#!/usr/bin/env python3
"""ROS entry point for V7 execution; warm nodes first, no mixed dry-run mode."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import xmlrpc.client
from episode_driver import EpisodeDriver,EpisodeFailure
from mission_ledger import OFFICIAL_EXAMPLE_INSTRUCTIONS
from ros_endpoints import BoundedXmlRpcTransport,bus_rows,missing_peers,INCOMING,OUTGOING


class RosTransport:
    def __init__(self):
        import rospy
        import rosgraph
        from std_msgs.msg import Bool,String
        self.rospy=rospy;self.Bool=Bool;self.String=String
        self.events=queue.Queue();self.lock=threading.Lock();self.scene=None;self.capture=None;self.capture_seen=0.
        rospy.init_node('tcei_episode',anonymous=True)
        self.publishers={topic:rospy.Publisher(topic,String,queue_size=1)
                         for topic in ('/tcei/request','/tcei/prepare_observation','/tcei/cancel_request')}
        self.subscribers=[rospy.Subscriber(topic,String,lambda message,topic=topic:self._receive(topic,message),queue_size=200)
                          for topic in ('/tcei/nine_status','/tcei/task_status','/tcei/stop_ack','/tcei/cancel_ack')]
        self.subscribers.extend([rospy.Subscriber('/tcei/candidates',String,self._scene,queue_size=1),
            rospy.Subscriber('/Jaka/gripper_is_captured',Bool,self._capture,queue_size=1)])
        self.node_uri=rospy.get_node_uri();self.node_name=rospy.get_name();self.connection_diagnostic=None
        self.master_uri=rosgraph.get_master_uri();self.peer_uris={};self.peer_uris_seen=0.

    def _receive(self,topic,message):
        try:
            data=json.loads(message.data)
            if not isinstance(data,dict):return
            self.events.put({'topic':topic,'message':data,'received_monotonic':time.monotonic()})
        except (ValueError,TypeError):return

    def _scene(self,message):
        try:data=json.loads(message.data)
        except (ValueError,TypeError):return
        if not isinstance(data,dict):return
        with self.lock:self.scene=data

    def _capture(self,message):
        with self.lock:self.capture=bool(message.data);self.capture_seen=time.monotonic()

    def execution_enabled(self):
        return all(self.rospy.get_param(name,False) is True
                   for name in ('/tcei_nine/execute','/tcei_controller/execute'))

    def is_shutdown(self):return self.rospy.is_shutdown()

    def latest_scene(self):
        with self.lock:return copy.deepcopy(self.scene)

    def next_event(self,timeout):
        try:return self.events.get(timeout=timeout)
        except queue.Empty:return None

    def publish(self,topic,data):
        deadline=data.get('deadline_monotonic') if isinstance(data,dict) else None
        if deadline is not None and time.monotonic()>=deadline:raise EpisodeFailure('request deadline expired before transport check')
        # A lost response channel must not prevent sending the stop request.
        # Physical stop still needs its independently received acknowledgement.
        missing=self.connection_missing(topic,require_response=topic!='/tcei/cancel_request')
        if missing:raise EpisodeFailure('intended command/response peer not connected: '+', '.join(missing))
        if deadline is not None and time.monotonic()>=deadline:raise EpisodeFailure('request deadline expired during transport check')
        self.publishers[topic].publish(self.String(json.dumps(data,ensure_ascii=False)))

    def connection_missing(self,command=None,require_response=True):
        try:
            with xmlrpc.client.ServerProxy(self.node_uri,transport=BoundedXmlRpcTransport(.5)) as api:
                rows=bus_rows(api.getBusInfo(self.node_name))
            if require_response:
                needed=set(INCOMING.values()) if command is None else {OUTGOING[command]}
                refresh=(time.monotonic()-self.peer_uris_seen>=2. or not needed<=self.peer_uris.keys() or
                         missing_peers(rows,command,require_response=True,node_uris=self.peer_uris))
                if refresh:
                    found={}
                    with xmlrpc.client.ServerProxy(self.master_uri,transport=BoundedXmlRpcTransport(.5)) as master:
                        for node in sorted(needed):
                            reply=master.lookupNode(self.node_name,node)
                            if (not isinstance(reply,(list,tuple)) or len(reply)!=3 or reply[0]!=1 or
                                    not isinstance(reply[2],str) or not reply[2].startswith('http://')):
                                raise ValueError('could not resolve expected ROS node: '+node)
                            found[node]=reply[2]
                    self.peer_uris.update(found);self.peer_uris_seen=time.monotonic()
            missing=missing_peers(rows,command,require_response=require_response,node_uris=self.peer_uris)
            self.connection_diagnostic={'checked_monotonic':time.monotonic(),'missing':missing,
                                        'bus_rows':rows,'resolved_node_uris':dict(self.peer_uris)}
            return missing
        except (OSError,ValueError,xmlrpc.client.Error) as error:
            self.connection_diagnostic={'checked_monotonic':time.monotonic(),'error':repr(error)}
            return ['bounded connection inspection failed: '+str(error)]

    def preflight(self,timeout):
        until=time.monotonic()+timeout
        while time.monotonic()<until and not self.is_shutdown():
            if not self.execution_enabled():raise EpisodeFailure('execution authorization is disabled')
            scene=self.latest_scene()
            with self.lock:captured=self.capture;capture_age=time.monotonic()-self.capture_seen
            connections=not self.connection_missing()
            if (scene is not None and type(scene.get('observed_at')) in (int,float)
                    and 0<=time.time()-scene['observed_at']<2 and captured is not None and capture_age<1.5 and connections):
                if captured:raise EpisodeFailure('initial gripper occupied; inspect before competition start')
                return scene
            time.sleep(.05)
        raise EpisodeFailure('prewarmed nodes/fresh camera/gripper feedback or subscribers unavailable')


def main(argv=None):
    parser=argparse.ArgumentParser()
    tasks=parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument('--instruction',action='append')
    tasks.add_argument('--official-example',action='store_true')
    parser.add_argument('--task-source',default=None)
    parser.add_argument('--output',required=True)
    parser.add_argument('--budget',type=float,default=600.)
    parser.add_argument('--preflight-timeout',type=float,default=30.)
    args=parser.parse_args(argv)
    if not 0<args.budget<=600:parser.error('budget must be greater than 0 and at most 600 seconds')
    if not 0<args.preflight_timeout<=180:parser.error('preflight timeout must be within 0..180 seconds')
    instructions=list(OFFICIAL_EXAMPLE_INSTRUCTIONS) if args.official_example else args.instruction
    task_source=args.task_source or ('official_example' if args.official_example else 'custom')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    before_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    def record(event):
        with (out/'events.jsonl').open('a',encoding='utf-8') as file:
            file.write(json.dumps(event,ensure_ascii=False)+'\n')
    driver=EpisodeDriver(RosTransport(),instructions,task_source,args.budget,record=record,
                         preflight_timeout=args.preflight_timeout)
    summary=driver.run()
    after_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    summary['code_sha256']=before_hashes
    changed=sorted(name for name in set(before_hashes)|set(after_hashes) if before_hashes.get(name)!=after_hashes.get(name))
    summary['code_changed_during_episode']=changed
    if changed:
        summary['result_before_code_change_invalidation']=summary['status'];summary['status']='invalidated_code_changed'
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print('SUMMARY',str(out/'summary.json'),flush=True)
    return 0 if summary['status']=='succeeded' else 1


if __name__=='__main__':raise SystemExit(main())
