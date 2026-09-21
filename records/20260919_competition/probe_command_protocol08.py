#!/usr/bin/python3
"""Native08 communication check with motion disabled; leaves a measured stop latched."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time
import uuid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--code',required=True);parser.add_argument('--outdir',required=True)
    args=parser.parse_args();code=Path(args.code);sys.path[:0]=[str(code),str(code.parent/'harness')]
    from run_episode import RosTransport
    from episode_driver import EpisodeDriver
    from mission_ledger import MissionLedger
    from ros_endpoints import OBSERVATION_PROTOCOL
    from probe_idle_stop import plain,stable,J,P,C,E,K
    import rospy
    from std_msgs.msg import String,Bool,Float32,Float32MultiArray
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from rosgraph_msgs.msg import Clock
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=False)
    stream=(out/'events.jsonl').open('x',encoding='utf-8',buffering=1)
    lock=threading.RLock();native=[];intents=[];subs=[]
    result={'status':'failed','scope':'disabled_execution_admission_and_explicit_cancel_only',
            'reset_sent':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def log(row):
        with lock:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    def record(topic,msg):
        row={'kind':'receive','topic':topic,'monotonic':time.monotonic(),'wall':time.time(),
             'value':plain(msg),'callerid':getattr(msg,'_connection_header',{}).get('callerid')}
        with lock:
            log(row);native.append(row)
            if topic in action_topics:intents.append(row)
    action_topics={'/Jaka/set_end_effector_pose':PoseStamped,'/Jaka/set_gripper_value':Float32,
                   '/tcei/execute_preview':String}
    topics={J:JointState,P:PoseStamped,C:Bool,E:Float32MultiArray,K:Clock,**action_topics}
    def flags_disabled():
        assert all(rospy.get_param(key) is False for key in ('/tcei_controller/execute','/tcei_nine/execute'))
    def wait_event(transport,rid,terminal):
        began=time.monotonic();rows=[]
        while time.monotonic()-began<3.:
            event=transport.next_event(.1)
            if event is None:continue
            log({'kind':'protocol_event',**event})
            if event['message'].get('request_id')!=rid:continue
            rows.append(event)
            if event['message'].get('status')==terminal:return rows
        raise TimeoutError('missing '+terminal+' for '+rid)
    def alarm(_signum,_frame):raise TimeoutError('bounded40second native protocol probe')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,40.)
    try:
        transport=RosTransport();flags_disabled()
        subs=[rospy.Subscriber(t,typ,lambda msg,t=t:record(t,msg),queue_size=1000) for t,typ in topics.items()]
        began=time.monotonic();until=began+10.
        while time.monotonic()<until:
            missing=transport.connection_missing()
            with lock:joints=[r for r in native if r['topic']==J]
            if not missing and joints:break
            time.sleep(.05)
        assert not missing and joints,transport.connection_diagnostic
        result['connections']=transport.connection_diagnostic
        names=joints[-1]['value']['name'];assert len(names)==8 and len(set(names))==8
        round_id='protocol08_'+uuid.uuid4().hex;requests=[]
        for index in (1,2):
            flags_disabled();sent=time.monotonic();rid=round_id+'_'+str(index)
            request={'request_id':rid,'round_id':round_id,'task_id':round_id+'-observation',
                     'deadline_monotonic':sent+30.,'started_at_monotonic':sent,'started_at_wall':time.time()}
            log({'kind':'publish_once','topic':'/tcei/prepare_observation','monotonic':sent,'value':request})
            transport.publish('/tcei/prepare_observation',request)
            rows=wait_event(transport,rid,'execution_disabled')
            accepted=[row for row in rows if row['message'].get('status')=='observation_accepted']
            assert len(accepted)==1 and accepted[0]['message']['round_id']==round_id
            assert accepted[0]['message']['observation_protocol']==OBSERVATION_PROTOCOL
            assert not any(row['message'].get('status')=='observation_started' for row in rows)
            requests.append({'request':request,'events':rows,'acceptance_seconds':accepted[0]['received_monotonic']-sent})
        result['observations']=requests
        # This request was never submitted: cancellation must still bind to the
        # actual stop while the controller reports its earlier current request.
        driver=EpisodeDriver(transport,['engineering cancellation only'],clock=time,record=log,stop_timeout=7.)
        started=time.monotonic();driver.ledger=MissionLedger(driver.round_id,driver.instructions,started,time.time())
        driver.current_request=driver.round_id+'-never-submitted'
        safety=driver._cancel_and_wait('protocol check for an unreceived request');result['safe_stop']=safety
        assert safety['state']=='stopped',safety
        assert safety['cancel_mapping']['request_id']==driver.current_request
        assert safety['cancel_mapping']['affected_request_id']==requests[-1]['request']['request_id']
        since=time.monotonic();until=since+7.;proof=None
        while time.monotonic()<until:
            flags_disabled()
            with lock:proof=stable(native,since,.25,time.monotonic(),names)
            if proof:break
            time.sleep(.05)
        assert proof is not None;result['independent_stationary']=proof
        req={**requests[-1]['request'],'request_id':round_id+'-late','deadline_monotonic':time.monotonic()+30.}
        transport.publish('/tcei/prepare_observation',req)
        result['late_request_events']=wait_event(transport,req['request_id'],'observation_request_rejected')
        assert not any(r['message'].get('status') in ('observation_accepted','observation_started','observation_completed')
                       for r in result['late_request_events'])
        flags_disabled()
        with lock:assert not intents,'unexpected pose/gripper command during disabled-execution check'
        result['status']='passed_native_protocol_only'
    except BaseException as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        for sub in subs:sub.unregister()
        rospy.signal_shutdown('native protocol check finished; no stop reset requested')
        result.update(finished_at=time.time(),pose_gripper_command_count=len(intents),native_rows=len(native))
        with lock:stream.close()
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ('status','error','pose_gripper_command_count','native_rows')}),flush=True)
    return 0 if result['status']=='passed_native_protocol_only' else 1


if __name__=='__main__':raise SystemExit(main())
