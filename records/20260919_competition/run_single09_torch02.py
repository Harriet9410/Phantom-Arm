#!/usr/bin/python3
"""One bounded engineering episode; preserves failure and closes owned recorders."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path('/root/tcei_competition_20260919')
RELEASE=ROOT/'releases/dev_v7_t1_09'
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
RUN=RUNS/'single09_torch02'
STACK=RUNS/'no_action09_01'
sys.path[:0]=[str(RELEASE/'tcei_stack'),str(RELEASE/'harness')]
import rospy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import JointState
from process_guard import current,read_record,identity
from ros_endpoints import CANCEL_PROTOCOL,BoundedXmlRpcTransport,bus_rows,missing_peers
import xmlrpc.client


def main():
    rospy.init_node('tcei_single09_supervisor',anonymous=True)
    flags=('/tcei_controller/execute','/tcei_nine/execute')
    assert all(rospy.get_param(key) is False for key in flags)
    assert not (RUN/'episode').exists()
    assert (RUN/'rgbd/reference.npz').exists() and (RUN/'scalars/ready.json').exists()
    assert json.loads((ROOT/'protocol09_E_02/summary.json').read_text())['status']=='passed_native_protocol_only'
    manifest=json.loads((RELEASE/'tcei_stack/BUILD_MANIFEST.json').read_text())
    assert all(hashlib.sha256((RELEASE/'tcei_stack'/name).read_bytes()).hexdigest()==sha
               for name,sha in manifest['files'].items())
    for name in ('sim','controller','perception','nine'):
        assert current(read_record(STACK/'pids'/(name+'.json'))),name
    for name in ('rgbd','scalars'):
        assert current(read_record(RUN/'pids'/(name+'.json'))),name
    assert (RUN/'visual_target_review.json').exists()
    review=json.loads((RUN/'visual_target_review.json').read_text())
    assert review['class']=='Torch' and review['side']=='left' and review['confirmed_unique_visible_target'] is True
    assert hashlib.sha256((RUN/'rgbd/reference.jpg').read_bytes()).hexdigest()==review['reference_jpg_sha256']
    scene=json.loads(rospy.wait_for_message('/tcei/candidates',String,timeout=5).data)
    torches=[c for c in scene['candidates'] if c['class']=='Torch' and c['identity_status']=='confirmed']
    assert len(torches)==1 and time.time()-scene['observed_at']<2.
    assert max(abs(a-b) for a,b in zip(torches[0]['pixel'],review['reference_pixel']))<6.
    assert rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data is False
    joint=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert len(joint.position)==8 and max(abs(v) for v in joint.velocity[:6])<.03
    assert max(abs(v) for v in joint.position[-2:])<.001
    (RUN/'before_scene.json').open('x').write(json.dumps(scene,ensure_ascii=False,indent=2))
    gold={'registered_at':time.time(),'instruction':'抓取最左方的军用手电筒，放到左侧传送带',
          'expected_class':'Torch','expected_side':'left','expected_source_pixel':torches[0]['pixel'],
          'pre_round_stable_id':torches[0]['stable_id'],
          'identity_note':'New round may reset stable IDs; verify physical source pixel/class plus actual model binding, not this old ID alone.',
          'scope':'engineering single-object; excluded from original30/new30 and official score'}
    (RUN/'expected_target.json').open('x').write(json.dumps(gold,ensure_ascii=False,indent=2))
    (STACK/'round_started.lock').open('x').write(json.dumps({'engineering_only':True,'run':str(RUN),'time':time.time()}))
    cancel_pub=rospy.Publisher('/tcei/cancel_request',String,queue_size=1)
    replies={}
    def receive(topic,msg):replies[topic]=(time.monotonic(),json.loads(msg.data))
    subs=[rospy.Subscriber(topic,String,lambda m,t=topic:receive(t,m),queue_size=20)
          for topic in ('/tcei/cancel_ack','/tcei/stop_ack')]
    def stop_after_failure(reason):
        start=time.monotonic();request={'protocol':CANCEL_PROTOCOL,'cancel_id':uuid.uuid4().hex,
            'round_id':'supervisor_'+uuid.uuid4().hex,'request_id':'single09_watchdog','reason':reason[:1000]}
        with xmlrpc.client.ServerProxy(rospy.get_node_uri(),transport=BoundedXmlRpcTransport(.5)) as api:
            missing=missing_peers(bus_rows(api.getBusInfo(rospy.get_name())),'/tcei/cancel_request',require_response=False)
        if missing:return {'state':'unconfirmed','reason':str(missing)}
        cancel_pub.publish(String(json.dumps(request)));mapping=None
        while time.monotonic()-start<8.:
            accepted=replies.get('/tcei/cancel_ack');stop=replies.get('/tcei/stop_ack')
            if accepted and accepted[0]>=start and accepted[1].get('cancel_id')==request['cancel_id']:
                mapping=accepted[1]
            if (mapping and mapping.get('state')=='accepted' and stop and stop[0]>=start and
                    stop[1].get('id')==mapping.get('stop_id') and stop[1].get('state')=='stopped'):
                return {'state':'stopped','mapping':mapping,'proof':stop[1]}
            time.sleep(.05)
        return {'state':'unconfirmed','mapping':mapping}
    command=['/usr/bin/python3','-u',str(RELEASE/'tcei_stack/run_episode.py'),'--instruction',gold['instruction'],
             '--task-source','engineering_single','--budget','180','--output',str(RUN/'episode')]
    result={'started_at':time.time(),'command':command,'budget_seconds':180,'watchdog_seconds':230,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'returncode':None}
    child=None
    try:
        for key in flags:rospy.set_param(key,True)
        with (RUN/'logs/episode.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=str(ROOT),env=os.environ.copy(),stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        owned=identity(child.pid);owned.update(command=command,cwd=str(ROOT),started_at=time.time())
        (RUN/'pids/episode.json').open('x').write(json.dumps(owned,indent=2))
        result['returncode']=child.wait(timeout=230.)
        summary_path=RUN/'episode/summary.json'
        if not summary_path.exists():raise RuntimeError('episode exited without summary')
        result['episode_status']=json.loads(summary_path.read_text()).get('status')
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        result['supervisor_stop']=stop_after_failure(str(error))
    finally:
        for key in flags:rospy.set_param(key,False)
        result['execution_disabled']=all(rospy.get_param(key) is False for key in flags)
        guard=['/usr/bin/python3',str(RELEASE/'harness/process_guard.py')]
        if child is not None and child.poll() is None:
            stopped=subprocess.run(guard+['stop',str(RUN/'pids/episode.json'),'--timeout','10'],capture_output=True,text=True)
            result['episode_cleanup']={'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr}
        time.sleep(2.)
        result['recorder_cleanup']=[]
        for name in ('rgbd','scalars'):
            stopped=subprocess.run(guard+['stop',str(RUN/'pids'/(name+'.json')),'--timeout','10'],capture_output=True,text=True)
            result['recorder_cleanup'].append({'name':name,'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})
        for sub in subs:sub.unregister()
        rospy.signal_shutdown('single-object supervisor finished')
        result['finished_at']=time.time()
        (RUN/'supervisor_finished.json').open('x').write(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result.get('episode_status')=='succeeded' else 1


if __name__=='__main__':
    try:exit_code=main()
    except BaseException as error:
        failure={'error':type(error).__name__+': '+str(error),'time':time.time(),'cleanup':[]}
        guard=['/usr/bin/python3',str(RELEASE/'harness/process_guard.py')]
        for name in ('rgbd','scalars'):
            record=RUN/'pids'/(name+'.json')
            if not record.exists():continue
            stopped=subprocess.run(guard+['stop',str(record),'--timeout','10'],capture_output=True,text=True)
            failure['cleanup'].append({'name':name,'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})
        if RUN.exists():(RUN/'supervisor_aborted.json').open('x').write(json.dumps(failure,ensure_ascii=False,indent=2))
        print(json.dumps(failure,ensure_ascii=False),flush=True);exit_code=1
    raise SystemExit(exit_code)
