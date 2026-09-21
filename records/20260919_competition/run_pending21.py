#!/usr/bin/python3
"""One bounded official five-example engineering episode; preserves failure and closes owned recorders."""
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
RELEASE=ROOT/'releases/dev_v7_t1_21'
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
RUN=RUNS/'pending21_A01'
STACK=RUNS/'pending21_stack01'
sys.path[:0]=[str(RELEASE/'tcei_stack'),str(RELEASE/'harness')]
import rospy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import JointState
from process_guard import current,read_record,identity
from ros_endpoints import CANCEL_PROTOCOL,BoundedXmlRpcTransport,bus_rows,missing_peers
import xmlrpc.client


def main():
    rospy.init_node('tcei_official18_supervisor',anonymous=True)
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
    from mission_ledger import OFFICIAL_EXAMPLE_INSTRUCTIONS
    instructions=['抓取左上方的烟雾弹，放到左侧传送带','抓取右下方的弹夹，放到右侧传送带',
                  '抓取最左方的军用手电筒，放到左侧传送带','抓取手雷，放到右侧传送带',
                  '抓取剩余的物品，放到左侧传送带']
    assert list(OFFICIAL_EXAMPLE_INSTRUCTIONS)==instructions
    review=json.loads((RUN/'visual_targets_review.json').read_text())
    assert review['instructions']==instructions and review['all_five_visible_identified'] is True
    assert hashlib.sha256((RUN/'rgbd/reference.jpg').read_bytes()).hexdigest()==review['reference_jpg_sha256']
    assert len(review['expected_targets'])==5
    scene=json.loads(rospy.wait_for_message('/tcei/candidates',String,timeout=5).data)
    assert time.time()-scene['observed_at']<2. and len(scene['candidates'])==5
    bound=[]
    for target in review['expected_targets']:
        matches=[c for c in scene['candidates'] if c['class']==target['class'] and
                 c['identity_status']=='confirmed' and max(abs(a-b) for a,b in zip(c['pixel'],target['pixel']))<6.]
        assert len(matches)==1,'independent visual target does not match current scene'
        bound.append(matches[0]['stable_id'])
    assert len(set(bound))==5
    assert rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data is False
    joint=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert len(joint.position)==8 and max(abs(v) for v in joint.velocity[:6])<.03
    assert max(abs(v) for v in joint.position[-2:])<.001
    (RUN/'before_scene.json').open('x').write(json.dumps(scene,ensure_ascii=False,indent=2))
    gold={'registered_at':time.time(),'instructions':instructions,'expected_targets':review['expected_targets'],
          'pre_round_stable_ids':bound,'identity_note':'IDs may reset; use independently reviewed appearance/pixel anchors and actual per-task bindings.',
          'scope':'declared evidence-gap test; first release must remain unverified without blocking other four or being regrasped as remaining'}
    (RUN/'expected_target.json').open('x').write(json.dumps(gold,ensure_ascii=False,indent=2))
    (STACK/'round_started.lock').open('x').write(json.dumps({'engineering_only':True,'run':str(RUN),'time':time.time()}))
    cancel_pub=rospy.Publisher('/tcei/cancel_request',String,queue_size=1)
    replies={}
    def receive(topic,msg):replies[topic]=(time.monotonic(),json.loads(msg.data))
    subs=[rospy.Subscriber(topic,String,lambda m,t=topic:receive(t,m),queue_size=20)
          for topic in ('/tcei/cancel_ack','/tcei/stop_ack')]
    def stop_after_failure(reason):
        start=time.monotonic();request={'protocol':CANCEL_PROTOCOL,'cancel_id':uuid.uuid4().hex,
            'round_id':'supervisor_'+uuid.uuid4().hex,'request_id':'official18_watchdog','reason':reason[:1000]}
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
    command=['/usr/bin/python3','-u',str(RELEASE/'tcei_stack/run_episode.py'),'--official-example',
             '--task-source','engineering_pending_placement','--budget','600','--output',str(RUN/'episode')]
    result={'started_at':time.time(),'command':command,'budget_seconds':600,'watchdog_seconds':660,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'returncode':None}
    child=None
    try:
        for key in flags:rospy.set_param(key,True)
        with (RUN/'logs/episode.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=str(ROOT),env=os.environ.copy(),stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        owned=identity(child.pid);owned.update(command=command,cwd=str(ROOT),started_at=time.time())
        (RUN/'pids/episode.json').open('x').write(json.dumps(owned,indent=2))
        result['returncode']=child.wait(timeout=660.)
        summary_path=RUN/'episode/summary.json'
        if not summary_path.exists():raise RuntimeError('episode exited without summary')
        summary=json.loads(summary_path.read_text());result['episode_status']=summary.get('status')
        if result['episode_status']=='succeeded':
            assert result['returncode']==0 and summary['verified_objects']==5
            assert [task['instruction'] for task in summary['tasks']]==instructions
            assert not summary.get('code_changed_during_episode')
            assert all(task['result']['status'] in ('task_succeeded','task_pending_verification') for task in summary['tasks'])
            assert all(task['status']=='verified' for task in summary['ledger']['tasks'])
        elif result['episode_status']=='completed_with_unverified_placements':
            assert [task['instruction'] for task in summary['tasks']]==instructions
            assert all(task['result']['status'] in ('task_succeeded','task_pending_verification') for task in summary['tasks'])
            assert all(task['status'] in ('verified','executed_pending_verification') for task in summary['ledger']['tasks'])
            assert not summary.get('code_changed_during_episode')
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error);result['episode_status']='supervisor_failed'
        result['supervisor_stop']=stop_after_failure(str(error))
    finally:
        for key in flags:rospy.set_param(key,False)
        result['execution_disabled']=all(rospy.get_param(key) is False for key in flags)
        guard=['/usr/bin/python3',str(RELEASE/'harness/process_guard.py')]
        if child is not None and child.poll() is None:
            stopped=subprocess.run(guard+['stop',str(RUN/'pids/episode.json'),'--timeout','20'],capture_output=True,text=True)
            result['episode_cleanup']={'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr}
        time.sleep(2.)
        result['recorder_cleanup']=[]
        for name in ('rgbd','scalars'):
            stopped=subprocess.run(guard+['stop',str(RUN/'pids'/(name+'.json')),'--timeout','20'],capture_output=True,text=True)
            result['recorder_cleanup'].append({'name':name,'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})
        result['evidence_closeout']={}
        for name in ('rgbd','scalars'):
            closed=RUN/name/'closed.json'
            result['evidence_closeout'][name]=json.loads(closed.read_text()) if closed.exists() else None
        result['evidence_complete']=(result['evidence_closeout']['rgbd'] is not None and
            result['evidence_closeout']['rgbd'].get('complete') is True and result['evidence_closeout']['scalars'] is not None and
            all(item['code']==0 for item in result['recorder_cleanup']))
        for sub in subs:sub.unregister()
        rospy.signal_shutdown('official five-example supervisor finished')
        result['finished_at']=time.time()
        (RUN/'supervisor_finished.json').open('x').write(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result.get('episode_status')=='succeeded' and result.get('evidence_complete') else 1


if __name__=='__main__':
    try:exit_code=main()
    except BaseException as error:
        failure={'error':type(error).__name__+': '+str(error),'time':time.time(),'cleanup':[]}
        guard=['/usr/bin/python3',str(RELEASE/'harness/process_guard.py')]
        for name in ('rgbd','scalars'):
            record=RUN/'pids'/(name+'.json')
            if not record.exists():continue
            stopped=subprocess.run(guard+['stop',str(record),'--timeout','20'],capture_output=True,text=True)
            failure['cleanup'].append({'name':name,'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})
        if RUN.exists():(RUN/'supervisor_aborted.json').open('x').write(json.dumps(failure,ensure_ascii=False,indent=2))
        print(json.dumps(failure,ensure_ascii=False),flush=True);exit_code=1
    raise SystemExit(exit_code)
