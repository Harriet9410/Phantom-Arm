from pathlib import Path
import hashlib,json,subprocess,os,time,rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
r=Path('/root/tcei_competition_20260919');runs=Path('/root/gpufree-data/tcei_competition_20260919');rel=r/'releases/dev_v7_t1_13'
manifest_path=rel/'tcei_stack/BUILD_MANIFEST.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='1ea4baff2a868b42856352b44b8024c63346a67e6f803f7ba18f645c11647ee2'
manifest=json.loads(manifest_path.read_text())
assert all(hashlib.sha256((rel/'tcei_stack'/name).read_bytes()).hexdigest()==sha for name,sha in manifest['files'].items())
previous=runs/'single13_torch01'
assert json.loads((previous/'episode/summary.json').read_text())['status']=='succeeded'
assert json.loads((previous/'supervisor_finished.json').read_text())['execution_disabled'] is True
assert json.loads((previous/'review/manual_visual_review.json').read_text())['status']=='visually_confirmed_correct_category_on_requested_left_belt'
rospy.init_node('tcei_prepare_official13_01',anonymous=True)
flags=('/tcei_controller/execute','/tcei_nine/execute')
assert all(rospy.get_param(k) is False for k in flags)
samples=[]
for _ in range(4):
    joint=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    cap=rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data
    assert len(joint.name)==8 and len(set(joint.name))==8 and cap is False
    assert max(abs(v) for v in joint.velocity[:6])<.03 and max(abs(q) for q in joint.position[-2:])<.001
    samples.append({'names':list(joint.name),'position':list(joint.position),'velocity':list(joint.velocity),'stamp':joint.header.stamp.to_sec(),'captured':cap})
    time.sleep(.1)
assert samples[-1]['stamp']>samples[0]['stamp']
assert max(abs(a-b) for a,b in zip(samples[0]['position'],samples[-1]['position']))<.001
assert all(rospy.get_param(k) is False for k in flags)
(r/'official13_01_pre_shutdown.json').open('x').write(json.dumps({'checked_at':time.time(),'robot_samples':samples,'source':'fresh native idle empty-arm after independently reviewed single13 completion; coldinitialize five objects for new full official-example engineering round'},indent=2))
guard=['/usr/bin/python3',str(rel/'harness/process_guard.py')];stops=[]
for name in ('nine','controller','perception','sim'):
    rec=runs/'no_action13_01'/'pids'/(name+'.json');p=subprocess.run(guard+['stop',str(rec),'--timeout','20'],capture_output=True,text=True)
    stops.append({'name':name,'record':str(rec),'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(r/'official13_01_stop_prior.json').write_text(json.dumps(stops,indent=2));assert p.returncode==0
    assert subprocess.run(guard+['check',str(rec)]).returncode!=0
rospy.set_param('/tcei_controller/observation_ready',False);rospy.signal_shutdown('owned old services stopped')
env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(rel/'tcei_stack'),TCEI_ROBOT_CALIBRATION=str(rel/'calibration/robot_projection.static_checked_v1.json'))
result=subprocess.run(['bash',str(rel/'harness/start_stack.sh'),'official13_stack01'],env=env)
(r/'official13_01_launcher_result.json').write_text(json.dumps({'returncode':result.returncode,'finished_at':time.time(),'run':'official13_stack01'}))
assert result.returncode==0;print('OFFICIAL13_STACK01_WARMUP_COMPLETE',flush=True)
