from pathlib import Path
import hashlib,json,subprocess,os,time,rospy
from std_msgs.msg import Bool,String,Float32MultiArray
from sensor_msgs.msg import JointState
r=Path('/root/tcei_competition_20260919');runs=Path('/root/gpufree-data/tcei_competition_20260919');rel=r/'releases/dev_v7_t1_13'
manifest_path=rel/'tcei_stack/BUILD_MANIFEST.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='1ea4baff2a868b42856352b44b8024c63346a67e6f803f7ba18f645c11647ee2'
manifest=json.loads(manifest_path.read_text())
assert all(hashlib.sha256((rel/'tcei_stack'/name).read_bytes()).hexdigest()==sha for name,sha in manifest['files'].items())
(rel/'PATCH_INSTALL_VERIFIED.json').write_text(json.dumps({'version':manifest['version'],'verified_at':time.time(),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'runtime_files':len(manifest['files'])},indent=2))
completed=runs/'single12_torch01'
assert json.loads((completed/'episode/summary.json').read_text())['status']=='stopped'
assert json.loads((completed/'supervisor_finished.json').read_text())['execution_disabled'] is True
rospy.init_node('tcei_prepare_full13',anonymous=True)
flags=('/tcei_controller/execute','/tcei_nine/execute')
assert all(rospy.get_param(k) is False for k in flags)
ack=json.loads(rospy.wait_for_message('/tcei/stop_ack',String,timeout=5).data)
assert ack['state']=='stopped' and ack['id']=='74f1ee589c76487882df960a1e72c891' and 0<=time.time()-ack['time']<1.5
efforts=list(rospy.wait_for_message('/Jaka/get_gripper_efforts',Float32MultiArray,timeout=5).data)
samples=[]
for _ in range(4):
    joint=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    cap=rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data
    assert len(joint.name)==8 and len(set(joint.name))==8 
    assert max(abs(v) for v in joint.velocity[:6])<.03
    samples.append({'names':list(joint.name),'position':list(joint.position),'velocity':list(joint.velocity),'stamp':joint.header.stamp.to_sec(),'captured':cap})
    time.sleep(.1)
assert samples[-1]['stamp']>samples[0]['stamp']
assert max(abs(a-b) for a,b in zip(samples[0]['position'],samples[-1]['position']))<.001
assert all(rospy.get_param(k) is False for k in flags)
(r/'full13_pre_shutdown.json').open('x').write(json.dumps({'checked_at':time.time(),'robot_samples':samples,'stop':ack,'efforts':efforts,'source':'fresh native stopped failed episode; hold preserved until owned simulator shutdown','operation':'cold initialize next engineering scene after archived failure, not recovery or success of old episode'},indent=2))
guard=['/usr/bin/python3',str(rel/'harness/process_guard.py')];stops=[]
for name in ('nine','controller','perception','sim'):
    rec=runs/'no_action12_01'/'pids'/(name+'.json');p=subprocess.run(guard+['stop',str(rec),'--timeout','20'],capture_output=True,text=True)
    stops.append({'name':name,'record':str(rec),'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(r/'full13_stop_prior.json').write_text(json.dumps(stops,indent=2));assert p.returncode==0
    assert subprocess.run(guard+['check',str(rec)]).returncode!=0
rospy.set_param('/tcei_controller/observation_ready',False);rospy.signal_shutdown('owned old services stopped')
env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(rel/'tcei_stack'),TCEI_ROBOT_CALIBRATION=str(rel/'calibration/robot_projection.static_checked_v1.json'))
result=subprocess.run(['bash',str(rel/'harness/start_stack.sh'),'no_action13_01'],env=env)
(r/'full13_launcher_result.json').write_text(json.dumps({'returncode':result.returncode,'finished_at':time.time(),'run':'no_action13_01'}))
assert result.returncode==0;print('FULL13_WARMUP_COMPLETE',flush=True)
