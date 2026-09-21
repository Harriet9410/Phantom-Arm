from pathlib import Path
import json,subprocess,os,time,rospy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import JointState
r=Path('/root/tcei_competition_20260919');runs=Path('/root/gpufree-data/tcei_competition_20260919');rel=r/'releases/dev_v7_t1_08'
assert (rel/'PATCH_INSTALL_VERIFIED.json').exists();rospy.init_node('tcei_prepare_full08',anonymous=True)
assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
joint=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5);diag={'source':'native_joint_state_while_stopped','all_joint_names':list(joint.name),'all_joint_positions':list(joint.position),'all_joint_velocities':list(joint.velocity),'stamp':joint.header.stamp.to_sec()};cap=rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data
ack=json.loads(rospy.wait_for_message('/tcei/stop_ack',String,timeout=5).data)
assert ack['state']=='stopped' and ack['id']=='8ef34ac878a14dfc9cdde44875438218'
assert len(joint.name)==8 and len(set(joint.name))==8 and cap is False and max(abs(v) for v in diag['all_joint_velocities'][:6])<.03 and max(abs(q) for q in diag['all_joint_positions'][-2:])<.001
(r/'full08_pre_shutdown.json').open('x').write(json.dumps({'checked_at':time.time(),'robot':diag,'captured':cap,'stop':ack},indent=2))
guard=['/usr/bin/python3',str(rel/'harness/process_guard.py')];stops=[]
for name in ('nine','controller','perception','sim'):
 rec=runs/'no_action07_01'/'pids'/(name+'.json');p=subprocess.run(guard+['stop',str(rec),'--timeout','20'],capture_output=True,text=True)
 stops.append({'name':name,'record':str(rec),'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(r/'full08_stop_prior.json').write_text(json.dumps(stops,indent=2));assert p.returncode==0
 assert subprocess.run(guard+['check',str(rec)]).returncode!=0
rospy.set_param('/tcei_controller/observation_ready',False);rospy.signal_shutdown('owned old services stopped')
env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(rel/'tcei_stack'),TCEI_ROBOT_CALIBRATION=str(rel/'calibration/robot_projection.static_checked_v1.json'))
result=subprocess.run(['bash',str(rel/'harness/start_stack.sh'),'no_action08_01'],env=env)
(r/'full08_launcher_result.json').write_text(json.dumps({'returncode':result.returncode,'finished_at':time.time(),'run':'no_action08_01'}))
assert result.returncode==0;print('FULL08_WARMUP_COMPLETE',flush=True)
