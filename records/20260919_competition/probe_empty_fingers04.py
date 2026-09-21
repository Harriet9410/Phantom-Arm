from pathlib import Path
import rospy,json,time,threading,subprocess,math
from std_msgs.msg import String,Bool,Float32,Float32MultiArray
r=Path('/root/tcei_competition_20260919');d=r/'empty_fingers04_D_01';d.mkdir(exist_ok=False);latest={};lock=threading.RLock();log=(d/'events.jsonl').open('x',buffering=1);result={'status':'failed','engineering_only':True,'object_grasp_requested':False,'commands':[]};rospy.init_node('tcei_empty_finger_check',anonymous=True)
def cb(k,msg):
 value=json.loads(msg.data) if k=='diag' else msg.data;row={'kind':k,'value':value,'time':time.monotonic()}
 with lock:latest[k]=row;log.write(json.dumps(row)+chr(10))
subs=[rospy.Subscriber('/tcei/joint_diagnostics',String,lambda m:cb('diag',m)),rospy.Subscriber('/Jaka/gripper_is_captured',Bool,lambda m:cb('captured',m)),rospy.Subscriber('/Jaka/get_gripper_efforts',Float32MultiArray,lambda m:cb('efforts',m))];pub=rospy.Publisher('/Jaka/set_gripper_value',Float32,queue_size=1)
def ready(value,timeout):
 end=time.monotonic()+timeout; count=0;seen=-1
 while time.monotonic()<end:
  with lock:rows={k:v for k,v in latest.items()}
  if all(k in rows and time.monotonic()-rows[k]['time']<1.5 for k in ('diag','captured','efforts')):
   x=rows['diag'];v=x['value'];assert rows['captured']['value'] is False;assert all(math.isfinite(e) and abs(e)<.2 for e in rows['efforts']['value']);assert v['tracking'] is False;assert math.dist(v['tcp_actual'],[.0662943895,.40,2.7503792615])<.01
   if x['time']!=seen:seen=x['time'];count=count+1 if max(abs(q-value) for q in v['all_joint_positions'][-2:])<.001 else 0
   if count>=3:return v
  time.sleep(.03)
 raise TimeoutError('measured finger target not settled')
try:
 assert rospy.get_param('/tcei_controller/execute') is False and rospy.get_param('/tcei_nine/execute') is False;result['before']=ready(0.,8);assert pub.get_num_connections()>0;pub.publish(Float32(.02));result['commands'].append(.02);result['half_closed']=ready(.02,5)
 source=(r/'probe_projection_initial04.py').read_text().replace('projection_initial04_01','projection_fingers04_01');p=r/'probe_projection_fingers04.py';p.open('x').write(source); run=subprocess.run(['/usr/bin/python3',str(p)],capture_output=True,text=True,timeout=20);result['projection_probe']={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr[-1000:]};assert run.returncode==0;result['status']='partial_closure_projection_captured'
except Exception as error:result['error']=repr(error)
finally:
 pub.publish(Float32(0.));result['commands'].append(0.)
 try:result['reopened']=ready(0.,5)
 except Exception as error:result['status']='failed';result['reopen_error']=repr(error)
 (d/'summary.json').write_text(json.dumps(result,indent=2));print('EMPTY_FINGER_RESULT',result['status'],flush=True)
