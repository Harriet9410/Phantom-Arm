from pathlib import Path
import json,time,uuid,threading,rospy
from std_msgs.msg import String,Bool
r=Path('/root/tcei_competition_20260919'); prior=json.loads((r/'H01_dynamic_B_02/summary.json').read_text()); assert prior['status']=='passed_dynamic_short_move' and prior['lock_state']=='reset_acknowledged'; d=r/'observation07_C_01';d.mkdir(exist_ok=False); log=(d/'events.jsonl').open('x',buffering=1);lock=threading.RLock(); latest={};events=[]; result={'engineering_test':True,'competition_round':False,'grasp_requested':False,'status':'running'}
rospy.init_node('tcei_observation_check',anonymous=True)
def cb(topic,msg):
 try:v=json.loads(msg.data)
 except Exception:v={'unparsed':msg.data}
 row={'topic':topic,'wall':time.time(),'monotonic':time.monotonic(),'value':v}
 with lock:latest[topic]=row;events.append(row);log.write(json.dumps(row,ensure_ascii=False)+chr(10))
subs=[rospy.Subscriber(t,String,lambda m,t=t:cb(t,m),queue_size=100) for t in ('/tcei/task_status','/tcei/planner_status','/tcei/stop_ack','/tcei/candidates','/tcei/joint_diagnostics')];pub=rospy.Publisher('/tcei/prepare_observation',String,queue_size=1);cancel=rospy.Publisher('/tcei/cancel',Bool,queue_size=1)
try:
 assert rospy.get_param('/tcei_nine/execute') is False and rospy.get_param('/tcei_controller/execute') is False
 until=time.monotonic()+10
 while time.monotonic()<until:
  with lock:ready=pub.get_num_connections()>0 and cancel.get_num_connections()>0 and all(t in latest for t in ('/tcei/joint_diagnostics','/tcei/candidates','/tcei/stop_ack'))
  if ready:break
  time.sleep(.05)
 assert ready;ack=latest['/tcei/stop_ack']['value'];assert ack['state']=='reset' and ack['id']==prior['stop']['id']; diag=latest['/tcei/joint_diagnostics'];assert time.time()-diag['wall']<1.5 and diag['value']['tracking'] is False
 began=time.monotonic(); wall=time.time();rid='observe_'+uuid.uuid4().hex;round_id='engineering_observe_'+uuid.uuid4().hex;payload={'request_id':rid,'round_id':round_id,'task_id':round_id+'-observation','deadline_monotonic':began+120.,'started_at_monotonic':began,'started_at_wall':wall};result.update(request=payload,initial_scene=latest['/tcei/candidates']['value']); (d/'request.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));rospy.set_param('/tcei_controller/execute',True);pub.publish(String(data=json.dumps(payload)));result['request_sent_once']=True
 terminal=None
 while time.monotonic()<began+112.:
  with lock:matches=[x for x in events if x['topic']=='/tcei/task_status' and x['value'].get('request_id')==rid and x['value'].get('status') in ('observation_completed','task_failed','observation_preparation_failed','observation_request_rejected','execution_disabled')]
  if matches:terminal=matches[-1];break
  time.sleep(.05)
 if terminal is None:raise TimeoutError('observation deadline approaching')
 result['terminal']=terminal;assert terminal['value']['status']=='observation_completed',str(terminal['value']);result.update(status='observation_completed',elapsed_wall=time.monotonic()-began,final_scene=latest['/tcei/candidates']['value'],final_robot=latest['/tcei/joint_diagnostics']['value'])
except Exception as error:
 result.update(status='failed',error=repr(error))
 if result.get('request_sent_once'):
  cancel.publish(Bool(data=True));result['cancel_sent_once']=True;end=time.monotonic()+7
  while time.monotonic()<end:
   with lock:row=latest.get('/tcei/stop_ack',{})
   if row.get('value',{}).get('state')=='stopped' and row.get('monotonic',0)>began:result['stopped_after_failure']=row;break
   time.sleep(.05)
finally:
 rospy.set_param('/tcei_controller/execute',False);result['execute_flags']={p:rospy.get_param(p) for p in ('/tcei_controller/execute','/tcei_nine/execute')};result['finished_at']=time.time();(d/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('OBSERVATION_RESULT',result['status'],flush=True)
