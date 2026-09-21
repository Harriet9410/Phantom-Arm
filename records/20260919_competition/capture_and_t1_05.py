from pathlib import Path
import json,time,threading,hashlib,struct,subprocess,copy
import rospy,cv2,numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
r=Path('/root/tcei_competition_20260919'); d=r/'t1_capture05_before01'; d.mkdir(exist_ok=False); rospy.init_node('tcei_capture05',anonymous=True); scenes={};images={};lock=threading.Lock();bridge=CvBridge()
def scene_cb(msg):
 data=json.loads(msg.data); k=data.get('scene_image_binding',{}).get('sensor_stamp_ns')
 with lock:
  scenes[k]=data
  while len(scenes)>30:scenes.pop(next(iter(scenes)))
def image_cb(msg):
 with lock:
  images[msg.header.stamp.to_nsec()]=msg
  while len(images)>30:images.pop(next(iter(images)))
rospy.Subscriber('/tcei/candidates',String,scene_cb,queue_size=10); rospy.Subscriber('/tcei/annotated_image',Image,image_cb,queue_size=2,buff_size=2**24); until=time.monotonic()+15; pair=None
while time.monotonic()<until:
 with lock:
  keys=set(scenes)&set(images)
  if keys:
   k=max(keys);s=scenes[k];m=images[k]
   if time.time()-s['observed_at']<2:pair=(s,m);break
 time.sleep(.02)
assert pair is not None;s,m=pair;arr=bridge.imgmsg_to_cv2(m,'bgr8');h,w=arr.shape[:2];digest=hashlib.sha256(b'tcei.numbered_scene.bgr8.v1'+bytes([0])+struct.pack('<II',w,h)+np.ascontiguousarray(arr).tobytes()).hexdigest(); assert digest==s['scene_image_binding']['canonical_pixel_sha256']; assert rospy.get_param('/tcei_nine/execute') is False and rospy.get_param('/tcei_controller/execute') is False
cv2.imwrite(str(d/'numbered_scene.png'),arr);(d/'scene.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)); old=json.loads((r/'t1_capture04_before01/scene.json').read_text()); before={c['stable_id']:c for c in old['candidates']};after={c['stable_id']:c for c in s['candidates']}; assert set(before)==set(after);assert old['spatial_context']==s['spatial_context'];assert old['scene_complete']==s['scene_complete'];assert len(old['unknown_regions'])==len(s['unknown_regions'])
for sid,c in after.items():
 assert c['class']==before[sid]['class'] and c['identity_status']=='confirmed';assert max(abs(a-b) for a,b in zip(c['pixel'],before[sid]['pixel']))<3
for a,b in zip(old['unknown_regions'],s['unknown_regions']):assert a['reason']==b['reason'] and max(abs(x-y) for x,y in zip(a['bbox'],b['bbox']))<3
oracle=json.loads((r/'t1_initial04_01_oracle.json').read_text()); oracle.update(version='dev_v7_t1_05',registered_at=time.time(),source_capture=str(d),pre_inference_inheritance='Original independently annotated targets preserved; stable membership, class, pixels, spatial context, scene coverage and unknown-region geometry checked unchanged before this run.',actions_enabled=False); op=r/'t1_initial05_01_oracle.json'; assert not op.exists();op.write_text(json.dumps(oracle,ensure_ascii=False,indent=2)); out=r/'t1_initial05_01'; assert not out.exists(); cmd=['/usr/bin/python3',str(r/'releases/dev_v7_t1_05/harness/t1_semantics.py'),'--output',str(out),'--timeout','40','--completion-timeout','5']
for case in oracle['cases']:cmd+=['--instruction',case['instruction']]
log=(r/'t1_initial05_01.log').open('x');job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); (r/'t1_initial05_01_job.json').write_text(json.dumps({'pid':job.pid,'output':str(out),'oracle':str(op),'log':str(r/'t1_initial05_01.log'),'started_at':time.time(),'image_binding':s['scene_image_binding']},indent=2));print('T1_05_STARTED',job.pid,flush=True)
