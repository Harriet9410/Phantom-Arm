from pathlib import Path
import json,time,threading,hashlib,struct,subprocess,copy
import rospy,cv2,numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
r=Path('/root/tcei_competition_20260919'); d=r/'t1_capture06_clear01'; d.mkdir(exist_ok=False); rospy.init_node('tcei_capture06_clear',anonymous=True); scenes={};images={};lock=threading.Lock();bridge=CvBridge()
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
cv2.imwrite(str(d/'numbered_scene.png'),arr);(d/'scene.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
print('CLEAR_CAPTURE',s['scene_complete'],len(s['candidates']),len(s['unknown_regions']),flush=True)
