from pathlib import Path
import sys,json,time,threading,collections,traceback
import numpy as np,cv2,rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
r=Path('/root/tcei_competition_20260919');release=r/'releases/dev_v7_t1_04';sys.path.insert(0,str(release/'tcei_stack'))
from robot_projection import RobotProjector
d=r/'projection_initial04_01';d.mkdir(exist_ok=False);result={'robot_commands_sent':False,'calibration_verified':False,'usable_for_control':False}
try:
 model=RobotProjector('/root/EAICON/Content/JAKA/robot/JAKA_C5_With_DH_PGI.urdf',package_paths={'Meshes':'/root/EAICON/Content/JAKA/robot/Meshes'})
 result['model_issues']=model.issues;assert not model.issues,str(model.issues)
 calibration=json.loads((release/'calibration/robot_projection.pending.json').read_text());assert calibration['verified'] is False
 rospy.init_node('tcei_projection_debug',anonymous=True);lock=threading.Lock();rgb=collections.deque(maxlen=10);diagnostics=collections.deque(maxlen=80);info=[]
 def on_rgb(m):
  with lock:rgb.append(m)
 def on_diag(m):
  with lock:diagnostics.append(json.loads(m.data))
 def on_info(m):
  with lock:info[:]=[m]
 rospy.Subscriber('/Jaka/camera/rgb',Image,on_rgb,queue_size=2,buff_size=2**24);rospy.Subscriber('/Jaka/camera/camera_info',CameraInfo,on_info,queue_size=1);rospy.Subscriber('/tcei/joint_diagnostics',String,on_diag,queue_size=40)
 deadline=time.monotonic()+15;pair=None
 while time.monotonic()<deadline and not rospy.is_shutdown():
  with lock:
   if rgb and diagnostics and info:
    msg=rgb[-1];stamp=msg.header.stamp.to_sec();q=min(diagnostics,key=lambda row:abs(row['simulation_time']-stamp))
    if abs(q['simulation_time']-stamp)<=.05 and 0<=time.time()-q['time']<2:pair=(msg,q,info[0]);break
  time.sleep(.02)
 assert pair is not None,'synchronized feedback unavailable'
 msg,q,k=pair;image=CvBridge().imgmsg_to_cv2(msg,'bgr8');joints=dict(zip(q['all_joint_names'],q['all_joint_positions']))
 proof=model.project_debug(joints,list(k.K),[k.width,k.height],joint_stamp=q['simulation_time'],image_stamp=msg.header.stamp.to_sec(),calibration=calibration)
 mask=proof.pop('debug_mask',None);proof.pop('mask',None);result.update(projection=proof,joint_diagnostics=q,camera={'K':list(k.K),'width':k.width,'height':k.height,'stamp':msg.header.stamp.to_sec(),'frame_id':msg.header.frame_id})
 assert mask is not None,proof['reason'];assert proof['usable_for_control'] is False
 overlay=image.copy();overlay[mask]=(.6*overlay[mask]+.4*np.array([0,0,255])).astype(np.uint8)
 contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(0,255,0),1);cv2.putText(overlay,'UNVERIFIED ROBOT PROJECTION - DEBUG ONLY',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
 cv2.imwrite(str(d/'raw_rgb.png'),image);cv2.imwrite(str(d/'robot_overlay.png'),overlay);cv2.imwrite(str(d/'debug_mask.png'),mask.astype(np.uint8)*255)
 root_camera=np.asarray(calibration['base_to_camera']);camera_world=np.asarray(calibration['camera_optical_from_world']);world_tcp=np.linalg.inv(camera_world)@root_camera@model.forward_kinematics(joints)['gripper_center']
 result.update(status='debug_projection_created',reported_tcp_consistency_error_m=float(np.linalg.norm(world_tcp[:3,3]-np.asarray(q['tcp_actual']))),stationary_max_joint_speed=float(np.max(np.abs(q['all_joint_velocities']))))
except Exception as error:result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
finally:
 (d/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(r/'projection_initial04_01_summary.txt').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('PROJECTION_DEBUG_RESULT',result['status'],flush=True)
