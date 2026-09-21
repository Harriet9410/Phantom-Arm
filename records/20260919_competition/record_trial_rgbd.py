#!/usr/bin/env python3
"""Read-only, timestamp-matched RGB-D evidence for replaying transit validation."""
import json
import queue
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String

rospy.init_node('tcei_rgbd_recorder',anonymous=True)
out=Path(rospy.get_param('~output'))
out.mkdir(parents=True,exist_ok=False)
bridge=CvBridge();lock=threading.Lock();jobs=queue.Queue(maxsize=60)
info=None;until=0.;last=0.;reference=False;sequence=0


def on_info(msg):
    global info
    with lock:info={'K':list(msg.K),'width':msg.width,'height':msg.height}


def on_event(msg):
    global until
    event=json.loads(msg.data)
    if event.get('status') in ('attempt','descend','release_started','trial_lift_started','trial_lift_evidence','grasp_contact_confirmed','grasp_verified','arrived','drop_detected','drop_recovery_started','placement_verified'):
        with lock:until=time.monotonic()+30
    with open(out/'events.jsonl','a') as f:f.write(msg.data+'\n')


def on_pair(rgb,depth):
    global last,reference
    now=time.monotonic()
    with lock:
        camera_info=info
        if camera_info is None or now-last<.1:return
        if reference and now>=until:return
        is_reference=not reference;reference=True;last=now
    color=bridge.imgmsg_to_cv2(rgb,'bgr8').copy()
    raw_depth=bridge.imgmsg_to_cv2(depth,'passthrough').astype(np.float32)
    if depth.encoding=='16UC1':raw_depth*=.001
    elif depth.encoding!='32FC1':raise ValueError('unexpected depth encoding')
    metadata={'captured_at':time.time(),'stamp':rgb.header.stamp.to_sec(),
              'sync_delta':abs((rgb.header.stamp-depth.header.stamp).to_sec()),'camera':camera_info}
    try:jobs.put_nowait((is_reference,color,raw_depth,metadata))
    except queue.Full:rospy.logerr('RGB-D evidence queue full; dropped pair')


rospy.Subscriber('/Jaka/camera/camera_info',CameraInfo,on_info,queue_size=1)
rospy.Subscriber('/tcei/task_status',String,on_event,queue_size=20)
rgb=message_filters.Subscriber('/Jaka/camera/rgb',Image)
depth=message_filters.Subscriber('/Jaka/camera/depth',Image)
sync=message_filters.ApproximateTimeSynchronizer([rgb,depth],5,.04)
sync.registerCallback(on_pair)
rospy.loginfo('read-only paired RGB-D recorder ready: %s',out)
while not rospy.is_shutdown():
    try:is_reference,color,raw_depth,metadata=jobs.get(timeout=.2)
    except queue.Empty:continue
    try:
        sequence+=1;stem='reference' if is_reference else 'pair_%04d'%sequence
        cv2.imwrite(str(out/(stem+'.jpg')),color,[cv2.IMWRITE_JPEG_QUALITY,92])
        np.savez_compressed(out/(stem+'.npz'),depth=raw_depth)
        metadata['stem']=stem
        with open(out/'index.jsonl','a') as f:f.write(json.dumps(metadata)+'\n')
    finally:jobs.task_done()
