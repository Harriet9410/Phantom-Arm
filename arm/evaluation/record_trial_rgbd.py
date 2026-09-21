#!/usr/bin/env python3
"""Read-only, timestamp-matched RGB-D evidence for replaying transit validation."""
import json
import queue
import signal
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
from rgbd_archive import DepthSeriesWriter

rospy.init_node('tcei_rgbd_recorder',anonymous=True,disable_signals=True)
duration=float(rospy.get_param('~duration',900.))
if not 1.<=duration<=1800.:raise ValueError('duration must be 1..1800 wall seconds')
expires=time.monotonic()+duration
out=Path(rospy.get_param('~output'))
out.mkdir(parents=True,exist_ok=False)
depth_writer=DepthSeriesWriter()
bridge=CvBridge();lock=threading.Lock();jobs=queue.Queue(maxsize=60)
info=None;until=0.;last=0.;reference=False;sequence=0
capture_window=float(rospy.get_param('~capture_window_seconds',0.))
if not 0<=capture_window<=15.:raise ValueError('read-only capture window must be 0..15 seconds')
if capture_window:until=time.monotonic()+capture_window
accepting=True;dropped=0;write_errors=[];stop_requested=threading.Event()
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop_requested.set())


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
    global last,reference,dropped
    now=time.monotonic()
    with lock:
        camera_info=info
        if not accepting or camera_info is None or now-last<.1:return
        if reference and now>=until:return
        is_reference=not reference;reference=True;last=now
    color=bridge.imgmsg_to_cv2(rgb,'bgr8').copy()
    raw_depth=bridge.imgmsg_to_cv2(depth,'passthrough').astype(np.float32)
    if depth.encoding=='16UC1':raw_depth*=.001
    elif depth.encoding!='32FC1':raise ValueError('unexpected depth encoding')
    metadata={'captured_at':time.time(),'stamp':rgb.header.stamp.to_sec(),
              'sync_delta':abs((rgb.header.stamp-depth.header.stamp).to_sec()),'camera':camera_info}
    with lock:
        if not accepting:return
        try:jobs.put_nowait((is_reference,color,raw_depth,metadata))
        except queue.Full:
            dropped+=1;rospy.logerr('RGB-D evidence queue full; dropped pair')


subscriptions=[rospy.Subscriber('/Jaka/camera/camera_info',CameraInfo,on_info,queue_size=1),
               rospy.Subscriber('/tcei/task_status',String,on_event,queue_size=20)]
rgb=message_filters.Subscriber('/Jaka/camera/rgb',Image)
depth=message_filters.Subscriber('/Jaka/camera/depth',Image)
sync=message_filters.ApproximateTimeSynchronizer([rgb,depth],5,.04)
sync.registerCallback(on_pair)
rospy.loginfo('read-only paired RGB-D recorder ready: %s',out)
drain_deadline=None;reason=None;received_end=None
try:
    while True:
        if accepting and (stop_requested.is_set() or rospy.is_shutdown() or time.monotonic()>=expires):
            with lock:accepting=False
            received_end=time.time();reason='requested_stop' if stop_requested.is_set() else 'bounded_duration_or_ros_shutdown'
            for sub in subscriptions:sub.unregister()
            rgb.unregister();depth.unregister();drain_deadline=time.monotonic()+15.
        if not accepting and (jobs.empty() or time.monotonic()>=drain_deadline):break
        try:is_reference,color,raw_depth,metadata=jobs.get(timeout=.2)
        except queue.Empty:continue
        try:
            sequence+=1;stem='reference' if is_reference else 'pair_%04d'%sequence
            if not cv2.imwrite(str(out/(stem+'.jpg')),color,[cv2.IMWRITE_JPEG_QUALITY,92]):
                raise IOError('RGB image write failed')
            depth_writer.write(out/(stem+'.npz'),raw_depth)
            metadata['stem']=stem
            with open(out/'index.jsonl','a') as f:f.write(json.dumps(metadata)+'\n')
        except Exception as error:
            write_errors.append({'stem':stem,'error':str(error)});rospy.logerr('RGB-D archive write failed: %s',error)
        finally:jobs.task_done()
finally:
    with lock:accepting=False
    for sub in subscriptions:sub.unregister()
    rgb.unregister();depth.unregister()
    (out/'closed.json').write_text(json.dumps({'closed_at':time.time(),'receiving_ended_at':received_end,
        'maximum_duration_seconds':duration,'stop_reason':reason,'recorded_pairs':sequence-len(write_errors),
        'pending_pairs':jobs.qsize(),'dropped_pairs':dropped,'write_errors':write_errors,
        'complete':reason is not None and jobs.empty() and dropped==0 and not write_errors,
        'capture_window_seconds':capture_window,'depth_encoding':'float32_xor_deflate_level1_v1'},indent=2))
    rospy.signal_shutdown('recorder flushed or bounded drain reached')
