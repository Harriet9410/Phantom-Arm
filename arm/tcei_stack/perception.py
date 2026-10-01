#!/usr/bin/env python3
"""Synchronized RGB-D scene, persistent identities and explicit unknown regions.

The numbered full image retains measured layout. The legacy crop sheet is only
an appearance aid and never supplies spatial relationships. No robot commands.
"""
import copy
import json
import threading
import time
import cv2
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from ultralytics import YOLO
from conveyor_observer import DepthConveyorTracker
from grasp_geometry import body_center
from core import in_source_bin
from rotation_perception import RotationDetector
from observation_view import spatial_context,normalized_xy
from perception_tracking import CandidateTracker
from frame_pairing import image_payload_binding


class Perception:
    def __init__(self):
        self.bridge = CvBridge()
        self.yolo_enabled=bool(rospy.get_param('~yolo_enabled',True))
        self.detector_mode=str(rospy.get_param('~detector','yolo'))
        self.nine_detector=None
        if self.yolo_enabled:
            self.model = YOLO(rospy.get_param('~weights', '/root/jaka/best.pt'))
            self.rotation_detector = RotationDetector(self.model)
        else:
            # 加分合规形态：YOLO 权重不加载（可审计），分类走九格（nine_classify）。
            self.model = None
            from nine_classify import NineRotationDetector
            self.rotation_detector = NineRotationDetector(timeout=float(rospy.get_param('~classify_timeout',10.)))
            self.nine_detector = self.rotation_detector
            self.detector_mode = 'nine'
        self.identities=CandidateTracker()
        self.identity_lock=threading.Lock()
        self.show_gui=bool(rospy.get_param('~show_gui',False))
        self.max_frame_age=min(2.,max(.1,float(rospy.get_param('~max_frame_age',2.))))
        self.frame_epoch=0;self.round_id=None;self.capture_barrier_wall=None;self.capture_barrier_stamp=None
        self.simulation_clock=None;self.clock_received_wall=0.
        self.round_published_frames=0;self.clock_fault=False
        self.info = None
        self.lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = None
        self.frame_ready = threading.Event()
        self.gui_lock = threading.Lock()
        self.gui_frames = None
        self.transport_lock = threading.Lock()
        self.transport = DepthConveyorTracker()
        self.last = 0.
        self.seq = 0
        self.max_fps = float(rospy.get_param('~max_fps',10.))
        self.pub = rospy.Publisher('/tcei/candidates', String, queue_size=1)
        self.identity_pub=rospy.Publisher('/tcei/identity_status',String,queue_size=10)
        self.sheet_pub = rospy.Publisher('/tcei/contact_sheet', Image, queue_size=1)
        self.img_pub = rospy.Publisher('/tcei/annotated_image', Image, queue_size=1)
        rospy.Subscriber('/Jaka/camera/camera_info', CameraInfo, self.on_info, queue_size=1)
        rospy.Subscriber('/clock',Clock,self.on_clock,queue_size=1)
        self.rgb = message_filters.Subscriber('/Jaka/camera/rgb', Image,queue_size=1,buff_size=2**24)
        self.depth = message_filters.Subscriber('/Jaka/camera/depth', Image,queue_size=1,buff_size=2**24)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb, self.depth], 3, .08)
        self.sync.registerCallback(self.on_frame)
        rospy.Subscriber('/tcei/task_status',String,self.on_task_status,queue_size=20)
        rospy.loginfo('TCEI perception ready; detector=%s classes=%s', self.detector_mode,
                      getattr(self.model,'names','nine_grid_classification'))
        self.worker=threading.Thread(target=self.process_latest,daemon=True)
        self.worker.start()

    def on_info(self, msg):
        self.info = msg

    def on_clock(self,msg):
        stamp=float(msg.clock.to_sec())
        if not np.isfinite(stamp):return
        with self.identity_lock:
            previous=self.simulation_clock
            self.simulation_clock=stamp;self.clock_received_wall=time.time()
            if self.round_id is not None and self.capture_barrier_stamp is None:
                self.capture_barrier_stamp=stamp;self.identities.round_image_after=stamp
            elif self.round_id is not None and previous is not None and stamp<previous:
                self.clock_fault=True

    def frame_is_current(self,received_at,image_stamp,epoch):
        if epoch!=getattr(self,'frame_epoch',0):return False
        barrier=getattr(self,'capture_barrier_wall',None)
        if barrier is not None and received_at<barrier:return False
        if getattr(self,'round_id',None) is not None:
            if self.clock_fault:return False
            image_barrier=self.capture_barrier_stamp;clock=self.simulation_clock
            if image_barrier is None or clock is None or time.time()-self.clock_received_wall>1.5:return False
            if not image_barrier<image_stamp<=clock+.08:return False
        return True

    def on_task_status(self,msg):
        event=json.loads(msg.data)
        new_round=event.get('status')=='round_started' and event.get('round_id') is not None and event['round_id']!=self.round_id
        with self.identity_lock:
            if new_round:
                self.frame_epoch+=1;self.round_id=event['round_id'];self.capture_barrier_wall=event.get('time',time.time())
                self.round_published_frames=0;self.clock_fault=False
                self.simulation_clock=None;self.clock_received_wall=0.
                stamp=event.get('simulation_time')
                self.capture_barrier_stamp=float(stamp) if isinstance(stamp,(int,float)) and np.isfinite(stamp) else None
            ack=self.identities.on_event(event)
        with self.transport_lock:self.transport.on_event(event)
        if new_round:
            with self.pending_lock:self.pending=None;self.frame_ready.clear()
        if ack is not None:
            ack['published_at']=time.time()
            self.identity_pub.publish(String(json.dumps(ack)))

    def on_frame(self, rgb_msg, depth_msg):
        # Subscriber callbacks never perform inference. New camera pairs replace
        # pending work so expensive rotation views cannot accumulate old images.
        with self.pending_lock:
            captured_at=time.time();epoch=self.frame_epoch
            if not self.frame_is_current(captured_at,rgb_msg.header.stamp.to_sec(),epoch):return
            self.pending=(rgb_msg,depth_msg,captured_at,epoch)
            self.frame_ready.set()

    def process_latest(self):
        while not rospy.is_shutdown():
            if not self.frame_ready.wait(.1):continue
            remaining=1./self.max_fps-(time.monotonic()-self.last)
            if remaining>0:time.sleep(remaining)
            with self.pending_lock:
                pending=self.pending;self.pending=None;self.frame_ready.clear()
            if pending is not None:self.process_frame(*pending)

    def process_frame(self, rgb_msg, depth_msg, received_at, frame_epoch=None):
        if frame_epoch is None:frame_epoch=getattr(self,'frame_epoch',0)
        if not self.frame_is_current(received_at,rgb_msg.header.stamp.to_sec(),frame_epoch):return
        if self.info is None or time.monotonic() - self.last < 1. / self.max_fps:
            return
        if not self.lock.acquire(False):
            return
        try:
            self.last = time.monotonic()
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8').copy()
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
            if depth_msg.encoding == '16UC1':
                depth *= .001
            elif depth_msg.encoding != '32FC1':
                raise ValueError('unsupported depth encoding: '+depth_msg.encoding)
            if rgb.shape[:2] != depth.shape or self.info.width != rgb.shape[1]:
                raise ValueError('RGB/depth/intrinsics size mismatch')
            if self.detector_mode=='nine' or not self.yolo_enabled:
                if self.nine_detector is None:
                    from nine_classify import NineRotationDetector
                    self.nine_detector=NineRotationDetector(self.model if self.yolo_enabled else None,
                                                            timeout=float(rospy.get_param('~classify_timeout',10.)))
                source,boxes,recognition_metrics = self.nine_detector.detect(rgb,depth,self.info.K)
            else:
                source,boxes,recognition_metrics = self.rotation_detector.detect(rgb,depth,self.info.K)
            h, w = depth.shape
            k = self.info.K
            context=spatial_context(k,depth.shape)
            observations = []
            R=np.array([[.99887577,-.04588472,-.01190774],[.04628524,.99828373,.03587854],[.01024103,-.03638936,.99928521]])
            for c in source:
                u,v=c['pixel'];z=c['depth']
                xc,yc=(u-k[2])*z/k[0],(v-k[5])*z/k[4]
                p=R@np.array([xc,yc,z])+np.array([.0470183,.03276388,.35385047]);p[0]=-p[0]
                c['position']=p.tolist();observations.append(c)
            boxes = sorted(boxes,key=lambda b:b['bbox'][0])
            for box in boxes:
                if box['class'] is None:continue
                x1,y1,x2,y2 = map(int, box['bbox'])
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                if x2-x1 < 4 or y2-y1 < 4: continue
                u,v = (x1+x2)/2, (y1+y2)/2
                rx,ry = max(2,(x2-x1)//4), max(2,(y2-y1)//4)
                patch = depth[max(0,int(v)-ry):min(h,int(v)+ry+1),max(0,int(u)-rx):min(w,int(u)+rx+1)]
                good = patch[np.isfinite(patch) & (patch>.1) & (patch<5)]
                if good.size < max(5, .6*patch.size): continue
                z = float(np.median(good))
                wx=-(u-k[2])*z/k[0];wy=(v-k[5])*z/k[4]+.1
                if abs(wx)<.38 and -.32<wy<.22:
                    # Source picks require a measured foreground body. Base
                    # detections are retained outside it for transport audits.
                    continue
                spread = float(np.percentile(good,90)-np.percentile(good,10))
                if spread > .08: continue
                sx1,sy1,sx2,sy2=max(0,x1-8),max(0,y1-8),min(w,x2+8),min(h,y2+8)
                surrounding=depth[sy1:sy2,sx1:sx2]
                ring=np.ones(surrounding.shape,dtype=bool)
                ring[y1-sy1:y2-sy1,x1-sx1:x2-sx1]=False
                values=surrounding[ring & np.isfinite(surrounding) & (surrounding>.1) & (surrounding<5)]
                if len(values)<12:continue
                surface_z=4.5-z
                support_z=4.5-float(np.percentile(values,80))
                grasp_method='box_center'
                if box['class'] in ('Grenade','Smokegrenade','Torch','Magazine'):
                    try:u,v,z=body_center(depth,[x1,y1,x2,y2],support_z)
                    except ValueError:continue
                    grasp_method='depth_body_core'
                roi = rgb[y1:y2,x1:x2]
                gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                contours,_ = cv2.findContours(cv2.Canny(gray,50,150),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                angle = 0.
                if contours:
                    (_, _),(rw,rh),a = cv2.minAreaRect(max(contours,key=cv2.contourArea))
                    angle = float(a if rw < rh else a-90)
                xc,yc = (u-k[2])*z/k[0], (v-k[5])*z/k[4]
                R=np.array([[.99887577,-.04588472,-.01190774],[.04628524,.99828373,.03587854],[.01024103,-.03638936,.99928521]])
                p=R@np.array([xc,yc,z])+np.array([.0470183,.03276388,.35385047])
                p[0] = -p[0]
                observations.append({'class':box['class'],
                  'confidence':box['confidence'],'pixel':[u,v],'bbox':[x1,y1,x2,y2],
                  'depth':z,'depth_spread':spread,'angle_deg':angle,'position':p.tolist(),
                  # Read-only geometry audit: optical camera axes to USD world.
                  # Keep the historical grasp calibration separate from this.
                  'world_position':[-xc,yc+.1,4.5-z],'support_z':support_z,'grasp_point_method':grasp_method})
            # Source components already come from the calibrated basket mask.
            # Do not reuse the old 2.50 m height cut-off: standing/stacked
            # objects remain visible candidates with grasp_ready=False.
            candidates=[dict(o) for o in source if o['confidence']>=.60]
            for i,c in enumerate(candidates):
                c['id']=str(i+1)
                c['image_stamp']=rgb_msg.header.stamp.to_sec()
            with self.identity_lock:
                if not self.frame_is_current(received_at,rgb_msg.header.stamp.to_sec(),frame_epoch):return
                with self.transport_lock:
                    tracked=self.transport.process(depth,k,received_at,rgb_msg.header.stamp.to_sec())
            observations.extend(tracked)
            self.seq += 1
            unknown=list(recognition_metrics.get('unknown_regions',[]))
            with self.identity_lock:
                if not self.frame_is_current(received_at,rgb_msg.header.stamp.to_sec(),frame_epoch):return
                if time.time()-received_at>=self.max_frame_age:
                    for candidate in candidates:
                        candidate.update(stable_id=None,identity_status='stale',identity_candidates=[],source_frame_id=self.seq)
                else:
                    candidates,unknown=self.identities.update(candidates,self.seq,received_at,unknown,
                                                             image_stamp=rgb_msg.header.stamp.to_sec())
                source_lifecycle=self.identities.source_lifecycle()
            for region in unknown:
                if region.get('pixel') is None and region.get('bbox'):
                    a,b,c,d=region['bbox'];region['pixel']=[(a+c)/2.,(b+d)/2.]
                if region.get('pixel') is not None:
                    region['normalized_xy']=normalized_xy(region['pixel'],context['basket_roi'])
            annotated=rgb.copy()
            x1,y1,x2,y2=map(round,context['basket_roi'])
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(190,150,30),1)
            cv2.putText(annotated,'CALIBRATED BASKET / IMAGE UP',(x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.42,(160,110,20),1)
            for index,region in enumerate(unknown):
                if not region.get('bbox'):continue
                a,b,c,d=map(lambda v:int(round(v)),region['bbox'])
                cv2.rectangle(annotated,(a,b),(c,d),(0,120,240),1)
                cv2.putText(annotated,'UNKNOWN '+str(index+1),(a,min(h-5,d+13)),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,80,200),1)
            for tracked_object in tracked:
                a,b,c,d=tracked_object['bbox']
                cv2.rectangle(annotated,(a,b),(c,d),(0,190,255),2)
                cv2.putText(annotated,'TRACK '+tracked_object['class'],(a,max(20,b-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,100,180),2)
            rows=max(1,(len(candidates)+2)//3)
            sheet=np.full((rows*280,3*280,3),245,np.uint8)
            for i,c in enumerate(candidates):
                x1,y1,x2,y2=c['bbox']
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,200,0),2)
                label=c['id']
                cv2.putText(annotated,label,(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
                margin=12
                crop=rgb[max(0,y1-margin):min(h,y2+margin),max(0,x1-margin):min(w,x2+margin)]
                ch,cw=crop.shape[:2]; scale=min(230/cw,220/ch)
                crop=cv2.resize(crop,(max(1,int(cw*scale)),max(1,int(ch*scale))))
                oy=(i//3)*280+48; ox=(i%3)*280+(280-crop.shape[1])//2
                sheet[oy:oy+crop.shape[0],ox:ox+crop.shape[1]]=crop
                cv2.putText(sheet,c['id'],((i%3)*280+90,(i//3)*280+30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
            now=time.time()
            sensor_issues=[]
            if now-received_at>=self.max_frame_age:sensor_issues.append('processing_stale')
            if abs((rgb_msg.header.stamp-depth_msg.header.stamp).to_sec())>.08:sensor_issues.append('rgb_depth_unsynchronized')
            for issue in sensor_issues:unknown.append({'bbox':context['basket_roi'],'reason':issue})
            complete=(bool(recognition_metrics.get('coverage_complete',False)) and not unknown
                      and all(c['identity_status']=='confirmed' for c in candidates))
            data={'frame_id':self.seq,'observed_at':received_at,'published_at':now,
                  'round_id':getattr(self,'round_id',None),'frame_epoch':frame_epoch,
                  'observed_at_kind':'camera_pair_received_wall','camera_capture_stamp':rgb_msg.header.stamp.to_sec(),
                  'freshness_limit_seconds':self.max_frame_age,
                  'processing_age_seconds':now-received_at,'stamp':rgb_msg.header.stamp.to_sec(),
                  'sync_delta':abs((rgb_msg.header.stamp-depth_msg.header.stamp).to_sec()),
                  'image_size':[w,h],'candidates':candidates,'observations':observations,
                  'recognition':recognition_metrics,'unknown_regions':unknown,
                  'spatial_context':context,'coverage_complete':complete,'scene_complete':complete,
                  'sensor_issues':sensor_issues,'visibility':recognition_metrics.get('visibility'),
                  'model_capabilities':self.rotation_detector.capabilities,
                  'identity_scope':'this explicit round; verified delivery required for retirement',
                  'coverage_scope':'visible_foreground_only; hidden fully occluded objects cannot be excluded',
                  'scene_image_topic':'/tcei/annotated_image',
                  'scene_image_binding':image_payload_binding(annotated,rgb_msg.header.stamp,rgb_msg.header.frame_id),
                  'crop_sheet_role':'appearance_only_not_spatial'}
            data.update(source_lifecycle)
            with self.transport_lock:data['transport_status']=dict(getattr(self.transport,'diagnostics',{}))
            with self.identity_lock:
                if not self.frame_is_current(received_at,rgb_msg.header.stamp.to_sec(),frame_epoch):return
                for arr,pub in [(sheet,self.sheet_pub),(annotated,self.img_pub)]:
                    msg=self.bridge.cv2_to_imgmsg(arr,'bgr8')
                    # rospy may rewrite seq in-place on serialization. Neither
                    # publisher may share or mutate the original camera header.
                    msg.header=copy.deepcopy(rgb_msg.header)
                    pub.publish(msg)
                self.pub.publish(String(json.dumps(data)))
                self.round_published_frames+=1
            # ApproximateTime callbacks can run on either ROS subscriber thread.
            # Qt must remain on the main thread or the second frame can deadlock.
            with self.gui_lock:
                self.gui_frames = (annotated, sheet)
            rospy.loginfo_throttle(10,'frame=%d candidates=%d sync=%.3fs',self.seq,len(candidates),data['sync_delta'])
        except Exception as e:
            rospy.logerr_throttle(3,'perception rejected frame: %s',e)
        finally:
            self.lock.release()

    def run(self):
        while not rospy.is_shutdown():
            with self.gui_lock:
                frames = self.gui_frames
            if frames is not None and self.show_gui:
                cv2.imshow('TCEI clean perception',frames[0])
                cv2.imshow('TCEI Nine candidate crops',frames[1])
                cv2.waitKey(1)
            time.sleep(.04)
        if self.show_gui:cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('tcei_perception')
    node=Perception()
    node.run()
