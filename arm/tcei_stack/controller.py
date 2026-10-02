#!/usr/bin/env python3
"""Bounded state machine for the supplied pure Isaac simulation.

No simulator physics edits. Motion cancellation uses the V7 simulator's measured
stop acknowledgement; withdrawing commands alone never counts as a stopped arm.
"""
import copy
import collections
import hashlib
import json
import math
import os
import queue
import threading
import time
import uuid
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image,CameraInfo,JointState
from cv_bridge import CvBridge
from arrival_window import ArrivalWindow
from core import rebind_candidate, legacy_quaternion, quaternion_error, validate_workspace, verify_transport, calibrated_grasp, reacquire_after_empty, verify_settled_placement, occlusion_depth,grasp_approach_heights,empty_finger_lowest_z
from frame_matching import matching_depth,reference_consistency
from observation_view import OBSERVATION_POSITION,TRANSFER_POSITION,basket_visibility
from drop_recovery import pickable_recovery_objects,reacquire_dropped
from contact_loss import ContactLossWindow
from gripper_closure import ContactConfirmationWindow,next_closure_target
from grasp_contact_monitor import GraspContactMonitor,ContactLiftWindow,EFFORT_PROTOCOL
from ros_endpoints import CANCEL_PROTOCOL,OBSERVATION_PROTOCOL,cancel_payload
from robot_projection import RobotProjector
from lift_evidence import evaluate_lift_frame,LiftEvidenceWindow,ContactSurfaceWindow
from grasp_candidates import generate_grasp_candidates,GraspAttemptHistory


class PlanningRejected(RuntimeError):
    pass


class GraspLost(RuntimeError):
    pass


class EmptyGrasp(RuntimeError):
    pass


class LiftUnverified(GraspLost):
    pass


class LiftUnobservable(RuntimeError):
    """Hold the object and stop for verification; not permission to open/regrasp."""
    pass


class Controller:
    def __init__(self):
        self.lock=threading.RLock();self.pose=None;self.pose_seen=0.
        self._init_command_protocol()
        self.captured=False;self.capture_seen=0.;self.snapshot=None
        self.simulation_time=None;self.clock_seen=0.
        self.depth=None;self.depth_seen=0.;self.depth_stamp=None;self.depth_bridge=CvBridge()
        # At the capped20Hz input rate, retain more than the scene's2s
        # freshness allowance;24 frames could evict a still-valid YOLO pair.
        self.depth_history=collections.deque(maxlen=48)
        self.scene_history=collections.deque(maxlen=1600)
        self.planner_status=None
        self.stop_id=None;self.stop_ack=None;self.stop_seen=0.
        self.gripper_efforts=None;self.efforts_seen=0.;self.joint_feedback=None;self.joints_seen=0.
        self.contact_confirmation=None
        self.contact_monitor=GraspContactMonitor(
            on_threshold=float(rospy.get_param('~contact_threshold',.2)),
            off_threshold=float(rospy.get_param('~loss_effort_threshold',.05)))
        self.contact_payload=None
        self.contact_history=collections.deque(maxlen=128)
        self.joint_diagnostics=collections.deque(maxlen=256)
        self.identity_acks={};self.trial_lift=None;self.last_contact_sim_time=None
        self.robot_projector=None;self.projection_calibration=None;self.projection_error='not loaded'
        self.grasp_history=GraspAttemptHistory();self.active_attempt_id=None;self.active_lift_verified=False
        self.camera_info=None;self.current_round=None;self.current_task=None
        self.active_object=None;self.active_side=None
        self.pending_deliveries={};self.last_pending_delivery_check=0.
        self.active_release_id=None
        self.active_release_context=None;self.mission_wall_offset=None
        self.require_planner=bool(rospy.get_param('~require_planner_feedback',False))
        self.arrival_seconds=float(rospy.get_param('~arrival_seconds',.2))
        ArrivalWindow(self.arrival_seconds)
        self.queue=queue.Queue(maxsize=1);self.busy=False;self.seen=set()
        self.cancelled=threading.Event();self.needs_recovery=False
        self.current_request=None;self.grip_value=0.;self.deadline=0.
        self.contact_loss=ContactLossWindow()
        with open(__file__,'rb') as f:self.code_sha256=hashlib.sha256(f.read()).hexdigest()
        self.ee_pub=rospy.Publisher('/Jaka/set_end_effector_pose',PoseStamped,queue_size=1)
        self.bound_preview_pub=rospy.Publisher('/tcei/execute_preview',String,queue_size=1)
        self.stop_pub=rospy.Publisher('/tcei/stop_request',String,queue_size=1)
        self.stop_reset_pub=rospy.Publisher('/tcei/stop_reset',String,queue_size=1)
        self.cancel_ack_pub=rospy.Publisher('/tcei/cancel_ack',String,queue_size=10)
        self.preview_pub=rospy.Publisher('/tcei/grasp_preview_request',String,queue_size=1)
        self.preview_response=None
        rospy.Subscriber('/tcei/grasp_preview_response',String,self.on_preview_response,queue_size=1)
        self.grip_pub=rospy.Publisher('/Jaka/set_gripper_value',Float32,queue_size=1)
        self.event_pub=rospy.Publisher('/tcei/task_status',String,queue_size=10,latch=True)
        self.log_dir=rospy.get_param('~log_dir','/root/tcei_runs/20260917_2216/improved')
        os.makedirs(self.log_dir,exist_ok=True)
        rospy.Subscriber('/Jaka/get_end_effector_pose',PoseStamped,self.on_pose,queue_size=1)
        rospy.Subscriber('/Jaka/gripper_is_captured',Bool,self.on_capture,queue_size=100)
        rospy.Subscriber('/Jaka/get_gripper_efforts',Float32MultiArray,self.on_efforts,queue_size=100)
        rospy.Subscriber('/tcei/gripper_contact',String,self.on_contact_feedback,queue_size=100)
        rospy.Subscriber('/Jaka/get_jointstate',JointState,self.on_joints,queue_size=100)
        rospy.Subscriber('/tcei/joint_diagnostics',String,self.on_joint_diagnostics,queue_size=100)
        rospy.Subscriber('/tcei/identity_status',String,self.on_identity_status,queue_size=20)
        rospy.Subscriber('/Jaka/camera/camera_info',CameraInfo,self.on_camera_info,queue_size=1)
        rospy.Subscriber('/tcei/stop_ack',String,self.on_stop_ack,queue_size=10)
        rospy.Subscriber('/clock',Clock,self.on_clock,queue_size=1)
        rospy.Subscriber('/Jaka/camera/depth',Image,self.on_depth,queue_size=1,buff_size=2**24)
        rospy.Subscriber('/tcei/candidates',String,self.on_candidates,queue_size=1)
        rospy.Subscriber('/tcei/planner_status',String,self.on_planner,queue_size=10)
        rospy.Subscriber('/tcei/plan',String,self.on_plan,queue_size=1)
        rospy.Subscriber('/tcei/cancel',Bool,self.on_cancel,queue_size=1)
        rospy.Subscriber('/tcei/prepare_observation',String,self.on_prepare_observation,queue_size=1)
        rospy.Subscriber('/tcei/cancel_request',String,self.on_cancel_request,queue_size=10)
        self.home_p=list(OBSERVATION_POSITION)
        self.transfer_p=list(TRANSFER_POSITION)
        self.home_q=[.0146265891,-.0000924757,.9998736759,-.0062198575]
        self.load_projection_capability()
        self.event('ready',execute=rospy.get_param('~execute',False),
                   lift_evidence_capable=self.robot_projector is not None,
                   lift_evidence_capability_reason=self.projection_error)

    def _init_command_protocol(self):
        self.command_epoch=0;self.cancelled_rounds=set();self.cancel_requests={}
        self.admitted_request=None;self.admitted_context=None

    @staticmethod
    def request_context(request):
        request=request if isinstance(request,dict) else {}
        return {key:request.get(key) for key in ('request_id','round_id','task_id')}

    def enqueue_command(self,request):
        # Admission and cancellation share the same lock as command emission.
        with self.lock:
            if (self.busy or self.admitted_request is not None or not self.queue.empty() or
                    self.needs_recovery or self.stop_id or self.cancelled.is_set() or
                    request.get('round_id') in self.cancelled_rounds):
                raise ValueError('busy, cancelled, stopped or recovery required')
            rid=request['request_id']
            if rid in self.seen:raise ValueError('duplicate request')
            self.queue.put_nowait({**request,'_command_epoch':self.command_epoch})
            self.seen.add(rid);self.admitted_request=rid
            self.admitted_context=self.request_context(request)
            kind='observation_accepted' if request.get('kind')=='prepare_observation' else 'plan_accepted'
            self.event(kind,**self.admitted_context,command_epoch=self.command_epoch,
                       observation_protocol=OBSERVATION_PROTOCOL if kind=='observation_accepted' else None)

    def invalidate_commands(self,request_round=None):
        # Call under self.lock. Never clear this latch merely to start a queue item.
        self.command_epoch+=1;self.cancelled.set();self.needs_recovery=True
        for ident in (request_round,getattr(self,'current_round',None)):
            if ident:self.cancelled_rounds.add(ident)
        pending=getattr(self,'queue',None)
        while pending is not None:
            try:item=pending.get_nowait()
            except queue.Empty:break
            self.event('queued_request_cancelled',**self.request_context(item),command_epoch=self.command_epoch)
            pending.task_done()
        if not getattr(self,'busy',False):
            self.admitted_request=None;self.admitted_context=None

    def acknowledge_cancel(self,request,stop_id,revision=0):
        response={'protocol':CANCEL_PROTOCOL,'cancel_id':request['cancel_id'],
                  'request_round_id':request['round_id'],'request_id':request['request_id'],
                  'state':'accepted','stop_id':stop_id,'mapping_revision':revision,
                  'affected_round_id':getattr(self,'current_round',None),
                  'affected_request_id':getattr(self,'current_request',None),
                  'monotonic':time.monotonic(),'time':time.time()}
        self.cancel_requests[request['cancel_id']]={'request':copy.deepcopy(request),'response':response}
        self.cancel_ack_pub.publish(String(json.dumps(response)))
        return response

    def load_projection_capability(self):
        """Missing calibration disables grasps, not observation or stop control."""
        self.robot_projector=None;self.projection_calibration=None
        try:
            path=rospy.get_param('~robot_projection_calibration','')
            if not path:raise ValueError('robot_projection_calibration_not_configured')
            with open(path,'r') as stream:calibration=json.load(stream)
            if calibration.get('verified') is not True or not calibration.get('id'):
                raise ValueError('robot_projection_calibration_unverified')
            camera_world=np.asarray(calibration.get('camera_optical_from_world'),dtype=float)
            if camera_world.shape!=(4,4) or not np.isfinite(camera_world).all():raise ValueError('camera_optical_from_world_missing')
            rotation=camera_world[:3,:3]
            if not np.allclose(camera_world[3],[0,0,0,1],atol=1e-8) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or abs(np.linalg.det(rotation)-1)>1e-5:
                raise ValueError('camera_optical_from_world_not_rigid')
            model=RobotProjector(rospy.get_param('~robot_urdf','/root/EAICON/Content/JAKA/robot/JAKA_C5_With_DH_PGI.urdf'),
                package_paths={'Meshes':rospy.get_param('~robot_mesh_dir','/root/EAICON/Content/JAKA/robot/Meshes')})
            if model.issues:raise ValueError('robot_projection_geometry_incomplete: '+str(model.issues))
            if calibration.get('root_link')!=model.root_link:raise ValueError('robot_projection_root_mismatch')
            self.robot_projector=model;self.projection_calibration=calibration;self.projection_error=None
            self.event('lift_evidence_capability',available=True,calibration_id=calibration['id'],urdf_sha256=model.urdf_sha256)
        except (OSError,ValueError,TypeError,KeyError) as error:
            self.projection_error=str(error)
            self.event('lift_evidence_capability',available=False,reason=self.projection_error,
                       observation_and_measured_stop_remain_available=True)

    def prepare_observation(self,deadline=None):
        """Clear the camera view within the supplied competition deadline."""
        self.busy=True
        self.deadline=float(deadline) if deadline is not None else time.monotonic()+180
        began=None
        try:
            while True:
                self.checkpoint()
                with self.lock:
                    ready=self.pose is not None and time.monotonic()-self.pose_seen<1.5 and self.planner_status is not None and self.depth is not None
                if ready:break
                time.sleep(.1)
            info=rospy.wait_for_message('/Jaka/camera/camera_info',CameraInfo,timeout=10)
            self.checkpoint()
            if self.holding():raise RuntimeError('cannot prepare observation with occupied gripper')
            began=time.monotonic();started_at=time.time()
            rospy.set_param('~observation_started_monotonic',began)
            rospy.set_param('~observation_started_wall',started_at)
            self.event('observation_preparation_started',target_position=self.home_p)
            self.open_gripper()
            self.move(self.home_p,self.home_q,'clear_basket_for_observation')
            seen=0.;consecutive=0;visibility=None
            until=time.monotonic()+5
            while time.monotonic()<until:
                self.checkpoint()
                with self.lock:d=self.depth;received=self.depth_seen
                if received>seen:
                    seen=received;visibility=basket_visibility(d,info.K)
                    consecutive=consecutive+1 if visibility['clear'] else 0
                    if consecutive>=3:break
                time.sleep(.05)
            if consecutive<3:raise RuntimeError('basket still occluded at observation pose: '+str(visibility))
            elapsed=time.monotonic()-began
            rospy.set_param('~observation_position',self.home_p)
            rospy.set_param('~observation_preparation_seconds',elapsed)
            rospy.set_param('~observation_ready',True)
            self.event('ready',execute=False,observation_prepared=True,observation_position=self.home_p,
                       observation_started_at=started_at,observation_preparation_seconds=elapsed,visibility=visibility)
        except Exception as error:
            rospy.set_param('~observation_ready',False)
            self.needs_recovery=True;self.event('observation_preparation_failed',reason=str(error))
            raise
        finally:self.busy=False

    def event(self,status,**data):
        if status=='round_started':
            stamp=getattr(self,'simulation_time',None)
            if (isinstance(stamp,(int,float)) and math.isfinite(stamp)
                    and time.monotonic()-getattr(self,'clock_seen',-math.inf)<1.5):
                data.setdefault('simulation_time',stamp)
        if self.active_object is not None and status in (
                'drop_detected','holding_feedback_lost','grasp_verified',
                'release_started','released','placement_verified','placement_pending_verification'):
            data.setdefault('stable_id',self.active_object.get('stable_id'))
            data.setdefault('side',self.active_side)
            data.setdefault('category',self.active_object.get('class'))
        row={'time':time.time(),'monotonic':time.monotonic(),'request_id':self.current_request,
             'round_id':self.current_round,'task_id':self.current_task,
             'status':status,'code_sha256':self.code_sha256,**data}
        encoded=json.dumps(row,ensure_ascii=False)
        self.event_pub.publish(String(encoded));rospy.loginfo(encoded)
        with open(os.path.join(self.log_dir,'control_events.jsonl'),'a') as f:f.write(encoded+'\n')

    def on_pose(self,msg):
        with self.lock:self.pose=msg;self.pose_seen=time.monotonic()

    def on_efforts(self,msg):
        with self.lock:
            self.gripper_efforts=list(msg.data);self.efforts_seen=time.monotonic()
            window=getattr(self,'contact_confirmation',None)
            if window is not None:window.observe_efforts(self.gripper_efforts,self.efforts_seen)

    def on_contact_feedback(self,msg):
        try:
            value=json.loads(msg.data);samples=value['samples']
            if (value.get('protocol')!=EFFORT_PROTOCOL or len(samples)!=2 or
                    [row['finger'] for row in samples]!=['left','right']):
                raise ValueError('invalid native effort protocol or finger ordering')
            published=value['published_wall'];simulation=value['simulation_time']
            if (not isinstance(published,(int,float)) or not math.isfinite(published) or
                    not 0<=time.time()-published<=1.5 or not isinstance(simulation,(int,float)) or
                    not math.isfinite(simulation)):
                raise ValueError('native effort publication stale or invalid')
            if any(row['valid'] and (not isinstance(row['stamp'],(int,float)) or
                   not math.isfinite(row['stamp']) or abs(row['stamp']-simulation)>.15) for row in samples):
                raise ValueError('native effort acquisition differs from simulator clock')
            with self.lock:
                self.contact_monitor.observe(time.monotonic(),tuple(row['stamp'] for row in samples),
                    tuple(row['value'] for row in samples),tuple(row['valid'] for row in samples))
                self.contact_payload=value
                self.contact_history.append({'received':time.monotonic(),'proof':self.contact_monitor.snapshot(time.monotonic())})
        except (ValueError,KeyError,TypeError) as error:
            with self.lock:self.contact_monitor.fault='invalid native effort feedback: '+str(error)

    def on_joints(self,msg):
        with self.lock:
            self.joint_feedback=msg;self.joints_seen=time.monotonic()
            window=getattr(self,'contact_confirmation',None)
            if window is not None:self.record_closure_joints(window,msg,self.joints_seen)

    @staticmethod
    def record_closure_joints(window,msg,seen):
        try:window.observe_joints(list(msg.name),list(msg.position),msg.header.stamp.to_sec(),seen)
        except (AttributeError,TypeError,ValueError):window.fail('measured finger joint feedback invalid')

    def on_joint_diagnostics(self,msg):
        try:
            value=json.loads(msg.data);stamp=float(value['simulation_time'])
            names=value['all_joint_names'];positions=value['all_joint_positions']
            tcp=np.asarray(value['tcp_actual'],dtype=float)
            rotation=np.asarray(value['tcp_rotation_matrix'],dtype=float)
            if not math.isfinite(stamp) or len(names)!=len(positions) or len(set(names))!=len(names):return
            if not all(isinstance(n,str) for n in names) or not all(math.isfinite(p) for p in positions):return
            if tcp.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(tcp).all() or not np.isfinite(rotation).all():return
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or abs(np.linalg.det(rotation)-1.)>1e-4:return
            value['_received_monotonic']=time.monotonic()
            with self.lock:
                if self.joint_diagnostics and stamp<=self.joint_diagnostics[-1]['simulation_time']:return
                self.joint_diagnostics.append(value)
        except (ValueError,TypeError,KeyError):return

    def on_identity_status(self,msg):
        try:
            value=json.loads(msg.data);ident=value['reacquire_id']
            if not isinstance(ident,str):return
            value['_received_monotonic']=time.monotonic()
            with self.lock:self.identity_acks[ident]=value
        except (ValueError,TypeError,KeyError):return

    def paired_depth_diagnostics(self,after_stamp):
        with self.lock:depths=list(self.depth_history);feedback=list(self.joint_diagnostics)
        now=time.monotonic();pairs=[]
        for depth,received,stamp in depths:
            if stamp<=after_stamp or not 0<=now-received<1.2:continue
            possible=[d for d in feedback if 0<=now-d['_received_monotonic']<1.2 and abs(d['simulation_time']-stamp)<=.05]
            if possible:pairs.append((depth,stamp,min(possible,key=lambda d:abs(d['simulation_time']-stamp))))
        return sorted(pairs,key=lambda row:row[1])

    def begin_trial_lift(self,candidate):
        began=time.monotonic();baseline=None
        after=(self.last_contact_sim_time-.051) if self.last_contact_sim_time is not None else -math.inf
        while time.monotonic()-began<2.:
            self.checkpoint()
            with self.lock:
                samples=[d for d in self.joint_diagnostics if d['simulation_time']>after
                         and 0<=time.monotonic()-d['_received_monotonic']<1.2]
            if samples:
                baseline=copy.deepcopy(samples[-1]);stamp=baseline['simulation_time'];break
            time.sleep(.03)
        if baseline is None:raise RuntimeError('fresh actual robot baseline unavailable for trial lift')
        with self.lock:info=self.camera_info
        self.trial_lift={'candidate':copy.deepcopy(candidate),'baseline':copy.deepcopy(baseline),
            'baseline_stamp':stamp,'last_stamp':stamp,'window':LiftEvidenceWindow(stamp),
            'contact_surface_window':ContactSurfaceWindow(stamp),
            'contact_lift_window':ContactLiftWindow(stamp),'last_contact_stamp':stamp,'contact_lift_result':None,
            'k':list(info.K) if info is not None else None,'verified':False,'last_proof':None,'matched_frames':0}
        self.active_lift_verified=False
        self.event('trial_lift_started',candidate=candidate,baseline_simulation_time=baseline['simulation_time'],
                   tcp_actual=baseline['tcp_actual'],calibration_id=(self.projection_calibration or {}).get('id'),
                   primary_confirmation='bilateral_contact_aperture_and_measured_trial_lift')
        return list(baseline['tcp_actual'])

    def trial_contact_proof(self,stamp,diagnostic):
        with self.lock:
            current=self.contact_monitor.snapshot(time.monotonic())
            history=list(self.contact_history)
        if current['state']!='holding' or not current['stable_contact']:return None
        matches=[row for row in history if 0<=time.monotonic()-row['received']<1.2 and
                 row['proof']['sensor_stamps'] and max(abs(t-stamp) for t in row['proof']['sensor_stamps'])<=.05]
        if not matches:return None
        contact=copy.deepcopy(min(matches,key=lambda row:max(abs(t-stamp) for t in row['proof']['sensor_stamps']))['proof'])
        baseline=self.trial_lift['baseline']
        old=dict(zip(baseline['all_joint_names'],baseline['all_joint_positions']))
        now=dict(zip(diagnostic['all_joint_names'],diagnostic['all_joint_positions']))
        names=('finger1_joint','finger2_joint');positions=[now[name] for name in names]
        drift=max(abs(now[name]-old[name]) for name in names)
        contact.update(nonempty_aperture=all(.001<p<.038 for p in positions) and drift<=.0015,
                       measured_finger_positions=positions,finger_drift_m=drift,
                       aperture_scope='engineering bounds for current0..0.04m finger stroke')
        return contact

    def check_live_projection(self):
        if self.robot_projector is None:raise RuntimeError('effective-grasp evidence capability unavailable: '+str(self.projection_error))
        began=time.monotonic();pairs=[]
        while time.monotonic()-began<2.:
            self.checkpoint();pairs=self.paired_depth_diagnostics(-math.inf)
            if pairs:break
            time.sleep(.03)
        with self.lock:info=self.camera_info
        if not pairs or info is None:raise RuntimeError('synchronized robot/camera feedback unavailable before grasp')
        depth,stamp,diagnostic=pairs[-1]
        result=self.robot_projector.project(dict(zip(diagnostic['all_joint_names'],diagnostic['all_joint_positions'])),
            list(info.K),[depth.shape[1],depth.shape[0]],joint_stamp=diagnostic['simulation_time'],
            image_stamp=stamp,calibration=self.projection_calibration)
        if not result['coverage_complete']:raise RuntimeError('robot projection unavailable before grasp: '+result['reason'])
        self.event('grasp_evidence_preflight',calibration_id=result['calibration_id'],
                   covered_pixels=result['covered_pixels'],image_stamp=stamp,joint_stamp=diagnostic['simulation_time'])

    def observe_trial_lift(self):
        trial=self.trial_lift
        if trial is None:return
        with self.lock:feedback=list(self.joint_diagnostics)
        for diagnostic in feedback:
            stamp=diagnostic['simulation_time']
            if stamp<=trial['last_contact_stamp'] or not 0<=time.monotonic()-diagnostic['_received_monotonic']<1.2:continue
            trial['last_contact_stamp']=stamp
            delta=(np.asarray(diagnostic['tcp_actual'])-np.asarray(trial['baseline']['tcp_actual'])).tolist()
            relative=np.asarray(diagnostic['tcp_rotation_matrix'])@np.asarray(trial['baseline']['tcp_rotation_matrix']).T
            angle=math.acos(float(np.clip((np.trace(relative)-1.)/2.,-1.,1.)))
            contact=self.trial_contact_proof(stamp,diagnostic)
            result=trial['contact_lift_window'].add(stamp,delta,angle,contact)
            trial['contact_lift_result']=result
            self.event('trial_lift_contact_evidence',joint_stamp=stamp,proof=result,contact=contact)
        if (self.robot_projector is None or not self.projection_calibration or trial['k'] is None
                or len(trial['candidate'].get('surface_reference',[]))<8):return
        for depth,stamp,diagnostic in self.paired_depth_diagnostics(trial['last_stamp']):
            trial['last_stamp']=stamp
            positions=dict(zip(diagnostic['all_joint_names'],diagnostic['all_joint_positions']))
            mask_proof=self.robot_projector.project(positions,trial['k'],[depth.shape[1],depth.shape[0]],
                joint_stamp=diagnostic['simulation_time'],image_stamp=stamp,calibration=self.projection_calibration)
            if not mask_proof['coverage_complete']:
                self.event('trial_lift_visual_unavailable',reason=mask_proof['reason'],image_stamp=stamp)
                continue
            camera_rotation=np.asarray(self.projection_calibration['camera_optical_from_world'])[:3,:3]
            translation=camera_rotation@(np.asarray(diagnostic['tcp_actual'])-np.asarray(trial['baseline']['tcp_actual']))
            relative=np.asarray(diagnostic['tcp_rotation_matrix'])@np.asarray(trial['baseline']['tcp_rotation_matrix']).T
            angle=math.acos(float(np.clip((np.trace(relative)-1.)/2.,-1.,1.)))
            proof=evaluate_lift_frame(trial['candidate'],depth,trial['k'],translation,stamp=stamp,
                robot_mask=mask_proof['mask'],robot_mask_stamp=mask_proof['mask_stamp'],orientation_delta_rad=angle)
            result=trial['window'].add(proof);trial['last_proof']=proof
            contact=self.trial_contact_proof(stamp,diagnostic) if proof.get('partial_match_candidate') else None
            fused=trial['contact_surface_window'].add(proof,contact)
            if fused['verified']:result=fused
            self.event('trial_lift_evidence',proof=proof,window=result,
                       contact=contact,contact_surface_window=fused,
                       robot_projection={'calibration_id':mask_proof['calibration_id'],'covered_pixels':mask_proof['covered_pixels'],
                                         'joint_stamp':diagnostic['simulation_time'],'image_stamp':stamp})
            if proof['status']=='source_unchanged':raise LiftUnverified('original target surface remained after measured trial lift')
            if result['verified'] and not trial['verified']:
                trial['verified']=True;self.active_lift_verified=True
                self.event('grasp_verified',candidate=trial['candidate'],proof=result,
                           evidence=result['evidence'],
                           first_verified_image_stamp=stamp)

    def trial_lift_and_verify(self,candidate,q,above):
        start=self.begin_trial_lift(candidate)
        target=[start[0],start[1],min(start[2]+.04,above[2])]
        if target[2]-start[2]<.025:
            self.trial_lift=None;raise RuntimeError('insufficient verified vertical trial-lift clearance')
        try:
            self.move(target,q,'trial_lift',held=True)
            self.observe_trial_lift()
            if not self.trial_lift['verified']:
                supported=self.trial_lift['contact_lift_result']
                with self.lock:current=self.contact_monitor.snapshot(time.monotonic())
                if supported and supported['verified'] and current['stable_contact']:
                    self.trial_lift['verified']=True;self.active_lift_verified=True
                    self.event('grasp_verified',candidate=candidate,proof=supported,evidence=supported['evidence'],
                               first_verified_joint_stamp=self.trial_lift['last_contact_stamp'],
                               visual_confirmation=False,visual_status=(self.trial_lift['last_proof'] or {}).get('status','unavailable'))
            if not self.trial_lift['verified']:
                reason=(self.trial_lift['contact_lift_result'] or {}).get('reason','no fresh contact/robot samples')
                with self.lock:contact=self.contact_monitor.snapshot(time.monotonic())
                if contact['retained_support']:
                    self.event('grasp_verification_inconclusive',reason=reason,contact=contact,
                               grip_preserved=True,automatic_open_or_regrasp=False)
                    raise LiftUnobservable('contact retained but lift verification incomplete: '+reason)
                raise LiftUnverified('effective grasp not established during first lift: '+reason)
        finally:self.trial_lift=None

    def on_camera_info(self,msg):
        with self.lock:self.camera_info=msg

    def on_stop_ack(self,msg):
        try:value=json.loads(msg.data)
        except (ValueError,TypeError):return
        with self.lock:
            self.stop_ack=value;self.stop_seen=time.monotonic()
            # A cancellation racing an internal reset must stay stopped. The
            # replacement ID is explicit; old/reset IDs never prove this stop.
            if (value.get('state')=='reset' and value.get('id')==self.stop_id and
                    self.cancelled.is_set()):
                self.stop_id=None
                ident=self.request_stop('cancellation remains active after reset')
                for row in list(self.cancel_requests.values()):
                    self.acknowledge_cancel(row['request'],ident,row['response']['mapping_revision']+1)

    def request_stop(self,reason):
        with self.lock:
            if self.stop_id is None:self.stop_id=uuid.uuid4().hex
            ident=self.stop_id
            self.stop_pub.publish(String(json.dumps({'id':ident,'reason':str(reason)})))
            self.event('controlled_stop_requested',stop_id=ident,reason=str(reason))
        return ident

    def wait_stopped(self,ident,timeout=7.):
        # Stopping is still required after the task deadline or cancellation.
        until=time.monotonic()+timeout
        while time.monotonic()<until and not rospy.is_shutdown():
            with self.lock:reply=copy.deepcopy(self.stop_ack);seen=self.stop_seen
            if reply and reply.get('id')==ident and time.monotonic()-seen<1.:
                if reply.get('state')=='stopped':
                    self.event('controlled_stop_verified',stop=reply);return reply
                if reply.get('state')=='fault':
                    self.event('controlled_stop_fault',stop=reply);return None
            time.sleep(.03)
        self.event('controlled_stop_unconfirmed',stop_id=ident)
        return None

    def reset_measured_stop(self,ident):
        with self.lock:
            self.checkpoint()
            if self.stop_id!=ident:raise RuntimeError('cannot reset a replaced stop')
            sent=time.monotonic()
            self.stop_reset_pub.publish(String(json.dumps({'id':ident})))
        while time.monotonic()-sent<2. and not rospy.is_shutdown():
            with self.lock:
                self.checkpoint();reply=copy.deepcopy(self.stop_ack);seen=self.stop_seen
                if self.stop_id!=ident:raise RuntimeError('stop changed during reset')
                if reply and reply.get('id')==ident and seen>=sent:
                    if reply.get('state')=='reset':
                        self.stop_id=None
                        self.event('controlled_stop_reset',stop_id=ident);return
                    if reply.get('state') in ('reset_rejected','fault'):
                        raise RuntimeError('measured stop reset rejected: '+str(reply))
            time.sleep(.03)
        raise TimeoutError('measured stop reset not acknowledged')

    def on_cancel(self,msg):
        if msg.data:
            with self.lock:
                self.invalidate_commands();self.request_stop('external cancellation')

    def on_cancel_request(self,msg):
        value=None
        try:
            value=cancel_payload(json.loads(msg.data))
            with self.lock:
                prior=self.cancel_requests.get(value['cancel_id'])
                if prior:
                    if prior['request']!=value:raise ValueError('cancel ID reused with different context')
                    self.acknowledge_cancel(value,prior['response']['stop_id'],prior['response']['mapping_revision'])
                    return
                self.invalidate_commands(value['round_id'])
                ident=self.request_stop('external cancellation: '+value['reason'])
                self.acknowledge_cancel(value,ident)
        except (ValueError,TypeError,KeyError) as error:
            self.cancel_ack_pub.publish(String(json.dumps({'protocol':CANCEL_PROTOCOL,'state':'rejected',
                'cancel_id':value.get('cancel_id') if isinstance(value,dict) else None,
                'request_round_id':value.get('round_id') if isinstance(value,dict) else None,
                'request_id':value.get('request_id') if isinstance(value,dict) else None,
                'reason':str(error),'monotonic':time.monotonic(),'time':time.time()})))

    def on_prepare_observation(self,msg):
        req=None
        try:
            req=json.loads(msg.data);rid=req['request_id'];deadline=req['deadline_monotonic']
            if not isinstance(rid,str) or not rid:raise ValueError('invalid observation request')
            self.event('observation_received',**self.request_context(req),observation_protocol=OBSERVATION_PROTOCOL)
            if type(deadline) not in (int,float) or not math.isfinite(deadline) or not 0<deadline-time.monotonic()<=600:
                raise ValueError('invalid competition deadline')
            self.enqueue_command({**req,'kind':'prepare_observation'})
        except Exception as error:
            self.event('observation_request_rejected',**self.request_context(req),
                       observation_protocol=OBSERVATION_PROTOCOL,reason=str(error))

    def on_preview_response(self,msg):
        with self.lock:self.preview_response=json.loads(msg.data)

    def preview_grasp(self,p,above,q,side):
        ident=uuid.uuid4().hex
        self.preview_pub.publish(String(json.dumps({'id':ident,'grasp':p,'above':above,
            'quaternion':q,'transfer':self.transfer_p,'side':side,
            'place_y':float(rospy.get_param('~place_y',.15))})))
        began=time.monotonic()
        while time.monotonic()-began<10:
            self.checkpoint()
            with self.lock:reply=copy.deepcopy(self.preview_response)
            if reply and reply.get('id')==ident:
                self.event('grasp_preview',proof=reply)
                if reply.get('failure_kind') in ('internal_error','invalid_request','cancelled'):
                    raise RuntimeError('planner '+reply['failure_kind']+' at '+str(reply.get('error_stage'))+': '+str(reply.get('reason')))
                # Preview runs on the simulator's main thread. Wait for fresh
                # camera/pose feedback after it yields before issuing motion.
                until=time.monotonic()+2.
                while time.monotonic()<until:
                    self.checkpoint()
                    with self.lock:scene=self.snapshot
                    if scene and time.time()-scene['observed_at']<.5:break
                    time.sleep(.05)
                return reply
            time.sleep(.03)
        raise TimeoutError('grasp path preview did not respond')
    def on_capture(self,msg):
        with self.lock:
            self.captured=bool(msg.data);self.capture_seen=time.monotonic()
            window=getattr(self,'contact_confirmation',None)
            if window is not None:window.observe_capture(self.captured,self.capture_seen)
    def on_clock(self,msg):
        with self.lock:
            self.simulation_time=msg.clock.to_sec();self.clock_seen=time.monotonic()
            window=getattr(self,'contact_confirmation',None)
            if window is not None:window.observe_clock(self.simulation_time,self.clock_seen)
    def on_candidates(self,msg):
        with self.lock:
            self.snapshot=json.loads(msg.data)
            self.scene_history.append(self.snapshot)

    def on_depth(self,msg):
        if time.monotonic()-self.depth_seen<.05:return
        if msg.encoding not in ('32FC1','16UC1'):return
        d=self.depth_bridge.imgmsg_to_cv2(msg,'passthrough').astype('float32')
        if msg.encoding=='16UC1':d*=.001
        with self.lock:
            self.depth=d;self.depth_seen=time.monotonic();self.depth_stamp=msg.header.stamp.to_sec()
            self.depth_history.append((d,self.depth_seen,self.depth_stamp))

    def observed_occluder(self,candidate,scene=None,matched=None):
        if scene is None:
            with self.lock:scene=self.snapshot;history=list(self.depth_history)
            if scene is None:return None
            matched=matching_depth(history,scene['stamp'],time.monotonic(),max_age=2.)
        if matched is None:return None
        d,_,_=matched
        if list(d.shape)!=list(reversed(scene['image_size'])):return None
        u,v=map(lambda x:int(round(x)),candidate['pixel'])
        if not (2<=u<d.shape[1]-2 and 2<=v<d.shape[0]-2):return None
        return occlusion_depth(candidate['depth'],d[v-2:v+3,u-2:u+3].reshape(-1).tolist())

    def reference_depth_consistent(self,candidate,scene=None,matched=None):
        if scene is None:
            with self.lock:scene=self.snapshot;history=list(self.depth_history)
            if scene is None:return None
            matched=matching_depth(history,scene['stamp'],time.monotonic(),max_age=2.)
        if matched is None:return None
        d,_,_=matched
        if list(d.shape)!=list(reversed(scene['image_size'])):return None
        points=candidate.get('depth_reference',[])
        if any(not (0<=u<d.shape[1] and 0<=v<d.shape[0]) for u,v,_ in points):return None
        return reference_consistency([z for _,_,z in points],[float(d[v,u]) for u,v,_ in points])

    def verify_approach_observation(self,selected,approach_scene,p,selected_at,timeout=2.5):
        """Wait at the raised pose for one coherent, newly captured RGB-D scene."""
        began=time.monotonic();not_before=time.time();until=min(began+timeout,self.deadline-8.)
        references=[c for c in approach_scene['candidates'] if c.get('stable_id')!=selected.get('stable_id')]
        last_frame=None;last_reason='no_fresh_post_arrival_scene'
        while time.monotonic()<until:
            self.checkpoint();scene=self.fresh_scene()
            if scene['frame_id']==last_frame or scene['observed_at']<not_before:
                time.sleep(.05);continue
            last_frame=scene['frame_id']
            try:
                self.rebind_target(selected,self.object_candidates(scene,selected))
                return
            except ValueError:pass
            actual,orientation=self.read_pose()
            short_age=time.monotonic()-selected_at<30.
            tool_above=(math.dist(actual[:2],p[:2])<.015 and actual[2]-p[2]>.10
                        and empty_finger_lowest_z(actual,orientation)>2.485238+.010)
            if not (short_age and tool_above and selected['confidence']>=.6):
                raise ValueError('raised-pose identity retention preconditions not met')
            with self.lock:history=list(self.depth_history)
            matched=matching_depth(history,scene['stamp'],time.monotonic(),max_age=2.)
            unchanged=hidden=depth_refs=0
            for reference in references:
                try:rebind_candidate(reference,scene['candidates']);unchanged+=1
                except ValueError:
                    if self.observed_occluder(reference,scene,matched) is not None:hidden+=1
                    elif self.reference_depth_consistent(reference,scene,matched) is not None:depth_refs+=1
            occluder=self.observed_occluder(selected,scene,matched)
            approved=(occluder is not None and unchanged+hidden+depth_refs==len(references))
            last_reason=('accepted' if approved else 'depth_pair_unavailable' if matched is None
                         else 'unexplained_target_or_context')
            self.event('occlusion_check',short_age=short_age,tool_above=tool_above,
                       confidence=selected['confidence'],occluder_depth=occluder,
                       unchanged_references=unchanged,hidden_references=hidden,depth_references=depth_refs,
                       total_references=len(references),frame_id=scene['frame_id'],image_stamp=scene['stamp'],
                       depth_stamp=None if matched is None else matched[2],reason=last_reason,
                       scene_age_seconds=time.time()-scene['observed_at'],
                       cached_depth_stamps=[history[0][2],history[-1][2]] if history else [],
                       wait_seconds=time.monotonic()-began)
            if approved:
                self.event('controlled_self_occlusion',candidate=selected,unchanged_references=unchanged,
                           hidden_references=hidden,depth_references=depth_refs,occluder_depth=occluder,
                           frame_id=scene['frame_id'],image_stamp=scene['stamp'],depth_stamp=matched[2],
                           limitation='Static scene assumption for this final descent; last verified target is retained.')
                return
            time.sleep(.05)
        raise ValueError('target not visible and self-occlusion conditions not met after bounded fresh-frame wait: '+last_reason)

    def on_planner(self,msg):
        with self.lock:self.planner_status=json.loads(msg.data)

    def on_plan(self,msg):
        rid=None;plan=None
        try:
            plan=json.loads(msg.data)
            rid=plan['request_id']
            if not isinstance(rid,str) or not rid:raise ValueError('invalid request ID')
            if self.busy or not self.queue.empty() or self.needs_recovery or self.stop_id:raise ValueError('busy or recovery required')
            if rid in self.seen:raise ValueError('duplicate plan')
            if plan['side'] not in ('left','right'):raise ValueError('unknown conveyor side')
            if not isinstance(plan['objects'],list) or not 0<=len(plan['objects'])<=10:raise ValueError('invalid objects')
            if not 0<=time.time()-plan['created_at']<3:raise ValueError('stale plan')
            if plan['objects'] and self.robot_projector is None:
                self.load_projection_capability()
                if self.robot_projector is None:
                    raise ValueError('effective-grasp evidence capability unavailable: '+str(self.projection_error))
            if not plan['objects']:
                semantic=plan.get('semantic',{});context=plan.get('task_context',{})
                with self.lock:scene=copy.deepcopy(self.snapshot)
                clear=(scene and 0<=time.time()-scene.get('observed_at',0)<2 and
                       scene.get('coverage_complete') is True and scene.get('scene_complete') is True and
                       not scene.get('unknown_regions') and not scene.get('candidates'))
                if not (plan['class']=='Remaining' and semantic.get('intent')=='remaining' and
                        semantic.get('count')==0 and semantic.get('ids')==[] and clear and
                        context.get('dependencies_satisfied') is True and context.get('remaining_complete') is True and
                        not context.get('reserved_ids') and not context.get('reserved_stable_ids')):
                    raise ValueError('empty plan requires real Nine remaining intent and complete fresh empty scene')
            for c in plan['objects']:
                if plan['class']!='Remaining' and c['class']!=plan['class']:raise ValueError('plan class mismatch')
                if c.get('grasp_ready') is False:raise ValueError('target geometry not approved for current grasp')
                if not isinstance(c.get('stable_id'),str) or not c['stable_id'] or c.get('identity_status')!='confirmed':
                    raise ValueError('target requires a confirmed persistent identity')
                validate_workspace(c['position'])
            if 'deadline_monotonic' in plan:
                deadline=plan['deadline_monotonic']
                if type(deadline) not in (int,float) or not math.isfinite(deadline) or not 0<deadline-time.monotonic()<=600:
                    raise ValueError('competition deadline expired or invalid')
            self.enqueue_command(plan)
        except Exception as e:self.event('plan_rejected',**self.request_context(plan),reason=str(e))

    def checkpoint(self):
        if rospy.is_shutdown() or self.cancelled.is_set():raise RuntimeError('cancelled: no further targets will be issued')
        if time.monotonic()>self.deadline:raise TimeoutError('total task budget exhausted')
        self.check_pending_deliveries()

    def check_pending_deliveries(self):
        """Read-only follow-up; never blocks another grasp or publishes motion."""
        pending=getattr(self,'pending_deliveries',{})
        if not pending or time.monotonic()-getattr(self,'last_pending_delivery_check',0.)<.25:return
        self.last_pending_delivery_check=time.monotonic()
        with self.lock:history=list(self.scene_history)
        for ident,context in list(pending.items()):
            candidate=context['candidate']
            metadata={'request_id':context['request_id'],'round_id':context['round_id'],'task_id':context['task_id'],
                      'stable_id':candidate['stable_id'],'category':candidate['class'],'side':context['side'],'release_id':ident}
            try:
                proof=verify_transport(history,candidate['class'],context['side'],
                    context['released_at'],context['drop_y'],context['request_id'],context['drop_x'],
                    stable_id=candidate['stable_id'],release_id=ident)
                if proof is None:
                    proof=verify_settled_placement(history,candidate['class'],context['side'],context['released_at'],
                        context['drop_y'],context['request_id'],context['home_arrived_at'],
                        stable_id=candidate['stable_id'],release_id=ident)
            except Exception as error:
                proof=None
                if context.get('verification_error')!=str(error):
                    self.event('placement_verification_error',**metadata,reason=str(error),continue_next_task=True)
                    context['verification_error']=str(error)
            timely=[sample['observed_at'] for sample in (proof or {}).get('samples',[])
                    if type(sample.get('observed_at')) in (int,float) and math.isfinite(sample['observed_at'])
                    and context['released_at']<=sample['observed_at']<=context['deadline_wall']]
            if proof and timely:
                self.event('placement_verified',**metadata,candidate=candidate,transport=proof,
                    released_at=context['released_at'],released_monotonic=context['released_monotonic'],
                    delivered_at=min(timely),verified_at=time.time(),delivery_time_source='first_qualified_belt_observation_wall',
                    background_verification=True,empty_release_proof=context['empty_release_proof'],
                    belt_observer=self.belt_observer_state())
                context['verified']=True;pending.pop(ident,None)
            elif time.time()>context['released_at']+35.:
                self.event('placement_verification_expired',**metadata,
                           reason='no timely bound belt proof; later task execution is unaffected',
                           belt_observer=self.belt_observer_state())
                pending.pop(ident,None)

    def belt_observer_state(self):
        """Latest belt-observer diagnostics, recorded beside placement outcomes.

        The observer runs inside the perception node and publishes its state with
        the scene message. Without it an unverified placement cannot be told
        apart from one the observer never looked for.
        """
        with self.lock:
            state=(self.snapshot or {}).get('transport_status')
        return copy.deepcopy(state)

    def fresh_scene(self):
        self.checkpoint()
        with self.lock:s=copy.deepcopy(self.snapshot)
        if s is None or not 0<=time.time()-s['observed_at']<2:raise RuntimeError('visual feedback stale')
        return s

    def read_pose(self,with_stamp=False):
        with self.lock:p=self.pose;t=self.pose_seen
        if p is None or time.monotonic()-t>1.5:raise RuntimeError('end-effector feedback missing or stale')
        xyz=[p.pose.position.x,p.pose.position.y,p.pose.position.z]
        q=[p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w]
        return (xyz,q,t) if with_stamp else (xyz,q)

    def holding(self):
        with self.lock:captured=self.captured;t=self.capture_seen
        if time.monotonic()-t>1.5:raise RuntimeError('gripper feedback missing or stale')
        return captured

    def require_retained_contact(self,phase):
        with self.lock:
            contact=self.contact_monitor.snapshot(time.monotonic());armed=self.contact_monitor.armed
        if contact['state']=='feedback_fault' or not armed:
            raise RuntimeError('native effort feedback unavailable during held motion: '+str(contact['fault']))
        if contact['contact_lost']:
            self.event('holding_feedback_lost',phase=phase,proof=contact)
            raise GraspLost('sustained contact support lost during '+phase)
        return contact

    def move(self,p,q,phase,held=False,timeout=45,preview_id=None):
        self.checkpoint();validate_workspace(p)
        if self.stop_id is not None:raise RuntimeError('motion blocked until measured stop is explicitly reset')
        if len(q)!=4 or not all(math.isfinite(v) for v in q):raise ValueError('invalid quaternion')
        norm=math.sqrt(sum(v*v for v in q))
        if norm<1e-6:raise ValueError('zero quaternion')
        q=[v/norm for v in q]
        actual,actual_q=self.read_pose()
        if held:self.require_retained_contact(phase)
        stop_reserve=float(rospy.get_param('~stop_reserve_seconds',8.))
        if not 5.<=stop_reserve<=30.:raise ValueError('stop reserve must remain within 5..30 seconds')
        estimate=6.+max(.9,math.dist(actual,p)/.12,quaternion_error(actual_q,q)/.4)+self.arrival_seconds+1.
        if self.deadline-time.monotonic()<estimate+stop_reserve:
            raise TimeoutError('insufficient budget for motion estimate and measured stop reserve: '+phase)
        target=PoseStamped();target.header.frame_id='base_link';target.header.stamp=rospy.Time.now()
        target.pose.position.x,target.pose.position.y,target.pose.position.z=p
        target.pose.orientation.x,target.pose.orientation.y,target.pose.orientation.z,target.pose.orientation.w=q
        self.event(phase,target=p,quaternion=q)
        # Exactly one command: avoid the stock simulator queueing repeated goals.
        connection_started=time.monotonic()
        motion_pub=self.bound_preview_pub if preview_id is not None else self.ee_pub
        while motion_pub.get_num_connections()==0:
            self.checkpoint()
            if time.monotonic()-connection_started>3:
                raise RuntimeError('no simulator subscriber for motion command')
            time.sleep(.05)
        sent_at=time.time()
        if self.deadline-time.monotonic()<estimate+stop_reserve:
            raise TimeoutError('motion budget consumed while waiting for command subscriber: '+phase)
        with self.lock:
            self.checkpoint()
            self.last_motion_sent_at=sent_at
            if preview_id is None:self.ee_pub.publish(target)
            else:motion_pub.publish(String(json.dumps({'id':preview_id,'position':list(p),'quaternion_wire':list(q)})))
        began=time.monotonic();lost=None;gap_reported=False;finished_at=None;duration_checked=False
        arrival=ArrivalWindow(self.arrival_seconds)
        while time.monotonic()-began<timeout:
            self.checkpoint()
            if time.monotonic()>=self.deadline-stop_reserve:
                raise TimeoutError('motion interrupted at measured-stop reserve: '+phase)
            with self.lock:
                pose_age=time.monotonic()-self.pose_seen
                grip_age=time.monotonic()-self.capture_seen
                planner=copy.deepcopy(self.planner_status)
            plan_finished=False
            if (planner and planner.get('time',0)>=sent_at and
                len(planner.get('position',[]))==3 and math.dist(planner['position'],p)<1e-5 and
                quaternion_error(planner.get('quaternion_wire',[0,0,0,0]),q)<1e-4):
                if planner['status']=='rejected':
                    raise PlanningRejected('planner rejected target: '+phase)
                plan_finished=planner['status']=='tracking_finished'
                duration=planner.get('timing',{}).get('duration')
                if not duration_checked and planner['status']=='planned' and isinstance(duration,(int,float)) and math.isfinite(duration):
                    duration_checked=True
                    if duration+self.arrival_seconds+stop_reserve>self.deadline-time.monotonic():
                        raise TimeoutError('accepted trajectory cannot finish before stop reserve: '+phase)
            # Stock TRRT runs synchronously in the simulator's publishing loop.
            # Wait for real feedback during a bounded initial planning gap;
            # never use the stale sample to claim arrival or retained grasp.
            if pose_age>1.5 or (held and grip_age>1.5):
                if time.monotonic()-began<15:
                    if not gap_reported:
                        self.event('planning_feedback_gap',phase=phase,pose_age=pose_age)
                        gap_reported=True
                    arrival.reset();self.contact_loss.reset();time.sleep(.05);continue
                raise RuntimeError('feedback did not resume within planning allowance: '+phase)
            actual,aq,sample_at=self.read_pose(with_stamp=True)
            if held:
                self.require_retained_contact(phase)
            if phase=='trial_lift' and self.trial_lift is not None:
                self.observe_trial_lift()
            dist=math.dist(actual,p);angle=quaternion_error(aq,q)
            if plan_finished:
                if finished_at is None:finished_at=time.monotonic()
                if time.monotonic()-finished_at>3 and (dist>=.008 or angle>=.07):
                    raise RuntimeError('trajectory ended without reaching target: %s; position_error=%.4f; orientation_error=%.4f'%(phase,dist,angle))
            eligible=dist<.008 and angle<.07 and (not self.require_planner or plan_finished)
            if arrival.observe(time.monotonic(),sample_at,actual,aq,eligible):
                self.event('arrived',phase=phase,position_error=dist,orientation_error=angle,
                           stable_seconds=time.monotonic()-arrival.since,fresh_samples=arrival.samples)
                return
            time.sleep(.05)
        raise TimeoutError('motion timeout: '+phase)

    def open_gripper(self):
        with self.lock:
            self.checkpoint();self.contact_loss.reset()
            self.contact_monitor.reset()
            self.contact_history.clear()
            self.grip_value=0.;self.grip_pub.publish(Float32(0.))
        began=time.monotonic();stable=None
        while time.monotonic()-began<4:
            self.checkpoint()
            if not self.holding():
                if stable is None:stable=time.monotonic()
                if time.monotonic()-stable>.3:return
            else:stable=None
            time.sleep(.05)
        raise TimeoutError('release was not confirmed')

    def move_held_linear(self,p,q,phase,max_step=.12):
        """Short observed segments retain the grasp attitude during transport.

        The supplied joint-space planner has no collision validator. Limiting
        each Cartesian displacement reduces long, low-sweeping joint motions;
        this is not a proof of whole-arm collision freedom.
        """
        start,_=self.read_pose()
        segments=max(1,int(math.ceil(math.dist(start,p)/max_step)))
        for i in range(1,segments+1):
            target=[a+(b-a)*i/segments for a,b in zip(start,p)]
            self.move(target,q,'%s_%02d_of_%02d'%(phase,i,segments),held=True)

    def close_gripper(self):
        began=time.monotonic();last_increment=began
        preload_base=None;preload_goal=None;preload_finished=False;preload_limited=False
        preload_limit_reason=None;preload_trimmed=False;preload_last_step_sim=None
        self.last_closure_failure=None
        window=ContactConfirmationWindow(threshold=float(rospy.get_param('~contact_threshold',.2)))
        max_lead=float(rospy.get_param('~closure_max_position_lead',.002))
        if not math.isfinite(max_lead) or not 0<max_lead<=.002:raise ValueError('closure loading cap must be within (0,.002]')
        preload_distance=float(rospy.get_param('~closure_preload_distance',.002))
        if not math.isfinite(preload_distance) or not 0<preload_distance<=.002:
            raise ValueError('gradual preload must be within (0,.002]')
        effort_stop=float(rospy.get_param('~preload_effort_stop',30.))
        if not math.isfinite(effort_stop) or not .2<effort_stop<=30.:
            raise ValueError('preload effort stop must be within (.2,30]')
        with self.lock:
            self.contact_confirmation=window
            window.observe_clock(self.simulation_time,self.clock_seen)
            self.record_closure_joints(window,self.joint_feedback,self.joints_seen)
            window.observe_capture(self.captured,self.capture_seen)
            window.observe_efforts(self.gripper_efforts,self.efforts_seen)
            last_step_sim=window.sim;last_step_joint=window.joint_stamp
        try:
            while time.monotonic()-began<12:
                self.checkpoint()
                with self.lock:
                    self.checkpoint()
                    now=time.monotonic();proof=window.snapshot(now);blocked=False
                    if proof['confirmed'] and preload_goal is None:
                        preload_base=self.grip_value;preload_goal=min(.04,preload_base+preload_distance)
                        window.restart_confirmation()
                        self.event('grasp_preload_started',command=preload_base,target=preload_goal,
                                   requested_distance_m=preload_distance,step_m=.00025,
                                   contact_efforts=proof['contact_efforts'],finger_positions=proof['finger_positions'],
                                   effort_stop=effort_stop,effort_stop_calibrated=False)
                    can_step=(now-last_increment>=.04 and proof['simulation_time']>last_step_sim
                              and proof['joint_stamp']>last_step_joint)
                    loading=(preload_goal is not None and not preload_finished and proof['contact_candidate'])
                    searching=(preload_goal is None and not proof['contact_candidate'])
                    # Preload must observe the response to the previous load,
                    # not consume a newer joint message with older effort data.
                    if loading and preload_last_step_sim is not None:
                        can_step=can_step and proof['feedback_simulation_time']-preload_last_step_sim>=.08
                    finish_reason=None
                    if loading and can_step and max(proof['contact_efforts'])>=effort_stop:
                        finish_reason='effort_stop';can_step=False
                    if can_step and (loading or searching):
                        target=next_closure_target(self.grip_value,proof['finger_positions'],max_lead=max_lead)
                        if target is None:
                            if loading:
                                # Contact can push a finger back slightly at
                                # the loading cap. Trim at most one normal step;
                                # never call positive, loaded contact an empty
                                # grasp merely because it cannot take more load.
                                target=min(proof['finger_positions'])+max_lead-.0001
                                if not preload_trimmed and 0<self.grip_value-target<=.00025:
                                    preload_trimmed=True;self.grip_value=target
                                    self.grip_pub.publish(Float32(target));last_increment=now
                                    last_step_sim=proof['simulation_time'];last_step_joint=proof['joint_stamp']
                                    finish_reason='measured_position_cap_trim'
                                else:blocked=True
                            else:blocked=True
                        else:
                            if loading:target=min(target,preload_goal)
                            moved=target>self.grip_value+1e-6
                            if moved:
                                self.grip_value=target;self.grip_pub.publish(Float32(target));last_increment=now
                                last_step_sim=proof['simulation_time'];last_step_joint=proof['joint_stamp']
                            if loading:
                                if moved:
                                    window.restart_confirmation();preload_last_step_sim=proof['simulation_time']
                                if not moved or target>=preload_goal-1e-6:
                                    finish_reason='measured_position_cap' if not moved else 'requested_distance'
                    if finish_reason is not None:
                        preload_finished=True;preload_limited=self.grip_value<preload_goal-1e-6
                        preload_limit_reason=finish_reason;window.restart_confirmation()
                        self.event('grasp_preload',command=self.grip_value,requested_target=preload_goal,
                            applied_distance_m=self.grip_value-preload_base,limit_reason=finish_reason,
                            limited_by_measured_position_lead=finish_reason.startswith('measured_position_cap'),
                            contact_efforts=proof['contact_efforts'],effort_stop=effort_stop,
                            finger_positions=proof['finger_positions'],position_lead_limit=max_lead)
                    proof=window.snapshot(now)
                    if preload_finished and self.grip_value-min(proof['finger_positions'])>max_lead+1e-6:
                        blocked=True
                    if proof['confirmed'] and preload_finished and not blocked:
                        joints=self.joint_feedback
                        self.contact_monitor.arm(now)
                        self.contact_loss.reset();self.last_contact_sim_time=proof['simulation_time']
                        confirmed=dict(command=self.grip_value,contact_efforts=proof['contact_efforts'],
                            stable_sim_seconds=proof['stable_sim_seconds'],fresh_contact_samples=proof['fresh_contact_samples'],
                            joint_names=list(joints.name),measured_joint_positions=list(joints.position),
                            native_contact=self.contact_monitor.snapshot(now),
                            position_lead_limit=max_lead,position_lead_limit_calibrated=False,
                            preload_distance_m=self.grip_value-preload_base,preload_limited=preload_limited,
                            preload_limit_reason=preload_limit_reason,preload_effort_stop=effort_stop,
                            preload_effort_stop_calibrated=False,
                            limitation='Contact plus preload only; actual trial-lift confirmation is still required.')
                    else:confirmed=None
                if confirmed is not None:
                    self.event('grasp_contact_confirmed',**confirmed)
                    return True
                if blocked:
                    self.last_closure_failure='measured_fingers_fell_behind_loading_cap'
                    self.event('closure_rejected' if preload_base is not None else 'empty_grasp',
                               command=self.grip_value,reason=self.last_closure_failure,
                               contact_candidate=proof['contact_candidate'],contact_efforts=proof['contact_efforts'],
                               finger_positions=proof['finger_positions'],preload_started=preload_base is not None)
                    return False
                time.sleep(.04)
            self.last_closure_failure='stable_loaded_contact_confirmation_timeout'
            self.event('empty_grasp',command=self.grip_value,reason=self.last_closure_failure)
            return False
        finally:
            with self.lock:self.contact_confirmation=None

    @staticmethod
    def belt_candidates(scene,category,side):
        sign=1 if side=='left' else -1
        return [c for c in scene['candidates'] if c['class']==category
                and sign*c['position'][0]>.43 and -.5<c['position'][1]<.9]

    def execute_object(self,selected,side):
        context=self.fresh_scene()
        other_references=[copy.deepcopy(c) for c in context['candidates'] if c.get('stable_id')!=selected.get('stable_id')]
        recoveries=0
        while True:
            try:
                self.active_object=copy.deepcopy(selected);self.active_lift_verified=False
                verified=self.execute_object_once(selected,side)
                self.finish_grasp_attempt('released_pending_verification' if verified is False else 'success')
                if recoveries:self.event('drop_recovery_completed',recoveries=recoveries,category=selected['class'])
                return verified
            except (GraspLost,EmptyGrasp) as error:
                outcome='empty_grasp' if isinstance(error,EmptyGrasp) else ('holding_feedback_lost' if self.active_lift_verified else 'no_effective_lift')
                self.finish_grasp_attempt(outcome,reason=str(error))
                recoveries+=1
                self.event('drop_recovery_started',recovery= recoveries,category=selected['class'],reason=str(error))
                selected=self.recover_dropped_object(selected,other_references,recoveries)
            except Exception as error:
                self.finish_grasp_attempt('system_aborted',reason=str(error));raise

    def finish_grasp_attempt(self,outcome,**details):
        if self.active_attempt_id is None:return
        record=self.grasp_history.record_outcome(self.active_attempt_id,outcome,now=time.monotonic(),details=details)
        self.active_attempt_id=None;self.event('grasp_attempt_outcome',record=record)

    def select_grasp_pose(self,selected,side):
        if self.robot_projector is None:raise RuntimeError('effective-grasp verification is not configured')
        camera_to_world=np.asarray(self.projection_calibration['camera_optical_from_world'])[:3,:3].T.tolist()
        first=self.rebind_with_wait(selected)
        generated=generate_grasp_candidates(first,camera_to_world)
        self.event('grasp_candidate_set',candidate=first,generated=generated)
        names=[item['candidate_id'] for item in generated['candidates']]
        # Preserve the normal vertical search and its successful fast path.
        # Only its exhausted, nonrepeated candidates reach the bounded tilt
        # family; all trials share the same object/global budgets and guards.
        for tilt in (0.,10.,-10.):
            if tilt:self.event('grasp_tilt_fallback_started',tilt_deg=tilt,
                reason='vertical candidates exhausted; measured geometry and full route still required')
            for layer in (0,1):
                for name in names:
                    for variant in (0,180):
                        self.checkpoint()
                        candidate=self.rebind_with_wait(selected,timeout=6.)
                        current=generate_grasp_candidates(candidate,camera_to_world)
                        choices=[item for item in current['candidates'] if item['candidate_id']==name]
                        if not choices:continue
                        item=choices[0];pose_candidate=item['pose_candidate']
                        decision=self.grasp_history.check(candidate,pose_candidate,variant,layer,
                            now=time.monotonic(),deadline=self.deadline,minimum_budget=20.,stop_reserve=5.,tilt_deg=tilt)
                        if not decision['allowed']:
                            self.event('grasp_candidate_skipped',candidate_id=name,yaw_variant=variant,tilt_deg=tilt,depth_layer=layer,decision=decision)
                            if decision['reason'] in ('object_attempt_limit_reached','object_recovery_budget_exhausted','global_budget_insufficient'):
                                raise RuntimeError('bounded grasp search exhausted: '+decision['reason'])
                            continue
                        self.active_attempt_id=self.grasp_history.begin_attempt(candidate,pose_candidate,variant,layer,
                            now=time.monotonic(),deadline=self.deadline,minimum_budget=20.,stop_reserve=5.,tilt_deg=tilt)
                        p,q=calibrated_grasp(pose_candidate,layer,variant,tilt_deg=tilt)
                        self.event('attempt',attempt_id=self.active_attempt_id,candidate=candidate,
                                   grasp_candidate=item,yaw_variant=variant,tilt_deg=tilt,depth_layer=layer)
                        for height in grasp_approach_heights(p,q if tilt else None):
                            latest=self.fresh_scene()
                            check=self.rebind_target(candidate,self.object_candidates(latest,candidate))
                            if math.dist(check['world_position'],candidate['world_position'])>.003:
                                raise ValueError('target moved during fixed grasp-candidate preview')
                            above=[p[0],p[1],height]
                            preview=self.preview_grasp(p,above,q,side)
                            if not preview.get('valid'):
                                self.event('grasp_orientation_rejected',grasp_candidate_id=name,yaw_variant=variant,tilt_deg=tilt,above_height=height,reason=preview.get('reason'))
                                continue
                            try:
                                self.move(above,q,'approach',preview_id=preview['id'])
                                self.event('grasp_orientation_selected',grasp_candidate_id=name,yaw_variant=variant,tilt_deg=tilt,
                                    depth_layer=layer,candidate=candidate,pose_candidate=pose_candidate,
                                    quaternion=q,above_height=height,preview=preview)
                                return candidate,p,q,above,preview['place'],latest,time.monotonic()
                            except PlanningRejected as error:
                                self.event('grasp_orientation_rejected',grasp_candidate_id=name,yaw_variant=variant,tilt_deg=tilt,reason=str(error))
                        self.finish_grasp_attempt('planning_rejected',grasp_candidate_id=name,yaw_variant=variant,tilt_deg=tilt,depth_layer=layer)
        raise PlanningRejected('all bounded nonrepeated grasp points, wrists, tilts, depths and heights rejected')

    def recover_dropped_object(self,selected,other_references,recovery):
        if time.monotonic()>self.deadline-60:raise TimeoutError('insufficient task time for drop recovery')
        stop_id=self.request_stop('grasp recovery: '+str(recovery))
        if self.wait_stopped(stop_id) is None:
            raise RuntimeError('recovery forbidden: active trajectory stop not confirmed')
        self.reset_measured_stop(stop_id)
        self.open_gripper()
        actual,orientation=self.read_pose()
        clearance=empty_finger_lowest_z(actual,orientation)
        if clearance<2.485238+.010:
            escape=[actual[0],actual[1],actual[2]+2.485238+.025-clearance]
            self.move(escape,orientation,'drop_recovery_vertical_escape')
        self.move(self.home_p,self.home_q,'drop_recovery_observation')
        previous=None;stable_since=None;last_frame=None;stable_samples=[];identity_requests=0
        recovery_started=self.grasp_history.recovery_started.get(selected.get('stable_id'),time.monotonic())
        recovery_until=min(self.deadline-60,recovery_started+self.grasp_history.limits['max_recovery_seconds'])
        while time.monotonic()<recovery_until:
            self.checkpoint();scene=self.fresh_scene()
            if scene['frame_id']==last_frame:time.sleep(.06);continue
            last_frame=scene['frame_id']
            try:
                found=reacquire_dropped(selected,other_references,pickable_recovery_objects(scene))
                if found.get('grasp_ready') is False:raise ValueError('reacquired geometry remains unready')
                stamp=float(scene['stamp'])
                if stable_samples and stamp<=stable_samples[-1]:time.sleep(.06);continue
                if previous is not None:
                    rebind_candidate(previous,[found],max_pixels=1.5,max_depth=.003)
                    if math.dist(previous['world_position'][:2],found['world_position'][:2])>.0015:
                        raise ValueError('dropped object is still moving')
                    if abs((found['angle_deg']-previous['angle_deg']+90.)%180.-90.)>2.:
                        raise ValueError('dropped object is still rotating')
                    stable_samples.append(stamp)
                    if stamp-stable_since>=.6 and len(stable_samples)>=3:
                        if found.get('stable_id')==selected.get('stable_id') and found.get('identity_status')=='confirmed':
                            self.event('drop_recovery_reacquired',recovery=recovery,candidate=found,identity_method='existing_confirmed_tracker_binding')
                            return found
                        identity_requests+=1
                        if identity_requests>4:raise RuntimeError('bounded recovery identity acknowledgement attempts exhausted')
                        recovered=self.acknowledge_reacquired_identity(selected,found,scene,other_references,stable_samples)
                        if recovered is not None:
                            self.event('drop_recovery_reacquired',recovery=recovery,candidate=recovered,identity_method='current_camera_tracker_acknowledgement')
                            return recovered
                        previous=None;stable_since=None;stable_samples=[]
                else:
                    stable_since=stamp;previous=copy.deepcopy(found);stable_samples=[stamp]
            except ValueError:
                previous=None;stable_since=None;stable_samples=[]
            time.sleep(.06)
        raise TimeoutError('dropped object was not stably and uniquely reidentified within the remaining budget')

    def acknowledge_reacquired_identity(self,selected,found,scene,other_references,stable_samples):
        stable=selected.get('stable_id')
        if not isinstance(stable,str) or not stable:raise ValueError('original persistent identity is missing')
        current=[c for c in scene.get('candidates',[]) if c.get('id')==found.get('id')]
        if len(current)!=1 or math.dist(current[0]['pixel'],found['pixel'])>1.:
            raise ValueError('recovery target is outside the current tracked candidate coverage')
        excluded=[]
        for reference in other_references:
            if reference.get('class')!=selected['class']:continue
            ident=reference.get('stable_id')
            matches=[c for c in scene['candidates'] if c.get('stable_id')==ident and c.get('identity_status')=='confirmed']
            if not ident or len(matches)!=1:raise ValueError('another same-class identity is not visibly excluded')
            excluded.append(ident)
        ident=uuid.uuid4().hex;sent=time.monotonic()
        self.event('identity_reacquired',reacquire_id=ident,stable_id=stable,candidate_id=found['id'],
                   frame_id=scene['frame_id'],**{'class':selected['class']},pixel=found['pixel'],
                   world_position=found['world_position'],proof={'uniquely_reidentified':True,
                   'stable_seconds':stable_samples[-1]-stable_samples[0],'sample_count':len(stable_samples),
                   'first_stamp':stable_samples[0],'last_stamp':stable_samples[-1],
                   'last_frame_id':scene['frame_id'],'excluded_same_class_ids':excluded})
        ack=None
        while time.monotonic()-sent<2.:
            self.checkpoint()
            with self.lock:ack=copy.deepcopy(self.identity_acks.get(ident))
            if ack and ack.get('_received_monotonic',0)>=sent:break
            time.sleep(.03)
        if not ack or ack.get('status')!='accepted':
            self.event('identity_reacquisition_not_accepted',reacquire_id=ident,ack=ack)
            return None
        until=time.monotonic()+3.
        while time.monotonic()<until:
            self.checkpoint();fresh=self.fresh_scene()
            if fresh['frame_id']>ack['effective_after_frame_id']:
                matches=[c for c in fresh['candidates'] if c.get('stable_id')==stable and
                         c.get('identity_status')=='confirmed' and c.get('class')==selected['class']]
                if len(matches)==1:
                    rebound=rebind_candidate(found,matches,max_pixels=2.,max_depth=.003)
                    rebound=copy.deepcopy(rebound);rebound['_drop_recovery']=True
                    return rebound
            time.sleep(.05)
        self.event('identity_reacquisition_followup_missing',reacquire_id=ident,ack=ack)
        return None

    @staticmethod
    def object_candidates(scene,selected):
        return pickable_recovery_objects(scene) if selected.get('_drop_recovery') else scene['candidates']

    @staticmethod
    def rebind_target(selected,current):
        stable_id=selected.get('stable_id')
        if not isinstance(stable_id,str) or not stable_id:
            raise ValueError('target has no persistent identity')
        matches=[c for c in current if c.get('stable_id')==stable_id and
                 c.get('identity_status')=='confirmed' and c.get('class')==selected.get('class')]
        if len(matches)!=1:raise ValueError('persistent target is not uniquely confirmed; reobserve')
        # The tracker must preserve a continuous, unambiguous association.
        # Still reject stale geometric binding; far-drop identity is explicitly
        # restored and re-anchored by recover_dropped_object before this call.
        return rebind_candidate(selected,matches,max_pixels=45. if selected.get('_drop_recovery') else 14.)

    def rebind_with_wait(self,selected,timeout=25.):
        """Reobserve-bounded rebind: post-delivery identity lag must not kill a task.

        rebind_target error text already prescribes reobserve -- this honors it:
        poll fresh scenes inside a bounded window; raise only when it expires.
        checkpoint enforces stop/deadline guards. Permanent defects (no
        persistent identity) fail immediately at entry.
        """
        if not isinstance(selected.get('stable_id'),str) or not selected.get('stable_id'):
            raise ValueError('target has no persistent identity')
        began=time.monotonic();last=None
        while time.monotonic()-began<timeout:
            self.checkpoint()
            scene=self.fresh_scene()
            try:
                return self.rebind_target(selected,self.object_candidates(scene,selected))
            except ValueError as error:
                last=error;time.sleep(.5)
        raise last

    def execute_object_once(self,selected,side):
        self.active_release_context=None
        self.check_live_projection()
        before=self.fresh_scene()
        initial_objects=self.object_candidates(before,selected)
        before_source_count=sum(c['class']==selected['class'] for c in initial_objects)
        unique_at_start=sum(c['class']==selected['class'] for c in initial_objects)==1
        grabbed=False
        for attempt in range(1):
            self.open_gripper()
            selected,p,q,above,preferred_place,approach_scene,selected_at=self.select_grasp_pose(selected,side)
            self.active_object=copy.deepcopy(selected)
            # Recheck visible targets. A top camera loses sight of an object
            # directly beneath the raised tool; retain its short-lived identity
            # only with a verified raised pose and unchanged visible context.
            actual,actual_q=self.read_pose()
            finger_clearance=empty_finger_lowest_z(actual,actual_q)-2.485238
            if finger_clearance<.010:
                raise ValueError('actual open fingers do not clear basket rim by10mm')
            self.event('approach_clearance_verified',finger_rim_clearance_m=finger_clearance,
                       position=actual,quaternion=actual_q)
            self.verify_approach_observation(selected,approach_scene,p,selected_at)
            self.move(p,q,'descend')
            if self.close_gripper():grabbed=True;break
        if not grabbed:raise EmptyGrasp(getattr(self,'last_closure_failure',None) or
                                       'bilateral stable contact not established for this bounded grasp candidate')
        self.trial_lift_and_verify(selected,q,above)
        self.move_held_linear(above,q,'lift',max_step=.04)
        self.move_held_linear(self.transfer_p,q,'transfer_clearance')
        drop_y=float(rospy.get_param('~place_y',.15))
        reached=False
        # Keep the same grasp attitude until release. The stock attachment code
        # mixes quaternion conventions, so wrist rotations can disturb contact.
        # Prefer upstream belt center for edge clearance and visible transit.
        place_options=[(.68,drop_y,q),(.65,.2,q),(.6,.35,q),(.6,.5,q)]
        if preferred_place:
            first=(abs(preferred_place[0]),preferred_place[1],q)
            place_options=[first]+[item for item in place_options if abs(item[0]-first[0])>1e-5 or abs(item[1]-first[1])>1e-5]
        for candidate_x,candidate_y,candidate_q in place_options:
            place=[candidate_x if side=='left' else -candidate_x,candidate_y,2.67]
            try:
                self.move_held_linear(place,candidate_q,'place_above')
                drop_y=candidate_y;place_q=candidate_q;reached=True;break
            except PlanningRejected as error:
                self.event('alternate_place_candidate',rejected_y=candidate_y,quaternion=candidate_q,reason=str(error))
        if not reached:raise PlanningRejected('all bounded place candidates rejected')
        down=[place[0],place[1],place[2]-.2]
        self.move(down,place_q,'place_descend',held=True)
        release_command_at=time.time()
        self.active_release_id=uuid.uuid4().hex
        self.event('release_started',side=side,category=selected['class'],drop_y=drop_y,drop_x=abs(place[0]),release_id=self.active_release_id)
        self.open_gripper()
        released_at=time.time();released_monotonic=time.monotonic()
        offset=self.mission_wall_offset if self.mission_wall_offset is not None else released_at-released_monotonic
        self.active_release_context={'candidate':copy.deepcopy(selected),'side':side,'request_id':self.current_request,
            'round_id':self.current_round,'task_id':self.current_task,'release_id':self.active_release_id,
            'released_at':released_at,'released_monotonic':released_monotonic,'drop_y':drop_y,'drop_x':abs(place[0]),
            'deadline_wall':self.deadline+offset,'home_arrived_at':None,'verified':False}
        self.event('released',release_started_at=release_command_at,released_at=released_at,
                   released_monotonic=released_monotonic,release_id=self.active_release_id)
        # Withdraw inward as well as upward so the camera sees the released item.
        retreat=[place[0]-(.22 if side=='left' else -.22),place[1],place[2]]
        self.move(retreat,place_q,'place_retreat')
        self.move(self.home_p,self.home_q,'return_home')
        home_arrived_at=time.time()
        self.active_release_context['home_arrived_at']=home_arrived_at
        began=time.monotonic()
        while time.monotonic()-began<6:
            self.checkpoint()
            with self.lock:history=list(self.scene_history);scene=self.snapshot
            # Source counts are diagnostic only. Missing camera frames after
            # a completed release must reach the pending-verification branch,
            # not abort the round through fresh_scene().
            after_source_count=None
            if scene and 0<=time.time()-scene.get('observed_at',-math.inf)<2.:
                after_source_count=sum(c['class']==selected['class'] for c in self.object_candidates(scene,selected))
            proof=verify_transport(history,selected['class'],side,released_at,drop_y,self.current_request,abs(place[0]),
                                   stable_id=selected['stable_id'],release_id=self.active_release_id)
            if proof is None:
                proof=verify_settled_placement(history,selected['class'],side,released_at,drop_y,self.current_request,home_arrived_at,
                                              stable_id=selected['stable_id'],release_id=self.active_release_id)
            if proof:proof.update(source_count_before=before_source_count,source_count_after=after_source_count,
                                  count_change_is_diagnostic_only=True)
            if proof and not self.holding():
                qualified_times=[sample.get('observed_at') for sample in proof.get('samples',[])
                                 if type(sample.get('observed_at')) in (int,float) and
                                 math.isfinite(sample['observed_at']) and sample['observed_at']>=released_at]
                delivered_at=min(qualified_times) if qualified_times else time.time()
                self.event('placement_verified',side=side,category=selected['class'],transport=proof,
                           release_id=self.active_release_id,
                           released_at=released_at,released_monotonic=released_monotonic,
                           delivered_at=delivered_at,verified_at=time.time(),
                           delivery_time_source=('first_qualified_belt_observation_wall' if qualified_times
                                                 else 'verification_wall_fallback'),
                           belt_observer=self.belt_observer_state(),
                           candidate=selected)
                self.active_release_context['verified']=True
                return True
            time.sleep(.15)
        return self.defer_pending_placement()

    def defer_pending_placement(self):
        context=self.active_release_context
        if (not context or context.get('verified') or not context.get('home_arrived_at')
                or not context.get('released_at') or context.get('release_id')!=self.active_release_id):
            raise RuntimeError('pending placement requires a completed current release and return')
        with self.lock:
            joints=self.joint_feedback;seen=self.joints_seen
            captured=self.captured;capture_seen=self.capture_seen
            fingers=dict(zip(joints.name,joints.position))
        empty=(time.monotonic()-seen<1.5 and time.monotonic()-capture_seen<1.5 and not captured
               and all(abs(fingers.get(name,math.inf))<.001 for name in ('finger1_joint','finger2_joint')))
        actual,_=self.read_pose()
        if not empty or math.dist(actual,self.home_p)>=.008:
            raise RuntimeError('release follow-up cannot continue without fresh empty gripper at observation pose')
        empty_proof={'release_completed':True,'empty_gripper':True,'at_observation_pose':True,
                     'finger_positions':{name:fingers[name] for name in ('finger1_joint','finger2_joint')},
                     'checked_at':time.time()}
        self.active_release_context['empty_release_proof']=empty_proof
        self.pending_deliveries[self.active_release_id]=copy.deepcopy(self.active_release_context)
        self.event('placement_pending_verification',release_id=self.active_release_id,proof=empty_proof,
                   reason='release completed; no coherent belt proof yet',continue_next_task=True,
                   belt_observer=self.belt_observer_state())
        return False

    def verify_pending_delivery_after_stop(self,stop_proof,timeout=2.):
        """Bounded read-only evidence collection; never resumes motion or grip."""
        context=self.active_release_context
        if (not context or context.get('verified') or not stop_proof or stop_proof.get('state')!='stopped'
                or stop_proof.get('id')!=self.stop_id):return False
        if not 0<timeout<=2.5:raise ValueError('post-stop evidence timeout exceeds readonly bound')
        candidate=context['candidate'];until=time.monotonic()+timeout
        while time.monotonic()<until and not rospy.is_shutdown():
            with self.lock:history=list(self.scene_history)
            proof=verify_transport(history,candidate['class'],context['side'],context['released_at'],
                context['drop_y'],context['request_id'],context['drop_x'],
                stable_id=candidate['stable_id'],release_id=context['release_id'])
            if proof is None and context.get('home_arrived_at') is not None:
                proof=verify_settled_placement(history,candidate['class'],context['side'],context['released_at'],
                    context['drop_y'],context['request_id'],context['home_arrived_at'],
                    stable_id=candidate['stable_id'],release_id=context['release_id'])
            timely=[sample['observed_at'] for sample in (proof or {}).get('samples',[])
                    if isinstance(sample.get('observed_at'),(int,float)) and math.isfinite(sample['observed_at'])
                    and context['released_at']<=sample['observed_at']<=context['deadline_wall']]
            try:empty=not self.holding()
            except RuntimeError:empty=False
            if proof and timely and empty:
                self.event('placement_verified',request_id=context['request_id'],round_id=context['round_id'],
                    task_id=context['task_id'],stable_id=candidate['stable_id'],side=context['side'],category=candidate['class'],
                    candidate=candidate,transport=proof,release_id=context['release_id'],
                    released_at=context['released_at'],released_monotonic=context['released_monotonic'],
                    delivered_at=min(timely),verified_at=time.time(),delivery_time_source='first_qualified_belt_observation_wall',
                    belt_observer=self.belt_observer_state(),
                    delayed_verification_after_stop=True,stop_id=self.stop_id,read_only_finalization=True)
                context['verified']=True;return True
            time.sleep(.05)
        self.event('post_stop_delivery_unverified',release_id=context['release_id'],stable_id=candidate['stable_id'],
                   reason='no bound timely belt proof with fresh empty-gripper feedback',read_only_finalization=True)
        return False

    def run(self):
        while not rospy.is_shutdown():
            try:plan=self.queue.get(timeout=.2)
            except queue.Empty:
                self.check_pending_deliveries();continue
            began=time.monotonic();done=0;pending_objects=[];claimed=False
            try:
                with self.lock:
                    if (self.cancelled.is_set() or self.needs_recovery or self.stop_id or
                            type(plan.get('_command_epoch')) is not int or plan['_command_epoch']!=self.command_epoch or
                            plan.get('round_id') in self.cancelled_rounds or plan.get('request_id')!=self.admitted_request):
                        self.event('queued_request_cancelled',**self.request_context(plan),
                                   reason='admission invalidated before worker start')
                        continue
                    self.busy=True;claimed=True;self.current_request=plan['request_id']
                    previous_round=self.current_round
                    self.current_round=plan.get('round_id',self.current_round)
                    self.current_task=plan.get('task_id');self.active_object=None;self.active_side=None
                # No parameter-server call when a fixed deadline was supplied.
                self.deadline=float(plan['deadline_monotonic'] if 'deadline_monotonic' in plan else
                    time.monotonic()+min(600,rospy.get_param('~task_budget',550)))
                start_wall=plan.get('started_at_wall');start_mono=plan.get('started_at_monotonic')
                self.mission_wall_offset=(float(start_wall)-float(start_mono)
                    if all(isinstance(value,(int,float)) and math.isfinite(value) for value in (start_wall,start_mono))
                    else time.time()-time.monotonic())
                if not rospy.get_param('~execute',False):
                    self.event('execution_disabled',plan=plan);continue
                self.checkpoint()
                if self.current_round and self.current_round!=previous_round:
                    self.grasp_history=GraspAttemptHistory();self.active_attempt_id=None
                    self.trial_lift=None;self.last_contact_sim_time=None;self.active_lift_verified=False
                    self.active_release_context=None;self.active_release_id=None
                    self.pending_deliveries={}
                    with self.lock:
                        self.joint_diagnostics.clear();self.depth_history.clear();self.identity_acks.clear()
                    self.event('round_started',deadline_monotonic=self.deadline,
                               started_at_monotonic=plan.get('started_at_monotonic'),
                               started_at_wall=plan.get('started_at_wall'))
                if plan.get('kind')=='prepare_observation':
                    self.event('observation_started',observation_protocol=OBSERVATION_PROTOCOL)
                    self.prepare_observation(deadline=self.deadline)
                    self.event('observation_completed',elapsed=time.monotonic()-began,
                               observation_protocol=OBSERVATION_PROTOCOL)
                    continue
                self.event('task_started')
                if not plan['objects']:
                    self.event('remaining_noop',proof={'complete_empty_scene':True,
                               'semantic':plan.get('semantic'),'automatic_points_claimed':False})
                for c in plan['objects']:
                    if time.monotonic()>self.deadline-60:raise TimeoutError('insufficient remaining time for another object')
                    self.active_object=copy.deepcopy(c);self.active_side=plan['side']
                    if self.execute_object(c,plan['side']) is False:pending_objects.append(c['stable_id'])
                    else:done+=1
                self.event('task_pending_verification' if pending_objects else 'task_succeeded',
                           placed_verified=done,pending_stable_ids=pending_objects,
                           motion_complete=True,elapsed=time.monotonic()-began)
            except Exception as e:
                if not claimed:
                    self.event('plan_rejected',**self.request_context(plan),reason='worker admission error: '+str(e))
                    continue
                self.needs_recovery=True
                stop_id=self.request_stop('task failure: '+str(e))
                stop_proof=self.wait_stopped(stop_id)
                try:
                    if stop_proof and self.verify_pending_delivery_after_stop(stop_proof):done+=1
                except Exception as finalization_error:
                    self.event('post_stop_delivery_check_failed',reason=str(finalization_error),read_only_finalization=True)
                self.event('task_failed',reason=str(e),placed_verified=done,elapsed=time.monotonic()-began,
                           stopped_verified=bool(stop_proof),stop=stop_proof,
                           limitation='Explicit reset or a new controlled run is required; stop success is based on measured feedback.')
            finally:
                with self.lock:
                    if claimed:self.busy=False
                    if self.admitted_request==self.request_context(plan)['request_id']:
                        self.admitted_request=None;self.admitted_context=None
                self.queue.task_done()


if __name__=='__main__':
    rospy.init_node('tcei_controller')
    node=Controller()
    if rospy.get_param('~prepare_observation',False):
        rospy.set_param('~observation_ready',False)
        node.prepare_observation()
    node.run()
