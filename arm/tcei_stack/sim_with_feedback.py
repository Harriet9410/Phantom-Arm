#!/usr/bin/env python3
"""Run the original scene/physics with corrected idle commands and planner ACKs.

Run from /root/EAICON using /root/isaacsim/python.sh. Original files stay intact.
No object pose, collision property, conveyor speed, clock or score is changed.
"""
import sys
import json
import time
import threading
import traceback
import numpy as np
import xml.etree.ElementTree as ET
from trajectory_timing import retime_path,parameterize_joint_path
from kinematic_guard import normalize_configuration,rotation_angle,quaternion_rotation
from joint_path import guarded_joint_line
from cartesian_path import continuation_path
from planning_candidates import ik_seed_groups,evaluate_ik_solution,validate_preview_binding
from controlled_stop import StopLatch
from tcei_workspace import validate_workspace
from grasp_contact_monitor import effort_payload

sys.path.insert(0,'/root/EAICON/Source/JAKA')
import jaka_sim as original
import jaka_env as original_controller

# [相机绑定] 官方 jaka_sim 把 ROS 相机挂在 camera_top 视口的 render product 上
# （`create_viewport_window(..., width=320, height=240)` 那个），于是相机分辨率跟着
# 仿真窗口走。窗口一变，内参 fx/cx/cy 就漂，下游"像素→机器人坐标"的换算全部失效
# —— 实测把桌面从 1512x1008 改成 1920x1080 后相机仍是 1280x720、内参不变，靠的就是
# 这段。它此前只以补丁形式存在于某台服务器上，换实例就丢；现在随包走。
# 做法：在 jaka_sim 建相机之前替换模块级的 add_ros1_camera，忽略传入的视口路径，
# 强制绑到 /World/Cameras/top 上一个固定的 1280x720 render product。官方文件不动。
_ROSCAM_CAMERA='/World/Cameras/top'
_ROSCAM_RESOLUTION=(1280,720)
_roscam_product=None
_add_ros1_camera_official=original.add_ros1_camera


def _add_ros1_camera_pinned(render_product_path,graph_path,rgb_topic,info_topic,depth_topic):
    """Ignore the viewport render product; pin the ROS camera to a fixed resolution."""
    global _roscam_product
    if _roscam_product is None:
        import omni.replicator.core as replicator
        _roscam_product=replicator.create.render_product(_ROSCAM_CAMERA,_ROSCAM_RESOLUTION)
        print('[TCEI] camera pinned: %s %dx%d -> %s'
              % (_ROSCAM_CAMERA,_ROSCAM_RESOLUTION[0],_ROSCAM_RESOLUTION[1],_roscam_product.path))
    _add_ros1_camera_official(str(_roscam_product.path),graph_path,rgb_topic,info_topic,depth_topic)


original.add_ros1_camera=_add_ros1_camera_pinned

import rospy
from std_msgs.msg import String


class StableController(original_controller.JakaRmpFlowController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.planner_pub=rospy.Publisher('/tcei/planner_status',String,queue_size=10,latch=True)
        self.last_goal=None
        self.last_joint_goal=None;self.last_diagnostic=0.
        self.preview_lock=threading.Lock();self.preview_pending=None
        self.preview_pub=rospy.Publisher('/tcei/grasp_preview_response',String,queue_size=1,latch=True)
        self.preview_sub=rospy.Subscriber('/tcei/grasp_preview_request',String,self.on_preview_request,queue_size=1)
        self.stop_lock=threading.RLock();self.stop_latch=StopLatch()
        self.stop_reset_pending=None;self.stop_gripper=None;self.last_stop_publish=0.
        self.stop_ack_pub=rospy.Publisher('/tcei/stop_ack',String,queue_size=10,latch=True)
        self.stop_measurement_pub=rospy.Publisher('/tcei/stop_measurement',String,queue_size=500)
        self.last_stop_measurement_stamp=None
        self.stop_sub=rospy.Subscriber('/tcei/stop_request',String,self.on_stop_request,queue_size=10)
        self.stop_reset_sub=rospy.Subscriber('/tcei/stop_reset',String,self.on_stop_reset,queue_size=1)
        self.preview_cache=None;self.preview_execution=None;self.ik_diagnostics=[]
        self.execution_sub=rospy.Subscriber('/tcei/execute_preview',String,self.on_execute_preview,queue_size=1)
        self.min_motion_seconds=float(rospy.get_param('~min_motion_seconds',.9))
        self.use_direct_path=bool(rospy.get_param('~use_direct_path',True))
        if not .5 <= self.min_motion_seconds <= 2.:
            raise ValueError('min_motion_seconds must be within 0.5..2.0')
        self.joint_diagnostic_pub=rospy.Publisher('/tcei/joint_diagnostics',String,queue_size=1)
        self.contact_pub=rospy.Publisher('/tcei/gripper_contact',String,queue_size=10)
        self.last_contact_pair=None
        robot=ET.parse(kwargs['robot_arm_urdf_path']).getroot()
        self.velocity_limits={j.attrib['name']:float(j.find('limit').attrib['velocity'])
                              for j in robot.findall('joint') if j.find('limit') is not None}
        self.report('ready')

    def get_gripper_efforts(self):
        left=self.left_sensor.get_sensor_reading(use_latest_data=True)
        right=self.right_sensor.get_sensor_reading(use_latest_data=True)
        if hasattr(self,'contact_pub'):
            payload=effort_payload(left,right,self._sim_ctx.current_time,time.time())
            self.last_native_contact=payload
            pair=tuple((row['stamp'],row['valid']) for row in payload['samples'])
            if pair!=self.last_contact_pair:
                self.contact_pub.publish(String(json.dumps(payload,allow_nan=False)))
                self.last_contact_pair=pair
        # Preserve the official legacy interface; the new channel retains the
        # validity lost when the legacy method substitutes zero for invalidity.
        return (left.value if left.is_valid else 0.,right.value if right.is_valid else 0.)

    def native_gripper_state(self):
        return {'captured':bool(self.gripper_is_close),'attachment_present':self.target is not None,
                'drive_command':float(self._manual_gripper_cmd)}

    def audit_native_gripper(self,operation,before):
        # Diagnostic log only. Never publishes an object identity/pose or
        # supplies evidence to task control, grasp verification or scoring.
        try:
            row={'operation':operation,'time':time.time(),'simulation_time':float(self._sim_ctx.current_time),
                 'before':before,'after':self.native_gripper_state(),
                 'contact':getattr(self,'last_native_contact',None),
                 'tcp_actual':np.asarray(self.get_end_effector_pose()[0]).tolist(),
                 'scope':'native implementation audit only; not effective grasp evidence'}
            rospy.loginfo('TCEI_NATIVE_GRIPPER '+json.dumps(row,allow_nan=False))
        except Exception:
            # Recording failure must not change the official plant operation.
            pass

    def open_gripper(self):
        before=self.native_gripper_state()
        result=super().open_gripper()
        self.audit_native_gripper('open',before)
        return result

    def close_gripper(self):
        before=self.native_gripper_state()
        result=super().close_gripper()
        self.audit_native_gripper('close',before)
        return result

    def report(self,status,**extra):
        row={'time':time.time(),'status':status,**(self.last_goal or {}),**extra}
        self.planner_pub.publish(String(json.dumps(row)))

    def on_stop_request(self,msg):
        try:
            value=json.loads(msg.data)
            with self.stop_lock:
                state=self.stop_latch.request(value['id'],value.get('reason','unspecified'),time.monotonic())
                self.stop_ack_pub.publish(String(json.dumps({'id':value['id'],'state':'requested',
                    'latched_state':state,'time':time.time()})))
        except Exception as error:
            self.stop_ack_pub.publish(String(json.dumps({'state':'request_rejected','reason':str(error)})))

    def on_stop_reset(self,msg):
        try:
            value=json.loads(msg.data)
            with self.stop_lock:
                if not self.stop_latch.active or value.get('id')!=self.stop_latch.request_id:
                    raise ValueError('stop reset id mismatch or no active stop')
                self.stop_reset_pending=value['id']
        except Exception as error:
            self.stop_ack_pub.publish(String(json.dumps({'state':'reset_rejected','reason':str(error)})))

    def check_planning_interrupt(self,deadline=None):
        if hasattr(self,'stop_latch') and self.stop_latch.active:raise RuntimeError('stop requested during planning')
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('planning computation budget')

    def clear_accepted_motion(self):
        self._tracking_in_progress=False;self.trajectory=None;self.time_from_start=None
        self.trajectory_start_time=None;self.preview_cache=None;self.preview_execution=None
        with self.preview_lock:self.preview_pending=None
        with original.g_data_lock:original.g_new_goal_received=False

    def process_controlled_stop(self):
        if not hasattr(self,'stop_latch'):return False
        with self.stop_lock:
            if not self.stop_latch.active:return False
            if self.stop_gripper is None:self.stop_gripper=float(self._manual_gripper_cmd)
            self.clear_accepted_motion()
            configuration_error=None
            try:
                physics=self._sim_ctx.get_physics_context()
                solver=physics.get_solver_type();physics_dt=float(self._sim_ctx.get_physics_dt())
            except Exception as error:
                solver=None;physics_dt=None;configuration_error=str(error)
            try:self.stop_latch.configure_solver(solver,physics_dt)
            except ValueError as error:
                self.stop_latch.fault_reason=str(error);self.stop_latch.state='fault'
            try:
                all_positions=np.asarray(self._articulation.get_joint_positions(),dtype=float)
                all_velocities=np.asarray(self._articulation.get_joint_velocities(),dtype=float)
                actual=all_positions[:self.num_arm_dof];velocity=all_velocities[:self.num_arm_dof]
                tcp,_=self.get_end_effector_pose()
            except Exception:
                all_positions=all_velocities=None
                actual=velocity=np.full(self.num_arm_dof,np.nan);tcp=np.full(3,np.nan)
            row=self.stop_latch.sample(actual,velocity,np.asarray(tcp).reshape(-1),
                                      float(self._sim_ctx.current_time),time.monotonic())
            row['time']=time.time()
            row['measurement_protocol']='tcei.stop_measurement.v2'
            row['configuration_error']=configuration_error
            try:self.publish_stop_measurement(all_positions,all_velocities,tcp,solver,physics_dt,configuration_error)
            except Exception as error:row['measurement_recording_error']=str(error)
            target=self.stop_latch.hold_position
            if target is not None:
                # Fixed drive target and zero target velocity brake motion via
                # existing physics; never set actual joint positions/velocities.
                full=self._pack_full(target,self.stop_gripper)
                try:
                    self._articulation.get_articulation_controller().apply_action(
                        original_controller.ArticulationAction(joint_positions=full,
                                                              joint_velocities=np.zeros(len(full))))
                except Exception as error:
                    self.stop_latch.fault_reason='hold drive action failed: '+str(error)
                    self.stop_latch.state='fault'
                    row.update(state='fault',fault_reason=self.stop_latch.fault_reason)
                self.last_joint_goal=target.copy()
            self._manual_gripper_cmd=self.stop_gripper
            if self.stop_reset_pending is not None:
                reset_id=self.stop_reset_pending;self.stop_reset_pending=None
                try:
                    reset=self.stop_latch.reset(reset_id)
                    # Discard commands received while stopped.  Reset itself
                    # causes no movement; caller must send a fresh goal afterward.
                    with original.g_data_lock:
                        original.g_new_goal_received=False
                        original.g_gripper_value=self.stop_gripper
                    self.stop_gripper=None
                    self.stop_ack_pub.publish(String(json.dumps(reset)))
                    return True
                except Exception as error:
                    self.stop_ack_pub.publish(String(json.dumps({'id':reset_id,'state':'reset_rejected','reason':str(error)})))
            if time.monotonic()-self.last_stop_publish>.1:
                self.stop_ack_pub.publish(String(json.dumps(row)));self.last_stop_publish=time.monotonic()
            return True

    def publish_stop_measurement(self,positions,velocities,tcp,solver,physics_dt,configuration_error):
        # Publishing diagnostics must never prevent the holding drive action.
        # Each consumer gets the unmodified state of every physics frame and
        # can independently derive speed instead of trusting a computed value.
        stamp=float(self._sim_ctx.current_time);key=(self.stop_latch.request_id,stamp)
        if not hasattr(self,'stop_measurement_pub') or key==getattr(self,'last_stop_measurement_stamp',None):return
        names=list(self._articulation.dof_names)
        valid=(positions is not None and velocities is not None and positions.shape==(len(names),)
               and velocities.shape==positions.shape and np.asarray(tcp).shape==(3,)
               and np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(tcp).all())
        sample={'protocol':'tcei.stop_measurement.v2','id':self.stop_latch.request_id,
            'simulation_time':stamp,'published_wall':time.time(),'solver_type':solver,
            'physics_dt':physics_dt,'configuration_error':configuration_error,'valid':bool(valid),
            'joint_names':names,'positions':positions.tolist() if valid else None,
            'reported_velocities':velocities.tolist() if valid else None,
            'tcp':np.asarray(tcp).tolist() if valid else None}
        self.stop_measurement_pub.publish(String(json.dumps(sample,allow_nan=False)))
        self.last_stop_measurement_stamp=key

    def forward_and_track(self,gripper_value=None):
        # Preserve grasp command during an active stop.  The official contact
        # model still runs; this does not force attachment or edit object state.
        if hasattr(self,'stop_latch') and self.stop_latch.active:
            if self.stop_gripper is None:self.stop_gripper=float(self._manual_gripper_cmd)
            gripper_value=self.stop_gripper
        super().forward_and_track(gripper_value)

    def on_execute_preview(self,msg):
        try:
            value=json.loads(msg.data)
            if not isinstance(value,dict):raise ValueError('preview execution must be an object')
            position=np.asarray(value['position'],dtype=float)
            wire=np.asarray(value['quaternion_wire'],dtype=float)
            if position.shape!=(3,) or wire.shape!=(4,) or not np.isfinite(position).all() or not np.isfinite(wire).all():
                raise ValueError('invalid preview execution target')
            if not isinstance(value.get('id'),str):raise ValueError('invalid preview execution id')
            with self.stop_lock:
                if self.stop_latch.active:raise ValueError('motion is stop-latched')
                with original.g_data_lock:
                    if self._tracking_in_progress or original.g_new_goal_received:
                        raise ValueError('another motion is already pending')
                    self.preview_execution=value
                    original.g_robotArm_ee_position=position.copy()
                    original.g_robotArm_ee_orientation=wire[[3,0,1,2]].copy()
                    original.g_new_goal_received=True
        except Exception as error:
            self.report('rejected',reason=str(error),planner_method='bound_preview',
                        position=value.get('position') if isinstance(locals().get('value'),dict) else None,
                        quaternion_wire=value.get('quaternion_wire') if isinstance(locals().get('value'),dict) else None)

    def apply_gripper_only(self,gripper_value):
        if self.process_controlled_stop():return
        self._manual_gripper_cmd=float(gripper_value)
        # Preserve the last arm position targets. Replacing them with measured
        # positions each frame removes restoring position error and allows drift.
        count=len(self._articulation.dof_names)
        indices=np.arange(self.num_arm_dof,count,dtype=np.int32)
        if not len(indices):return
        self._articulation.get_articulation_controller().apply_action(
            original_controller.ArticulationAction(
                joint_positions=np.full(len(indices),self._manual_gripper_cmd),
                joint_indices=indices))
        if hasattr(self,'joint_diagnostic_pub'):self.publish_joint_diagnostics()
        if hasattr(self,'preview_lock'):self.process_preview()

    def on_preview_request(self,msg):
        try:value=json.loads(msg.data)
        except Exception:return
        if not isinstance(value,dict):return
        with self.preview_lock:self.preview_pending=value

    def process_preview(self):
        if self._tracking_in_progress:return
        with self.preview_lock:
            request=self.preview_pending;self.preview_pending=None
        if request is None:return
        began=time.monotonic();deadline=began+6.
        result={'id':request.get('id'),'valid':False,'failure_kind':'path_rejected',
                'branch_results':[],'approach_ik_diagnostics':[]}
        self.preview_cache=None;self.ik_diagnostics=[];self.last_ik_call=None
        stage='request_validation'
        try:
            self.check_planning_interrupt(deadline)
            if not isinstance(request.get('id'),str) or not request['id']:raise ValueError('preview requires id')
            grasp=np.asarray(validate_workspace(request['grasp']));above=np.asarray(validate_workspace(request['above']))
            transfer=np.asarray(validate_workspace(request['transfer']));q=np.asarray(request['quaternion'],dtype=float)
            if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.)>.001:raise ValueError('invalid preview orientation')
            if request['side'] not in ('left','right'):raise ValueError('invalid preview side')
            place_y=float(request['place_y'])
            validate_workspace([.68,place_y,2.67])
            stage='read_robot_state'
            start=np.asarray(self._articulation.get_joint_positions()[:self.num_arm_dof]).copy()
            checked=[]
            # First assess the normal group, then bounded extra seeds only if
            # no complete route passed.  At most three distinct approach branches.
            for expanded in (False,True):
                before=len(self.ik_diagnostics)
                stage='approach_ik'
                candidates=self.clearance_ik_candidates(above,q,start,False,expanded,deadline)
                result['approach_ik_diagnostics'].extend(self.ik_diagnostics[before:])
                for _,goal in candidates:
                    if any(np.max(abs(goal-old))<1e-4 for old in checked):continue
                    if len(checked)>=3:break
                    checked.append(goal.copy());branch={'joint_goal':goal.tolist()}
                    result['branch_results'].append(branch)
                    stage='approach_path';approach=self.path_to_joint_goal(start,goal,False,deadline)
                    if approach is None:
                        branch.update(valid=False,reason='approach path failed guards',
                                      rejection=getattr(self,'last_path_rejection',None));continue
                    state=goal.copy();points=len(approach);failed=False
                    # R4-3 口径对齐（10/7）：held 分段与执行一致
                    # （controller ~held_transfer_step=.04，原预演 0.12 偏松）。
                    for phase,target,held,step in [('descend',grasp,False,None),('lift',above,True,.04),('transfer',transfer,True,.04)]:
                        stage=phase+'_path'
                        path=self.preview_move(state,target,q,held,step,deadline=deadline)
                        if path is None:
                            branch.update(valid=False,reason='no guarded executable path',phase=phase)
                            failed=True;break
                        points+=len(path);state=path[-1]
                    if failed:continue
                    sign=1 if request['side']=='left' else -1
                    # R4-3 口径对齐（10/7）：预演与执行共用同一份有序投放候选
                    # （controller.py Fix E：y≤0.21 可证明带），预演接受的路线
                    # 才是实际将走的路线；旧 (.6,.35)/(.6,.5) 远端候选已废弃。
                    for x,y in [(.68,place_y),(.65,.2),(.6,.15)]:
                        place=np.array([sign*x,y,2.67])
                        stage='place_across_path'
                        across=self.preview_move(state,place,q,True,.04,deadline=deadline)
                        if across is None:continue
                        stage='place_descend_path'
                        down=self.preview_move(across[-1],place-np.array([0,0,.2]),q,True,None,deadline=deadline)
                        if down is None:continue
                        self.check_planning_interrupt(deadline)
                        branch.update(valid=True)
                        result.update(valid=True,failure_kind=None,place=place.tolist(),approach_joint_goal=goal.tolist(),
                            start_joints=start.tolist(),checked_points=points+len(across)+len(down),
                            checked_stages=['approach_path','descend','lift','transfer','place_above','place_descend'],
                            limitation='Joint bounds, sampled link/table clearance and held attitude only; no full mesh, obstacle or self-collision certification.')
                        self.preview_cache={'id':request['id'],'position':above.tolist(),'orientation':q.tolist(),
                            'start_joints':start.tolist(),'path':approach.copy(),'created_at':time.monotonic()}
                        break
                    if result['valid']:break
                    branch.update(valid=False,reason='no continuous same-attitude placement route')
                if result['valid'] or len(checked)>=3:break
            if not result['valid']:result['reason']='bounded approach branches exhausted'
        except Exception as error:
            kind=('cancelled' if getattr(self,'stop_latch',None) is not None and self.stop_latch.active else
                  'planning_timeout' if isinstance(error,TimeoutError) else
                  'invalid_request' if stage=='request_validation' and isinstance(error,(ValueError,TypeError,KeyError)) else 'internal_error')
            result.update(reason=str(error),failure_kind=kind,error_type=type(error).__name__,
                          error_stage=stage,error_traceback=traceback.format_exc(),ik_call=self.last_ik_call)
        result['seconds']=time.monotonic()-began
        result['ik_diagnostics']=self.ik_diagnostics[-120:]
        self.preview_pub.publish(String(json.dumps(result)))

    def preview_move(self,start,target,orientation,held,max_step,deadline):
        frame=self._art_kin.get_end_effector_frame()
        origin,_=self._kin_solver.compute_forward_kinematics(frame,start)
        origin=np.asarray(origin).reshape(-1);target=np.asarray(target)
        segments=max(1,int(np.ceil(np.linalg.norm(target-origin)/max_step))) if max_step else 1
        paths=[];state=np.asarray(start)
        for index in range(1,segments+1):
            point=origin+(target-origin)*(index/segments)
            path=self.joint_line_candidate(state,point,orientation,held,deadline=deadline)
            if path is None:path=self.cartesian_candidate(state,point,orientation,deadline=deadline)
            if path is None:return None
            paths.append(path if not paths else path[1:]);state=path[-1]
        return np.concatenate(paths)

    def publish_joint_diagnostics(self):
        simulation_time=float(self._sim_ctx.current_time)
        if (time.monotonic()-self.last_diagnostic<.03 or
                simulation_time==getattr(self,'last_robot_simulation_time',None)):return
        self.last_diagnostic=time.monotonic()
        self.last_robot_simulation_time=simulation_time
        all_positions=np.asarray(self._articulation.get_joint_positions(),dtype=float)
        all_velocities=np.asarray(self._articulation.get_joint_velocities(),dtype=float)
        names=list(self._articulation.dof_names)
        if all_positions.shape!=(len(names),) or not np.isfinite(all_positions).all():return
        actual=all_positions[:self.num_arm_dof]
        actual_tcp,actual_quaternion=self.get_end_effector_pose()
        _,actual_rotation=self._kin_solver.compute_forward_kinematics(
            self._art_kin.get_end_effector_frame(),actual)
        goal_tcp=None
        if self.last_joint_goal is not None:
            goal_tcp,_=self._kin_solver.compute_forward_kinematics(
                self._art_kin.get_end_effector_frame(),self.last_joint_goal)
        row={'time':time.time(),'tracking':bool(self._tracking_in_progress),
             # This entire sample is acquired on the same simulation thread
             # without advancing physics.  Camera ROS stamps use simulation
             # time; wall time is transport diagnostics, never camera pairing.
             'simulation_time':simulation_time,'all_joint_names':names,
             'all_joint_positions':all_positions.tolist(),
             'all_joint_velocities':all_velocities.tolist() if all_velocities.shape==all_positions.shape and np.isfinite(all_velocities).all() else None,
             'joint_target':self.last_joint_goal.tolist() if self.last_joint_goal is not None else None,
             'joint_actual':actual.tolist(),
             'joint_error':(self.last_joint_goal-actual).tolist() if self.last_joint_goal is not None else None,
             'tcp_actual':np.asarray(actual_tcp).reshape(-1).tolist(),
             'tcp_rotation_matrix':np.asarray(actual_rotation).tolist(),
             'tcp_quaternion_wire':[float(actual_quaternion[i]) for i in (1,2,3,0)],
             'tcp_goal_fk':np.asarray(goal_tcp).reshape(-1).tolist() if goal_tcp is not None else None,
             'goal':self.last_goal}
        self.joint_diagnostic_pub.publish(String(json.dumps(row)))

    def plan_and_execute_trajectory(self,target_pos,target_rot_wxyz,duration=5.):
        self.last_goal={'position':list(map(float,target_pos)),
                       'quaternion_wire':[float(target_rot_wxyz[i]) for i in (1,2,3,0)]}
        self.report('planning')
        began=time.monotonic()
        try:
            self.check_planning_interrupt()
            start_position,_=self.get_end_effector_pose()
            if self._tracking_in_progress:
                self.report('rejected',reason='previous trajectory still tracking')
                return
            deadline=began+6.;self.active_planning_deadline=deadline
            binding=self.preview_execution;self.preview_execution=None
            cached=self.preview_cache;self.preview_cache=None
            bound=False
            if binding is not None:
                orientation=np.array([target_rot_wxyz[i] for i in (1,2,3,0)])
                start=self._articulation.get_joint_positions()[:self.num_arm_dof]
                path=validate_preview_binding(cached,binding['id'],target_pos,orientation,start,time.monotonic())
                # Measured drift within the tight binding tolerance still needs
                # a checked connection; a changed branch never executes silently.
                prefix=guarded_joint_line(start,path[0],lambda state:self.state_is_valid(state,deadline=deadline))
                if prefix is None:raise ValueError('preview start connection invalid')
                path,path_times=parameterize_joint_path(np.concatenate([prefix,path[1:]]),duration)
                self.set_trajectory(path,duration);self.time_from_start=path_times;bound=True
            local=self.use_direct_path and np.linalg.norm(target_pos-start_position)<=.35
            # Preserve the validated local joint motion when its guards pass.
            # A constrained Cartesian continuation is the next bounded option,
            # before falling back to the original unconstrained sampler.
            direct=not bound and local and self.prepare_direct_trajectory(target_pos,target_rot_wxyz,duration)
            cartesian=not bound and not direct and local and self.prepare_cartesian_trajectory(target_pos,target_rot_wxyz,duration)
            if not bound and not cartesian and not direct:
                orientation=np.array([target_rot_wxyz[i] for i in (1,2,3,0)])
                start=self._articulation.get_joint_positions()[:self.num_arm_dof]
                gripping=abs(self._manual_gripper_cmd)>.001
                candidates=self.clearance_ik_candidates(target_pos,orientation,start,gripping,False,deadline)
                candidates+=self.clearance_ik_candidates(target_pos,orientation,start,gripping,True,deadline)
                for _,goal in candidates[:3]:
                    path=self.path_to_joint_goal(start,goal,gripping,deadline)
                    if path is not None:self.set_trajectory(path,duration);break
            timing={'planner_method':('bound_preview' if bound else 'guarded_cartesian_continuation' if cartesian else 'guarded_joint_line' if direct else 'guarded_trrt_fallback')}
            if bound:timing['route_parameterization']='joint_arc_length'
            if self._tracking_in_progress:
                self.last_joint_goal=self.trajectory[-1].copy()
                goal_tcp,goal_rotation=self._kin_solver.compute_forward_kinematics(
                    self._art_kin.get_end_effector_frame(),self.last_joint_goal)
                timing['goal_fk_error']=float(np.linalg.norm(np.asarray(goal_tcp).reshape(-1)-target_pos))
                if timing['goal_fk_error']>.005:
                    raise ValueError('IK result does not reproduce requested position')
                gripping=abs(self._manual_gripper_cmd)>.001
                max_attitude_deviation=0.
                for state in self.trajectory:
                    if not self.state_is_valid(state,deadline=deadline):
                        raise ValueError('sampled arm path violates joint limits or table clearance')
                    if gripping:
                        _,rotation=self._kin_solver.compute_forward_kinematics(self._art_kin.get_end_effector_frame(),state)
                        max_attitude_deviation=max(max_attitude_deviation,rotation_angle(rotation,goal_rotation))
                timing['max_held_attitude_deviation_rad']=max_attitude_deviation
                if gripping and max_attitude_deviation>.035:
                    raise ValueError('held path changes tool attitude by %.2f degrees'%np.degrees(max_attitude_deviation))
                desired=max(self.min_motion_seconds,min(5.,float(np.linalg.norm(target_pos-start_position))/.15))
                timing['desired_seconds']=desired
                timing['min_motion_seconds']=self.min_motion_seconds
                names=self._articulation.dof_names[:self.num_arm_dof]
                limits=[.5*self.velocity_limits[name] for name in names]
                self.trajectory,self.time_from_start,retimed=retime_path(self.trajectory,self.time_from_start,desired,limits,2.)
                # Retiming interpolates the route.  Check every resulting sample,
                # not a fixed 60-point subset, with the same held-attitude guard.
                for state in self.trajectory:
                    if not self.state_is_valid(state,goal_rotation if gripping else None,deadline):
                        raise ValueError('retimed path violates kinematic guards')
                timing.update(retimed)
                self.trajectory_duration=timing['duration']
            self.check_planning_interrupt(deadline)
            self.report('planned' if self._tracking_in_progress else 'rejected',
                        reason=None if self._tracking_in_progress else 'bounded guarded paths exhausted',
                        planning_seconds=time.monotonic()-began,timing=timing,
                        rejection=getattr(self,'last_path_rejection',None))
        except Exception as error:
            self._tracking_in_progress=False
            self.trajectory=None;self.time_from_start=None;self.trajectory_start_time=None
            self.report('rejected',reason=str(error),planning_seconds=time.monotonic()-began)
            return
        finally:self.active_planning_deadline=None

    def set_trajectory(self,path,duration):
        self.trajectory=np.asarray(path);self.time_from_start=np.linspace(0.,duration,len(path))
        self.trajectory_duration=duration;self.trajectory_start_time=None;self._tracking_in_progress=True

    def state_is_valid(self,state,held_rotation=None,deadline=None):
        self.check_planning_interrupt(deadline)
        props=self._articulation.dof_properties
        state=np.asarray(state)
        if state.shape!=(self.num_arm_dof,) or not np.isfinite(state).all():
            self.last_path_rejection={'reason':'invalid_joint_state'};return False
        if np.any(state<props['lower'][:self.num_arm_dof]) or np.any(state>props['upper'][:self.num_arm_dof]):
            self.last_path_rejection={'reason':'path_joint_limits','joints':state.tolist()};return False
        clearance=self.table_clearance_rejection(state)
        if clearance is not None:
            self.last_path_rejection={'reason':'path_table_clearance',**clearance};return False
        if held_rotation is not None:
            _,rotation=self._kin_solver.compute_forward_kinematics(self._art_kin.get_end_effector_frame(),state)
            angle=rotation_angle(rotation,held_rotation)
            if angle>.035:
                self.last_path_rejection={'reason':'path_held_attitude','angle':angle,'maximum':.035};return False
        return True

    def path_to_joint_goal(self,start,goal,holding,deadline):
        _,rotation=self._kin_solver.compute_forward_kinematics(self._art_kin.get_end_effector_frame(),goal)
        valid=lambda state:self.state_is_valid(state,rotation if holding else None,deadline)
        direct=guarded_joint_line(start,goal,valid)
        if direct is not None:return direct
        if not valid(start) or not valid(goal):return None
        props=self._articulation.dof_properties
        limits=list(zip(props['lower'][:self.num_arm_dof],props['upper'][:self.num_arm_dof]))
        # The original algorithm and constraints are retained, but its former
        # unconditional validator is replaced.  Avoid cubic splines overshooting
        # limits by preserving and linearly subdividing every sampled edge.
        raw=original_controller.trrt_star_optimized(start,goal,valid,limits,step_size=.01,radius=.5)
        if raw is None:return None
        edges=[]
        for first,second in zip(raw[:-1],raw[1:]):
            edge=guarded_joint_line(first,second,valid)
            if edge is None:return None
            edges.append(edge if not edges else edge[1:])
        return np.concatenate(edges) if edges else None

    def continuation_ik(self,position,orientation,reference):
        props=self._articulation.dof_properties
        lower,upper=props['lower'][:self.num_arm_dof],props['upper'][:self.num_arm_dof]
        seeds=[np.asarray(reference).copy()]
        for direction in (-1.,1.):
            seed=np.asarray(reference).copy();seed[1]+=.08*direction;seed[2]-=.08*direction;seeds.append(seed)
            seed=np.asarray(reference).copy();seed[-1]+=.12*direction;seeds.append(seed)
        frame=self._art_kin.get_end_effector_frame()
        desired_rotation=quaternion_rotation(orientation)
        for seed in seeds:
            self.check_planning_interrupt(getattr(self,'active_planning_deadline',None))
            joints,ok=self._kin_solver.compute_inverse_kinematics(frame,position,orientation,seed,.0008,.006)
            self.check_planning_interrupt(getattr(self,'active_planning_deadline',None))
            if not ok:continue
            try:joints=normalize_configuration(joints,reference,lower,upper)
            except ValueError:continue
            if np.max(abs(joints-reference))>.20 or not self.has_table_clearance(joints):continue
            actual,rotation=self._kin_solver.compute_forward_kinematics(frame,joints)
            if np.linalg.norm(np.asarray(actual).reshape(-1)-position)>.002:continue
            if rotation_angle(rotation,desired_rotation)>.012:continue
            return joints
        return None

    def cartesian_candidate(self,start,target,orientation,deadline=None):
        frame=self._art_kin.get_end_effector_frame()
        position,rotation=self._kin_solver.compute_forward_kinematics(frame,start)
        desired=quaternion_rotation(orientation)
        if rotation_angle(rotation,desired)>.035:return None
        def valid(state):
            self.check_planning_interrupt(deadline)
            if not self.state_is_valid(state,deadline=deadline):return False
            _,r=self._kin_solver.compute_forward_kinematics(frame,state)
            return rotation_angle(r,desired)<=.035
        def solve(position,reference):
            self.check_planning_interrupt(deadline)
            return self.continuation_ik(position,orientation,reference)
        return continuation_path(start,np.asarray(position).reshape(-1),target,solve,valid)

    def prepare_cartesian_trajectory(self,target_pos,target_rot_wxyz,duration):
        orientation=np.array([target_rot_wxyz[i] for i in (1,2,3,0)])
        start=self._articulation.get_joint_positions()[:self.num_arm_dof]
        path=self.cartesian_candidate(start,target_pos,orientation,deadline=getattr(self,'active_planning_deadline',None))
        if path is None:return False
        self.trajectory=path;self.time_from_start=np.linspace(0.,duration,len(path))
        self.trajectory_duration=duration;self.trajectory_start_time=None;self._tracking_in_progress=True
        return True

    def prepare_direct_trajectory(self,target_pos,target_rot_wxyz,duration):
        """Local candidate; retain the original planner when any guard fails."""
        orientation=np.array([target_rot_wxyz[i] for i in (1,2,3,0)])
        start=self._articulation.get_joint_positions()[:self.num_arm_dof]
        gripping=abs(self._manual_gripper_cmd)>.001
        path=self.joint_line_candidate(start,target_pos,orientation,gripping,
                                       deadline=getattr(self,'active_planning_deadline',None),emit=True)
        if path is None:return False
        self.trajectory=path
        self.time_from_start=np.linspace(0.,duration,len(path))
        self.trajectory_duration=duration
        self.trajectory_start_time=None
        self._tracking_in_progress=True
        return True

    def joint_line_candidate(self,start,target_pos,orientation,gripping,deadline=None,emit=False):
        self.check_planning_interrupt(deadline)
        tried=[]
        for expanded in (False,True):
            candidates=self.clearance_ik_candidates(target_pos,orientation,start,gripping,expanded,deadline)
            for _,goal in candidates[:3]:
                if any(np.max(abs(goal-old))<1e-4 for old in tried):continue
                tried.append(goal)
                _,rotation=self._kin_solver.compute_forward_kinematics(self._art_kin.get_end_effector_frame(),goal)
                path=guarded_joint_line(start,goal,lambda state:self.state_is_valid(state,rotation if gripping else None,deadline))
                if path is not None:
                    if emit:self.report('ik_selected',joint_goal=goal.tolist(),valid_candidates=len(candidates))
                    return path
                self.ik_diagnostics.append({'reason':'joint_line_guard_rejected','joint_goal':goal.tolist(),
                                           'rejection':getattr(self,'last_path_rejection',None)})
        return None

    def has_table_clearance(self,joints):
        # The robot is mounted on this table plane. These link-center margins
        # are conservative checks, not whole-mesh/self-collision certification.
        return self.table_clearance_rejection(joints) is None

    def table_clearance_rejection(self,joints):
        base_z=float(self._articulation.get_world_pose()[0][2])
        for name,margin in [('Link_03',.09),('Link_04',.07)]:
            p,_=self._kin_solver.compute_forward_kinematics(name,joints)
            height=float(np.asarray(p).reshape(-1)[2])
            if not np.isfinite(height):return {'link':name,'detail':'nonfinite forward kinematics'}
            if height<base_z+margin:return {'link':name,'height':height,'minimum_height':base_z+margin}
        return None

    def clearance_ik_candidates(self,target_position,target_orientation,current,gripping,expanded=False,deadline=None):
        self.check_planning_interrupt(deadline)
        defaults=getattr(self._kin_solver,'_default_cspace_seeds',None)
        self.last_ik_call={'step':'joint_limits','solver_type':type(self._kin_solver).__module__+'.'+type(self._kin_solver).__name__,
                          'target_position':np.asarray(target_position).tolist(),
                          'target_orientation':None if target_orientation is None else np.asarray(target_orientation).tolist(),
                          'default_seeds_type':type(defaults).__name__,
                          'default_seed_count':len(defaults) if isinstance(defaults,(list,tuple)) else None}
        current=np.asarray(current)
        props=self._articulation.dof_properties
        lower,upper=props['lower'][:self.num_arm_dof],props['upper'][:self.num_arm_dof]
        names=self._articulation.dof_names[:self.num_arm_dof]
        self.last_ik_call['step']='joint_order';solver_names=self._kin_solver.get_joint_names()
        self.last_ik_call['solver_joint_names_type']=type(solver_names).__name__
        if list(solver_names)!=list(names):raise ValueError('kinematics/articulation joint ordering differs')
        self.last_ik_call['step']='seed_generation'
        groups=ik_seed_groups(current,lower,upper)
        seeds=groups[1] if expanded else groups[0]
        frame=self._art_kin.get_end_effector_frame();candidates=[];diagnostics=[]
        for name,seed in seeds:
            self.check_planning_interrupt(deadline)
            began=time.monotonic()
            self.last_ik_call.update(step='inverse_kinematics',seed_name=name,warm_start=seed.tolist())
            joints,ok=self._kin_solver.compute_inverse_kinematics(frame,target_position,target_orientation,seed,.002,.02)
            self.last_ik_call.update(step='solution_validation',solver_success=bool(ok))
            self.check_planning_interrupt(deadline)
            row={'seed':name,'seconds':time.monotonic()-began,'target_position':list(map(float,target_position))}
            if not ok:row['reason']='ik_not_converged'
            else:
                if joints is None:raise RuntimeError('IK reported success without joint positions')
                joints,detail=evaluate_ik_solution(joints,current,lower,upper,gripping,
                    target_position,target_orientation,lambda q:self._kin_solver.compute_forward_kinematics(frame,q),
                    self.table_clearance_rejection)
                row.update(detail)
                if joints is not None:
                    if any(np.max(abs(joints-old))<1e-4 for _,old in candidates):row['reason']='duplicate_branch'
                    else:candidates.append((float(np.linalg.norm(joints-current)),joints))
            diagnostics.append(row)
        self.ik_diagnostics.extend(diagnostics)
        self.ik_diagnostics=self.ik_diagnostics[-500:]
        return sorted(candidates,key=lambda pair:pair[0])

    def solve_clearance_ik(self,target_position,target_orientation=None,position_tolerance=None,orientation_tolerance=None,emit=True,reference=None,holding=None):
        current=self._articulation.get_joint_positions()[:self.num_arm_dof] if reference is None else np.asarray(reference)
        gripping=abs(self._manual_gripper_cmd)>.001 if holding is None else bool(holding)
        candidates=self.clearance_ik_candidates(target_position,target_orientation,current,gripping,
                                                False,getattr(self,'active_planning_deadline',None))
        if not candidates:candidates=self.clearance_ik_candidates(target_position,target_orientation,current,gripping,
                                                                  True,getattr(self,'active_planning_deadline',None))
        if not candidates:
            if emit:self.report('ik_no_table_clearance_solution',diagnostics=self.ik_diagnostics[-12:])
            return original_controller.ArticulationAction(),False
        _,selected=min(candidates,key=lambda pair:pair[0])
        if emit:self.report('ik_selected',joint_goal=selected.tolist(),valid_candidates=len(candidates))
        return original_controller.ArticulationAction(joint_positions=selected),True

    def update_trajectory_tracking(self,gripper_value):
        if self.process_controlled_stop():return
        was_tracking=self._tracking_in_progress
        super().update_trajectory_tracking(gripper_value)
        self.publish_joint_diagnostics()
        if was_tracking and not self._tracking_in_progress:self.report('tracking_finished')


original.JakaRmpFlowController=StableController
if __name__=='__main__':
    environment=original.SimEnvironment(kit=original.kit)
    try:
        environment.play()
        while original.kit.is_running() and not rospy.is_shutdown():environment.step()
    finally:
        environment.close()
