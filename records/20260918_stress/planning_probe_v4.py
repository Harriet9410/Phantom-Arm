"""Isaac kinematic regression only: no robot targets and no object manipulation."""
import json,sys,math,time
from pathlib import Path
import numpy as np

root=Path('/root/tcei_stress_20260918')
code=root/'candidate_v4/tcei_stack';sys.path.insert(0,str(code))
import sim_with_feedback as runtime
from core import calibrated_grasp
from kinematic_guard import rotation_angle,quaternion_rotation,normalize_configuration

env=runtime.original.SimEnvironment(kit=runtime.original.kit)
output=[]
try:
    env.play();controller=env.robotArmController;art=controller._articulation
    original_positions=art.get_joint_positions
    for case in json.loads((root/'planner_regression_cases.json').read_text()):
        started=time.monotonic();index=case['round'];start=np.array(case['start_diagnostics']['joint_actual'])
        goal=np.array(case['target']['target']);orientation=np.array(case['target']['quaternion'])
        full=original_positions().copy();full[:len(start)]=start
        art.get_joint_positions=lambda *args,**kwargs:full.copy()
        controller._manual_gripper_cmd=.04 if index==2 else 0.
        path=controller.cartesian_candidate(start,goal,orientation)
        fk,fr=controller._kin_solver.compute_forward_kinematics(controller._art_kin.get_end_effector_frame(),start)
        props=art.dof_properties
        result={'round':index,'old_failure':case['rejection'].get('reason'),
                'continuation_points':len(path) if path is not None else 0}
        result.update(start_fk=np.asarray(fk).reshape(-1).tolist(),recorded_tcp=case['start_diagnostics']['tcp_actual'],
                      start_angle_degrees=math.degrees(rotation_angle(fr,quaternion_rotation(orientation))),
                      start_clearance=controller.has_table_clearance(start),
                      base_pose=[x.tolist() for x in art.get_world_pose()],
                      joint_lower=props['lower'][:controller.num_arm_dof].tolist(),joint_upper=props['upper'][:controller.num_arm_dof].tolist())
        samples=[]
        point=np.asarray(fk).reshape(-1)+(goal-np.asarray(fk).reshape(-1))*.1
        for label,position in [('first10percent',point),('goal',goal)]:
            for ptol,rtol in [(.0008,.006),(.002,.02)]:
                joints,ok=controller._kin_solver.compute_inverse_kinematics(controller._art_kin.get_end_effector_frame(),position,orientation,start,ptol,rtol)
                item={'point':label,'ptol':ptol,'rtol':rtol,'ok':bool(ok)}
                if ok:
                    from kinematic_guard import normalize_configuration
                    joints=normalize_configuration(joints,start,props['lower'][:controller.num_arm_dof],props['upper'][:controller.num_arm_dof])
                    p2,r2=controller._kin_solver.compute_forward_kinematics(controller._art_kin.get_end_effector_frame(),joints)
                    item.update(joints=joints.tolist(),max_jump=float(np.max(abs(joints-start))),clearance=controller.has_table_clearance(joints),position_error=float(np.linalg.norm(np.asarray(p2).reshape(-1)-position)),angle_error_deg=math.degrees(rotation_angle(r2,quaternion_rotation(orientation))))
                samples.append(item)
        result['raw_ik']=samples
        run=root/'deploy_runs'/('formal30_v3_%02d'%index)
        events=[json.loads(s) for s in (run/'episode/events.jsonl').read_text().splitlines()]
        candidate=[e['candidate'] for e in events if e['status']=='attempt'][-1]
        height_scan=[];controller._manual_gripper_cmd=0.
        for variant in (0,180):
            p,q=calibrated_grasp(candidate,yaw_variant=variant)
            for z in sorted(set([p[2]+.10,p[2]+.12,p[2]+.14,p[2]+.16,2.579508])):
                above=np.array([p[0],p[1],z]);action,ok=controller.solve_clearance_ik(above,np.array(q))
                item={'variant':variant,'height':z,'lift_delta':z-p[2],'ik_valid':bool(ok)}
                if ok:
                    down=controller.cartesian_candidate(np.asarray(action.joint_positions),np.asarray(p),np.array(q))
                    back=controller.cartesian_candidate(down[-1],above,np.array(q)) if down is not None else None
                    item.update(down_points=len(down) if down is not None else 0,up_points=len(back) if back is not None else 0)
                height_scan.append(item)
        result['height_scan']=height_scan
        if path is not None:
            rotations=[controller._kin_solver.compute_forward_kinematics(controller._art_kin.get_end_effector_frame(),q)[1] for q in path]
            result['max_attitude_degrees']=math.degrees(max(rotation_angle(r,quaternion_rotation(orientation)) for r in rotations))
            result['max_joint_step']=float(np.max(abs(np.diff(path,axis=0))))
        if index==4:
            run=root/'deploy_runs/formal30_v3_04'
            events=[json.loads(s) for s in (run/'episode/events.jsonl').read_text().splitlines()]
            candidate=[e['candidate'] for e in events if e['status']=='attempt'][-1]
            variants=[]
            for variant in (0,180):
                p,q=calibrated_grasp(candidate,yaw_variant=variant);above=np.array(p)+[0,0,.16]
                action,ok=controller.solve_clearance_ik(above,np.array(q))
                variants.append({'variant':variant,'approach_position':above.tolist(),'ik_valid':bool(ok),
                                 'joints':action.joint_positions.tolist() if ok else None})
                valid=[];raw_success=0
                for shoulder in (.3,1.3,2.3):
                    for elbow in (-.5,-1.5,-2.5):
                        for wrist in (2.5,4.5,6.):
                            for roll in (-2.,0.,2.):
                                seed=start.copy();seed[1]=shoulder;seed[2]=elbow;seed[3]=wrist;seed[5]=roll
                                joints,success=controller._kin_solver.compute_inverse_kinematics(controller._art_kin.get_end_effector_frame(),above,np.array(q),seed,.002,.02)
                                if success:
                                    raw_success+=1
                                    try:joints=normalize_configuration(joints,start,props['lower'][:controller.num_arm_dof],props['upper'][:controller.num_arm_dof])
                                    except ValueError:continue
                                    if controller.has_table_clearance(joints):valid.append(joints.tolist())
                variants[-1].update(grid_raw_success=raw_success,grid_valid_count=len(valid),grid_valid_examples=valid[:3])
            result['wrist_variants']=variants
        result['seconds']=time.monotonic()-started;output.append(result)
        print('PROBE_RESULT',json.dumps(result),flush=True)
        art.get_joint_positions=original_positions
    (root/(Path(__file__).stem+'.json')).write_text(json.dumps(output,indent=2))
finally:
    try:art.get_joint_positions=original_positions
    except Exception:pass
    env.close()
