#!/usr/bin/env python3
"""Test-only initial placement and read-only truth logging; frozen runtime executes tasks."""
import hashlib
import collections
import json
import math
import os
from pathlib import Path
import random
import runpy
import sys
import time
import numpy as np

CODE=Path(os.environ['TCEI_CODE']).resolve()
OUT=Path(os.environ['TCEI_RUN_DIR']).resolve()
CASE_PATH=Path(os.environ['STRESS_CASE_FILE']).resolve()
CASE=json.loads(CASE_PATH.read_text())
assert Path(os.environ['TCEI_INSTANCE_FILE']).read_text().strip()==os.environ['TCEI_INSTANCE_ID']
from launch_stack import verify_original_case
verify_original_case(CASE_PATH,Path(os.environ['TCEI_CASE_REGISTER']))
manifest=json.loads((CODE/'BUILD_MANIFEST.json').read_text())
assert all(hashlib.sha256((CODE/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
random.seed(CASE['planner_seed'])
np.random.seed(CASE['planner_seed'])
sys.path.insert(0,str(CODE))
sys.path.insert(0,'/root/EAICON/Source/JAKA')
import jaka_sim as original
import rospy
from isaacsim.core.prims import SingleRigidPrim
from pxr import UsdGeom,Usd


def rotate(q, vector):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q)
    v=np.asarray(vector,dtype=float)
    return v+2*np.cross(q[1:],np.cross(q[1:],v)+q[0]*v)


class RandomizedEnvironment(original.SimEnvironment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.test_objects=[]
        self.truth_last=0.
        self.mutation_started=time.time()
        assert not rospy.get_param('/tcei_controller/execute',False)
        assert not rospy.get_param('/tcei_nine/execute',False)
        # Canonicalize the asset transforms BEFORE PhysX builds its actors.
        # Constructing a wrapper after play can leave unitsResolve transforms
        # inconsistent with already-cooked actor geometry in this image.
        for item in CASE['objects']:
            prim=self.stage.GetPrimAtPath(item['path'])
            assert prim.IsValid(),item['path']
            body=SingleRigidPrim(item['path'],name='stress_'+item['object_id'],
                position=np.array(item['requested_position']),
                orientation=np.array(item['requested_quaternion_wxyz']))
            self.world.scene.add(body)
            center=(np.array(item['bbox_min'])+np.array(item['bbox_max']))/2
            q=item['quaternion_wxyz']
            offset=rotate([q[0],-q[1],-q[2],-q[3]],center-np.array(item['position']))
            self.test_objects.append((item,body,offset))
        self.mutation_finished=time.time()

    def play(self):
        try:
            super().play()
            stable=0
            pose_window=collections.deque(maxlen=31)
            settling=[]
            for frame in range(3600):
                self.world.step(render=True)
                poses=[body.get_world_pose() for _,body,_ in self.test_objects]
                pose_window.append(poses)
                position_span=0.;orientation_span=0.
                if len(pose_window)==31:
                    for index in range(len(poses)):
                        positions=np.array([row[index][0] for row in pose_window])
                        quats=np.array([row[index][1] for row in pose_window])
                        dots=np.abs(quats@quats[0]);angles=2*np.arccos(np.clip(dots,0.,1.))
                        position_span=max(position_span,float(np.linalg.norm(np.ptp(positions,axis=0))))
                        orientation_span=max(orientation_span,float(angles.max()))
                else:position_span=orientation_span=float('inf')
                slow=position_span<.002 and orientation_span<.035
                stable=stable+1 if slow else 0
                if frame%30==0:
                    settling.append({'frame':frame,'position_span_m':position_span,
                        'orientation_span_rad':orientation_span,
                        'linear_speeds':[float(np.linalg.norm(body.get_linear_velocity())) for _,body,_ in self.test_objects],
                        'angular_speeds':[float(np.linalg.norm(body.get_angular_velocity())) for _,body,_ in self.test_objects]})
                if stable>=30:break
            (OUT/'settling_trace.json').write_text(json.dumps(settling,indent=2))
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
            truth=self.read_truth()
            xmin,xmax,ymin,ymax=CASE['inner_xy']
            valid=stable>=30
            for row,(item,body,offset) in zip(truth,self.test_objects):
                box=cache.ComputeWorldBound(self.stage.GetPrimAtPath(item['path'])).ComputeAlignedRange()
                row['bbox_min']=list(box.GetMin());row['bbox_max']=list(box.GetMax())
                lo,hi=row['bbox_min'],row['bbox_max']
                row['valid_in_basket']=(xmin-.005<=lo[0] and hi[0]<=xmax+.005 and ymin-.005<=lo[1] and hi[1]<=ymax+.005 and 2.34<lo[2]<2.43 and hi[2]<2.55)
                row['reference_xy_error_m']=float(np.linalg.norm(np.array(row['reference_world'][:2])-item['reference_xy']))
                actual=np.array(row['quaternion_wxyz']);wanted=np.array(item['requested_quaternion_wxyz'])
                dot=abs(float(np.dot(actual,wanted)/np.linalg.norm(actual)/np.linalg.norm(wanted)))
                row['orientation_error_rad']=2*math.acos(min(1.,dot))
                # Actual settled angles are recorded, not forced to remain at
                # the requested attitude against normal gravity/contact.
                # A rounded asset can rotate while settling. Its measured
                # reference may move by about 20 mm without leaving the basket.
                # Bound that settling drift at 40 mm and still enforce the
                # physical in-basket and non-overlap checks independently.
                valid=valid and row['valid_in_basket'] and row['reference_xy_error_m']<.04
            for i,a in enumerate(truth):
                for j,b in enumerate(truth[i+1:],i+1):
                    gap=math.dist(a['reference_world'][:2],b['reference_world'][:2])-self.test_objects[i][0]['footprint_radius_m']-self.test_objects[j][0]['footprint_radius_m']
                    valid=valid and gap>=0
            report={'runtime_instance':os.environ['TCEI_INSTANCE_ID'],'source_instance':CASE.get('instance'),'case_id':CASE['case_id'],'seed':CASE['seed'],'case_sha256':hashlib.sha256(CASE_PATH.read_bytes()).hexdigest(),
                    'mutation_started_at':self.mutation_started,'mutation_finished_at':self.mutation_finished,
                    'ready_at':time.time(),'settle_frames':frame+1,'stable_frames':stable,
                    'maximum_reference_settling_drift_m':.04,'settling_limit_frames':3600,
                    'settling_rule':'31-frame pose window spans less than 2 mm and 0.035 rad, repeated for 30 consecutive frames',
                    'valid':bool(valid),'objects':truth,'ground_truth_not_published_to_controller':True}
            (OUT/'scene_initialized.json').write_text(json.dumps(report,indent=2))
            if not valid:raise RuntimeError('randomized physical scene did not settle within the legal basket region')
            self.log_truth(force=True)
            print('STRESS_SCENE_READY',CASE['case_id'],CASE['seed'],flush=True)
        except Exception as error:
            (OUT/'scene_setup_error.json').write_text(json.dumps({'time':time.time(),'error':repr(error),'case_id':CASE['case_id']},indent=2))
            try:
                from PIL import ImageGrab
                ImageGrab.grab(xdisplay=os.environ['DISPLAY']).save(str(OUT/'setup_failure.png'))
            except Exception:pass
            raise

    def read_truth(self):
        rows=[]
        for item,body,offset in self.test_objects:
            p,q=body.get_world_pose()
            center=np.asarray(p)+rotate(q,offset)
            matrix=UsdGeom.Xformable(self.stage.GetPrimAtPath(item['path'])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            rows.append({'object_id':item['object_id'],'class':item['category'],'path':item['path'],
                         'position':p.tolist(),'quaternion_wxyz':q.tolist(),'reference_world':center.tolist(),
                         # Preserve the authored world transform independently:
                         # the stock grabber temporarily disables rigid physics.
                         'usd_world_matrix':[list(row) for row in matrix],
                         'linear_velocity':body.get_linear_velocity().tolist(),'angular_velocity':body.get_angular_velocity().tolist()})
        return rows

    def log_truth(self,force=False):
        if force or time.monotonic()-self.truth_last>=.2:
            self.truth_last=time.monotonic()
            row={'observed_at':time.time(),'simulation_time':float(self.world.current_time),'objects':self.read_truth()}
            with (OUT/'ground_truth.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')

    def step(self):
        super().step()
        self.log_truth()


original.SimEnvironment=RandomizedEnvironment
runpy.run_path(str(CODE/'sim_with_feedback.py'),run_name='__main__')
