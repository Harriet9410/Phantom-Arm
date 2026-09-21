from pathlib import Path
import json, traceback
r=Path('/root/tcei_competition_20260919');app=None;out={'runtime_alignment_verified':False}
try:
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'renderer':'RayTracedLighting'})
 from pxr import Usd,UsdGeom
 import numpy as np
 scene='/root/EAICON/Content/JAKA/scene.usd';stage=Usd.Stage.Open(scene);cache=UsdGeom.XformCache();camera=stage.GetPrimAtPath('/World/Cameras/top')
 roots=[p for p in stage.Traverse() if p.GetName()=='Link_00' and 'RobotArm' in str(p.GetPath())]
 out.update(scene=scene,stage_units=UsdGeom.GetStageMetersPerUnit(stage),robot_roots=[str(p.GetPath()) for p in roots],camera_valid=bool(camera))
 if len(roots)!=1 or not camera:raise ValueError('Robot root or camera missing/ambiguous')
 world_camera=np.array(cache.GetLocalToWorldTransform(camera),dtype=float).T;world_root=np.array(cache.GetLocalToWorldTransform(roots[0]),dtype=float).T
 optical_world=np.diag([1.,-1.,-1.,1.])@np.linalg.inv(world_camera)
 out.update(status='static_geometry_read',world_from_usd_camera=world_camera.tolist(),world_from_Link_00=world_root.tolist(),camera_optical_from_world=optical_world.tolist(),camera_optical_from_Link_00=(optical_world@world_root).tolist(),source='static robot/camera only; runtime overlay pending')
 cam=UsdGeom.Camera(camera);out['camera_lens']={k:float(v.Get()) for k,v in [('focal_length',cam.GetFocalLengthAttr()),('horizontal_aperture',cam.GetHorizontalApertureAttr()),('vertical_aperture',cam.GetVerticalApertureAttr())]}
except Exception as e:out.update(status='failed',error=str(e),traceback=traceback.format_exc())
finally:
 (r/'static_geometry_runtime.txt').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('STATIC_GEOMETRY_RESULT',json.dumps(out),flush=True)
 if app is not None:app.close()
