from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from pathlib import Path
import json
s=Usd.Stage.Open('/root/EAICON/Content/JAKA/scene.usd');c=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
rows=[]
for p in s.Traverse():
 if 'RobotArm' in str(p.GetPath()):continue
 if p.HasAPI(UsdPhysics.RigidBodyAPI) or any(w in p.GetName().lower() for w in ['basket','tray','bin','box']):
  try:
   b=c.ComputeWorldBound(p).ComputeAlignedRange();x=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default());q=Gf.Transform(x).GetRotation().GetQuat()
   rows.append({'path':str(p.GetPath()),'rigid':p.HasAPI(UsdPhysics.RigidBodyAPI),'position':list(x.ExtractTranslation()),'quaternion_wxyz':[q.GetReal()]+list(q.GetImaginary()),'bbox_min':list(b.GetMin()),'bbox_max':list(b.GetMax()),'xform_ops':[str(o.GetOpName()) for o in UsdGeom.Xformable(p).GetOrderedXformOps()]})
  except Exception as e:rows.append({'path':str(p.GetPath()),'error':str(e)})
Path('/root/tcei_stress_20260918/scene_inventory.json').write_text(json.dumps(rows,indent=2))
print('INVENTORY_READY',len(rows),flush=True)
app.close()
