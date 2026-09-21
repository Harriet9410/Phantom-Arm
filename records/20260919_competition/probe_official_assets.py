from pathlib import Path
import json,re,traceback
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting'})
r=Path('/root/tcei_competition_20260919');out={'world_stepped':False,'robot_commands_sent':False}
try:
 from pxr import Usd
 root=Path('/root/EAICON/Content');files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.usd','.usda','.usdc','.usdz','.obj','.fbx','.stl'))
 out['asset_files']=[str(p.relative_to(root)) for p in files]
 stage=Usd.Stage.Open('/root/EAICON/Content/JAKA/scene.usd')
 traverse=getattr(stage,'TraverseAll',stage.Traverse);prims=list(traverse())
 out['inactive_prims_included']=hasattr(stage,'TraverseAll');out['prim_count']=len(prims)
 out['world_children']=[{'path':str(p.GetPath()),'type':p.GetTypeName(),'active':p.IsActive(),'display_name':str(p.GetAllMetadata().get('displayName',''))} for p in stage.GetPrimAtPath('/World').GetChildren()]
 pattern=re.compile('food|ration|compressed|biscuit|干粮|饼干',re.I)
 out['name_matches']=[{'path':str(p.GetPath()),'display_name':str(p.GetAllMetadata().get('displayName',''))} for p in prims if pattern.search(str(p.GetPath())+' '+str(p.GetAllMetadata().get('displayName','')))]
 out['referenced_layers']=[str(layer.identifier) for layer in stage.GetUsedLayers()]
 out['status']='read_success';out['scope']='Authored prim names/display names and used layer/file paths only; no visual class inference and no proof that generically named embedded geometry cannot be food.'
except Exception as e:out.update(status='failed',error=repr(e),traceback=traceback.format_exc())
finally:
 (r/'official_asset_inventory.txt').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');app.close()
