from pathlib import Path
import json,subprocess,hashlib,time,ast,rospy
r=Path('/root/tcei_competition_20260919');runs=Path('/root/gpufree-data/tcei_competition_20260919');rel=r/'releases/dev_v7_t1_06';run=runs/'services06_02';run.mkdir(exist_ok=False)
for name in ('pids','logs','events'):(run/name).mkdir()
assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'));assert json.loads((r/'observation04_C_01/summary.json').read_text())['status']=='observation_completed'
manifest=json.loads((rel/'tcei_stack/BUILD_MANIFEST.json').read_text())
for name,sha in manifest['files'].items():assert hashlib.sha256((rel/'tcei_stack'/name).read_bytes()).hexdigest()==sha
oldsim=r/'releases/dev_v7_t1_04/tcei_stack';todo=['sim_with_feedback'];seen=[]
while todo:
 name=todo.pop()
 if name in seen:continue
 path=oldsim/(name+'.py');new=rel/'tcei_stack'/(name+'.py')
 if name=='core':
  left=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_workspace');right=next(n for n in ast.parse(new.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_workspace');assert ast.dump(left)==ast.dump(right);seen.append('core.validate_workspace_AST_identical');continue
 assert path.read_bytes()==new.read_bytes();seen.append(name);tree=ast.parse(path.read_text())
 for node in ast.walk(tree):
  names=([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [])
  todo.extend(x for x in names if (oldsim/(x+'.py')).exists() and x not in seen)
guard=['/usr/bin/python3',str(rel/'harness/process_guard.py')];stops=[]
for name,origin in [('nine','semantic05_01'),('controller','no_action04_01'),('perception','no_action04_01')]:
 rec=runs/origin/'pids'/(name+'.json');p=subprocess.run(guard+['stop',str(rec),'--timeout','20'],capture_output=True,text=True);stops.append({'name':name,'record':str(rec),'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(run/'prior_stop_results.json').write_text(json.dumps(stops,indent=2));assert p.returncode==0;assert subprocess.run(guard+['check',str(rec)]).returncode!=0
commands=[('controller',['/usr/bin/python3','-u',str(rel/'tcei_stack/controller.py'),'_execute:=false','_prepare_observation:=false','_require_planner_feedback:=true','_log_dir:='+str(run/'events')]),('perception',['/opt/conda/envs/yolov8/bin/python','-u',str(rel/'tcei_stack/perception.py'),'_show_gui:=false','_weights:=/root/jaka/best.pt']),('nine',['/opt/conda/envs/inference/bin/python','-u',str(rel/'tcei_stack/nine_node.py'),'_execute:=false','_model_path:=/root/inference/FM9G4B-V','_log_dir:='+str(run/'events')])]
for name,command in commands:subprocess.run(guard+['start',str(run/'pids'/(name+'.json')),str(rel/'tcei_stack'),str(run/'logs'/(name+'.log')),'--']+command,check=True)
(run/'components.json').write_text(json.dumps({'version':'dev_v7_t1_06','started_at':time.time(),'sim_original_run':'no_action04_01','sim_dependency_verification':seen,'scope':'development test with retained04 simulator; core.validate_workspace identical AST, all other listed simulation dependencies identical bytes; final full-stack restart still required','runtime_manifest':manifest,'motion_enabled':False,'competition_round':False},indent=2));print('SERVICES06_STARTED',flush=True)
