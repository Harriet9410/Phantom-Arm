from pathlib import Path
import json,subprocess,sys,time,hashlib,rospy
r=Path('/root/tcei_competition_20260919'); old=Path('/root/gpufree-data/tcei_competition_20260919/no_action04_01'); rel=r/'releases/dev_v7_t1_05'; run=old.parent/'semantic05_01'; run.mkdir(exist_ok=False)
for d in ('pids','logs','events'):(run/d).mkdir()
assert rospy.get_param('/tcei_nine/execute') is False and rospy.get_param('/tcei_controller/execute') is False
m=json.loads((rel/'tcei_stack/BUILD_MANIFEST.json').read_text()); oldrel=r/'releases/dev_v7_t1_04'
for name,sha in m['files'].items():
 assert hashlib.sha256((rel/'tcei_stack'/name).read_bytes()).hexdigest()==sha
 if name not in ('planning_prompt.py','nine_node.py'):assert (oldrel/'tcei_stack'/name).read_bytes()==(rel/'tcei_stack'/name).read_bytes()
guard=['/usr/bin/python3',str(rel/'harness/process_guard.py')]
stopped=subprocess.run(guard+['stop',str(old/'pids/nine.json'),'--timeout','20'],capture_output=True,text=True); (run/'stop_prior_nine.json').write_text(json.dumps({'code':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr})); assert stopped.returncode==0
assert subprocess.run(guard+['check',str(old/'pids/nine.json')]).returncode!=0
cmd=['/opt/conda/envs/inference/bin/python','-u',str(rel/'tcei_stack/nine_node.py'),'_execute:=false','_model_path:=/root/inference/FM9G4B-V','_log_dir:='+str(run/'events')]
subprocess.run(guard+['start',str(run/'pids/nine.json'),str(rel/'tcei_stack'),str(run/'logs/nine.log'),'--']+cmd,check=True)
(run/'components.json').write_text(json.dumps({'mode':'development_semantic_only_no_action','nine_release':str(rel),'unchanged_supporting_run':str(old),'unchanged_source_verified':True,'manifest':m,'started_at':time.time()},indent=2)); print('NINE05_STARTED',flush=True)
