"""Create read-only evidence recorders before independent image review."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import rospy
ROOT=Path('/root/tcei_competition_20260919');RELEASE=ROOT/'releases/dev_v7_t1_18'
RUNS=Path('/root/gpufree-data/tcei_competition_20260919');RUN=RUNS/'official18_round01'
sys.path.insert(0,str(RELEASE/'harness'))
from process_guard import current,read_record
assert json.loads((ROOT/'full17_launcher_result.json').read_text())['returncode']==0
rospy.init_node('tcei_prepare_official18_recorders',anonymous=True)
assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
assert all(current(read_record(RUNS/'official18_stack01/pids'/(n+'.json'))) for n in ('sim','controller','perception','nine'))
RUN.mkdir(exist_ok=False)
for name in ('pids','logs'):(RUN/name).mkdir()
guard=['/usr/bin/python3',str(RELEASE/'harness/process_guard.py')]
helpers=('record_trial_rgbd12.py','rgbd_archive.py','record_scalar_evidence12.py')
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in helpers}
assert hashes['record_scalar_evidence12.py']=='797ff9add8055cecb104e020c58e2d35156db21711ddaeb397fccc577ec36b78'
commands={'rgbd':['/usr/bin/python3','-u',str(ROOT/'record_trial_rgbd12.py'),'_output:='+str(RUN/'rgbd'),'_duration:=900'],
          'scalars':['/usr/bin/python3','-u',str(ROOT/'record_scalar_evidence12.py'),'--output',str(RUN/'scalars'),'--duration','900']}
for name,command in commands.items():
    subprocess.run(guard+['start',str(RUN/'pids'/(name+'.json')),str(ROOT),str(RUN/'logs'/(name+'.log')),'--']+command,check=True)
until=time.monotonic()+30.
while not ((RUN/'rgbd/reference.jpg').exists() and (RUN/'scalars/ready.json').exists()):
    assert time.monotonic()<until,'evidence recorder readiness timeout'
    assert all(current(read_record(RUN/'pids'/(n+'.json'))) for n in commands)
    time.sleep(.15)
(RUN/'preparation.json').open('x').write(json.dumps({'prepared_at':time.time(),'helper_sha256':hashes,
    'recorders':{name:read_record(RUN/'pids'/(name+'.json')) for name in commands},
    'purpose':'same-layout repetition; compare preload/holding losses and verify nonblocking placement follow-up, no physics parameter changes'},indent=2))
config_path=ROOT/'review_ui_02/config.json';config=json.loads(config_path.read_text())
config['episode_dir']=str(RUN/'episode');config['stack_dir']=str(RUNS/'official18_stack01')
config['label']='official18_round01 官方五条原句：渐进预紧、待核验继续、传送带关联修正'
(RUN/'previous_panel_config.json').open('x').write(config_path.read_text())
config['review_image']=str(RUNS/'official16_round01/delivery_review16_01/task02_release_plus_3000ms.jpg')
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
print('OFFICIAL18_RECORDERS_READY',flush=True)
