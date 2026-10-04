"""Start a separately owned V7 no-action stack; never move to observation here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

HERE=Path(__file__).resolve().parent


def configuration(environ):
    return {key:environ.get('TCEI_'+key.upper(),'') for key in
        ('code','runs','scene','weights','model','isaac_py','yolo_py','nine_py','robot_calibration','instance_file','instance_id')}


def verify_resources(config):
    required_files=[Path(config['scene'])/'Content/JAKA/scene.usd',Path(config['weights']),
                    Path(config['code'])/'BUILD_MANIFEST.json',Path(config['instance_file'])]
    for path in required_files:
        if not path.is_file():raise ValueError('required file missing: '+str(path))
    for key in ('isaac_py','yolo_py','nine_py'):
        if not os.access(config[key],os.X_OK):raise ValueError('required executable missing: '+config[key])
    if not Path(config['model']).is_dir():raise ValueError('Nine model directory missing: '+config['model'])
    if Path(config['instance_file']).read_text(encoding='utf-8').strip()!=config['instance_id']:raise ValueError('simulation instance identity mismatch')
    manifest=json.loads((Path(config['code'])/'BUILD_MANIFEST.json').read_text(encoding='utf-8'))
    files=manifest.get('files',{})
    for name in ('controller.py','perception.py','nine_node.py','sim_with_feedback.py','run_episode.py','episode_driver.py','mission_ledger.py'):
        if name not in files:raise ValueError('frozen runtime manifest lacks '+name)
    for name,expected in files.items():
        if Path(name).name!=name:raise ValueError('manifest paths must be flat runtime filenames')
        path=Path(config['code'])/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('runtime differs from frozen manifest: '+str(path))
    if config['robot_calibration'] and not Path(config['robot_calibration']).is_file():
        raise ValueError('configured robot projection calibration file missing')
    return manifest


DETECTORS=('nine','yolo')


def commands(config,run_dir,case_file=None,detector='nine'):
    code=Path(config['code']);logs=str(Path(run_dir)/'events')
    sim=[config['isaac_py'],str(code/'sim_with_feedback.py')]
    if case_file:
        sim=['/usr/bin/env','STRESS_CASE_FILE='+str(case_file),config['isaac_py'],str(HERE/'original_layout_sim.py')]
    controller=['/usr/bin/python3','-u',str(code/'controller.py'),'_execute:=false',
                '_prepare_observation:=false','_require_planner_feedback:=true','_log_dir:='+logs,
                '_held_transfer_step:=0.04']
    if config['robot_calibration']:controller+=['_robot_projection_calibration:='+config['robot_calibration']]
    # The detector choice drives BOTH perception switches together, so the
    # inconsistent combination (YOLO weights loaded while classifying by nine)
    # is unreachable.  'nine' keeps the competition form: YOLO weights not
    # loaded, classification via the nine-grid model.
    perception=[config['yolo_py'],'-u',str(code/'perception.py'),'_show_gui:=false',
                '_weights:='+config['weights'],
                '_yolo_enabled:='+('true' if detector=='yolo' else 'false'),
                '_detector:='+detector,'_classify_timeout:=25',
                '_classify_fastpath:=true']
    # The nine node is launched in BOTH modes: it owns the semantic plan and the
    # /tcei/nine_status evidence chain that the episode driver requires, so a
    # round cannot complete without it.  Switching the detector only changes who
    # produces the class boxes feeding that plan.
    return [('sim',config['scene'],sim),('controller',str(code),controller),
        ('perception',str(code),perception),
        ('nine',str(code),[config['nine_py'],'-u',str(code/'nine_node.py'),'_execute:=false','_model_path:='+config['model'],'_log_dir:='+logs])]


def verify_original_case(path,register_path):
    if not register_path.is_file():raise ValueError('B01 original-layout register missing: '+str(register_path))
    data=json.loads(path.read_text(encoding='utf-8'));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    registered=json.loads(register_path.read_text(encoding='utf-8')).get('cases',[])
    matching=[row for row in registered if row.get('sha256')==digest and row.get('case_id')==data.get('case_id')
              and row.get('seed')==data.get('seed') and row.get('planner_seed')==data.get('planner_seed')]
    if len(matching)!=1:raise ValueError('case is not an exact registered B01 layout; do not substitute the earlier 20260918xx seeds')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('--case',type=Path)
    parser.add_argument('--case-register',type=Path,default=HERE.parent/'B01_REGISTER.json')
    parser.add_argument('--detector',choices=DETECTORS,default=os.environ.get('TCEI_DETECTOR') or 'nine',
                        help="perception detector: 'nine' (default; competition form) or 'yolo'")
    args=parser.parse_args()
    if args.detector not in DETECTORS:
        parser.error('invalid detector (from env TCEI_DETECTOR): %r'%args.detector)
    detector=args.detector
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',args.run):parser.error('invalid unique run name')
    config=configuration(os.environ);manifest=verify_resources(config)
    master=urlparse(os.environ.get('ROS_MASTER_URI','http://localhost:11311')).hostname
    if master not in ('localhost','127.0.0.1',socket.gethostname()):raise ValueError('only a local simulation ROS master is allowed')
    display=os.environ.get('DISPLAY')
    if not display or not shutil.which('xdpyinfo'):raise ValueError('verified DISPLAY and xdpyinfo required by the supplied graphical scene launcher')
    subprocess.run(['xdpyinfo','-display',display],check=True,timeout=10,stdout=subprocess.DEVNULL)
    guard=[sys.executable,str(HERE/'process_guard.py')]
    subprocess.run(guard+['conflicts'],check=True)
    run_dir=Path(config['runs'])/args.run;run_dir.mkdir(parents=True,exist_ok=False)
    for name in ('pids','logs','events'): (run_dir/name).mkdir()
    os.environ['TCEI_RUN_DIR']=str(run_dir);os.environ['TCEI_CODE']=config['code']
    os.environ['TCEI_DETECTOR']=detector
    os.environ.pop('STRESS_CASE_FILE',None)
    if args.case:
        data=verify_original_case(args.case,args.case_register)
        os.environ['TCEI_CASE_REGISTER']=str(args.case_register.resolve())
        if not data.get('objects'):raise ValueError('case objects invalid')
        shutil.copyfile(args.case,run_dir/'case.json')
    spec=commands(config,run_dir,run_dir/'case.json' if args.case else None,detector)
    (run_dir/'launch.json').write_text(json.dumps({'created_at':time.time(),'config':config,'display':display,
        'detector':detector,'runtime_manifest':manifest,'commands':spec,'mode':'no_action_warmup','observation_moved_before_t0':False},indent=2))
    def start(name,cwd,command):
        subprocess.run(guard+['start',str(run_dir/'pids'/('%s.json'%name)),cwd,
                             str(run_dir/'logs'/('%s.log'%name)),'--']+command,check=True)
    master_running=subprocess.run(['rosnode','list'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5).returncode==0
    if not master_running:
        start('roscore',config['scene'],['/usr/bin/python3','/opt/ros/noetic/bin/roscore'])
        for _ in range(30):
            if subprocess.run(['rosnode','list'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=3).returncode==0:break
            time.sleep(1)
        else:raise TimeoutError('owned ROS master startup failed')
    # Avoid inherited true values during simulator/case initialization.
    subprocess.run(['rosparam','set','/tcei_controller/execute','false'],check=True)
    subprocess.run(['rosparam','set','/tcei_nine/execute','false'],check=True)
    for name,cwd,command in spec:
        start(name,cwd,command)
        if name=='sim' and args.case:
            deadline=time.monotonic()+240
            while not (run_dir/'scene_initialized.json').exists():
                if (run_dir/'scene_setup_error.json').exists():raise RuntimeError((run_dir/'scene_setup_error.json').read_text())
                if time.monotonic()>deadline:raise TimeoutError('initial physical layout did not settle')
                subprocess.run(guard+['check',str(run_dir/'pids/sim.json')],check=True)
                time.sleep(.2)
            if not json.loads((run_dir/'scene_initialized.json').read_text()).get('valid'):raise ValueError('physical layout invalid')
    subprocess.run(['/usr/bin/python3',str(HERE/'probe_no_action.py'),'--timeout','180',
                    '--output',str(run_dir/'read_only_ready.json')],check=True)
    (run_dir/'warmup_complete.txt').write_text('No trajectory or observation request sent. Execution remains disabled.\n')
    print(json.dumps({'run_dir':str(run_dir),'mode':'no_action_warmup','detector':detector,'execution_enabled':False}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr)
        print('No automatic task was started. Preserve logs/PID records; do not reuse this run name.',file=sys.stderr)
        raise SystemExit(1)
