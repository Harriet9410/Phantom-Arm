"""Restart the owned failed no-action stack after its simulator exited."""
from pathlib import Path
import hashlib,json,subprocess,os,time,sys,rospy

ROOT=Path('/root/tcei_competition_20260919')
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
RELEASE=ROOT/'releases/dev_v7_t1_13'
sys.path.insert(0,str(RELEASE/'harness'))
from process_guard import current,read_record

prior=json.loads((ROOT/'official13_01_launcher_result.json').read_text())
assert prior['returncode']==1
old=RUNS/'official13_stack01'
assert not (old/'round_started.lock').exists()
assert not (RUNS/'official13_stack02').exists()
assert hashlib.sha256((RELEASE/'tcei_stack/BUILD_MANIFEST.json').read_bytes()).hexdigest()=='1ea4baff2a868b42856352b44b8024c63346a67e6f803f7ba18f645c11647ee2'
rospy.init_node('restart_official13_stack02',anonymous=True)
flags=('/tcei_controller/execute','/tcei_nine/execute')
assert all(rospy.get_param(key) is False for key in flags)
before={name:current(read_record(old/'pids'/(name+'.json'))) for name in ('sim','controller','perception','nine')}
assert before['sim'] is False,'original simulator is unexpectedly live; inspect before restart'
(ROOT/'official13_stack02_restart_from.json').open('x').write(json.dumps({
    'checked_at':time.time(),'prior_result':prior,'owned_live_before':before,
    'scope':'failed warmup, no episode began; user explicitly requested restart'},indent=2))
guard=['/usr/bin/python3',str(RELEASE/'harness/process_guard.py')];stops=[]
for name in ('nine','controller','perception','sim'):
    record=old/'pids'/(name+'.json')
    result=subprocess.run(guard+['stop',str(record),'--timeout','20'],capture_output=True,text=True)
    stops.append({'name':name,'code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    (ROOT/'official13_stack02_stop_prior.json').write_text(json.dumps(stops,indent=2))
    assert result.returncode==0
    assert not current(read_record(record))
rospy.set_param('/tcei_controller/observation_ready',False)
rospy.signal_shutdown('owned failed warmup services stopped')
env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(RELEASE/'tcei_stack'),
    TCEI_ROBOT_CALIBRATION=str(RELEASE/'calibration/robot_projection.static_checked_v1.json'))
result=subprocess.run(['bash',str(RELEASE/'harness/start_stack.sh'),'official13_stack02'],env=env)
(ROOT/'official13_stack02_launcher_result.json').write_text(json.dumps({
    'returncode':result.returncode,'finished_at':time.time(),'run':'official13_stack02'}))
assert result.returncode==0
print('OFFICIAL13_STACK02_WARMUP_COMPLETE',flush=True)
