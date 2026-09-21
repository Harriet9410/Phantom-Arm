"""Fresh frozen18 scene for explicitly labelled loaded-stop fault testing."""
from pathlib import Path
import hashlib,json,math,os,subprocess,sys,time,uuid
import rospy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import JointState

ROOT=Path('/root/tcei_competition_20260919');REL=ROOT/'releases/dev_v7_t1_20'
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
OLD=RUNS/'pending19_stack01';STACK=RUNS/'pending20_stack01';RUN=RUNS/'pending20_A01'
import lzma,shutil
OLDREL=ROOT/'releases/dev_v7_t1_19'
archive=ROOT/'runtime20.json.xz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='f4156ae4e3079470cb9cdc88060afe0236dae8ede7b3da5f77e920576a323bfd'
bundle=json.loads(lzma.decompress(archive.read_bytes()))
assert bundle['label']=='dev_v7_t1_20' and not REL.exists()
manifest=json.loads(bundle['files']['BUILD_MANIFEST.json'])
assert hashlib.sha256(bundle['files']['BUILD_MANIFEST.json'].encode()).hexdigest()=='4fc1eb03ae62862ce7e9bf857d3f1effee8d2f95170a66268243fa129b72de4f'
assert set(bundle['files'])==set(manifest['files'])|{'BUILD_MANIFEST.json'}
for name,content in bundle['files'].items():
    assert Path(name).name==name
    if name!='BUILD_MANIFEST.json':assert hashlib.sha256(content.encode()).hexdigest()==manifest['files'][name]
prior=json.loads((OLDREL/'tcei_stack/BUILD_MANIFEST.json').read_text())
assert all(hashlib.sha256((OLDREL/'tcei_stack'/n).read_bytes()).hexdigest()==h for n,h in prior['files'].items())
assert {n for n,h in manifest['files'].items() if prior['files'].get(n)!=h}=={'controller.py','perception.py','perception_tracking.py','nine_node.py'}
shutil.copytree(OLDREL,REL,ignore=shutil.ignore_patterns('__pycache__','PATCH_INSTALL_VERIFIED.json'))
for name,content in bundle['files'].items():(REL/'tcei_stack'/name).write_bytes(content.encode())
(REL/'PATCH_INSTALL_VERIFIED.json').open('x').write(json.dumps({'manifest_sha256':'4fc1eb03ae62862ce7e9bf857d3f1effee8d2f95170a66268243fa129b72de4f','files':len(manifest['files']),'t0_passed':664,'installed_at':time.time()},indent=2))

sys.path.insert(0,str(REL/'harness'))
from process_guard import current,read_record

def main():
    assert not STACK.exists() and not RUN.exists()
    finished=json.loads((RUNS/'pending19_A01/supervisor_finished.json').read_text())
    assert finished['episode_status']=='stopped' and finished['execution_disabled'] and finished['evidence_complete']
    assert not current(read_record(RUNS/'pending19_A01/pids/episode.json'))
    manifest=json.loads((REL/'tcei_stack/BUILD_MANIFEST.json').read_text())
    assert manifest['version']=='dev_v7_t1_20'
    assert all(hashlib.sha256((REL/'tcei_stack'/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
    rospy.init_node('tcei_pending20_prepare',anonymous=True)
    assert all(rospy.get_param(k,None) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
    assert rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data is False
    joints=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert len(joints.position)==8 and all(math.isfinite(v) for v in joints.velocity)
    assert max(abs(v) for v in joints.position[-2:])<.001
    previous=json.loads((RUNS/'pending19_A01/episode/summary.json').read_text())
    assert 'unknown foreground' in previous['reason']
    proof=json.loads(rospy.wait_for_message('/tcei/stop_ack',String,timeout=5).data)
    assert proof['id']=='c5a4b4f6046b4636a0b526621c0e4c89' and proof['state']=='stopped'
    assert proof.get('fault_reason') is None and 0<=time.time()-proof['time']<1.5
    assert proof['solver_type']=='TGS' and proof['velocity_source']=='consecutive_physics_step_position_difference' and proof['max_joint_speed']<=.03
    measurements=[]
    for _ in range(4):
        j=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
        assert len(j.position)==8 and all(math.isfinite(v) for v in j.velocity)
        measurements.append({'stamp':j.header.stamp.to_sec(),'position':list(j.position),'velocity':list(j.velocity)})
    assert measurements[-1]['stamp']>measurements[0]['stamp']
    (ROOT/'pending20_pre_shutdown.json').open('x').write(json.dumps({'stop':proof,'joints':measurements},indent=2))
    guard=['/usr/bin/python3',str(REL/'harness/process_guard.py')];stops=[]
    for name in ('nine','controller','perception','sim'):
        p=OLD/'pids'/(name+'.json');res=subprocess.run(guard+['stop',str(p),'--timeout','20'],capture_output=True,text=True)
        stops.append({'name':name,'code':res.returncode,'stdout':res.stdout,'stderr':res.stderr})
        (ROOT/'pending20_stopped_prior.json').write_text(json.dumps(stops,indent=2))
        assert res.returncode==0 and not current(read_record(p))
    rospy.set_param('/tcei_controller/observation_ready',False)
    rospy.signal_shutdown('prior loaded-stop evidence preserved; cold source-contact test scene')
    env=os.environ.copy();env.update(DISPLAY=':20',TCEI_CODE=str(REL/'tcei_stack'),TCEI_V7_CODE=str(REL/'tcei_stack'),
        TCEI_ROBOT_CALIBRATION=str(REL/'calibration/robot_projection.static_checked_v1.json'))
    result=subprocess.run(['/usr/bin/python3','-u',str(ROOT/'launch_pending20.py'),STACK.name],env=env)
    assert result.returncode==0,'frozen18 warmup failed'
    RUN.mkdir();(RUN/'pids').mkdir();(RUN/'logs').mkdir()
    commands={'rgbd':['/usr/bin/python3','-u',str(ROOT/'record_trial_rgbd12.py'),'_output:='+str(RUN/'rgbd'),'_duration:=1200'],
        'scalars':['/usr/bin/python3','-u',str(ROOT/'record_scalar_evidence12.py'),'--output',str(RUN/'scalars'),'--duration','1200']}
    for name,command in commands.items():
        subprocess.run(guard+['start',str(RUN/'pids'/(name+'.json')),str(ROOT),str(RUN/'logs'/(name+'.log')),'--']+command,check=True,env=env)
    until=time.monotonic()+30.
    while not ((RUN/'rgbd/reference.jpg').exists() and (RUN/'scalars/ready.json').exists()):
        assert time.monotonic()<until
        assert all(current(read_record(RUN/'pids'/(n+'.json'))) for n in commands)
        time.sleep(.1)
    config=ROOT/'review_ui_02/config.json';prior=config.read_text();value=json.loads(prior)
    (RUN/'previous_panel_config.json').open('x').write(prior)
    value.update(episode_dir=str(RUN/'episode'),stack_dir=str(STACK),
        label='放置待核验专项：仅屏蔽第一件传送带证据，原图保留；验证继续后续任务')
    config.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    (ROOT/'pending20_prepared.json').open('x').write(json.dumps({'prepared_at':time.time(),
        'stack':str(STACK),'run':str(RUN),'recorders':{n:read_record(RUN/'pids'/(n+'.json')) for n in commands},
        'robot_episode_started':False,'fault_injection':'withhold first release-bound belt observations from controller only; no physical fault and no image edits'},indent=2))
    print('H01_LOADED18_PREPARED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        (ROOT/'pending20_prepare_failed.json').open('x').write(json.dumps({'error':repr(error),'time':time.time()}))
        raise
