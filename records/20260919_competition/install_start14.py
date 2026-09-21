"""Install frozen14 beside13, archive measured stopped state, start owned stack."""
from pathlib import Path
import hashlib,json,lzma,os,shutil,subprocess,sys,time
import rospy
from std_msgs.msg import String,Bool,Float32MultiArray
from sensor_msgs.msg import JointState

ROOT=Path('/root/tcei_competition_20260919')
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
OLD=ROOT/'releases/dev_v7_t1_13';NEW=ROOT/'releases/dev_v7_t1_14'
archive=ROOT/'runtime14.json.xz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='bc19e2bf23f9ac591c039494d76343038a11bdbd1533df552438a53c857e8575'
bundle=json.loads(lzma.decompress(archive.read_bytes()))
assert bundle['label']=='dev_v7_t1_14' and not NEW.exists()
assert not (RUNS/'official14_stack01').exists()
manifest=json.loads(bundle['files']['BUILD_MANIFEST.json'])
assert hashlib.sha256(bundle['files']['BUILD_MANIFEST.json'].encode()).hexdigest()=='617c2ecc8df5393467eb376b47a938f9c8ad3a86363022505d99a66fb1d286ac'
assert set(bundle['files'])==set(manifest['files'])|{'BUILD_MANIFEST.json'}
for name,source in bundle['files'].items():
    assert Path(name).name==name
    if name!='BUILD_MANIFEST.json':assert hashlib.sha256(source.encode()).hexdigest()==manifest['files'][name]
old_manifest=json.loads((OLD/'tcei_stack/BUILD_MANIFEST.json').read_text())
assert all(hashlib.sha256((OLD/'tcei_stack'/n).read_bytes()).hexdigest()==h for n,h in old_manifest['files'].items())
assert {n for n,h in manifest['files'].items() if old_manifest['files'].get(n)!=h}=={'controller.py','sim_with_feedback.py'}
shutil.copytree(OLD,NEW,ignore=shutil.ignore_patterns('__pycache__','PATCH_INSTALL_VERIFIED.json'))
for name,source in bundle['files'].items():(NEW/'tcei_stack'/name).write_bytes(source.encode())
(NEW/'PATCH_INSTALL_VERIFIED.json').open('x').write(json.dumps({'version':manifest['version'],
    'verified_at':time.time(),'manifest_sha256':hashlib.sha256((NEW/'tcei_stack/BUILD_MANIFEST.json').read_bytes()).hexdigest(),
    'files':len(manifest['files']),'source':'verified runtime14.json.xz','t0_passed':606},indent=2))
assert json.loads((RUNS/'official13_round01/supervisor_finished.json').read_text())['execution_disabled'] is True
rospy.init_node('tcei_prepare_full14',anonymous=True)
assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
ack=json.loads(rospy.wait_for_message('/tcei/stop_ack',String,timeout=5).data)
assert ack['state']=='stopped' and ack['id']=='5cd14ab950064eb69c4b1bb7a5a657ad' and 0<=time.time()-ack['time']<1.5
samples=[]
for _ in range(4):
    joints=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert len(joints.position)==8 and max(abs(v) for v in joints.velocity[:6])<.03
    samples.append({'stamp':joints.header.stamp.to_sec(),'position':list(joints.position),
        'captured':rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data,
        'efforts':list(rospy.wait_for_message('/Jaka/get_gripper_efforts',Float32MultiArray,timeout=5).data)})
assert samples[-1]['stamp']>samples[0]['stamp']
assert max(abs(a-b) for a,b in zip(samples[0]['position'],samples[-1]['position']))<.001
(ROOT/'full14_pre_shutdown.json').open('x').write(json.dumps({'stop':ack,'samples':samples,
    'scope':'preserve failed official13 round; cold restart for same-layout diagnostic repetition, not a recovery success'},indent=2))
sys.path.insert(0,str(NEW/'harness'))
from process_guard import current,read_record
guard=['/usr/bin/python3',str(NEW/'harness/process_guard.py')];stops=[]
for name in ('nine','controller','perception','sim'):
    record=RUNS/'official13_stack02/pids'/(name+'.json')
    result=subprocess.run(guard+['stop',str(record),'--timeout','20'],capture_output=True,text=True)
    stops.append({'name':name,'code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    (ROOT/'full14_stop_prior.json').write_text(json.dumps(stops,indent=2))
    assert result.returncode==0 and not current(read_record(record))
rospy.set_param('/tcei_controller/observation_ready',False)
rospy.signal_shutdown('owned13 stack stopped')
env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(NEW/'tcei_stack'),
    TCEI_ROBOT_CALIBRATION=str(NEW/'calibration/robot_projection.static_checked_v1.json'))
result=subprocess.run(['bash',str(NEW/'harness/start_stack.sh'),'official14_stack01'],env=env)
(ROOT/'full14_launcher_result.json').write_text(json.dumps({'returncode':result.returncode,'finished_at':time.time(),'run':'official14_stack01'}))
assert result.returncode==0
print('FULL14_WARMUP_COMPLETE',flush=True)
