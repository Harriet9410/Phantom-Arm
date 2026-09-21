"""Install frozen18 beside16, archive failed stop without reset, start owned stack."""
from pathlib import Path
import hashlib,json,lzma,os,shutil,subprocess,sys,time
import rospy
from std_msgs.msg import String,Bool,Float32MultiArray
from sensor_msgs.msg import JointState

ROOT=Path('/root/tcei_competition_20260919')
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
OLD=ROOT/'releases/dev_v7_t1_18';NEW=ROOT/'releases/dev_v7_t1_19'
archive=ROOT/'runtime19.json.xz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='36b7dd07a987e5e234d9773870031acf1b6e87d6d6328f4c422eb6d6ea06a1b3'
bundle=json.loads(lzma.decompress(archive.read_bytes()))
assert bundle['label']=='dev_v7_t1_19' and not NEW.exists()
assert not (RUNS/'H01_source_contact19_stack01').exists()
manifest=json.loads(bundle['files']['BUILD_MANIFEST.json'])
assert hashlib.sha256(bundle['files']['BUILD_MANIFEST.json'].encode()).hexdigest()=='a0cd92e8087833d3531c0be344865bae30d3f467f2c7dfcc42f57cf9313970ff'
assert set(bundle['files'])==set(manifest['files'])|{'BUILD_MANIFEST.json'}
for name,source in bundle['files'].items():
    assert Path(name).name==name
    if name!='BUILD_MANIFEST.json':assert hashlib.sha256(source.encode()).hexdigest()==manifest['files'][name]
old_manifest=json.loads((OLD/'tcei_stack/BUILD_MANIFEST.json').read_text())
assert all(hashlib.sha256((OLD/'tcei_stack'/n).read_bytes()).hexdigest()==h for n,h in old_manifest['files'].items())
assert {n for n,h in manifest['files'].items() if old_manifest['files'].get(n)!=h}=={'controlled_stop.py','sim_with_feedback.py'}
shutil.copytree(OLD,NEW,ignore=shutil.ignore_patterns('__pycache__','PATCH_INSTALL_VERIFIED.json'))
for name,source in bundle['files'].items():(NEW/'tcei_stack'/name).write_bytes(source.encode())
(NEW/'PATCH_INSTALL_VERIFIED.json').open('x').write(json.dumps({'version':manifest['version'],
    'verified_at':time.time(),'manifest_sha256':hashlib.sha256((NEW/'tcei_stack/BUILD_MANIFEST.json').read_bytes()).hexdigest(),
    'files':len(manifest['files']),'source':'verified runtime19.json.xz','t0_passed':652},indent=2))
assert json.loads((RUNS/'H01_source_contact18_A01/supervisor_finished.json').read_text())['execution_disabled'] is True
rospy.init_node('tcei_prepare_full19',anonymous=True)
assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
ack=json.loads(rospy.wait_for_message('/tcei/stop_ack',String,timeout=5).data)
assert ack['state']=='fault' and ack['id']=='797e63490b1d45da963f29007df36850' and 0<=time.time()-ack['time']<1.5
assert ack['fault_reason']=='stop confirmation timed out'
summary=json.loads((RUNS/'H01_source_contact18_A01/episode/summary.json').read_text())
assert summary['status']=='failed' and summary['safe_stop']['state']=='fault'
samples=[]
for _ in range(8):
    joints=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert len(joints.position)==8 and len(joints.velocity)==8
    samples.append({'stamp':joints.header.stamp.to_sec(),'position':list(joints.position),
        'velocity':list(joints.velocity),'captured':rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data,
        'efforts':list(rospy.wait_for_message('/Jaka/get_gripper_efforts',Float32MultiArray,timeout=5).data)})
assert samples[-1]['stamp']>samples[0]['stamp']
assert max(max(abs(a-b) for a,b in zip(samples[0]['position'][:6],s['position'][:6])) for s in samples)<.001
(ROOT/'full19_pre_shutdown.json').open('x').write(json.dumps({'stop':ack,'samples':samples,
    'measured_stop_verified':False,'fault_reset_sent':False,
    'scope':'Failed native18 source-contact stop is preserved as failed. Retire owned simulator processes with episode and recorders closed; cold start a separate test, not a successful robot recovery.'},indent=2))
sys.path.insert(0,str(NEW/'harness'))
from process_guard import current,read_record
assert not current(read_record(RUNS/'H01_source_contact18_A01/pids/episode.json'))
guard=['/usr/bin/python3',str(NEW/'harness/process_guard.py')];stops=[]
for name in ('nine','controller','perception','sim'):
    record=RUNS/'H01_source_contact18_stack01/pids'/(name+'.json')
    result=subprocess.run(guard+['stop',str(record),'--timeout','20'],capture_output=True,text=True)
    stops.append({'name':name,'code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    (ROOT/'full19_stop_prior.json').write_text(json.dumps(stops,indent=2))
    assert result.returncode==0 and not current(read_record(record))
rospy.set_param('/tcei_controller/observation_ready',False)
rospy.signal_shutdown('owned18 source-contact stack stopped')
env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(NEW/'tcei_stack'),
    TCEI_ROBOT_CALIBRATION=str(NEW/'calibration/robot_projection.static_checked_v1.json'))
result=subprocess.run(['bash',str(NEW/'harness/start_stack.sh'),'H01_source_contact19_stack01'],env=env)
(ROOT/'full19_launcher_result.json').write_text(json.dumps({'returncode':result.returncode,'finished_at':time.time(),'run':'H01_source_contact19_stack01'}))
assert result.returncode==0
print('FULL18_WARMUP_COMPLETE',flush=True)
