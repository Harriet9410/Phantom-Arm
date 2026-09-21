"""Fresh frozen18 scene for explicitly labelled loaded-stop fault testing."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,uuid
import rospy
from std_msgs.msg import String,Bool
from sensor_msgs.msg import JointState

ROOT=Path('/root/tcei_competition_20260919');REL=ROOT/'releases/dev_v7_t1_18'
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
OLD=RUNS/'official18_stack01';STACK=RUNS/'H01_loaded18_stack01';RUN=RUNS/'H01_loaded18_A01'
sys.path.insert(0,str(REL/'harness'))
from process_guard import current,read_record

def main():
    assert not STACK.exists() and not RUN.exists()
    finished=json.loads((RUNS/'official18_round01/supervisor_finished.json').read_text())
    assert finished['episode_status']=='succeeded' and finished['execution_disabled'] and finished['evidence_complete']
    assert not current(read_record(RUNS/'official18_round01/pids/episode.json'))
    manifest=json.loads((REL/'tcei_stack/BUILD_MANIFEST.json').read_text())
    assert manifest['version']=='dev_v7_t1_18'
    assert all(hashlib.sha256((REL/'tcei_stack'/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
    rospy.init_node('tcei_h01_loaded18_prepare',anonymous=True)
    assert all(rospy.get_param(k,None) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
    assert rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data is False
    joints=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert len(joints.position)==8 and max(abs(v) for v in joints.velocity[:6])<.03
    assert max(abs(v) for v in joints.position[-2:])<.001
    acks=[];sub=rospy.Subscriber('/tcei/stop_ack',String,lambda m:acks.append((time.monotonic(),json.loads(m.data))),queue_size=20)
    pub=rospy.Publisher('/tcei/stop_request',String,queue_size=1,latch=False)
    until=time.monotonic()+3.
    while pub.get_num_connections()<1:
        assert time.monotonic()<until,'stop receiver unavailable';time.sleep(.02)
    sid='H01_prepare_'+uuid.uuid4().hex;sent=time.monotonic()
    pub.publish(String(json.dumps({'id':sid,'reason':'archive successful18 then cold start isolated loaded-stop test'})))
    proof=None
    while time.monotonic()-sent<7.:
        matching=[a for seen,a in acks if seen>=sent and a.get('id')==sid]
        if matching and matching[-1].get('state')=='fault':raise RuntimeError(str(matching[-1]))
        if matching and matching[-1].get('state')=='stopped':proof=matching[-1];break
        time.sleep(.03)
    assert proof is not None,'prior idle stop not confirmed; no restart'
    measurements=[]
    for _ in range(4):
        j=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
        assert max(abs(v) for v in j.velocity[:6])<.03
        measurements.append({'stamp':j.header.stamp.to_sec(),'position':list(j.position),'velocity':list(j.velocity)})
    assert measurements[-1]['stamp']>measurements[0]['stamp']
    (ROOT/'h01_loaded18_pre_shutdown.json').open('x').write(json.dumps({'stop':proof,'joints':measurements},indent=2))
    guard=['/usr/bin/python3',str(REL/'harness/process_guard.py')];stops=[]
    for name in ('nine','controller','perception','sim'):
        p=OLD/'pids'/(name+'.json');res=subprocess.run(guard+['stop',str(p),'--timeout','20'],capture_output=True,text=True)
        stops.append({'name':name,'code':res.returncode,'stdout':res.stdout,'stderr':res.stderr})
        (ROOT/'h01_loaded18_stopped_prior.json').write_text(json.dumps(stops,indent=2))
        assert res.returncode==0 and not current(read_record(p))
    sub.unregister();pub.unregister();rospy.set_param('/tcei_controller/observation_ready',False)
    rospy.signal_shutdown('prior18 preserved; start fresh test scene')
    env=os.environ.copy();env.update(DISPLAY=':20',TCEI_V7_CODE=str(REL/'tcei_stack'),
        TCEI_ROBOT_CALIBRATION=str(REL/'calibration/robot_projection.static_checked_v1.json'))
    result=subprocess.run(['bash',str(REL/'harness/start_stack.sh'),STACK.name],env=env)
    assert result.returncode==0,'frozen18 warmup failed'
    RUN.mkdir();(RUN/'pids').mkdir();(RUN/'logs').mkdir()
    commands={'rgbd':['/usr/bin/python3','-u',str(ROOT/'record_trial_rgbd12.py'),'_output:='+str(RUN/'rgbd'),'_duration:=600'],
        'scalars':['/usr/bin/python3','-u',str(ROOT/'record_scalar_evidence12.py'),'--output',str(RUN/'scalars'),'--duration','600']}
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
        label='H01 带载停止专项：试抬后主动取消；不计入正常搬运回合')
    config.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    (ROOT/'h01_loaded18_prepared.json').open('x').write(json.dumps({'prepared_at':time.time(),
        'stack':str(STACK),'run':str(RUN),'recorders':{n:read_record(RUN/'pids'/(n+'.json')) for n in commands},
        'robot_episode_started':False,'fault_injection':'one declared cancellation during measured loaded lift'},indent=2))
    print('H01_LOADED18_PREPARED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        (ROOT/'h01_loaded18_prepare_failed.json').open('x').write(json.dumps({'error':repr(error),'time':time.time()}))
        raise
