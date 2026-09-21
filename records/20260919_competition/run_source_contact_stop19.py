"""H01: one declared cancellation during a measured loaded lift; no reset/open."""
from pathlib import Path
import copy,hashlib,json,math,os,subprocess,sys,threading,time,uuid
import rospy
from std_msgs.msg import String,Bool,Float32,Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock

ROOT=Path('/root/tcei_competition_20260919');REL=ROOT/'releases/dev_v7_t1_19'
RUN=Path('/root/gpufree-data/tcei_competition_20260919/H01_source_contact19_A01')
sys.path[:0]=[str(REL/'tcei_stack'),str(REL/'harness'),str(ROOT)]
from process_guard import identity,current,read_record
from probe_idle_stop import plain
from ros_endpoints import CANCEL_PROTOCOL,BoundedXmlRpcTransport,bus_rows,missing_peers
from loaded_stop_checks import J,P,C,E,K,position
from full_step_stop_checks import M,held_frame_stationary
from loaded_stop_result_audit import validate_interrupted_closeout
import xmlrpc.client

T='/tcei/task_status';D='/tcei/joint_diagnostics';A='/tcei/stop_ack';ACK='/tcei/cancel_ack'
POSE='/Jaka/set_end_effector_pose';GRIP='/Jaka/set_gripper_value';CANCEL='/tcei/cancel_request'
TOPICS={J:JointState,P:PoseStamped,C:Bool,E:Float32MultiArray,K:Clock,
    T:String,D:String,A:String,ACK:String,POSE:PoseStamped,GRIP:Float32,
    '/tcei/execute_preview':String,CANCEL:String,M:String}

def main():
    case=RUN/'probe';case.mkdir(exist_ok=False);log=(case/'events.jsonl').open('x',buffering=1)
    lock=threading.RLock();rows=[];last={};errors=[];subs=[];child=None
    result={'case':'H01_source_contact19_A01','status':'failed','mode':'declared_fault_injection',
        'injection':'one cancellation after loaded preload and before any trial lift',
        'normal_round':False,'velocity_basis':'TGS consecutive physical frame displacement; original velocities preserved','reset_sent':False,'gripper_or_pose_commands_sent_by_probe':0,
        'cancel_sent':False,'limits':{'arm_speed':.03,'arm_drift':.0015,'tcp_drift':.0015,
            'stable_sim_seconds':.25,'stop_confirmation_wall_seconds':7.,'braking_tcp_distance_m':.02},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def record(topic,value):
        with lock:
            row={'topic':topic,'mono':time.monotonic(),'wall':time.time(),'value':value}
            log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');rows.append(row);last[topic]=row
            if len(rows)>100000:raise RuntimeError('probe evidence limit reached')
            return row
    def callback(topic,message):
        try:
            value=json.loads(message.data) if TOPICS[topic] is String else plain(message)
            record(topic,value)
        except Exception as error:errors.append(repr(error))
    def snapshot():
        with lock:return list(rows),copy.deepcopy(last)
    cancel_id=uuid.uuid4().hex;cancel_pub=None
    def send_cancel(reason,context=None):
        if result['cancel_sent']:return
        request={'protocol':CANCEL_PROTOCOL,'cancel_id':cancel_id,
            'round_id':(context or {}).get('round_id','H01_source_contact19_probe'),
            'request_id':(context or {}).get('request_id','H01_source_contact19_guard'),'reason':reason}
        with xmlrpc.client.ServerProxy(rospy.get_node_uri(),transport=BoundedXmlRpcTransport(.5)) as api:
            missing=missing_peers(bus_rows(api.getBusInfo(rospy.get_name())),CANCEL,require_response=False)
        if missing:raise RuntimeError('cancel transport unavailable: '+str(missing))
        result['cancel_sent']=True;result['cancel_at']=time.monotonic();result['cancel_request']=request
        cancel_pub.publish(String(json.dumps(request)))
    try:
        rospy.init_node('tcei_loaded18_stop_probe',anonymous=True)
        assert all(rospy.get_param(k,None) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
        assert all(current(read_record(RUN/'pids'/(n+'.json'))) for n in ('rgbd','scalars'))
        supervisor=ROOT/'H01_source_contact19_supervisor.py'
        assert hashlib.sha256(supervisor.read_bytes()).hexdigest()=='8684af99fe315bcc1ebd7d0bba3feeb9b2a1b94adca206a5bf8ad5331754179d'
        subs=[rospy.Subscriber(t,kind,lambda m,t=t:callback(t,m),queue_size=1000) for t,kind in TOPICS.items()]
        cancel_pub=rospy.Publisher(CANCEL,String,queue_size=1,latch=False)
        until=time.monotonic()+10.
        while not all(t in last for t in (J,P,C,E,K,D)):
            assert time.monotonic()<until,'native preflight feedback unavailable';time.sleep(.02)
        _,values=snapshot();assert values[C]['value']['data'] is False
        assert max(abs(v) for v in values[J]['value']['velocity'][:6])<.03
        code,_,graph=rospy.get_master().getSystemState();assert code==1
        publishers=dict(graph[0]);sim=publishers[J]
        assert len(sim)==1 and len(publishers[K])==1 and all(publishers[t]==sim for t in (P,C,E,D,A,M))
        result['native_publisher']=sim[0];result['started_at']=time.time()
        with (RUN/'logs/supervisor.log').open('xb') as out:
            child=subprocess.Popen(['/usr/bin/python3','-u',str(supervisor)],cwd=str(ROOT),env=os.environ.copy(),
                stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        owned=identity(child.pid);owned.update(started_at=time.time(),command=['/usr/bin/python3','-u',str(supervisor)],cwd=str(ROOT))
        (RUN/'pids/supervisor.json').open('x').write(json.dumps(owned,indent=2))
        phase=None;verified=None;origin=None;until=time.monotonic()+100.
        while time.monotonic()<until:
            if errors:raise RuntimeError(str(errors))
            stream,values=snapshot()
            if child.poll() is not None:raise RuntimeError('episode ended before cancellation trigger')
            for row in stream:
                e=row['value']
                if phase is None and row['topic']==T and e.get('status')=='grasp_preload':
                    phase=e
                    prior=[r for r in stream if r['topic']==P and r['mono']<=row['mono']]
                    if not prior:raise RuntimeError('no measured source-contact origin')
                    origin=position(prior[-1]['value']);result['phase']=phase;result['source_contact_origin']=origin
            if phase:
                if any(r['topic']==T and r['value'].get('status')=='trial_lift_started' for r in stream):
                    raise RuntimeError('missed source-contact test window; do not claim source test')
                d=values[D]['value']
                if values[D]['mono']>=phase['monotonic'] and time.monotonic()-values[D]['mono']<.3:
                    assert d['tracking'] is False and math.dist(d['tcp_actual'],origin)<.001
                    assert values[C]['value']['data'] is True
                    result['source_contact_before_cancel']=values[D];result['native_at_cancel']=values[P]
                    send_cancel('H01 declared loaded source-contact stop test',phase);break
            time.sleep(.01)
        if not result['cancel_sent']:raise TimeoutError('no verified loaded-motion cancellation trigger')
        first_stopped=None;until=result['cancel_at']+7.
        while time.monotonic()<until:
            stream,values=snapshot()
            mappings=[r for r in stream if r['topic']==ACK and r['mono']>=result['cancel_at']
                      and r['value'].get('cancel_id')==cancel_id and r['value'].get('state')=='accepted']
            if mappings:
                mapping=mappings[-1];result['cancel_mapping']=mapping
                acks=[r for r in stream if r['topic']==A and r['mono']>=result['cancel_at']
                      and r['value'].get('id')==mapping['value']['stop_id']]
                if acks:
                    ack=acks[-1];result['last_stop_ack']=ack
                    if ack['value'].get('state')=='fault':raise RuntimeError('native stop fault: '+str(ack['value']))
                    if ack['value'].get('state')=='stopped' and not ack['value'].get('fault_reason'):
                        if first_stopped is None:first_stopped=ack
                        proof=held_frame_stationary(stream,first_stopped['mono'],time.monotonic(),mapping['value']['stop_id'])
                        if proof:
                            result['independent_loaded_stop']=proof
                            result['first_stopped_ack']=first_stopped
                            result['stop_confirmation_seconds']=first_stopped['mono']-result['cancel_at']
                            break
            if errors:raise RuntimeError(str(errors))
            time.sleep(.02)
        assert result.get('independent_loaded_stop'),'loaded native stop not independently confirmed'
        result['supervisor_returncode']=child.wait(timeout=30.)
        assert (RUN/'supervisor_finished.json').exists()
        finished=json.loads((RUN/'supervisor_finished.json').read_text())
        assert finished['execution_disabled'] and finished['evidence_complete']
        validate_interrupted_closeout(finished,json.loads((RUN/'episode/summary.json').read_text()),result['cancel_mapping']['value']['stop_id'])
        result['closed']=finished;stream,values=snapshot()
        pc=position(result['native_at_cancel']['value'])
        poses=[position(r['value']) for r in stream if r['topic']==P and r['mono']>=result['cancel_at']]
        result['max_tcp_travel_after_cancel']=max(math.dist(pc,p) for p in poses)
        assert result['max_tcp_travel_after_cancel']<=.02,'braking excursion exceeded registered bound'
        # Allow the command already in transport at cancellation receipt; no
        # new pose/gripper target may follow the controller accepted mapping.
        accepted=result['cancel_mapping']['mono']
        late=[r for r in stream if r['topic'] in (POSE,GRIP,'/tcei/execute_preview') and r['mono']>accepted+.1]
        result['late_action_commands']=late;assert not late,'motion/gripper command after accepted cancellation'
        result['status']='passed_loaded_source_contact_stop'
    except BaseException as error:
        result['error']=repr(error)
        if child is not None and child.poll() is None:
            try:send_cancel('H01 probe failure guard',result.get('phase'))
            except BaseException as extra:result['cancel_error']=repr(extra)
            try:result['supervisor_returncode']=child.wait(timeout=35.)
            except subprocess.TimeoutExpired:result['supervisor_still_running']=True
    finally:
        if child is None:
            guard=['/usr/bin/python3',str(REL/'harness/process_guard.py')]
            result['preflight_recorder_cleanup']=[]
            for name in ('rgbd','scalars'):
                stopped=subprocess.run(guard+['stop',str(RUN/'pids'/(name+'.json')),'--timeout','20'],capture_output=True,text=True)
                result['preflight_recorder_cleanup'].append({'name':name,'code':stopped.returncode,
                    'stdout':stopped.stdout,'stderr':stopped.stderr})
        for sub in subs:sub.unregister()
        if cancel_pub:cancel_pub.unregister()
        result['finished_at']=time.time();result['callback_errors']=errors;result['rows']=len(rows)
        if errors:result['status']='failed'
        with lock:log.close()
        (case/'summary.json').open('x').write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        rospy.signal_shutdown('H01 loaded stop probe finished; stop latch intentionally not reset')
    print('H01_LOADED18_RESULT',result['status'],result.get('error'),flush=True)
    return 0 if result['status']=='passed_loaded_source_contact_stop' else 1

if __name__=='__main__':raise SystemExit(main())
