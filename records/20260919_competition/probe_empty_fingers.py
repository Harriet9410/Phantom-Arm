#!/usr/bin/python3
"""D02 empty-finger ramp/projection engineering probe; native Noetic, 60 s budget.
EffortSensor measures joint effort, not object contact. No arm target is published.
Requires frozen probe_idle_stop.py and probe_dynamic_stop.py beside this file.
"""
import argparse, hashlib, json, math, signal, socket, subprocess, threading, time
from pathlib import Path
from probe_dynamic_stop import prerequisite
from probe_idle_stop import J, P, C, E, K, D, A, TOPICS, FLAGS, plain, seconds, vector, stable
G, M = '/Jaka/set_gripper_value', '/Jaka/set_end_effector_pose'
MONITOR = (J, P, C, E, K, D, A, G, M)


def probe(rospy, types, outdir, idle_summary, observation_summary, projection_script=None, projection_result=None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=False); log = (out/'events.jsonl').open('x', encoding='utf-8', buffering=1)
    start = time.monotonic(); end = start+60.; work_end = start+40.; last = {}; rows = []; fatal = []; subs = []; pub = None; child = None; childlog = None
    lock = threading.RLock(); captured_since = [None]; lag_since = [None]; breaks=[0]; baseline = [None]; command = {'target': 0., 'at': start, 'sim': -1., 'joint_stamp': -1.}
    result = {'status': 'failed', 'phase': 'setup', 'commands': [], 'recovery': 'not_needed_no_command', 'object_grasped': False,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sensor': 'EffortSensor joint effort, not contact sensor',
              'limits': {'total_wall': 60., 'work_wall': 40., 'step': .00025, 'step_wall_min': .04, 'lead_limit': .0005,
                         'persistent_lag_wall': .2, 'persistent_captured_wall': .2, 'fresh_wall': 1., 'settled_sim': .25, 'position_tolerance': .001, 'absolute_joint_effort_below': .2}}
    def record(topic, value, kind='receive', caller=None):
        with lock:
            row = {'topic': topic, 'kind': kind, 'value': value, 'monotonic': time.monotonic(), 'wall': time.time(), 'callerid': caller}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); rows.append(row)
            if kind == 'receive': last[topic] = row
            return row
    def callback(topic, msg):
        try:
            raw = plain(msg); caller = getattr(msg, '_connection_header', {}).get('callerid'); record(topic, raw, 'raw_ros', caller)
            value = json.loads(raw['data']) if TOPICS[topic] == 'std_msgs/String' else raw
            with lock:
                if topic == K and K in last and seconds(value['clock']) < seconds(last[K]['value']['clock']): fatal.append('simulation clock rollback')
                record(topic, value, caller=caller)
                if topic==C:
                    if value['data'] is False and captured_since[0] is not None and time.monotonic()-captured_since[0]>.2: fatal.append('captured interval exceeded 0.2 wall seconds')
                    captured_since[0] = (captured_since[0] or time.monotonic()) if value['data'] is True else None
                if (topic==C and value['data'] is not False) or (topic==E and max(abs(x) for x in vector(value['data'],2))>=.2) or (topic==D and max(abs(x-command['target']) for x in vector(value['all_joint_positions'],8)[6:])>.001): breaks[0]+=1
                if result['phase']=='projection' and ((topic==C and value['data'] is not False) or (topic==E and max(abs(x) for x in vector(value['data'],2))>=.2)): fatal.append('held state changed during projection')
                if topic==D and baseline[0] and (max(abs(a-b) for a,b in zip(value['all_joint_positions'][:6],baseline[0]['q']))>.0015 or math.dist(value['tcp_actual'],baseline[0]['p'])>.0015): fatal.append('observed arm drift')
                if topic == M or (topic == G and caller != rospy.get_name()): fatal.append('foreign arm/gripper command')
                if topic == A and value.get('state') in ('requested', 'holding', 'stopped', 'fault'): fatal.append('active stop prevents gripper test')
        except Exception as error: fatal.append('callback: '+repr(error))
    def sample(recover=False):
        now = time.monotonic()
        if now >= end-.1 or rospy.is_shutdown(): raise TimeoutError('total wall budget/ROS shutdown')
        flags = {k: rospy.get_param(k, None) for k in FLAGS}; result['execute_flags'] = flags
        if (fatal and not recover) or any(v is not False for v in flags.values()): raise ValueError('unsafe flags/intent: '+str(fatal))
        if last.get(A,{}).get('value',{}).get('state') in ('requested','holding','stopped','fault'): raise ValueError('active stop prevents gripper recovery')
        if any(t not in last or now-last[t]['monotonic'] > 1. for t in (J,P,C,E,K,D)): raise ValueError('missing/stale native feedback')
        d = last[D]['value']; names = d['all_joint_names']; q = vector(d['all_joint_positions'],8); v = vector(d['all_joint_velocities'],8)
        jq = last[J]['value']; vector(jq['position'],8); vector(jq['velocity'],8); efforts = vector(last[E]['value']['data'],2)
        if len(set(names)) != 8 or names != jq['name'] or d.get('tracking') is not False or abs(d['simulation_time']-seconds(last[K]['value']['clock'])) > .15: raise ValueError('invalid robot/clock association')
        p = vector(d['tcp_actual'],3)
        if baseline[0] and (max(abs(a-b) for a,b in zip(q[:6],baseline[0]['q'])) > .0015 or math.dist(p,baseline[0]['p']) > .0015): raise ValueError('arm drift')
        if max(abs(x) for x in v[:6]) > .03: raise ValueError('arm not stationary')
        if not recover and captured_since[0] is not None and now-captured_since[0] > .2: raise ValueError('captured persists beyond 0.2 wall seconds')
        lead = max(abs(command['target']-x) for x in q[6:]); lag_since[0] = (lag_since[0] or now) if lead > .0005+1.e-9 else None
        if not recover and lag_since[0] is not None and now-lag_since[0] > .2: raise ValueError('finger command lag persists')
        return d, q[6:], efforts
    def send(value, reason):
        d = last.get(D, {}).get('value', {}); positions=d.get('all_joint_positions')
        row = record(G, {'target': value, 'reason': reason, 'actual_fingers': positions[6:] if isinstance(positions,list) else None, 'simulation_time': d.get('simulation_time')}, 'publish')
        try: stamp=seconds(last[J]['value']['header']['stamp'])
        except (KeyError,TypeError): stamp=-1.
        command.update(target=value, at=row['monotonic'], sim=d.get('simulation_time', -1.), joint_stamp=stamp)
        result['commands'].append(row); pub.publish(types['std_msgs/Float32'](data=value))
    def ramp(target, deadline, recover=False):
        result['phase'] = 'opening' if recover else 'closing'
        while abs(command['target']-target) > 1.e-9:
            if time.monotonic() >= deadline: raise TimeoutError('ramp wall timeout')
            with lock:
                d, actual, _ = sample(recover); nxt = command['target']+max(-.00025,min(.00025,target-command['target']))
                fresh = d['simulation_time'] > command['sim'] and seconds(last[J]['value']['header']['stamp']) > command['joint_stamp'] and last[J]['monotonic'] > command['at']
                if fresh and time.monotonic()-command['at'] >= .04 and max(abs(nxt-x) for x in actual) <= .0005+1.e-9: send(nxt,'ramp_open' if recover else 'ramp_close')
            time.sleep(.01)
    def settled(target, deadline, recover=False):
        since = None; initial = None; previous = None; count = 0; epoch=breaks[0]
        while time.monotonic() < deadline:
            with lock:
                d, actual, efforts = sample(recover); stamp = d['simulation_time']
                if epoch!=breaks[0]: since=None;initial=None;count=0;epoch=breaks[0]
                good = last[C]['value']['data'] is False and max(abs(x) for x in efforts) < .2 and max(abs(x-target) for x in actual) <= .001
                if not good: since = None; initial = None; count = 0
                elif stamp != previous:
                    if since is None: since = stamp; initial = list(actual)
                    if max(abs(a-b) for a,b in zip(actual,initial)) > .0002: since = stamp; initial = list(actual); count = 0
                    count += 1
                    if stamp-since >= .25 and count >= 3: return {'simulation_span':stamp-since,'fresh_frames':count,'fingers':actual,'joint_efforts':efforts,'captured':False,'verified_monotonic':time.monotonic()}
                previous = stamp
            time.sleep(.01)
        raise TimeoutError('settled empty fingers not confirmed')
    def stop_child():
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=.5)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=.5)
    try:
        a02, raw, _ = prerequisite(idle_summary); (out/'idle_prerequisite.json').write_bytes(raw)
        result['idle_prerequisite_sha256']=hashlib.sha256(raw).hexdigest()
        raw = Path(observation_summary).read_bytes(); c01 = json.loads(raw); (out/'observation_prerequisite.json').write_bytes(raw)
        result['observation_prerequisite_sha256']=hashlib.sha256(raw).hexdigest()
        if (c01.get('status')!='observation_completed' or c01.get('engineering_test') is not True or c01.get('competition_round') is not False or c01.get('grasp_requested') is not False or c01.get('request_sent_once') is not True
                or c01.get('execute_flags')!={k:False for k in FLAGS} or c01['terminal']['value'].get('status')!='observation_completed' or c01['terminal']['value'].get('request_id')!=c01['request']['request_id']): raise ValueError('C01 preparation prerequisite invalid')
        if bool(projection_script)!=bool(projection_result): raise ValueError('projection script and result must be supplied together')
        if projection_script and (not Path(projection_script).is_file() or Path(projection_result).exists()): raise ValueError('projection script missing or old result exists')
        rospy.init_node('tcei_empty_fingers_probe', anonymous=True, disable_signals=True)
        subs = [rospy.Subscriber(t,types[TOPICS[t]],lambda m,t=t:callback(t,m),queue_size=1000) for t in MONITOR]
        pub = rospy.Publisher(G,types['std_msgs/Float32'],queue_size=1,latch=False)
        while time.monotonic() < start+8. and any(t not in last for t in (J,P,C,E,K,D)): time.sleep(.02)
        with lock:
            d, actual, _ = sample(); baseline[0] = {'q':list(d['all_joint_positions'][:6]),'p':list(d['tcp_actual'])}
            if not 2.70 <= d['tcp_actual'][2] <= 2.945 or math.dist(d['tcp_actual'],c01['final_robot']['tcp_actual']) > .0015 or max(abs(x) for x in actual) > .001: raise ValueError('not at open empty C01 high observation pose')
            if d['all_joint_names']!=c01['final_robot']['all_joint_names'] or max(abs(a-b) for a,b in zip(d['all_joint_positions'][:6],vector(c01['final_robot']['all_joint_positions'],8)[:6]))>.0015: raise ValueError('C01 arm branch changed')
        result['preflight'] = settled(0.,min(start+10.,work_end))
        code,_,state = rospy.get_master().getSystemState(); pubs = dict(state[0]); subscribers = dict(state[1]); sim = a02['graph_snapshots'][-1]['value']['publishers'][J][0]
        record('graph',state,'metadata')
        if code!=1 or any(pubs.get(t)!=[sim] for t in (J,P,C,E,D,A)) or sim not in subscribers.get(G,[]) or pub.get_num_connections()<2: raise ValueError('native provider/command transport differs from A02')
        send(0.,'initial_open'); ramp(.02,min(time.monotonic()+15.,work_end)); result['half_closed'] = settled(.02,min(time.monotonic()+5.,work_end))
        if projection_script:
            result['phase']='projection'
            result['projection_script_sha256'] = hashlib.sha256(Path(projection_script).read_bytes()).hexdigest(); childlog = (out/'projection.log').open('x'); began = time.time()
            child = subprocess.Popen(['/usr/bin/python3',str(Path(projection_script).resolve())],stdout=childlog,stderr=subprocess.STDOUT,start_new_session=True); result['projection_pid'] = child.pid; until = min(time.monotonic()+20.,work_end)
            while child.poll() is None:
                d,actual,efforts = sample()
                if time.monotonic()>=until or last[C]['value']['data'] is not False or max(abs(x) for x in efforts)>=.2 or max(abs(x-.02) for x in actual)>.001: raise ValueError('projection timed out or held finger state changed')
                time.sleep(.02)
            d,actual,efforts=sample()
            if last[C]['value']['data'] is not False or max(abs(x) for x in efforts)>=.2 or max(abs(x-.02) for x in actual)>.001: raise ValueError('held state invalid at projection completion')
            result['projection_returncode']=child.returncode; path = Path(projection_result); result['projection'] = json.loads(path.read_text(encoding='utf-8'))
            if child.returncode!=0 or path.stat().st_mtime<began or result['projection'].get('status')!='debug_projection_created': raise ValueError('projection capture failed/stale')
        result['work_completed'] = True
    except BaseException as error: result['error'] = type(error).__name__+': '+str(error)
    finally:
        try: stop_child()
        except Exception as error: result['child_stop_error'] = repr(error)
        if result['commands']:
            try: ramp(0.,min(time.monotonic()+15.,end-6.),True); result['open_confirmation'] = settled(0.,min(time.monotonic()+5.,end-1.),True); result['recovery']='verified_open'
            except BaseException as error:
                result['recovery']='unknown'; result['recovery_error']=repr(error)
                try: send(0.,'single_fallback_zero_recovery_unknown'); result['fallback_zero_sent']=True
                except BaseException as final_error: result['fallback_error']=repr(final_error)
        for endpoint in subs+([pub] if pub else []):
            try: endpoint.unregister()
            except Exception as error: fatal.append(repr(error))
        if result.get('work_completed') and result['recovery']=='verified_open' and not fatal and 'child_stop_error' not in result: result['status']='passed_empty_finger_engineering'
        if childlog: childlog.close()
        result.update(finished_wall=time.time(),elapsed_wall=time.monotonic()-start,callback_errors=fatal,recorded_rows=len(rows)); log.close()
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('outdir','idle-summary','observation-summary'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--projection-script'); parser.add_argument('--projection-result'); args=parser.parse_args(); socket.setdefaulttimeout(.5)
    import rospy
    from std_msgs.msg import String,Bool,Float32,Float32MultiArray
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Clock
    types={t._type:t for t in (String,Bool,Float32,Float32MultiArray,JointState,PoseStamped,Clock)}
    def alarm(_s,_f): raise TimeoutError('60 second total wall guard')
    signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,60.)
    try: result=probe(rospy,types,args.outdir,args.idle_summary,args.observation_summary,args.projection_script,args.projection_result)
    finally: signal.setitimer(signal.ITIMER_REAL,0.)
    print(json.dumps({k:result.get(k) for k in ('status','phase','recovery','error','recovery_error','elapsed_wall','recorded_rows')})); return 0 if result['status']=='passed_empty_finger_engineering' else 1


if __name__=='__main__': raise SystemExit(main())
