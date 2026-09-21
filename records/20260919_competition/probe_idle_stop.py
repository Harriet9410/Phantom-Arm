#!/usr/bin/python3
"""Native Noetic idle stop/reset probe. Never publishes motion, gripper or cancel.

Run on the ROS host: /usr/bin/python3 probe_idle_stop.py --outdir NEW_DIRECTORY
Exit 0 means idle API evidence only; dynamic H01/H02 and physical safety remain untested.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import signal
import threading
import time
import uuid

J, P, C, E, K = '/Jaka/get_jointstate', '/Jaka/get_end_effector_pose', '/Jaka/gripper_is_captured', '/Jaka/get_gripper_efforts', '/clock'
D, A, S, R = '/tcei/joint_diagnostics', '/tcei/stop_ack', '/tcei/stop_request', '/tcei/stop_reset'
INTENTS = {'/Jaka/set_end_effector_pose': 'geometry_msgs/PoseStamped', '/Jaka/set_gripper_value': 'std_msgs/Float32',
           '/tcei/execute_preview': 'std_msgs/String', '/tcei/prepare_observation': 'std_msgs/String',
           '/tcei/plan': 'std_msgs/String', '/tcei/cancel': 'std_msgs/Bool', '/tcei/request': 'std_msgs/String'}
TOPICS = dict(INTENTS, **{J: 'sensor_msgs/JointState', P: 'geometry_msgs/PoseStamped', C: 'std_msgs/Bool',
         E: 'std_msgs/Float32MultiArray', K: 'rosgraph_msgs/Clock', D: 'std_msgs/String', A: 'std_msgs/String',
         S: 'std_msgs/String', R: 'std_msgs/String', '/tcei/planner_status': 'std_msgs/String', '/tcei/task_status': 'std_msgs/String'})
FLAGS = ('/tcei_controller/execute', '/tcei_nine/execute')


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, '__slots__'): return {k: plain(getattr(value, k)) for k in value.__slots__}
    if isinstance(value, float) and not math.isfinite(value): return {'nonfinite_float': repr(value)}
    return value


def seconds(stamp): return stamp['secs'] + stamp['nsecs'] / 1.e9
def vector(values, size):
    if not isinstance(values, list) or len(values) != size or any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
        raise ValueError('invalid finite vector')
    return values


def stable(rows, since, duration, now, names, anchor=None):
    """Independent native observations; receipt clock overlap is not exact sensor/sim pairing."""
    streams = {topic: [r for r in rows if r['kind'] == 'receive' and r['topic'] == topic and r['monotonic'] >= since] for topic in (J, P, C, E, K)}
    for topic in (J, P, K):
        values = streams[topic]; stamps = [seconds(r['value']['clock'] if topic == K else r['value']['header']['stamp']) for r in values]
        if any(b < a for a, b in zip(stamps, stamps[1:])): raise ValueError('source clock moved backwards')
        streams[topic] = [r for i, r in enumerate(values) if i == 0 or stamps[i] > stamps[i-1]]
    if any(len(v) < 3 or now - v[-1]['monotonic'] > 1.5 for v in streams.values()): return None
    if any(max(b['monotonic'] - a['monotonic'] for a, b in zip(v, v[1:])) > 1.5 for v in streams.values()):
        raise ValueError('native feedback gap exceeds 1.5 wall seconds')
    low = max(v[0]['monotonic'] for v in streams.values()); high = min(v[-1]['monotonic'] for v in streams.values())
    clocks = [seconds(r['value']['clock']) for r in streams[K] if low <= r['monotonic'] <= high]
    if len(clocks) < 3 or clocks[-1] - clocks[0] < duration: return None
    joints = [r['value'] for r in streams[J]]; poses = [r['value']['pose'] for r in streams[P]]
    if any(q['name'] != names for q in joints): raise ValueError('native joint ordering differs from diagnostic')
    qs = [vector(q['position'], 8) for q in joints]; vs = [vector(q['velocity'], 8) for q in joints]
    ps = [vector([p['position'][x] for x in ('x', 'y', 'z')], 3) for p in poses]
    rots = [vector([p['orientation'][x] for x in ('x', 'y', 'z', 'w')], 4) for p in poses]
    if any(abs(sum(x*x for x in q) - 1.) > .002 for q in rots): raise ValueError('nonunit native quaternion')
    anchor = anchor or {'q': qs[0], 'p': ps[0], 'rotation': rots[0]}
    drift = [max(abs(q[i] - anchor['q'][i]) for q in qs) for i in range(8)]
    tcp = max(math.sqrt(sum((p[i] - anchor['p'][i])**2 for i in range(3))) for p in ps)
    angle = max(2*math.acos(min(1., abs(sum(a*b for a, b in zip(q, anchor['rotation']))))) for q in rots)
    speed = max(abs(x) for v in vs for x in v[:6])
    if speed > .03 or max(drift[:6]) > .0015 or max(drift[6:]) > .0002 or tcp > .0015 or angle > .01:
        raise ValueError('independent native feedback is not stationary')
    if any(r['value']['data'] is not False for r in streams[C]): raise ValueError('gripper captured or invalid capture feedback')
    if any(any(abs(x) >= .2 for x in vector(r['value']['data'], 2)) for r in streams[E]):
        raise ValueError('empty gripper not supported by fresh low contact efforts')
    return {'clock_span': clocks[-1] - clocks[0], 'clock_start': clocks[0], 'clock_end': clocks[-1],
            'native_counts': {k: len(v) for k, v in streams.items()}, 'max_arm_speed': speed,
            'joint_drift': drift, 'tcp_drift': tcp, 'rotation_drift_rad': angle, 'anchor': anchor,
            'clock_pairing': 'overlapping_receipt_interval_only_not_atomic_sensor_pairing'}


def probe(rospy, types, outdir):
    """Injectable ROS surface for offline negative tests; all real waits use monotonic wall time."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=False)
    log = (outdir / 'events.jsonl').open('x', encoding='utf-8', buffering=1)
    lock = threading.RLock(); rows = []; last = {}; fatal = []; subscriptions = []; pubs = {}
    sid = 'H01_idle_' + uuid.uuid4().hex
    result = {'status': 'failed', 'scope': 'idle_native_stop_reset_only', 'dynamic_stop_tested': False,
              'physical_safety_certified': False, 'stop_id': sid, 'stop_sent': False, 'reset_sent': False,
              'lock_state': 'not_requested', 'phase': 'setup', 'topic_types': TOPICS,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': {'preflight_wall': 10., 'stop_wall': 7., 'reset_ack_wall': 3., 'post_reset_wall': 10.,
                         'stable_sim': .25, 'post_reset_sim': 1., 'fresh_wall': 1.5,
                         'arm_speed': .03, 'arm_drift': .0015, 'finger_drift': .0002, 'tcp_drift': .0015,
                         'rotation_drift_rad': .01, 'each_contact_effort_below': .2}}
    def record(topic, value, kind='receive'):
        with lock:
            row = {'kind': kind, 'topic': topic, 'monotonic': time.monotonic(), 'wall': time.time(), 'value': value}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); rows.append(row)
            if len(rows) > 100000: raise ValueError('evidence row limit exceeded')
            if kind == 'receive': last[topic] = row
            return row
    def callback(topic, message):
        try:
            raw = plain(message); record(topic, raw, 'raw_ros')
            value = json.loads(raw['data']) if TOPICS[topic] == 'std_msgs/String' else raw
            with lock:
                if topic == K and K in last and seconds(value['clock']) < seconds(last[K]['value']['clock']): fatal.append('simulation clock moved backwards')
                record(topic, value)
                if topic in INTENTS: fatal.append('unexpected control intent: ' + topic)
                if topic == C and value['data'] is not False: fatal.append('gripper is not empty')
                if topic in (S, R) and value.get('id') != sid: fatal.append('foreign stop/reset request')
                if topic == A:
                    state = value.get('state'); active = state in ('requested', 'holding', 'stopped', 'fault')
                    if (active and value.get('id') != sid) or state in ('fault', 'request_rejected', 'reset_rejected'): fatal.append('stop protocol conflict/fault: ' + str(value))
                    if state == 'fault' and value.get('id') == sid: result['lock_state'] = 'fault_latched'
                    if state == 'reset' and value.get('id') == sid and not result['reset_sent']: fatal.append('unsolicited reset')
        except Exception as error:
            with lock: fatal.append('callback: ' + repr(error))
    def graph():
        code, _, state = rospy.get_master().getSystemState(); code2, _, pairs = rospy.get_master().getTopicTypes()
        if code != 1 or code2 != 1: raise ValueError('ROS master graph query failed')
        publishers, subscribers = dict(state[0]), dict(state[1]); topic_types = dict(pairs)
        result.setdefault('graph_snapshots', []).append(record('graph', {'publishers': publishers, 'subscribers': subscribers, 'types': topic_types}, 'metadata'))
        for topic in (J, P, C, E, K, D, A):
            if len(publishers.get(topic, [])) != 1 or topic_types.get(topic) != TOPICS[topic]: raise ValueError('missing/duplicate publisher or wrong type: ' + topic)
        sim = publishers[J][0]
        if any(publishers[t][0] != sim for t in (P, C, E, D, A)): raise ValueError('native feedback and stop ACK have different providers')
        if any(sim not in subscribers.get(t, []) or pubs[t].get_num_connections() < 2 for t in (S, R)):
            raise ValueError('stop/reset simulator and recorder subscriptions not connected')
    def checks():
        flags = {name: rospy.get_param(name, None) for name in FLAGS}
        if flags != result.get('execute_flags'): result['execute_flags'] = flags; record('execute_flags', flags, 'metadata')
        if any(v is not False for v in flags.values()): raise ValueError('both execute parameters must explicitly be false')
        if fatal: raise ValueError('; '.join(fatal))
    def wait(phase, timeout, predicate):
        result['phase'] = phase; until = time.monotonic() + timeout
        while time.monotonic() < until:
            if rospy.is_shutdown(): raise RuntimeError('ROS shutdown')
            checks()
            with lock:
                if fatal: raise ValueError('; '.join(fatal))
                answer = predicate()
                if answer: return answer
            time.sleep(.05)
        raise TimeoutError(phase + ' wall timeout')
    def proof(since, duration, diagnostic=False):
        if D not in last: return None
        diag = last[D]; names = diag['value'].get('all_joint_names', [])
        if len(names) != 8 or len(set(names)) != 8: raise ValueError('eight unique diagnostic joint names required')
        if diagnostic:
            if diag['monotonic'] < since or time.monotonic() - diag['monotonic'] > 1.5: return None
            if diag['value'].get('tracking') is not False: raise ValueError('trajectory tracking is active or unknown')
            if not 0 <= time.time() - diag['value']['time'] <= 1.5: return None
            if K not in last or abs(seconds(last[K]['value']['clock']) - diag['value']['simulation_time']) > .15: return None
        return stable(rows, since, duration, time.monotonic(), names, result.get('preflight', {}).get('anchor'))
    def send(topic, payload):
        checks()
        marker = 'stop_sent' if topic == S else 'reset_sent'; result[marker] = True
        result['lock_state'] = 'requested_unconfirmed' if topic == S else 'reset_requested_unconfirmed'
        result[marker + '_at'] = record(topic, payload, 'publish_once')['monotonic']
        pubs[topic].publish(types['std_msgs/String'](data=json.dumps(payload)))
    try:
        rospy.init_node('tcei_idle_stop_probe', anonymous=True, disable_signals=True)
        result['use_sim_time'] = rospy.get_param('/use_sim_time', None)
        record('configuration', result, 'metadata')
        subscriptions = [rospy.Subscriber(t, types[k], lambda m, t=t: callback(t, m), queue_size=1000) for t, k in TOPICS.items()]
        pubs = {t: rospy.Publisher(t, types['std_msgs/String'], queue_size=1, latch=False) for t in (S, R)}
        start = time.monotonic(); result['preflight'] = wait('preflight', 10., lambda: proof(start, .25, True)); graph()
        if not proof(start, .25, True): raise ValueError('preflight evidence expired during graph query')
        send(S, {'id': sid, 'reason': 'H01_idle_api_check'})
        def stopped():
            ack = last.get(A, {}); value = ack.get('value', {})
            if value.get('id') != sid or value.get('state') != 'stopped' or value.get('fault_reason') or ack.get('monotonic', 0) < result['stop_sent_at']: return None
            if time.monotonic() - ack['monotonic'] > 1.5: return None
            stamp, speed, elapsed = vector([value.get('simulation_time'), value.get('max_joint_speed'), value.get('seconds')], 3)
            if speed < 0 or speed > .03 or elapsed < 0 or abs(stamp - seconds(last[K]['value']['clock'])) > .15: return None
            independent = proof(result['stop_sent_at'], .25)
            return {'ack': ack, 'independent': independent} if independent else None
        result['stopped'] = wait('stop_confirmation', max(0., result['stop_sent_at'] + 7. - time.monotonic()), stopped); result['lock_state'] = 'measured_stopped'; graph()
        if not stopped(): raise ValueError('stop evidence expired during graph query; reset withheld')
        send(R, {'id': sid})
        result['reset_ack'] = wait('reset_ack', 3., lambda: last.get(A) if last.get(A, {}).get('value', {}).get('id') == sid and last[A]['value'].get('state') == 'reset' and last[A]['monotonic'] >= result['reset_sent_at'] else None)
        result['lock_state'] = 'reset_acknowledged'
        result['post_reset'] = wait('post_reset_observation', 10., lambda: proof(result['reset_ack']['monotonic'], 1., True))
        result['status'] = 'passed_idle_api'; result['phase'] = 'complete'
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for endpoint in subscriptions + list(pubs.values()):
            try: endpoint.unregister()
            except Exception as error: fatal.append('unregister: ' + repr(error))
        if fatal: result['status'] = 'failed'; result['callback_or_cleanup_errors'] = fatal
        result['finished_wall'] = time.time(); result['finished_monotonic'] = time.monotonic()
        with lock: result['recorded_rows'] = len(rows); log.close()
        (outdir / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--outdir', required=True); args = parser.parse_args()
    socket.setdefaulttimeout(1.5)  # Bound ROS master XML-RPC socket waits; phase waits never use rospy.sleep.
    import rospy
    from std_msgs.msg import String, Bool, Float32, Float32MultiArray
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Clock
    types = {t._type: t for t in (String, Bool, Float32, Float32MultiArray, JointState, PoseStamped, Clock)}
    def alarm(_signum, _frame): raise TimeoutError('40 second total wall guard, including ROS setup/RPC')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, 40.)
    try: result = probe(rospy, types, args.outdir)
    finally: signal.setitimer(signal.ITIMER_REAL, 0.)
    print(json.dumps({k: result.get(k) for k in ('status', 'phase', 'stop_id', 'stop_sent', 'reset_sent', 'lock_state', 'error', 'recorded_rows')}))
    return 0 if result['status'] == 'passed_idle_api' else 1


if __name__ == '__main__': raise SystemExit(main())
