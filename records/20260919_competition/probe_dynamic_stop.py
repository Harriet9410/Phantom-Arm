#!/usr/bin/python3
"""One +5 mm world-Z command, measured in-motion stop, bounded reset observation.

Native Noetic only. Requires the successful A02 JSON and its frozen idle probe beside
this file. Never publishes gripper commands, changes speed, pauses physics or sets q.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import socket
import threading
import time
import uuid
import probe_idle_stop as idle
from probe_idle_stop import J, P, C, E, K, D, A, S, R, TOPICS, INTENTS, FLAGS, plain, seconds, vector, stable

M, PLANNER = '/Jaka/set_end_effector_pose', '/tcei/planner_status'
IDLE_SHA = '4a804eca7d637eff9e829a22a39b047b7cac431d300e36db392c63a1e81d2b2a'


def pose_values(message):
    p = message['pose']; return vector([p['position'][k] for k in ('x', 'y', 'z')], 3), vector([p['orientation'][k] for k in ('x', 'y', 'z', 'w')], 4)


def prerequisite(path):
    data = Path(path).read_bytes(); report = json.loads(data)
    source_hash = hashlib.sha256(Path(idle.__file__).read_bytes()).hexdigest()
    if source_hash != IDLE_SHA or report.get('probe_sha256') != IDLE_SHA: raise ValueError('A02/frozen idle source hash mismatch')
    if (report.get('status') != 'passed_idle_api' or report.get('stop_sent') is not True or report.get('reset_sent') is not True
            or report.get('lock_state') != 'reset_acknowledged' or report.get('execute_flags') != {k: False for k in FLAGS}):
        raise ValueError('successful A02 with both execute=false required')
    sid = report['stop_id']; stopped = report['stopped']; reset = report['reset_ack']['value']
    if stopped['ack']['value']['id'] != sid or reset.get('id') != sid or reset.get('state') != 'reset': raise ValueError('A02 stop/reset identity mismatch')
    if stopped['independent']['clock_span'] < .25 or report['post_reset']['clock_span'] < 1.: raise ValueError('A02 independent evidence incomplete')
    return report, data, source_hash


def matches(value, target):
    return value.get('position') == target['position'] and value.get('quaternion_wire') == target['quaternion_wire']


def motion_proof(rows, last, sent, start, target, now):
    diag = last.get(D, {}); value = diag.get('value', {})
    if diag.get('monotonic', 0) < sent or now-diag.get('monotonic', 0) > .5 or value.get('tracking') is not True or not matches(value.get('goal') or {}, target): return None
    if K not in last or abs(value['simulation_time']-seconds(last[K]['value']['clock'])) > .15: return None
    poses = [r for r in rows if r['kind'] == 'receive' and r['topic'] == P and r['monotonic'] >= sent]
    unique = []; previous = None
    for row in poses:
        stamp = seconds(row['value']['header']['stamp'])
        if previous is not None and stamp < previous: raise ValueError('native pose stamp moved backwards')
        if stamp != previous: unique.append(row)
        previous = stamp
    if len(unique) < 2 or now-unique[-2]['monotonic'] > .5: return None
    a, b = unique[-2:]; pa, _ = pose_values(a['value']); pb, _ = pose_values(b['value'])
    for p in (pa, pb):
        if not .0005 <= p[2]-start[2] < .0045 or math.hypot(p[0]-start[0], p[1]-start[1]) > .0005: return None
    if pb[2]-pa[2] <= .00001: return None
    if math.dist(pb, target['position']) <= .0005: return None
    return {'tracking': diag, 'native_tcp_samples': [a, b], 'delta_z': pb[2]-start[2], 'remaining_to_goal': math.dist(pb, target['position'])}


def stop_metrics(rows, entry, until):
    ref = entry['reference']; since = entry['sent_at']; p0, _ = pose_values(ref[P]['value']); q0 = ref[J]['value']['position']
    selected = [r for r in rows if r['kind'] == 'receive' and since <= r['monotonic'] <= until]
    ps = [p0] + [pose_values(r['value'])[0] for r in selected if r['topic'] == P]
    qs = [q0] + [vector(r['value']['position'], 8) for r in selected if r['topic'] == J]
    ack = entry.get('proof', {}).get('first_stopped_ack'); origin_sim = seconds(ref[K]['value']['clock'])
    requested = next((r for r in selected if r['topic'] == A and r['value'].get('id') == entry['id'] and r['value'].get('state') == 'requested'), None)
    return {'reference': ref, 'reference_age_wall': {t: since-r['monotonic'] for t, r in ref.items()},
            'max_joint_excursion': [max(abs(q[i]-q0[i]) for q in qs) for i in range(8)],
            'max_tcp_excursion': max(math.dist(p, p0) for p in ps),
            'tcp_sampled_path_length': sum(math.dist(a, b) for a, b in zip(ps, ps[1:])),
            'request_ack_wall_latency': None if requested is None else requested['monotonic']-since,
            'stopped_ack_wall_latency': None if ack is None else ack['monotonic']-since,
            'independent_confirmation_wall_latency': None if 'proof' not in entry else entry['proof']['confirmed_at']-since,
            'observed_sim_delta_to_stop_ack': None if ack is None else ack['value']['simulation_time']-origin_sim,
            'clock_mapping': 'latest clock at receipt; native header retained; not an exact sensor/sim pairing',
            'max_receive_gap_wall': {t: max([b-a for a, b in zip(v, v[1:])] or [0.]) for t in (J, P, C, E, K)
                                     for v in [[r['monotonic'] for r in selected if r['topic'] == t]]}}


def probe(rospy, types, outdir, idle_summary):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=False)
    log = (outdir/'events.jsonl').open('x', encoding='utf-8', buffering=1)
    rows = []; last = {}; fatal = []; subs = []; pubs = {}; lock = threading.RLock(); own_ids = set()
    result = {'status': 'failed', 'scope': 'single_5mm_high_empty_arm_dynamic_stop', 'physical_safety_certified': False,
              'motion_sent': False, 'motion_echo_count': 0, 'dynamic_stop_tested': False, 'reset_sent': False, 'lock_state': 'not_requested', 'phase': 'setup',
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'limits': {'dz': .005, 'motion_wall': 2.,
              'stop_wall': 7., 'preflight_wall': 10., 'reset_ack_wall': 3., 'post_reset_wall': 15., 'post_reset_sim_minimum': 1.}}
    def record(topic, value, kind='receive', caller=None):
        with lock:
            row = {'kind': kind, 'topic': topic, 'value': value, 'monotonic': time.monotonic(), 'wall': time.time(), 'callerid': caller,
                   'last_received_clock': seconds(last[K]['value']['clock']) if K in last else None}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); rows.append(row)
            if len(rows) > 100000: raise ValueError('evidence row limit exceeded')
            if kind == 'receive': last[topic] = row
            return row
    def callback(topic, message):
        try:
            raw = plain(message); caller = getattr(message, '_connection_header', {}).get('callerid'); record(topic, raw, 'raw_ros', caller)
            value = json.loads(raw['data']) if TOPICS[topic] == 'std_msgs/String' else raw
            with lock:
                if topic == K and K in last and seconds(value['clock']) < seconds(last[K]['value']['clock']): fatal.append('simulation clock moved backwards')
                record(topic, value, caller=caller)
                if topic == M:
                    result['motion_echo_count'] += 1
                    if (not result['motion_sent'] or caller != rospy.get_name() or result['motion_echo_count'] != 1
                            or value['pose'] != result['motion_command']['pose']): fatal.append('unexpected/duplicate motion command')
                elif topic in INTENTS: fatal.append('unexpected control intent: '+topic)
                if topic == C and value['data'] is not False: fatal.append('gripper not empty')
                if topic == E and any(abs(x) >= .2 for x in vector(value['data'], 2)): fatal.append('unexpected contact effort')
                if topic in (S, R) and value.get('id') not in own_ids: fatal.append('foreign stop/reset request')
                if topic == A and value.get('state') in ('requested', 'holding', 'stopped', 'fault') and value.get('id') not in own_ids: fatal.append('foreign active stop')
                if topic == A and value.get('state') in ('request_rejected', 'reset_rejected'): fatal.append('stop protocol rejected: '+str(value))
                if topic == A and value.get('state') == 'fault': fatal.append('stop fault: '+str(value))
                if topic == A and value.get('state') == 'reset' and value.get('id') in own_ids and not result['reset_sent']: fatal.append('unsolicited reset')
        except Exception as error:
            with lock: fatal.append('callback: '+repr(error))
    def checks(strict=True):
        flags = {k: rospy.get_param(k, None) for k in FLAGS}
        if flags != result.get('execute_flags'): result['execute_flags'] = flags; record('execute_flags', flags, 'metadata')
        if any(v is not False for v in flags.values()) and 'execute switch changed' not in fatal: fatal.append('execute switch changed')
        if rospy.is_shutdown(): raise RuntimeError('ROS shutdown')
        if strict and fatal: raise ValueError('; '.join(fatal))
    def wait(phase, deadline, predicate, strict=True):
        result['phase'] = phase
        while time.monotonic() < deadline:
            checks(strict)
            with lock:
                answer = predicate()
                if answer: return answer
            time.sleep(.02)
        raise TimeoutError(phase+' wall timeout')
    def graph(report):
        code, _, state = rospy.get_master().getSystemState(); code2, _, pairs = rospy.get_master().getTopicTypes()
        if code != 1 or code2 != 1: raise ValueError('ROS graph query failed')
        publishers, subscribers, topic_types = dict(state[0]), dict(state[1]), dict(pairs)
        record('graph', {'publishers': publishers, 'subscribers': subscribers, 'types': topic_types}, 'metadata')
        for t in (J, P, C, E, K, D, A, PLANNER):
            if len(publishers.get(t, [])) != 1 or topic_types.get(t) != TOPICS[t]: raise ValueError('missing/duplicate/wrong-type publisher: '+t)
        sim = publishers[J][0]
        if publishers[J] != report['graph_snapshots'][-1]['value']['publishers'][J]: raise ValueError('A02 and current native simulator node differ')
        if any(publishers[t] != [sim] for t in (P, C, E, D, A, PLANNER)): raise ValueError('native/stop/planner providers differ')
        if any(sim not in subscribers.get(t, []) or pubs[t].get_num_connections() < 2 for t in (S, R, M)): raise ValueError('command or stop transport not ready')
    def proof(since, duration, diagnostic=False, anchor=None):
        if D not in last: return None
        diag = last[D]; names = diag['value'].get('all_joint_names', [])
        if len(names) != 8 or len(set(names)) != 8: raise ValueError('eight unique joint names required')
        if diagnostic:
            if diag['monotonic'] < since or time.monotonic()-diag['monotonic'] > 1.5: return None
            if diag['value'].get('tracking') is not False: raise ValueError('unexpected active trajectory')
            if not 0 <= time.time()-diag['value']['time'] <= 1.5 or K not in last or abs(seconds(last[K]['value']['clock'])-diag['value']['simulation_time']) > .15: return None
        return stable(rows, since, duration, time.monotonic(), names, anchor)
    def send_stop(key, reason):
        if key in result: return result[key]
        sid = 'H01_dynamic_'+uuid.uuid4().hex; own_ids.add(sid)
        with lock:
            entry = {'id': sid, 'reference': {t: copy.deepcopy(last[t]) for t in (J, P, K)}, 'reason': reason}
            result[key] = entry; entry['sent_at'] = record(S, {'id': sid, 'reason': reason}, 'publish_once')['monotonic']
            result['lock_state'] = 'requested_unconfirmed'; pubs[S].publish(types['std_msgs/String'](data=json.dumps({'id': sid, 'reason': reason})))
        return entry
    def confirm_stop(entry):
        first = [None]
        def measured():
            row = last.get(A, {}); value = row.get('value', {})
            if value.get('id') != entry['id'] or row.get('monotonic', 0) < entry['sent_at']: return None
            if value.get('state') == 'fault': result['lock_state'] = 'fault_latched'; raise ValueError('measured stop fault: '+str(value))
            if value.get('state') != 'stopped' or value.get('fault_reason') or time.monotonic()-row['monotonic'] > 1.5: return None
            stamp, speed = vector([value.get('simulation_time'), value.get('max_joint_speed')], 2)
            if not 0 <= speed <= .03 or abs(stamp-seconds(last[K]['value']['clock'])) > .15: return None
            if first[0] is None: first[0] = copy.deepcopy(row)
            # Deliberately begins AFTER measured stopped; braking travel is not stationary drift.
            independent = proof(first[0]['monotonic'], .25)
            return {'first_stopped_ack': first[0], 'last_ack': row, 'independent': independent, 'confirmed_at': time.monotonic()} if independent else None
        entry['proof'] = wait('stop_confirmation', entry['sent_at']+7., measured, strict=False)
        result['lock_state'] = 'measured_stopped'; return entry['proof']
    try:
        report, raw, idle_hash = prerequisite(idle_summary); (outdir/'idle_prerequisite.json').write_bytes(raw)
        result['idle_prerequisite'] = {'path': str(Path(idle_summary).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'idle_source_sha256': idle_hash, 'stop_id': report['stop_id']}
        rospy.init_node('tcei_dynamic_stop_probe', anonymous=True, disable_signals=True); result['use_sim_time'] = rospy.get_param('/use_sim_time', None)
        subs = [rospy.Subscriber(t, types[k], lambda m, t=t: callback(t, m), queue_size=1000) for t, k in TOPICS.items()]
        pubs = {t: rospy.Publisher(t, types[TOPICS[t]], queue_size=1, latch=False) for t in (S, R, M)}
        start = time.monotonic(); result['preflight'] = wait('preflight', start+10., lambda: proof(start, .25, True)); graph(report)
        checks()
        with lock:
            if not proof(start, .25, True): raise ValueError('preflight expired')
            actual = copy.deepcopy(last[P]); p, q = pose_values(actual['value'])
            if actual['value']['header']['frame_id'] != 'base_link' or not (-.9 <= p[0] <= .9 and -.6 <= p[1] <= .9 and 2.70 <= p[2] <= 2.945): raise ValueError('not in permitted high empty-arm region')
            goal = types['geometry_msgs/PoseStamped'](); goal.header.frame_id = 'base_link'; goal.header.stamp = rospy.Time.now()
            goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = p[0], p[1], p[2]+.005
            goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = q
            target = {'position': [p[0], p[1], p[2]+.005], 'quaternion_wire': q}; result['target'] = target
            result['motion_command'] = plain(goal); result['motion_start_native'] = actual; result['motion_sent'] = True
            sent = record(M, result['motion_command'], 'publish_once'); result['motion_sent_at'] = sent['monotonic']; pubs[M].publish(goal)
        def moving():
            planned = [r for r in rows if r['kind'] == 'receive' and r['topic'] == PLANNER and r['monotonic'] >= sent['monotonic']
                       and type(r['value'].get('time')) in (int, float) and r['value']['time'] >= sent['wall'] and matches(r['value'], target)]
            if any(r['value'].get('status') == 'rejected' for r in planned): return {'outcome': 'planner_rejected', 'planner': planned}
            latest_p, _ = pose_values(last[P]['value'])
            if math.dist(latest_p, target['position']) <= .0005: return {'outcome': 'endpoint_reached_before_stop'}
            if latest_p[2] < p[2]-.0005 or math.hypot(latest_p[0]-p[0], latest_p[1]-p[1]) > .002: raise ValueError('unexpected motion direction')
            verified = motion_proof(rows, last, sent['monotonic'], p, target, time.monotonic())
            good = [r for r in planned if r['value'].get('status') == 'planned']
            return {'outcome': 'moving_before_endpoint', 'evidence': verified, 'planner': good[-1]} if verified and good else None
        try: result['motion'] = wait('motion_observation', sent['monotonic']+2., moving)
        except TimeoutError: result['motion'] = {'outcome': 'no_verified_motion_within_2_wall_seconds'}
        result['dynamic_stop_tested'] = result['motion']['outcome'] == 'moving_before_endpoint'
        entry = send_stop('stop', result['motion']['outcome']); confirm_stop(entry)
        checks()
        if result['motion']['outcome'] != 'moving_before_endpoint': raise ValueError('dynamic cancellation not established; retain measured stop')
        duration = result['motion']['planner']['value'].get('timing', {}).get('duration')
        duration = vector([duration], 1)[0]
        if not 0 < duration <= 5.: raise ValueError('planned duration unavailable or exceeds bounded post-reset observation; retain stop')
        result['post_reset_sim_required'] = max(1., duration+.5); result['remaining_duration_policy'] = 'full original duration upper bound plus 0.5 sim seconds'
        if result['motion_echo_count'] != 1: raise ValueError('own single motion publication not observed')
        checks(); result['reset_sent'] = True; result['reset_sent_at'] = record(R, {'id': entry['id']}, 'publish_once')['monotonic']
        result['lock_state'] = 'reset_requested_unconfirmed'; pubs[R].publish(types['std_msgs/String'](data=json.dumps({'id': entry['id']})))
        result['reset_ack'] = wait('reset_ack', result['reset_sent_at']+3., lambda: last.get(A) if last.get(A, {}).get('value', {}).get('id') == entry['id'] and last[A]['value'].get('state') == 'reset' and last[A]['monotonic'] >= result['reset_sent_at'] else None)
        result['lock_state'] = 'reset_acknowledged'
        def after_reset():
            since, anchor = result['reset_ack']['monotonic'], entry['proof']['independent']['anchor']
            # Early short-window check is only an abort watch; success still needs the FULL interval.
            if not proof(since, 0., True, anchor): return None
            return proof(since, result['post_reset_sim_required'], True, anchor)
        result['post_reset'] = wait('post_reset_observation', time.monotonic()+15., after_reset)
        result['status'] = 'passed_dynamic_short_move'; result['phase'] = 'complete'
    except BaseException as error:
        result['error'] = type(error).__name__+': '+str(error)
        # Any uncertain motion publication still gets ONE prepared cancellation. Never resend a stop ID.
        if result['motion_sent']:
            try:
                key = 'emergency_stop' if result['reset_sent'] else 'stop'
                entry = send_stop(key, 'probe_failure_hold');
                if 'proof' not in entry: confirm_stop(entry)
            except BaseException as stop_error: result['stop_error'] = type(stop_error).__name__+': '+str(stop_error)
    finally:
        for endpoint in subs+list(pubs.values()):
            try: endpoint.unregister()
            except Exception as error: fatal.append('unregister: '+repr(error))
        if fatal: result['status'] = 'failed'; result['callback_or_cleanup_errors'] = fatal
        with lock:
            for key in ('stop', 'emergency_stop'):
                if key in result:
                    try: result[key]['metrics'] = stop_metrics(rows, result[key], result[key].get('proof', {}).get('confirmed_at', time.monotonic()))
                    except Exception as error: result['status'] = 'failed'; result['metrics_error'] = repr(error)
            result['finished_wall'] = time.time(); result['finished_monotonic'] = time.monotonic(); result['recorded_rows'] = len(rows); log.close()
        (outdir/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--outdir', required=True); parser.add_argument('--idle-summary', required=True); args = parser.parse_args()
    socket.setdefaulttimeout(1.5)
    import rospy
    from std_msgs.msg import String, Bool, Float32, Float32MultiArray
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Clock
    types = {t._type: t for t in (String, Bool, Float32, Float32MultiArray, JointState, PoseStamped, Clock)}
    def alarm(_signum, _frame): raise TimeoutError('50 second wall guard including ROS setup/RPC')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, 50.)
    try: result = probe(rospy, types, args.outdir, args.idle_summary)
    finally: signal.setitimer(signal.ITIMER_REAL, 0.)
    print(json.dumps({k: result.get(k) for k in ('status', 'phase', 'motion_sent', 'motion_echo_count', 'reset_sent', 'lock_state', 'error', 'stop_error', 'recorded_rows')}))
    return 0 if result['status'] == 'passed_dynamic_short_move' else 1


if __name__ == '__main__': raise SystemExit(main())
