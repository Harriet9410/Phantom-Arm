"""Independent TGS stop audit. Never rewrites raw JointState velocity values."""
import math
from loaded_stop_checks import J,P,C,E,K,vector,position,stamp
M='/tcei/stop_measurement'

def held_frame_stationary(rows,since,now,stop_id,seconds=.25):
    streams={t:[r for r in rows if r['topic']==t and r['mono']>=since] for t in (J,P,C,E,K,M)}
    if any(len(v)<3 or now-v[-1]['mono']>1.5 for v in streams.values()):return None
    if any(any(b['mono']-a['mono']>1.5 for a,b in zip(v,v[1:])) for v in streams.values()):
        raise ValueError('feedback stream gap')
    frames=[];previous=None
    for r in streams[M]:
        f=r['value'];t=f.get('simulation_time');dt=f.get('physics_dt')
        if (f.get('id')!=stop_id or f.get('protocol')!='tcei.stop_measurement.v2' or
                f.get('valid') is not True or f.get('solver_type')!='TGS' or f.get('configuration_error') is not None):
            raise ValueError('unverified TGS measurement source')
        if type(t) not in (int,float) or not math.isfinite(t) or type(dt) not in (int,float) or not math.isfinite(dt) or not 0<dt<=.02:
            raise ValueError('invalid physics sample time')
        vector(f['positions'],8);vector(f['reported_velocities'],8);vector(f['tcp'],3)
        if len(f['joint_names'])!=8 or len(set(f['joint_names']))!=8:raise ValueError('invalid joint names')
        if previous:
            old=previous['value'];gap=t-old['simulation_time']
            if gap==0:
                if f!=old:raise ValueError('changed state at identical physics stamp')
                continue
            if gap<0 or abs(gap-dt)>max(1e-7,dt*.01) or dt!=old['physics_dt'] or f['joint_names']!=old['joint_names']:
                raise ValueError('missing, changed or reversed physics frame')
        frames.append(r);previous=r
    if len(frames)<3 or frames[-1]['value']['simulation_time']-frames[0]['value']['simulation_time']<seconds:return None
    q0=frames[0]['value']['positions'];p0=frames[0]['value']['tcp']
    speeds=[max(abs(b['value']['positions'][i]-a['value']['positions'][i]) for i in range(6))/
            (b['value']['simulation_time']-a['value']['simulation_time']) for a,b in zip(frames,frames[1:])]
    drift=[max(abs(r['value']['positions'][i]-q0[i]) for r in frames) for i in range(8)]
    tcp=max(math.dist(r['value']['tcp'],p0) for r in frames)
    if max(speeds)>.03 or max(drift[:6])>.0015 or max(drift[6:])>.0002 or tcp>.0015:return None
    # Cross-check the new atomic stream against the original native topics.
    for topic in (J,P,K):
        times=[stamp(r['value']['clock'] if topic==K else r['value']['header']['stamp']) for r in streams[topic]]
        if any(b<a for a,b in zip(times,times[1:])):raise ValueError('original feedback time rollback')
        if len(set(times))<3:return None
    for r in streams[J]:
        q=vector(r['value']['position'],8);vector(r['value']['velocity'],8)
        peer=min(frames,key=lambda f:abs(f['mono']-r['mono']))
        if abs(peer['mono']-r['mono'])>.05:return None
        if r['value']['name']!=peer['value']['joint_names'] or max(abs(a-b) for a,b in zip(q,peer['value']['positions']))>.0002:
            raise ValueError('original and atomic joint feedback disagree')
        if not all(.001<abs(v)<.038 for v in q[-2:]):return None
    for r in streams[P]:
        peer=min(frames,key=lambda f:abs(f['mono']-r['mono']))
        if abs(peer['mono']-r['mono'])>.05:return None
        if math.dist(position(r['value']),peer['value']['tcp'])>.0002:raise ValueError('original and atomic TCP disagree')
    if any(r['value'].get('data') is not True for r in streams[C]):return None
    if any(min(vector(r['value']['data'],2))<.2 for r in streams[E]):return None
    return {'method':'independently_recomputed_full_physics_step_velocity',
        'simulation_span':frames[-1]['value']['simulation_time']-frames[0]['value']['simulation_time'],
        'physics_dt':frames[0]['value']['physics_dt'],'solver_type':'TGS','physics_frames':len(frames),
        'max_full_step_arm_speed':max(speeds),
        'max_reported_arm_speed':max(abs(x) for r in streams[J] for x in r['value']['velocity'][:6]),
        'joint_drift':drift,'tcp_drift':tcp,'captured_continuously':True,'bilateral_effort_continuously':True,
        'native_topic_cross_check':True,'raw_velocities_modified':False}
