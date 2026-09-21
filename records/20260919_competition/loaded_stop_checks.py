"""Independent measured-feedback checks for a deliberately cancelled load test."""
import math

J='/Jaka/get_jointstate';P='/Jaka/get_end_effector_pose';C='/Jaka/gripper_is_captured'
E='/Jaka/get_gripper_efforts';K='/clock'

def vector(value,n):
    if (not isinstance(value,list) or len(value)!=n or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in value)):
        raise ValueError('invalid measured vector')
    return value

def stamp(value):
    sec=value['secs'];ns=value['nsecs']
    if type(sec) is not int or type(ns) is not int or sec<0 or not 0<=ns<1000000000:
        raise ValueError('invalid native source time')
    return sec+ns/1e9

def position(value):return vector([value['pose']['position'][k] for k in ('x','y','z')],3)

def moving_lift(origin,goal,diagnostic):
    p=vector(diagnostic['tcp_actual'],3);target=vector(goal,3);start=vector(origin,3)
    if diagnostic.get('tracking') is not True:return False
    actual_goal=diagnostic.get('goal',{}).get('position')
    if actual_goal is None or math.dist(actual_goal,target)>.0001:return False
    if math.hypot(p[0]-start[0],p[1]-start[1])>.002:raise ValueError('unexpected lateral motion')
    return p[2]-start[2]>=.002 and math.dist(p,target)>=.005

def held_stationary(rows,since,now,seconds=.25):
    streams={t:[r for r in rows if r['topic']==t and r['mono']>=since] for t in (J,P,C,E,K)}
    for topic in (J,P,K):
        filtered=[];last=None
        for r in streams[topic]:
            value=stamp(r['value']['clock'] if topic==K else r['value']['header']['stamp'])
            if last is not None and value<last:raise ValueError('native source time moved backwards')
            if last is None or value>last:filtered.append(r)
            last=value
        streams[topic]=filtered
    if any(len(v)<3 or now-v[-1]['mono']>1.5 for v in streams.values()):return None
    low=max(v[0]['mono'] for v in streams.values());high=min(v[-1]['mono'] for v in streams.values())
    clocks=[stamp(r['value']['clock']) for r in streams[K] if low<=r['mono']<=high]
    if len(clocks)<3 or clocks[-1]-clocks[0]<seconds:return None
    if any(any(b['mono']-a['mono']>1.5 for a,b in zip(v,v[1:])) for v in streams.values()):
        raise ValueError('native feedback gap')
    joints=[r['value'] for r in streams[J]]
    names=joints[0]['name']
    if len(names)!=8 or len(set(names))!=8 or any(j['name']!=names for j in joints):
        raise ValueError('joint identity changed')
    qs=[vector(j['position'],8) for j in joints];vs=[vector(j['velocity'],8) for j in joints]
    ps=[position(r['value']) for r in streams[P]]
    rots=[vector([r['value']['pose']['orientation'][k] for k in ('x','y','z','w')],4) for r in streams[P]]
    if any(abs(sum(x*x for x in q)-1.)>.002 for q in rots):raise ValueError('invalid native quaternion')
    speed=max(abs(x) for v in vs for x in v[:6])
    drift=[max(abs(q[i]-qs[0][i]) for q in qs) for i in range(8)]
    tcp=max(math.dist(p,ps[0]) for p in ps)
    angle=max(2*math.acos(min(1.,abs(sum(a*b for a,b in zip(q,rots[0]))))) for q in rots)
    if speed>.03 or max(drift[:6])>.0015 or max(drift[6:])>.0002 or tcp>.0015 or angle>.01:return None
    if any(r['value'].get('data') is not True for r in streams[C]):return None
    if any(min(vector(r['value']['data'],2))<.2 for r in streams[E]):return None
    if any(not all(.001<abs(v)<.038 for v in q[-2:]) for q in qs):return None
    return {'native_counts':{t:len(v) for t,v in streams.items()},'simulation_span':clocks[-1]-clocks[0],
            'max_arm_speed':speed,'joint_drift':drift,'tcp_drift':tcp,'rotation_drift':angle,
            'captured_continuously':True,'bilateral_effort_continuously':True,
            'nonempty_measured_aperture':True,'anchor_tcp':ps[0],
            'time_pairing':'overlapping receipt interval; not atomic across ROS topics'}
