"""Follow a Cartesian segment using adjacent IK solutions, checking its interior."""
import numpy as np


def continuation_path(start_joints,start_position,goal_position,solve_near,state_valid,
                      cartesian_step=.005,max_joint_jump=.20,max_joint_step=.025):
    q=np.asarray(start_joints,dtype=float)
    start=np.asarray(start_position,dtype=float);goal=np.asarray(goal_position,dtype=float)
    if q.ndim!=1 or not len(q) or start.shape!=(3,) or goal.shape!=(3,):
        raise ValueError('invalid Cartesian continuation dimensions')
    if not all(np.isfinite(a).all() for a in (q,start,goal)) or min(cartesian_step,max_joint_jump,max_joint_step)<=0:
        raise ValueError('invalid Cartesian continuation values')
    if not state_valid(q):return None
    count=max(1,int(np.ceil(np.linalg.norm(goal-start)/cartesian_step)))
    path=[q.copy()]
    for index in range(1,count+1):
        position=start+(goal-start)*(index/count)
        solved=solve_near(position,q.copy())
        if solved is None:return None
        next_q=np.asarray(solved,dtype=float)
        if next_q.shape!=q.shape or not np.isfinite(next_q).all():return None
        jump=float(np.max(np.abs(next_q-q)))
        if jump>max_joint_jump:return None
        subdivisions=max(1,int(np.ceil(60/count)),int(np.ceil(jump/max_joint_step)))
        for fraction in np.linspace(0.,1.,subdivisions+1)[1:]:
            state=q+fraction*(next_q-q)
            if not state_valid(state):return None
            path.append(state)
        q=next_q
    return np.asarray(path)
