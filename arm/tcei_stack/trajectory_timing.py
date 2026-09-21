"""Retiming only: preserve the sampled joint path, bound commanded v and a."""
import numpy as np


def parameterize_joint_path(positions,duration):
    """Preserve route vertices while assigning time by joint-space distance.

    A checked connection to a cached preview can have 61 near-stationary samples.
    Sample-index timing gives that tiny connection as much time as a long edge,
    creating a velocity jump that stretches the whole route during retiming.
    Duplicate vertices have no motion; genuine corners and endpoints stay exact.
    This supplies a geometric parameter, not permission to exceed motion limits.
    """
    p=np.asarray(positions,dtype=float)
    if p.ndim!=2 or len(p)<2 or not p.shape[1] or not np.isfinite(p).all():
        raise ValueError('invalid joint route')
    if not np.isfinite(duration) or duration<=0:raise ValueError('invalid route duration')
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
    keep=np.r_[True,lengths>0.]
    clean=p[keep]
    if len(clean)==1:return np.vstack([p[0],p[-1]]),np.array([0.,duration])
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(clean,axis=0),axis=1))]
    times=duration*distance/distance[-1]
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('joint route parameter precision exhausted')
    return clean,times


def retime_path(positions,times,desired_seconds,velocity_limits,acceleration_limit=2.):
    p=np.asarray(positions,dtype=float);t=np.asarray(times,dtype=float)
    limits=np.asarray(velocity_limits,dtype=float)
    if p.ndim!=2 or len(p)<2 or len(t)!=len(p) or limits.shape!=(p.shape[1],):
        raise ValueError('trajectory dimensions')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('trajectory values or time order')
    if desired_seconds<=0 or not np.isfinite(desired_seconds) or np.any(limits<=0) or not np.isfinite(limits).all() or acceleration_limit<=0 or not np.isfinite(acceleration_limit):
        raise ValueError('invalid motion limits')
    count=max(101,int(desired_seconds*100)+1)
    u=np.linspace(0.,1.,count)
    phase=np.clip(10*u**3-15*u**4+6*u**5,0.,1.)
    old_t=t[0]+phase*(t[-1]-t[0])
    new_p=np.column_stack([np.interp(old_t,t,p[:,j]) for j in range(p.shape[1])])
    new_p[0]=p[0];new_p[-1]=p[-1]
    dt=desired_seconds/(count-1)
    v=np.diff(new_p,axis=0)/dt
    a=np.diff(np.vstack([np.zeros((1,p.shape[1])),v,np.zeros((1,p.shape[1]))]),axis=0)/dt
    scale=max(1.,float(np.max(np.abs(v)/limits)),float(np.sqrt(np.max(np.abs(a))/acceleration_limit)))
    if scale>1.:scale*=1.01
    duration=desired_seconds*scale
    new_t=np.linspace(0.,duration,count)
    metadata={'duration':duration,'peak_command_velocity':np.max(np.abs(v)/scale,axis=0).tolist(),
              'peak_command_acceleration':np.max(np.abs(a)/(scale*scale),axis=0).tolist()}
    return new_p,new_t,metadata
