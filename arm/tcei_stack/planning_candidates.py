"""Bounded IK search and preview binding.  No scene truth or physical edits."""
import numpy as np
from kinematic_guard import normalize_configuration,rotation_angle,quaternion_rotation


def ik_seed_groups(current,lower,upper):
    current=np.asarray(current,dtype=float)
    if current.shape!=(6,):raise ValueError('JAKA search expects six arm joints')
    lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if (lower.shape!=current.shape or upper.shape!=current.shape or
            not all(np.isfinite(x).all() for x in (current,lower,upper)) or np.any(lower>=upper)):
        raise ValueError('invalid seed reference or physical limits')
    # Keep the four V6 seeds as the normal fast group.  Seed clipping merely
    # chooses an initial guess; returned solutions still face exact limits.
    base=[('current',current.copy()),('home',np.array([0.,1.3,-1.5,4.5,1.6,0.]))]
    for shoulder,elbow in [(.8,-1.3),(1.4,-2.)]:
        seed=current.copy();seed[1]=shoulder;seed[2]=elbow;seed[3]=3.9
        base.append(('legacy_%.1f_%.1f'%(shoulder,elbow),seed))
    extra=[]
    for shoulder,elbow,wrist in [(.4,-.8,3.2),(1.8,-2.4,4.8),
                                 (-.4,.8,-3.2),(-1.3,1.5,-4.5)]:
        seed=current.copy();seed[1:4]=[shoulder,elbow,wrist]
        extra.append(('alternate_%.1f_%.1f'%(shoulder,elbow),seed))
    for delta in (-.5,.5):
        seed=current.copy();seed[0]+=delta;seed[3]-=delta
        extra.append(('base_wrist_%+.1f'%delta,seed))
        seed=current.copy();seed[4]+=delta;seed[5]-=delta
        extra.append(('wrist_%+.1f'%delta,seed))
    seen=[];groups=[]
    for group in (base,extra):
        out=[]
        for name,seed in group:
            seed=np.clip(seed,np.asarray(lower),np.asarray(upper))
            if not np.isfinite(seed).all():raise ValueError('nonfinite seed or bounds')
            if any(np.allclose(seed,old,atol=1e-8,rtol=0) for old in seen):continue
            seen.append(seed);out.append((name,seed))
        groups.append(out)
    return groups


def evaluate_ik_solution(joints,reference,lower,upper,holding,position,orientation,
                         forward,clearance):
    """Return a legal solution and an explicit rejection, with V6 constraints."""
    try:q=normalize_configuration(joints,reference,lower,upper)
    except ValueError as error:return None,{'reason':'joint_limits','detail':str(error)}
    if holding and np.max(np.abs(q-reference))>np.pi:
        return None,{'reason':'held_joint_unwind'}
    rejection=clearance(q)
    if rejection is not None:return None,{'reason':'table_clearance',**rejection}
    actual,rotation=forward(q)
    error=float(np.linalg.norm(np.asarray(actual).reshape(-1)-position))
    if not np.isfinite(error) or error>.005:
        return None,{'reason':'fk_position_error','position_error':error}
    angle=rotation_angle(rotation,quaternion_rotation(orientation)) if orientation is not None else 0.
    if angle>.02:return None,{'reason':'fk_orientation_error','orientation_error':angle}
    return q,{'reason':'accepted','position_error':error,'orientation_error':angle,
              'joint_distance':float(np.linalg.norm(q-reference))}


def validate_preview_binding(cache,request_id,position,orientation,current,now,
                             max_age=10.,joint_tolerance=.005):
    if not cache or cache.get('id')!=request_id:raise ValueError('preview id unavailable')
    age=float(now-cache['created_at'])
    if not np.isfinite(age) or age<0 or age>max_age:raise ValueError('preview expired')
    position=np.asarray(position,dtype=float);current=np.asarray(current,dtype=float)
    if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('invalid preview target')
    if np.linalg.norm(position-np.asarray(cache['position']))>1e-5:
        raise ValueError('preview target changed')
    if rotation_angle(quaternion_rotation(orientation),quaternion_rotation(cache['orientation']))>1e-5:
        raise ValueError('preview attitude changed')
    old=np.asarray(cache['start_joints'])
    if current.shape!=old.shape or not np.isfinite(current).all() or np.max(np.abs(current-old))>joint_tolerance:
        raise ValueError('preview start joints changed')
    return np.asarray(cache['path'],dtype=float).copy()
