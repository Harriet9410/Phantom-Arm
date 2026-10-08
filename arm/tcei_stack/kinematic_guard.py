"""Joint-angle normalization within physical limits; no simulator mutations."""
import math
import numpy as np


def normalize_configuration(configuration,current,lower,upper):
    q=np.asarray(configuration,dtype=float)
    current,lower,upper=map(lambda x:np.asarray(x,dtype=float),(current,lower,upper))
    if q.ndim!=1 or any(x.shape!=q.shape for x in (current,lower,upper)):
        raise ValueError('joint configuration dimensions')
    if not all(np.isfinite(x).all() for x in (q,current,lower,upper)) or np.any(lower>=upper):
        raise ValueError('invalid joint configuration or limits')
    result=[]
    for angle,reference,lo,hi in zip(q,current,lower,upper):
        start=math.ceil((lo-angle)/(2*math.pi));end=math.floor((hi-angle)/(2*math.pi))
        options=[angle+2*math.pi*k for k in range(start,end+1)]
        if not options:raise ValueError('no equivalent angle inside physical limits')
        result.append(min(options,key=lambda x:abs(x-reference)))
    return np.asarray(result)


def rotation_angle(first,second):
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if a.shape!=(3,3) or b.shape!=(3,3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid orientation matrix')
    return math.acos(float(np.clip((np.trace(a.T@b)-1.)/2.,-1.,1.)))


def quaternion_rotation(quaternion_wxyz):
    q=np.asarray(quaternion_wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
        raise ValueError('invalid orientation quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
