"""Camera-only basket visibility, with tall objects distinct from occlusion.

The footprint is calibrated independently of detections. An isolated object
10 cm above the floor is not proof of robot occlusion. Connected near-camera
foreground entering from outside the basket remains an occlusion hypothesis.
"""
import cv2
import numpy as np

OBSERVATION_POSITION=[.0662943895,.40,2.7503792615]
TRANSFER_POSITION=[.0662943895,.0862721270,2.7503792615]
FLOOR_DEPTH=4.5-2.37212
# Fixed, previously calibrated inner basket footprint, not an object envelope.
BASKET_BOUNDS=(-.333,.319,-.269,.178)


def basket_roi(k,shape):
    """Image left/top/right/bottom from the fixed basket floor projection."""
    if len(k)!=9 or not np.isfinite(k).all() or k[0]<=0 or k[4]<=0:
        raise ValueError('invalid camera intrinsics')
    h,w=shape[:2]
    xmin,xmax,ymin,ymax=BASKET_BOUNDS
    x1=k[2]-xmax*k[0]/FLOOR_DEPTH
    x2=k[2]-xmin*k[0]/FLOOR_DEPTH
    y1=k[5]+(ymin-.1)*k[4]/FLOOR_DEPTH
    y2=k[5]+(ymax-.1)*k[4]/FLOOR_DEPTH
    if min(x1,y1)<0 or x2>w or y2>h or x2-x1<10 or y2-y1<10:
        raise ValueError('calibrated basket is outside the camera image')
    return [float(x1),float(y1),float(x2),float(y2)]


def basket_mask(k,shape):
    x1,y1,x2,y2=basket_roi(k,shape)
    yy,xx=np.indices(shape[:2],dtype=np.float32)
    return (xx>x1)&(xx<x2)&(yy>y1)&(yy<y2)


def spatial_context(k,shape):
    return {'reference':'camera_image','definition':'basket_quadrants_v1',
            'basket_roi':basket_roi(k,shape),
            'pixel_to_reference':[[1,0,0],[0,1,0],[0,0,1]],
            'transform_declared':True,'uncertainty_px':5.,
            'assumption':'Image left/up in calibrated basket footprint; referee reference convention unconfirmed.'}


def normalized_xy(pixel,roi):
    x1,y1,x2,y2=roi
    return [(float(pixel[0])-x1)/(x2-x1),(float(pixel[1])-y1)/(y2-y1)]


def visibility_layers(depth,k,robot_mask=None):
    """Return visibility metadata and conservative exclusion mask.

    Supplied robot masks must come from measured robot/camera geometry. Without
    one, ambiguous very high foreground is blocked, not asserted to be robot.
    """
    depth=np.asarray(depth)
    inside=basket_mask(k,depth.shape)
    valid=np.isfinite(depth)&(depth>.1)&(depth<5.)
    if inside.sum()<100:raise ValueError('invalid camera calibration')
    high=valid&(depth<FLOOR_DEPTH-.10)
    count,labels,stats,_=cv2.connectedComponentsWithStats(high.astype(np.uint8),8)
    excluded=np.zeros(depth.shape,dtype=bool)
    regions=[];tall_pixels=0
    for label in range(1,count):
        component=labels==label
        in_part=component&inside
        pixels=int(in_part.sum())
        if pixels<10:continue
        maximum=float(np.max(FLOOR_DEPTH-depth[in_part]))
        outside_pixels=int((component&~inside).sum())
        reason=None
        if maximum>.35:reason='unresolved_nearfield_foreground'
        elif outside_pixels>=10 and maximum>.20:
            reason='connected_external_high_foreground'
        if reason:
            excluded|=component
            ys,xs=np.nonzero(in_part)
            regions.append({'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                            'reason':reason,'pixels':pixels,'identity':'unknown',
                            'max_height_m':maximum})
        else:tall_pixels+=pixels
    if robot_mask is not None:
        robot_mask=np.asarray(robot_mask,dtype=bool)
        if robot_mask.shape!=depth.shape:raise ValueError('robot mask shape mismatch')
        selected=robot_mask&inside
        if selected.any():
            ys,xs=np.nonzero(selected)
            regions.append({'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                            'reason':'measured_robot_mask','pixels':int(selected.sum()),'identity':'robot'})
            excluded|=robot_mask
    invalid_fraction=1.-float(valid[inside].mean())
    occluded=int((excluded&inside).sum())
    meta={'clear':occluded<10 and invalid_fraction<.005,
          'high_foreground_pixels':int((high&inside).sum()),
          'tall_object_or_unknown_pixels':tall_pixels,
          'basket_pixels':int(inside.sum()),'invalid_fraction':invalid_fraction,
          'occlusion_fraction':occluded/int(inside.sum()),'occluded_regions':regions,
          'source':'current_depth_fixed_basket_projection',
          'robot_identity_verified':robot_mask is not None}
    if occluded>=10:meta['reason']='occluded_or_unresolved_nearfield'
    elif invalid_fraction>=.005:meta['reason']='invalid_basket_depth'
    return meta,excluded


def basket_visibility(depth,k,robot_mask=None):
    try:return visibility_layers(depth,k,robot_mask)[0]
    except (ValueError,IndexError,TypeError) as error:
        return {'clear':False,'reason':'invalid camera calibration: '+str(error),
                'occluded_regions':[],'invalid_fraction':1.}
