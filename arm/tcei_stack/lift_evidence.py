"""Camera surface evidence for a short, translation-only trial lift.

No ROS, simulator body state, expected class inventory or target attachment is
read. The caller supplies actual end-effector translation in optical camera
coordinates and, for positive evidence, a contemporaneous robot mask obtained
from measured robot geometry. An arbitrary/all-background mask is not a valid
substitute for that external geometry contract.

Source vacancy alone is never positive evidence. Without a robot mask this
module can reject an unchanged source but cannot confirm a lifted object.
Even with a mask, evidence is geometric consistency of the same surface under
a short rigid translation, not a universal visual identity guarantee.
"""
import math
import numpy as np


def _points(candidate):
    raw=candidate.get('surface_reference',[])
    points=np.asarray(raw,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<8:
        raise ValueError('insufficient surface reference')
    points=points[np.isfinite(points).all(axis=1)&(points[:,2]>.1)&(points[:,2]<5.)]
    if len(points)<8:raise ValueError('insufficient finite surface reference')
    # Duplicated reference pixels cannot manufacture independent evidence.
    _,indices=np.unique(np.rint(points[:,:2]).astype(int),axis=0,return_index=True)
    return points[np.sort(indices)]


def _sample(depth,pixels,robot_mask):
    h,w=depth.shape
    integer=np.rint(pixels).astype(int)
    inside=(integer[:,0]>=0)&(integer[:,0]<w)&(integer[:,1]>=0)&(integer[:,1]<h)
    values=np.full(len(pixels),np.nan)
    robot=np.zeros(len(pixels),dtype=bool)
    values[inside]=depth[integer[inside,1],integer[inside,0]]
    if robot_mask is not None:robot[inside]=robot_mask[integer[inside,1],integer[inside,0]]
    valid=inside&np.isfinite(values)&(values>.1)&(values<5.)
    return values,valid,robot


def _spread(points,selected):
    """Require support distributed over reference object, not one image patch."""
    if not np.any(selected):return {'cells':0,'x_fraction':0.,'y_fraction':0.}
    lo=points[:,:2].min(axis=0);extent=np.maximum(points[:,:2].max(axis=0)-lo,1.)
    normalized=np.clip((points[selected,:2]-lo)/extent,0.,.999999)
    cells=np.floor(normalized*4).astype(int)
    span=np.ptp(points[selected,:2],axis=0)/extent
    return {'cells':len({tuple(cell) for cell in cells}),
            'x_fraction':float(span[0]),'y_fraction':float(span[1])}


def _distributed(spread,min_cells=4):
    return spread['cells']>=min_cells and spread['x_fraction']>=.35 and spread['y_fraction']>=.35


def evaluate_lift_frame(candidate,depth,k,camera_translation,*,stamp,
                        robot_mask=None,robot_mask_stamp=None,orientation_delta_rad=None,
                        depth_tolerance=.004,min_points=8,min_cells=4):
    """Classify matched/source_unchanged/occluded/unobservable surface evidence.

    camera_translation: measured EE displacement, in optical x-right/y-down/
    z-forward metres. Do not pass command displacement or untransformed world
    axes. Positive evidence requires >=10 mm motion toward camera, a valid
    same-frame robot exclusion mask and dispersed surface points. Rotation
    >0.03 rad is unsupported here. During a vertical lift the object may still
    project over its original source, so empty floor is not a prerequisite.

    robot_mask_stamp must match the depth frame within 0.05 s. Mask provenance
    and camera/robot extrinsics are external validation responsibilities.
    """
    proof={'status':'unobservable','stamp':stamp,'positive':False,
           'source_vacancy_is_proof':False,'reason':'insufficient_evidence'}
    try:
        stamp=float(stamp);proof['stamp']=stamp
        points=_points(candidate)
        depth=np.asarray(depth,dtype=float);k=np.asarray(k,dtype=float)
        delta=np.asarray(camera_translation,dtype=float)
        if depth.ndim!=2 or k.shape!=(9,) or not np.isfinite(k).all() or min(k[0],k[4])<=0:
            raise ValueError('invalid image or intrinsics')
        if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('invalid measured translation')
        if not math.isfinite(stamp):raise ValueError('invalid frame stamp')
        if orientation_delta_rad is None:raise ValueError('measured_orientation_change_required')
        if not math.isfinite(orientation_delta_rad) or abs(orientation_delta_rad)>.03:
            raise ValueError('rotation_not_supported_by_translation_verifier')
        if np.linalg.norm(delta)>.10:raise ValueError('translation_exceeds_trial_lift_scope')
        expected_k=candidate.get('surface_reference_meta',{}).get('intrinsics')
        if expected_k is not None and not np.allclose(k,expected_k,rtol=0,atol=1e-5):
            raise ValueError('reference_and_current_intrinsics_differ')
        if robot_mask is not None:
            robot_mask=np.asarray(robot_mask,dtype=bool)
            if robot_mask.shape!=depth.shape:raise ValueError('robot_mask_shape_mismatch')
            if robot_mask_stamp is None or not math.isfinite(robot_mask_stamp) or abs(robot_mask_stamp-stamp)>.05:
                raise ValueError('robot_mask_not_synchronized')
        proof['camera_translation']=delta.tolist()
        proof['orientation_delta_rad']=float(orientation_delta_rad)
        proof['reference_points']=len(points)
        proof['robot_mask_available']=robot_mask is not None
        z=points[:,2]
        xyz=np.column_stack(((points[:,0]-k[2])*z/k[0],(points[:,1]-k[5])*z/k[4],z))
        moved=xyz+delta
        if np.any(moved[:,2]<=.1):raise ValueError('projected_surface_behind_camera')
        projected=np.column_stack((moved[:,0]*k[0]/moved[:,2]+k[2],moved[:,1]*k[4]/moved[:,2]+k[5]))
        original_values,original_valid,original_robot=_sample(depth,points[:,:2],robot_mask)
        values,valid,robot=_sample(depth,projected,robot_mask)
        source_visible=original_valid&~original_robot
        unchanged=source_visible&(np.abs(original_values-z)<=depth_tolerance)
        source_spread=_spread(points,unchanged)
        source_vacated=source_visible&(original_values>z+depth_tolerance)
        vacated_spread=_spread(points,source_vacated)
        unmasked=valid&~robot
        occluded=valid&((values<moved[:,2]-depth_tolerance)|robot)
        matched=unmasked&(np.abs(values-moved[:,2])<=depth_tolerance)
        spread=_spread(points,matched)
        visible_pixels=np.rint(projected[matched]).astype(int)
        pixel_span=np.ptp(visible_pixels,axis=0).tolist() if len(visible_pixels) else [0,0]
        proof.update({'source_visible_points':int(source_visible.sum()),
                      'source_unchanged_points':int(unchanged.sum()),
                      'source_vacated_points':int(source_vacated.sum()),
                      'matched_points':int(matched.sum()),'unmasked_valid_points':int(unmasked.sum()),
                      'occluded_points':int(occluded.sum()),'matched_spread':spread,
                      'source_unchanged_spread':source_spread,'source_vacated_spread':vacated_spread,
                      'matched_fraction':float(matched.sum()/max(1,unmasked.sum()))})
        proof.update(matched_reference_indices=np.flatnonzero(matched).tolist(),
                     matched_unique_pixels=len({tuple(pixel) for pixel in visible_pixels}),
                     matched_pixel_span=pixel_span)
        # Only a requested/measured change large enough to distinguish the
        # original surface makes unchanged source samples a useful rejection.
        enough_motion=delta[2]<=-.010
        if enough_motion and unchanged.sum()>=min_points and _distributed(source_spread,min_cells) and unchanged.sum()>=.65*max(1,source_visible.sum()):
            proof.update(status='source_unchanged',reason='original_surface_still_measured_after_trial_lift')
        elif not enough_motion:
            proof['reason']='insufficient_measured_vertical_lift'
        elif robot_mask is None:
            proof['reason']='robot_exclusion_mask_required_for_positive_evidence'
        elif matched.sum()>=min_points and _distributed(spread,min_cells) and proof['matched_fraction']>=.70:
            proof.update(status='matched',positive=True,reason='dispersed_surface_matches_measured_translation_excluding_robot')
        elif occluded.sum()>=.50*len(points):
            proof.update(status='occluded',reason='predicted_object_surface_occluded')
        else:proof['reason']='insufficient_distributed_surface_match'
        # A narrow exposed end is insufficient by itself. Keep it explicitly
        # nonpositive; a separate window may combine it with stamped bilateral
        # contact, nonempty aperture and a longer same-surface motion sequence.
        proof['partial_match_candidate']=bool(proof['status']=='occluded' and robot_mask is not None and
            delta[2]<=-.015 and np.linalg.norm(delta[:2])<=.005 and abs(orientation_delta_rad)<=.015 and
            matched.sum()>=min_points and proof['matched_unique_pixels']>=min_points and
            min(pixel_span)>=2 and spread['cells']>=3 and proof['matched_fraction']>=.90)
    except (ValueError,TypeError,KeyError,IndexError) as error:
        proof['reason']=str(error)
    return proof


class LiftEvidenceWindow:
    """Require fresh consecutive frames and a measured-motion baseline."""
    def __init__(self,reference_stamp,min_frames=3,min_span=.15,max_gap=.35,
                 min_motion_baseline=.008):
        self.reference_stamp=float(reference_stamp);self.min_frames=min_frames
        self.min_span=min_span;self.max_gap=max_gap;self.min_motion_baseline=min_motion_baseline
        self.frames=[];self.last_stamp=self.reference_stamp

    def add(self,proof):
        stamp=proof.get('stamp')
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<=self.last_stamp:
            self.frames=[]
            return {'verified':False,'reason':'stale_or_duplicate_frame','frames':0}
        gap=stamp-self.last_stamp;self.last_stamp=stamp
        if proof.get('status')!='matched' or not proof.get('positive'):
            self.frames=[]
            return {'verified':False,'reason':proof.get('reason','nonmatching_frame'),'frames':0,
                    'rejected':proof.get('status')=='source_unchanged'}
        if gap>self.max_gap:self.frames=[]
        current=np.asarray(proof.get('camera_translation',[]),dtype=float)
        if current.shape!=(3,) or not np.isfinite(current).all():
            self.frames=[]
            return {'verified':False,'reason':'invalid_measured_translation','frames':0}
        if self.frames:
            previous=np.asarray(self.frames[-1]['camera_translation'])
            # A reversing/downward target cannot be pooled into a clean lift.
            if current[2]>previous[2]+.002:
                self.frames=[]
                return {'verified':False,'reason':'measured_lift_reversed','frames':0}
        self.frames.append(dict(proof))
        span=stamp-self.frames[0]['stamp']
        baseline=float(abs(current[2]-self.frames[0]['camera_translation'][2]))
        verified=len(self.frames)>=self.min_frames and span>=self.min_span and baseline>=self.min_motion_baseline
        return {'verified':verified,'reason':'surface_following_verified' if verified else 'collecting_fresh_motion_baseline',
                'frames':len(self.frames),'span_seconds':span,'vertical_baseline_m':baseline,
                'evidence':'surface_geometry_excluding_measured_robot_mask'}


class ContactSurfaceWindow:
    """Five aligned observations of the same visible patch plus stable contact.

    Force alone, source vacancy, a robot-colored patch, a stationary object and
    fully occluded depth cannot pass. Partial support has a stronger temporal
    and displacement requirement than the dispersed-surface-only verifier.
    """
    def __init__(self,reference_stamp):
        self.last_stamp=float(reference_stamp);self.frames=[];self.common=set()

    def add(self,proof,contact):
        stamp=proof.get('stamp');reason=None
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<=self.last_stamp:
            reason='stale_or_duplicate_frame'
        if reason is None:
            gap=stamp-self.last_stamp;self.last_stamp=stamp
            if gap>.35:self.frames=[];self.common=set()
            if not proof.get('partial_match_candidate'):reason='no_qualified_partial_surface'
            elif not isinstance(contact,dict) or not contact.get('stable_contact') or not contact.get('nonempty_aperture'):
                reason='stable_nonempty_contact_required'
            elif contact.get('state')!='holding' or contact.get('contact_lost') or contact.get('fault'):
                reason='contact_not_supported'
            else:
                stamps=contact.get('sensor_stamps',[])
                if len(stamps)!=2 or not all(isinstance(t,(int,float)) and math.isfinite(t) and abs(t-stamp)<=.05 for t in stamps):
                    reason='contact_and_depth_not_synchronized'
        delta=np.asarray(proof.get('camera_translation',[]),dtype=float)
        if reason is None and (delta.shape!=(3,) or not np.isfinite(delta).all()):reason='invalid_measured_translation'
        if reason is None and self.frames and delta[2]>self.frames[-1]['z']+.002:reason='measured_lift_reversed'
        ids=proof.get('matched_reference_indices',[])
        if reason is None and (not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids) or len(set(ids))<8):
            reason='insufficient_unique_reference_points'
        if reason is not None:
            self.frames=[];self.common=set()
            return {'verified':False,'reason':reason,'frames':0}
        common=set(ids) if not self.frames else self.common.intersection(ids)
        if len(common)<8:self.frames=[];common=set(ids)
        self.common=common
        self.frames.append({'stamp':stamp,'z':float(delta[2]),'sensor_stamps':tuple(contact['sensor_stamps'])})
        span=stamp-self.frames[0]['stamp'];baseline=float(abs(delta[2]-self.frames[0]['z']))
        acquisitions=len({row['sensor_stamps'] for row in self.frames})
        verified=len(self.frames)>=5 and span>=.25-1e-9 and baseline>=.012-1e-9 and acquisitions>=3
        return {'verified':verified,'reason':'contact_and_partial_surface_following_verified' if verified else 'collecting_contact_surface_baseline',
                'frames':len(self.frames),'span_seconds':span,'vertical_baseline_m':float(baseline),
                'common_reference_points':len(self.common),'distinct_contact_acquisitions':acquisitions,
                'evidence':'stamped_bilateral_contact_nonempty_aperture_and_same_unmasked_surface_motion'}
