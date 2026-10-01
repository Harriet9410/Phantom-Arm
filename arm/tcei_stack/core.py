"""Pure, testable guards shared by perception, Nine and execution."""
import json
import math
import statistics
import re
from tcei_workspace import validate_workspace

CLASSES = ('Magazine', 'Torch', 'Grenade', 'Smokegrenade', 'CompressedFood')
from semantics import ALIASES, parse_model_plan, parse_compact_selection



def occlusion_depth(reference_depth,values):
    """Return measured high foreground only when a mostly valid patch agrees."""
    if not values or not math.isfinite(reference_depth):return None
    good=[float(v) for v in values if math.isfinite(v) and .1<v<5.]
    if len(good)<.8*len(values):return None
    median=statistics.median(good)
    return median if median<reference_depth-.10 and median<2.0 else None


def parse_plan(text, candidates, instruction, scene=None, task_context=None, allow_legacy=True):
    """Reject Nine answers that drop task constraints or select wrong geometry."""
    return parse_model_plan(text, candidates, instruction, scene, task_context, allow_legacy)


def parse_selection(text,candidates,instruction,scene,task_context=None):
    """Explicit compact Nine wire protocol with unchanged full task guards."""
    return parse_compact_selection(text,candidates,instruction,scene,task_context)


def rebind_candidate(selected, current, max_pixels=14.0, max_depth=.025):
    possible = []
    for c in current:
        if c['class'] != selected['class']:
            continue
        d = math.dist(c['pixel'], selected['pixel'])
        if d <= max_pixels and abs(c['depth'] - selected['depth']) <= max_depth:
            possible.append((d, c))
    possible.sort(key=lambda x: x[0])
    if not possible or (len(possible) > 1 and possible[1][0] - possible[0][0] < 5):
        raise ValueError('scene changed or correspondence ambiguous; reobserve')
    return possible[0][1]


def legacy_quaternion(angle_deg):
    """Wire compatibility only: this image stores Isaac wxyz in ROS xyzw fields.

    Preserve the simulator, explicitly isolate its convention at the boundary.
    Downward tool + yaw, never normalize Euler angles as quaternion components.
    Grasp axis is undirected (180 degrees periodic).
    """
    if not math.isfinite(angle_deg):
        raise ValueError('nonfinite angle')
    angle_deg = (angle_deg + 90.0) % 180.0 - 90.0
    a = math.radians(angle_deg) / 2
    return [0.0, math.sin(a), math.cos(a), 0.0]


def quaternion_error(a, b):
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return math.inf
    dot = abs(sum(x*y for x, y in zip(a, b)) / (na * nb))
    return 2 * math.acos(min(1., dot))


def calibrated_grasp(candidate,attempt=0,yaw_variant=0,tilt_deg=0.):
    """Current scene: measured optical/world transform + URDF jaw/TCP geometry.

    Jaw midpoint is 12 mm behind the TCP in tool X; audited distal finger Z is
    69.27 mm. Rotate the XY compensation with the grasp orientation and keep
    the finger tips above the support surface, including on the second attempt.
    """
    world=candidate.get('world_position');support=candidate.get('support_z')
    if world is None or support is None or len(world)!=3:
        raise ValueError('missing measured grasp geometry')
    if not all(math.isfinite(x) for x in world) or not math.isfinite(support) or world[2]-support<.002:
        raise ValueError('invalid object/support geometry')
    # Depth body geometry suppresses thin appendages before estimating yaw.
    # The class-independent circular-body fallback is applied in perception.
    if yaw_variant not in (0,180):raise ValueError('unsupported grasp symmetry variant')
    if isinstance(tilt_deg,bool) or not math.isfinite(tilt_deg) or tilt_deg not in (0.,-10.,10.):
        raise ValueError('unsupported bounded grasp tilt')
    angle=(candidate['angle_deg']+90.)%180.-90.+yaw_variant
    # Preserve the chosen wrist branch: a 180-degree tool yaw swaps the two
    # fingers but must rotate the 12 mm TCP offset as well.
    half=math.radians(angle)/2.
    q=[0.,math.sin(half),math.cos(half),0.]
    rad=math.radians(angle)
    penetration=min(.012,max(.004,(world[2]-support)/2.))
    if candidate['class'] in ('Grenade','Smokegrenade','Torch'):
        # Round body: fingertips must reach below its mid-height so lifting
        # does not simply squeeze the upper hemisphere downward out of the jaws.
        penetration=min(.028,max(.004,.65*(world[2]-support)))
    z=max(world[2]+.06927-penetration-attempt*.004,support+.06927+.002)
    target=[world[0]-.012*math.cos(rad),world[1]+.012*math.sin(rad),z]
    if tilt_deg:
        # Roll about the closing axis, preserving the measured jaw closing
        # direction. Compensate the original local jaw midpoint, not a made-up
        # world offset. No wrist reorientation takes place while carrying.
        c,s=math.cos(math.radians(tilt_deg)/2),math.sin(math.radians(tilt_deg)/2)
        w,x,y,zq=q
        q=[w*c-x*s,w*s+x*c,y*c+zq*s,zq*c-y*s]
        w,x,y,zq=q
        rotation=((1-2*(y*y+zq*zq),2*(x*y-zq*w),2*(x*zq+y*w)),
                  (2*(x*y+zq*w),1-2*(x*x+zq*zq),2*(y*zq-x*w)),
                  (2*(x*zq-y*w),2*(y*zq+x*w),1-2*(x*x+y*y)))
        contact=(-.012,0.,.06927-penetration-attempt*.004)
        target=[world[i]-sum(rotation[i][j]*contact[j] for j in range(3)) for i in range(3)]
        lowest_offset=empty_finger_lowest_z([0.,0.,0.],q)
        target[2]=max(target[2],support+.002-lowest_offset)
    validate_workspace(target)
    return target,q


def grasp_approach_heights(grasp_position,quaternion=None):
    """Measured basket rim + finger length + 25 mm empty-tool clearance.

    The held object first travels inward to the central high transfer point;
    it does not cross a basket rim at the initial lift height.
    """
    minimum=2.485238+.06927+.025
    lowest=max(minimum,float(grasp_position[2])+.12)
    # Preserve the established 160 mm lift where the complete route is valid.
    # Lower alternatives are used only when that route fails preview.
    # Last resort only, after both symmetric wrist poses at standard heights.
    # Retain15mm commanded nominal finger/rim clearance. The controller also
    # verifies >=10mm actual clearance using the padded open-finger mesh bounds.
    fallback=max(2.485238+.06927+.015,float(grasp_position[2])+.11)
    values=(max(lowest,grasp_position[2]+.16),max(lowest,grasp_position[2]+.14),lowest,fallback)
    if quaternion is not None:
        minimum_actual=2.485238+.011-empty_finger_lowest_z([0.,0.,0.],quaternion)
        values=tuple(max(z,minimum_actual) for z in values)
    return list(dict.fromkeys(round(z,6) for z in values))


def empty_finger_lowest_z(position,quaternion):
    """WorldZ lower bound from both open-finger STL bounds relative to TCP.

    Bounds include1mm padding. Wire quaternion is the scene's isolated wxyz
    convention. Rotation is measured, so tilted fingers are not treated as
    perfectly vertical when approving the lower approach fallback.
    """
    if len(position)!=3 or len(quaternion)!=4 or not all(math.isfinite(v) for v in list(position)+list(quaternion)):
        raise ValueError('invalid measured finger pose')
    norm=math.sqrt(sum(v*v for v in quaternion))
    if abs(norm-1.)>.01:raise ValueError('invalid measured finger quaternion')
    w,x,y,z=[v/norm for v in quaternion]
    row=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))
    bounds=((- .073,.051),(-.0312,.028),(-.0185,.0703))
    return float(position[2])+sum(min(a*lo,a*hi) for a,(lo,hi) in zip(row,bounds))


def reacquire_after_empty(selected,current,unique_at_start):
    if not unique_at_start:
        return rebind_candidate(selected,current)
    try:return rebind_candidate(selected,current,max_pixels=45.,max_depth=.04)
    except ValueError:pass
    options=[]
    for c in current:
        p=c.get('world_position')
        if c['class']!=selected['class'] or c['confidence']<.6 or p is None:continue
        if not (abs(p[0])<.33 and -.27<p[1]<.17 and 2.34<p[2]<2.50):continue
        if math.dist(c['pixel'],selected['pixel'])>90:continue
        options.append(c)
    if len(options)!=1:raise ValueError('empty-grasp reacquisition is not unique')
    return options[0]


def _current_release_observation(candidate,request_id,stable_id,release_id,released_at):
    """Unassociated class detections cannot prove a particular delivery."""
    return bool(all(isinstance(value,str) and value for value in (request_id,stable_id,release_id))
        and candidate.get('method')=='depth_foreground'
        and candidate.get('identity_source')=='verified_grasp_and_release'
        and candidate.get('request_id')==request_id and candidate.get('stable_id')==stable_id
        and candidate.get('release_id')==release_id
        and isinstance(candidate.get('released_at'),(int,float))
        and math.isfinite(candidate['released_at']) and abs(candidate['released_at']-released_at)<.05)


def _verify_transport_segment(history, category, side, released_at, drop_y=0., request_id=None,drop_x=.6,stable_id=None,release_id=None):
    """Use the release-time trajectory, even when the item has already left.

    Current unmodified USD has belt surface velocity along world +Y. The input
    world_position is measured from RGB-D and audited fixed camera extrinsics,
    not a simulator object-state feed. Low-confidence observations alone cannot
    initiate a grasp; here they need a coherent post-release belt trajectory.
    """
    sign=1 if side=='left' else -1
    track=[]
    for frame in history:
        if not released_at<=frame['observed_at']<=released_at+35:continue
        eligible=[]
        for c in frame.get('observations',frame.get('candidates',[])):
            p=c.get('world_position')
            if c['class']!=category or p is None:continue
            if not _current_release_observation(c,request_id,stable_id,release_id,released_at):continue
            if not all(math.isfinite(v) for v in p):continue
            if not (.43<sign*p[0]<1.15 and 2.22<p[2]<2.55):continue
            if not drop_y-.15<p[1]<1.4:continue
            if not track:
                if abs(p[0]-sign*drop_x)>.2 or p[1]>drop_y+.55:continue
                distance=abs(p[0]-sign*drop_x)+abs(p[1]-drop_y)
            else:
                dt=frame['stamp']-track[-1]['stamp']
                if dt<=0:continue
                previous=track[-1]['world_position']
                if abs(p[0]-track[0]['world_position'][0])>.08:continue
                dy=p[1]-previous[1]
                if dy<-.02 or dy>max(.12,dt*1.5):continue
                distance=abs(dy-.1*dt)+abs(p[0]-previous[0])
            eligible.append((distance,c))
        eligible.sort(key=lambda x:x[0])
        if not eligible:continue
        if len(eligible)>1 and eligible[1][0]-eligible[0][0]<.025:
            return None
        c=eligible[0][1]
        track.append({'stamp':frame['stamp'],'observed_at':frame['observed_at'],
                      'world_position':c['world_position'],'pixel':c['pixel'],
                      'request_id':request_id,'stable_id':stable_id,'release_id':release_id,'released_at':released_at,
                      'confidence':c.get('confidence'),'method':c.get('method','yolo'),'image_size':frame['image_size']})
    # Falling payload / low gripper fragments can precede belt contact. Require
    # a final track with stable height, rather than counting that drop as travel.
    settled=[];heights=[]
    for sample in reversed(track):
        height=sample['world_position'][2]
        if heights and max(max(heights),height)-min(min(heights),height)>.025:break
        settled.append(sample);heights.append(height)
    track=list(reversed(settled))
    if len(track)<3:return None
    elapsed=track[-1]['stamp']-track[0]['stamp']
    travel=track[-1]['world_position'][1]-track[0]['world_position'][1]
    if elapsed<=0 or travel<.08 or not .015<travel/elapsed<1.5:return None
    near_exit=track[-1]['pixel'][1]>track[-1]['image_size'][1]-45
    later=[f for f in history if f['observed_at']>track[-1]['observed_at']+.5]
    return {'class':category,'side':side,'samples':track,'travel_m':travel,
            'request_id':request_id,'stable_id':stable_id,'release_id':release_id,'released_at':released_at,
            'speed_m_s':travel/elapsed,'transport_seen':True,
            'left_observation_area':bool(near_exit and later)}


def verify_transport(history,category,side,released_at,drop_y=0.,request_id=None,drop_x=.6,stable_id=None,release_id=None):
    """Independent detector/track segments cannot be stitched into one proof."""
    def key(c):return c.get('method','yolo'),c.get('track_segment',0)
    keys={key(c) for f in history for c in f.get('observations',f.get('candidates',[]))
          if c.get('class')==category}
    for segment in sorted(keys,key=str):
        isolated=[{**f,'observations':[c for c in f.get('observations',f.get('candidates',[]))
                                     if key(c)==segment]} for f in history]
        proof=_verify_transport_segment(isolated,category,side,released_at,drop_y,request_id,drop_x,stable_id,release_id)
        if proof:
            proof['evidence_track']={'method':segment[0],'segment':segment[1]}
            return proof
    return None


def in_source_bin(candidate):
    p=candidate.get('world_position')
    return bool(p is not None and len(p)==3 and all(math.isfinite(v) for v in p)
                and abs(p[0])<.38 and -.32<p[1]<.22 and 2.365<p[2]<2.50)


def verify_settled_placement(history,category,side,released_at,drop_y,request_id,observed_after,stable_id=None,release_id=None):
    """A newly released item can remain on the belt instead of leaving the image.

    Caller separately verifies an empty gripper. Count decrease is diagnostic,
    never required. Same object/request/release and continuous segment only.
    """
    if not request_id or observed_after<released_at:return None
    sign=1 if side=='left' else -1
    stable=[]
    for frame in history:
        seen=frame['observed_at']
        if not max(released_at,observed_after)<=seen<=released_at+60:continue
        choices=[]
        for c in frame.get('observations',[]):
            p=c.get('world_position')
            if c['class']!=category or not _current_release_observation(c,request_id,stable_id,release_id,released_at):continue
            if p is None or not all(math.isfinite(v) for v in p):continue
            # Height band must admit everything the tracker can publish. The
            # producer (DepthConveyorTracker) keeps 2.22..2.49, so a narrower
            # consumer band silently made some detected payloads unverifiable:
            # a taller object pushed p[2] past 2.45 and no proof could ever form.
            if not (.52<sign*p[0]<1.05 and drop_y-.15<p[1]<drop_y+.8 and 2.22<p[2]<2.49):continue
            choices.append(c)
        if len(choices)!=1:stable=[];continue
        c=choices[0]
        sample={'stamp':frame['stamp'],'observed_at':seen,'world_position':c['world_position'],
                'request_id':request_id,'stable_id':stable_id,'release_id':release_id,'released_at':released_at,
                'pixel':c['pixel'],'method':'depth_foreground','confidence':None,'track_segment':c.get('track_segment',0)}
        if stable:
            if sample['track_segment']!=stable[-1]['track_segment']:stable=[]
        if stable:
            dt=sample['stamp']-stable[-1]['stamp']
            if dt<=0:continue
            positions=[r['world_position'] for r in stable]+[sample['world_position']]
            span=[max(p[j] for p in positions)-min(p[j] for p in positions) for j in range(3)]
            if dt>.5 or span[0]>.012 or span[1]>.012 or span[2]>.01:stable=[]
        stable.append(sample)
        if len(stable)>=6 and stable[-1]['stamp']-stable[0]['stamp']>=1.:
            return {'class':category,'side':side,'request_id':request_id,'stable_id':stable_id,
                    'release_id':release_id,'released_at':released_at,'settled_on_belt':True,
                    'verification':'stable_foreground_after_return_home','samples':stable}
    return None
