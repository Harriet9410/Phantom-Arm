"""RGB-D foreground proposals and scale-preserving rotation-aware classification.

Only camera data and fixed camera/basket calibration enter this module.  No
simulator object state, expected object counts, seed, or class inventory is used.
Previous frames retain only useful image rotation angles, never detections.
"""
import time
import cv2
import numpy as np
from observation_view import basket_mask,basket_roi,visibility_layers,normalized_xy
from perception_tracking import canonical_class,CLASS_ALIASES


def surface_samples(depth,x,y,distance,k,grid=4,per_cell=8):
    """Stratify measured body surface, including periphery but avoiding edges.

    Samples remain original optical pixels/depth, not grasp coordinates or
    simulator geometry. They permit trial-lift matching outside the gripper's
    central occlusion. A mixed/unknown component is still not grasp-ready.
    """
    ys,xs=np.nonzero(distance>=1.5)
    if not len(xs):return [],{'grid':[grid,grid],'intrinsics':list(k)}
    height,width=distance.shape
    cells_x=np.minimum(grid-1,(xs*grid//max(1,width)).astype(int))
    cells_y=np.minimum(grid-1,(ys*grid//max(1,height)).astype(int))
    rows=[]
    for cy in range(grid):
        for cx in range(grid):
            choices=np.flatnonzero((cells_x==cx)&(cells_y==cy))
            if not len(choices):continue
            take=choices[np.linspace(0,len(choices)-1,min(per_cell,len(choices)),dtype=int)]
            for index in take:
                u,v=x+int(xs[index]),y+int(ys[index]);z=float(depth[v,u])
                if np.isfinite(z) and .1<z<5.:rows.append([u,v,round(z,6)])
    return rows,{'grid':[grid,grid],'max_per_cell':per_cell,'intrinsics':[float(v) for v in k],
                 'coordinate_system':'optical_pixels_and_depth_m','source':'current_rgbd_body_surface',
                 'image_size':[int(depth.shape[1]),int(depth.shape[0])],
                 'edge_exclusion_pixels':1.5}


def source_components(depth, k, return_metadata=False):
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    footprint=basket_mask(k,depth.shape)
    visibility,excluded=visibility_layers(depth,k)
    valid=np.isfinite(depth) & (depth>.1) & (depth<5.)
    wx=-(xx-k[2])*depth/k[0]
    wy=(yy-k[5])*depth/k[4]+.1
    # Wall/rim exclusion uses each measured 3-D point, as in the V6 camera
    # calibration. Semantic quadrants use the fixed floor-projected ROI;
    # neither region is derived from the detected-object bounding envelope.
    inside = valid & (wx>-.335) & (wx<.322) & (wy>-.272) & (wy<.181)
    unknown=[]
    invalid=(footprint&~valid).astype(np.uint8)
    count_invalid,_,stats_invalid,_=cv2.connectedComponentsWithStats(invalid,8)
    for stat in stats_invalid[1:]:
        x,y,bw,bh,area=map(int,stat)
        if area>=9:unknown.append({'bbox':[x,y,x+bw,y+bh],'reason':'invalid_depth','area':area})
    unknown.extend(dict(r) for r in visibility['occluded_regions'])
    metadata={'unknown_regions':unknown,'visibility':visibility,'floor_estimation':'camera_histogram'}
    values = depth[inside]
    hist, edges = np.histogram(values, bins=np.arange(1.95,2.251,.001))
    if not len(values) or hist.max()<100:
        metadata['unknown_regions'].append({'bbox':basket_roi(k,depth.shape),'reason':'support_not_observable'})
        return ([],None,metadata) if return_metadata else ([],None)
    index = int(hist.argmax())
    floor_samples = values[(values>=edges[index]) & (values<edges[index+1])]
    floor = float(np.median(floor_samples))
    # Retain tall/stacked foreground. Height changes readiness, not existence.
    mask = (inside & ~excluded & (floor-depth>.004)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    proposals = []
    for index in range(1,count):
        x,y,bw,bh,area = map(int,stats[index])
        if area<9:continue
        body = (labels[y:y+bh,x:x+bw]==index).astype(np.uint8)
        heights=floor-depth[y:y+bh,x:x+bw][body.astype(bool)]
        too_large=area>10000 or max(bw,bh)*floor/min(k[0],k[4])>.155
        reasons=[]
        if area<40:reasons.append('small_unresolved_foreground')
        if too_large:reasons.append('merged_or_large_foreground')
        if float(np.percentile(heights,95))>.075:reasons.append('tall_or_stacked_geometry')
        distance = cv2.distanceTransform(np.pad(body,1),cv2.DIST_L2,5)[1:-1,1:-1]
        radius = float(distance.max())
        if radius<3:
            metadata['unknown_regions'].append({'bbox':[x,y,x+bw,y+bh],
                'reason':'thin_unresolved_foreground','area':area})
            continue
        # Thick body centre excludes slender handles and the grenade lever.
        core = distance>=.70*radius
        core_count,_,core_stats,_=cv2.connectedComponentsWithStats(core.astype(np.uint8),8)
        substantial_cores=sum(int(stat[4])>=9 for stat in core_stats[1:])
        if substantial_cores>1:reasons.append('multiple_body_cores')
        ys,xs = np.nonzero(core)
        weights = distance[core]
        u = x+float(np.average(xs,weights=weights))
        v = y+float(np.average(ys,weights=weights))
        z = float(np.median(depth[y:y+bh,x:x+bw][core]))
        # A less aggressive erosion retains the long body axis, removing thin
        # appendages.  Undirected axis has 180 degree periodicity.
        axis_y,axis_x = np.nonzero(distance>=.30*radius)
        points = np.column_stack((axis_x,axis_y)).astype(np.float64)
        eig, vectors = np.linalg.eigh(np.cov(points.T))
        principal = vectors[:,int(eig.argmax())]
        theta = float(np.degrees(np.arctan2(principal[1],principal[0])))
        ratio = float(eig.max()/max(eig.min(),1e-6))
        if ratio>4.:
            # A torch's wider head must not pull the grasp to one end. Select
            # the middle cross-section of the long body, then its jaw midpoint.
            projection=points@principal
            middle=float(sum(np.percentile(projection,[5,95]))/2.)
            section=abs(projection-middle)<=2.
            cross=points[section]
            if len(cross)>=5:
                u=x+float(cross[:,0].mean());v=y+float(cross[:,1].mean())
                pixels=np.rint(cross).astype(int)
                z=float(np.median(depth[y+pixels[:,1],x+pixels[:,0]]))
        angle = (theta-90.+90.)%180.-90.
        if ratio<1.20:
            # A nearly round body has no reliable yaw. Any cross-body direction
            # is equivalent; a fixed direction prevents noise-driven flips.
            angle = 0.
        xc,yc = (u-k[2])*z/k[0],(v-k[5])*z/k[4]
        sample_indices=np.linspace(0,len(axis_x)-1,min(32,len(axis_x)),dtype=int)
        signature=[[int(x+axis_x[i]),int(y+axis_y[i]),round(float(depth[y+axis_y[i],x+axis_x[i]]),5)] for i in sample_indices]
        surface_reference,surface_reference_meta=surface_samples(depth,x,y,distance,k)
        proposals.append({'bbox':[x,y,x+bw,y+bh], 'pixel':[u,v], 'depth':z,
            'depth_spread':float(np.ptp(depth[y:y+bh,x:x+bw][core])),
            'world_position':[-xc,yc+.1,4.5-z], 'support_z':4.5-floor,
            'angle_deg':angle, 'body_axis_deg':theta, 'axis_ratio':ratio,
            'grasp_point_method':'depth_body_center_and_axis', 'area':area,
            'body_mask':body,'depth_reference':signature,
            'surface_reference':surface_reference,'surface_reference_meta':surface_reference_meta,
            'normalized_xy':normalized_xy([u,v],basket_roi(k,depth.shape)),
            'height_m':float(np.percentile(heights,95)),
            'grasp_ready':not reasons,'grasp_uncertainty':reasons})
    answer=(sorted(proposals,key=lambda p:p['pixel'][0]),4.5-floor)
    return (*answer,metadata) if return_metadata else answer


def associate(box, proposals):
    x1,y1,x2,y2=box
    choices=[]
    for index,p in enumerate(proposals):
        u,v=p['pixel'];a,b,c,d=p['bbox']
        if not x1-3<=u<=x2+3 or not y1-3<=v<=y2+3:
            continue
        intersection=max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))
        union=(c-a)*(d-b)+(x2-x1)*(y2-y1)-intersection
        score=intersection/max(union,1.)
        if score>=.25:choices.append((score,index))
    choices.sort(reverse=True)
    if not choices or len(choices)>1 and choices[1][0]>.65*choices[0][0]:
        return None
    return choices[0][1]


class RotationDetector:
    def __init__(self,model):
        self.model=model
        self.preferred=[]
        names=list(model.names.values()) if isinstance(model.names,dict) else list(model.names)
        supported=sorted({canonical_class(name) for name in names if canonical_class(name)})
        self.capabilities={'declared_classes':supported,'raw_names':[str(n) for n in names],
            'missing_official_classes':[name for name in CLASS_ALIASES if name not in supported],
            'source':'loaded_weights.names','accuracy_verified':False}

    def detect(self,rgb,depth,k):
        began=time.monotonic()
        proposals,support,metadata=source_components(depth,k,return_metadata=True)
        h,w=depth.shape
        angles=[0.]+self.preferred
        for p in proposals:
            for shift in (0.,90.,180.,270.):
                angles.append(round((p['body_axis_deg']+shift)/15.)*15.%360.)
        angles+=list(range(0,360,15))
        schedule=[]
        for angle in angles:
            angle=float(angle)%360.
            if angle not in schedule:schedule.append(angle)
        found=[{} for _ in proposals]
        base=[];used=[]
        for angle in schedule:
            matrix=cv2.getRotationMatrix2D((w/2.,h*.38),angle,1.)
            view=rgb if angle==0 else cv2.warpAffine(rgb,matrix,(w,h),borderValue=(245,245,245))
            result=self.model.predict(view,conf=.20,iou=.4,imgsz=960,verbose=False)[0]
            inverse=cv2.invertAffineTransform(matrix)
            used.append(angle)
            view_matches=[[] for _ in proposals]
            for box in result.boxes:
                a,b,c,d=map(float,box.xyxy[0].cpu().tolist())
                corners=np.array([[a,b,1.],[c,b,1.],[c,d,1.],[a,d,1.]])@inverse.T
                lo=corners.min(axis=0);hi=corners.max(axis=0)
                bbox=[max(0,float(lo[0])),max(0,float(lo[1])),min(w,float(hi[0])),min(h,float(hi[1]))]
                raw_category=self.model.names[int(box.cls[0])]
                category=canonical_class(raw_category);confidence=float(box.conf[0])
                row={'class':category,'raw_class':str(raw_category),'confidence':confidence,
                     'bbox':bbox,'recognition_rotation_deg':angle}
                if angle==0:base.append(row)
                if category is not None and confidence>=.60:
                    for proposal_index,proposal in enumerate(proposals):
                        x1,y1,x2,y2=bbox;a0,b0,c0,d0=proposal['bbox']
                        area=max(0,min(c0,x2)-max(a0,x1))*max(0,min(d0,y2)-max(b0,y1))
                        if area>=.18*max(1,(c0-a0)*(d0-b0)) and area>=.40*max(1,(x2-x1)*(y2-y1)):
                            view_matches[proposal_index].append(row)
                index=associate(bbox,proposals)
                if index is not None:
                    label=category if category is not None else 'unmapped:'+str(raw_category)
                    old=found[index].get(label)
                    if old is None or confidence>old['confidence']:found[index][label]=row
            for proposal,matches in zip(proposals,view_matches):
                merged=False
                for i,left in enumerate(matches):
                    for right in matches[i+1:]:
                        a,b,c,d=left['bbox'];x1,y1,x2,y2=right['bbox']
                        overlap=max(0,min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))
                        iou=overlap/max(1,(c-a)*(d-b)+(x2-x1)*(y2-y1)-overlap)
                        if iou<.25 and np.hypot((a+c-x1-x2)/2.,(b+d-y1-y2)/2.)>8:
                            merged=True
                if merged and 'multiple_detector_instances' not in proposal['grasp_uncertainty']:
                    proposal['grasp_uncertainty'].append('multiple_detector_instances');proposal['grasp_ready']=False
            if not proposals or all(max((b['confidence'] for b in f.values()),default=0)>=.75 for f in found):
                break
        source=[];usefulness={}
        unknown=metadata['unknown_regions']
        for proposal,classes in zip(proposals,found):
            ordered=sorted(classes.values(),key=lambda b:b['confidence'],reverse=True)
            reason=None
            if not ordered:reason='unrecognized_foreground'
            elif ordered[0]['class'] is None:reason='unsupported_model_class'
            elif ordered[0]['confidence']<.60:reason='low_class_confidence'
            elif len(ordered)>1 and ordered[0]['confidence']-ordered[1]['confidence']<.10:reason='class_conflict'
            if reason:
                unknown.append({key:value for key,value in proposal.items() if key!='body_mask'})
                unknown[-1].update({'reason':reason,
                                    'class_hypotheses':[{'class':row['class'],'raw_class':row['raw_class'],'confidence':row['confidence']} for row in ordered],
                                    'grasp_ready':False})
                continue
            chosen=dict(ordered[0]);angle=chosen['recognition_rotation_deg']
            usefulness[angle]=usefulness.get(angle,0)+1
            chosen.update({key:value for key,value in proposal.items() if key!='body_mask'})
            if proposal['grasp_uncertainty']:
                unknown.append({'bbox':proposal['bbox'],'pixel':proposal['pixel'],
                    'normalized_xy':proposal['normalized_xy'],'class':chosen['class'],
                    'reason':proposal['grasp_uncertainty'][0],
                    'reasons':proposal['grasp_uncertainty'],'recognized_region':True})
            source.append(chosen)
        self.preferred=sorted(usefulness,key=lambda a:(-usefulness[a],used.index(a)))
        return source,base,{'views':used,'seconds':time.monotonic()-began,
            'foreground_count':len(proposals),'recognized_count':len(source),'support_z':support,
            'unknown_regions':unknown,'visibility':metadata['visibility'],
            'model_capabilities':self.capabilities,
            'coverage_complete':not unknown and metadata['visibility']['clear'],
            'coverage_scope':'visible_foreground_only; hidden fully occluded objects cannot be excluded'}
