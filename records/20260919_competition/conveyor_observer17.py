"""Track a released, previously identified item using measured depth foreground.

This does not classify an unknown blob. Identity is associated only after the
same object/request produced attempt -> grasp_verified -> release_started ->
released, with a unique release_id. Old contact-only events cannot activate it.
"""
import cv2
import numpy as np
import copy
import math


class DepthConveyorTracker:
    def __init__(self,reference_frames=12):
        self.reference_frames=reference_frames;self.references=[];self.background=None
        self.context={};self.active=None;self.last=None
        self.latest_depth=None
        self.segment=0
        self.reference_background=None;self.latest_components=[];self.latest_stamp=None
        self.diagnostics={'state':'not_armed'}
        self.preexisting_state=None
        self.round_id=None
        self.background_coverage=None;self.reference_coverages=[]
        self.last_reference_stamp=None

    def on_event(self,event):
        rid=event.get('request_id');status=event.get('status')
        if status=='round_started':
            if event.get('round_id') is None or event.get('round_id')==self.round_id:return
            self.round_id=event['round_id']
            self.references=[];self.background=None;self.reference_background=None
            self.context={};self.active=None;self.last=None;self.latest_depth=None
            self.latest_components=[];self.latest_stamp=None;self.diagnostics={'state':'round_reset'}
            self.preexisting_state=None
            self.background_coverage=None;self.reference_coverages=[]
            self.last_reference_stamp=None
            return
        if not rid:return
        if status=='attempt':
            # Keep the calibrated static background. A moving prior payload
            # must be explicitly excluded, not baked into a depth snapshot
            # that would later erase a new payload crossing its old pixels.
            if self.reference_background is not None:self.background=self.reference_background.copy()
            self.active=None;self.last=None
            candidate=event.get('candidate',{})
            self.context[rid]={'class':candidate.get('class'),'stable_id':candidate.get('stable_id'),'grasp':False}
            # Record other belt occupants while the tool is at observation
            # pose, before this grasp. Do not call the held target/tool at the
            # release pose a pre-existing item merely because it is visible.
            self.preexisting_state={'side':None,'preexisting':copy.deepcopy(self.latest_components),
                                    'provenance_uncertain':False}
        elif status=='grasp_verified' and rid in self.context:
            candidate=event.get('candidate',{});stable=event.get('stable_id') or candidate.get('stable_id')
            previous=self.context[rid]
            if (isinstance(stable,str) and stable and stable==previous.get('stable_id')
                    and (event.get('proof') or {}).get('verified') is True
                    and (event.get('category') or candidate.get('class'))==previous.get('class')):
                previous['grasp']=True
        elif status=='release_started':
            previous=self.context.get(rid,{})
            if (previous.get('grasp') and previous.get('class')==event.get('category') and event.get('side') in ('left','right')
                    and previous.get('stable_id')==event.get('stable_id') and isinstance(event.get('release_id'),str) and event['release_id']):
                self.active={'request_id':rid,'class':event['category'],'side':event['side'],
                             'stable_id':event['stable_id'],'release_id':event['release_id'],
                             'started':event['time'],'drop_y':event['drop_y'],
                             'drop_x':float(event.get('drop_x',.6)),'released':False,
                             'tool_clear_at':None,
                             'preexisting':copy.deepcopy((self.preexisting_state or {}).get('preexisting',[])),
                             'provenance_uncertain':bool((self.preexisting_state or {}).get('provenance_uncertain',True))}
                self.last=None;self.segment=0
                self.diagnostics={'state':'release_pending','release_id':event['release_id']}
        elif status=='released' and self.active and self.active['request_id']==rid:
            if event.get('stable_id')!=self.active['stable_id'] or event.get('release_id')!=self.active['release_id']:return
            released_at=event.get('released_at',event.get('time'))
            if not isinstance(released_at,(int,float)) or not math.isfinite(released_at) or released_at<self.active['started']:return
            self.active.update(released=True,released_at=released_at)
            self.diagnostics={'state':'released_awaiting_tool_clearance','release_id':event['release_id']}
        elif status=='arrived' and event.get('phase')=='place_retreat':
            active=self.active;at=event.get('time')
            if (active and active['released'] and active['request_id']==rid
                    and type(at) in (int,float) and math.isfinite(at) and at>=active['released_at']):
                active['tool_clear_at']=at
                self.diagnostics={'state':'tracking_current_release','release_id':active['release_id']}
        elif status in ('drop_detected','holding_feedback_lost','task_failed'):
            if rid in self.context:self.context[rid]['grasp']=False
            if self.active and self.active['request_id']==rid:self.active=None;self.last=None

    def _belt_components(self,depth,K,simulation_time):
        reference=self.reference_background
        if reference is None:return []
        h,w=depth.shape;yy,xx=np.indices(depth.shape,dtype=np.float32)
        wx=-(xx-K[2])*depth/K[0];wy=(yy-K[5])*depth/K[4]+.1;wz=4.5-depth
        valid=np.isfinite(depth)&np.isfinite(reference)&(depth>.1)
        high=cv2.dilate((valid&(wz>2.50)).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        mask=(valid&~high&(abs(wx)>.52)&(abs(wx)<1.05)&(wy>-.2)&(wy<1.4)
              &(wz>2.22)&(wz<2.49)&(reference-depth>.007)).astype(np.uint8)
        if self.background_coverage is not None:mask&=self.background_coverage.astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8);rows=[]
        for index in range(1,count):
            x,y,bw,bh,area=map(int,stats[index])
            if not 40<=area<=6000:continue
            region=labels==index
            rows.append({'world_position':[float(np.median(a[region])) for a in (wx,wy,wz)],
                         'bbox':[x,y,x+bw,y+bh],'stamp':simulation_time})
        return rows

    def _empty_belt_reference(self,depth,K):
        """Certify observed empty support regions, never absorb a static payload.

        Only the convex interior of a well-observed dominant belt plane is
        covered. Rails/outside/unobserved regions are not silently considered
        empty and cannot later supply delivery evidence. This check uses actual
        depth; no expected object count or simulator identities are involved.
        """
        yy,xx=np.indices(depth.shape,dtype=np.float32)
        valid=np.isfinite(depth)&(depth>.1)&(depth<5.)
        wx=-(xx-K[2])*depth/K[0];wy=(yy-K[5])*depth/K[4]+.1
        coverage=np.zeros(depth.shape,bool);checks=[]
        for side,sign in [('left',1),('right',-1)]:
            region=valid&(sign*wx>.52)&(sign*wx<1.05)&(wy>-.2)&(wy<1.4)
            values=depth[region&(depth>2.)&(depth<2.35)]
            histogram,edges=np.histogram(values,bins=np.arange(2.,2.351,.001))
            if len(values)<1000 or histogram.max()<500:
                return None,{'state':'background_support_unobservable','side':side}
            index=int(histogram.argmax());mode=(edges[index]+edges[index+1])/2.
            support=region&(abs(depth-mode)<.003)
            ys,xs=np.nonzero(support)
            if len(xs)<1000 or np.ptp(xs)<20 or np.ptp(ys)<40:
                return None,{'state':'background_support_unobservable','side':side}
            hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
            interior=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(interior,hull,1)
            interior=cv2.erode(interior,np.ones((5,5),np.uint8)).astype(bool)
            area=int(interior.sum())
            # A roller conveyor is corrugated: its common upper tangent plane
            # is visible in repeated strips, not at every pixel. Require that
            # measured support is spread over the belt, rather than demand a
            # mostly flat surface (which incorrectly rejects the real rollers).
            supported=interior&support;py,px=np.nonzero(supported)
            bx=np.minimum(3,((px-xs.min())*4/max(1,np.ptp(xs))).astype(int))
            by=np.minimum(7,((py-ys.min())*8/max(1,np.ptp(ys))).astype(int))
            cells=np.bincount(by*4+bx,minlength=32) if len(px) else np.zeros(32)
            supported_cells=int((cells>=max(10,.05*area/32.)).sum())
            if (area<max(1000,.60*len(values)) or supported.sum()<.25*area
                    or supported_cells<24 or (valid&interior).sum()<.98*area):
                return None,{'state':'background_support_or_visibility_insufficient','side':side}
            raised=(interior&valid&(mode-depth>.007)).astype(np.uint8)
            count,_,stats,_=cv2.connectedComponentsWithStats(raised,8)
            suspicious=[list(map(int,row)) for row in stats[1:] if int(row[4])>=40]
            if suspicious:
                return None,{'state':'background_contains_possible_payload','side':side,
                             'raised_components_xywh_area':suspicious,'support_depth':float(mode)}
            coverage|=interior
            checks.append({'side':side,'support_depth':float(mode),'covered_pixels':area,
                           'support_fraction':float((support&interior).sum()/area),
                           'support_grid_cells':supported_cells,'support_grid_total':32})
        return coverage,{'state':'observed_empty_support_regions','checks':checks,
                         'scope':'visible_plane_interiors_only_not_whole_belt_or_hidden_regions'}

    def _preexisting_exclusions(self,components,simulation_time,height):
        active=self.active or self.preexisting_state;excluded=[];kept=[]
        if active is None:return excluded
        sign=1 if active['side']=='left' else -1
        for old in active['preexisting']:
            p=old['world_position']
            if active['side'] is not None and not .52<sign*p[0]<1.05:continue
            dt=max(0.,simulation_time-old['stamp'])
            matches=[c for c in components if abs(c['world_position'][0]-p[0])<.08
                     and -.025<=c['world_position'][1]-p[1]<=max(.12,1.5*dt)]
            excluded.extend(matches)
            if len(matches)==1:kept.append(copy.deepcopy(matches[0]))
            elif len(matches)>1:
                active['provenance_uncertain']=True;kept.append(old)
            elif old['bbox'][3]<height-45:
                active['provenance_uncertain']=True;kept.append(old)
        active['preexisting']=kept
        return excluded

    def process(self,depth,K,wall_time,simulation_time):
        depth=np.asarray(depth,dtype=np.float32)
        self.latest_depth=depth.copy()
        self.latest_stamp=simulation_time
        if self.background is not None and self.background.shape!=depth.shape:
            self.background=None;self.references=[];self.active=None;self.last=None
            self.background_coverage=None;self.reference_coverages=[]
        if self.background is None:
            if self.last_reference_stamp is not None and simulation_time<=self.last_reference_stamp:
                self.diagnostics={'state':'background_requires_new_capture_frames'}
                return []
            self.last_reference_stamp=simulation_time
            coverage,proof=self._empty_belt_reference(depth,K)
            if coverage is None:
                self.references=[];self.reference_coverages=[];self.diagnostics=proof
                return []
            self.references.append(depth.copy())
            self.reference_coverages.append(coverage)
            if len(self.references)>=self.reference_frames:
                self.background=np.nanmedian(np.stack(self.references),axis=0)
                self.reference_background=self.background.copy()
                self.background_coverage=np.logical_and.reduce(self.reference_coverages)
                self.references=[]
                self.reference_coverages=[];self.diagnostics={**proof,'reference_frames':self.reference_frames}
            return []
        self.latest_components=self._belt_components(depth,K,simulation_time)
        excluded=self._preexisting_exclusions(self.latest_components,simulation_time,depth.shape[0])
        active=self.active
        if not active or not active['released'] or not active['released_at']<=wall_time<active['started']+35:return []
        # Low finger tips lie in the same height band as a falling payload.
        # Only establish an identity after the measured retreat completes and
        # a NEW camera exposure arrives; stale queued images are not clearance.
        if active['tool_clear_at'] is None or wall_time<active['tool_clear_at']:
            self.diagnostics={'state':'released_awaiting_tool_clearance','release_id':active['release_id']}
            return []
        if active['provenance_uncertain']:
            self.diagnostics={'state':active.get('provenance_failure','preexisting_object_provenance_ambiguous'),
                              'release_id':active['release_id']};return []
        if self.last is not None and simulation_time-self.last['stamp']>.6:
            self.diagnostics={'state':'release_track_continuity_lost','release_id':active['release_id']}
            active['provenance_uncertain']=True;active['provenance_failure']='release_track_continuity_lost';return []
        h,w=depth.shape
        xx=(np.arange(w,dtype=np.float32)[None,:]-K[2])/K[0]
        yy=(np.arange(h,dtype=np.float32)[:,None]-K[5])/K[4]
        wx=-xx*depth;wy=yy*depth+.1;wz=4.5-depth
        sign=1 if active['side']=='left' else -1
        finite=np.isfinite(depth)&np.isfinite(self.background)&(depth>.1)
        mask=finite&(sign*wx>.52)&(sign*wx<1.05)&(wy>active['drop_y']-.18)&(wy<1.4)
        mask&=(wz>2.22)&(wz<2.49)&((self.background-depth)>.007)
        if self.background_coverage is None:return []
        mask&=self.background_coverage
        # Reject pixels near the higher robot rather than calling them payload.
        high=(finite&(wz>2.50)).astype(np.uint8)
        high=cv2.dilate(high,np.ones((7,7),np.uint8))
        mask=(mask&~high.astype(bool)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8)
        options=[]
        for index in range(1,count):
            x,y,bw,bh,area=map(int,stats[index])
            if not 40<=area<=6000:continue
            region=labels==index
            p=[float(np.median(a[region])) for a in (wx,wy,wz)]
            if any(max(0,min(x+bw,old['bbox'][2]+3)-max(x,old['bbox'][0]-3)) *
                   max(0,min(y+bh,old['bbox'][3]+3)-max(y,old['bbox'][1]-3))>0 for old in excluded):continue
            if abs(p[0]-sign*active['drop_x'])>.2 or not active['drop_y']-.12<p[1]<active['drop_y']+.8:continue
            if self.last is not None:
                dt=simulation_time-self.last['stamp']
                if dt<=0:continue
                old=self.last['world_position'];dy=p[1]-old[1]
                if abs(p[0]-old[0])>.08 or dy<-.025 or dy>max(.12,dt*1.5):continue
                distance=abs(p[0]-old[0])+abs(dy-.15*dt)
            else:distance=abs(p[0]-sign*active['drop_x'])+abs(p[1]-active['drop_y'])
            observation={'class':active['class'],'confidence':None,'method':'depth_foreground',
                         'identity_source':'verified_grasp_and_release',
                         'stable_id':active['stable_id'],'release_id':active['release_id'],
                         'released_at':active['released_at'],
                         'request_id':active['request_id'],'track_segment':self.segment,'pixel':centers[index].tolist(),
                         'bbox':[x,y,x+bw,y+bh],'world_position':p,
                         'depth':float(np.median(depth[region])),'foreground_pixels':area}
            options.append((distance,observation))
        options.sort(key=lambda x:x[0])
        if not options:return []
        if len(options)>1 and (self.last is None or options[1][0]-options[0][0]<.03):return []
        observation=options[0][1]
        self.last={'stamp':simulation_time,'world_position':observation['world_position']}
        return [observation]
