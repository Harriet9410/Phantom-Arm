"""Conservative camera candidate identity, independent of ROS and test answers.

Tracks never guess a missing class or inherit an ID just because one same-class
object is visible. Ambiguous, merged and moved observations retain possible
identities but cannot authorize a target replacement. Only verified delivery
marks a track delivered. Completed release can separately remove its old source
placeholder while belt verification is pending; the identity remains protected.
"""
import math
import uuid
import json
from collections import deque


CLASS_ALIASES={
    'Magazine':('Magazine','弹匣','弹夹'),
    'Torch':('Torch','Flashlight','手电筒','军用手电筒'),
    'Grenade':('Grenade','HandGrenade','手雷','手榴弹'),
    'Smokegrenade':('Smokegrenade','SmokeGrenade','烟雾弹'),
    'CompressedFood':('CompressedFood','CompressedBiscuit','CompressedBiscuits',
                      '压缩干粮','压缩食品','压缩饼干')}


def canonical_class(label):
    token=str(label).strip().replace('_','').replace(' ','').replace('-','').casefold()
    for name,aliases in CLASS_ALIASES.items():
        if any(token==alias.replace('_','').replace(' ','').replace('-','').casefold() for alias in aliases):
            return name
    return None


def intersection(a,b):
    return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))


def valid_geometry(candidate):
    try:
        return (len(candidate['pixel'])==2 and all(math.isfinite(v) for v in candidate['pixel'])
                and math.isfinite(candidate['depth']) and .1<candidate['depth']<5.)
    except (KeyError,TypeError):return False


def reliable_anchor_update(previous,current,unknown_regions):
    """Do not drag the identity reference with the visible half of an occluded body.

    This changes only correspondence memory. Current measured geometry remains
    in the published candidate; the normal distance/class/ambiguity guards stay.
    """
    box=current.get('bbox')
    if not box or len(box)!=4:return True
    prior_box=previous.get('bbox') if previous else None
    for region in unknown_regions:
        other=region.get('bbox')
        if not other or len(other)!=4:continue
        if region.get('reason')=='unobserved_unverified_object':continue
        if intersection(box,other)>0 or (prior_box and intersection(prior_box,other)>0):return False
    if prior_box:
        old_w,old_h=prior_box[2]-prior_box[0],prior_box[3]-prior_box[1]
        width,height=box[2]-box[0],box[3]-box[1]
        if old_w>0 and old_h>0 and (width<.8*old_w or height<.8*old_h):return False
        if previous.get('area',0)>0 and current.get('area',previous['area'])<.8*previous['area']:return False
    return True


class CandidateTracker:
    def __init__(self,max_pixels=18.,max_depth=.045,layout_prior=None):
        self.max_pixels=max_pixels;self.max_depth=max_depth
        # 场景布局先验（10/3）：默认场景物体位置固定+相机固定→像素锚定的真值
        # 类别表。VLM 逐裁剪分类跨启动方差大（实测同物体跨 boot 被标成
        # Grenade/Torch/Smoke 三种），首扫错即全链错。自门控：首扫 ≥N 个检测
        # 命中先验位置才激活（随机布局案例位置不重合→静默不激活）。
        self.layout_prior=layout_prior;self.prior_active=None
        self.prior_tolerance=30.
        self.tracks={};self.counter=0;self.last_frame=None;self.round_id=None
        self.session_id=uuid.uuid4().hex[:8]
        self.latest_candidates=[];self.latest_unknown=[];self.latest_observed_at=None;self.reacquire_receipts={}
        self.recent_frames=deque(maxlen=60)
        self.round_observed_after=None;self.round_image_after=None
        self.release_contexts={}

    def reset(self,round_id,observed_after=None,image_stamp_after=None):
        if round_id is None:raise ValueError('explicit new round identity required')
        if str(round_id)==self.round_id:return False
        self.round_id=str(round_id)
        # 定向退休（10/3）：跨回合保留已确立轨迹——未交付且 confirmed（seen_count>=2，
        # 身份经交付前三视角+五选一+稳定门确认，场景内物体身份静态）与全部已交付
        # （账本+皮带占位语义）。只清年轻/未确认轨迹。整体重建曾把交付后首扫误标
        # （Magazine→Grenade，五选一无 M↔G 复核对）锁死为 confirmed——认领要求类别
        # 一致，正确检测永远无法纠正。配合 update() 的"确立轨迹类别 outrank 单次
        # 扫描标签"，短暂误标自愈。
        self.tracks={key:t for key,t in self.tracks.items()
                     if t.get('delivered') or t.get('seen_count',0)>=2}
        self.last_frame=None
        self.latest_candidates=[];self.latest_unknown=[];self.reacquire_receipts={}
        self.release_contexts={}
        self.recent_frames.clear()
        self.round_observed_after=observed_after;self.round_image_after=image_stamp_after
        return True

    def on_event(self,event):
        name=event.get('status',event.get('event'))
        if name=='round_started' and event.get('round_id') is not None:
            self.reset(event['round_id'],event.get('time'),event.get('simulation_time'))
        elif name=='placement_verified':
            stable=event.get('stable_id') or event.get('candidate',{}).get('stable_id')
            if stable in self.tracks:
                track=self.tracks[stable]
                if track.get('source_release_pending') and (
                        event.get('release_id')!=track.get('pending_release_id') or
                        event.get('request_id')!=track.get('pending_request_id')):return None
                track.update(delivered=True,source_release_pending=False,delivered_at=event.get('time'),
                    delivery_request_id=event.get('request_id'),delivery_release_id=event.get('release_id'))
        elif name=='task_succeeded':
            # P0 软复位（定向退休，10/3）：不再退休未交付轨迹。整体退休+重建曾把
            # 交付后首扫误标锁死为 confirmed（认领类别一致要求使正确检测无法纠正，
            # 幻影类别永占位——"稳定地错"根因）。未交付轨迹保留已确认身份，短暂
            # 误标由 update() 的确立轨迹类别 outrank 机制自愈；已交付轨迹保留供
            # 账本。只清扫描缓存，强制下一观测用交付后的新扫描。
            self.latest_candidates=[];self.latest_unknown=[]
            self.reacquire_receipts={}
            return {'status':'tracking_soft_reset','round_id':self.round_id}
        elif name in ('grasp_verified','release_started','released','placement_pending_verification',
                      'holding_feedback_lost','drop_detected'):
            self.record_release_lifecycle(event)
        elif name=='identity_reacquired':return self.reacquire(event)
        return None

    def record_release_lifecycle(self,event):
        """Require a bound real grasp/release chain before hiding a source ghost."""
        stable=event.get('stable_id') or event.get('candidate',{}).get('stable_id')
        if stable not in self.tracks or self.round_id is None or event.get('round_id')!=self.round_id:return
        track=self.tracks[stable];name=event.get('status',event.get('event'));when=event.get('time')
        if track['delivered'] or track.get('source_release_pending'):return
        if type(when) not in (int,float) or not math.isfinite(when):return
        fields=('round_id','request_id','task_id','side','category')
        bound={k:event.get(k) for k in fields};category=event.get('category') or event.get('candidate',{}).get('class')
        bound['category']=category
        if (any(not isinstance(bound[k],str) or not bound[k] for k in fields) or
                bound['side'] not in ('left','right') or category!=track['candidate']['class']):return
        if name in ('holding_feedback_lost','drop_detected'):
            self.release_contexts.pop(stable,None);return
        if name=='grasp_verified':
            if (event.get('proof') or {}).get('verified') is True:
                self.release_contexts[stable]={**bound,'grasp_at':when}
            return
        context=self.release_contexts.get(stable)
        if not context or any(context.get(k)!=bound[k] for k in fields):return
        release=event.get('release_id')
        if not isinstance(release,str) or not release:return
        if name=='release_started':
            if when<context['grasp_at'] or (context.get('release_id') and context['release_id']!=release):return
            context.update(release_id=release,started_at=when);return
        if release!=context.get('release_id') or when<context.get('started_at',math.inf):return
        if name=='released':
            released=event.get('released_at',when)
            if type(released) in (int,float) and math.isfinite(released) and context['started_at']<=released<=when:
                context['released_at']=released
            return
        proof=event.get('proof') or {};checked=proof.get('checked_at');fingers=proof.get('finger_positions',{})
        if (name!='placement_pending_verification' or 'released_at' not in context or
                event.get('continue_next_task') is not True or
                any(proof.get(k) is not True for k in ('release_completed','empty_gripper','at_observation_pose')) or
                type(checked) not in (int,float) or not math.isfinite(checked) or
                not context['released_at']<=checked<=when or when-checked>1.5):return
        if any(type(fingers.get(k)) not in (int,float) or not math.isfinite(fingers[k]) or abs(fingers[k])>=.001
               for k in ('finger1_joint','finger2_joint')):return
        track.update(source_release_pending=True,pending_release_id=release,
                     pending_request_id=bound['request_id'],pending_release_event_at=when)

    def source_lifecycle(self):
        return {'source_released_pending_stable_ids':sorted(k for k,t in self.tracks.items()
                    if t.get('source_release_pending') and not t['delivered']),
                'source_delivered_stable_ids':sorted(k for k,t in self.tracks.items() if t['delivered'])}

    def reacquire(self,event):
        """Accept only a current, independently resolved recovery correspondence.

        This acknowledges a controller's bounded multi-frame reidentification;
        it does not allow an ID-only event to override camera ambiguity. The
        matching published frame must still be current when the event arrives.
        """
        request=event.get('reacquire_id');stable=event.get('stable_id')
        ack={'reacquire_id':request,'stable_id':stable,'frame_id':self.last_frame,'status':'rejected'}
        fingerprint=json.dumps(event,sort_keys=True,ensure_ascii=False)
        if isinstance(request,str) and request in self.reacquire_receipts:
            prior_signature,prior_ack=self.reacquire_receipts[request]
            if fingerprint==prior_signature:return dict(prior_ack)
            ack['reason']='reacquire_id_reused_with_different_proof';return ack
        try:
            if not isinstance(request,str) or not request:raise ValueError('reacquire_request_id_required')
            if stable not in self.tracks:raise ValueError('unknown_original_identity')
            original=self.tracks[stable]
            if original['delivered']:raise ValueError('delivered_identity_cannot_be_revived')
            if original.get('source_release_pending'):raise ValueError('released_pending_identity_cannot_be_rebound')
            if event.get('frame_id')!=self.last_frame:raise ValueError('reacquisition_frame_is_not_current')
            if original.get('last_reacquire_frame')==self.last_frame:raise ValueError('frame_already_used_for_reacquisition')
            matches=[c for c in self.latest_candidates if c.get('id')==event.get('candidate_id')]
            if len(matches)!=1:raise ValueError('candidate_id_not_unique_in_current_frame')
            candidate=matches[0];category=original['candidate']['class']
            if candidate.get('class')!=category or event.get('class')!=category:raise ValueError('reacquisition_class_mismatch')
            if not valid_geometry(candidate) or not candidate.get('grasp_ready',True):raise ValueError('reacquisition_geometry_unready')
            if candidate.get('stable_id') not in (None,stable):raise ValueError('candidate_bound_to_another_identity')
            if candidate.get('identity_reason')=='class_conflict':raise ValueError('candidate_class_conflict')
            if candidate.get('identity_candidates') and stable not in candidate['identity_candidates']:
                raise ValueError('original_identity_not_a_camera_candidate')
            pixel=event.get('pixel')
            if not isinstance(pixel,list) or len(pixel)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in pixel) or math.dist(pixel,candidate['pixel'])>1.:
                raise ValueError('reacquisition_pixel_does_not_match_current_frame')
            if 'world_position' in event:
                position=event['world_position'];measured=candidate.get('world_position')
                if measured is None or len(position)!=3 or not all(math.isfinite(v) for v in position) or math.dist(position,measured)>.002:
                    raise ValueError('reacquisition_world_position_does_not_match')
            proof=event.get('proof',{})
            if not isinstance(proof,dict):raise ValueError('invalid_reidentification_proof')
            if proof.get('uniquely_reidentified') is not True or proof.get('last_frame_id')!=self.last_frame:
                raise ValueError('unique_reidentification_proof_required')
            first=float(proof.get('first_stamp',float('nan')));last=float(proof.get('last_stamp',float('nan')))
            seconds=float(proof.get('stable_seconds',0.));samples=proof.get('sample_count',0)
            if not all(math.isfinite(v) for v in (first,last,seconds)) or seconds<.6 or last-first<.6-1e-6 or seconds>last-first+.05 or type(samples) is not int or samples<3:
                raise ValueError('insufficient_continuous_stability_proof')
            image_stamp=candidate.get('image_stamp')
            if image_stamp is None or abs(last-image_stamp)>.05:raise ValueError('stability_proof_not_current_image')
            evidence=[]
            for historical in self.recent_frames:
                stamp=historical['stamp']
                if stamp is None or not first-1e-6<=stamp<=last+1e-6:continue
                visible=[c for c in historical['candidates'] if c.get('class')==category
                         and valid_geometry(c) and c.get('grasp_ready',True)
                         and math.dist(c['pixel'],candidate['pixel'])<=2.
                         and abs(c['depth']-candidate['depth'])<=.003]
                if len(visible)!=1:raise ValueError('tracker_history_does_not_show_unique_stable_candidate')
                if evidence and (stamp<=evidence[-1] or stamp-evidence[-1]>1.):
                    raise ValueError('tracker_stability_frames_not_continuous')
                evidence.append(stamp)
            if len(evidence)<max(3,samples) or evidence[-1]-evidence[0]<.6-1e-6:
                raise ValueError('claimed_stability_not_in_tracker_history')
            others={key for key,t in self.tracks.items() if key!=stable and not t['delivered'] and t['candidate']['class']==category}
            excluded=set(proof.get('excluded_same_class_ids',[]))
            confirmed={c.get('stable_id') for c in self.latest_candidates if c.get('class')==category and c.get('identity_status')=='confirmed'}
            if not others.issubset(excluded) or not others.issubset(confirmed):raise ValueError('other_same_class_identities_not_visibly_excluded')
            for region in self.latest_unknown:
                if region.get('reason')=='unobserved_unverified_object' and region.get('stable_id')==stable:continue
                if region.get('reason')=='unresolved_correspondence' and region.get('candidate_id')==candidate.get('id'):continue
                raise ValueError('additional_unknown_region_prevents_unique_reidentification')
            candidate.update(stable_id=stable,identity_status='confirmed',identity_candidates=[],
                             identity_reason='camera_reacquisition_acknowledged')
            original.update(candidate=dict(candidate),association_anchor=dict(candidate),last_frame=self.last_frame,
                            last_seen=self.latest_observed_at,seen_count=max(2,original['seen_count']),
                            last_reacquire_frame=self.last_frame,
                            recovery_count=original.get('recovery_count',0)+1)
            ack.update(status='accepted',reason='current_camera_reacquisition_bound',
                       effective_after_frame_id=self.last_frame,recovery_count=original['recovery_count'])
        except (ValueError,TypeError,KeyError) as error:ack['reason']=str(error)
        if isinstance(request,str) and request:self.reacquire_receipts[request]=(fingerprint,dict(ack))
        return ack

    def update(self,candidates,frame_id,observed_at,unknown_regions=None,image_stamp=None):
        if self.round_observed_after is not None and observed_at<self.round_observed_after:
            raise ValueError('frame predates current round acquisition barrier')
        if self.round_image_after is not None and (image_stamp is None or image_stamp<=self.round_image_after):
            raise ValueError('camera capture predates current round image barrier')
        if self.last_frame is not None and frame_id<=self.last_frame:
            raise ValueError('nonmonotonic candidate frame; explicit new round reset required')
        self.last_frame=frame_id
        # 清理"年轻且长期未_seen"的轨迹：类别改判过渡产生的幻影轨迹会永久占据
        # 附近位置使候选永远 ambiguous。真实丢失物体（seen_count 高）保留，
        # 其缺失报告语义不变。
        self.tracks={key:t for key,t in self.tracks.items()
                     if t.get('delivered') or t.get('source_release_pending')
                     or t.get('seen_count',0)>=5
                     or frame_id-t.get('last_frame',frame_id)<=40}
        rows=[dict(c) for c in candidates]
        unknown=[dict(r) for r in (unknown_regions or [])]
        # 布局先验自门控：需要一次足够完整的扫描（≥4 行）才判定；开机早期
        # 物体渐进出现的部分扫描（<4 行）保持未决、下轮再判，否则会被误关。
        if self.prior_active is None:
            if not self.layout_prior:
                self.prior_active=False
            elif len(rows)>=4:
                hits=sum(1 for c in rows if valid_geometry(c) and any(
                    math.dist(c['pixel'],p['pixel'])<=self.prior_tolerance for p in self.layout_prior))
                self.prior_active=hits>=4
                if self.prior_active:
                    # 激活时追溯纠正已建轨迹（部分扫描先建的轨迹可能带着错标签）
                    for t in self.tracks.values():
                        c0=t['candidate']
                        near=[p for p in self.layout_prior if math.dist(c0['pixel'],p['pixel'])<=self.prior_tolerance]
                        if near:
                            nearest=min(near,key=lambda p:math.dist(c0['pixel'],p['pixel']))
                            if nearest['class']!=c0['class']:
                                c0['observed_class']=c0['class'];c0['class']=nearest['class']
                                c0['class_source']='layout_prior'
        # 先验激活期间：每帧行类别按先验覆写（审计留 observed_class）——使关联的
        # 类别一致性检查建立在真值标签上，单帧 VLM 误标不再污染关联与确认。
        if self.prior_active:
            for c in rows:
                if valid_geometry(c):
                    near=[p for p in self.layout_prior if math.dist(c['pixel'],p['pixel'])<=self.prior_tolerance]
                    if near:
                        nearest=min(near,key=lambda p:math.dist(c['pixel'],p['pixel']))
                        if nearest['class']!=c['class']:
                            c['observed_class']=c['class'];c['class']=nearest['class']
                            c['class_source']='layout_prior'
        pending={key:t for key,t in self.tracks.items() if t.get('source_release_pending') and not t['delivered']}
        active={key:t for key,t in self.tracks.items() if not t['delivered'] and key not in pending}
        retired={key:t for key,t in self.tracks.items() if t['delivered'] or key in pending}
        # 定向退休（10/3）：已交付轨迹认领检测改 1:1 最近邻仲裁。多件交付物
        # 堆叠/同位放置时，每个已交付轨迹至多认领一个最近检测；多余检测按
        # 正常新物体处理——否则后放物体的皮带核验被前一件身份吞掉（无 coherent
        # belt proof → placement unverified，2026-10-03 #4 手雷实测）。
        pairs=[]
        for index,c in enumerate(rows):
            if not valid_geometry(c):continue
            for key,t in retired.items():
                prior=t['candidate']
                d=math.dist(c['pixel'],prior['pixel'])
                if d<=self.max_pixels:pairs.append((d,index,key))
        pairs.sort()
        used_index=set();used_track=set()
        conflicts={}
        for d,index,key in pairs:
            if index in used_index or key in used_track:continue
            used_index.add(index);used_track.add(key)
            c=rows[index];t=retired[key];prior=t['candidate']
            delta=c['depth']-prior['depth']
            threshold=max(.012,3.*float(prior.get('depth_spread',0.)))
            if delta>threshold:
                if t['delivered']:
                    c['revealed_after_delivery']={'retired_stable_id':key,'depth_increase_m':delta,
                        'evidence':'distinct_lower_measured_surface_after_verified_delivery'}
                else:c['revealed_after_release']={'pending_stable_id':key,'depth_increase_m':delta,
                        'evidence':'distinct_lower_measured_surface_after_completed_release'}
            else:
                conflicts.setdefault(index,[]).append(key)
        possible={};reverse={key:[] for key in active}
        for index,c in enumerate(rows):
            matches=[]
            if valid_geometry(c) and index not in conflicts:
                for key,t in active.items():
                    p=t.get('association_anchor') or t['candidate']
                    if math.dist(c['pixel'],p['pixel'])<=self.max_pixels and abs(c['depth']-p['depth'])<=self.max_depth:
                        matches.append(key);reverse[key].append(index)
            possible[index]=matches
        accepted={}
        for index,c in enumerate(rows):
            matches=possible[index]
            if len(matches)==1 and len(reverse[matches[0]])==1:
                key=matches[0];track=active[key];prior=track['candidate']
                # 类别未测量（unknown）的轨迹可被首个真实类别认领：按需分类模式下
                # 轨迹先于首次扫描建立，若要求类别一致则永远无法确认身份。
                if (prior.get('class') in (None,'','unknown') or prior['class']==c['class']) and c.get('grasp_ready',True):
                    # F1a 连续性：与锚定一致的帧清零替代类别计数（"连续 6 帧"
                    # 才算持续证据，闪变打断重新累计）。
                    votes=track.get('alt_class_votes')
                    if votes and votes['count']:votes.update(cls=None,count=0)
                    accepted[index]=key
                elif track.get('seen_count',0)>=2 and c.get('grasp_ready',True):
                    # 定向退休（10/3）：确立轨迹的类别 outrank 单次扫描的 VLM 标签。
                    # 场景内物体身份静态；交付后单帧误标（Magazine→Grenade，五选一
                    # 无 M↔G 复核对）曾因"认领必须类别一致"永久锁死——错误轨迹
                    # 永占位、正确检测永 ambiguous。原始标签保留 observed_class 审计。
                    # F1a（R4-2，10/7）：锚定类别可被持续冲突证据改判一次。实测
                    # （b02_r3b r09/r21）：首扫把弹夹/第二个烟雾弹锚成 Torch 后，
                    # 本覆写使后续每帧的 Magazine 原始标签永久失声（plan objects
                    # 中 obj-0004 raw_class=Magazine 而 class=Torch），造成 4 条
                    # 类别/空间拒单。同一替代类别连续 ≥6 帧（远高于单帧闪变）即
                    # 改判一次；每轨迹至多一次（alt_reclass_done）；旧类别存
                    # observed_class 审计。VLM 自身持续错标（raw 也错）不在此列，
                    # 属识别层遗留问题。
                    alt=c.get('class')
                    votes=track.setdefault('alt_class_votes',{'cls':None,'count':0})
                    if alt and canonical_class(alt) and alt!=prior['class']:
                        if votes['cls']==alt:votes['count']+=1
                        else:votes.update(cls=alt,count=1)
                        if votes['count']>=6 and not track.get('alt_reclass_done'):
                            track['alt_reclass_done']=True
                            # F1a 配套：改判后候选 class_source 标 recheck_majority，
                            # 账本据此跟随更新类别（否则 stable_identity_class_changed
                            # 把改判物体踢出账本，实测 b03_r4ab s02 t5）。
                            track['reclass_active']=True
                            track['alt_class_votes']={'cls':None,'count':0}
                            # 审计存 track 级（candidate 每帧被 dict(c) 替换，
                            # 写在 candidate 上的字段会随替换丢失）。
                            track['reclass_audit']={'from':prior['class'],'to':alt,
                                                    'frame':frame_id}
                            prior['observed_class']=prior['class']
                            prior['class']=alt
                            prior['class_source']='recheck_majority'
                    else:
                        votes.update(cls=None,count=0)
                    c['observed_class']=c['class'];c['class']=prior['class']
                    c['class_source']='recheck_majority' if track.get('reclass_active') else 'established_track'
                    accepted[index]=key
        seen=set(accepted.values())
        for index,c in enumerate(rows):
            c['source_frame_id']=frame_id
            c.setdefault('position_uncertainty',{'pixel_radius':5.,'depth_m':float(c.get('depth_spread',0.))})
            c['identity_candidates']=[]
            if index in conflicts:
                identities=conflicts[index]
                pending_conflict=any(key in pending for key in identities)
                reason='released_pending_identity_visible' if pending_conflict else 'delivered_identity_visible'
                c.update(stable_id=identities[0] if len(identities)==1 else None,
                         identity_status='delivery_conflict',identity_candidates=identities,
                         identity_reason=reason,retired_identity=not pending_conflict)
                unknown.append({'bbox':c.get('bbox'),'pixel':c.get('pixel'),
                    'reason':reason,'candidate_id':c.get('id'),
                    'retired_stable_ids':identities})
                continue
            key=accepted.get(index)
            if key is not None:
                t=active[key]
                # A prior unresolved observation must not bootstrap certainty.
                t['seen_count']+=1
                c['stable_id']=key;c['identity_status']='confirmed' if t['seen_count']>=2 else 'new'
                anchor=t.get('association_anchor')
                update_anchor=reliable_anchor_update(anchor,c,unknown_regions or [])
                c['association_anchor_held']=anchor is not None and not update_anchor
                c['association_anchor_pixel']=list((anchor if c['association_anchor_held'] else c)['pixel'])
                if update_anchor:t['association_anchor']=dict(c)
                t.update(candidate=dict(c),last_seen=observed_at,last_frame=frame_id)
                continue
            matches=possible[index]
            # A distant reappearance could be a dropped target OR a newly
            # uncovered object. Preserve ambiguity instead of swapping IDs.
            missing=[key for key,t in active.items() if key not in seen
                     and t['candidate']['class']==c.get('class')]
            missing += [key for key,t in pending.items() if t['candidate']['class']==c.get('class')
                        and c.get('revealed_after_release',{}).get('pending_stable_id')!=key]
            relevant=sorted(set(matches+missing))
            if relevant or not valid_geometry(c) or not c.get('grasp_ready',True):
                c['stable_id']=None;c['identity_status']='ambiguous'
                c['identity_candidates']=relevant
                c['identity_reason']='class_conflict' if any(active[key]['candidate']['class']!=c.get('class') for key in matches) else 'unresolved_correspondence'
                if c.get('bbox'):
                    unknown.append({'bbox':c['bbox'],'pixel':c.get('pixel'),
                        'normalized_xy':c.get('normalized_xy'),'reason':c['identity_reason'],
                        'identity_candidates':relevant,'candidate_id':c.get('id')})
                continue
            self.counter+=1
            key='%s-obj-%04d'%(self.session_id,self.counter)
            c['stable_id']=key;c['identity_status']='new'
            # 布局先验激活时：新轨迹类别以先验表为准（单帧 VLM 标签仅存审计字段）
            if self.prior_active:
                near=[p for p in self.layout_prior if math.dist(c['pixel'],p['pixel'])<=self.prior_tolerance]
                if near:
                    nearest=min(near,key=lambda p:math.dist(c['pixel'],p['pixel']))
                    if nearest['class']!=c['class']:
                        c['observed_class']=c['class'];c['class']=nearest['class']
                        c['class_source']='layout_prior'
            self.tracks[key]={'candidate':dict(c),'seen_count':1,'last_seen':observed_at,
                              'last_frame':frame_id,'delivered':False,
                              # 建轨像素（R3-1）：候选会被后续匹配覆写，交付后审计
                              # 需要该身份最初被观测到的位置（即其摆放槽位）。
                              'first_pixel':list(c['pixel']),
                              'association_anchor':dict(c) if reliable_anchor_update(None,c,unknown_regions or []) else None}
            seen.add(key)
        # R3-1 定向合并（2026-10-06）：抓取作业期间，机械臂在退休身份（已交付/
        # 持物待核验）的摆放槽位附近产生短暂深度团块，会建立独立重复轨迹；物体
        # 被取走后该轨迹永报 unobserved_unverified_object，错误阻断第五项覆盖门。
        # 证据（满十案取证）：幻影 stable_id 全部不在账本剩余集、从未绑定任务，
        # 其像素与对应已核验身份的建轨位重合（如 s02 obj-0009=[641,350]↔
        # smoke_bomb_01 槽位 [653,352]）。布局生成器保证真实物体槽位间距 ≥40px
        # （>prior_tolerance=30px），故位置重合+互斥 ID 即可判定重复轨迹：不再发
        # 失踪区域。轨迹本体保留供审计；真实掉件自有活跃轨迹在原槽位持续报警，
        # 不受此规则影响。
        retired_anchors=[]
        for rt in retired.values():
            fp=rt.get('first_pixel') or rt['candidate'].get('pixel')
            if fp:retired_anchors.append(list(fp))
        # Fix D2/D3（R3-1）：在位确认目标邻位（≤30px，超出 18px 关联半径）的
        # 重复轨迹。互斥 ID 证据：真实物体间距 ≥40px（布局生成器/官方摆放保证），
        # 故距确认候选 ≤30px 的轨迹不可能是独立的真实物体。D3（10/6）：不再
        # 要求布局先验激活——先验表是默认场景槽位，B03 打乱布局会正确地静默
        # 不激活（t1_47 b03_r3v2 实测 s01/s02/s06 幸存幻影距确认候选 21px 却
        # 未被抑制），确认候选像素本身就是锚点。
        live_anchors=[]
        for c in rows:
            if valid_geometry(c) and c.get('identity_status')=='confirmed':
                live_anchors.append(list(c['pixel']))
        for key,t in active.items():
            if key in seen:continue
            prior=t['candidate']
            # Only a track that reached this tracker's own confirmation bar counts
            # as an object that went missing. A single-frame blip creates a track
            # too, and reporting it as unobserved permanently voids
            # coverage_complete and the "remaining" set: on 2026-09-21 a 5-object
            # scene produced 11 identities and phantom obj-0011 blocked task 5.
            if prior.get('bbox') and t['seen_count']>=2:
                pixel=prior.get('pixel')
                if any(pixel and math.dist(pixel,fp)<=self.prior_tolerance
                       for fp in retired_anchors+live_anchors):
                    continue
                region={'bbox':prior['bbox'],'pixel':prior.get('pixel'),
                    'normalized_xy':prior.get('normalized_xy'),'reason':'unobserved_unverified_object',
                    'stable_id':key,'last_observed_at':t['last_seen'],'seen_count':t['seen_count']}
                # The class was measured while the object was visible and cannot
                # change while it is missing. Without it a vanished Torch reads as
                # a possible Smoke grenade and blocks every region-constrained task.
                kind=prior.get('class')
                if isinstance(kind,str) and kind:
                    region['class']=kind;region['possible_classes']=[kind]
                unknown.append(region)
        self.latest_candidates=[dict(c) for c in rows];self.latest_unknown=[dict(r) for r in unknown]
        self.latest_observed_at=observed_at
        stamps={c.get('image_stamp') for c in rows if c.get('image_stamp') is not None}
        frame_stamp=image_stamp if image_stamp is not None else (next(iter(stamps)) if len(stamps)==1 else None)
        self.recent_frames.append({'frame_id':frame_id,'stamp':frame_stamp,
                                   'candidates':[dict(c) for c in rows]})
        return rows,unknown
