"""Task identity, one competition clock and evidence-bounded local scoring.

No ROS, simulator truth, object manipulation, or language-model replacement.
The ledger checks evidence association; callers still validate spatial semantics
and physical proofs.  Its local estimate never substitutes for referee scoring.
"""
import copy
import hashlib
import json
import math
import re
import time


OFFICIAL_EXAMPLE_INSTRUCTIONS=(
    '抓取左上方的烟雾弹，放到左侧传送带',
    '抓取右下方的弹夹，放到右侧传送带',
    '抓取最左方的军用手电筒，放到左侧传送带',
    '抓取手雷，放到右侧传送带',
    '抓取剩余的物品，放到左侧传送带')
OFFICIAL_FULL_POINTS=(15,15,15,20,15)
OFFICIAL_CLASSES=('Smokegrenade','Magazine','Torch','Grenade',None)


def _number(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(label+' must be a finite number')
    return float(value)


def _instruction(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('empty instruction')
    return re.sub(r'\s+','',value).rstrip('。.!！')


def _remaining_instruction(value):
    return any(word in value for word in ('剩余','剩下'))


def _side(value):
    found=re.findall(r'(?:放到|放置到|放在|送到)\s*(左|右)侧传送带',value)
    return {'左':'left','右':'right'}.get(found[0]) if len(set(found))==1 else None


class MissionLedger:
    def __init__(self,round_id,instructions,started_monotonic,started_wall,
                 budget_seconds=600,task_source='custom',allow_reorder=False):
        if not isinstance(round_id,str) or not round_id:raise ValueError('round id required')
        if not isinstance(instructions,(list,tuple)) or not instructions:raise ValueError('instructions required')
        self.round_id=round_id;self.instructions=tuple(_instruction(x) for x in instructions)
        self._started_monotonic=_number(started_monotonic,'start monotonic')
        self.started_wall=_number(started_wall,'start wall')
        self.budget_seconds=_number(budget_seconds,'budget')
        if not 0<self.budget_seconds<=600:raise ValueError('budget must be within 0..600 seconds')
        self._deadline=self._started_monotonic+self.budget_seconds
        self.task_source=str(task_source);self.allow_reorder=bool(allow_reorder)
        self.official_score_eligible=(self.task_source=='official_example' and
            self.instructions==tuple(_instruction(x) for x in OFFICIAL_EXAMPLE_INSTRUCTIONS))
        self.tasks=[{'task_id':'%s-task-%02d'%(round_id,i+1),'instruction':text,
            'status':'pending','requests':[],'targets':{},'grasps':{},'placements':{},
            'active_grasps':{},'releases':{},'release_contexts':{},'dropped':[],'feedback_losses':[],
            'unverified_drop_reports':[],'pending_losses':{},'pending_placements':{},
            'events':[],'nine_evidence':{},'empty_remaining':False}
            for i,text in enumerate(self.instructions)]
        self.known={};self._visible_by_id={};self._visible_frame=None;self._request_index={}
        self._event_keys={};self._last_frame=None;self._last_context={}
        self.round_events=[];self.milestones={'judge_started_monotonic':self._started_monotonic,
                                            'judge_started_wall':self.started_wall}

    @property
    def deadline(self):return self._deadline

    @property
    def started_monotonic(self):return self._started_monotonic

    def remaining(self,now=None):
        now=time.monotonic() if now is None else _number(now,'now')
        if now<self.started_monotonic:raise ValueError('now precedes the fixed competition start')
        return max(0.,self.deadline-now)

    def can_start(self,required_seconds,now=None):
        required=_number(required_seconds,'required seconds')
        if required<0:raise ValueError('required duration cannot be negative')
        remaining=self.remaining(now)
        return remaining>0 and required<=remaining

    def _task(self,index):
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(self.tasks):
            raise ValueError('invalid task index')
        return self.tasks[index]

    def _dependencies(self,index):
        # A task that failed without touching its target is resolved, not pending:
        # nothing is held and nothing is owed, so it must not block later tasks.
        return all(task['status'] in ('verified','executed_pending_verification')
                   or task.get('released_after_failure') for task in self.tasks[:index])

    def is_remaining(self,index):return _remaining_instruction(self._task(index)['instruction'])

    def _delivered(self):
        return {key for task in self.tasks for key in task['placements']}

    def _pending_released(self):
        return {key for task in self.tasks for key in task['pending_placements'] if key not in task['placements']}

    def release_failed_task(self,index,reason,now=None):
        """Return a failed task's untouched targets to the shared pool.

        Nothing was grasped, so the objects are still on the table.  Reserving
        them would make a later "remaining" instruction unsatisfiable: its
        remaining set can never be complete while reserved targets stay
        reserved, and the driver would stop the whole episode there.
        """
        task=self._task(index)
        if task['status'] in ('verified','executed_pending_verification'):return False
        if task['grasps'] or task['placements'] or task['pending_placements']:return False
        if not task['targets']:return False
        at=_number(now,'release time') if now is not None else time.monotonic()
        task['released_after_failure']={'reason':str(reason)[:500],'monotonic':at,
                                        'released_stable_ids':sorted(task['targets'])}
        return True

    def release_rejected_task(self,index,reason,now=None):
        """Mark a planner-rejected task resolved, so later tasks are not blocked.

        A rejection (semantic validation exhausted, request refused, plan rejected)
        arrives before any plan is accepted, so no target was ever bound: nothing is
        held and nothing is owed.  release_failed_task() cannot serve this case -- it
        requires bound targets, which only exist once a plan is accepted -- yet
        _dependencies must still see the task as resolved or the next task cannot even
        be requested.  Grasp/placement evidence still blocks: if anything was taken the
        caller keeps its hard stop, because there is no measured stop to fall back on.
        """
        task=self._task(index)
        if task['status'] in ('verified','executed_pending_verification'):return False
        if task['grasps'] or task['placements'] or task['pending_placements']:return False
        at=_number(now,'reject time') if now is not None else time.monotonic()
        task['released_after_failure']={'reason':str(reason)[:500],'monotonic':at,
                                        'released_stable_ids':sorted(task['targets'])}
        return True

    def task_context(self,scene,index,now=None,wall_now=None):
        task=self._task(index);now=time.monotonic() if now is None else _number(now,'now')
        wall_now=(self.started_wall+now-self.started_monotonic if wall_now is None else _number(wall_now,'wall now'))
        if scene.get('round_id',self.round_id)!=self.round_id:raise ValueError('scene belongs to another round')
        frame=scene.get('frame_id')
        if frame is None:raise ValueError('scene frame id required')
        if self._last_frame is not None and frame<self._last_frame:raise ValueError('older scene cannot replace current context')
        self._last_frame=frame;visible={};by_stable={};issues=[]
        try:
            age=wall_now-_number(scene['observed_at'],'observed_at')
            fresh=0<=age<=2.
        except (KeyError,ValueError):fresh=False
        if not fresh:issues.append('scene_not_fresh')
        for candidate in scene.get('candidates',[]):
            ident=candidate.get('id');stable=candidate.get('stable_id')
            if not isinstance(ident,str) or ident in visible:raise ValueError('invalid or duplicate frame candidate id')
            visible[ident]=copy.deepcopy(candidate)
            confirmed=(isinstance(stable,str) and stable and stable!=ident and
                       candidate.get('identity_status')=='confirmed')
            if not confirmed:
                issues.append('candidate_identity_unconfirmed');continue
            if stable in by_stable:raise ValueError('one stable identity assigned to two visible candidates')
            by_stable[stable]=candidate
            if candidate.get('class') not in ('Magazine','Torch','Grenade','Smokegrenade','CompressedFood'):
                # An unrecognized class must never enter the known table: the scan
                # settles a few frames after the round starts, so caching the
                # transient 'unknown' would make the later, correct class look like
                # a change and abort the round ('bound identity class changed').
                issues.append('candidate_class_unrecognized');continue
            prior=self.known.get(stable)
            if prior and prior['class']!=candidate.get('class'):
                issues.append('stable_identity_class_changed');continue
            self.known[stable]={'class':candidate.get('class'),'last_frame':frame,
                                'last_observed_at':scene.get('observed_at')}
        self._visible_by_id=visible;self._visible_frame=frame
        delivered=self._delivered();pending_released=self._pending_released()
        reserved={key for i,other in enumerate(self.tasks) if i!=index and other['status']!='verified'
                  and not other.get('released_after_failure')
                  for key in other['targets'] if key not in delivered and key not in pending_released}
        remaining_stable=sorted(key for key in by_stable if key not in delivered and key not in reserved and key not in pending_released)
        missing=sorted(key for key in self.known if key not in by_stable and key not in delivered and key not in pending_released)
        if missing:issues.append('unobserved_unverified_identities')
        if scene.get('unknown_regions'):issues.append('unknown_regions')
        if scene.get('sensor_issues'):issues.append('sensor_issues')
        coverage=scene.get('coverage_complete') is True and scene.get('scene_complete',True) is True
        if not coverage:issues.append('coverage_incomplete')
        dependencies=self._dependencies(index)
        if not dependencies:issues.append('prior_tasks_unverified')
        if dependencies and reserved:issues.append('reserved_targets_remain_after_dependencies')
        if pending_released.intersection(by_stable):issues.append('pending_release_identity_visible_in_source')
        if (self.is_remaining(index) and
                any(not _remaining_instruction(other['instruction']) for other in self.tasks[index+1:])):
            issues.append('remaining_before_future_tasks')
        if any(key in delivered for key in by_stable):issues.append('delivered_identity_visible')
        context={'task_id':task['task_id'],'task_index':index,'round_id':self.round_id,
            'dependencies_satisfied':dependencies,'allow_reorder':self.allow_reorder,
            'remaining_ids':[by_stable[key]['id'] for key in remaining_stable],
            'remaining_stable_ids':remaining_stable,
            'reserved_ids':[by_stable[key]['id'] for key in sorted(reserved) if key in by_stable],
            'reserved_stable_ids':sorted(reserved),'stable_ids':sorted(by_stable),
            'released_pending_stable_ids':sorted(pending_released),
            'candidate_stable_ids':{c['id']:key for key,c in by_stable.items()},
            'remaining_complete':not issues,'remaining_incomplete_reasons':sorted(set(issues)),
            'missing_stable_ids':missing,'scene_frame_id':frame,'scene_fresh':fresh,
            'deadline_monotonic':self.deadline,'remaining_seconds':self.remaining(now),
            'coverage_scope':scene.get('coverage_scope','unspecified')}
        self._last_context[index]=copy.deepcopy(context)
        return context

    def mark_requested(self,index,request_id):
        task=self._task(index)
        if not isinstance(request_id,str) or not request_id:raise ValueError('request id required')
        if request_id in self._request_index:
            if self._request_index[request_id]!=index:raise ValueError('request id reused across tasks')
            return False
        if task['status']=='verified':raise ValueError('verified task cannot be restarted')
        if not self._dependencies(index) and (not self.allow_reorder or _remaining_instruction(task['instruction'])):
            raise ValueError('task dependencies not satisfied')
        self._request_index[request_id]=index;task['requests'].append(request_id)
        task['nine_evidence'][request_id]={};task['status']='requested'
        return True

    def bind_plan(self,index,plan,scene=None):
        task=self._task(index)
        request_id=plan.get('request_id')
        if request_id is not None and self._request_index.get(request_id)!=index:
            raise ValueError('plan request is not registered to this task')
        if plan.get('instruction') is not None and _instruction(plan['instruction'])!=task['instruction']:
            raise ValueError('plan instruction differs from registered task')
        side=plan.get('side')
        if side not in ('left','right'):raise ValueError('plan target side required')
        if any(target['side']!=side for target in task['targets'].values()):
            raise ValueError('retry cannot change the bound conveyor side')
        expected_side=_side(task['instruction'])
        if expected_side is not None and side!=expected_side:raise ValueError('plan contradicts instructed conveyor side')
        source={c['id']:c for c in scene.get('candidates',[])} if scene is not None else self._visible_by_id
        source_frame=scene.get('frame_id') if scene is not None else self._visible_frame
        if plan.get('frame_id') is not None and source_frame!=plan['frame_id']:source={}
        objects=plan.get('objects')
        if objects is None:
            ids=plan.get('ids',[]);stable_ids=plan.get('stable_ids')
            if not isinstance(ids,list):raise ValueError('plan ids must be a list')
            if any(ident not in source for ident in ids):raise ValueError('plan requires observed stable identities')
            objects=[source[ident] for ident in ids]
            if stable_ids is not None and stable_ids!=[c.get('stable_id') for c in objects]:
                raise ValueError('plan frame ids and stable ids disagree')
        if not isinstance(objects,list):raise ValueError('plan objects must be a list')
        if any(not isinstance(obj,dict) for obj in objects):raise ValueError('plan objects must contain records')
        ids=[obj.get('id') for obj in objects];stable_ids=[obj.get('stable_id') for obj in objects]
        if plan.get('ids',ids)!=ids or plan.get('stable_ids',stable_ids)!=stable_ids:
            raise ValueError('plan object ids disagree with explicit id lists')
        if any(not isinstance(ident,str) or not isinstance(key,str) or not key or key==ident for ident,key in zip(ids,stable_ids)):
            raise ValueError('persistent identity required; frame ids are not stable ids')
        if len(set(ids))!=len(ids) or len(set(stable_ids))!=len(stable_ids):raise ValueError('duplicate planned object')
        is_remaining=_remaining_instruction(task['instruction'])
        if is_remaining:
            context=self._last_context.get(index,{})
            if not context.get('remaining_complete'):raise ValueError('remaining set is not complete')
            if set(stable_ids)!=set(context['remaining_stable_ids']):raise ValueError('plan does not cover complete remaining set')
        elif not objects:raise ValueError('ordinary category task requires a target')
        if self.official_score_eligible and index<4:
            if len(objects)!=1 or objects[0].get('class')!=OFFICIAL_CLASSES[index]:
                raise ValueError('official example target category or quantity mismatch')
        if task['targets']:
            pending=set(task['targets'])-set(task['placements'])
            if task['status']=='verified':
                if set(stable_ids)!=set(task['targets']):raise ValueError('verified task targets cannot change')
                return copy.deepcopy(task['targets'])
            if ((is_remaining and not pending.issubset(set(stable_ids))) or
                    (not is_remaining and set(stable_ids)!=pending)):
                raise ValueError('retry cannot replace the already bound undelivered identities')
        delivered=self._delivered()
        for obj in objects:
            key=obj['stable_id'];ident=obj['id']
            if obj.get('source_frame_id') is not None and plan.get('frame_id') is not None and obj['source_frame_id']!=plan['frame_id']:
                raise ValueError('plan object comes from a different scene frame')
            if obj.get('identity_status','confirmed')!='confirmed':raise ValueError('unconfirmed plan identity')
            if obj.get('class') not in ('Magazine','Torch','Grenade','Smokegrenade','CompressedFood'):
                raise ValueError('unknown objects cannot be bound for sorting')
            if ident in source and source[ident].get('stable_id')!=key:raise ValueError('plan does not match current candidate identity')
            if key in delivered:raise ValueError('target has already been delivered')
            if any(key in other['targets'] for other in self.tasks if other is not task):
                raise ValueError('target reserved by another task')
            prior=self.known.get(key)
            if prior and prior['class']!=obj.get('class'):raise ValueError('bound identity class changed')
        for obj in objects:
            key=obj['stable_id']
            task['targets'].setdefault(key,{'stable_id':key,'side':side,'class':obj.get('class'),
                                          'initial_frame_id':obj['id']})
            self.known.setdefault(key,{'class':obj.get('class'),'last_frame':plan.get('frame_id'),
                                      'last_observed_at':plan.get('created_at')})
        if task['status']=='verified':return copy.deepcopy(task['targets'])
        task['empty_remaining']=is_remaining and not objects;task['status']='planned'
        return copy.deepcopy(task['targets'])

    def _event_time(self,event):
        if event.get('monotonic') is not None:return _number(event['monotonic'],'event monotonic')
        if event.get('time') is not None:return self.started_monotonic+_number(event['time'],'event wall')-self.started_wall
        return None

    def _semantic_evidence(self,task):
        for evidence in task['nine_evidence'].values():
            names=('infer_started','model_answer','plan_published')
            if not all(name in evidence for name in names):continue
            times=[evidence[name]['monotonic'] for name in names]
            if not all(t is not None for t in times) or times!=sorted(times):continue
            decision=evidence['plan_published'].get('decision_monotonic',times[-1])
            if decision+1e-6<times[1] or decision>times[-1]+1e-6:continue
            first_grasp=min((row['monotonic'] for row in task['grasps'].values()),default=None)
            boundary=first_grasp
            if boundary is None and task['empty_remaining']:boundary=task.get('completed_monotonic')
            if boundary is None or decision<=boundary+1e-6:return True
        return False

    def record_event(self,index,event):
        task=self._task(index);event=copy.deepcopy(event)
        status=event.get('status',event.get('event'))
        raw=json.dumps(event,sort_keys=True,ensure_ascii=False,allow_nan=False)
        digest=hashlib.sha256(raw.encode('utf-8')).hexdigest()
        key=(index,event.get('event_id') or digest)
        if key in self._event_keys:
            prior=self._event_keys[key]
            if prior['digest']!=digest:raise ValueError('event id reused with different content')
            return {'accepted':prior['record']['accepted'],'duplicate':True,
                    'reason':prior['record'].get('reason')}
        record={'event':event,'accepted':False};task['events'].append(record)
        self._event_keys[key]={'digest':digest,'record':record}
        try:
            if event.get('round_id',self.round_id)!=self.round_id:raise ValueError('event round mismatch')
            if event.get('task_id',task['task_id'])!=task['task_id']:raise ValueError('event task mismatch')
            rid=event.get('request_id')
            if rid is not None and self._request_index.get(rid)!=index:raise ValueError('event request mismatch')
            at=self._event_time(event)
            if at is not None and at<self.started_monotonic:raise ValueError('task event precedes competition start')
            if status in ('infer_started','model_answer','plan_published'):
                if rid is None:raise ValueError('Nine evidence must identify actual request')
                evidence=task['nine_evidence'][rid]
                if status=='infer_started':
                    if _instruction(event.get('instruction',''))!=task['instruction']:
                        raise ValueError('Nine inferred a different instruction')
                    if event.get('frame_id') is None or not event.get('image'):raise ValueError('Nine image/frame evidence missing')
                elif status=='model_answer':
                    if not isinstance(event.get('answer'),str) or not event['answer'].strip():raise ValueError('Nine answer missing')
                else:
                    if event['plan'].get('request_id')!=rid:raise ValueError('published plan request differs from model call')
                    self.bind_plan(index,event['plan'])
                evidence[status]={'event_digest':digest,'monotonic':at}
                if status=='plan_published' and event['plan'].get('created_at') is not None:
                    evidence[status]['decision_monotonic']=(self.started_monotonic+
                        _number(event['plan']['created_at'],'plan created_at')-self.started_wall)
            elif status=='placement_pending_verification':
                stable=event.get('stable_id');proof=event.get('proof',{});context=task['release_contexts'].get(stable,{})
                if (stable not in task['targets'] or event.get('side')!=task['targets'][stable]['side']
                        or stable not in task['grasps'] or stable not in task['releases']):
                    raise ValueError('pending placement lacks bound grasp/release evidence')
                if not context.get('release_id') or event.get('release_id')!=context['release_id']:
                    raise ValueError('pending placement release identity differs')
                if (at is None or at<task['releases'][stable] or not isinstance(proof,dict)
                        or not all(proof.get(k) is True for k in ('release_completed','empty_gripper','at_observation_pose'))):
                    raise ValueError('pending placement lacks completed release and empty observation-pose proof')
                if stable not in task['placements']:
                    task['pending_placements'][stable]={'release_id':context['release_id'],'monotonic':at,
                                                       'event_digest':digest,'proof':copy.deepcopy(proof)}
            elif status in ('grasp_verified','placement_verified','released','drop_detected',
                            'holding_feedback_lost','physical_drop_verified'):
                stable=event.get('stable_id') or event.get('candidate',{}).get('stable_id')
                if stable not in task['targets']:raise ValueError('event is not for a bound target')
                target=task['targets'][stable]
                if event.get('side')!=target['side']:raise ValueError('event conveyor side contradicts task')
                if at is None:raise ValueError('physical event time required')
                if status in ('drop_detected','holding_feedback_lost','physical_drop_verified'):
                    effective=task['active_grasps'].pop(stable,None)
                    if effective is None:effective=task['pending_losses'].get(stable,{}).get('effective_grasp_at')
                    loss={'stable_id':stable,'monotonic':at,'effective_grasp_at':effective,
                          'event_digest':digest,'status':status}
                    physical=event.get('physical_drop_proof')
                    independent=(isinstance(physical,dict) and physical.get('verified') is True
                        and physical.get('source') in ('rgbd_object_trajectory','camera_object_trajectory','independent_object_observation')
                        and isinstance(physical.get('samples'),list) and len(physical['samples'])>=2)
                    if independent:
                        for sample in physical['samples']:
                            if not isinstance(sample,dict):independent=False;break
                            position=sample.get('world_position',[])
                            if (not isinstance(position,(list,tuple)) or len(position)!=3 or
                                    any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in position)):
                                independent=False;break
                    if independent and effective is not None:
                        # A camera-confirmed report may follow the original
                        # holding signal loss; do not count one incident twice.
                        if not any(row['stable_id']==stable and row['effective_grasp_at']==effective for row in task['dropped']):
                            task['dropped'].append({**loss,'proof':copy.deepcopy(physical)})
                    else:
                        if status in ('drop_detected','holding_feedback_lost'):
                            task['feedback_losses'].append(loss)
                        if status=='physical_drop_verified' or event.get('physical_drop_proof'):
                            task['unverified_drop_reports'].append({**loss,'reason':
                                'no_prior_effective_grasp' if effective is None else 'independent_physical_drop_evidence_missing'})
                    task['pending_losses'][stable]=loss
                elif status=='released':
                    if at>self.deadline:raise ValueError('release occurred after competition deadline')
                    task['releases'][stable]=at
                    release_id=event.get('release_id')
                    if release_id is not None:
                        if not isinstance(release_id,str) or not release_id:raise ValueError('invalid release id')
                        task['release_contexts'][stable]={'release_id':release_id,'request_id':rid,
                            'released_at':event.get('released_at',self.started_wall+at-self.started_monotonic)}
                else:
                    if not any(event.get(name) for name in ('proof','evidence','transport','verification','lift_proof')):
                        raise ValueError('physical verification evidence missing')
                    if status=='grasp_verified':
                        if at>self.deadline:raise ValueError('grasp occurred after competition deadline')
                        if stable in task['placements']:raise ValueError('delivered object cannot be grasped again')
                        task['grasps'].setdefault(stable,{'monotonic':at,'event_digest':digest})
                        task['active_grasps'][stable]=at
                        task['pending_losses'].pop(stable,None)
                    else:
                        if stable in task['placements']:
                            record['accepted']=True
                            return {'accepted':True,'duplicate':True}
                        if stable not in task['active_grasps']:raise ValueError('placement lacks current verified grasp')
                        # Opening the gripper does not establish belt arrival.
                        # Prefer a qualified belt observation; absent that, use
                        # the later verification time conservatively.
                        delivered=at;time_source='verification_event_time'
                        transport=event.get('transport') or {}
                        if not isinstance(transport,dict):raise ValueError('invalid transport evidence')
                        samples=transport.get('samples',[])
                        qualified=(transport.get('transport_seen') is True or transport.get('settled_on_belt') is True)
                        if qualified:
                            if not isinstance(samples,list) or not samples or any(not isinstance(sample,dict) for sample in samples):
                                raise ValueError('qualified transport samples missing or invalid')
                            sample_times=[self.started_monotonic+_number(sample.get('observed_at'),'belt observed_at')-self.started_wall
                                          for sample in samples]
                            if sample_times!=sorted(sample_times) or sample_times[-1]>at+1e-6:
                                raise ValueError('qualified transport sample chronology invalid')
                            if transport.get('side',target['side'])!=target['side'] or transport.get('class',target['class'])!=target['class']:
                                raise ValueError('transport class or conveyor side mismatch')
                        release=task['release_contexts'].get(stable)
                        if release is not None:
                            if event.get('release_id')!=release['release_id']:
                                raise ValueError('placement release id mismatch')
                            expected={'stable_id':stable,'request_id':release['request_id'],'release_id':release['release_id']}
                            for proof in [transport]+(samples if isinstance(samples,list) else []):
                                if not isinstance(proof,dict) or any(proof.get(k)!=v for k,v in expected.items()):
                                    raise ValueError('transport object/request/release association mismatch')
                                stamp=_number(proof.get('released_at'),'transport released_at')
                                if abs(stamp-_number(release['released_at'],'recorded released_at'))>1e-6:
                                    raise ValueError('transport release time mismatch')
                        observed_delivery=None
                        if qualified:
                            delivered=sample_times[0]
                            observed_delivery=delivered
                            time_source='first_qualified_belt_observation_wall'
                        explicit_times=[]
                        if event.get('delivered_at') is not None:
                            explicit_times.append(self.started_monotonic+_number(event['delivered_at'],'delivered_at')-self.started_wall)
                            time_source=event.get('delivery_time_source','explicit_delivery_wall')
                        if event.get('delivered_monotonic') is not None:
                            explicit_times.append(_number(event['delivered_monotonic'],'delivered monotonic'))
                            time_source=event.get('delivery_time_source','explicit_delivery_monotonic')
                        if explicit_times:
                            if max(explicit_times)-min(explicit_times)>1e-6:
                                raise ValueError('explicit delivery clocks contradict one another')
                            if observed_delivery is not None and min(explicit_times)<observed_delivery-1e-6:
                                raise ValueError('explicit delivery precedes qualified transport samples')
                            # Floating point conversion must never backdate the
                            # raw sample across the competition deadline.
                            delivered=max(explicit_times+([] if observed_delivery is None else [observed_delivery]))
                        if delivered<task['active_grasps'][stable] or delivered>at:
                            raise ValueError('delivery time order invalid')
                        if delivered>self.deadline:raise ValueError('delivery occurred after competition deadline')
                        if stable in task['releases'] and delivered<task['releases'][stable]-.001:
                            raise ValueError('belt delivery evidence precedes recorded release')
                        if at>self.deadline and (time_source=='verification_event_time' or stable not in task['releases']):
                            raise ValueError('late verification requires qualified timely delivery and recorded release')
                        task['placements'].setdefault(stable,{'monotonic':delivered,'verified_monotonic':at,
                            'event_digest':digest,'side':target['side'],'delivery_time_source':time_source,
                            'released_monotonic':task['releases'].get(stable)})
                        task['active_grasps'].pop(stable,None)
                        task['pending_placements'].pop(stable,None)
                        self.milestones['last_delivery_monotonic']=max(delivered,self.milestones.get('last_delivery_monotonic',delivered))
                        self.milestones['last_verification_monotonic']=max(at,self.milestones.get('last_verification_monotonic',at))
                        if (task.get('controller_status')=='task_pending_verification'
                                and set(task['placements'])==set(task['targets']) and self._semantic_evidence(task)):
                            task.update(status='verified',completed_monotonic=at,
                                        completion_basis='background_verified_timely_delivery_after_completed_release')
            elif status=='task_pending_verification':
                covered=set(task['placements'])|set(task['pending_placements'])
                if (at is None or not task['targets'] or covered!=set(task['targets']) or event.get('motion_complete') is not True
                        or not self._semantic_evidence(task)):
                    raise ValueError('continuation requires completed bound releases and actual Nine evidence')
                expected=set(task['targets'])-set(task['placements'])
                if set(event.get('pending_stable_ids',[]))-set(task['placements'])!=expected:
                    raise ValueError('pending task targets differ from unresolved releases')
                task['controller_status']=status;task['operationally_completed_monotonic']=at
                task['status']='executed_pending_verification' if expected else 'verified'
                task['completion_basis']='release_complete_placement_unverified' if expected else 'background_verified_timely_delivery'
                if not expected:task['completed_monotonic']=at
            elif status=='task_succeeded':
                task['controller_status']=status
                if set(task['placements'])!=set(task['targets']) or (not task['targets'] and not task['empty_remaining']):
                    raise ValueError('task success lacks every bound target delivery')
                decisions=[e['plan_published'].get('decision_monotonic',e['plan_published']['monotonic'])
                           for e in task['nine_evidence'].values() if 'plan_published' in e]
                if at is not None and decisions and all(decision is not None and decision>at+1e-6 for decision in decisions):
                    raise ValueError('task completion precedes the Nine decision')
                task['status']='verified';task['completed_monotonic']=at
                task['completion_basis']='controller_result_with_verified_delivery_evidence'
            elif status in ('task_failed','rejected','plan_rejected','task_deferred'):
                task['controller_status']=status
                if task['status']!='verified':task['status']='deferred' if status=='task_deferred' else 'incomplete'
            record['accepted']=True
            return {'accepted':True,'duplicate':False}
        except (ValueError,KeyError,TypeError) as error:
            record['reason']=str(error)
            return {'accepted':False,'duplicate':False,'reason':str(error)}

    def finalize_verified_deliveries(self,index,stopped_proof,verified_at):
        """Derive task evidence after stopping; never emit a controller success.

        Operational failure remains in controller_status.  This only credits
        already observed, deadline-compliant deliveries and actual Nine calls.
        """
        task=self._task(index)
        if task['status']=='verified':return False
        if not isinstance(stopped_proof,dict) or stopped_proof.get('state')!='stopped' or not stopped_proof.get('id'):
            return False
        if not self._dependencies(index) or not self._semantic_evidence(task):return False
        if set(task['placements'])!=set(task['targets']):return False
        if not task['targets'] and not task['empty_remaining']:return False
        decisions=[e['plan_published'].get('decision_monotonic',e['plan_published']['monotonic'])
                   for e in task['nine_evidence'].values() if 'plan_published' in e]
        if not any(at is not None and at<=self.deadline for at in decisions):return False
        if any(row['monotonic']>self.deadline for row in task['placements'].values()):return False
        task['status']='verified';task['completed_monotonic']=_number(verified_at,'read-only verification time')
        task['completion_basis']='verified_timely_deliveries_after_measured_stop'
        task['completion_stop_id']=stopped_proof['id']
        return True

    def observe(self,event):
        rid=event.get('request_id')
        if rid in self._request_index:return self.record_event(self._request_index[rid],event)
        task_id=event.get('task_id')
        for index,task in enumerate(self.tasks):
            if task_id==task['task_id']:return self.record_event(index,event)
        status=event.get('status',event.get('event'))
        if status in ('preparation_started','judge_started','round_started','motion_started',
                      'returned_home','stop_requested','stopped','aborted'):
            at=self._event_time(event)
            if at is None:raise ValueError('round event time required')
            if event.get('round_id',self.round_id)!=self.round_id:raise ValueError('round event mismatch')
            if status in ('judge_started','round_started') and abs(at-self.started_monotonic)>.001:
                raise ValueError('competition start cannot be reset')
            if status=='preparation_started' and at>self.started_monotonic:raise ValueError('preparation must precede start')
            self.round_events.append(copy.deepcopy(event))
            self.milestones.setdefault(status+'_monotonic',at)
            return {'accepted':True}
        return {'accepted':False,'reason':'event has no registered task or round milestone'}

    def summary(self,now=None):
        remaining=self.remaining(now);tasks=[];estimated=0;pending=[]
        for index,task in enumerate(self.tasks):
            semantic=self._semantic_evidence(task)
            score=None;score_status='not_official_scoring_source'
            if self.official_score_eligible:
                if task['status']=='verified' and task['targets'] and semantic:
                    score=OFFICIAL_FULL_POINTS[index];score_status='local_full_task_estimate';estimated+=score
                elif task['status']=='verified' and not task['targets']:
                    score_status='empty_remaining_requires_referee';pending.append(task['task_id'])
                elif task['grasps'] or task['placements']:
                    score_status='partial_or_missing_semantic_evidence_requires_referee';pending.append(task['task_id'])
                else:score=0;score_status='no_verified_scoring_evidence'
            tasks.append({'task_id':task['task_id'],'instruction':task['instruction'],'status':task['status'],
                'target_stable_ids':list(task['targets']),'verified_grasps':len(task['grasps']),
                'verified_placements':len(task['placements']),'nine_semantic_evidence':semantic,
                'estimated_points':score,'score_status':score_status,'drop_count':len(task['dropped']),
                'controller_status':task.get('controller_status'),
                'completion_basis':task.get('completion_basis'),
                'physical_drop_count':len(task['dropped']),'holding_feedback_loss_count':len(task['feedback_losses']),
                'unverified_drop_report_count':len(task['unverified_drop_reports']),
                'rejected_events':sum(not row['accepted'] for row in task['events']),
                'targets':copy.deepcopy(task['targets']),'request_ids':list(task['requests']),
                'grasp_evidence':copy.deepcopy(task['grasps']),'placement_evidence':copy.deepcopy(task['placements']),
                'pending_placement_evidence':copy.deepcopy(task['pending_placements']),
                'released_after_failure':copy.deepcopy(task.get('released_after_failure')),
                'nine_evidence':copy.deepcopy(task['nine_evidence'])})
        return {'round_id':self.round_id,'task_source':self.task_source,'started_monotonic':self.started_monotonic,
            'started_wall':self.started_wall,'deadline_monotonic':self.deadline,'remaining_seconds':remaining,
            'tasks':tasks,'verified_objects':len(self._delivered()),
            'all_tasks_verified':all(t['status']=='verified' for t in self.tasks),
            'mandatory_nine_evidence_complete':all(self._semantic_evidence(t) for t in self.tasks),
            'official_score_eligible':self.official_score_eligible,
            'estimated_base_points':estimated if self.official_score_eligible else None,
            'pending_referee_tasks':pending,'multimodal_bonus':{'points':None,'status':'not_specialty_validated'},
            'milestones':copy.deepcopy(self.milestones),'scoring_scope':'local evidence estimate; not referee score'}
