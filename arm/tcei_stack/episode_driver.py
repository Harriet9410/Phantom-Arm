"""One-clock competition driver with injectable transport, no ROS or truth reads."""
import copy
import collections
import hashlib
import json
import math
import time
import uuid
from mission_ledger import MissionLedger
from ros_endpoints import CANCEL_PROTOCOL,OBSERVATION_PROTOCOL


class EpisodeFailure(RuntimeError):pass


class EpisodeDriver:
    def __init__(self,transport,instructions,task_source='custom',budget=600.,clock=time,
                 record=None,round_id=None,minimum_task_seconds=60.,observation_timeout=60.,
                 scene_timeout=10.,stop_timeout=8.,preflight_timeout=30.,
                 dependency_timeout=2.,evidence_grace_seconds=3.,observation_acceptance_timeout=3.):
        self.transport=transport;self.instructions=list(instructions);self.task_source=task_source
        self.budget=float(budget);self.clock=clock;self.record=record or (lambda event:None)
        self.round_id=round_id or 'round_'+uuid.uuid4().hex
        self.minimum_task_seconds=minimum_task_seconds;self.observation_timeout=observation_timeout
        self.scene_timeout=scene_timeout;self.stop_timeout=stop_timeout;self.preflight_timeout=preflight_timeout
        self.dependency_timeout=float(dependency_timeout);self.evidence_grace_seconds=float(evidence_grace_seconds)
        self.observation_acceptance_timeout=float(observation_acceptance_timeout)
        if not 0<self.observation_acceptance_timeout<=10:raise ValueError('invalid observation acceptance timeout')
        if not 0<self.dependency_timeout<=10 or not 0<=self.evidence_grace_seconds<=10:
            raise ValueError('bounded evidence wait must be within configured limits')
        self.ledger=None;self.current_request=None;self.latest_stop=None;self.expected_stop_ids=set()
        self.active_stop_id=None;self.retired_stop_ids=set();self.stop_acks={}
        self.cancel_epoch=0;self.cancel_started=None;self.cancel_started_wall=None;self.cancel_stop_ids=set()
        self.cancel_request=None;self.cancel_mapping=None;self.cancel_ack_error=None
        self.pending_events=[];self.ready_events=collections.deque();self.late_evidence_errors=[]
        self.minimum_scene_wall=None;self.actions_requested=False;self.current_sent_wall=None
        self.task_interval_started=None;self.task_interval_ended=None
        self.summary={'status':'not_started','verified_objects':0,'elapsed_seconds':0.,
                      'round_id':self.round_id,'task_source':task_source,'budget_seconds':self.budget,'tasks':[]}

    def metadata(self,task_id):
        return {'round_id':self.round_id,'task_id':task_id,'deadline_monotonic':self.ledger.deadline,
                'started_at_monotonic':self.ledger.started_monotonic,'started_at_wall':self.ledger.started_wall}

    def checkpoint(self):
        if self.transport.is_shutdown():raise EpisodeFailure('ROS shutdown during episode')
        if self.ledger.remaining(self.clock.monotonic())<=0:raise TimeoutError('fixed episode deadline exhausted')
        if not self.transport.execution_enabled():raise EpisodeFailure('execution authorization was disabled')

    def _retire_stop(self,ident):
        if not ident:return
        self.retired_stop_ids.add(ident);self.cancel_stop_ids.discard(ident)
        if self.active_stop_id==ident:self.active_stop_id=None
        if self.latest_stop and self.latest_stop['message'].get('id')==ident:self.latest_stop=None

    def _update_stop(self,topic,event,received):
        if topic=='/tcei/cancel_ack':
            expected=self.cancel_request
            if (expected is None or event.get('protocol')!=CANCEL_PROTOCOL or
                    event.get('cancel_id')!=expected['cancel_id'] or
                    event.get('request_round_id')!=expected['round_id'] or
                    event.get('request_id')!=expected['request_id'] or received<self.cancel_started):return
            at=event.get('monotonic')
            if type(at) not in (int,float) or not math.isfinite(at) or not self.cancel_started<=at<=self.clock.monotonic()+1e-6:return
            if event.get('state')=='rejected':self.cancel_ack_error=event.get('reason','cancel rejected');return
            ident=event.get('stop_id');revision=event.get('mapping_revision')
            if (event.get('state')!='accepted' or not isinstance(ident,str) or not ident or
                    type(revision) is not int or revision<0 or ident in self.retired_stop_ids):return
            prior=self.cancel_mapping
            if prior and (revision<prior['mapping_revision'] or
                          (revision==prior['mapping_revision'] and ident!=prior['stop_id'])):return
            self.cancel_mapping=copy.deepcopy(event);self.cancel_stop_ids={ident}
            self.active_stop_id=ident;self.expected_stop_ids.add(ident)
            return
        if topic=='/tcei/stop_ack':
            ident=event.get('id');row={'message':event,'received_monotonic':received}
            self.latest_stop=row
            if ident:self.stop_acks[ident]=row
            if event.get('state')=='reset':self._retire_stop(ident)
            return
        ours=(event.get('request_id')==self.current_request or event.get('round_id')==self.round_id)
        if not ours:return
        status=event.get('status');ident=event.get('stop_id')
        if status=='controlled_stop_reset':self._retire_stop(ident)
        elif status=='controlled_stop_requested' and ident and ident not in self.retired_stop_ids:
            self.expected_stop_ids.add(ident)
            # Task-scoped legacy events are useful evidence, but only the
            # explicit cancel_id response can bind an external cancellation.
            if self.cancel_started is None:self.active_stop_id=ident

    def _metadata_error(self,event):
        if event.get('status')!='plan_published':return None
        rid=event.get('request_id')
        if self.ledger is None or rid not in self.ledger._request_index:return None
        plan=event.get('plan',{});index=self.ledger._request_index[rid]
        expected=self.metadata(self.ledger.tasks[index]['task_id'])
        for key in ('round_id','task_id','deadline_monotonic','started_at_monotonic','started_at_wall'):
            if plan.get(key)!=expected[key]:return 'Nine plan changed or omitted fixed mission metadata: '+key
        if plan.get('request_id')!=rid:return 'Nine published a mismatched request id'
        return None

    def _dependency(self,event):
        rid=event.get('request_id');status=event.get('status')
        if self.ledger is None or rid not in self.ledger._request_index:return None
        task=self.ledger.tasks[self.ledger._request_index[rid]]
        nine=task['nine_evidence'][rid]
        if status=='model_answer' and 'infer_started' not in nine:return 'Nine infer_started'
        if status=='plan_published' and not all(key in nine for key in ('infer_started','model_answer')):
            return 'Nine inference and answer'
        physical=('grasp_verified','released','placement_verified','drop_detected',
                  'holding_feedback_lost','physical_drop_verified','task_succeeded',
                  'placement_pending_verification','task_pending_verification')
        if status in physical and 'plan_published' not in nine:return 'Nine plan binding'
        stable=event.get('stable_id') or event.get('candidate',{}).get('stable_id')
        if stable in task['targets'] and status in ('released','placement_verified','placement_pending_verification') and stable not in task['grasps']:
            return 'current verified grasp'
        if status in ('placement_verified','placement_pending_verification') and stable in task['targets'] and stable not in task['releases']:
            return 'recorded release'
        if status=='task_succeeded':
            if not self.ledger._semantic_evidence(task):return 'complete Nine evidence chain'
            if set(task['placements'])!=set(task['targets']):return 'all bound target placements'
        if status=='task_pending_verification':
            if not self.ledger._semantic_evidence(task):return 'complete Nine evidence chain'
            if set(task['placements'])|set(task['pending_placements'])!=set(task['targets']):return 'all completed bound releases'
        return None

    def _apply_event(self,event,validate):
        rid=event.get('request_id')
        if self.ledger is not None and rid in self.ledger._request_index:
            result=self.ledger.observe(event)
            critical=('infer_started','model_answer','plan_published','grasp_verified','released',
                      'placement_verified','drop_detected','holding_feedback_lost','physical_drop_verified','task_succeeded',
                      'placement_pending_verification','task_pending_verification')
            if event.get('status') in critical and not result.get('accepted'):
                reason='task evidence rejected: '+str(result.get('reason'))
                if validate:raise EpisodeFailure(reason)
                self.late_evidence_errors.append({'request_id':rid,'status':event.get('status'),'reason':reason})
                return
        if event.get('status') in ('task_succeeded','task_pending_verification','task_failed','rejected','plan_rejected'):
            for task in self.summary['tasks']:
                if task['request_id']==rid:
                    task['result']=copy.deepcopy(event)
                    task['elapsed_seconds']=max(0.,self.clock.time()-task['sent_at'])
        self.ready_events.append(event)

    def _drain_pending(self,validate):
        progress=True
        while progress:
            progress=False
            for pending in list(self.pending_events):
                reason=self._dependency(pending['event'])
                if reason is None:
                    self.pending_events.remove(pending);self._apply_event(pending['event'],validate);progress=True
                else:pending['reason']=reason
        expired=[pending for pending in self.pending_events
                 if self.clock.monotonic()-pending['received_monotonic']>self.dependency_timeout]
        if expired and validate:
            raise EpisodeFailure('bounded evidence dependency timeout: '+expired[0]['reason'])
        # During read-only closure keep unresolved records until the grace
        # boundary, then report them; never invent the missing dependency.

    def _consume(self,envelope,validate=True):
        if envelope is not None:
            topic=envelope['topic'];event=copy.deepcopy(envelope['message'])
            received=envelope.get('received_monotonic',self.clock.monotonic())
            self._update_stop(topic,event,received)
            if topic in ('/tcei/stop_ack','/tcei/cancel_ack'):
                label='stop_ack' if topic=='/tcei/stop_ack' else 'cancel_ack'
                self.record({'status':label,'stop' if label=='stop_ack' else 'response':event,
                             'source_topic':topic,'received_monotonic':received})
                self.ready_events.append(event)
            else:
                self.record({**event,'source_topic':topic,'received_monotonic':received})
                if (validate and event.get('status') in ('request_rejected','observation_request_rejected') and
                        self.current_sent_wall is not None and type(event.get('time')) in (int,float) and
                        event['time']>=self.current_sent_wall and event.get('request_id')!=self.current_request):
                    raise EpisodeFailure('fresh unassociated request rejection: '+str(event.get('reason')))
                error=self._metadata_error(event)
                if error:
                    if validate:raise EpisodeFailure(error)
                    self.late_evidence_errors.append({'reason':error,'status':event.get('status')})
                else:
                    reason=self._dependency(event)
                    if reason:
                        fingerprint=hashlib.sha256(json.dumps(event,sort_keys=True).encode('utf-8')).hexdigest()
                        if not any(p['fingerprint']==fingerprint for p in self.pending_events):
                            if len(self.pending_events)>=256:raise EpisodeFailure('bounded dependency buffer capacity exceeded')
                            self.pending_events.append({'event':event,'received_monotonic':received,
                                                        'reason':reason,'fingerprint':fingerprint})
                    else:self._apply_event(event,validate)
        self._drain_pending(validate)
        return self.ready_events.popleft() if self.ready_events else None

    def _next(self,timeout=.2,validate=True):
        self._drain_pending(validate)
        if self.ready_events:return self.ready_events.popleft()
        return self._consume(self.transport.next_event(timeout),validate=validate)

    def _prepare_observation(self):
        rid='observation_'+uuid.uuid4().hex;self.current_request=rid
        request={**self.metadata(self.round_id+'-observation'),'request_id':rid}
        self.summary['observation']={'request_id':rid,'sent_at':self.clock.time()}
        self.current_sent_wall=self.clock.time();sent=self.clock.monotonic()
        self.actions_requested=True;self.transport.publish('/tcei/prepare_observation',request)
        until=min(self.ledger.deadline,sent+self.observation_timeout)
        accepted_until=min(until,sent+self.observation_acceptance_timeout);accepted=False
        while self.clock.monotonic()<until:
            if not accepted and self.clock.monotonic()>=accepted_until:
                raise TimeoutError('observation request acceptance not confirmed within shared budget')
            self.checkpoint()
            wait_until=until if accepted else accepted_until
            event=self._next(timeout=min(.2,max(0.,wait_until-self.clock.monotonic())))
            if not event or event.get('request_id')!=rid:continue
            status=event.get('status')
            if status in ('observation_accepted','observation_started','observation_completed'):
                if (event.get('observation_protocol')!=OBSERVATION_PROTOCOL or
                        event.get('round_id')!=self.round_id or event.get('task_id')!=request['task_id']):
                    raise EpisodeFailure('observation acknowledgement context/protocol mismatch')
                accepted=True;self.summary['observation'].setdefault('accepted_at',self.clock.time())
                if status=='observation_started':self.summary['observation']['worker_started_at']=self.clock.time()
            if status=='observation_completed':
                self.summary['observation']['result']=event;self.minimum_scene_wall=self.clock.time();return
            if status in ('task_failed','observation_request_rejected','observation_preparation_failed','execution_disabled','queued_request_cancelled'):
                raise EpisodeFailure('observation preparation failed: '+str(event.get('reason',status)))
        raise TimeoutError('basket observation preparation did not complete within shared budget')

    def _context(self,index):
        until=min(self.ledger.deadline,self.clock.monotonic()+self.scene_timeout)
        last_reason='fresh observation unavailable'
        last_audit=-math.inf
        while self.clock.monotonic()<until:
            self.checkpoint();scene=self.transport.latest_scene()
            if scene is not None:
                observed=scene.get('observed_at')
                fresh=(type(observed) in (int,float) and 0<=self.clock.time()-observed<2.)
                if fresh and (self.minimum_scene_wall is None or observed>=self.minimum_scene_wall):
                    context=self.ledger.task_context(scene,index,now=self.clock.monotonic(),wall_now=self.clock.time())
                    if not self.ledger.is_remaining(index) or context['remaining_complete']:return scene,context
                    last_reason='remaining set incomplete: '+','.join(context['remaining_incomplete_reasons'])
                    if self.clock.monotonic()-last_audit>=1.:
                        self.record({'status':'remaining_context_incomplete','time':self.clock.time(),
                            'task_index':index,'context':copy.deepcopy(context),
                            'candidates':[{key:c.get(key) for key in ('id','class','pixel','bbox','depth',
                                'stable_id','identity_status','identity_candidates','identity_reason',
                                'association_anchor_held','association_anchor_pixel')} for c in scene.get('candidates',[])],
                            'unknown_regions':[{key:r.get(key) for key in ('reason','bbox','pixel','stable_id',
                                'candidate_id','identity_candidates','retired_stable_ids')} for r in scene.get('unknown_regions',[])]})
                        last_audit=self.clock.monotonic()
            self._next(timeout=.1)
        raise EpisodeFailure(last_reason)

    def _run_task(self,index):
        scene,context=self._context(index)
        empty=self.ledger.is_remaining(index) and context['remaining_complete'] and not context['remaining_ids']
        if not self.ledger.can_start(1. if empty else self.minimum_task_seconds,self.clock.monotonic()):
            raise TimeoutError('insufficient shared budget for another task and safe termination')
        rid='episode_'+uuid.uuid4().hex;self.current_request=rid;self.ledger.mark_requested(index,rid)
        request={**self.metadata(context['task_id']),'request_id':rid,'instruction':self.instructions[index],'task_context':context}
        task={'request_id':rid,'task_id':context['task_id'],'instruction':self.instructions[index],
              'sent_at':self.clock.time(),'scene_frame_id':scene['frame_id'],'empty_remaining_requested':empty}
        self.summary['tasks'].append(task);began=self.clock.monotonic();self.current_sent_wall=self.clock.time()
        if self.task_interval_started is None:self.task_interval_started=began
        self.transport.publish('/tcei/request',request)
        while self.clock.monotonic()<self.ledger.deadline:
            self.checkpoint();event=self._next()
            if not event or event.get('request_id')!=rid:continue
            status=event.get('status')
            if status=='task_succeeded':
                if self.ledger.tasks[index]['status']!='verified':raise EpisodeFailure('task lacks verified delivery evidence')
                task.update(result=event,elapsed_seconds=self.clock.monotonic()-began)
                self.task_interval_ended=self.clock.monotonic()
                self.minimum_scene_wall=self.clock.time();return
            if status=='task_pending_verification':
                if self.ledger.tasks[index]['status'] not in ('executed_pending_verification','verified'):
                    raise EpisodeFailure('pending task has no completed release evidence')
                task.update(result=event,elapsed_seconds=self.clock.monotonic()-began)
                self.task_interval_ended=self.clock.monotonic()
                self.minimum_scene_wall=self.clock.time();return
            if status=='task_failed':
                task.update(result=event,elapsed_seconds=self.clock.monotonic()-began)
                self.task_interval_ended=self.clock.monotonic()
                self.minimum_scene_wall=self.clock.time()
                proof=event.get('stop') if isinstance(event.get('stop'),dict) else None
                if not (event.get('stopped_verified') is True and proof and proof.get('state')=='stopped'):
                    raise EpisodeFailure('task stopped after task_failed without a measured stop: '+str(event.get('reason','')))
                released=self.ledger.release_failed_task(index,event.get('reason',''),self.clock.monotonic())
                self.record({'status':'task_failed_continuing' if released else 'task_failed_unresolved',
                             'time':self.clock.time(),'task_index':index,'task_id':context['task_id'],
                             'reason':event.get('reason'),'stop_id':proof.get('id'),
                             'targets_released':released})
                if not released:
                    raise EpisodeFailure('task stopped after task_failed with unfinished release evidence: '+str(event.get('reason','')))
                return {'task_index':index,'task_id':context['task_id'],'status':'task_failed',
                        'reason':event.get('reason',''),'stop_id':proof.get('id')}
            if status in ('rejected','request_rejected','plan_rejected'):
                task.update(result=event,elapsed_seconds=self.clock.monotonic()-began)
                self.task_interval_ended=self.clock.monotonic()
                self.minimum_scene_wall=self.clock.time()
                # 计划被拒发生在"计划被接受"之前：没有绑定目标、没有抓取，物体仍在桌上。
                # 按账本自身的规则（失败且未触碰其目标的任务算已了结），继续下一条，而不是
                # 让整轮死在第一条失败指令上；一旦有抓取/放置证据就维持原有的硬停（那种
                # 情形没有可用的"已停稳"测量，不能带着手里的东西往下走）。
                resolved=self.ledger.release_rejected_task(index,event.get('reason',''),self.clock.monotonic())
                self.record({'status':'task_rejected_continuing' if resolved else 'task_rejected_unresolved',
                             'time':self.clock.time(),'task_index':index,'task_id':context['task_id'],
                             'reason':event.get('reason'),'reject_status':status,'resolved':resolved})
                if not resolved:
                    raise EpisodeFailure('task stopped after '+status+' with existing grasp or placement evidence: '+str(event.get('reason','')))
                return {'task_index':index,'task_id':context['task_id'],'status':'task_rejected',
                        'reason':event.get('reason',''),'reject_status':status}
            if status=='execution_disabled':
                # 无动作/预热模式的模式标志，不是任务失败：维持硬停，避免把配置问题
                # 伪装成"5 条都跑过、只是都被拒"。
                task.update(result=event,elapsed_seconds=self.clock.monotonic()-began)
                raise EpisodeFailure('task stopped after execution_disabled: '+str(event.get('reason','')))
        raise TimeoutError('fixed episode deadline exhausted while waiting for task result')

    def _cancel_and_wait(self,reason):
        began=self.clock.monotonic()
        self.cancel_epoch+=1;self.cancel_started=began;self.cancel_started_wall=self.clock.time()
        self.cancel_stop_ids=set();self.cancel_mapping=None;self.cancel_ack_error=None
        self.cancel_request={'protocol':CANCEL_PROTOCOL,'cancel_id':uuid.uuid4().hex,
                             'round_id':self.round_id,'request_id':self.current_request or 'no_action_request',
                             'reason':str(reason)[:1000]}
        self.record({'status':'cancel_requested','request':copy.deepcopy(self.cancel_request),
                     'monotonic':began,'time':self.cancel_started_wall})
        try:self.transport.publish('/tcei/cancel_request',self.cancel_request)
        except Exception as error:return {'state':'unconfirmed','reason':'cancel publication failed: '+str(error),'seconds':0.}
        # Controller owns the stop ID; a competing second ID would conflict
        # with its already-latched stop following the same task failure.
        while self.clock.monotonic()-began<self.stop_timeout and not self.transport.is_shutdown():
            latest=self.stop_acks.get(self.active_stop_id)
            if latest is not None:
                reply=latest['message']
                if (reply.get('id') in self.cancel_stop_ids and reply.get('id') not in self.retired_stop_ids and
                        latest['received_monotonic']>=began and self.clock.monotonic()-latest['received_monotonic']<1.):
                    if reply.get('state') in ('stopped','fault'):
                        if reply['state']=='stopped':
                            self.ledger.observe({'status':'stopped','round_id':self.round_id,
                                                 'monotonic':self.clock.monotonic(),'proof':reply})
                        return {'state':reply['state'],'proof':reply,'seconds':self.clock.monotonic()-began,
                                'cancel_epoch':self.cancel_epoch,'cancel_mapping':copy.deepcopy(self.cancel_mapping),
                                'received_monotonic':latest['received_monotonic']}
            self._next(timeout=.1,validate=False)
        return {'state':'unconfirmed','reason':self.cancel_ack_error or reason,
                'cancel_mapping':copy.deepcopy(self.cancel_mapping),'seconds':self.clock.monotonic()-began}

    def _read_only_closeout(self,safety):
        began=self.clock.monotonic();before=len(self.ledger._delivered())
        until=began+self.evidence_grace_seconds
        while self.clock.monotonic()<until and not self.transport.is_shutdown():
            self._next(timeout=min(.1,max(0.,until-self.clock.monotonic())),validate=False)
            proof=safety.get('proof',{})
            if (safety.get('state')=='stopped' and proof.get('id')==self.active_stop_id and
                    proof.get('id') not in self.retired_stop_ids):
                for index,task in enumerate(self.ledger.tasks):
                    if self.ledger.finalize_verified_deliveries(index,proof,self.clock.monotonic()):
                        self.record({'status':'task_evidence_finalized','task_id':task['task_id'],
                            'round_id':self.round_id,'monotonic':self.clock.monotonic(),
                            'basis':task['completion_basis'],'controller_status':task.get('controller_status'),
                            'stop_id':proof['id'],'derived_locally':True})
            if (all(t['status']=='verified' for t in self.ledger.tasks) and
                    all(self.ledger._semantic_evidence(t) for t in self.ledger.tasks) and
                    all(t.get('controller_status') is not None for t in self.ledger.tasks) and not self.pending_events):break
        self.summary['read_only_closeout']={'seconds':self.clock.monotonic()-began,
            'new_verified_objects':len(self.ledger._delivered())-before,'actions_allowed':False}
        self.summary['unresolved_evidence']=[{'request_id':p['event'].get('request_id'),
            'status':p['event'].get('status'),'reason':p['reason']} for p in self.pending_events]
        self.summary['late_evidence_errors']=copy.deepcopy(self.late_evidence_errors)

    def run(self):
        started=None
        try:
            if not self.transport.execution_enabled():raise EpisodeFailure('Nine and controller execution must already be authorized')
            started=self.clock.monotonic();started_wall=self.clock.time()
            self.ledger=MissionLedger(self.round_id,self.instructions,started,started_wall,budget_seconds=self.budget,task_source=self.task_source)
            self.summary.update(started_at=started_wall,started_monotonic=started,deadline_monotonic=self.ledger.deadline,
                                start_source='driver_start_after_execution_authorization')
            start={'status':'judge_started','round_id':self.round_id,'monotonic':started,'time':started_wall,
                   'source':'driver_start_event'}
            self.record(start);self.ledger.observe(start)
            # Nodes must already be warm.  Even connection/freshness waiting
            # inside the driver consumes T0+600; readiness never resets T0.
            initial=self.transport.preflight(min(self.preflight_timeout,self.ledger.remaining(self.clock.monotonic())))
            self.summary['initial_scene']=initial;self.checkpoint();self._prepare_observation()
            failed_tasks=[]
            for index in range(len(self.instructions)):
                outcome=self._run_task(index)
                if outcome is None:continue
                failed_tasks.append(outcome)
                if index+1<len(self.instructions):
                    # The controller states that an explicit reset is required after a
                    # failed task; re-preparing the observation is that reset, and it
                    # fails closed if the controller refuses to run it again.
                    self._prepare_observation()
            self.summary['failed_tasks']=failed_tasks
            if any(t['status']=='executed_pending_verification' for t in self.ledger.tasks):
                self._read_only_closeout({'state':'not_requested'})
            evidence=self.ledger.summary(self.clock.monotonic())
            if not evidence['mandatory_nine_evidence_complete']:
                raise EpisodeFailure('complete task or mandatory Nine evidence is missing')
            complete=evidence['all_tasks_verified']
            unresolved=any(t['status']=='incomplete' for t in self.ledger.tasks)
            if complete:status='succeeded'
            elif unresolved:status='completed_with_failed_tasks'
            else:status='completed_with_unverified_placements'
            self.summary.update(status=status,
                safe_stop={'state':'not_requested','reason':('normal verified completion' if complete else
                    'some commanded tasks failed; the remaining instructions were still attempted' if unresolved else
                    'all commanded releases completed; some placement evidence remains unverified')})
        except Exception as error:
            if self.task_interval_started is not None:self.task_interval_ended=self.clock.monotonic()
            deadline_expired=(self.ledger is not None and self.clock.monotonic()>=self.ledger.deadline)
            self.summary.update(status='failed' if started is not None else 'not_started',reason=str(error))
            if self.actions_requested and self.ledger is not None:
                try:safety=self._cancel_and_wait(str(error))
                except Exception as stop_error:
                    safety={'state':'unconfirmed','reason':'stop acknowledgement processing failed: '+str(stop_error)}
                self.summary['safe_stop']=safety
                if safety['state']=='stopped':self.summary['status']='stopped'
                try:self._read_only_closeout(safety)
                except Exception as closing_error:
                    self.summary['read_only_closeout_error']=str(closing_error)
                ident=safety.get('proof',{}).get('id')
                if safety['state']=='stopped' and (ident in self.retired_stop_ids or self.active_stop_id!=ident):
                    self.summary['safe_stop']={'state':'unconfirmed','reason':'stop reset or replaced during read-only closure',
                                               'prior_proof':safety}
                    self.summary['status']='failed'
                current_ack=self.stop_acks.get(ident)
                if (self.summary['safe_stop']['state']=='stopped' and current_ack is not None and
                        current_ack['received_monotonic']>safety.get('received_monotonic',-1) and
                        current_ack['message'].get('state')!='stopped'):
                    self.summary['safe_stop']={'state':'fault' if current_ack['message'].get('state')=='fault' else 'unconfirmed',
                        'reason':'later stop feedback no longer confirms stationary hold','proof':current_ack['message']}
                    self.summary['status']='failed'
                if (deadline_expired and self.summary['safe_stop']['state']=='stopped' and
                        all(t['status']=='verified' for t in self.ledger.tasks) and
                        all(self.ledger._semantic_evidence(t) for t in self.ledger.tasks) and
                        not self.pending_events and not self.late_evidence_errors):
                    self.summary['status']='succeeded'
                    self.summary['completion_verification_phase']='read_only_after_deadline'
            else:self.summary['safe_stop']={'state':'not_requested','reason':'this driver issued no action request'}
        finally:
            ended=self.clock.monotonic();self.summary['elapsed_seconds']=max(0.,ended-started) if started is not None else 0.
            self.summary['task_interval_elapsed_seconds']=(None if self.task_interval_started is None else
                max(0.,(self.task_interval_ended if self.task_interval_ended is not None else ended)-self.task_interval_started))
            self.summary['task_interval_scope']='first Nine request to last task result or episode failure; observation preparation and final stop wait excluded'
            if self.ledger is not None:
                evidence=self.ledger.summary(max(ended,self.ledger.started_monotonic))
                self.summary['ledger']=evidence;self.summary['verified_objects']=evidence['verified_objects']
                last=evidence['milestones'].get('last_delivery_monotonic')
                self.summary['last_delivery_elapsed_seconds']=None if last is None else last-self.ledger.started_monotonic
                self.summary['time_scope']='one start includes observation, Nine, motion, recovery and verification; stop may extend after deadline'
        return self.summary
