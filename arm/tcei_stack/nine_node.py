#!/usr/bin/env python3
"""Nine remains the multimodal planner; strict guards reject unsafe answers."""
import copy
import json
import os
import queue
import threading
import time
import uuid
import cv2
import rospy
import torch
from PIL import Image as PILImage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transformers import AutoModel, AutoTokenizer
from core import parse_plan, rebind_candidate
from semantic_repair import infer_validated_selection
from frame_pairing import stamp_key,latest_fresh_pair,received_image_record,validate_image_binding
from planning_prompt import planning_prompt,feedback_principle_zh
from semantics import (bind_task_context, scene_unchanged, audit_instruction,
                       validate_selection, observation_readiness, ObservationRequired,
                       ModelNeedsConfirmation,MODEL_SELECTION_PROTOCOL)

REQUEST_CONTEXT_KEYS = {'task_context', 'round_id', 'task_id', 'deadline_monotonic',
                        'started_at_wall', 'started_at_monotonic'}


def read_inference_settings(get_param):
    """Bounded startup configuration; bools/strings/floats are not integers."""
    values={name:get_param('~'+name,1) for name in ('vision_batch_size','num_beams')}
    for name,maximum in (('vision_batch_size',16),('num_beams',3)):
        if type(values[name]) is not int or not 1<=values[name]<=maximum:
            raise ValueError('%s must be a strict integer within 1..%d'%(name,maximum))
    return values


def inference_configuration(model,settings):
    """Audit actual vision config and the explicit chat generation arguments."""
    config=model.config
    actual=getattr(config,'vision_batch_size',None)
    if type(actual) is not int or actual!=settings['vision_batch_size']:
        raise RuntimeError('model.config.vision_batch_size differs from frozen startup configuration')
    slice_config=getattr(config,'slice_config',None)
    slices=(slice_config.get('max_slice_nums') if isinstance(slice_config,dict) else
            getattr(slice_config,'max_slice_nums',None))
    source='model.config.slice_config.max_slice_nums'
    if slices is None:
        slices=getattr(config,'max_slice_nums',None);source='model.config.max_slice_nums'
    return {'vision_batch_size':actual,'vision_batch_size_source':'model.config.vision_batch_size',
            'num_beams':settings['num_beams'],'num_beams_source':'explicit_chat_kwarg',
            'max_slice_nums':slices,'max_slice_nums_source':source if slices is not None else 'not_exposed',
            'max_slice_nums_override':None,'full_image_preserved':True,
            'sampling':False,'max_new_tokens':400,'model_dtype':str(getattr(model,'dtype','unavailable')),
            'configuration_scope':'validated_once_at_node_start'}


class Nine:
    def __init__(self):
        self.lock=threading.Lock(); self.images={}; self.scenes={}; self.snapshot=None
        self.queue=queue.Queue(maxsize=1); self.seen=set(); self.busy=False
        self.bridge=CvBridge()
        self.plan_pub=rospy.Publisher('/tcei/plan',String,queue_size=1)
        self.status_pub=rospy.Publisher('/tcei/nine_status',String,queue_size=10,latch=True)
        self.log_dir=rospy.get_param('~log_dir','/root/tcei_runs/20260917_2216/improved')
        os.makedirs(self.log_dir,exist_ok=True)
        self.inference_settings=read_inference_settings(rospy.get_param)
        path=rospy.get_param('~model_path','/root/inference/FM9G4B-V')
        self.model=AutoModel.from_pretrained(path,trust_remote_code=True,attn_implementation='sdpa',torch_dtype=torch.bfloat16).eval().to('cuda')
        self.model.config.vision_batch_size=self.inference_settings['vision_batch_size']
        self.tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        # Only the numbered full scene preserves left/up/extreme relations.
        rospy.Subscriber('/tcei/annotated_image',Image,self.on_image,queue_size=1,buff_size=2**24)
        rospy.Subscriber('/tcei/candidates',String,self.on_candidates,queue_size=1)
        rospy.Subscriber('/tcei/request',String,self.on_request,queue_size=1)
        self.event('ready',inference_configuration=inference_configuration(self.model,self.inference_settings),
                   model_protocol=MODEL_SELECTION_PROTOCOL)

    def event(self,status,**data):
        row={'time':time.time(),'status':status,**data}
        encoded=json.dumps(row,ensure_ascii=False)
        self.status_pub.publish(String(encoded));rospy.loginfo(encoded)
        with open(os.path.join(self.log_dir,'nine_events.jsonl'),'a') as f:f.write(encoded+'\n')

    def on_image(self,msg):
        key=stamp_key(msg.header.stamp.to_sec())
        image=self.bridge.imgmsg_to_cv2(msg,'bgr8').copy()
        record=received_image_record(image,msg.header.stamp,msg.header.frame_id,msg.header.seq)
        with self.lock:
            self.images.pop(key,None);self.images[key]=record
            for k in list(self.images)[:-12]:del self.images[k]

    def on_candidates(self,msg):
        scene=json.loads(msg.data);key=stamp_key(scene['stamp'])
        with self.lock:
            self.snapshot=scene;self.scenes.pop(key,None);self.scenes[key]=scene
            for k in list(self.scenes)[:-12]:del self.scenes[k]

    def on_request(self,msg):
        rid=None
        try:
            req=json.loads(msg.data) if msg.data.lstrip().startswith('{') else {'instruction':msg.data,'request_id':str(uuid.uuid4())}
            if isinstance(req,dict) and isinstance(req.get('request_id'),str):rid=req['request_id']
            if (not {'instruction','request_id'} <= set(req) or
                    set(req) - {'instruction','request_id'} - REQUEST_CONTEXT_KEYS or
                    not isinstance(req['instruction'],str)):
                raise ValueError('request schema')
            if 'task_context' in req and not isinstance(req['task_context'],dict):
                raise ValueError('task context schema')
            if 'deadline_monotonic' in req:
                deadline=req['deadline_monotonic']
                if type(deadline) not in (int,float) or not time.monotonic()<deadline<float('inf'):
                    raise ValueError('expired or invalid absolute task deadline')
            rid=req['request_id']
            if not isinstance(rid,str) or not rid or len(rid)>80:raise ValueError('request ID')
            if rid in self.seen:raise ValueError('duplicate request')
            if self.busy or not self.queue.empty():raise ValueError('planner busy')
            self.seen.add(rid); self.queue.put_nowait(req)
        except Exception as e:self.event('request_rejected',request_id=rid,reason=str(e))

    def wait_for_observation(self,req,started,global_deadline,observation_seconds):
        """Obtain a fresh matched scene after transient identities stabilize.

        Waiting has its own bounded allocation, charged to the same absolute
        competition deadline. No model call or target selection occurs here.
        """
        until=min(started+observation_seconds,global_deadline)
        last_reason='fresh matched image and candidates unavailable'
        waiting_reported=False
        while time.monotonic()<until and not rospy.is_shutdown():
            with self.lock:
                pair=latest_fresh_pair(self.scenes,self.images,time.time())
                if pair is not None:
                    snap=copy.deepcopy(pair[0]);image_seq=pair[1]['transport_seq'];image=pair[1]['image'].copy()
                else:
                    snap=None
                    paired_times=[s for key,s in self.scenes.items() if key in self.images
                                  and 0<=time.time()-s['observed_at']<2.]
                    if paired_times:
                        newest=max(paired_times,key=lambda s:s['observed_at'])
                        try:validate_image_binding(newest,self.images[stamp_key(newest['stamp'])])
                        except (ValueError,TypeError,KeyError) as error:last_reason=str(error)
            if snap is not None:
                context=req.get('task_context') or {};minimum_frame=context.get('scene_frame_id')
                expected_pending=set(context.get('released_pending_stable_ids',[]))
                applied=set(snap.get('source_released_pending_stable_ids',[]))|set(snap.get('source_delivered_stable_ids',[]))
                if type(minimum_frame) is int and snap.get('frame_id',-1)<minimum_frame:
                    readiness={'ready':False,'reason':'camera frame predates requested task context'}
                elif not expected_pending<=applied:
                    readiness={'ready':False,'reason':'perception has not applied completed pending releases'}
                else:readiness=observation_readiness(snap,req['instruction'])
                if readiness['ready']:
                    return snap,image_seq,image,time.monotonic()-started
                last_reason=readiness['reason']
            if not waiting_reported:
                self.event('observation_waiting',request_id=req['request_id'],reason=last_reason,
                           observation_deadline_monotonic=until,global_deadline_monotonic=global_deadline)
                waiting_reported=True
            remaining=until-time.monotonic()
            if remaining>0:time.sleep(min(.02,remaining))
        if time.monotonic()>=global_deadline:
            raise TimeoutError('global deadline exhausted while waiting for observation')
        raise ObservationRequired('observation wait budget exhausted: '+last_reason)

    def run(self):
        while not rospy.is_shutdown():
            try:req=self.queue.get(timeout=.2)
            except queue.Empty:continue
            self.busy=True;rid=req['request_id'];started=time.monotonic()
            terminal_status=None;stage='observation';model_calls=0;model_answers=0
            try:
                observation_seconds=float(rospy.get_param('~observation_wait_seconds',3.))
                if not 0<observation_seconds<=5.:
                    raise ValueError('observation_wait_seconds must be within 0..5 seconds')
                global_deadline=req.get('deadline_monotonic',started+observation_seconds+30.)
                snap,image_seq,image,observation_elapsed=self.wait_for_observation(
                    req,started,global_deadline,observation_seconds)
                stage='input_preparation'
                pil=PILImage.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                artifact=str(uuid.uuid4())
                pil.save(os.path.join(self.log_dir,artifact+'.png'))
                task_context=bind_task_context(req.get('task_context'),snap)
                hint_keys=('id','class','pixel','bbox','normalized_xy','confidence','depth',
                           'grasp_ready','identity_status','pixel_uncertainty_px')
                hints=[{key:c[key] for key in hint_keys if key in c} for c in snap['candidates']]
                prompt=planning_prompt(req['instruction'],hints,snap,task_context)
                actual_inference=inference_configuration(self.model,self.inference_settings)
                model_started=time.monotonic()
                deadline=min(model_started+30.,global_deadline)
                self.event('infer_started',request_id=rid,instruction=req['instruction'],
                           model_protocol=MODEL_SELECTION_PROTOCOL,
                           frame_id=snap['frame_id'],image=artifact+'.png',image_seq=image_seq,
                           transport_seq=image_seq,sequence_role='ROS_transport_audit_only',
                           scene_image_binding=snap['scene_image_binding'],
                           paired_sensor_stamp=snap['stamp'],prompt=prompt,task_context=task_context,
                           observation_wait_seconds=observation_elapsed,model_budget_seconds=30.,
                           inference_configuration=actual_inference,
                           model_deadline_monotonic=deadline,global_deadline_monotonic=global_deadline)
                def infer(attempt,previous,feedback):
                    nonlocal stage,model_calls,model_answers
                    if time.monotonic()>=deadline:raise TimeoutError('model/global planning budget exceeded')
                    actual_prompt=prompt
                    if feedback:
                        actual_prompt=planning_prompt(req['instruction'],hints,snap,task_context,feedback=feedback)
                        self.event('repair_started',request_id=rid,attempt=attempt,feedback=feedback,
                                   feedback_guidance_zh=feedback_principle_zh(feedback),
                                   previous_answer=previous,prompt=actual_prompt)
                    stage='model_inference';model_calls+=1
                    inference_configuration(self.model,self.inference_settings)
                    with torch.inference_mode():
                        answer=self.model.chat(image=None,msgs=[{'role':'user','content':[pil,actual_prompt]}],tokenizer=self.tokenizer,max_new_tokens=400,sampling=False,num_beams=self.inference_settings['num_beams'])
                    model_answers+=1;stage='plan_validation'
                    self.event('model_answer',request_id=rid,answer=str(answer),attempt=attempt,elapsed=time.monotonic()-started)
                    if time.monotonic()>=deadline:raise TimeoutError('model/global planning budget exceeded')
                    return answer
                def rejected(attempt,answer,error):
                    self.event('validation_failed',request_id=rid,attempt=attempt,answer=answer,reason=error)
                selection=infer_validated_selection(infer,snap['candidates'],req['instruction'],
                    scene=snap,task_context=task_context,on_reject=rejected)
                parsed=selection['internal_v2']
                stage='plan_publication'
                # ROS parameter access can block. It must finish before the
                # final observation/deadline gate, not after that gate.
                execute_enabled=rospy.get_param('~execute',False)
                if time.monotonic()>=deadline:raise TimeoutError('model/global planning budget exceeded')
                with self.lock:
                    current=copy.deepcopy(self.snapshot)
                    if current is None or not 0<=time.time()-current['observed_at']<2:
                        raise ObservationRequired('perception stopped during inference')
                    if current['stamp']<snap['stamp']:
                        raise ObservationRequired('late old scene after model inference')
                    associations=scene_unchanged(snap,current,rebind_candidate)
                    rebound=[associations[i] for i in parsed['ids']]
                    current_task_context=bind_task_context(req.get('task_context'),current)
                    rebound_answer={**parsed,'ids':[c['id'] for c in rebound]}
                    validate_selection(rebound_answer,current['candidates'],audit_instruction(req['instruction']),
                                       current,current_task_context)
                    plan={key:copy.deepcopy(req[key]) for key in REQUEST_CONTEXT_KEYS if key in req}
                    plan.update({'request_id':rid,'instruction':req['instruction'],'created_at':time.time(),
                          'side':parsed['side'],'class':parsed['class'],'objects':rebound,'frame_id':current['frame_id'],
                          'stamp':current['stamp'],'source_frame_id':snap['frame_id'],'source_stamp':snap['stamp'],
                          'source_image_binding':snap['scene_image_binding'],'source_transport_seq':image_seq,
                          'semantic':parsed,'task_context':current_task_context,'model_source':'FM9G4B-V',
                          'model_protocol':selection['model_protocol'],'model_selection':selection['model_selection'],
                          'audited_constraints':selection['audited_constraints'],
                          'semantic_provenance':selection['semantic_provenance']})
                    message=String(json.dumps(plan))
                    # Serialization is also charged to the deadline. Snapshot
                    # replacement cannot race this final short gate; model work
                    # and parameter/network reads never run under the lock.
                    if time.monotonic()>=deadline:raise TimeoutError('model/global planning budget exceeded')
                    if not 0<=time.time()-current['observed_at']<2:
                        raise ObservationRequired('scene expired while preparing plan publication')
                    if execute_enabled:self.plan_pub.publish(message)
                terminal_status='plan_published' if execute_enabled else 'dry_run_validated'
                self.event(terminal_status,request_id=rid,plan=plan)
            except Exception as e:
                terminal_status='rejected'
                needs_observation=isinstance(e,ObservationRequired)
                model_confirmation=isinstance(e,ModelNeedsConfirmation)
                failure_kind=('model_needs_confirmation' if model_confirmation else
                              'observation_required' if needs_observation else
                              'budget_exhausted' if isinstance(e,TimeoutError) else
                              'semantic_validation' if stage=='plan_validation' and isinstance(e,ValueError) else
                              'runtime_error')
                self.event('rejected',request_id=rid,reason=str(e),elapsed=time.monotonic()-started,
                           requires_reobservation=needs_observation,failure_kind=failure_kind,
                           error_type=type(e).__name__,failure_stage=stage,
                           model_protocol=MODEL_SELECTION_PROTOCOL,
                           model_response=copy.deepcopy(e.model_response) if model_confirmation else None)
            finally:
                self.busy=False;self.queue.task_done()
                self.event('request_finished',request_id=rid,scope='nine_inference_request',
                           completion_protocol='tcei.nine.request_finished.v1',terminal_status=terminal_status,
                           idle=not self.busy and self.queue.empty(),model_call_attempts=model_calls,
                           model_answers_received=model_answers)


if __name__=='__main__':
    rospy.init_node('tcei_nine')
    Nine().run()
