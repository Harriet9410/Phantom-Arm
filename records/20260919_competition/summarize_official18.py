"""Compact audit of real18 episode and unchanged original camera evidence."""
from pathlib import Path
import collections,hashlib,json,shutil

ROOT=Path('/root/tcei_competition_20260919')
RUN=Path('/root/gpufree-data/tcei_competition_20260919/official18_round01')
summary=json.loads((RUN/'episode/summary.json').read_text())
finished=json.loads((RUN/'supervisor_finished.json').read_text())
events=[json.loads(line) for line in (RUN/'episode/events.jsonl').read_text().splitlines()]
counts=dict(collections.Counter(e.get('status') for e in events))
keep=('time','status','task_id','request_id','category','side','stable_id','reason','command',
      'requested_target','applied_distance_m','limit_reason','contact_efforts','finger_positions',
      'preload_distance_m','preload_limited','stable_sim_seconds','fresh_contact_samples','release_id',
      'stopped_verified','placed_verified','pending_stable_ids','elapsed','visual_confirmation')
statuses={'grasp_preload','grasp_contact_confirmed','grasp_verified','closure_rejected','empty_grasp',
    'holding_feedback_lost','drop_recovery_started','controlled_stop_fault','controlled_stop_verified',
    'placement_pending_verification','placement_verified','task_succeeded','task_failed','task_pending_verification',
    'released','model_answer'}
compact=[]
for e in events:
    if e.get('status') not in statuses:continue
    row={k:e[k] for k in keep if k in e}
    if e.get('candidate'):row['candidate']={k:e['candidate'].get(k) for k in ('class','stable_id','pixel','angle_deg')}
    if e.get('status')=='model_answer':row.update(answer=e.get('answer'),elapsed=e.get('elapsed'))
    if e.get('transport'):
        row['transport']={k:e['transport'].get(k) for k in ('travel_m','speed_m_s','transport_seen','side','class','release_id')}
        row['transport']['samples']=len(e['transport'].get('samples',[]))
    compact.append(row)

frames=[];review=RUN/'delivery_review18';review.mkdir(exist_ok=False)
index=[json.loads(line) for line in (RUN/'rgbd/index.jsonl').read_text().splitlines()]
for e in events:
    if e.get('status')!='released':continue
    number=e['task_id'].rsplit('-',1)[-1]
    for delay in (1.5,3.):
        frame=min(index,key=lambda f:abs(f['captured_at']-(e['time']+delay)))
        source=RUN/'rgbd'/(frame['stem']+'.jpg')
        dest=review/('task'+number+'_'+e['release_id'][:8]+'_'+str(int(delay*1000))+'ms.jpg')
        assert not dest.exists()
        shutil.copyfile(source,dest)
        frames.append({'task_id':e['task_id'],'release_id':e.get('release_id'),'delay':delay,
            'source':str(source),'display':str(dest),'captured_at':frame['captured_at'],
            'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})

out={'summary':{k:summary.get(k) for k in ('status','elapsed_seconds','verified_objects','reason','safe_stop',
    'code_changed_during_episode','last_delivery_elapsed_seconds')},'counts':counts,'events':compact,
    'tasks':[{k:t.get(k) for k in ('instruction','status','verified_grasps','verified_placements','estimated_points',
        'score_status','physical_drop_count','holding_feedback_loss_count','rejected_events')} for t in summary['ledger']['tasks']],
    'closed':finished,'frames':frames,'raw_hashes':{name:hashlib.sha256((RUN/name).read_bytes()).hexdigest()
        for name in ('episode/summary.json','episode/events.jsonl','supervisor_finished.json')},
    'scope':'Native evidence summary. Raw images still require independent visual review. Not a30-round campaign result.'}
(ROOT/'official18_final_review.txt').open('x').write(json.dumps(out,ensure_ascii=False,indent=2))
(review/'evidence_manifest.json').open('x').write(json.dumps({'frames':frames,'raw_hashes':out['raw_hashes'],
    'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print('OFFICIAL18_REVIEW_READY',summary['status'],summary['verified_objects'],len(frames),flush=True)
