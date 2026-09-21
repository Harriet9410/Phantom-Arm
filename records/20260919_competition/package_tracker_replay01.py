from pathlib import Path
import json,sys,cv2,time,hashlib,numpy as np
p=Path('/root/tcei_package_validation01/tcei_260920v2');run=Path('/root/gpufree-data/tcei_package_validation01/package_smoke_round01');r=Path('/root/tcei_competition_20260919');sys.path[:0]=[str(p/'tcei_stack'),str(p/'evaluation')]
from ultralytics import YOLO
from rotation_perception import RotationDetector
from perception_tracking import CandidateTracker
from rgbd_archive import read_depth_archive
e=[json.loads(x) for x in (run/'episode/events.jsonl').read_text().splitlines()];start=[x for x in e if x.get('status')=='infer_started'][-1];hints=json.loads(start['prompt'].split('候选提示：')[1].split('\n任务上下文：')[0]);tracker=CandidateTracker();tracker.round_id=start['task_context']['round_id'];tracker.counter=10;tracker.session_id='replay';tracker.last_frame=start['frame_id']
for c in hints:
 key=start['task_context']['candidate_stable_ids'][c['id']];tracker.tracks[key]={'candidate':dict(c,stable_id=key),'seen_count':10,'last_seen':start['time'],'last_frame':start['frame_id'],'delivered':False}
model=RotationDetector(YOLO('/root/jaka/best.pt'));idx=[json.loads(x) for x in (run/'rgbd/index.jsonl').read_text().splitlines() if json.loads(x)['captured_at']>=start['time']];events=[x for x in e if x.get('time',0)>=start['time']];cursor=0;previous=None;previous_name=None;out={'scope':'JPEG RGB-D replay, seeded only with last real confirmed two-object observation; not a physical rerun','changes':[]};last=None
for n,row in enumerate(idx):
 path=run/'rgbd'/(row['stem']+'.npz')
 with np.load(path,allow_pickle=False) as f:
  if 'depth' in f:depth=f['depth']
  elif str(f['previous'].item())==previous_name:depth=np.bitwise_xor(f['depth_xor'],previous.view(np.uint32)).view(np.float32)
  else:depth=read_depth_archive(path)
  assert hashlib.sha256(depth.tobytes()).hexdigest()==str(f['depth_sha256'].item())
 previous=depth;previous_name=path.name
 while cursor<len(events) and events[cursor].get('time',0)<=row['captured_at']:tracker.on_event(events[cursor]);cursor+=1
 source,boxes,metrics=model.detect(cv2.imread(str(path.with_suffix('.jpg'))),depth,row['camera']['K'])
 for i,c in enumerate(source):c.update(id=str(i+1),image_stamp=row['stamp'])
 candidates,unknown=tracker.update(source,start['frame_id']+n+1,row['captured_at'],metrics.get('unknown_regions',[]),row['stamp']);small=[{k:c.get(k) for k in ('class','pixel','depth','stable_id','identity_status','identity_candidates','identity_reason')} for c in candidates];signature=str([(x['class'],x['stable_id'],x['identity_status'],x['identity_candidates']) for x in small])
 if signature!=last:
  out['changes'].append({'stem':row['stem'],'wall':row['captured_at'],'candidates':small,'tracks':{k:{'candidate':{a:t['candidate'].get(a) for a in ('class','pixel','depth')},'delivered':t['delivered']} for k,t in tracker.tracks.items()}});last=signature
 if n%50==0:(r/'package_tracker_replay_progress.json').write_text(json.dumps({'frames':n,'total':len(idx),'changes':len(out['changes'])}))
out['processed_frames']=len(idx);out['final_unknown']=[{k:x.get(k) for k in ('reason','stable_id','retired_stable_ids','pixel')} for x in unknown];(r/'package_tracker_diagnosisA.txt').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('TRACKER_REPLAY_DONE',len(idx),flush=True)
