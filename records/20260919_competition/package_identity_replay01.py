from pathlib import Path
import json,sys,cv2,time
p=Path('/root/tcei_package_validation01/tcei_260920v2');run=Path('/root/gpufree-data/tcei_package_validation01/package_smoke_round01');r=Path('/root/tcei_competition_20260919');sys.path[:0]=[str(p/'tcei_stack'),str(p/'evaluation')]
from ultralytics import YOLO
from rotation_perception import RotationDetector
from rgbd_archive import read_depth_archive
e=[json.loads(x) for x in (run/'episode/events.jsonl').read_text().splitlines()];idx=[json.loads(x) for x in (run/'rgbd/index.jsonl').read_text().splitlines()];last_infer=[x for x in e if x.get('status')=='infer_started'][-1]
o={'scope':'recorded images only, no ROS publishers or commands','last_infer':{k:last_infer[k] for k in ('time','frame_id','image','prompt','task_context')},'frames':[]}
model=RotationDetector(YOLO('/root/jaka/best.pt'));picks=[min(idx,key=lambda y:abs(y['captured_at']-(last_infer['time']+delta))) for delta in (0,10,20,35,48,55,60,65)]
for row in picks:
 path=run/'rgbd'/(row['stem']+'.npz');source,boxes,metrics=model.detect(cv2.imread(str(path.with_suffix('.jpg'))),read_depth_archive(path),row['camera']['K']);o['frames'].append({'stem':row['stem'],'wall':row['captured_at'],'stamp':row['stamp'],'source':[{k:c.get(k) for k in ('class','pixel','depth','bbox','confidence','grasp_ready','grasp_uncertainty')} for c in source],'unknown_regions':metrics.get('unknown_regions'),'coverage_complete':metrics.get('coverage_complete')})
(r/'package_identity_diagnosisA.txt').write_text(json.dumps(o,ensure_ascii=False,indent=2));print('IDENTITY_REPLAY_DONE',len(o['frames']),flush=True)
