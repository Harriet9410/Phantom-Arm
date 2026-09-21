"""Read-only offline validation of actual camera recordings."""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np
from ultralytics import YOLO

parser=argparse.ArgumentParser()
parser.add_argument('--code',required=True)
parser.add_argument('--inputs',required=True)
args=parser.parse_args();sys.path.insert(0,args.code)
from rotation_perception import RotationDetector
p=Path(args.inputs);rgb=cv2.imread(str(p/'current_rgb.png'))
depth=np.load(p/'current_depth.npz')['depth'];k=json.loads((p/'input_metadata.json').read_text())['K']
detector=RotationDetector(YOLO('/root/jaka/best.pt'))
rows=[]
for i in range(3):
    source,base,metrics=detector.detect(rgb,depth,k)
    print(json.dumps({'iteration':i,'metrics':metrics,'objects':[{n:c[n] for n in ('class','confidence','pixel','angle_deg','axis_ratio','recognition_rotation_deg')} for c in source]}),flush=True)
    rows.append({'iteration':i,'metrics':metrics,'objects':source})
image=rgb.copy()
for c in source:
    a,b,d,e=map(int,c['bbox']);u,v=map(int,c['pixel'])
    cv2.rectangle(image,(a,b),(d,e),(0,210,0),2);cv2.circle(image,(u,v),3,(0,0,255),-1)
    cv2.putText(image,'%s %.2f yaw%.1f'%(c['class'],c['confidence'],c['angle_deg']),(a,max(20,b-7)),0,.5,(0,0,0),1)
cv2.imwrite(str(p/'rotation_fixed_annotated.png'),image)
(p/'rotation_fixed_evaluation.json').write_text(json.dumps(rows,indent=2))
