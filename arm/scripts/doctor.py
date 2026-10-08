"""Read-only official-image and split-environment deployment checks."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    reports=[]
    commands=[['/usr/bin/python3',str(ROOT/'scripts/verify_image.py')],
        ['/usr/bin/python3','-c','import rospy,numpy,cv2,cv_bridge,tkinter;from PIL import Image;print("ROS/NumPy/OpenCV/Tk/Pillow OK")'],
        [os.environ['TCEI_YOLO_PY'],'-c','import torch,ultralytics,numpy,rospy;print("YOLO",ultralytics.__version__,"torch",torch.__version__)'],
        [os.environ['TCEI_NINE_PY'],'-c','import torch,transformers,PIL,rospy;print("Nine torch",torch.__version__,"transformers",transformers.__version__)'],
        ['xdpyinfo','-display',os.environ['DISPLAY']]]
    for command in commands:
        r=subprocess.run(command,capture_output=True,text=True,timeout=120)
        reports.append({'command':command,'returncode':r.returncode,'stdout':r.stdout[-20000:],'stderr':r.stderr[-4000:]})
        print(('通过' if r.returncode==0 else '未通过')+'：'+command[0])
    out={'checked_at':time.time(),'ok':all(r['returncode']==0 for r in reports),'checks':reports,
        'scope':'Read-only dependencies; does not prove loaded model inference or completed robot tasks.'}
    path=ROOT/'state'/('doctor_'+str(time.time_ns())+'.json');path.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('检查记录：'+str(path));return 0 if out['ok'] else 1

if __name__=='__main__':raise SystemExit(main())
