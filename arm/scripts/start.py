"""Start a new owned stack; execution remains disabled until run is invoked."""
import argparse,subprocess,time
from common import ROOT,run_path,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--case');p.add_argument('--case-register')
    p.add_argument('--detector',choices=('nine','yolo'),
                   help="perception detector: 'nine' (default; competition form) or 'yolo'")
    args=p.parse_args()
    stack=run_path(args.name)
    command=['/usr/bin/python3',str(ROOT/'harness/launch_stack.py'),args.name]
    if args.case:command+=['--case',args.case]
    if args.case_register:command+=['--case-register',args.case_register]
    if args.detector:command+=['--detector',args.detector]
    result=subprocess.run(command)
    if stack.exists():atomic_json(ROOT/'state/active_stack.json',
        {'stack_dir':str(stack),'ready':result.returncode==0,'recorded_at':time.time()})
    if result.returncode==0:print('已就绪，尚未下发抓取任务。运行：bash robot.sh run 本轮名称 --official-example')
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
