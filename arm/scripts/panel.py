"""Launch the packaged read-only desktop review UI without a command publisher."""
import argparse,time
from common import ROOT,RUNS,stack_path,run_path,atomic_json,start_owned,current,read_record

def configure(stack,episode,label):
    path=ROOT/'state/panel_config.json'
    atomic_json(path,{'stack_dir':str(stack),'episode_dir':str(episode/'episode'),
                     'label':label,'runs_root':str(RUNS)})
    return path

def ensure_panel(config):
    record=ROOT/'state/panel_pid.json'
    if record.exists():
        if current(read_record(record)):return
        record.rename(record.with_name('panel_pid_'+str(time.time_ns())+'.json'))
    out=RUNS/('review_'+str(time.time_ns()));out.mkdir()
    start_owned(record,ROOT,out/'panel.log',['/usr/bin/python3','-u',str(ROOT/'evaluation/desktop_review.py'),
        '--config',str(config),'--output',str(out)])
    deadline=time.monotonic()+10.
    while not (out/'ready.json').exists():
        if not current(read_record(record)) or time.monotonic()>deadline:
            raise RuntimeError('核对面板未就绪，请查看 '+str(out/'panel.log'))
        time.sleep(.1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stack');p.add_argument('--episode',default='manual_review')
    args=p.parse_args();config=configure(stack_path(args.stack),run_path(args.episode),'部署核对：相机、YOLO、九格、执行状态')
    ensure_panel(config);print('核对面板启动请求已发送。请在平台远程桌面查看。')

if __name__=='__main__':main()
