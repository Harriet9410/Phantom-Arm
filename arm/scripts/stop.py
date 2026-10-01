"""Stop only this package's recorded workers after native stop confirmation."""
import argparse,time
from common import stack_path,owned,guard,write_new
from cancel import cancel_and_confirm

def main():
    p=argparse.ArgumentParser();p.add_argument('--stack');args=p.parse_args();stack=stack_path(args.stack)
    live=owned(stack);result={'started_at':time.time(),'workers_before':live,'stops':[]}
    if live['sim'] and live['controller']:
        proof=cancel_and_confirm('package stack shutdown')
        result['stop_confirmation']=proof
        write_new(stack/('shutdown_stop_'+str(time.time_ns())+'.json'),proof)
        if proof['state']!='stopped':raise RuntimeError('尚未确认停止；保留仿真和日志，请查看核对面板')
    elif live['sim']:
        raise RuntimeError('仿真存活但控制器缺失，不能自动确认停止；先检查进程和桌面')
    # A whole simulation shutdown, never a reset inside an active round.
    for name in ('nine','controller','perception','sim'):
        r=guard('stop',stack/'pids'/(name+'.json'),'--timeout','20')
        result['stops'].append({'name':name,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        if r.returncode:raise RuntimeError('进程仍在退出：'+name)
    result['finished_at']=time.time();write_new(stack/('shutdown_'+str(time.time_ns())+'.json'),result)
    print('本轮机械臂、识别和九格进程已退出。历史数据已保留。')

if __name__=='__main__':main()
