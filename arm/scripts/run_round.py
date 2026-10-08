"""One recorded, bounded episode. Default entry always uses production control."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from common import ROOT,RUNS,run_path,stack_path,owned,start_owned,guard,write_new,atomic_json,current,read_record
from cancel import cancel_and_confirm
from panel import configure,ensure_panel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name');p.add_argument('--stack')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--official-example',action='store_true')
    group.add_argument('--instructions',type=str,help='UTF-8 JSON array of original instruction strings')
    p.add_argument('--task-source');p.add_argument('--budget',type=float,default=600.)
    # 可视化开关（10/3）：默认全关（竞赛干净桌面）；训练按需加
    p.add_argument('--viewer',action='store_true',help='弹出 bbox 实时小窗')
    p.add_argument('--panel',action='store_true',help='弹出核对面板')
    args=p.parse_args()
    if not 0<args.budget<=600:p.error('预算必须大于0且不超过600秒')
    import rospy
    from std_msgs.msg import Bool,String
    from sensor_msgs.msg import JointState
    from process_guard import identity
    from launch_stack import verify_resources,configuration
    from mission_ledger import OFFICIAL_EXAMPLE_INSTRUCTIONS
    instructions=list(OFFICIAL_EXAMPLE_INSTRUCTIONS) if args.official_example else json.loads(open(args.instructions,encoding='utf-8-sig').read())
    if not isinstance(instructions,list) or not instructions or any(not isinstance(x,str) or not x.strip() for x in instructions):
        raise ValueError('指令文件必须是非空字符串数组')
    stack=stack_path(args.stack);run=run_path(args.name)
    if run.exists() or (stack/'round_started.lock').exists():raise RuntimeError('本轮名称已使用，或当前场景已运行；保留原记录并新建场景')
    if not all(owned(stack).values()):raise RuntimeError('四个运行节点尚未就绪，请先start/status')
    if not (stack/'warmup_complete.txt').exists():raise RuntimeError('预热未完成')
    manifest=verify_resources(configuration(os.environ))
    if shutil.disk_usage(RUNS).free<8*1024**3:raise RuntimeError('数据盘少于8GiB，暂停新轮次以保护完整证据')
    rospy.init_node('tcei_package_supervisor',anonymous=True,disable_signals=True)
    flags=('/tcei_controller/execute','/tcei_nine/execute')
    if any(rospy.get_param(k,False) is not False for k in flags):raise RuntimeError('执行开关已开启，禁止重叠运行')
    capture=rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=8)
    joints=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=8)
    if capture.data or len(joints.position)!=8 or max(abs(v) for v in joints.position[-2:])>=.001:
        raise RuntimeError('初始夹爪不是已张开的空载状态')
    scene=json.loads(rospy.wait_for_message('/tcei/candidates',String,timeout=8).data)
    if not 0<=time.time()-scene['observed_at']<2:raise RuntimeError('相机数据过期')
    run.mkdir();(run/'pids').mkdir();(run/'logs').mkdir()
    write_new(run/'before_scene.json',scene)
    write_new(run/'instructions.json',instructions)
    write_new(run/'runtime_manifest.json',manifest)
    write_new(stack/'round_started.lock',{'run':str(run),'created_at':time.time(),'task_source':args.task_source})
    atomic_json(ROOT/'state/active_episode.json',{'name':args.name,'stack':stack.name,'started_at':time.time()})
    result={'started_at':time.time(),'runtime':manifest['version'],'instructions':instructions,
            'scope':'normal production control; no injected grasp, release, observation, or physical faults'}
    child=None;recorders=[];interrupted=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:interrupted.append(s))
    try:
        length=str(min(1800,int(args.budget)+150))
        commands={
            'rgbd':['/usr/bin/python3','-u',str(ROOT/'evaluation/record_trial_rgbd.py'),'_output:='+str(run/'rgbd'),'_duration:='+length],
            'scalars':['/usr/bin/python3','-u',str(ROOT/'evaluation/record_scalar_evidence.py'),'--output',str(run/'scalars'),'--duration',length]}
        for name,command in commands.items():
            start_owned(run/'pids'/(name+'.json'),ROOT,run/'logs'/(name+'.log'),command);recorders.append(name)
        until=time.monotonic()+120.
        while not ((run/'rgbd/reference.jpg').exists() and (run/'scalars/ready.json').exists()):
            if interrupted:raise InterruptedError('operator interrupted preparation')
            if time.monotonic()>until or not all(current(read_record(run/'pids'/(n+'.json'))) for n in recorders):
                # 120 秒：回合结束后仿真正在重载场景（1~2 分钟），相机话题恢复
                # 需要时间；30 秒曾导致连续回合"证据记录器未就绪"空跑。
                raise RuntimeError('证据记录器未就绪')
            time.sleep(.1)
        # 可视化（10/3）：面板从无条件自动弹改为 --panel 门控；--viewer 弹 bbox 小窗
        # （只读旁观件，进程挂回合会话下，回合结束后随窗口关闭自然回收）
        if args.panel:
            ensure_panel(configure(stack,run,'实际执行：'+args.name))
        if args.viewer:
            subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'evaluation/bbox_window.py')],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        command=['/usr/bin/python3','-u',str(ROOT/'tcei_stack/run_episode.py'),
            '--output',str(run/'episode'),'--budget',str(args.budget),
            '--task-source',args.task_source or ('official_example' if args.official_example else 'custom')]
        for instruction in instructions:command+=['--instruction',instruction]
        result['command']=command
        for key in flags:rospy.set_param(key,True)
        with (run/'logs/episode.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=str(ROOT),env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=identity(child.pid)
        if record is None:raise RuntimeError('任务入口提前退出')
        record.update(command=command,cwd=str(ROOT),started_at=time.time());write_new(run/'pids/episode.json',record)
        deadline=time.monotonic()+args.budget+60.
        while child.poll() is None:
            if interrupted:raise InterruptedError('operator interrupted task')
            if time.monotonic()>=deadline:raise TimeoutError('episode supervisor watchdog expired')
            time.sleep(.2)
        result['returncode']=child.returncode
        summary=json.loads((run/'episode/summary.json').read_text())
        result['episode_status']=summary['status'];result['verified_objects']=summary.get('verified_objects',0)
        result['elapsed_seconds']=summary.get('elapsed_seconds')
        if summary.get('code_changed_during_episode'):raise RuntimeError('运行中源码发生变化')
        if summary.get('safe_stop',{}).get('state') in ('fault','unconfirmed'):
            result['stop_attention_required']=True
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        result['supervisor_stop']=cancel_and_confirm(str(error))
        result['stop_attention_required']=result['supervisor_stop']['state']!='stopped'
    finally:
        for key in flags:rospy.set_param(key,False)
        result['execution_disabled']=all(rospy.get_param(k) is False for k in flags)
        if child is not None and child.poll() is None and (run/'pids/episode.json').exists():
            r=guard('stop',run/'pids/episode.json','--timeout','20')
            result['episode_cleanup']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        time.sleep(2.)
        result['recorders']={}
        for name in recorders:
            r=guard('stop',run/'pids'/(name+'.json'),'--timeout','20')
            closed=run/name/'closed.json'
            result['recorders'][name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,
                                     'closed':json.loads(closed.read_text()) if closed.exists() else None}
        result['evidence_complete']=(len(result['recorders'])==2 and
            all(x['returncode']==0 and x['closed'] for x in result['recorders'].values()) and
            result['recorders']['rgbd']['closed'].get('complete') is True)
        result['finished_at']=time.time();write_new(run/'supervisor_finished.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        rospy.signal_shutdown('bounded episode closed')
    return 0 if result.get('episode_status')=='succeeded' and result.get('evidence_complete') else 1

if __name__=='__main__':raise SystemExit(main())
