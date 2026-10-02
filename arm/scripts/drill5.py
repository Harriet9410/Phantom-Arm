#!/usr/bin/env python3
"""drill5.py · 官方五条指令逐条连续刷题（同栈短回合 + 回合间完整复位）。

为什么不是单回合五连发：episode_driver 对 rejected（语义/空间校验拒绝）
按设计整轮终止（episode_driver.py:295-297 抛 EpisodeFailure），且
controller 的 cancelled 锁存只能重启进程清除（官方注释：绝不为开始下一个
队列项而清锁存）——单回合内被拒即停，架构上无解（2026-10-02 分析留痕）。

本脚本：每条指令一个独立短回合（同栈复用，sim 不重载），回合间执行完整
复位（杀残留 → stop_reset → 重启 controller 并回写 pids → 清锁 → 等
感知就绪 + 夹爪张开），单条失败不拖停后续指令。全部走受控链路
robot.sh run --instructions，与正式回合零差异。

用法： bash robot.sh drill5 [--only N] [--ready-budget 90]
输出： /root/gpufree-data/tcei_260920v2/drill5_results.json（增量落盘）
"""
import argparse, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('/root/gpufree-data/tcei_260920v2')
STACK_STATE = ROOT / 'state/active_stack.json'
INSTRUCTIONS = [
    '抓取左上方的烟雾弹，放到左侧传送带',
    '抓取右下方的弹夹，放到右侧传送带',
    '抓取最左方的军用手电筒，放到左侧传送带',
    '抓取手雷，放到右侧传送带',
    '抓取剩余的物品，放到左侧传送带',
]
# 单物体指令给 210s（含场景变更后 ~50s 稳定门 + 动作 + 验证）；
# 指令5 清剩余（最多 4 物体 × ~60s）单独放宽到 420s。
BUDGETS = [210, 210, 210, 210, 420]
LABELS = ['烟雾弹→左带', '弹夹→右带', '手电筒→左带', '手雷→右带', '剩余物品→左带']

RESET_SCRIPT = r'''import json, time, os, subprocess, glob
import rospy
from std_msgs.msg import String
S = json.load(open('/root/tcei_final_v2_23/tcei_260920v2/state/active_stack.json'))['stack_dir']
t0 = time.time()
def log(*a):
    print('[%.1fs]' % (time.time() - t0), *a, flush=True)
rospy.init_node('drill5_reset', anonymous=True)
for k in ('/tcei_controller/execute', '/tcei_nine/execute'):
    try: rospy.set_param(k, False)
    except Exception: pass
subprocess.run('pkill -9 -f "run_episode[.]py"; pkill -9 -f "record_trial_rgbd"; pkill -9 -f "record_scalar_evidence"', shell=True)
# 锁先清：即使后面 controller 重启失败，下一轮也不会被锁拦
lock = S + '/round_started.lock'
if os.path.exists(lock):
    os.remove(lock); log('lock cleared')
else:
    log('lock clean')
try:
    m = rospy.wait_for_message('/tcei/stop_ack', String, timeout=4)
    d = json.loads(m.data); sid = d.get('id')
    if sid:
        pub = rospy.Publisher('/tcei/stop_reset', String, queue_size=1, latch=True)
        time.sleep(.5); pub.publish(String(json.dumps({'id': sid}))); time.sleep(1.5)
        log('latch reset', sid[:8])
    else:
        log('latch clean')
except Exception as e:
    log('latch', e)
subprocess.run('pkill -9 -f "controller[.]py"', shell=True)
time.sleep(2)
# Popen + start_new_session：绝不用 bash -c '... & echo' + capture_output
# （管道被后台 controller 继承，subprocess.run 会挂到超时——首跑教训）
logf = open(S + '/logs/controller.log', 'a')
subprocess.Popen(
    ['/usr/bin/python3', '-u', 'controller.py',
     '_execute:=false', '_prepare_observation:=false', '_require_planner_feedback:=true',
     '_log_dir:=' + S + '/events',
     '_robot_projection_calibration:=/root/tcei_final_v2_23/tcei_260920v2/calibration/robot_projection.static_checked_v1.json'],
    cwd='/root/tcei_final_v2_23/tcei_260920v2/tcei_stack',
    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True)
env = dict(os.environ)
pid = None
for _ in range(20):
    time.sleep(.5)
    r = subprocess.run('pgrep -f "controller[.]py" | head -1', shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        pid = int(r.stdout.strip()); break
if pid:
    st = open('/proc/%d/stat' % pid).read(); f = st[st.rfind(')') + 2:].split()
    rec = json.load(open(S + '/pids/controller.json'))
    rec['pid'] = pid; rec['pgid'] = int(f[2]); rec['start_ticks'] = int(f[19]); rec['started_at'] = time.time()
    json.dump(rec, open(S + '/pids/controller.json', 'w'), indent=2)
    log('controller restarted', pid)
else:
    log('controller FAIL')
log('RESET_DONE')
'''

READY_TEMPLATE = r'''import json, time
import rospy
from std_msgs.msg import String, Bool
rospy.init_node('drill5ready', anonymous=True)
BUDGET = __BUDGET__
t0 = time.time()
last = None
while time.time() - t0 < BUDGET and not rospy.is_shutdown():
    try:
        g = rospy.wait_for_message('/Jaka/gripper_is_captured', Bool, timeout=3)
        if g.data:
            print('OCCUPIED'); break
        sc = json.loads(rospy.wait_for_message('/tcei/candidates', String, timeout=3).data)
        cs = sc.get('candidates', [])
        fresh = time.time() - sc.get('observed_at', 0) < 2
        unknown = [c for c in cs if str(c.get('class', '')).upper() == 'UNKNOWN']
        low = [c for c in cs if c.get('confidence', 0) < 0.5]
        sig = (len(cs), len(unknown), len(low), fresh)
        if fresh and len(cs) >= 1 and not unknown and not low:
            print('READY n=' + str(len(cs))); break
        if sig != last:
            print('WAIT ' + str(sig)); last = sig
        time.sleep(3)
    except Exception as e:
        print('WAIT exc ' + type(e).__name__); time.sleep(3)
else:
    print('NOT_READY')
'''


def sh(cmd, timeout=180):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, executable='/bin/bash')
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return 'ERR ' + str(e)


def reset_round():
    Path('/tmp/drill5_reset.py').write_text(RESET_SCRIPT)
    return sh("bash -c 'source %s/scripts/env.sh && /usr/bin/python3 /tmp/drill5_reset.py'" % ROOT, timeout=90)


def wait_ready(budget):
    Path('/tmp/drill5_ready.py').write_text(READY_TEMPLATE.replace('__BUDGET__', str(int(budget))))
    out = sh("bash -c 'source %s/scripts/env.sh && /usr/bin/python3 /tmp/drill5_ready.py'" % ROOT, timeout=budget + 30)
    if 'OCCUPIED' in out:
        return 'OCCUPIED', out
    if 'READY' in out:
        return 'READY', out
    return 'NOT_READY', out


def save(results):
    results['updated_at'] = time.time()
    (RUNS / 'drill5_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', type=int, default=None, help='只跑第 N 条指令（1-5）')
    p.add_argument('--ready-budget', type=int, default=90)
    args = p.parse_args()
    results = {'started_at': time.time(),
               'stack': json.loads(STACK_STATE.read_text())['stack_dir'],
               'budgets': BUDGETS, 'rounds': []}
    print('栈：', results['stack'])
    for i, instr in enumerate(INSTRUCTIONS):
        if args.only and i + 1 != args.only:
            continue
        row = {'index': i + 1, 'instruction': instr, 'label': LABELS[i], 'budget': BUDGETS[i]}
        print('\n=== [%d/5] %s（预算 %ds）===' % (i + 1, instr, BUDGETS[i]))
        # 1) 完整复位
        out = reset_round()
        row['reset_tail'] = out[-400:]
        print('  复位：', [l for l in out.splitlines() if l][:6])
        if 'controller FAIL' in out:
            row['status'] = 'reset_failed'
            results['rounds'].append(row); save(results)
            print('  ✗ controller 重启失败，中止')
            break
        # 2) 等就绪（夹爪张开 + 候选新鲜 + 无 UNKNOWN + 无低置信）
        state, probe = wait_ready(args.ready_budget)
        row['ready'] = state; row['ready_tail'] = probe[-300:]
        print('  就绪：', state)
        if state != 'READY':
            row['status'] = 'occupied_aborted' if state == 'OCCUPIED' else 'not_ready'
            results['rounds'].append(row); save(results)
            if state == 'OCCUPIED':
                print('  ✗ 夹爪占用（上轮遗留抓持），中止后续（需人工检查）')
                break
            print('  ✗ 感知未就绪，跳过本条')
            continue
        # 3) 发射单指令短回合（不设 timeout 包壳：run_round 自带 budget+60 看门狗）
        round_name = 'drill5_%d_%d' % (i + 1, int(time.time()))
        instr_file = RUNS / ('drill5_instr_%d.json' % (i + 1))
        instr_file.write_text(json.dumps([instr], ensure_ascii=False), encoding='utf-8')
        budget = BUDGETS[i]
        print('  发射回合 %s …' % round_name)
        sh('cd %s && nohup bash robot.sh run %s --instructions %s --budget %d > /tmp/%s.out 2>&1 &'
           % (ROOT, round_name, instr_file, budget, round_name), timeout=50)
        # 4) 等回合结束（预算 + 复位/证据缓冲 180s）；.out 出现 Traceback 提前判失败
        sf = RUNS / round_name / 'supervisor_finished.json'
        out_file = Path('/tmp/%s.out' % round_name)
        deadline = time.monotonic() + budget + 180
        early_fail = False
        while time.monotonic() < deadline and not sf.exists():
            if not early_fail and out_file.exists():
                txt = out_file.read_text(errors='replace')
                if 'Traceback' in txt or 'RuntimeError' in txt:
                    early_fail = True
                    row['early_error'] = txt[-400:]
                    break
            time.sleep(8)
        row['round'] = round_name
        if early_fail:
            row['status'] = 'round_failed'
            results['rounds'].append(row); save(results)
            print('  ✗ 回合启动失败（见 early_error）')
            continue
        if not sf.exists():
            row['status'] = 'round_timeout'
            sh('pkill -9 -f "run_episode[.]py"; pkill -9 -f "record_trial_rgbd"; pkill -9 -f "record_scalar_evidence"', timeout=20)
            results['rounds'].append(row); save(results)
            print('  ✗ 回合超时（已强杀残留，下轮复位兜底）')
            continue
        sup = json.loads(sf.read_text())
        ep = RUNS / round_name / 'episode' / 'summary.json'
        if ep.exists():
            s = json.loads(ep.read_text())
            tasks = s.get('tasks', [])
            t = tasks[0] if tasks else {}
            res = t.get('result') or {}
            row.update(episode_status=s.get('status'), verified=s.get('verified_objects'),
                       task_status=res.get('status'), placed_verified=res.get('placed_verified'),
                       task_elapsed=res.get('elapsed') or t.get('elapsed_seconds'),
                       reason=res.get('reason'), supervisor_elapsed=sup.get('elapsed_seconds'))
        else:
            row.update(episode_status=None, supervisor_error=sup.get('error'),
                       status='preflight_failed' if sup.get('error') else 'no_episode')
        if 'status' not in row:
            row['status'] = {'task_succeeded': 'succeeded', 'rejected': 'rejected',
                             'task_failed': 'task_failed'}.get(
                row.get('task_status'), row.get('episode_status') or 'unknown')
        results['rounds'].append(row); save(results)
        mark = '✓' if row['status'] == 'succeeded' else '✗'
        extra = ('— ' + str(row.get('reason'))[:80]) if row.get('reason') else ''
        print('  %s %s（placed=%s，%ss）%s' % (mark, row['status'], row.get('placed_verified'),
                                              round(row.get('task_elapsed') or 0, 1), extra))
    # 汇总
    ok = sum(1 for r in results['rounds'] if r.get('status') == 'succeeded')
    results['summary'] = {'succeeded': ok, 'attempted': len(results['rounds']),
                          'per_instruction': {r['label']: r.get('status') for r in results['rounds']}}
    save(results)
    print('\n===== 五条连续刷题汇总 =====')
    for r in results['rounds']:
        reason = ('（' + str(r.get('reason'))[:70] + '）') if r.get('reason') else ''
        print('指令%d %-12s : %s%s' % (r['index'], r['label'], r.get('status'), reason))
    print('成功 %d/%d  结果文件：%s' % (ok, len(results['rounds']), RUNS / 'drill5_results.json'))


if __name__ == '__main__':
    main()
