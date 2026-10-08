#!/usr/bin/env python3
"""drill5 v2 · 官方五条指令逐条连续刷题（每条独立短回合）。

v1 教训（2026-10-02 实测）：
1. episode_driver 对 rejected 按设计整轮终止 + controller cancelled 锁存只能
   重启清除——单回合内被拒即停，架构上无解。逐条循环是唯一不动竞赛代码的路。
2. 共享栈不重载场景：jaka_sim 只在启动时 open_stage 一次，交付物残留
   （n=5→4→3），后续指令指向已交付物体 → 空间校验拒绝。**每条指令需要原始
   5 物体场景 = 每轮全栈重启**（fresh 模式，默认）。
3. reset 竞态：先 stop_reset 再杀 controller → 旧 controller 收到 reset ack
   时 cancelled 仍置位 → 立即以新 id 重锁 sim（controller.py:488-491）→
   下轮观测规划被拒（clear_basket_for_observation）。**修复：先杀 controller
   再清 sim latch，带验证循环**（shared 模式）。

用法：
  bash robot.sh drill5 [--mode fresh|shared] [--only N] [--ready-budget 120]
  fresh（默认）：每条指令 stop+start 全栈重启（新场景，~5 分钟/条）
  shared：同栈复位循环（快，但场景残留，仅用于机制验证）
输出：/root/gpufree-data/tcei_260920v2/drill5_results.json（增量落盘）
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
BUDGETS = [210, 210, 210, 210, 560]
LABELS = ['烟雾弹→左带', '弹夹→右带', '手电筒→左带', '手雷→右带', '剩余物品→左带']

# shared 模式复位：先杀 controller（cancelled 锁存随进程消亡，杜绝重锁竞态），
# 再清 sim 侧 StopLatch（带验证循环），最后起新 controller 并回写 pids。
RESET_SCRIPT = r'''import json, time, os, subprocess
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
lock = S + '/round_started.lock'
if os.path.exists(lock):
    os.remove(lock); log('lock cleared')
else:
    log('lock clean')
# 先杀 controller：它的 cancelled 锁存在收到 reset ack 后会以新 id 重锁 sim
subprocess.run('pkill -9 -f "controller[.]py"', shell=True)
time.sleep(2)
log('controller killed')
# 清 sim 侧 StopLatch：验证循环，直到 stop_ack 不再显示活跃停止
for attempt in range(3):
    try:
        m = rospy.wait_for_message('/tcei/stop_ack', String, timeout=4)
        d = json.loads(m.data); sid = d.get('id'); state = d.get('state')
    except Exception as e:
        log('stop_ack', type(e).__name__); sid = None; state = None
    if not sid or state in ('reset', None):
        log('sim latch clear (attempt %d, state=%s)' % (attempt, state)); break
    pub = rospy.Publisher('/tcei/stop_reset', String, queue_size=1, latch=True)
    time.sleep(.5); pub.publish(String(json.dumps({'id': sid}))); time.sleep(2)
    log('sim stop_reset %s (attempt %d)' % (sid[:8], attempt))
else:
    log('sim latch verify timeout (continuing)')
# 起新 controller
logf = open(S + '/logs/controller.log', 'a')
subprocess.Popen(
    ['/usr/bin/python3', '-u', 'controller.py',
     '_execute:=false', '_prepare_observation:=false', '_require_planner_feedback:=true',
     '_log_dir:=' + S + '/events',
     '_robot_projection_calibration:=/root/tcei_final_v2_23/tcei_260920v2/calibration/robot_projection.static_checked_v1.json'],
    cwd='/root/tcei_final_v2_23/tcei_260920v2/tcei_stack',
    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True)
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
EXPECT = __EXPECT__
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
        # fresh 模式要求候选数达到期望（原始场景物体数）——场景未就绪时物体不全
        if fresh and len(cs) >= EXPECT and not unknown and not low:
            print('READY n=' + str(len(cs))); break
        if sig != last:
            print('WAIT ' + str(sig)); last = sig
        time.sleep(3)
    except Exception as e:
        print('WAIT exc ' + type(e).__name__); time.sleep(3)
else:
    print('NOT_READY')
'''

# shared 模式（同场景连测）就绪：交付后场景变化 → 必须等 tracker 类别签名稳定
# 才发射，否则空间校验撞"boundary uncertainty"墙（视频轮 task-02 三连拒实证）。
SHARED_READY_TEMPLATE = r'''import json, time
import rospy
from std_msgs.msg import String, Bool
rospy.init_node('drill5ready_sh', anonymous=True)
BUDGET = __BUDGET__
TARGET = __TARGET__
STABLE_N = 6
t0 = time.time()
last_sig = None
stable = 0
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
        has_target = (not TARGET) or any(c.get('class') == TARGET for c in cs)
        sig = tuple(sorted(str(c.get('class', '?')) for c in cs))
        if fresh and cs and not unknown and not low and has_target:
            if sig == last_sig:
                stable += 1
            else:
                stable = 0
                last_sig = sig
            print('WAIT stable=%d/%d n=%d sig=%s' % (stable, STABLE_N, len(cs), list(sig)))
            if stable >= STABLE_N:
                print('READY n=%d sig=%s' % (len(cs), list(sig)))
                break
        else:
            stable = 0
            print('WAIT fresh=%s n=%d unk=%d low=%d tgt=%s' % (fresh, len(cs), len(unknown), len(low), has_target))
        time.sleep(5)
    except Exception as e:
        print('WAIT exc ' + type(e).__name__); time.sleep(3)
else:
    print('NOT_READY')
'''

# 每条指令的目标类别（shared 模式就绪门校验目标仍在场景中）
TARGET_CLASS = {1: 'Smokegrenade', 2: 'Magazine', 3: 'Torch', 4: 'Grenade', 5: None}


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


def wait_ready(budget, expect=1):
    Path('/tmp/drill5_ready.py').write_text(
        READY_TEMPLATE.replace('__BUDGET__', str(int(budget))).replace('__EXPECT__', str(int(expect))))
    out = sh("bash -c 'source %s/scripts/env.sh && /usr/bin/python3 /tmp/drill5_ready.py'" % ROOT,
             timeout=budget + 30)
    if 'OCCUPIED' in out:
        return 'OCCUPIED', out
    if 'READY' in out:
        return 'READY', out
    return 'NOT_READY', out


def wait_ready_shared(budget, target=None):
    """shared 模式就绪：目标类别在场 + 类别签名连续 6 次（30s）不变才放行。"""
    tgt = repr(target) if target else 'None'
    Path('/tmp/drill5_ready_sh.py').write_text(
        SHARED_READY_TEMPLATE.replace('__BUDGET__', str(int(budget))).replace('__TARGET__', tgt))
    out = sh("bash -c 'source %s/scripts/env.sh && /usr/bin/python3 /tmp/drill5_ready_sh.py'" % ROOT,
             timeout=budget + 30)
    if 'OCCUPIED' in out:
        return 'OCCUPIED', out
    if 'READY' in out:
        return 'READY', out
    return 'NOT_READY', out


def save(results):
    results['updated_at'] = time.time()
    (RUNS / 'drill5_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))


def fresh_cycle(stack_name, idx, instr, budget, ready_budget):
    """全栈重启一轮：stop → start（新场景）→ 等就绪 → 单指令回合。"""
    row = {'index': idx, 'instruction': instr, 'label': LABELS[idx - 1],
           'budget': budget, 'mode': 'fresh'}
    print('  stop 旧栈…')
    sh('cd %s && timeout 90 bash robot.sh stop 2>&1 | tail -2' % ROOT, timeout=100)
    sh('pkill -9 -f "sim_with_feedback[.]py"; pkill -9 -f "controller[.]py"; '
       'pkill -9 -f "perception[.]py"; pkill -9 -f "nine_node[.]py"; sleep 3', timeout=20)
    # 清历史锁
    sh('find %s -name round_started.lock -delete 2>/dev/null' % RUNS, timeout=20)
    print('  起新栈 %s（新场景，~4 分钟）…' % stack_name)
    out = sh('cd %s && timeout 360 bash robot.sh start %s 2>&1 | tail -3' % (ROOT, stack_name), timeout=380)
    if 'Traceback' in out or 'Error' in out:
        row.update(status='start_failed', detail=out[-300:])
        return row
    print('  等待感知就绪（期望 5 物体）…')
    state, probe = wait_ready(ready_budget, expect=5)
    row['ready'] = state; row['ready_tail'] = probe[-300:]
    print('  就绪：', state)
    if state != 'READY':
        row['status'] = 'occupied_aborted' if state == 'OCCUPIED' else 'not_ready'
        return row
    return run_single_round(row, idx, instr, budget)


def shared_cycle(idx, instr, budget, ready_budget):
    """同栈复位循环：修复版 reset（先杀 controller）→ 等就绪 → 单指令回合。"""
    row = {'index': idx, 'instruction': instr, 'label': LABELS[idx - 1],
           'budget': budget, 'mode': 'shared'}
    out = reset_round()
    row['reset_tail'] = out[-400:]
    print('  复位：', [l for l in out.splitlines() if l][:6])
    if 'controller FAIL' in out:
        row['status'] = 'reset_failed'
        return row
    state, probe = wait_ready_shared(ready_budget, TARGET_CLASS[idx])
    row['ready'] = state; row['ready_tail'] = probe[-300:]
    print('  就绪：', state)
    if state != 'READY':
        row['status'] = 'occupied_aborted' if state == 'OCCUPIED' else 'not_ready'
        return row
    return run_single_round(row, idx, instr, budget)


def run_single_round(row, idx, instr, budget):
    round_name = 'drill5_%d_%d' % (idx, int(time.time()))
    instr_file = RUNS / ('drill5_instr_%d.json' % (idx))
    instr_file.write_text(json.dumps([instr], ensure_ascii=False), encoding='utf-8')
    print('  发射回合 %s …' % round_name)
    sh('cd %s && nohup bash robot.sh run %s --instructions %s --budget %d > /tmp/%s.out 2>&1 &'
       % (ROOT, round_name, instr_file, budget, round_name), timeout=50)
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
        return row
    if not sf.exists():
        row['status'] = 'round_timeout'
        sh('pkill -9 -f "run_episode[.]py"; pkill -9 -f "record_trial_rgbd"; pkill -9 -f "record_scalar_evidence"',
           timeout=20)
        return row
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
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['fresh', 'shared'], default='fresh')
    p.add_argument('--only', type=int, default=None, help='只跑第 N 条指令（1-5）')
    p.add_argument('--ready-budget', type=int, default=120)
    args = p.parse_args()
    results = {'started_at': time.time(), 'mode': args.mode,
               'stack': 'fresh-per-round' if args.mode == 'fresh'
               else json.loads(STACK_STATE.read_text())['stack_dir'],
               'budgets': BUDGETS, 'rounds': []}
    print('模式：', args.mode)
    for i, instr in enumerate(INSTRUCTIONS, 1):
        if args.only and i != args.only:
            continue
        print('\n=== [%d/5] %s（预算 %ds）===' % (i, instr, BUDGETS[i - 1]))
        if args.mode == 'fresh':
            stack_name = 'drill5f_%d_%d' % (i, int(time.time()) % 100000)
            row = fresh_cycle(stack_name, i, instr, BUDGETS[i - 1], args.ready_budget)
        else:
            row = shared_cycle(i, instr, BUDGETS[i - 1], args.ready_budget)
        results['rounds'].append(row); save(results)
        mark = '✓' if row.get('status') == 'succeeded' else '✗'
        extra = ('— ' + str(row.get('reason') or row.get('early_error') or '')[:80]) if (
            row.get('reason') or row.get('early_error')) else ''
        print('  %s %s（placed=%s，%ss）%s' % (mark, row.get('status'), row.get('placed_verified'),
                                              round(row.get('task_elapsed') or 0, 1), extra))
    ok = sum(1 for r in results['rounds'] if r.get('status') == 'succeeded')
    results['summary'] = {'succeeded': ok, 'attempted': len(results['rounds']),
                          'per_instruction': {r['label']: r.get('status') for r in results['rounds']}}
    save(results)
    print('\n===== 五条连续刷题汇总（%s 模式）=====' % args.mode)
    for r in results['rounds']:
        reason = ('（' + str(r.get('reason') or r.get('early_error') or '')[:70] + '）') if (
            r.get('reason') or r.get('early_error')) else ''
        print('指令%d %-12s : %s%s' % (r['index'], r['label'], r.get('status'), reason))
    print('成功 %d/%d  结果文件：%s' % (ok, len(results['rounds']), RUNS / 'drill5_results.json'))


if __name__ == '__main__':
    main()
