#!/usr/bin/env python3
"""P3 · 30 场景随机抽 5 刷题：纯九格管线泛化验证。

每个案例：robot.sh stop → robot.sh start <stack> --case <案例> →
等就绪（五类别全确认）→ robot.sh run --official-example（入口 A 5 指令）→
收成绩。全部走受控链路，与正式回合零差异。输出每案例成功率汇总 JSON。
"""
import argparse, json, random, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('/root/gpufree-data/tcei_260920v2')
REGISTER = ROOT / 'B02_AVAILABLE30_REGISTER.json'
INSTRUCTIONS = ['抓取左上方的烟雾弹，放到左侧传送带', '抓取右下方的弹夹，放到右侧传送带',
                '抓取最左方的军用手电筒，放到左侧传送带', '抓取手雷，放到右侧传送带',
                '抓取剩余的物品，放到左侧传送带']


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, executable='/bin/bash')
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return 'ERR ' + str(e)


def wait_ready(stack_dir, budget=180):
    """等感知就绪：≥5 候选、全 confirmed、无 unknown 类别。
    随机布局下提案 id 与类别不对应官方位置表，故不校验 id→class。"""
    deadline = time.monotonic() + budget
    probe = ('import json,rospy,time\n'
             'from std_msgs.msg import String\n'
             "rospy.init_node('drillready',anonymous=True)\n"
             't0=time.time()\n'
             'while time.time()-t0<%d and not rospy.is_shutdown():\n'
             '    try:\n'
             '        sc=json.loads(rospy.wait_for_message("/tcei/candidates",String,timeout=5).data)\n'
             '        cs=sc.get("candidates",[])\n'
             '        conf=[c for c in cs if c.get("identity_status")=="confirmed"]\n'
             '        known=[c for c in cs if c.get("class") not in (None,"","unknown")]\n'
             '        if len(cs)>=5 and len(conf)==5 and len(known)==5:\n'
             '            print("READY");break\n'
             '        time.sleep(4.)\n'
             '    except Exception: time.sleep(4.)\n'
             'else: print("NOT_READY")\n' % budget)
    Path('/tmp/drillready.py').write_text(probe)
    out = sh("bash -c 'source %s/scripts/env.sh && python3 /tmp/drillready.py'" % ROOT, timeout=budget + 30)
    return 'READY' in out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--stack', default='drill_stack')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--skip-start', action='store_true', help='复用当前栈（仅首次验证用）')
    args = p.parse_args()
    reg = json.loads(REGISTER.read_text(encoding='utf-8'))
    cases = reg['cases']
    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    picked = rng.sample(cases, args.count)
    results = {'started_at': time.time(), 'seed': args.seed, 'cases': []}
    print('抽中案例：', [c['case_id'] for c in picked])
    for idx, case in enumerate(picked, 1):
        cid = case['case_id']
        case_file = str(ROOT / case['case_file'])
        # 每案例独立栈名：robot.sh start 拒绝复用已存在的运行目录
        # （首次刷题实测 5 案例全部 FileExistsError，教训入库）
        stack_name = '%s_%s' % (args.stack, cid)
        row = {'case_id': cid, 'index': idx, 'stack': stack_name}
        print('\n=== [%d/%d] %s ===' % (idx, args.count, cid))
        # 1) 停旧栈 + 清残留
        sh('cd %s && timeout 60 bash robot.sh stop 2>&1 | tail -1' % ROOT)
        sh('pkill -f "sim_with_feedback[.]py"; pkill -f "controller[.]py"; '
           'pkill -f "perception[.]py"; pkill -f "nine_node[.]py"; sleep 2')
        # 2) 起新栈（案例布局）
        print('  起栈 %s（案例 %s，约 4 分钟）…' % (stack_name, cid))
        out = sh('cd %s && timeout 300 bash robot.sh start %s --case %s '
                 '--case-register %s 2>&1 | tail -3' % (ROOT, stack_name, case_file, REGISTER),
                 timeout=320)
        print('  start:', out[-200:])
        # 真失败信号=Traceback/FileExistsError/"Error:"；就绪 JSON 里的
        # "errors": [] 是成功标志，不能用 'error' in out 误判（教训入库）
        if 'Traceback' in out or 'FileExistsError' in out or 'Error:' in out:
            row.update(status='start_failed', detail=out[-300:])
            results['cases'].append(row)
            continue
        # 3) 等就绪
        print('  等待感知就绪…')
        ready = wait_ready(RUNS / stack_name)
        row['ready'] = ready
        if not ready:
            row['status'] = 'not_ready'
            results['cases'].append(row)
            print('  ✗ 未就绪，跳过')
            continue
        # 4) 跑入口 A 回合
        round_name = 'drill_%s_%d' % (cid, int(time.time()))
        print('  发射回合 %s …' % round_name)
        sh('cd %s && timeout 30 bash robot.sh run %s --official-example '
           '> /tmp/%s.out 2>&1 &' % (ROOT, round_name, round_name), timeout=40)
        # 5) 等回合结束（预算 600 + 缓冲）
        deadline = time.monotonic() + 720
        sf = RUNS / round_name / 'supervisor_finished.json'
        while time.monotonic() < deadline and not sf.exists():
            time.sleep(10)
        if not sf.exists():
            row['status'] = 'round_timeout'
            results['cases'].append(row)
            print('  ✗ 回合超时')
            continue
        data = json.loads(sf.read_text())
        ep = RUNS / round_name / 'episode' / 'summary.json'
        tasks = []
        if ep.exists():
            s = json.loads(ep.read_text())
            for t in s.get('tasks', []):
                res = t.get('result') or {}
                tasks.append({'instruction': t.get('instruction'), 'status': res.get('status'),
                              'placed_verified': res.get('placed_verified'),
                              'elapsed': t.get('elapsed_seconds')})
        ok = sum(1 for t in tasks if t['status'] == 'task_succeeded')
        row.update(status='ok', verified=data.get('verified_objects', 0),
                   tasks_ok=ok, tasks_total=len(tasks), tasks=tasks,
                   elapsed=data.get('elapsed_seconds'))
        results['cases'].append(row)
        print('  ✓ 成功 %d/%d 任务，verified=%s' % (ok, max(len(tasks), 1), row['verified']))
        # 落盘
        results['finished_at'] = time.time()
        (RUNS / 'drill_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    # 汇总
    ok_cases = sum(1 for r in results['cases'] if r.get('status') == 'ok' and (r.get('tasks_ok') or 0) >= 1)
    all_ok = sum(1 for r in results['cases'] if r.get('status') == 'ok' and r.get('tasks_ok') == 5)
    results['summary'] = {'cases_with_any_success': ok_cases, 'cases_all5': all_ok,
                          'total': len(results['cases'])}
    (RUNS / 'drill_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print('\n===== 刷题汇总 =====')
    print(json.dumps(results['summary'], ensure_ascii=False))
    for r in results['cases']:
        print('%s: %s (成功 %s/%s)' % (r['case_id'], r.get('status'), r.get('tasks_ok'), r.get('tasks_total')))


if __name__ == '__main__':
    main()
