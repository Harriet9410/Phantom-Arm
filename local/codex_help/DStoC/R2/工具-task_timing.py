#!/usr/bin/env python3
"""Per-task time accounting and held-path accounting from one raw event stream.

Serves the review's step 3 asks:
  * 每任务的 模型等待/规划 · 动作 · 核验 时间（找 600 s 预算瓶颈）
  * 持物阶段的路径长度与分段数（"走—停—走"的停顿次数）——用于量化同侧折返

Single source: `<stack>/events/control_events.jsonl`, which the controller/driver write
with `task_id` + `status` + `target` + `time`/`monotonic`, so nothing is inferred.

Usage:
  python3 task_timing.py <stack_dir> [--task N] [--md]
  python3 task_timing.py <batch_dir>          # 逐案汇总（每案取 <batch>_<case>_stack）
"""
import json
import math
import pathlib
import statistics
import sys

HELD_PHASES = ('lift', 'transfer_clearance', 'place_above', 'place_descend')


def read_rows(path):
    rows = []
    for line in pathlib.Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def phase_of(status):
    """'transfer_clearance_03_of_09' -> 'transfer_clearance'"""
    parts = str(status).split('_')
    for i in range(len(parts), 0, -1):
        name = '_'.join(parts[:i])
        if name in HELD_PHASES or name in ('approach', 'descend', 'place_descend', 'place_retreat',
                                           'release', 'return_home', 'trial_lift'):
            return name
    return str(status)


def analyse(rows):
    per_task = {}
    for row in rows:
        task = row.get('task_id')
        if not task:
            continue
        per_task.setdefault(task, []).append(row)

    out = []
    for task in sorted(per_task):
        events = sorted(per_task[task], key=lambda r: r.get('monotonic') or 0)
        marks = {}
        for row in events:
            status = str(row.get('status'))
            marks.setdefault(status, row.get('monotonic'))
        # 关键里程碑（取首次出现）
        def at(*names):
            for name in names:
                if name in marks and isinstance(marks[name], (int, float)):
                    return marks[name]
            return None

        started = at('task_started', 'ready')
        requested = at('observation_completed', 'plan_accepted')
        planned = at('plan_accepted', 'plan_published')
        grasped = at('grasp_verified')
        released = at('released')
        verified = at('placement_verified', 'task_succeeded')

        # 运动分段：同一 phase 的连续目标
        motion = [r for r in events if r.get('target') and isinstance(r.get('target'), list)]
        held = [r for r in motion if phase_of(r.get('status')) in HELD_PHASES]
        span = 0.0
        phase_path = {}
        phase_time = {}
        for phase in HELD_PHASES:
            pts = [(r.get('monotonic'), r['target']) for r in held if phase_of(r.get('status')) == phase]
            if not pts:
                continue
            length = 0.0
            for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
                length += math.dist(p0[:2], p1[:2])
            phase_path[phase] = round(length, 3)
            phase_time[phase] = round((pts[-1][0] - pts[0][0]), 1)
            span += length
        gaps = []
        mono = [r.get('monotonic') for r in motion if isinstance(r.get('monotonic'), (int, float))]
        if len(mono) > 1:
            gaps = [b - a for a, b in zip(mono, mono[1:]) if 0 < b - a < 30]
        out.append({
            'task': task[-7:],
            'instruction': '',   # 由调用方补齐（summary.json）
            'start_to_plan_s': round((planned - started), 1) if planned and started else None,
            'plan_to_grasp_s': round((grasped - planned), 1) if grasped and planned else None,
            'held_transit_s': round((released - grasped), 1) if released and grasped else None,
            'verify_s': round((verified - released), 1) if verified and released else None,
            'total_s': round((verified - started), 1) if verified and started else None,
            'motion_segments': len(motion),
            'held_segments': len(held),
            'held_path_m': round(span, 3),
            'phase_path_m': phase_path,
            'phase_time_s': phase_time,
            'median_segment_gap_s': round(statistics.median(gaps), 2) if gaps else None,
            'outcome': 'verified' if verified else ('grasp only' if grasped else 'no grasp'),
        })
    return out


def print_table(rows, title):
    print('### %s' % title)
    print('| 任务 | 起始→计划 | 计划→抓稳 | 持物搬运 | 核验 | 合计 | 运动段 | 持物段 | 持物路径 | 段间中位 | 结局 |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for r in rows:
        print('| %s | %s | %s | %s | %s | %s | %s | %s | %s m | %s s | %s |' % (
            r['task'], r['start_to_plan_s'], r['plan_to_grasp_s'], r['held_transit_s'],
            r['verify_s'], r['total_s'], r['motion_segments'], r['held_segments'],
            r['held_path_m'], r['median_segment_gap_s'], r['outcome']))
    totals = [r['total_s'] for r in rows if r['total_s']]
    if totals:
        print()
        print('任务合计 %s s｜持物路径合计 %.3f m｜持物段合计 %s' % (
            round(sum(totals), 1), sum(r['held_path_m'] for r in rows),
            sum(r['held_segments'] for r in rows)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    target = pathlib.Path(sys.argv[1]).resolve()
    if target.name.endswith('_stack'):
        print_table(analyse(read_rows(target / 'events' / 'control_events.jsonl')), target.name)
        return 0
    # 批次目录：逐案
    for stack in sorted(target.parent.glob(target.name + '_scramble_*_stack')):
        rows = analyse(read_rows(stack / 'events' / 'control_events.jsonl'))
        if rows:
            print_table(rows, stack.name)
            print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
