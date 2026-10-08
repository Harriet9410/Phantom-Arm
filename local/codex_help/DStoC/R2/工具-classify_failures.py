#!/usr/bin/env python3
"""Give every failed CASE one root-cause category plus the evidence path, and print the
raw per-task verdict strings the runner recorded.

Deliverable served (review §六 step 1): per failure, ONE primary cause drawn from
{识别类别, 身份绑定, 语义拒单, 抓取与掉件, 放置核验, 超时} with the file that proves it —
so a failure is never explained from a single video guess.

Reads only what a batch already wrote (stdlib only, no ROS):
  <batch>/campaign.json                        case status / terminal reason / task_results / supervisor
  <batch>_<case>_round/episode/summary.json    per-task instruction list
  <batch>_<case>_stack/logs/perception.log     candidate lines (id:class:identity:area)  ← 识别证据
  <batch>_<case>_stack/events/control_events.jsonl   raw failing event text

Usage: python3 classify_failures.py <batch_dir> [--json]
"""
import json
import pathlib
import re
import sys

CATEGORIES = [
    ('识别类别', re.compile(r'category disagreement|recognition capability|unknown objects cannot|class changed', re.I)),
    ('身份绑定', re.compile(r'identity|ambiguous|unconfirmed|stable_id|correspondence|multiple_body_cores', re.I)),
    ('语义拒单', re.compile(r'implicit quantity|contradicts explicit instruction|violates spatial|not the unique|'
                            r'model side|unsupported exact|remaining set|plan does not|target reserved', re.I)),
    ('抓取与掉件', re.compile(r'holding_feedback_lost|drop_recovery|empty grasp|sustained contact|slip|preload|grasp_', re.I)),
    ('放置核验', re.compile(r'unverified|pending_verification|placement|conveyor', re.I)),
    ('超时', re.compile(r'budget|deadline|timeout|timed out|insufficient', re.I)),
]


def classify(text):
    for name, pattern in CATEGORIES:
        if pattern.search(str(text or '')):
            return name
    return '未归类'


def read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def read_lines(path):
    try:
        return pathlib.Path(path).read_text(encoding='utf-8').splitlines()
    except Exception:
        return []


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    batch = pathlib.Path(sys.argv[1]).resolve()
    if not batch.is_dir():
        print('batch dir not readable:', batch)
        return 1
    campaign = read_json(batch / 'campaign.json') or {}
    runs_root = batch.parent
    cases, details = [], []

    for result in campaign.get('results', []):
        case = result.get('case_id')
        round_dir = runs_root / (batch.name + '_' + case + '_round')
        stack_dir = runs_root / (batch.name + '_' + case + '_stack')
        summary = read_json(round_dir / 'episode' / 'summary.json') or {}
        supervisor = result.get('supervisor') or {}
        reason = str(result.get('reason') or '')
        verified = result.get('verified_objects')
        tasks = summary.get('tasks') or []
        entries = [e if isinstance(e, str) else json.dumps(e, ensure_ascii=False) for e in (result.get('task_results') or [])]
        perception = read_lines(stack_dir / 'logs' / 'perception.log')
        cand_lines = [l for l in perception if 'candidates=' in l]

        cases.append({
            'case': case, 'status': result.get('status'),
            'verified': verified, 'episode': supervisor.get('episode_status'),
            'elapsed': round(result.get('elapsed_seconds') or supervisor.get('elapsed_seconds') or 0),
            'cause': classify(reason) if reason else '—（无终态拒单）',
            'reason': reason[:170],
            'evidence': str(stack_dir.relative_to(runs_root)) + '/logs/perception.log',
            'cand_tail': (cand_lines[-1].strip()[:200] if cand_lines else '(no candidate line)'),
        })
        details.append({'case': case, 'tasks': [t.get('instruction') or '' for t in tasks],
                        'task_results': entries,
                        'round': str(round_dir.relative_to(runs_root)),
                        'stack': str(stack_dir.relative_to(runs_root))})

    print('# 逐案失败归类 — 批次 %s' % batch.name)
    print('# 批次状态: %s（%s/%s 案完成）| autoVerified=%s' % (
        campaign.get('status'), len(campaign.get('results', [])), campaign.get('planned_rounds'),
        campaign.get('automatic_verified_rounds')))
    print()
    print('| 案例 | 验证 | episode | 用时 | 单一主因 | 终态原文 |')
    print('| --- | ---: | --- | ---: | --- | --- |')
    for row in cases:
        print('| %s | %s/5 | %s | %ss | **%s** | %s |' % (
            row['case'], row['verified'], row['episode'], row['elapsed'], row['cause'],
            row['reason'].replace('|', '/') or '-'))
    print()
    print('## 证据路径（识别类看候选行，其余看回合事件）')
    for row in cases:
        print('- **%s** `%s`' % (row['case'], row['evidence']))
        print('  - 候选行尾部：`%s`' % row['cand_tail'].replace('|', '/'))
    print()
    print('## 逐条原文（runner 记录的 task_results，按序）')
    for d in details:
        print('- **%s** round=`%s`' % (d['case'], d['round']))
        for i, t in enumerate(d['tasks']):
            raw = d['task_results'][i] if i < len(d['task_results']) else '(no entry)'
            print('  %d. %s → `%s`' % (i + 1, t[:26], str(raw)[:180].replace('|', '/')))
    if '--json' in sys.argv:
        print()
        print(json.dumps({'cases': cases, 'details': details}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
