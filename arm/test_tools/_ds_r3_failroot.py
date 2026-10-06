#!/usr/bin/env python3
"""R3-2 forensics v2: key evidence events for every failed task in
b03_full_10051355 (7 rejected + 5 task_pending_verification).

Per case exports:
  * failed-task list (from episode summary: index/instruction/status/reason)
  * ALL key events (full payload, truncated): validation / answers / plans /
    grasp / release / placement chain — ordered, time-stamped
  * ground-truth object list from cases_scramble10/<case>.json
Output: one JSON on stdout.
"""
import json
import pathlib

RUNS = pathlib.Path('/root/gpufree-data/tcei_260920v2')
BATCH = RUNS / 'b03_full_10051355'
PKG = pathlib.Path('/root/tcei_final_v2_23/tcei_260920v2')
KEY = ('model_answer', 'validation_failed', 'rejected', 'task_rejected_continuing',
       'plan_published', 'plan_accepted', 'plan_rejected', 'task_started',
       'task_succeeded', 'task_pending_verification', 'task_failed',
       'placement_pending_verification', 'placement_verified', 'placement_failed',
       'grasp_verified', 'grasp_attempt_outcome', 'release_started', 'released',
       'holding_feedback_lost', 'drop_detected', 'judge_started', 'judge_result')
LONG = {'validation_failed', 'rejected', 'task_pending_verification', 'model_answer',
        'judge_result', 'placement_failed'}


def read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def trunc(v, n):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + '...<%d more>' % (len(s) - n)


def main():
    campaign = read_json(BATCH / 'campaign.json') or {}
    out = {'batch': BATCH.name, 'cases': {}}
    for result in campaign.get('results') or []:
        case = result['case_id']
        rd = BATCH.parent / (BATCH.name + '_' + case + '_round')
        summary = read_json(rd / 'episode' / 'summary.json') or {}
        tasks_summary = []
        for index, task in enumerate(summary.get('tasks') or []):
            res = task.get('result') or {}
            tasks_summary.append({'index': index + 1,
                                  'instruction': task.get('instruction', ''),
                                  'status': res.get('status') or 'not_initiated',
                                  'reason': str(res.get('reason') or '')})
        failed = [t for t in tasks_summary if t['status'] in ('rejected', 'task_pending_verification')]
        cf = read_json(PKG / 'cases_scramble10' / (case + '.json'))
        gt = None
        if cf:
            objs = cf.get('objects') or cf.get('items') or cf.get('ground_truth') or []
            if isinstance(objs, list):
                gt = [{'class': o.get('class') or o.get('category'),
                       'pos': o.get('position') or o.get('pos'),
                       'name': o.get('name')} for o in objs]
            else:
                gt = trunc(cf, 600)
        evs = []
        p = rd / 'episode' / 'events.jsonl'
        for line in p.read_text(errors='ignore').splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            st = str(e.get('status') or '')
            if st in KEY:
                n = 700 if st in LONG else 300
                evs.append({'t': e.get('time'), 'st': st,
                            'task_index': e.get('task_index'),
                            'stable': str(e.get('stable_id') or (e.get('candidate') or {}).get('stable_id') or '')[-9:],
                            'd': trunc(e, n)})
        out['cases'][case] = {
            'episode_status': (result.get('supervisor') or {}).get('episode_status'),
            'verified_objects': result.get('verified_objects'),
            'failed_tasks': failed,
            'ground_truth': gt,
            'key_events': evs,
        }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
