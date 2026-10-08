#!/usr/bin/env python3
"""DS 检视：逐任务打印某案的完整证据链（分类回答/校验/拒单/身份歧义）。
用法：/usr/bin/python3 /root/ds_inspect.py <batch> <case> [task_suffix]
例：  /usr/bin/python3 /root/ds_inspect.py b03_r7cand_190538 scramble_04 task-03
"""
import json
import sys

RUNS = '/root/gpufree-data/tcei_260920v2'


def main():
    batch, case = sys.argv[1], sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else None
    rd = '%s/%s_%s_round' % (RUNS, batch, case)
    ev = rd + '/episode/events.jsonl'

    key_status = ('task_started', 'task_succeeded', 'task_failed', 'rejected',
                  'task_rejected_continuing', 'task_pending_verification',
                  'placement_verified', 'placement_pending_verification',
                  'model_answer', 'validation_failed', 'repair_started',
                  'plan_published', 'plan_accepted', 'plan_rejected',
                  'candidate_class_conflict', 'stable_identity_class_changed',
                  'classification_completed', 'nine_scan_class_audit',
                  'grasp_candidate_set', 'judge_started',
                  'remaining_context_incomplete')

    evs = []
    for line in open(ev, errors='ignore'):
        line = line.strip()
        if not line:
            continue
        try:
            evs.append(json.loads(line))
        except Exception:
            pass

    print('total events:', len(evs))
    for e in evs:
        st = str(e.get('status') or '')
        tid = str(e.get('task_id') or '')
        hit_task = want and (want in tid or want.replace('-0', '-') in tid)
        if not (hit_task or st in key_status):
            continue
        if want and not hit_task and st not in ('remaining_context_incomplete',
                                                'candidate_class_conflict',
                                                'stable_identity_class_changed'):
            continue
        row = {k: e.get(k) for k in ('time', 'status', 'task_id', 'reason',
                                     'instruction', 'detail', 'message',
                                     'object_id', 'stable_id', 'class',
                                     'candidates_count', 'quantity')
               if e.get(k) is not None}
        # 保留候选摘要（若存在）
        cands = e.get('candidates')
        if isinstance(cands, list) and cands:
            row['candidates'] = [{'id': c.get('stable_id'), 'class': c.get('class'),
                                  'raw_class': c.get('raw_class'),
                                  'status': c.get('identity_status'),
                                  'class_source': c.get('class_source')} for c in cands]
        print(json.dumps(row, ensure_ascii=False)[:900])


if __name__ == '__main__':
    main()
