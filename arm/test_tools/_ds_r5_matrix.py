#!/usr/bin/env python3
"""R5 P0 forensics: per-case fault matrix extraction (t1_52 gate batch).

Targets the four chains from Codex R5 task book:
  (1) readiness wait   : s04/s05/s07 t1 -- was the paired frame's candidate class
                         `unknown` when selection ran? (perf: wait/timeout seconds)
  (2) misclassify after classified : s07 t2
  (3) missing target class : s08 t4
  (4) t5 class drift : s01/s04/s06/s07/s08/s10 -- FIRST occurrence of
      stable_identity_class_changed per stable_id, with nine-scan audit.

Per case exports:
  * task results (index/instruction/status/reason)
  * for every task: model answers (first + repairs), validation reasons,
    paired frame_id, candidate ids/classes/identity_status at that frame
  * nine scan audit events (nine scan class audit / classify_answered tags)
  * tracker alt-class vote evidence + ledger first class_changed per stable_id
  * t5 not-initiated: full remaining_context_incomplete tail + first-abnormal time
Output: one JSON on stdout.
"""
import glob
import json
import os
import sys

RUNS = '/root/gpufree-data/tcei_260920v2'
PKG = '/root/tcei_final_v2_23/tcei_260920v2'
BLOB = {'code_sha256', 'image', 'preview', 'depth_references', 'hidden_references',
        'depth_layer', 'inference_configuration', 'quaternion', 'target',
        'grasp_candidate', 'pose_candidate', 'joint_stamp', 'cached_depth_stamps',
        'depth_stamp', 'calibration_id'}
SEM = ('model_answer', 'validation_failed', 'repair_started', 'plan_published',
       'plan_accepted', 'plan_rejected', 'task_started', 'rejected',
       'task_rejected_continuing', 'task_succeeded', 'task_pending_verification',
       'task_failed', 'placement_verified', 'placement_pending_verification',
       'grasp_verified', 'released', 'judge_started', 'grasp_candidate_set',
       'classification_completed', 'nine_scan_class_audit', 'classify_answered',
       'nine_scan', 'observation_accepted', 'observation_received')


def slim(e, n=1500):
    out = {}
    for k, v in e.items():
        if k in BLOB:
            continue
        if isinstance(v, str) and len(v) > n:
            out[k] = v[:n] + '...<%d>' % (len(v) - n)
        else:
            out[k] = v
    return out


def read_json(p):
    try:
        return json.loads(open(p, encoding='utf-8').read())
    except Exception:
        return None


def find_batch(prefix):
    hits = [x for x in sorted(glob.glob(RUNS + '/' + prefix + '*'))
            if os.path.isdir(x) and not x.endswith('_stack') and not x.endswith('_round')]
    return hits[-1] if hits else None


def main():
    prefix = 'b03_r5gate_'
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
    batch = find_batch(prefix)
    if not batch:
        print(json.dumps({'error': 'no %s* batch found' % prefix}))
        return
    camp = read_json(batch + '/campaign.json') or {}
    out = {'batch': os.path.basename(batch), 'campaign_status': camp.get('status'),
           'runtime': camp.get('runtime_version'), 'cases': {}}
    for row in camp.get('results') or []:
        case = row.get('case_id')
        rd = batch + '_' + case + '_round'
        summary = read_json(rd + '/episode/summary.json') or {}
        tasks = []
        for i, t in enumerate(summary.get('tasks') or []):
            r = t.get('result') or {}
            tasks.append({'index': i + 1, 'instruction': t.get('instruction', ''),
                          'status': r.get('status') or 'not_initiated',
                          'reason': str(r.get('reason') or '')})
        evs = []
        p = rd + '/episode/events.jsonl'
        if os.path.exists(p):
            for line in open(p, errors='ignore'):
                if line.strip():
                    try:
                        evs.append(json.loads(line))
                    except Exception:
                        pass
        # 语义/审计事件（按顺序）
        chain = [slim(e) for e in evs if e.get('status') in SEM]
        # 每个 t5 未发起案的类改判首次时刻
        drift = {}
        for e in evs:
            st = str(e.get('status') or '')
            if st in ('stable_identity_class_changed', 'identity_class_changed',
                      'candidate_class_conflict') or 'class_changed' in st:
                drift.setdefault(str(e.get('time')), []).append(slim(e, 600))
        # 九格扫描审计（t1_52 新增，走 rospy.loginfo -> logs/perception.log）。
        # 实测（R5 b03_r5gate）：审计写在 <case>_stack/logs/perception.log，
        # 不在 <case>_round/logs/ 下——两处都扫以兼容。
        audits = []
        for logs_dir in (rd + '/logs', rd.replace('_round', '_stack') + '/logs'):
            if not os.path.isdir(logs_dir):
                continue
            for lf in sorted(glob.glob(logs_dir + '/*.log')):
                for line in open(lf, errors='ignore'):
                    if 'nine scan class audit' in line:
                        try:
                            payload = line.split('nine scan class audit:', 1)[1].strip()
                            audits.append({'log': os.path.basename(lf), 'audit': json.loads(payload)})
                        except Exception:
                            audits.append({'log': os.path.basename(lf), 'raw': line[-800:]})
        # 就绪等待证据：classify_answered（带 tag）与 readiness 相关事件
        readiness = [slim(e, 500) for e in evs if 'readiness' in str(e.get('status') or '').lower()
                     or e.get('status') in ('classify_answered', 'observation_accepted',
                                            'observation_received', 'observation_prepared')]
        # t5 末状态
        rem = [slim(e, 2000) for e in evs if e.get('status') == 'remaining_context_incomplete']
        t5ev = [slim(e, 800) for e in evs if str(e.get('task_id') or '').endswith(('-task-05', '-task-5'))]
        out['cases'][case] = {
            'episode_status': (row.get('supervisor') or {}).get('episode_status'),
            'verified_objects': row.get('verified_objects'),
            'tasks': tasks,
            'semantic_chain': chain,
            'readiness_events': readiness,
            'class_changed_events': drift,
            'nine_scan_audits': audits,
            't5_last_remaining': rem[-1] if rem else None,
            't5_events': t5ev,
        }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
