#!/usr/bin/env python3
"""R4-1 forensics: per-failure full evidence extraction.

Targets: the 8 rejections + 1 pending verification in t1_48 batches
(b03_r3v3 s01/s02, b02_r3b r09/r21/r24).

Per failure exports:
  * instruction, terminal status + driver reason, request_id
  * ALL events of that task (blob fields stripped, candidates/task_context kept)
  * the three model answers + validation feedback verbatim
  * ledger context (remaining/reserved/released_pending) at task start
  * case ground truth (cases_*.json objects)
  * for the pending task: release->verify window with conveyor observer states
Output: one JSON on stdout.
"""
import glob
import json
import os

RUNS = '/root/gpufree-data/tcei_260920v2'
PKG = '/root/tcei_final_v2_23/tcei_260920v2'
BLOB = {'code_sha256', 'image', 'preview', 'depth_references', 'hidden_references',
        'depth_layer', 'inference_configuration', 'quaternion', 'target',
        'grasp_candidate', 'pose_candidate', 'joint_stamp', 'cached_depth_stamps',
        'depth_stamp', 'calibration_id'}
KEEP_BIG = {'candidates', 'task_context', 'answer', 'feedback_guidance_zh',
            'model_response', 'reason', 'belt_observer', 'feedback'}

TARGETS = [
    ('b03_r3v3_10062357', 'scramble_01', 3),
    ('b03_r3v3_10062357', 'scramble_02', 2),
    ('b03_r3v3_10062357', 'scramble_02', 3),
    ('b03_r3v3_10062357', 'scramble_02', 5),
    ('b02_r3b_10070029', 'official_random_09', 1),
    ('b02_r3b_10070029', 'official_random_09', 2),
    ('b02_r3b_10070029', 'official_random_21', 1),
    ('b02_r3b_10070029', 'official_random_21', 2),
    ('b02_r3b_10070029', 'official_random_24', 1),
]


def slim(e):
    out = {}
    for k, v in e.items():
        if k in BLOB:
            continue
        if isinstance(v, str) and len(v) > 400 and k not in KEEP_BIG:
            out[k] = v[:400] + '...'
        elif isinstance(v, str) and len(v) > 1200 and k in KEEP_BIG:
            out[k] = v[:1200] + '...'
        else:
            out[k] = v
    return out


def case_truth(batch, case):
    if case.startswith('scramble'):
        p = PKG + '/cases_scramble10/%s.json' % case
    else:
        p = PKG + '/cases_official30_v2/%s.json' % case
    try:
        cf = json.load(open(p, encoding='utf-8'))
    except Exception:
        return None
    objs = cf.get('objects') or cf.get('items') or []
    return [{'name': o.get('name'), 'class': o.get('class') or o.get('category'),
             'pos': o.get('position') or o.get('pos')} for o in objs]


def main():
    out = {'failures': []}
    for batch, case, tidx in TARGETS:
        rd = RUNS + '/%s_%s_round' % (batch, case)
        p = rd + '/episode/events.jsonl'
        if not os.path.exists(p):
            out['failures'].append({'batch': batch, 'case': case, 'task': tidx, 'error': 'no events'})
            continue
        events = []
        for line in open(p, errors='ignore'):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        tevents = [e for e in events
                   if str(e.get('task_id') or '').endswith('-task-%02d' % tidx)
                   or str(e.get('task_id') or '').endswith('-task-%d' % tidx)]
        summary = {}
        try:
            s = json.load(open(rd + '/episode/summary.json'))
            t = s.get('tasks', [])[tidx - 1]
            r = t.get('result') or {}
            summary = {'instruction': t.get('instruction'),
                       'status': r.get('status'), 'reason': str(r.get('reason') or '')}
        except Exception as e:
            summary = {'error': str(e)}
        chain = [slim(e) for e in tevents]
        request_ids = sorted({e.get('request_id') for e in tevents if e.get('request_id')})
        # 模型应答/校验事件不带 task_id，按 request_id 关联补入
        rid_set = set(request_ids)
        related = []
        for e in events:
            if e.get('request_id') in rid_set and e not in tevents:
                if e.get('status') in ('model_answer', 'validation_failed', 'repair_started',
                                       'judge_started', 'plan_published', 'plan_accepted',
                                       'plan_rejected', 'grasp_candidate_set'):
                    related.append(slim(e))
        related.sort(key=lambda e: e.get('time') or 0)
        # 释放/核验窗口（仅 pending 任务用，但统一导出）
        window = []
        rel = next((e for e in tevents if e.get('status') == 'released'), None)
        pen = next((e for e in tevents if e.get('status') == 'placement_pending_verification'), None)
        if rel and pen:
            t0, t1 = rel['time'], pen['time']
            for e in events:
                if t0 - 1 <= e.get('time', 0) <= t1 + 1:
                    window.append(slim(e))
        out['failures'].append({
            'batch': batch, 'case': case, 'task_index': tidx,
            'summary': summary,
            'request_ids': [str(x)[-16:] for x in request_ids],
            'n_task_events': len(chain),
            'chain': chain,
            'model_related_events': related,
            'release_window_events': window,
            'ground_truth': case_truth(batch, case),
        })
    print(json.dumps(out, ensure_ascii=False))



SEM_KEYS = ('model_answer', 'validation_failed', 'plan_published', 'plan_accepted',
            'plan_rejected', 'task_started', 'rejected', 'task_rejected_continuing',
            'task_succeeded', 'task_pending_verification', 'judge_started',
            'grasp_candidate_set', 'grasp_evidence_preflight', 'repair_started')


def semantic_chain(events):
    return [slim(e) for e in events if e.get('status') in SEM_KEYS]


def main2():
    out = {'semantic_chains': {}}
    for batch, case in [('b03_r3v3_10062357', 'scramble_01'),
                        ('b03_r3v3_10062357', 'scramble_02'),
                        ('b02_r3b_10070029', 'official_random_09'),
                        ('b02_r3b_10070029', 'official_random_21'),
                        ('b02_r3b_10070029', 'official_random_24')]:
        p = RUNS + '/%s_%s_round/episode/events.jsonl' % (batch, case)
        events = []
        for line in open(p, errors='ignore'):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        chain = semantic_chain(events)
        # 案例真值
        truth = case_truth(batch, case)
        out['semantic_chains'][batch + '_' + case] = {
            'n_events': len(events), 'truth': truth, 'chain': chain}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main2__':
    pass



def main3():
    out = {'plans': {}}
    for batch, case in [('b03_r3v3_10062357', 'scramble_01'),
                        ('b03_r3v3_10062357', 'scramble_02'),
                        ('b02_r3b_10070029', 'official_random_09'),
                        ('b02_r3b_10070029', 'official_random_21'),
                        ('b02_r3b_10070029', 'official_random_24')]:
        p = RUNS + '/%s_%s_round/episode/events.jsonl' % (batch, case)
        events = []
        for line in open(p, errors='ignore'):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        plans = []
        for e in events:
            if e.get('status') != 'plan_published':
                continue
            plan = e.get('plan') or {}
            tc = plan.get('task_context') or {}
            objs = []
            for o in plan.get('objects') or []:
                objs.append({k: o.get(k) for k in (
                    'class', 'raw_class', 'confidence', 'bbox', 'pixel', 'depth',
                    'world_position', 'support_z', 'axis_ratio', 'grasp_point_method',
                    'stable_id', 'id', 'identity_status', 'position_uncertainty',
                    'depth_spread', 'recognition_rotation_deg', 'angle_deg')})
            plans.append({'time': e.get('time'), 'instruction': plan.get('instruction'),
                          'side': plan.get('side'), 'class': plan.get('class'),
                          'task_index': tc.get('task_index'),
                          'candidate_stable_ids': tc.get('candidate_stable_ids'),
                          'remaining_ids': tc.get('remaining_ids'),
                          'reserved_ids': tc.get('reserved_ids'),
                          'released_pending': tc.get('released_pending_stable_ids'),
                          'objects': objs})
        out['plans'][batch + '_' + case] = plans
    print(json.dumps(out, ensure_ascii=False))




def main4():
    out = {'repair_prompts': {}}
    for batch, case in [('b03_r3v3_10062357', 'scramble_01'),
                        ('b03_r3v3_10062357', 'scramble_02'),
                        ('b02_r3b_10070029', 'official_random_09'),
                        ('b02_r3b_10070029', 'official_random_21'),
                        ('b02_r3b_10070029', 'official_random_24')]:
        p = RUNS + '/%s_%s_round/episode/events.jsonl' % (batch, case)
        items = []
        for line in open(p, errors='ignore'):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get('status') in ('repair_started', 'model_answer', 'validation_failed'):
                items.append({'status': e.get('status'), 'time': e.get('time'),
                              'request_id': str(e.get('request_id') or '')[-16:],
                              'attempt': e.get('attempt'),
                              'answer': e.get('answer'),
                              'previous_answer': e.get('previous_answer'),
                              'reason': e.get('reason'),
                              'feedback_guidance_zh': e.get('feedback_guidance_zh'),
                              'prompt': e.get('prompt')})
        out['repair_prompts'][batch + '_' + case] = items
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main4()


