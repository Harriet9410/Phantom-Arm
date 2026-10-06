#!/usr/bin/env python3
"""R3-0/R3-1 evidence extraction for the b03_full_10051355 batch.

Per case produces:
  * R3-0 matrix: per task 1-5 status (initiated/rejected/pending/verified/not_initiated)
  * R3-0 snapshot: server git/manifests/disk/procs (printed once)
  * R3-1 forensics: last remaining_context_incomplete -> unobserved stable_ids,
    each traced against ALL placement_verified/grasp_verified/released events,
    classified into the four R3 categories
  * R3-2 detail: per rejected task -> instruction/validation answers; per
    task_pending_verification -> its release/placement evidence chain

Output: one JSON printed to stdout (piped to a file by the caller).
"""
import glob
import json
import pathlib
import subprocess

RUNS = pathlib.Path('/root/gpufree-data/tcei_260920v2')
BATCH = RUNS / 'b03_full_10051355'
PKG = pathlib.Path('/root/tcei_final_v2_23/tcei_260920v2')


def read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def driver_events(rd):
    out = []
    p = rd / 'episode' / 'events.jsonl'
    for line in p.read_text(errors='ignore').splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def task_status_of(result):
    """Per-task status list (1-5) from the driver summary + campaign audit."""
    summary = read_json(result['run'] + '/episode/summary.json') or {}
    rows = []
    for index, task in enumerate(summary.get('tasks') or []):
        res = task.get('result') or {}
        rows.append({'index': index + 1, 'instruction': task.get('instruction', ''),
                     'status': res.get('status') or 'not_initiated',
                     'reason': str(res.get('reason') or '')[:120]})
    return rows


def case_forensics(result):
    case = result['case_id']
    rd = BATCH.parent / (BATCH.name + '_' + case + '_round')
    events = driver_events(rd)
    out = {'case': case, 'episode_status': (result.get('supervisor') or {}).get('episode_status'),
           'verified': result.get('verified_objects'), 'elapsed': round(result.get('elapsed_seconds') or 0)}

    # --- per-task status ---
    out['tasks'] = task_status_of(result)

    # --- last remaining_context_incomplete ---
    last_rem = None
    placement_verified = {}   # stable_id -> [event dicts]
    grasp_verified = {}       # stable_id -> [times]
    released = {}             # stable_id -> [event dicts]
    task_status_events = []   # task-level statuses in order
    for e in events:
        st = str(e.get('status') or '')
        stable = e.get('stable_id') or (e.get('candidate') or {}).get('stable_id')
        if st == 'remaining_context_incomplete':
            last_rem = e
        if st == 'placement_verified' and stable:
            placement_verified.setdefault(stable, []).append(
                {'time': e.get('time'), 'release_id': e.get('release_id'),
                 'request_id': str(e.get('request_id') or '')[-12:], 'side': e.get('side')})
        if st == 'grasp_verified' and stable:
            grasp_verified.setdefault(stable, []).append(e.get('time'))
        if st == 'released' and stable:
            released.setdefault(stable, []).append(
                {'time': e.get('time'), 'release_id': e.get('release_id'), 'side': e.get('side')})
        if st in ('task_started', 'task_succeeded', 'task_failed', 'rejected',
                  'task_pending_verification', 'plan_rejected') and e.get('task_id'):
            task_status_events.append({'time': e.get('time'), 'task': str(e['task_id'])[-7:],
                                       'status': st, 'reason': str(e.get('reason') or '')[:100]})
    out['placement_verified_events'] = {k: v for k, v in placement_verified.items()}
    out['grasp_verified_stables'] = {k: len(v) for k, v in grasp_verified.items()}
    out['released_events'] = {k: v for k, v in released.items()}

    if last_rem is None:
        out['remaining'] = None
        return out
    ctx = last_rem.get('context') or {}
    regions = last_rem.get('unknown_regions') or []
    cands = last_rem.get('candidates') or []
    unobserved = [r for r in regions if r.get('reason') == 'unobserved_unverified_object']
    other_regions = [r for r in regions if r.get('reason') != 'unobserved_unverified_object']

    remaining_ids = ctx.get('remaining_stable_ids') or ctx.get('remaining_ids') or []
    ledger_delivered = []
    # ledger 的已交付集合：从各任务的 targets/placements 推（summary 里没有），
    # 这里用 placement_verified 事件近似，再与 context 字段互证。
    for r in unobserved:
        sid = r.get('stable_id')
        cls = r.get('class')
        has_pv = sid in placement_verified
        has_gv = sid in grasp_verified
        if has_pv:
            cat = '3_已核验交付但区域未退休'
            ev = placement_verified[sid]
        elif has_gv:
            cat = '2_抓取后未核验或掉件'
            ev = {'grasp_times': grasp_verified[sid]}
        else:
            cat = '1_真实未交付且不可见（待定：遮挡 or 伪影）'
            ev = {}
        r['_classification'] = cat
        r['_in_ledger_remaining'] = sid in remaining_ids
        r['_placement_events'] = ev if isinstance(ev, list) else [ev]
    out['remaining'] = {
        'time': last_rem.get('time'),
        'reasons': ctx.get('remaining_incomplete_reasons'),
        'remaining_stable_ids': remaining_ids,
        'unobserved': [{'stable_id': r.get('stable_id'), 'class': r.get('class'),
                        'bbox': r.get('bbox'), 'seen_count': r.get('seen_count'),
                        'last_observed_at': r.get('last_observed_at'),
                        'classification': r.get('_classification'),
                        'in_ledger_remaining': r.get('_in_ledger_remaining'),
                        'placement_events': r.get('_placement_events')} for r in unobserved],
        'other_regions': [{'reason': r.get('reason'), 'bbox': r.get('bbox'),
                           'stable_id': r.get('stable_id')} for r in other_regions],
        'visible_candidates': [{'id': c.get('id'), 'class': c.get('class'),
                                'stable_id': str(c.get('stable_id') or '')[-8:],
                                'identity': c.get('identity_status')} for c in cands],
    }
    out['task_status_events_tail'] = task_status_events[-8:]
    return out


def main():
    campaign = read_json(BATCH / 'campaign.json') or {}
    out = {'batch': BATCH.name, 'campaign_status': campaign.get('status'),
           'runtime': campaign.get('runtime_version'),
           'autoVerified': campaign.get('automatic_verified_rounds'),
           'cases': {}}
    matrix = []
    for result in campaign.get('results', []):
        case = result['case_id']
        f = case_forensics(result)
        out['cases'][case] = f
        row = {'case': case, 'verified': result.get('verified_objects')}
        for t in (f.get('tasks') or []):
            row['task%d' % t['index']] = t['status']
        matrix.append(row)
    # R3-0 矩阵
    print('=== R3-0 十案×五项矩阵 ===')
    for row in matrix:
        print(row)
    print(json.dumps({'summary': {'total_verified': sum((r.get('verified') or 0) for r in campaign.get('results', []))}}, ensure_ascii=False))
    pathlib.Path('/root/gpufree-data/tcei_260920v2/r3_extract.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written: r3_extract.json')


if __name__ == '__main__':
    main()
