#!/usr/bin/env python3
"""R8-A：用修正后的验收口径离线重算 R7 两批（不重跑仿真、不改动原始证据）。

服务器执行： /usr/bin/python3 /root/ds_r8a_recompute.py
- 调用新版 run_registered_batch.audit()，结果写 <run>/independent_audit_r8a.json
  （与原 independent_audit.json 并存，原始证据不改）。
- 汇总写 /root/gpufree-data/r8a_recompute/baseline_r7_batches.json
"""
import json
import sys
from pathlib import Path

PKG = Path('/root/tcei_final_v2_23/tcei_260920v2')
RUNS = Path('/root/gpufree-data/tcei_260920v2')
OUT = Path('/root/gpufree-data/r8a_recompute')
sys.path.insert(0, str(PKG / 'test_tools'))
import run_registered_batch as B  # noqa: E402

BATCHES = {'b03_r7cand_190538': ['scramble_02', 'scramble_04'],
           'b03_r7cand2_192941': ['scramble_03', 'scramble_01']}
KEYS = ('task_index', 'object_id', 'truth_class', 'published_class', 'class_label_correct',
        'object_matches_instruction', 'side_matches_instruction', 'physically_on_belt',
        'instruction_completed', 'expected_object_ids', 'expected_side')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for batch, cases in BATCHES.items():
        camp = json.loads((RUNS / batch / 'campaign.json').read_text(encoding='utf-8'))
        out[batch] = {'runtime_version': camp.get('runtime_version'),
                      'register_sha256': camp.get('register_sha256'), 'cases': {}}
        for case_id in cases:
            case = json.loads((PKG / 'cases_scramble10' / (case_id + '.json')).read_text(encoding='utf-8'))
            ins_path = RUNS / batch / (case_id + '_instructions.json')
            ins = json.loads(ins_path.read_text(encoding='utf-8')) if ins_path.exists() else case['instructions']
            run = RUNS / (batch + '_' + case_id + '_round')
            stack = RUNS / (batch + '_' + case_id + '_stack')
            if not (run / 'episode/summary.json').exists():
                out[batch]['cases'][case_id] = {'error': 'missing run summary'}
                continue
            if not (stack / 'ground_truth.jsonl').exists():
                out[batch]['cases'][case_id] = {'error': 'missing ground truth'}
                continue
            try:
                res = B.audit(run, stack, case, ins, audit_name='independent_audit_r8a.json')
            except Exception as e:
                out[batch]['cases'][case_id] = {'error': '%s: %s' % (type(e).__name__, e)}
                continue
            out[batch]['cases'][case_id] = {
                'status': res['status'], 'task_results': res['task_results'],
                'verified_objects': res['verified_objects'],
                'independent_placements': res['independent_placements'],
                'instruction_tasks_completed': res['instruction_tasks_completed'],
                'classification_labels': res['classification_labels'],
                'rest_task': res['rest_task'], 'physical_completed': res['physical_completed'],
                'expected_metadata_source': res['expected_metadata_source'],
                'identity_conflicts': res['identity_conflicts'],
                'release_count': len(res['release_checks']),
                'releases': [{k: r.get(k) for k in KEYS} for r in res['release_checks']],
            }
    (OUT / 'baseline_r7_batches.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for batch, b in out.items():
        for cid, v in b['cases'].items():
            if 'error' in v:
                print('%s %s ERROR %s' % (batch, cid, v['error']))
                continue
            print('%s %s status=%s machine=%s physical=%s instr=%s labels=%s' % (
                batch, cid, v['status'], v['verified_objects'], v['independent_placements'],
                v['instruction_tasks_completed'], v['classification_labels']))
            if v['rest_task']:
                r = v['rest_task']
                print('     rest: plan=%s basket@start=%s ref=%s delivered=%s missing=%s extra=%s complete=%s planned_match=%s' % (
                    r['expected_set'], r.get('basket_set_at_start'), r.get('reference_set'),
                    r['delivered_set'], r['missing'], r['extra'], r['set_complete'],
                    r.get('matches_planned_set')))
    print('written:', OUT / 'baseline_r7_batches.json')


if __name__ == '__main__':
    main()
