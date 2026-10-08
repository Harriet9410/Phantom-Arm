#!/usr/bin/env python3
"""DS R7-1b: export candidate-batch evidence (READ-ONLY; no code change, no simulation).

用途：对 codex_v0_r7_candidate_20261008 的定向批次，逐案导出
  1. campaign 中该案的三段命令返回码、supervisor.episode_status、verified_objects；
  2. episode/summary.json 的五条任务 index/instruction/status/reason（含未发起）；
  3. episode/events.jsonl 的任务生命周期序列 + 类别变化事件 + remaining_context_incomplete 尾部；
  4. perception.log 的 verbatim 行：candidate class lineage / nine scan class audit /
     nine scan started / nine scan conflict images / frame= 摘要；
  5. 从 lineage 提取的 tracker 改判审计（protocol/stable_id/from/to/frame/两扫描 ID）；
  6. layout_prior_active 的逐帧取值（lineage 直接记录，替代此前的几何推断）。
输出目录：/root/gpufree-data/r7cand_export/（含 _SHA256.txt）
用法：/usr/bin/python3 /root/ds_r7cand_export.py [batch_name]
"""
import glob
import hashlib
import json
import os
import re
import sys

RUNS = '/root/gpufree-data/tcei_260920v2'
OUT = '/root/gpufree-data/r7cand_export'
DEFAULT_BATCH = 'b03_r7cand_190538'
LIFE = ('task_started', 'task_succeeded', 'task_failed', 'rejected',
        'task_rejected_continuing', 'task_pending_verification',
        'placement_verified', 'placement_pending_verification',
        'stable_identity_class_changed', 'identity_class_changed',
        'candidate_class_conflict')
TSPAT = re.compile(r'\[(\d+\.\d+)\]')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def lines_of(p):
    try:
        return open(p, errors='ignore').read().splitlines()
    except Exception:
        return []


def read_json(p):
    try:
        return json.loads(open(p, encoding='utf-8').read())
    except Exception:
        return None


def dump(name, text):
    p = os.path.join(OUT, name)
    open(p, 'w', encoding='utf-8').write(text)
    return p


def slim(e, n=1200):
    out = {}
    for k, v in e.items():
        if isinstance(v, str) and len(v) > n:
            out[k] = v[:n] + '...<%d>' % (len(v) - n)
        else:
            out[k] = v
    return out


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BATCH
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    camp = read_json(os.path.join(RUNS, batch, 'campaign.json')) or {}
    bundle = {
        'batch': batch,
        'campaign': {k: v for k, v in camp.items() if k != 'results'},
        'cases': {},
    }
    rows = camp.get('results') or []

    for row in rows:
        case = row.get('case_id')
        st = os.path.join(RUNS, batch + '_' + case + '_stack')
        rd = os.path.join(RUNS, batch + '_' + case + '_round')
        per = os.path.join(st, 'logs', 'perception.log')
        ev = os.path.join(rd, 'episode', 'events.jsonl')

        plines = lines_of(per)
        elines = lines_of(ev)

        # 1. campaign row (without huge command text)
        cmds = [{k: c.get(k) for k in ('phase', 'returncode', 'finished_at')}
                for c in row.get('commands') or []]
        sup = row.get('supervisor') or {}

        # 2. task results
        summary = read_json(os.path.join(rd, 'episode', 'summary.json')) or {}
        tasks = []
        for i, t in enumerate(summary.get('tasks') or []):
            r = t.get('result') or {}
            tasks.append({'index': i + 1, 'instruction': t.get('instruction', ''),
                          'status': r.get('status') or 'not_initiated',
                          'reason': str(r.get('reason') or ''),
                          'verified': r.get('verified')})

        # 3. events: lifecycle chain + class changes + remaining tail
        evs = []
        for l in elines:
            if not l.strip():
                continue
            try:
                evs.append(json.loads(l))
            except Exception:
                pass
        chain = [slim(e) for e in evs if str(e.get('status') or '') in LIFE]
        rem = [slim(e, 800) for e in evs if e.get('status') == 'remaining_context_incomplete']

        # 4. perception.log verbatim lines
        lineage = [l for l in plines if 'candidate class lineage' in l]
        audits = [l for l in plines if 'nine scan class audit' in l]
        started = [l for l in plines if 'nine scan started' in l]
        conflicts = [l for l in plines if 'nine scan conflict images' in l]
        frames = [l for l in plines if 'frame=' in l]
        unresolved = [l for l in plines if 'unresolved' in l]
        manifest.append(dump('lineage_lines_%s.txt' % case, '\n'.join(lineage) + '\n'))
        manifest.append(dump('audit_lines_%s.txt' % case, '\n'.join(audits) + '\n'))
        manifest.append(dump('scan_started_lines_%s.txt' % case, '\n'.join(started) + '\n'))
        manifest.append(dump('conflict_lines_%s.txt' % case, '\n'.join(conflicts) + '\n'))
        manifest.append(dump('frame_lines_%s.txt' % case, '\n'.join(frames) + '\n'))

        # 5. tracker reclass audits + 6. layout_prior_active, parsed from lineage
        reclass = []
        prior_vals = {}
        for l in lineage:
            ts = TSPAT.search(l)
            ts = float(ts.group(1)) if ts else None
            body = l.split('candidate class lineage:', 1)[-1].strip()
            try:
                rec = json.loads(body)
            except Exception:
                m = re.search(r'\{.*\}', body)
                rec = json.loads(m.group(0)) if m else {}
            if not isinstance(rec, dict):
                rec = {}
            ra = rec.get('reclass_audit')
            if ra:
                try:
                    ra = json.loads(ra) if isinstance(ra, str) else ra
                except Exception:
                    pass
                reclass.append({'t': ts, 'stable_id': rec.get('stable_id'),
                                'frame_id': rec.get('frame_id'),
                                'published_class': rec.get('published_class'),
                                'pretracker_class': rec.get('pretracker_class'),
                                'class_source': rec.get('class_source'),
                                'reclass_audit': ra})
            pa = rec.get('layout_prior_active')
            if pa is not None:
                prior_vals[str(pa)] = prior_vals.get(str(pa), 0) + 1

        bundle['cases'][case] = {
            'campaign_row': {'case_id': case, 'seed': row.get('seed'),
                             'case_sha256': row.get('case_sha256'),
                             'commands': cmds,
                             'episode_status': sup.get('episode_status'),
                             'verified_objects': row.get('verified_objects'),
                             'instructions': sup.get('instructions')},
            'tasks': tasks,
            'n_tasks': len(tasks),
            'n_succeeded': sum(1 for t in tasks if t['status'] in ('succeeded', 'task_succeeded')),
            'n_rejected': sum(1 for t in tasks if t['status'] == 'rejected'),
            'n_not_initiated': sum(1 for t in tasks if t['status'] == 'not_initiated'),
            'lifecycle_chain': chain,
            'remaining_tail_last': rem[-1] if rem else None,
            'n_remaining_events': len(rem),
            'counts': {'lineage': len(lineage), 'audit': len(audits),
                       'scan_started': len(started), 'conflict': len(conflicts),
                       'frames': len(frames), 'unresolved_hits': len(unresolved)},
            'layout_prior_active_values': prior_vals,
            'reclass_audits': reclass,
            'unresolved_lines': [l[-400:] for l in unresolved[:20]],
        }

    bf = os.path.join(OUT, 'bundle_%s.json' % batch)
    json.dump(bundle, open(bf, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    manifest.append(bf)
    with open(os.path.join(OUT, '_SHA256.txt'), 'w') as f:
        for p in manifest:
            if os.path.exists(p):
                f.write('%s  %s\n' % (sha(p), os.path.basename(p)))
    for p in manifest:
        if os.path.exists(p):
            print(os.path.basename(p), os.path.getsize(p))
    # 控制台摘要
    for case, c in bundle['cases'].items():
        print('%s: tasks=%d succeeded=%d status=%s verified=%s prior=%s reclass=%d' % (
            case, c['n_tasks'], c['n_succeeded'], c['campaign_row']['episode_status'],
            c['campaign_row']['verified_objects'], c['layout_prior_active_values'],
            len(c['reclass_audits'])))


if __name__ == '__main__':
    main()
