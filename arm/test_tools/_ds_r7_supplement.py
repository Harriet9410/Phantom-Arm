#!/usr/bin/env python3
"""R7-1a read-only evidence supplement (NO code change, NO simulation).

Per Codex R7 acceptance doc §"R7 下一步":
  1. s04 window 1791449888-1791449900 and s02 window 1791449404-1791449430:
     ALL candidate/class-related lines (frame= summaries, nine scan lines,
     process exit) from _stack/logs/perception.log, plus the same windows from
     _round/episode/events.jsonl.
  2. Mark t5 wait start / deadline-exit / nine scan start+complete; compare
     scan completion vs the decision window.
  3. layout_prior enablement for the B03 batch (code + logs).
Outputs to /root/gpufree-data/r7_supplement/ + _SHA256.txt
"""
import glob
import hashlib
import json
import os
import re

RUNS = '/root/gpufree-data/tcei_260920v2'
BATCH = 'b03_r6tgt_10081642'
OUT = '/root/gpufree-data/r7_supplement'
WINDOWS = {'scramble_02': (1791449404.0, 1791449430.0),
           'scramble_04': (1791449888.0, 1791449900.0)}
TSPAT = re.compile(r'\[(\d+\.\d+)\]')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def ts_of(line):
    m = TSPAT.search(line)
    return float(m.group(1)) if m else None


def dump(name, text):
    p = os.path.join(OUT, name)
    open(p, 'w').write(text)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    summary = {'batch': BATCH, 'cases': {}}

    for case, (t0, t1) in WINDOWS.items():
        st = os.path.join(RUNS, BATCH + '_' + case + '_stack')
        rd = os.path.join(RUNS, BATCH + '_' + case + '_round')
        per = os.path.join(st, 'logs', 'perception.log')
        ev = os.path.join(rd, 'episode', 'events.jsonl')

        # --- 1a. perception.log: window lines (ALL lines in window, verbatim) ---
        plines = open(per, errors='ignore').read().splitlines()
        inwin = [l for l in plines if (ts_of(l) is not None and t0 <= ts_of(l) <= t1)]
        manifest.append(dump('window_perception_%s.txt' % case, '\n'.join(inwin) + '\n'))

        # --- 1b. episode events in window (verbatim json lines) ---
        elines = [l for l in open(ev, errors='ignore').read().splitlines() if l.strip()]
        ewin = []
        for l in elines:
            try:
                e = json.loads(l)
            except Exception:
                continue
            t = e.get('time')
            if isinstance(t, (int, float)) and t0 <= t <= t1:
                ewin.append(l)
        manifest.append(dump('window_events_%s.jsonl' % case, '\n'.join(ewin) + '\n'))

        # --- 2. timing markers ---
        # t5 wait window: from last task terminal to episode end
        task_ev = []
        rem_ev = []
        for l in elines:
            try:
                e = json.loads(l)
            except Exception:
                continue
            stt = str(e.get('status') or '')
            if stt in ('task_started', 'task_succeeded', 'task_failed', 'rejected',
                       'task_rejected_continuing', 'task_pending_verification',
                       'placement_verified', 'placement_pending_verification'):
                task_ev.append({'t': e.get('time'), 'status': stt,
                                'task_id': str(e.get('task_id') or '')[-8:]})
            if stt == 'remaining_context_incomplete':
                rem_ev.append(e.get('time'))
        scans = [{'t': ts_of(l), 'kind': 'audit' if 'nine scan class audit' in l else 'other',
                  'sid': (json.loads(l.split('nine scan class audit:', 1)[1].strip())[0].get('scan_id')
                          if 'nine scan class audit' in l else None),
                  'raw': l[:160]}
                 for l in plines if 'nine scan' in l]
        # nine scan START time: not present in log (only completion) -> gap
        summary['cases'][case] = {
            'perception_log': per, 'episode_events': ev,
            'log_ts_range': [ts_of(plines[0]), ts_of(plines[-1])] if plines else None,
            'n_perception_lines': len(plines), 'n_window_lines': len(inwin),
            'n_events_window': len(ewin),
            'task_markers_tail': task_ev[-14:],
            'remaining_context_times': rem_ev,
            'first_remaining_time': (rem_ev[0] if rem_ev else None),
            'last_remaining_time': (rem_ev[-1] if rem_ev else None),
            'nine_scan_lines': scans,
            'nine_scan_start_times_present': False,
            'gap_note': 'perception.log 只记录 nine scan 完成行（含 audit），未见扫描开始时间戳；'
                        '故只能比较“扫描完成 vs 决策窗口”，无法比较扫描开始。'
            }
        manifest.append(per)

    # --- 3. layout_prior enablement ---
    pkg = '/root/tcei_final_v2_23/tcei_260920v2'
    pri = os.path.join(pkg, 'tcei_stack', 'scene_layout_prior.json')
    pri_info = {'path': pri, 'exists': os.path.exists(pri), 'sha256': sha(pri) if os.path.exists(pri) else None}
    try:
        d = json.load(open(pri))
        pri_info['scene'] = d.get('scene')
        pri_info['tolerance_px'] = d.get('tolerance_px')
        pri_info['gate_min_matches'] = d.get('gate_min_matches')
        pri_info['objects'] = d.get('objects')
    except Exception as e:
        pri_info['error'] = str(e)
    # registration told to the stack for this batch
    reg_pri = {}
    for f in glob.glob(os.path.join(RUNS, BATCH + '*_stack', 'runtime_manifest.json')):
        try:
            rm = json.load(open(f))
            reg_pri[os.path.basename(os.path.dirname(f))] = {
                k: rm.get(k) for k in ('layout_prior', 'classify_fastpath', 'detector',
                                       'yolo_enabled', 'case', 'case_register') if k in rm}
        except Exception as e:
            reg_pri[os.path.basename(os.path.dirname(f))] = str(e)
    summary['layout_prior'] = {'file': pri_info, 'runtime_manifest': reg_pri}
    # first-scan activation evidence: count of candidates in first frame line vs prior object count
    act = {}
    for case in WINDOWS:
        per = os.path.join(RUNS, BATCH + '_' + case + '_stack', 'logs', 'perception.log')
        for l in open(per, errors='ignore'):
            if 'frame=' in l:
                act[case] = l.split(']', 1)[-1].strip()[:220]
                break
    summary['first_frame_line'] = act
    manifest.append(dump('layout_prior_status.json', json.dumps(summary['layout_prior'], ensure_ascii=False, indent=1)))

    sf = os.path.join(OUT, 'supplement_summary.json')
    json.dump(summary, open(sf, 'w'), ensure_ascii=False, indent=1)
    manifest.append(sf)

    # --- 3b. server file SHAs (for Codex table cross-check) ---
    rows = []
    for rel in ('tcei_stack/nine_classify.py', 'tcei_stack/perception.py',
                'tcei_stack/perception_tracking.py', 'tcei_stack/episode_driver.py',
                'tcei_stack/BUILD_MANIFEST.json', 'PACKAGE_MANIFEST.json'):
        fp = os.path.join(pkg, rel)
        rows.append('%-40s %s' % (rel, sha(fp) if os.path.exists(fp) else 'MISSING'))
    manifest.append(dump('server_file_sha256.txt', '\n'.join(rows) + '\n'))

    with open(os.path.join(OUT, '_SHA256.txt'), 'w') as f:
        for p in manifest:
            if os.path.exists(p):
                f.write('%s  %s\n' % (sha(p), os.path.basename(p)))
    for p in manifest:
        if os.path.exists(p):
            print(os.path.basename(p), os.path.getsize(p))


if __name__ == '__main__':
    main()
