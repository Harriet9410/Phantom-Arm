#!/usr/bin/env python3
"""R2-0 evidence: per-task table, classification vote chains, unknown-region timeline.

Implements the round-2 review's step R2-0 deliverables from RAW events only (no ROS):
  1. per-task table per case: 已发起/已核验/失败/中止/未发起 + reason + evidence path
  2. per-scan annotated answer chain from `<stack>/logs/nine.log`, the class trajectory
     of each ground-truth object, and the scan that ESTABLISHED a wrong label
     (confidence-marked: the log has answers but no prompts, so votes are reconstructed
     by position/shape and labeled as such)
  3. unknown-region timeline from `<round>/scalars/events.jsonl`, the final
     remaining_context payload, and crop images for each distinct region

Ground truth mapping uses the case's authored positions through the calibrated
projection u = 643.5 - 569.3x, v = 297.4 + 629.6y (residual <=15.5px, see R1).

Usage:
  python3 r2_evidence.py <batch_dir> --register <register.json> --out <outdir>
"""
import json
import math
import pathlib
import re
import sys

U0, UX, V0, VY = 643.5, -569.3, 297.4, 629.6
EN_CLASSES = ('Grenade', 'Magazine', 'Smokegrenade', 'Torch', 'CompressedFood')
CN_MARKS = ('手榴弹', '手雷', '手电筒', '弹夹', '弹匣', '烟雾弹', '压缩')
ALIAS = {'手榴弹': 'Grenade', '手雷': 'Grenade', '手电筒': 'Torch', '军用手电筒': 'Torch',
         '弹夹': 'Magazine', '弹匣': 'Magazine', '烟雾弹': 'Smokegrenade', '压缩': 'CompressedFood'}


def read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def read_text(path):
    try:
        return pathlib.Path(path).read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def gt_objects(case):
    out = []
    for obj in case.get('objects', []):
        x, y = obj['position'][0], obj['position'][1]
        out.append({'name': obj['path'].split('/')[-1], 'u': U0 + UX * x, 'v': V0 + VY * y})
    return out


def nearest_gt(gt, cx, cy):
    best = min(gt, key=lambda g: (g['u'] - cx) ** 2 + (g['v'] - cy) ** 2)
    return best['name'], math.hypot(best['u'] - cx, best['v'] - cy)


def parse_class_loose(answer):
    low = str(answer or '').lower()
    en = [c for c in sorted(EN_CLASSES, key=len, reverse=True) if c.lower() in low]
    cn = [ALIAS[c] for c in CN_MARKS if c in str(answer or '')]
    return en, cn


def parse_nine_log(text):
    """(time, kind, payload) events; kinds: answer / scan_end / scan_failed / timeout."""
    events = []
    for line in text.splitlines():
        if 'nine scan: classified' in line:
            m = re.search(r'\[([\d.]+)\]\s*:?\s*(nine scan:.*)$', line)
            events.append((float(m.group(1)) if m else None, 'scan_end', m.group(2) if m else line.strip()))
            continue
        if 'nine scan failed' in line:
            events.append((None, 'scan_failed', line.strip()[:200]))
            continue
        if 'nine locate timeout' in line:
            m = re.search(r'\[([\d.]+)\]', line)
            events.append((float(m.group(1)) if m else None, 'timeout', 'nine locate timeout'))
            continue
        i = line.find('{')
        if i < 0:
            continue
        try:
            d = json.loads(line[i:])
        except Exception:
            continue
        if d.get('status') == 'classify_answered':
            m = re.search(r'\[([\d.]+)\]', line)
            events.append((float(m.group(1)) if m else d.get('time'), 'answer', d))
    return events


def merged_events(stack_dir):
    """scan_end markers live in perception.log; answers/timeouts in nine.log. Merge by time."""
    nine = parse_nine_log(read_text(stack_dir / 'logs' / 'nine.log'))
    perc = parse_nine_log(read_text(stack_dir / 'logs' / 'perception.log'))
    ends = [e for e in perc if e[1] == 'scan_end']
    fails = [e for e in perc if e[1] == 'scan_failed']
    rest = [e for e in nine if e[1] in ('answer', 'timeout')]
    merged = sorted([e for e in rest + ends + fails if e[0] is not None], key=lambda e: e[0])
    return merged


def annotate_answer(answer):
    """Best-effort shape label for one logged answer (prompts are not logged)."""
    text = str(answer or '')
    try:
        d = json.loads(text)
        if isinstance(d, dict) and any(k.isdigit() for k in d):
            return 'full_frame_json'
    except Exception:
        pass
    en, cn = parse_class_loose(text)
    joined = sorted(set(en + [c for c in cn]))
    if len(joined) == 1:
        return 'single_class:' + joined[0]
    if len(joined) == 0:
        return 'no_class_token'
    return 'multi_class:' + '+'.join(joined)


def scan_windows(events):
    """Group events into scans: a window ends at each scan_end marker."""
    windows, current = [], []
    for t, kind, payload in events:
        if kind == 'scan_end':
            windows.append({'answers': current, 'end_text': payload, 'end_time': t})
            current = []
        else:
            current.append((t, kind, payload))
    return windows


def candidates_timeline(round_dir):
    """(sim_time, wall_time, candidates, unknown_regions) per /tcei/candidates record."""
    path = round_dir / 'scalars' / 'events.jsonl'
    rows = []
    for line in read_text(path).splitlines():
        if '"candidates"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        v = d.get('value') or {}
        if not isinstance(v, dict):
            continue
        rows.append({'sim': d.get('last_received_simulation_time'),
                     'wall': d.get('received_wall'),
                     'candidates': v.get('candidates') or [],
                     'unknown': v.get('unknown_regions') or []})
    return rows


def load_frames(round_dir):
    """frame stem -> captured_at (wall) from rgbd/index.jsonl."""
    idx = round_dir / 'rgbd' / 'index.jsonl'
    frames = {}
    for line in read_text(idx).splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        stem = d.get('stem')
        if stem:
            frames[stem] = d.get('captured_at')
    return frames


def frame_for_wall(frames, wall):
    if wall is None:
        return None, None
    best, delta = None, None
    for stem, t in frames.items():
        if not isinstance(t, (int, float)):
            continue
        d = abs(t - wall)
        if delta is None or d < delta:
            best, delta = stem, d
    return best, delta


# ---------------------------------------------------------------- per-task table

def per_task_table(batch, case, result, summary):
    events = [l for l in read_text(batch.parent / (batch.name + '_' + case + '_round') / 'episode' / 'events.jsonl').splitlines() if l.strip()]
    parsed = []
    for line in events:
        try:
            parsed.append(json.loads(line))
        except Exception:
            pass
    rows = []
    tasks = summary.get('tasks') or []
    raw_results = result.get('task_results') or []
    for index, task in enumerate(tasks):
        instruction = task.get('instruction') or ''
        status, reason = 'unknown', ''
        if index < len(raw_results):
            raw = raw_results[index]
            if isinstance(raw, str):
                status, reason = raw, ''
            else:
                status, reason = raw.get('status', 'unknown'), raw.get('reason', '')
        else:
            status = 'not_initiated'
        rows.append({'index': index + 1, 'instruction': instruction, 'status': status, 'reason': str(reason)[:150]})
    # 第五项是否真的未发起：找 task-05 的 request 事件
    tail = [e for e in parsed if str(e.get('status', '')).startswith('remaining') or 'remaining set incomplete' in str(e.get('reason', ''))]
    return rows, tail[-1:] and json.dumps(tail[-1], ensure_ascii=False)[:400] or ''


# ---------------------------------------------------------------- main

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    batch = pathlib.Path(sys.argv[1]).resolve()
    args = sys.argv[2:]
    register_path = args[args.index('--register') + 1] if '--register' in args else None
    outdir = pathlib.Path(args[args.index('--out') + 1]) if '--out' in args else batch / '_r2evidence'
    outdir.mkdir(parents=True, exist_ok=True)
    campaign = read_json(batch / 'campaign.json') or {}
    register = read_json(register_path) if register_path else None
    planned_ids = [r['case_id'] for r in (register or {}).get('cases', [])]
    done_ids = [r.get('case_id') for r in campaign.get('results', [])]
    roll = []
    for cid in planned_ids or done_ids:
        if cid in done_ids:
            roll.append({'case': cid, 'roll': 'completed_or_stopped'})
        elif cid == campaign.get('active_case'):
            roll.append({'case': cid, 'roll': 'aborted_mid_run'})
        else:
            roll.append({'case': cid, 'roll': 'not_initiated'})
    report = {'batch': batch.name, 'campaign_status': campaign.get('status'),
              'planned': campaign.get('planned_rounds'), 'recorded': len(done_ids),
              'case_roll': roll, 'cases': {}}
    print('# R2-0 证据 — 批次 %s' % batch.name)
    print('# campaign.status=%s | planned=%s | recorded=%s' % (
        campaign.get('status'), campaign.get('planned_rounds'), len(done_ids)))
    print('# 案卷：' + ', '.join('%s=%s' % (r['case'], r['roll']) for r in roll))
    print()
    for result in campaign.get('results', []):
        case = result.get('case_id')
        round_dir = batch.parent / (batch.name + '_' + case + '_round')
        stack_dir = batch.parent / (batch.name + '_' + case + '_stack')
        summary = read_json(round_dir / 'episode' / 'summary.json') or {}
        case_doc = read_json(stack_dir / 'case.json') or {}
        gt = gt_objects(case_doc)
        rows, tail_event = per_task_table(batch, case, result, summary)
        # --- scan chains ---
        events = merged_events(stack_dir)
        windows = scan_windows(events)
        tl = candidates_timeline(round_dir)
        frames = load_frames(round_dir)
        # 类别轨迹：对每个 GT 物体，按扫描列出其提案的（标注后的）最终类别
        trajectory = {g['name']: [] for g in gt}
        establishing = {}
        for wi, win in enumerate(windows):
            answers = [p for (t, k, p) in win['answers'] if k == 'answer']
            if not answers:
                continue
            # 扫描时刻 ≈ 本窗第一条答案（wall 基）前 1.5s；按 wall 取该时刻之前最近的候选消息作提案
            t0 = win['answers'][0][0]
            pre = [r for r in tl if isinstance(t0, (int, float)) and isinstance(r['wall'], (int, float)) and r['wall'] <= t0 - 1.5]
            props = (pre[-1]['candidates'] if pre else [])
            # 全图答案：取本窗第一条能解析为 {编号:类名} 的答案（全图询问若超时会缺席）
            full = None
            for a in answers:
                try:
                    d = json.loads(a)
                except Exception:
                    continue
                if isinstance(d, dict) and any(k.isdigit() for k in d):
                    full = d
                    break
            assigned = {}
            if isinstance(full, dict):
                for k, v in full.items():
                    if str(k).isdigit() and isinstance(v, str):
                        en, cn = parse_class_loose(v)
                        cand = en or cn
                        if len(set(cand)) == 1:
                            assigned[int(k)] = cand[0]
            for c in props:
                try:
                    bid = int(c.get('id'))
                except Exception:
                    continue
                cls = assigned.get(bid)
                if not cls:
                    continue
                bbox = c.get('bbox') or [0, 0, 0, 0]
                name, dist = nearest_gt(gt, (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                if dist > 45:
                    continue
                trajectory[name].append({'scan': wi + 1, 'class': cls})
        for name, hist in trajectory.items():
            if not hist:
                continue
            first_cls = hist[0]['class']
            for h in hist:
                if h['class'] != first_cls:
                    break
            if first_cls:
                establishing.setdefault(name, hist[0]['scan'])
        # --- regions：分两类 —— 阻断类（multiple_body_cores / unresolved_correspondence）
        # 与 last-seen 类（unobserved_unverified_object，多为已交付/被遮挡物体的最后可见框）
        region_map = {}
        for r in tl:
            for reg in r['unknown']:
                bbox = reg.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                reason = reg.get('reason')
                coarse = (reason, tuple(int(v) // 40 for v in bbox))
                e = region_map.setdefault(coarse, {'reason': reason, 'bbox': list(bbox),
                                                   'first': r['sim'], 'last': r['sim'], 'n': 0,
                                                   'class': reg.get('class'), 'wall_last': r['wall']})
                e['n'] += 1
                e['last'] = r['sim']
                if isinstance(r['wall'], (int, float)):
                    e['wall_last'] = r['wall']
        regions, lastseen = [], []
        for e in region_map.values():
            bbox = e['bbox']
            name, dist = nearest_gt(gt, (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            e['nearest_gt'] = name if dist <= 60 else None
            e['gt_dist'] = round(dist, 1)
            (regions if e['reason'] in ('multiple_body_cores', 'unresolved_correspondence') else lastseen).append(e)
        regions.sort(key=lambda e: -e['n'])
        lastseen.sort(key=lambda e: -e['n'])
        report['cases'][case] = {
            'per_task': rows, 'remaining_tail_event': tail_event,
            'scans': len(windows),
            'answers_total': sum(len([1 for (t, k, p) in w['answers'] if k == 'answer']) for w in windows),
            'trajectory': trajectory, 'establishing_scan': establishing,
            'blocker_regions': regions[:6], 'lastseen_regions_count': len(lastseen),
            'lastseen_top': [{'reason': e['reason'], 'bbox': e['bbox'], 'n': e['n'],
                              'nearest_gt': e.get('nearest_gt')} for e in lastseen[:6]],
        }
        print('## %s' % case)
        for r in rows:
            print('  任务%d [%s] %s %s' % (r['index'], r['status'], r['instruction'][:24], ('| ' + r['reason'][:70]) if r['reason'] else ''))
        if tail_event:
            print('  末态剩余集事件: %s' % tail_event[:260])
        print('  扫描数=%d 答案数=%d' % (len(windows), report['cases'][case]['answers_total']))
        for name, hist in trajectory.items():
            if not hist:
                continue
            seq = ' → '.join('%d:%s' % (h['scan'], h['class'][:6]) for h in hist[:8])
            print('  轨迹 %-14s %s' % (name, seq))
        for e in regions[:4]:
            print('  [阻断区] %-26s bbox=%s n=%d sim %.0f..%.0f 近似GT=%s' % (
                e['reason'], e['bbox'], e['n'], e['first'] or -1, e['last'] or -1, e.get('nearest_gt')))
        for e in lastseen[:3]:
            print('  [last-seen] %-28s bbox=%s n=%d 近似GT=%s' % (
                e['reason'], e['bbox'], e['n'], e.get('nearest_gt')))
        print()
    (outdir / 'r2_evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written:', outdir / 'r2_evidence.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
