#!/usr/bin/env python3
"""M1 offline model replay (review R2-1 adoption evidence).

Builds a ground-truth-labelled crop set from the five baseline cases' FIRST scan
(candidates from scalars, GT identity via the calibrated projection), asks the model
the SAME crop prompt the runtime uses, then parses every raw answer with the OLD
(dev_v7_t1_41) and NEW (M1) parsers and reports per-class correct/wrong/abstain.

The model is loaded exactly like nine_node does (FM9G4B-V + adapter loaded then
DISABLED for classification, greedy num_beams=1), so answers are the runtime's own.

Run with the inference env python AFTER all stacks are stopped:
  /opt/conda/envs/inference/bin/python _ds_m1_replay.py --out <outdir>
"""
import argparse
import json
import math
import pathlib
import re
import sys
import time

U0, UX, V0, VY = 643.5, -569.3, 297.4, 629.6
RUNS = pathlib.Path('/root/gpufree-data/tcei_260920v2')
BATCH = RUNS / 'b03_base_10042140'
MODEL = '/root/inference/FM9G4B-V'
ADAPTER = '/root/sft_v2/outputs/qlora_crop/step_27000'
CN_CROP = u'这个物资是什么类别？只回答类名：Grenade、Magazine、Smokegrenade、Torch。'
CLASSES = ('Grenade', 'Magazine', 'Smokegrenade', 'Torch', 'CompressedFood')
ALIAS = {'手榴弹': 'Grenade', '手雷': 'Grenade', '军用手电筒': 'Torch', '手电筒': 'Torch',
         '弹夹': 'Magazine', '弹匣': 'Magazine', '烟雾弹': 'Smokegrenade', '压缩': 'CompressedFood'}


def old_parse(text):
    low = str(text or '').lower()
    for cls in sorted(CLASSES, key=len, reverse=True):
        if cls.lower() in low:
            return cls
    return None


def new_parse(text):
    """Mirror of nine_classify._parse_class_answer (M1) for offline replay."""
    if not text:
        return None
    raw = str(text)
    low = raw.lower()
    spans = []
    for cls in sorted(CLASSES, key=len, reverse=True):
        for m in re.finditer(re.escape(cls.lower()), low):
            spans.append([m.start(), m.end(), cls])
    for canonical, aliases in ALIAS.items():
        for m in re.finditer(re.escape(canonical), raw):
            spans.append([m.start(), m.end(), ALIAS[canonical]])
    if not spans:
        return None
    kept = [s for s in spans
            if not any(o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0]) for o in spans)]
    negs = []
    for neg in ('不是', '并不是', '并非', '不算', '没有', '非', 'not', 'non-'):
        for m in re.finditer(re.escape(neg.lower()), low):
            negs.append((m.start(), m.end()))
    final = []
    for s in kept:
        hit = False
        for ns, ne in negs:
            gap = max(0, s[0] - ne) + max(0, ns - s[1])
            if gap <= 6:
                hit = True
                break
        if not hit:
            final.append(s)
    classes = {s[2] for s in final}
    return classes.pop() if len(classes) == 1 else None


def gt_objects(case):
    out = []
    for obj in case.get('objects', []):
        x, y = obj['position'][0], obj['position'][1]
        out.append({'name': obj['path'].split('/')[-1], 'u': U0 + UX * x, 'v': V0 + VY * y})
    return out


def nearest_gt(gt, cx, cy):
    best = min(gt, key=lambda g: (g['u'] - cx) ** 2 + (g['v'] - cy) ** 2)
    return best['name'], math.hypot(best['u'] - cx, best['v'] - cy)


def build_samples():
    """First-scan candidates per baseline case -> GT-labelled crops."""
    from PIL import Image
    samples = []
    for case in ['scramble_01', 'scramble_02', 'scramble_03', 'scramble_04', 'scramble_05']:
        rd = RUNS / ('b03_base_10042140_' + case + '_round')
        sd = RUNS / ('b03_base_10042140_' + case + '_stack')
        case_doc = json.loads((sd / 'case.json').read_text(encoding='utf-8'))
        gt = gt_objects(case_doc)
        # 首次扫描完成时刻（perception.log 的第一条 nine scan）
        first_scan_wall = None
        for line in (sd / 'logs' / 'perception.log').read_text(errors='ignore').splitlines():
            m = re.search(r'\[([\d.]+)\].*nine scan', line)
            if m:
                first_scan_wall = float(m.group(1))
                break
        # 首扫前最近的候选消息（wall 基）
        picked = None
        for line in (rd / 'scalars' / 'events.jsonl').read_text(errors='ignore').splitlines():
            if '"candidates"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            v = d.get('value') or {}
            if not isinstance(v, dict) or not v.get('candidates'):
                continue
            w = d.get('received_wall')
            if first_scan_wall and isinstance(w, (int, float)) and w <= first_scan_wall:
                picked = (w, v['candidates'])
        if not picked:
            print(case, ': no pre-scan candidates, skip')
            continue
        wall, cands = picked
        # 帧：captured_at 最接近 wall 的帧
        idx_rows = [json.loads(l) for l in (rd / 'rgbd' / 'index.jsonl').read_text(errors='ignore').splitlines() if l.strip()]
        best = min(idx_rows, key=lambda r: abs((r.get('captured_at') or 0) - wall))
        frame = rd / 'rgbd' / (best['stem'] + '.jpg')
        img = Image.open(frame).convert('RGB')
        W, H = img.size
        for c in cands:
            bbox = c.get('bbox')
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            name, dist = nearest_gt(gt, cx, cy)
            if dist > 45:
                continue                                   # 不确定对应关系的不收
            mx, my = int((bbox[2] - bbox[0]) * 0.6), int((bbox[3] - bbox[1]) * 0.6)
            box = (max(0, bbox[0] - mx), max(0, bbox[1] - my), min(W, bbox[2] + mx), min(H, bbox[3] + my))
            crop = img.crop(box)
            cw, ch = crop.size
            scale = max(2, 160 // max(1, max(ch, cw)))
            up = crop.resize((cw * scale, ch * scale), Image.BICUBIC)
            samples.append({'case': case, 'frame': best['stem'], 'bbox': bbox,
                            'gt': name, 'dist': round(dist, 1), 'image': up})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = build_samples()
    print('GT-labelled crops:', len(samples))
    from collections import Counter
    print('per-class:', dict(Counter(s['gt'] for s in samples)))

    import torch
    from transformers import AutoModel, AutoTokenizer
    began = time.time()
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().to('cuda')
    model.config.vision_batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print('model loaded %.1fs' % (time.time() - began))

    rows = []
    for i, s in enumerate(samples):
        with model.disable_adapter():
            answer = model.chat(image=None, msgs=[{'role': 'user', 'content': [s['image'], CN_CROP]}],
                                tokenizer=tokenizer, max_new_tokens=400, sampling=False, num_beams=1)
        raw = str(answer)
        o, n = old_parse(raw), new_parse(raw)
        rows.append({'case': s['case'], 'frame': s['frame'], 'bbox': s['bbox'], 'gt': s['gt'],
                     'raw': raw[:200], 'old': o, 'new': n,
                     'old_ok': o == s['gt'], 'new_ok': n == s['gt']})
        print('%2d %-12s GT=%-13s old=%-13s new=%-13s | %s' % (
            i, s['case'], s['gt'], o, n, raw.replace('\n', ' ')[:60]))

    def tally(key_ok, key_parse):
        from collections import Counter
        t = {}
        for r in rows:
            c = r['gt']
            t.setdefault(c, Counter())
            if r[key_parse] is None:
                t[c]['abstain'] += 1
            elif r[key_ok]:
                t[c]['correct'] += 1
            else:
                t[c]['wrong'] += 1
        return t
    summary = {'n': len(rows), 'old': tally('old_ok', 'old'), 'new': tally('new_ok', 'new'),
               'flips': [{'case': r['case'], 'gt': r['gt'], 'old': r['old'], 'new': r['new'],
                          'raw': r['raw']} for r in rows if r['old'] != r['new']]}
    (outdir / 'm1_replay.json').write_text(json.dumps({'rows': rows, 'summary': summary},
                                                      ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary['old'], ensure_ascii=False))
    print(json.dumps(summary['new'], ensure_ascii=False))
    print('flips:', len(summary['flips']))
    print('written:', outdir / 'm1_replay.json')


if __name__ == '__main__':
    sys.exit(main())
