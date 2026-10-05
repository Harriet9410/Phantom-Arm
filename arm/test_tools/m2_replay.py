#!/usr/bin/env python3
"""M2 replay on the ORIGINAL label-establishing frames (review R2-2).

For each frame+moment where a wrong label was first established (from R2-0 evidence),
re-run the runtime's full classification pipeline with PER-VOTE instrumentation:
full-frame numbered query, per-candidate EN/CN crops, GS/GT pairwise rechecks, final
four-way — each vote logged with raw text and old/new parse.  Then reconstruct the
final assignment exactly as nine_classify would (majority + anchor tie-break + recheck
overwrites + fastpath) and answer: did the GS/GT recheck overwrite a correct class on
THIS frame, and did fastpath skip the final verdict?

Run with the inference env python after all stacks are stopped:
  /opt/conda/envs/inference/bin/python _ds_m2_replay.py --out <outdir>
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

# prompts copied verbatim from nine_classify.py
EN_CROP = u'这个物资是什么类别？只回答类名：Grenade、Magazine、Smokegrenade、Torch。'
CN_CROP = u'这个物资是手雷、弹夹、烟雾弹、军用手电筒中的哪一种？只回答类名：Grenade、Magazine、Smokegrenade、Torch。'
GS = (u'仔细看这个物体的形状：烟雾弹（Smokegrenade）是圆柱形容器，'
      u'常带绿色环带；手雷（Grenade）是小型椭球体。'
      u'这个物体是哪一类？只回答 Smokegrenade 或 Grenade。')
GT_P = (u'仔细看这个物体的形状：手雷（Grenade）是小型椭球体，表面常有'
        u'网格状防滑纹；军用手电筒（Torch）是细长圆柱形，一端有尾盖或按钮。'
        u'这个物体是哪一类？只回答 Grenade 或 Torch。')
FIVE = (u'这是军用物资，四选一：Smokegrenade（烟雾弹，圆柱形容器，'
        u'常带绿色环带）、Grenade（手雷，小型椭球体，整体圆润带网格状'
        u'防滑纹）、Torch（军用手电筒，细长圆柱形，一端有尾盖或按钮）、'
        u'Magazine（弹夹，扁平长条形弹匣，一侧平直，常可见排列的弹壳'
        u'或供弹口）。这个物体是哪一类？只回答类名。')
FULL = (u'图中篮筐内有编号%s的物体。请分别判断每个编号物体的物资类别。'
        u'类别只能是：%s。只输出一个 JSON 对象，形如 {%s}，不要输出其他文字。')
NAMES = u'、'.join('%s（%s）' % (cls, zh) for zh, cls in
                   [('烟雾弹', 'Smokegrenade'), ('弹夹', 'Magazine'),
                    ('手雷', 'Grenade'), ('军用手电筒', 'Torch')])
CLASSES = ('Grenade', 'Magazine', 'Smokegrenade', 'Torch', 'CompressedFood')
ALIAS = {'手榴弹': 'Grenade', '手雷': 'Grenade', '军用手电筒': 'Torch', '手电筒': 'Torch',
         '弹夹': 'Magazine', '弹匣': 'Magazine', '烟雾弹': 'Smokegrenade', '压缩': 'CompressedFood'}


def new_parse(text):
    if not text:
        return None
    raw = str(text)
    low = raw.lower()
    spans = []
    for cls in sorted(CLASSES, key=len, reverse=True):
        for m in re.finditer(re.escape(cls.lower()), low):
            spans.append([m.start(), m.end(), cls])
    for canonical in set(ALIAS.values()):
        for alias, canon in [(a, c) for a, c in ALIAS.items() if c == canonical]:
            if not any('\u4e00' <= ch <= '\u9fff' for ch in alias):
                continue
            for m in re.finditer(re.escape(alias), raw):
                spans.append([m.start(), m.end(), canon])
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
        if any(max(0, s[0] - ne) + max(0, ns - s[1]) <= 6 for ns, ne in negs):
            continue
        final.append(s)
    classes = {s[2] for s in final}
    return classes.pop() if len(classes) == 1 else None


def gt_objects(case_doc):
    out = []
    for obj in case_doc.get('objects', []):
        x, y = obj['position'][0], obj['position'][1]
        out.append({'name': obj['path'].split('/')[-1], 'u': U0 + UX * x, 'v': V0 + VY * y})
    return out


def nearest_gt(gt, cx, cy):
    best = min(gt, key=lambda g: (g['u'] - cx) ** 2 + (g['v'] - cy) ** 2)
    return best['name'], math.hypot(best['u'] - cx, best['v'] - cy)


def candidates_at_wall(rd, wall):
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
        if isinstance(w, (int, float)) and w <= wall:
            picked = v['candidates']
    return picked or []


def frame_at(rd, wall):
    rows = [json.loads(l) for l in (rd / 'rgbd' / 'index.jsonl').read_text(errors='ignore').splitlines() if l.strip()]
    best = min(rows, key=lambda r: abs((r.get('captured_at') or 0) - wall))
    return rd / 'rgbd' / (best['stem'] + '.jpg'), best['stem']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        {'case': 'scramble_01', 'wall': 1791121351.7304425, 'note': '首扫：手雷被标 Smokegrenade'},
        {'case': 'scramble_03', 'wall': 1791122036.0473115, 'note': '首扫：手雷被标 Smokegrenade + 手电筒框被标 Grenade'},
        {'case': 'scramble_04', 'wall': 1791122457.5586126, 'note': '首扫：手雷标对 Grenade + 手电筒框被标 Grenade'},
    ]
    from PIL import Image
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

    def ask(image, prompt):
        with model.disable_adapter():
            return str(model.chat(image=None, msgs=[{'role': 'user', 'content': [image, prompt]}],
                                  tokenizer=tokenizer, max_new_tokens=400, sampling=False, num_beams=1))

    report = []
    for tgt in targets:
        case = tgt['case']
        rd = RUNS / ('b03_base_10042140_' + case + '_round')
        sd = RUNS / ('b03_base_10042140_' + case + '_stack')
        case_doc = json.loads((sd / 'case.json').read_text(encoding='utf-8'))
        gt = gt_objects(case_doc)
        cands = candidates_at_wall(rd, tgt['wall'])
        frame, stem = frame_at(rd, tgt['wall'])
        img = Image.open(frame).convert('RGB')
        W, H = img.size
        entry = {'case': case, 'note': tgt['note'], 'frame': stem,
                 'candidates': [], 'assign_old_pipeline': {}}
        print('=== %s frame=%s (%s)' % (case, stem, tgt['note']))
        # 全图编号查询（运行时的锚点，计双票）
        ids = [str(i + 1) for i in range(len(cands))]
        raw_full = ask(img, FULL % ('、'.join(ids), NAMES, ','.join('"%s":"类名"' % i for i in ids)))
        print('  FULL raw:', raw_full.replace('\n', ' ')[:150])
        full_map = {}
        try:
            full_map = {int(k): v for k, v in json.loads(raw_full).items() if str(k).isdigit()}
        except Exception:
            pass
        votes = {}
        props = []
        for i, c in enumerate(cands):
            bbox = c.get('bbox')
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            name, dist = nearest_gt(gt, cx, cy)
            mx, my = int((bbox[2] - bbox[0]) * 0.6), int((bbox[3] - bbox[1]) * 0.6)
            box = (max(0, bbox[0] - mx), max(0, bbox[1] - my), min(W, bbox[2] + mx), min(H, bbox[3] + my))
            crop = img.crop(box)
            cw, ch = crop.size
            scale = max(2, 160 // max(1, max(ch, cw)))
            up = crop.resize((cw * scale, ch * scale), Image.BICUBIC)
            props.append({'pid': i + 1, 'bbox': bbox, 'gt': name, 'dist': round(dist, 1), 'crop': up})
        for p in props:
            v = []
            anchor = full_map.get(p['pid'])
            if anchor and new_parse(anchor) and new_parse(anchor) in CLASSES:
                v.append({'vote': 'full', 'raw': anchor, 'parsed': new_parse(anchor)})
                v.append({'vote': 'full', 'raw': anchor, 'parsed': new_parse(anchor)})
            r_en = ask(p['crop'], EN_CROP)
            v.append({'vote': 'crop_en', 'raw': r_en[:120], 'parsed': new_parse(r_en)})
            r_cn = ask(p['crop'], CN_CROP)
            v.append({'vote': 'crop_cn', 'raw': r_cn[:120], 'parsed': new_parse(r_cn)})
            # 运行时的多数+锚点裁决
            tally = {}
            for x in v:
                if x['parsed']:
                    tally[x['parsed']] = tally.get(x['parsed'], 0) + 1
            best_cls, best_n = None, 0
            for cls in sorted(tally):
                if tally[cls] > best_n or (tally[cls] == best_n and cls == (v[0]['parsed'] if v else None)):
                    best_cls, best_n = cls, tally[cls]
            assign = best_cls if best_n >= 2 else (v[0]['parsed'] if v and v[0]['parsed'] else (v[-1]['parsed'] if v else None))
            # 配对复核（GS/GT，按运行时条件）+ 四选一终审
            rechecks = []
            if assign in ('Grenade', 'Smokegrenade'):
                a1, a2 = new_parse(ask(p['crop'], GS)), new_parse(ask(p['crop'], GS))
                if a1 and a1 == a2:
                    rechecks.append({'recheck': 'GS', 'raw': a1, 'overwrote_to': a1})
                    assign = a1
            if assign in ('Grenade', 'Torch'):
                a1, a2 = new_parse(ask(p['crop'], GT_P)), new_parse(ask(p['crop'], GT_P))
                if a1 and a1 == a2:
                    rechecks.append({'recheck': 'GT', 'raw': a1, 'overwrote_to': a1})
                    assign = a1
            all_same = len(v) >= 4 and len({x['parsed'] for x in v if x['parsed']}) == 1 and all(x['parsed'] for x in v)
            fastpath_skipped = bool(all_same)
            if not fastpath_skipped:
                a1, a2 = new_parse(ask(p['crop'], FIVE)), new_parse(ask(p['crop'], FIVE))
                if a1 and a1 == a2 and a1 != assign:
                    rechecks.append({'recheck': 'FIVE', 'raw': a1, 'overwrote_to': a1})
                    assign = a1
            p['votes'] = v
            p['rechecks'] = rechecks
            p['fastpath_skipped'] = fastpath_skipped
            p['final_assign'] = assign
            p['correct'] = assign == p['gt']
            entry['candidates'].append({k: p[k] for k in ('pid', 'bbox', 'gt', 'dist', 'final_assign', 'correct',
                                                          'fastpath_skipped', 'votes', 'rechecks')})
            entry['assign_old_pipeline'][p['pid']] = assign
            print('  #%d GT=%-13s final=%-13s correct=%s fastpath=%s rechecks=%s' % (
                p['pid'], p['gt'], assign, p['correct'], fastpath_skipped,
                ';'.join(r['recheck'] + '->' + r['overwrote_to'] for r in rechecks) or '-'))
        report.append(entry)

    (outdir / 'm2_replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written:', outdir / 'm2_replay.json')


if __name__ == '__main__':
    sys.exit(main())
