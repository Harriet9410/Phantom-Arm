#!/usr/bin/env python3
"""R8-B4 离线分类门：在保存的原图上验证新协议（不启 ROS、不驱动机器人）。

服务器执行： /usr/bin/python3 /root/ds_r8b_offline.py
- 载入一次 /root/inference/FM9G4B-V 基座模型（与 nine_node 的 grounding=False 路径一致）。
- 对每个标注实例：原图两视图裁剪（边距 0.25 / 0.85）+ 结构化短协议（max_new_tokens=32）。
- 两视图一致才判类别；否则记 unknown。逐实例记录原始回答与解析原因。
- 输出混淆矩阵与四项计数（正确/错误/unknown/协议失败），写 r8b_offline.json

真值来自 R7/R8 报告已确认的实例；真值只在评价端使用。
"""
import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

RUNS = Path('/root/gpufree-data/tcei_260920v2')
OUT = Path('/root/gpufree-data/r8b_offline')
MODEL = '/root/inference/FM9G4B-V'
CLASS_ZH = {'Smokegrenade': u'烟雾弹', 'Magazine': u'弹夹', 'Torch': u'军用手电筒', 'Grenade': u'手雷'}
DECLARED = ['Grenade', 'Magazine', 'Smokegrenade', 'Torch']
MAX_NEW_TOKENS = 32
VIEW_MARGINS = (0.25, 0.85)
FIELD_RE = re.compile(r'"class"\s*:\s*"([A-Za-z_]+)"')

PROMPT = (u'这是军用物资分类任务。可选类别只有：%s。'
          u'只输出一个 JSON 对象，形如 {"class":"类名"}，类名必须是上面之一；'
          u'实在无法确定时输出 {"class":"unknown"}。不要输出任何其他文字、解释或标点。'
          % u'、'.join(u'%s（%s）' % (c, CLASS_ZH[c]) for c in DECLARED))

# (batch, case, scan_id, proposal_index, px, bbox, truth, note)
# 真值仅取 R7/R8 报告已确证的实例；bbox 取自冲突日志里的真实 proposals。
INSTANCES = [
    ('b03_r7cand2_192941', 'scramble_01', '9b0d586878', 3, [671.5, 349.5], [641, 339, 697, 361],
     'Smokegrenade', 'F1: model answered Smokegrenade twice, program discarded it'),
    ('b03_r7cand_190538', 'scramble_02', '3b2a35810c', 3, [671.5, 349.5], [641, 339, 697, 361],
     'Smokegrenade', 's02: this object was later established as Torch (wrong)'),
    ('b03_r7cand2_192941', 'scramble_03', '1a81b4d6b1', 3, [660.04, 352.13], [644, 341, 696, 364],
     'Grenade', 'F6: hand_grenade; the later Torch reclass was wrong'),
    ('b03_r7cand_190538', 'scramble_04', 'a270e75518', 1, [485.26, 189.03], [470, 178, 522, 201],
     'Grenade', 'control: the real grenade'),
    ('b03_r7cand_190538', 'scramble_04', 'a270e75518', 2, [491.5, 355.5], [465, 344, 522, 367],
     'Smokegrenade', 'control: the second smoke bomb'),
    ('b03_r7cand_190538', 'scramble_04', 'a270e75518', 3, [671.5, 349.5], [641, 339, 697, 361],
     'Smokegrenade', 'F2: the real smoke bomb that became Torch'),
    ('b03_r7cand_190538', 'scramble_04', 'a270e75518', 4, [794.15, 182.45], [767, 162, 823, 204],
     'Magazine', 'control: the real magazine'),
    ('b03_r7cand_190538', 'scramble_04', 'a270e75518', 5, [794.38, 350.90], [762, 344, 826, 358],
     'Torch', 'control: the real flashlight'),
]


def load_model():
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16).eval().to('cuda')
    token = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    return model, token


def parse(text):
    if not text:
        return None, 'empty_answer'
    found = FIELD_RE.findall(str(text))
    if not found:
        return None, 'protocol_failure:no_class_field'
    if len(set(found)) != 1:
        return None, 'protocol_failure:multiple_conclusions'
    if '{' not in str(text) or '}' not in str(text):
        return None, 'protocol_failure:truncated_or_unbracketed'
    cls = found[0]
    if cls.lower() == 'unknown':
        return None, 'model_reported_unknown'
    if cls not in DECLARED:
        return None, 'protocol_failure:undeclared_class'
    return cls, 'ok'


def crop_view(rgb, bbox, margin):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    crop = rgb[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    scale = max(2, 160 // max(1, max(ch, cw)))
    up = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
    return up


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('loading model ...', flush=True)
    t0 = time.time()
    model, token = load_model()
    print('model ready in %.1fs' % (time.time() - t0), flush=True)
    rows = []
    conf = {}
    for batch, case, scan, pidx, px, bbox, truth, note in INSTANCES:
        img = RUNS / (batch + '_' + case + '_stack') / 'events/classification_conflicts' / (scan + '_raw.jpg')
        if not img.exists():
            rows.append({'case': case, 'scan': scan, 'truth': truth, 'error': 'missing image %s' % img})
            continue
        rgb = cv2.imread(str(img))
        if rgb is None:
            rows.append({'case': case, 'scan': scan, 'truth': truth, 'error': 'unreadable image'})
            continue
        views, details = [], []
        for margin in VIEW_MARGINS:
            up = crop_view(rgb, bbox, margin)
            pil = PILImage.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            torch.cuda.empty_cache()
            with torch.inference_mode():
                ans = model.chat(image=None, msgs=[{'role': 'user', 'content': [pil, PROMPT]}],
                                 tokenizer=token, max_new_tokens=MAX_NEW_TOKENS,
                                 sampling=False, num_beams=1)
            cls, why = parse(str(ans))
            views.append(cls)
            details.append({'margin': margin, 'raw': str(ans)[:160], 'parse': why,
                            'sha16': __import__('hashlib').sha256(up.tobytes()).hexdigest()[:16]})
            print('  %s %s m=%.2f -> %s (%s)' % (case, scan, margin, cls, why), flush=True)
        agree = views[0] is not None and views[0] == views[1]
        decided = views[0] if agree else None
        verdict = ('correct' if decided == truth else
                   'wrong' if decided is not None else
                   'unknown' if all(v is None for v in views) else 'unknown')
        conf[(truth, decided or 'unknown')] = conf.get((truth, decided or 'unknown'), 0) + 1
        rows.append({'case': case, 'scan': scan, 'proposal_index': pidx, 'pixel': px,
                     'truth': truth, 'note': note, 'views': views, 'details': details,
                     'decided': decided, 'verdict': verdict})
    total = len([r for r in rows if 'error' not in r])
    summary = {
        'instances': total,
        'correct': sum(1 for r in rows if r.get('verdict') == 'correct'),
        'wrong': sum(1 for r in rows if r.get('verdict') == 'wrong'),
        'unknown': sum(1 for r in rows if r.get('verdict') == 'unknown'),
        'protocol_failures': sum(1 for r in rows for d in r.get('details', [])
                                 if str(d.get('parse', '')).startswith('protocol_failure')),
        'views_requested': 2 * total,
        'confusion': {'%s -> %s' % k: v for k, v in sorted(conf.items())},
        'protocol': {'max_new_tokens': MAX_NEW_TOKENS, 'view_margins': list(VIEW_MARGINS),
                     'prompt': PROMPT, 'model': MODEL, 'adapter': 'disabled (base model)'},
        'rows': rows,
    }
    (OUT / 'r8b_offline.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('instances', 'correct', 'wrong', 'unknown',
                                              'protocol_failures', 'confusion')},
                     ensure_ascii=False, indent=1))
    print('written:', OUT / 'r8b_offline.json')


if __name__ == '__main__':
    main()
