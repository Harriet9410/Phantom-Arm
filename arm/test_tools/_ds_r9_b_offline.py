#!/usr/bin/env python3
"""R9 | B 离线补证：实际 5 类声明 + 在线等效模型包装 + 预固定 40 观测。

服务器执行： /usr/bin/python3 /root/ds_r9_b_offline.py

严格按 R9 要求：
  * 复用**候选自身**的 _class_prompt() 与 _parse_structured_class()（同一源码文件，
    与拟部署实现一致），类别集合取真实的 sorted(CLASS_ALIASES)（含 CompressedFood，共 5 类）。
  * 模型包装与在线一致：基座 + PeftModel(adapter) + 关闭适配器（nine_node grounding=False 路径）；
    若适配器缺失或包装失败，如实记录为偏差，不假装等效。
  * 样本预先固定（本文件常量，运行前确定）：每案最早两个含 5 件实物的扫描 × 5 件 = 40 观测。
    真值由各案 case.json 的物体布局 + make_scramble10 投影 + 该扫描的 proposals 关联确认。
  * 两臂：主臂=候选两视图裁剪协议；备臂=**事先定好的唯一替代输入**（全图+画出目标框）。
  * 逐条记录：原始文本、输入图哈希、提示、类别集合、解析原因、输出类别、真值、耗时。

输出： /root/gpufree-data/r9_b_offline/r9_b_offline.json
"""
import ast
import hashlib
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

PKG = Path('/root/tcei_final_v2_23/tcei_260920v2')
RUNS = Path('/root/gpufree-data/tcei_260920v2')
MODEL = '/root/inference/FM9G4B-V'
ADAPTER = '/root/sft_v2/outputs/qlora_crop/step_27000'
OUT = Path('/root/gpufree-data/r9_b_offline')
CANDIDATE = Path('/root/r9_candidate_nine_classify.py')
MAX_NEW_TOKENS = 32
VIEW_MARGINS = (0.25, 0.85)

# 预先固定的样本：每案最早两个含 5 件实物的扫描（运行前确定，不因结果改动）
SAMPLE = {'scramble_01': ['9b0d586878', '644ceabb3c'],
          'scramble_02': ['3b2a35810c', 'fd1fb8f0f8'],
          'scramble_03': ['1a81b4d6b1', 'ec1f1dea7a'],
          'scramble_04': ['0a97d261cc', 'a270e75518']}
CLASS_OF_OBJECT = {'smoke_bomb': 'Smokegrenade', 'smoke_bomb_01': 'Smokegrenade',
                   'Magazines': 'Magazine', 'Flashlight': 'Torch', 'hand_grenade': 'Grenade'}
ASSOC_TOL_PX = 30.0

sys.path.insert(0, str(PKG / 'test_tools'))
import make_scramble10 as M  # noqa: E402  (authoritative projection + pixel_xy)


def load_candidate_methods():
    """Exec the candidate's own prompt/parse/crop/classify methods (same file that ships)."""
    tree = ast.parse(CANDIDATE.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'NineRotationDetector')
    names = ('_class_prompt', '_parse_structured_class', '_classify_view_crop', '_classify_proposals')
    methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names), [m.name for m in methods]
    return methods


def declared_classes():
    """The real declared set, from CLASS_ALIASES (never a hand-written list)."""
    src = (PKG / 'tcei_stack/perception_tracking.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'CLASS_ALIASES' for t in node.targets):
            return sorted(ast.literal_eval(node.value))
    raise RuntimeError('CLASS_ALIASES not found')


def build_prompt_fn(methods, ns):
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    return ns


def proposals_for(case, scan_id):
    """The scan's proposals (bbox+pixel) from the run's own perception log."""
    log = RUNS / ('b03_r7cand_190538_' + case + '_stack') / 'logs/perception.log'
    if not log.exists():
        log = RUNS / ('b03_r7cand2_192941_' + case + '_stack') / 'logs/perception.log'
    for line in log.read_text(encoding='utf-8', errors='ignore').splitlines():
        if 'nine scan conflict images' not in line:
            continue
        m = re.search(r'nine scan conflict images:\s*(\{.*\})', line)
        if not m:
            continue
        try:
            d = json.loads(m.group(1))
        except Exception:
            continue
        if d.get('scan_id') == scan_id:
            return d.get('proposals', [])
    return []


def truth_map(case):
    """object_id -> (expected pixel, official class) for the case's five objects."""
    cj = json.loads((PKG / 'cases_scramble10' / (case + '.json')).read_text(encoding='utf-8'))
    out = {}
    for o in cj['objects']:
        px = M.pixel_xy(*o['reference_xy'])
        out[o['object_id']] = (px, CLASS_OF_OBJECT[o['object_id']])
    return out


def associate(proposals, truth):
    """Nearest unique proposal <-> object match by projected pixel; returns rows or a reason."""
    rows, used = [], set()
    for i, pr in enumerate(proposals, 1):
        px = pr.get('pixel')
        if not px:
            continue
        cands = sorted(((((px[0] - v[0][0]) ** 2 + (px[1] - v[0][1]) ** 2) ** .5), k)
                       for k, v in truth.items())
        if not cands or cands[0][0] > ASSOC_TOL_PX or cands[0][1] in used:
            rows.append({'proposal_index': i, 'pixel': px, 'bbox': pr.get('bbox'),
                         'truth': None, 'assoc_px': round(cands[0][0], 1) if cands else None,
                         'assoc_reason': 'out_of_tolerance' if cands and cands[0][0] > ASSOC_TOL_PX
                                         else 'already_used'})
            continue
        used.add(cands[0][1])
        rows.append({'proposal_index': i, 'pixel': px, 'bbox': pr.get('bbox'),
                     'object_id': cands[0][1], 'truth': truth[cands[0][1]][1],
                     'assoc_px': round(cands[0][0], 1)})
    return rows


def crop(rgb, bbox, margin):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    c = rgb[cy1:cy2, cx1:cx2]
    ch, cw = c.shape[:2]
    s = max(2, 160 // max(1, max(ch, cw)))
    return cv2.resize(c, (cw * s, ch * s), interpolation=cv2.INTER_CUBIC)


def marked_full(rgb, bbox):
    """The one pre-declared alternative input: the whole frame with the target box drawn."""
    out = rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declared = declared_classes()
    methods = load_candidate_methods()
    ns = {'cv2': cv2, 'np': np, 're': re, 'json': json, 'uuid': __import__('uuid'),
          'time': types.SimpleNamespace(monotonic=time.monotonic),
          'rospy': types.SimpleNamespace(logwarn_throttle=lambda *a: None),
          'CLASS_ZH': {'Smokegrenade': u'烟雾弹', 'Magazine': u'弹夹',
                       'Torch': u'军用手电筒', 'Grenade': u'手雷',
                       'CompressedFood': u'压缩干粮'},
          'CLASS_MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'VIEW_MARGINS': VIEW_MARGINS}
    ns = build_prompt_fn(methods, ns)

    from transformers import AutoModel, AutoTokenizer
    print('declared classes:', declared, flush=True)
    print('loading base model ...', flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16).eval().to('cuda')
    token = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    adapter_state = 'absent'
    disable_adapter = None
    if Path(ADAPTER).is_dir():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER)
            model.eval()
            disable_adapter = model.disable_adapter
            adapter_state = 'wrapped_and_disabled'
        except Exception as e:
            adapter_state = 'wrap_failed: %s' % e
    print('model ready in %.1fs; adapter=%s' % (time.time() - t0, adapter_state), flush=True)

    prompt = ns['_class_prompt'](types.SimpleNamespace(capabilities={'declared_classes': declared}))
    import contextlib

    def adapter_ctx():
        # disable_adapter() is single-use, so build a fresh context per call
        return disable_adapter() if disable_adapter else contextlib.nullcontext()

    rows = []
    for case, scans in SAMPLE.items():
        truth = truth_map(case)
        for scan in scans:
            img = None
            for batch in ('b03_r7cand_190538', 'b03_r7cand2_192941'):
                p = RUNS / (batch + '_' + case + '_stack') / 'events/classification_conflicts' / (scan + '_raw.jpg')
                if p.exists():
                    img = p
                    break
            if img is None:
                rows.append({'case': case, 'scan': scan, 'error': 'image missing'})
                continue
            rgb = cv2.imread(str(img))
            props = proposals_for(case, scan)
            obs = associate(props, truth)
            print('%s %s: %d proposals, %d associated' % (case, scan, len(props),
                                                          sum(1 for o in obs if o.get('truth'))), flush=True)
            for o in obs:
                rec = {'case': case, 'scan': scan, 'image': str(img), 'bbox': o['bbox'],
                       'truth': o.get('truth'), 'object_id': o.get('object_id'),
                       'assoc_px': o.get('assoc_px'), 'assoc_note': o.get('assoc_reason'),
                       'arms': {}}
                if not o.get('bbox'):
                    rec['arms']['error'] = 'no bbox for proposal'
                    rows.append(rec)
                    continue
                # 主臂：候选两视图裁剪
                views = []
                for margin in VIEW_MARGINS:
                    up = crop(rgb, o['bbox'], margin)
                    pil = PILImage.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
                    t1 = time.time()
                    torch.cuda.empty_cache()
                    with torch.inference_mode(), adapter_ctx():
                        ans = model.chat(image=None, msgs=[{'role': 'user', 'content': [pil, prompt]}],
                                         tokenizer=token, max_new_tokens=MAX_NEW_TOKENS,
                                         sampling=False, num_beams=1)
                    views.append({'margin': margin, 'raw': str(ans)[:200],
                                  'ms': round((time.time() - t1) * 1000),
                                  'sha16': hashlib.sha256(up.tobytes()).hexdigest()[:16],
                                  'parse': ns['_parse_structured_class'](
                                      types.SimpleNamespace(capabilities={'declared_classes': declared}),
                                      str(ans))[1],
                                  'class': ns['_parse_structured_class'](
                                      types.SimpleNamespace(capabilities={'declared_classes': declared}),
                                      str(ans))[0]})
                agree = views[0]['class'] is not None and views[0]['class'] == views[1]['class']
                rec['arms']['two_view'] = {'views': views, 'decided': views[0]['class'] if agree else None}
                # 备臂：全图 + 目标框（唯一事先定好的替代输入）
                full = marked_full(rgb, o['bbox'])
                pil = PILImage.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
                t1 = time.time()
                torch.cuda.empty_cache()
                with torch.inference_mode(), adapter_ctx():
                    ans = model.chat(image=None, msgs=[{'role': 'user', 'content': [pil, prompt]}],
                                     tokenizer=token, max_new_tokens=MAX_NEW_TOKENS,
                                     sampling=False, num_beams=1)
                cls, why = ns['_parse_structured_class'](
                    types.SimpleNamespace(capabilities={'declared_classes': declared}), str(ans))
                rec['arms']['marked_full'] = {'raw': str(ans)[:200], 'ms': round((time.time() - t1) * 1000),
                                              'sha16': hashlib.sha256(full.tobytes()).hexdigest()[:16],
                                              'parse': why, 'class': cls, 'decided': cls}
                rows.append(rec)

    def tally(arm):
        known_right = known_wrong = unknown = proto = 0
        conf = {}
        for r in rows:
            if 'error' in r or not r.get('truth'):
                continue
            a = r['arms'].get(arm)
            if not a:
                continue
            d = a.get('decided')
            fails = [v for v in (a.get('views') or [a]) if str(v.get('parse', '')).startswith('protocol_failure')]
            proto += len(fails)
            if d is None:
                unknown += 1
                key = '%s -> unknown' % r['truth']
            elif d == r['truth']:
                known_right += 1
                key = '%s -> %s' % (r['truth'], d)
            else:
                known_wrong += 1
                key = '%s -> %s' % (r['truth'], d)
            conf[key] = conf.get(key, 0) + 1
        return {'correct_known': known_right, 'wrong_known': known_wrong,
                'unknown': unknown, 'protocol_failures': proto, 'confusion': dict(sorted(conf.items()))}

    per_case = {}
    for case in SAMPLE:
        rs = [r for r in rows if r.get('case') == case and r.get('truth')]
        per_case[case] = {'observations': len(rs),
                          'two_view_known': sum(1 for r in rs if r['arms']['two_view']['decided']),
                          'marked_full_known': sum(1 for r in rs if r['arms']['marked_full']['decided'])}
    summary = {'declared_classes': declared, 'adapter': adapter_state, 'model': MODEL,
               'max_new_tokens': MAX_NEW_TOKENS, 'view_margins': list(VIEW_MARGINS),
               'sample': SAMPLE, 'assoc_tol_px': ASSOC_TOL_PX,
               'observations_planned': 40, 'observations_used': len([r for r in rows if r.get('truth')]),
               'arms': {'two_view': tally('two_view'), 'marked_full': tally('marked_full')},
               'per_case_coverage': per_case, 'rows': rows}
    (OUT / 'r9_b_offline.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'observations_used': summary['observations_used'], 'adapter': adapter_state,
                      'two_view': summary['arms']['two_view'], 'marked_full': summary['arms']['marked_full'],
                      'per_case': per_case}, ensure_ascii=False, indent=1))
    print('written:', OUT / 'r9_b_offline.json')


if __name__ == '__main__':
    main()
