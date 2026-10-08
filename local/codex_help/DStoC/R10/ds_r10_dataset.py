#!/usr/bin/env python3
"""R10：从已保存冲突原图构建完整双视图分类数据集（不加载模型、不跑仿真）。

服务器执行： /opt/conda/envs/inference/bin/python3 /root/ds_r10_dataset.py

做法
  * 遍历两个 R7 批次的全部九格冲突原图（`_raw.jpg`，覆盖 4 个场景、多个时刻）。
  * 用**同一套已确证的关联**（case 布局 + make_scramble10 投影 + 该扫描 proposals，容差 30px，
    唯一匹配）为每个提案确定真值类别。
  * 按候选协议在原图上按两个边距裁剪；保留 Torch 窄视图，
    因为在线候选没有额外的 24px 过滤。
  * **按场景划分**训练/留出：s01+s02 训练，s03+s04 留出（R9 要求：未见摆位验证）。
  * 输出清单与统计；不做任何模型调用。

输出： /root/gpufree-data/r10_dataset/{manifest.json,summary.json} 与 crops/<split>/<class>/*.jpg
"""
import ast
import collections
import json
import re
import sys
from pathlib import Path

import cv2

PKG = Path('/root/tcei_final_v2_23/tcei_260920v2')
RUNS = Path('/root/gpufree-data/tcei_260920v2')
OUT = Path('/root/gpufree-data/r10_dataset')
BATCHES = ('b03_r7cand_190538', 'b03_r7cand2_192941')
TRAIN_SCENES = ('scramble_01', 'scramble_02')
HOLDOUT_SCENES = ('scramble_03', 'scramble_04')
CLASS_OF_OBJECT = {'smoke_bomb': 'Smokegrenade', 'smoke_bomb_01': 'Smokegrenade',
                   'Magazines': 'Magazine', 'Flashlight': 'Torch', 'hand_grenade': 'Grenade'}
ASSOC_TOL_PX = 30.0
MARGINS = (0.25, 0.85)          # 与候选 _classify_view_crop 一致
MIN_CROP_PX = 1                # 只排除空裁剪，与在线候选一致

sys.path.insert(0, str(PKG / 'test_tools'))
import make_scramble10 as M  # noqa: E402


def conflict_scans(case):
    """(scan_id, proposals) for every saved conflict image of this case."""
    found = {}
    for batch in BATCHES:
        log = RUNS / (batch + '_' + case + '_stack') / 'logs/perception.log'
        if not log.exists():
            continue
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
            sid = d.get('scan_id')
            if sid:
                found[sid] = d.get('proposals', [])
    return found


def image_path(case, scan):
    for batch in BATCHES:
        p = RUNS / (batch + '_' + case + '_stack') / 'events/classification_conflicts' / (scan + '_raw.jpg')
        if p.exists():
            return p
    return None


def truth_map(case):
    cj = json.loads((PKG / 'cases_scramble10' / (case + '.json')).read_text(encoding='utf-8'))
    return {o['object_id']: (M.pixel_xy(*o['reference_xy']), CLASS_OF_OBJECT[o['object_id']])
            for o in cj['objects']}


def associate(proposals, truth):
    rows, used = [], set()
    for i, pr in enumerate(proposals, 1):
        px, bbox = pr.get('pixel'), pr.get('bbox')
        if not px or not bbox:
            continue
        cands = sorted(((((px[0] - v[0][0]) ** 2 + (px[1] - v[0][1]) ** 2) ** .5), k)
                       for k, v in truth.items())
        if not cands or cands[0][0] > ASSOC_TOL_PX or cands[0][1] in used:
            continue
        used.add(cands[0][1])
        rows.append({'proposal_index': i, 'bbox': [int(v) for v in bbox], 'pixel': px,
                     'object_id': cands[0][1], 'truth': truth[cands[0][1]][1],
                     'assoc_px': round(cands[0][0], 1)})
    return rows


def crop(rgb, bbox, margin):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    c = rgb[cy1:cy2, cx1:cx2]
    ch, cw = c.shape[:2]
    if ch < MIN_CROP_PX or cw < MIN_CROP_PX:
        return None
    s = max(2, 160 // max(1, max(ch, cw)))
    return cv2.resize(c, (cw * s, ch * s), interpolation=cv2.INTER_CUBIC)


def main():
    (OUT / 'crops').mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], collections.Counter()
    scenes = list(TRAIN_SCENES) + list(HOLDOUT_SCENES)
    for case in scenes:
        split = 'train' if case in TRAIN_SCENES else 'holdout'
        truth = truth_map(case)
        scans = conflict_scans(case)
        for scan, props in sorted(scans.items()):
            img = image_path(case, scan)
            if img is None:
                skipped['image_missing'] += 1
                continue
            rgb = cv2.imread(str(img))
            if rgb is None:
                skipped['unreadable'] += 1
                continue
            for o in associate(props, truth):
                saved = []
                for margin in MARGINS:
                    c = crop(rgb, o['bbox'], margin)
                    if c is None:
                        skipped['crop_too_small'] += 1
                        continue
                    d = OUT / 'crops' / split / o['truth']
                    d.mkdir(parents=True, exist_ok=True)
                    name = '%s_%s_%s_m%02d.jpg' % (case, scan, o['object_id'], int(margin * 100))
                    p = d / name
                    cv2.imwrite(str(p), c, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    saved.append({'margin': margin, 'path': str(p.relative_to(OUT)),
                                  'shape': list(c.shape[:2])})
                if saved:
                    manifest.append({'case': case, 'scan': scan, 'split': split,
                                     'object_id': o['object_id'], 'truth': o['truth'],
                                     'bbox': o['bbox'], 'pixel': o['pixel'],
                                     'assoc_px': o['assoc_px'], 'source_image': str(img),
                                     'crops': saved})
    per_split = collections.Counter(r['split'] for r in manifest)
    per_class = collections.Counter(r['truth'] for r in manifest)
    per_split_class = collections.Counter((r['split'], r['truth']) for r in manifest)
    per_scene = collections.Counter(r['case'] for r in manifest)
    # 泄漏检查：同一 (case, object_id) 不得同时出现在两个 split
    leak = [k for k, v in collections.Counter((r['case'], r['object_id']) for r in manifest).items()
            if len({r['split'] for r in manifest if (r['case'], r['object_id']) == k}) > 1]
    summary = {'observations': len(manifest),
               'crops': sum(len(r['crops']) for r in manifest),
               'scenes': scenes, 'train_scenes': list(TRAIN_SCENES),
               'holdout_scenes': list(HOLDOUT_SCENES),
               'per_split': dict(per_split), 'per_class': dict(per_class),
               'per_scene': dict(per_scene),
               'per_split_class': {'%s/%s' % k: v for k, v in sorted(per_split_class.items())},
               'split_leakage': leak, 'skipped': dict(skipped),
               'margins': list(MARGINS), 'assoc_tol_px': ASSOC_TOL_PX,
               'provenance': {'source': 'R7 saved nine-grid conflict frames (_raw.jpg)',
                              'truth': 'case layout + make_scramble10.pixel_xy + scan proposals, unique match',
                              'note': 'labels are used for training/evaluation only; never by the runtime'}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print('written:', OUT / 'manifest.json')


if __name__ == '__main__':
    main()
