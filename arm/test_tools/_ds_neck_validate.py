#!/usr/bin/env python3
"""Replicate the runtime mask pipeline on pair_0401 and evaluate the neck criterion."""
import json, pathlib, re
import numpy as np, cv2

RUNS = pathlib.Path("/root/gpufree-data/tcei_260920v2")
R3 = RUNS / "b03_base_10042140_scramble_03_round"

def decode_chain(target, files):
    def is_kf(p):
        with np.load(p, allow_pickle=False) as a:
            return "depth" in a.files
    kf = max(k for k in files if k <= target and is_kf(files[k]))
    with np.load(files[kf], allow_pickle=False) as a:
        cur = a["depth"].copy()
    n = kf
    while n < target:
        n += 1
        if n not in files:
            continue
        with np.load(files[n], allow_pickle=False) as a:
            if "depth_xor" in a.files:
                cur = np.bitwise_xor(cur.view(np.uint32), a["depth_xor"]).view(np.float32)
    return cur

files = {}
for p in R3.glob("rgbd/pair_*.npz"):
    m = re.search(r"pair_(\d+)", p.name)
    if m:
        files[int(m.group(1))] = p

K = [1343.83203125, 0.0, 640.0, 0.0, 1343.83203125, 360.0, 0.0, 0.0, 1.0]
depth = decode_chain(401, files)
k = np.array(K).reshape(3, 3)

# --- runtime 管线复刻（rotation_perception.source_components 40-130 行）---
inside = ((depth > 1.9) & (depth < 2.6)).astype(np.uint8)  # 筐深度带（观测位约 2.1-2.3）
# 保守起见直接复刻：inside = 筐 ROI 内的深度带 —— 运行时用标定 ROI，这里用
# 已知 basket 区域（全部物体都在其中）近似：x 350..950, y 120..430
roi = np.zeros_like(inside, dtype=bool)
roi[120:430, 350:950] = True
inside = (inside.astype(bool) & roi).astype(np.uint8)
values = depth[inside.astype(bool)]
hist, edges = np.histogram(values, bins=np.arange(1.95, 2.251, .001))
index = int(hist.argmax())
floor_samples = values[(values >= edges[index]) & (values < edges[index + 1])]
floor = float(np.median(floor_samples))
print("floor=%.3f (帧 pair_0401)" % floor)
mask = (inside.astype(bool) & (floor - depth > .004)).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
for i in range(1, count):
    x, y, bw, bh, area = map(int, stats[i])
    if area < 9:
        continue
    body = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8)
    heights = floor - depth[y:y+bh, x:x+bw][body.astype(bool)]
    dist = cv2.distanceTransform(np.pad(body, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    radius = float(dist.max())
    core = dist >= .70 * radius
    cc, _, cstats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    comps = [(int(s[0]), int(s[1]), int(s[4])) for s in cstats[1:] if s[4] >= 9]
    substantial = len(comps)
    if abs(x - 762) < 12 and abs(y - 344) < 12:
        print("★ 手电筒 blob: bbox=(%d,%d,%d,%d) area=%d radius=%.1f substantial_cores=%d comps=%s"
              % (x, y, bw, bh, area, radius, substantial, comps))
        # --- 颈部判据数值评估 ---
        if substantial >= 2:
            # 两个最大核心的质心连线上的 body 宽度剖面
            order = sorted(comps, key=lambda c: -c[2])[:2]
            n2, l2, s2, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
            big = [j for j in range(1, n2) if s2[j, 4] >= 9]
            big.sort(key=lambda j: -s2[j, 4])
            cents = []
            for j in big[:2]:
                ys, xs = np.nonzero(l2 == j)
                cents.append((float(xs.mean()), float(ys.mean())))
            if len(cents) == 2:
                (x1, y1), (x2, y2) = cents
                horizontal = bw >= bh
                steps = int(round(x2 - x1)) if horizontal else int(round(y2 - y1))
                widths = []
                for s in range(max(1, abs(steps) + 1)):
                    if horizontal:
                        px = int(round(x1 + (x2 - x1) * s / max(1, abs(steps))))
                        widths.append(int(body[:, px].sum()))
                    else:
                        py = int(round(y1 + (y2 - y1) * s / max(1, abs(steps))))
                        widths.append(int(body[py, :].sum()))
                print("  两核质心:", [(round(a), round(b)) for a, b in cents],
                      "| 连线 body 宽度剖面:", widths, "| 最窄颈部:", min(widths) if widths else None)
