#!/usr/bin/env python3
"""Depth forensics for the multiple_body_cores torch region (R2-2 evidence)."""
import json, pathlib, glob, re
import numpy as np, cv2

RUNS = pathlib.Path("/root/gpufree-data/tcei_260920v2")
R3 = RUNS / "b03_base_10042140_scramble_03_round"
files = {int(re.search(r"pair_(\d+)", p).group(1)): p for p in glob.glob(str(R3 / "rgbd/pair_*.npz"))}

with np.load(files[257], allow_pickle=False) as a:
    print("257 keys:", a.files)
    cur = a["depth"].copy()
print("257 shape:", cur.shape, "size:", cur.size)
for n in (258, 259, 260):
    if n not in files:
        print(n, "missing"); continue
    try:
        with np.load(files[n], allow_pickle=False) as a:
            print(n, "keys:", a.files)
            if "depth_xor" in a.files:
                dx = a["depth_xor"]
                cur = np.bitwise_xor(cur.view(np.uint32), dx).view(np.float32)
                print(n, "depth min/max: %.3f/%.3f" % (np.nanmin(cur), np.nanmax(cur)))
    except Exception as e:
        print(n, "FAIL:", str(e)[:140])

blob = cur[344:358, 762:826]
valid = blob[(blob > 1.5) & (blob < 4.0)]
print("手电筒 bbox valid px:", len(valid))
if len(valid):
    floor = float(np.median(valid))
    print("floor=%.3f" % floor)
    inside = ((blob > 1.5) & (blob < 4.0) & (floor - blob > .004)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inside, 8)
    print("连通体:", [int(stats[i, 4]) for i in range(1, count) if stats[i, 4] >= 9])
    for i in range(1, count):
        if stats[i, 4] < 60:
            continue
        body = (labels == i).astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(body, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        radius = float(dist.max())
        core = dist >= 0.70 * radius
        cc, _, cstats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
        areas = sorted((int(s[4]) for s in cstats[1:]), reverse=True)
        ratio = round(areas[1] / areas[0], 2) if len(areas) > 1 else None
        print("  blob=%d radius=%.1f core_areas=%s ratio2/1=%s" % (int(stats[i, 4]), radius, areas[:4], ratio))
