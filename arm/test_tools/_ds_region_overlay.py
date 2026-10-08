#!/usr/bin/env python3
"""R2-0.3 temporal overlay strips for the persistent torch region (CPU only)."""
import json, pathlib, re
from PIL import Image, ImageDraw

RUNS = pathlib.Path("/root/gpufree-data/tcei_260920v2")
R01 = RUNS / "b03_base_10042140_scramble_03_round"
OUT = RUNS / "r2_overlay"
OUT.mkdir(exist_ok=True)

idx = [json.loads(l) for l in (R01 / "rgbd/index.jsonl").read_text(errors="ignore").splitlines() if l.strip()]
scalars = []
for line in (R01 / "scalars/events.jsonl").read_text(errors="ignore").splitlines():
    if '"unknown_regions"' not in line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    v = d.get("value") or {}
    st = d.get("last_received_simulation_time")
    w = d.get("received_wall")
    if not isinstance(st, (int, float)):
        continue
    scalars.append((st, w, v.get("unknown_regions") or []))
print("候选消息:", len(scalars))

strips = []
for tag, sim_t in (("t1_首扫后", 90), ("t2_中段", 220), ("t3_末态", 320)):
    bbox, wall = None, None
    for st, w, regions in scalars:
        if st > sim_t:
            continue
        for reg in regions:
            if reg.get("reason") in ("multiple_body_cores", "unresolved_correspondence"):
                b = reg.get("bbox")
                if isinstance(b, list) and len(b) == 4 and abs((b[0]+b[2])/2 - 794) < 45 and abs((b[1]+b[3])/2 - 351) < 45:
                    bbox, wall = b, w
    if bbox is None:
        print(tag, ": 该时刻无手电筒区域")
        continue
    best = min(idx, key=lambda r: abs((r.get("captured_at") or 0) - wall))
    jpg = R01 / "rgbd" / (best["stem"] + ".jpg")
    img = Image.open(jpg).convert("RGB")
    pad = 34
    box = (max(0, int(bbox[0]) - pad), max(0, int(bbox[1]) - pad),
           min(img.width, int(bbox[2]) + pad), min(img.height, int(bbox[3]) + pad))
    crop = img.crop(box)
    cw, ch = crop.size
    scale = max(2, 360 // max(cw, ch))
    crop = crop.resize((cw * scale, ch * scale), Image.BICUBIC)
    dr = ImageDraw.Draw(crop)
    dr.rectangle((pad * scale, pad * scale,
                  pad * scale + (bbox[2] - bbox[0]) * scale,
                  pad * scale + (bbox[3] - bbox[1]) * scale), outline=(255, 40, 40), width=2)
    dr.text((6, 4), tag.split("_")[0] + " " + best["stem"], fill=(20, 20, 20))
    p = OUT / ("手电筒区域_%s.png" % tag)
    crop.save(p)
    strips.append(crop)
    print(tag, "->", best["stem"], "bbox", [int(v) for v in bbox], "crop", crop.size)

if strips:
    h = max(c.height for c in strips)
    total = sum(c.width for c in strips) + 24 * (len(strips) - 1)
    sheet = Image.new("RGB", (total, h), (255, 255, 255))
    xx = 0
    for c in strips:
        sheet.paste(c, (xx, 0))
        xx += c.width + 24
    sheet.save(OUT / "手电筒区域时序条.png")
    print("时序条:", OUT / "手电筒区域时序条.png", sheet.size)
