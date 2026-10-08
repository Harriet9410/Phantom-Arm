"""Measure image-space separation between objects in every registered scene.

Calibrates the world->pixel mapping from an observed frame instead of trusting the
generator's rough fit, then reports, per case, the smallest axis-aligned box gap
between any two objects.  Gap is what drives depth segmentation merging two bodies
into one 'multiple_body_cores' region (observed: flashlight/grenade pair at 8-11 px
-> flagged ambiguous -> dropped -> planner had no Torch candidate at all).

Run: python local/1003/v1/_measure_pixel_gap.py
"""
import json
import math
import sys
from pathlib import Path

ARM = Path(__file__).resolve().parents[3] / 'arm'

# Observed in b03_03_10041656 (read_only_ready.json candidate boxes, matched to the
# case's ground-truth object positions).  Used to calibrate the projection.
OBSERVED = [
    ((0.257, -0.180), (494.5, 182.5), 'Magazines'),
    ((0.257,  0.086), (495.5, 349.0), 'smoke_bomb_01'),
    ((-0.122, 0.028), (728.5, 321.5), 'hand_grenade'),
    ((-0.271, -0.178), (794.0, 185.5), 'smoke_bomb'),
    ((-0.277, 0.089), (794.0, 351.0), 'Flashlight'),
]


def least_squares(pairs):
    """pairs: [(x, u)] -> (a, b) for u = a + b*x, plus max residual."""
    n = len(pairs)
    sx = sum(p[0] for p in pairs); su = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs); sxu = sum(p[0] * p[1] for p in pairs)
    b = (n * sxu - sx * su) / (n * sxx - sx * sx)
    a = (su - b * sx) / n
    res = max(abs(a + b * x - u) for x, u in pairs)
    return a, b, res


ua, ub, ures = least_squares([(w[0], p[0]) for w, p, _ in OBSERVED])
va, vb, vres = least_squares([(w[1], p[1]) for w, p, _ in OBSERVED])
print('calibrated: u = %.1f %+.1f*x  (max residual %.1f px)' % (ua, ub, ures))
print('calibrated: v = %.1f %+.1f*y  (max residual %.1f px)' % (va, vb, vres))
print('generator fit in make_scramble10.py: u = 643 - 653x , v = 280 + 592y')
print()

U_SPAN, V_SPAN = abs(ub), abs(vb)


def boxes(case):
    out = []
    for obj in case['objects']:
        # The pixel box is about the object's bbox centre; 'position' is the prim
        # origin, offset from that centre by up to half the object (a ~28 px error on
        # a 0.1 m part).  Use reference_xy when the case carries it.
        x, y = obj['reference_xy'] if 'reference_xy' in obj else obj['position'][:2]
        sx = obj['bbox_max'][0] - obj['bbox_min'][0]
        sy = obj['bbox_max'][1] - obj['bbox_min'][1]
        u, v = ua + ub * x, va + vb * y
        du, dv = U_SPAN * sx, V_SPAN * sy
        out.append({'name': obj['path'].split('/')[-1],
                    'box': [u - du / 2, v - dv / 2, u + du / 2, v + dv / 2],
                    'ref_xy': (x, y)})
    return out


def gap(a, b):
    """Axis-aligned clearance: separated along an axis => that axis gap counts."""
    gx = max(a[0] - b[2], b[0] - a[2])
    gy = max(a[1] - b[3], b[1] - a[3])
    return max(gx, gy)


def report(label, path):
    case = json.loads(path.read_text(encoding='utf-8'))
    bs = boxes(case)
    worst, pair = None, None
    for i, a in enumerate(bs):
        for b in bs[i + 1:]:
            g = gap(a['box'], b['box'])
            if worst is None or g < worst:
                worst, pair = g, (a['name'], b['name'])
    flag = 'VIOLATION' if worst < 45 else 'ok'
    print('%-34s min gap %6.1f px  %-38s %s'
          % (label, worst, '%s <-> %s' % pair, flag))
    return worst, pair


print('=== B03 (10 scramble cases) ===')
b03_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ARM / 'cases_scramble10')
results = []
for i in range(1, 11):
    p = b03_dir / ('scramble_%02d.json' % i)
    results.append((report('scramble_%02d' % i, p), p))

print()
print('=== official 30 (reference layouts) ===')
off = sorted((ARM / 'cases_official30_v2').glob('*.json'))
offs = [(report(f.stem, f), f) for f in off]

print()
print('=== B01 (30 cases) ===')
b01 = sorted((ARM / 'cases_b01').glob('*.json'))
b1s = [(report(f.stem, f), f) for f in b01]

for label, rows in (('B03', results), ('official30', offs), ('b01', b1s)):
    bad = [r for r in rows if r[0][0] < 45]
    print()
    print('%s: %d/%d cases below 45 px' % (label, len(bad), len(rows)))
    for (w, pair), path in bad:
        print('   %-24s %6.1f px  %s <-> %s' % (path.stem, w, pair[0], pair[1]))
