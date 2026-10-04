#!/usr/bin/env python3
"""Read-only dump of the default scene's authored object poses (the official neat placement).

Run on the instance with the Isaac python (pxr only, no Kit/GUI):
  /root/isaacsim/python.sh test_tools/dump_default_poses.py [out.json]

Writes default_poses.json: per prim, the authored world transform (position +
quaternion wxyz) and the world AABB at that pose.  These are the "official
neat" attitudes B03 cases must reuse -- not the lying poses of B01/B02 cases.
Never modifies scene.usd.
"""
import json
import sys

from pxr import Usd, UsdGeom

SCENE = '/root/EAICON/Content/JAKA/scene.usd'
PRIMS = {
    '/World/smoke_bomb': 'Smokegrenade',
    '/World/smoke_bomb_01': 'Smokegrenade',
    '/World/hand_grenade': 'Grenade',
    '/World/Magazines': 'Magazine',
    '/World/Flashlight': 'Torch',
}

stage = Usd.Stage.Open(SCENE)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])

raw = {}
for path in PRIMS:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise SystemExit('prim missing: ' + path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    raw[path] = {'t': list(t), 'q': [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]],
                 'lo': list(lo), 'hi': list(hi)}

# Normalize units: the composed transform may be in stage units (cm) -- rescale so
# the objects sit at the known basket height (~2.38-2.42 m).
z0 = raw['/World/smoke_bomb']['t'][2]
if 2.3 < abs(z0) < 2.5:
    scale = 1.0
elif 230.0 < abs(z0) < 250.0:
    scale = 0.01
else:
    raise SystemExit('unexpected basket height, unit unknown: z=%r' % z0)

out = {'scene': SCENE, 'scale_applied': scale, 'objects': {}}
for path, o in raw.items():
    out['objects'][path] = {
        'category': PRIMS[path],
        'position': [v * scale for v in o['t']],
        'quaternion_wxyz': o['q'],
        'bbox_min': [v * scale for v in o['lo']],
        'bbox_max': [v * scale for v in o['hi']],
    }

outpath = sys.argv[1] if len(sys.argv) > 1 else '/root/tcei_final_v2_23/tcei_260920v2/test_tools/default_poses.json'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('WROTE', outpath, 'scale=', scale)
for p, o in out['objects'].items():
    print('%-24s %-12s pos=%s zspan=[%.3f,%.3f]' % (
        p, o['category'], [round(v, 4) for v in o['position']], o['bbox_min'][2], o['bbox_max'][2]))
