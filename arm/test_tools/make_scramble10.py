#!/usr/bin/env python3
"""Generate the 10 B03 "scramble" cases: default-scene neat attitudes + scrambled
positions + matching instructions.  Pure json/math -- runs anywhere.

Usage:  python make_scramble10.py <template_case> <default_poses.json> <out_dir> <register_out>
"""
import copy
import hashlib
import json
import math
import random
import sys

SCENES = [
    {'id': 'scramble_01', 'cells': {'smoke_bomb': 'LT', 'smoke_bomb_01': 'C', 'hand_grenade': 'RT',
                                    'Flashlight': 'LB', 'Magazines': 'RB'},
     'instructions': ['抓取左上方的烟雾弹，放到左侧传送带', '抓取右下方的弹夹，放到右侧传送带',
                      '抓取最左方的军用手电筒，放到左侧传送带', '抓取手雷，放到右侧传送带',
                      '抓取剩余的物品，放到左侧传送带'],
     'sides': ['left', 'right', 'left', 'right', 'left']},
    {'id': 'scramble_02', 'cells': {'smoke_bomb': 'LT', 'smoke_bomb_01': 'RB', 'hand_grenade': 'C',
                                    'Flashlight': 'RT', 'Magazines': 'LB'},
     'instructions': ['抓取左上方的烟雾弹，放到左侧传送带', '抓取左下方的弹夹，放到右侧传送带',
                      '抓取右上方的军用手电筒，放到左侧传送带', '抓取手雷，放到右侧传送带',
                      '抓取剩余的物品，放到左侧传送带'],
     'sides': ['left', 'right', 'left', 'right', 'left']},
    {'id': 'scramble_03', 'cells': {'smoke_bomb': 'RT', 'smoke_bomb_01': 'LB', 'hand_grenade': 'C',
                                    'Flashlight': 'RB', 'Magazines': 'LT'},
     'instructions': ['抓取左上方的弹夹，放到右侧传送带', '抓取右下方的军用手电筒，放到左侧传送带',
                      '抓取右上方的烟雾弹，放到左侧传送带', '抓取手雷，放到右侧传送带',
                      '抓取剩余的物品，放到左侧传送带'],
     'sides': ['right', 'left', 'left', 'right', 'left']},
    {'id': 'scramble_04', 'cells': {'smoke_bomb': 'LB', 'smoke_bomb_01': 'C', 'hand_grenade': 'LT',
                                    'Flashlight': 'RB', 'Magazines': 'RT'},
     'instructions': ['抓取左上方的手雷，放到右侧传送带', '抓取右上方的弹夹，放到左侧传送带',
                      '抓取右下方的军用手电筒，放到左侧传送带', '抓取左下方的烟雾弹，放到右侧传送带',
                      '抓取剩余的物品，放到左侧传送带'],
     'sides': ['right', 'left', 'left', 'right', 'left']},
    {'id': 'scramble_05', 'cells': {'smoke_bomb': 'RT', 'smoke_bomb_01': 'LB', 'hand_grenade': 'C',
                                    'Flashlight': 'LT', 'Magazines': 'RB'},
     'instructions': ['抓取左上方的军用手电筒，放到左侧传送带', '抓取右下方的弹夹，放到右侧传送带',
                      '抓取右上方的烟雾弹，放到右侧传送带', '抓取手雷，放到左侧传送带',
                      '抓取剩余的物品，放到左侧传送带'],
     'sides': ['left', 'right', 'right', 'left', 'left']},
    {'id': 'scramble_06', 'cells': {'smoke_bomb': 'RT', 'smoke_bomb_01': 'RB', 'hand_grenade': 'LT',
                                    'Flashlight': 'ML', 'Magazines': 'C'},
     'instructions': ['抓取最左方的军用手电筒，放到左侧传送带', '抓取右上方的烟雾弹，放到右侧传送带',
                      '抓取手雷，放到左侧传送带', '抓取弹夹，放到右侧传送带',
                      '抓取剩余的物品，放到左侧传送带'],
     'sides': ['left', 'right', 'left', 'right', 'left']},
    {'id': 'scramble_07', 'cells': {'smoke_bomb': 'LT', 'smoke_bomb_01': 'RB', 'hand_grenade': 'C',
                                    'Flashlight': 'RT', 'Magazines': 'LB'},
     'instructions': ['抓取左上方的烟雾弹，放到右侧传送带', '抓取右下方的烟雾弹，放到左侧传送带',
                      '抓取右上方的军用手电筒，放到右侧传送带', '抓取弹夹，放到左侧传送带',
                      '抓取剩余的物品，放到右侧传送带'],
     'sides': ['right', 'left', 'right', 'left', 'right']},
    {'id': 'scramble_08', 'cells': {'smoke_bomb': 'RB', 'smoke_bomb_01': 'C', 'hand_grenade': 'RT',
                                    'Flashlight': 'LB', 'Magazines': 'LT'},
     'instructions': ['抓取右上方的手雷，放到左侧传送带', '抓取左上方的弹夹，放到右侧传送带',
                      '抓取左下方的军用手电筒，放到右侧传送带', '抓取右下方的烟雾弹，放到左侧传送带',
                      '抓取剩余的物品，放到右侧传送带'],
     'sides': ['left', 'right', 'right', 'left', 'right']},
    {'id': 'scramble_09', 'cells': {'smoke_bomb': 'LT', 'smoke_bomb_01': 'LB', 'hand_grenade': 'C',
                                    'Flashlight': 'MR', 'Magazines': 'RB'},
     'instructions': ['抓取最右方的军用手电筒，放到右侧传送带', '抓取左上方的烟雾弹，放到左侧传送带',
                      '抓取左下方的烟雾弹，放到右侧传送带', '抓取弹夹，放到左侧传送带',
                      '抓取剩余的物品，放到右侧传送带'],
     'sides': ['right', 'left', 'right', 'left', 'right']},
    {'id': 'scramble_10', 'cells': {'smoke_bomb': 'LT', 'smoke_bomb_01': 'C', 'hand_grenade': 'LB',
                                    'Flashlight': 'RB', 'Magazines': 'RT'},
     'instructions': ['抓取右上方的弹夹，放到左侧传送带', '抓取左下方的手雷，放到右侧传送带',
                      '抓取右下方的军用手电筒，放到右侧传送带', '抓取左上方的烟雾弹，放到左侧传送带',
                      '抓取剩余的物品，放到右侧传送带'],
     'sides': ['left', 'right', 'right', 'left', 'right']},
]

# Duplicate-object guard: RT2 is RT shifted inward so the flashlight does not share
# a cell with the smoke grenade in S2/S7 (both referenced there).

# v5 attitude rule: keep each object's template (lying, zero-tilt) attitude, and add
# a pure yaw about the WORLD vertical axis so its long axis points along world X.
# A vertical-axis yaw leaves the relation to gravity untouched (no tilt) and is
# exactly the perturbation the training generator randomized -> in-distribution.
Q_YAW_M90 = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)   # R_z(-90): world +Y -> +X


def qmul(a, b):
    """Hamilton product a (x) b -- b applied first, then a."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qrot(q, v):
    """Rotate vector v by quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    tx = 2 * (y * v[2] - z * v[1])
    ty = 2 * (z * v[0] - x * v[2])
    tz = 2 * (x * v[1] - y * v[0])
    return [v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx)]


def qnorm(q):
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def cell_xy(cell, inner, ins):
    """Deep-quadrant spots (v2): 4 corners + 4 quadrant-inner slots.
    Nothing sits on or near the quadrant midlines (image x=644/y=268), which
    caused 'boundary uncertainty' rejections in v1.  Image-left = +x, top = -y."""
    xmin, xmax, ymin, ymax = inner
    mx, my = (xmin + xmax) / 2, (ymin + ymax) / 2
    xL, xR = xmax - ins, xmin + ins          # image-left / image-right
    yT, yB = ymin + ins, ymax - ins          # image-top / image-bottom
    cix, ciy = (xL + mx) / 2, (yT + my) / 2  # quadrant-inner (toward LT)
    table = {
        'LT': (xL, yT), 'RT': (xR, yT), 'LB': (xL, yB), 'RB': (xR, yB),
        'ML': (cix, ciy), 'MR': (xR + (mx - xR) * 0.45, (yT + my) / 2),
        'C': (xR + (mx - xR) * 0.45, yB + (my - yB) * 0.45),
        'TC': (mx, yT), 'BC': (mx, yB),
    }
    return table[cell]


def main():
    template_path, poses_path, out_dir, register_path = sys.argv[1:5]
    template = json.load(open(template_path, encoding='utf-8'))
    poses = json.load(open(poses_path, encoding='utf-8'))['objects']
    inner = template['inner_xy']  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = inner

    cases, rows = [], []
    for n, spec in enumerate(SCENES, 1):
        rng = random.Random(2026101100 + n)
        case = copy.deepcopy(template)
        case['case_id'] = spec['id']
        case['case_index'] = n
        case['seed'] = 2026101100 + n
        case['planner_seed'] = 2026102100 + n
        case['instructions'] = spec['instructions']
        case['expected_sides'] = spec['sides']
        placed = []
        for obj in case['objects']:
            cell = spec['cells'][obj['object_id']]
            tx, ty = cell_xy(cell, inner, template.get('wall_margin_m', 0.035) + 0.055)
            # v5: keep template attitude; if the long axis lies along world Y, add a
            # pure vertical-axis yaw so it points along X (horizontal in the
            # observation camera).  Zero tilt preserved; yaw is in-distribution.
            q_old = qnorm(tuple(obj['quaternion_wxyz']))
            span_old = [obj['bbox_max'][k] - obj['bbox_min'][k] for k in range(3)]
            bc_old = [(obj['bbox_min'][k] + obj['bbox_max'][k]) / 2 for k in range(3)]
            # prim origin offset in the object's local frame (invariant under yaw)
            q_inv = (q_old[0], -q_old[1], -q_old[2], -q_old[3])
            off_local = qrot(q_inv, [bc_old[k] - obj['position'][k] for k in range(3)])
            if span_old[1] > span_old[0]:          # long axis along Y -> swing to X
                q_new = qnorm(qmul(Q_YAW_M90, q_old))
                span_new = [span_old[1], span_old[0], span_old[2]]
            else:                                   # already along X
                q_new = q_old
                span_new = list(span_old)
            hx, hy = span_new[0] / 2, span_new[1] / 2
            tx = min(max(tx, xmin + hx + 0.006), xmax - hx - 0.006)
            ty = min(max(ty, ymin + hy + 0.006), ymax - hy - 0.006)
            bc_new = [tx, ty, obj['bbox_min'][2] + span_new[2] / 2]
            pos_new = [bc_new[k] - qrot(q_new, off_local)[k] for k in range(3)]
            obj['position'] = pos_new
            obj['requested_position'] = list(pos_new)
            obj['quaternion_wxyz'] = list(q_new)
            obj['requested_quaternion_wxyz'] = list(q_new)
            obj['bbox_min'] = [bc_new[k] - span_new[k] / 2 for k in range(3)]
            obj['bbox_max'] = [bc_new[k] + span_new[k] / 2 for k in range(3)]
            obj['reference_xy'] = [bc_new[0], bc_new[1]]
            obj['footprint_radius_m'] = 0.5 * math.hypot(span_new[0], span_new[1])
            placed.append(obj)
        # self-checks
        xmin, xmax, ymin, ymax = inner
        for o in placed:
            assert xmin - 0.005 <= o['bbox_min'][0] and o['bbox_max'][0] <= xmax + 0.005, (spec['id'], o['object_id'], 'x')
            assert ymin - 0.005 <= o['bbox_min'][1] and o['bbox_max'][1] <= ymax + 0.005, (spec['id'], o['object_id'], 'y')
            assert 2.34 < o['bbox_min'][2] < 2.43 and o['bbox_max'][2] < 2.55, (spec['id'], o['object_id'], 'z')
            # anti-midline: no object may sit near a quadrant boundary in image space
            # (empirical projection fits: u = 643 - 653*x, v = 280 + 592*y)
            u = 643 - 653 * o['reference_xy'][0]
            v = 280 + 592 * o['reference_xy'][1]
            assert abs(u - 644) >= 25 and abs(v - 268) >= 25, (spec['id'], o['object_id'], 'midline', round(u), round(v))
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                gap = math.dist(a['reference_xy'], b['reference_xy']) - a['footprint_radius_m'] - b['footprint_radius_m']
                assert gap >= 0.005, (spec['id'], a['object_id'], b['object_id'], round(gap, 4))
        body = json.dumps(case, ensure_ascii=False, indent=1) + '\n'
        out_path = '%s/%s.json' % (out_dir, spec['id'])
        open(out_path, 'w', encoding='utf-8').write(body)
        sha = hashlib.sha256(open(out_path, 'rb').read()).hexdigest()
        rows.append({'case_id': spec['id'], 'seed': case['seed'], 'planner_seed': case['planner_seed'],
                     'sha256': sha, 'case_file': 'cases_scramble10/%s.json' % spec['id']})
        print(spec['id'], 'ok')
        for o in placed:
            print('   %-14s ref_xy=%s fp=%.3f' % (o['object_id'], [round(v, 3) for v in o['reference_xy']], o['footprint_radius_m']))
    cases = None
    register = {'batch': 'B03_scramble10_default_pose_v1', 'status': 'registered_not_executed', 'count': len(rows),
                'note': 'default-scene neat attitudes (scene.usd authored) + scrambled positions + per-scene instructions; see cases_scramble10/default_poses.json',
                'cases': rows}
    open(register_path, 'w', encoding='utf-8').write(json.dumps(register, ensure_ascii=False, indent=1) + '\n')
    print('REGISTER', register_path, len(rows), 'cases')


if __name__ == '__main__':
    main()
