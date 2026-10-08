#!/usr/bin/env python3
"""Generate the 10 B03 "scramble" cases: default-scene neat attitudes + scrambled
positions + matching instructions.  Pure json/math -- runs anywhere.

Usage:  python make_scramble10.py <template_case> <default_poses.json> <out_dir> <register_out>
"""
import copy
import hashlib
import itertools
import json
import math
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
    {'id': 'scramble_08', 'cells': {'smoke_bomb': 'RB', 'smoke_bomb_01': 'MR', 'hand_grenade': 'RT',
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


# Image-space packing guard.  Two objects whose projected boxes nearly touch get
# merged by depth segmentation into one 'multiple_body_cores' region, which the
# tracker then marks ambiguous and drops -- the planner sees no candidate for that
# object at all.  Observed 10/4: a torch 8-11 px from the grenade was dropped, so
# "lower-right torch" became unsatisfiable and the round was rejected.  Projection
# calibrated from one observed frame (b03_03_10041656: five ground-truth object
# positions against their published candidate boxes):
U0, UX = 643.5, -569.3        # u = U0 + UX*x   (max residual 15.5 px)
V0, VY = 297.4, 629.6         # v = V0 + VY*y   (max residual  6.5 px)
MID_U, MID_V, MID_MARGIN = 644.0, 268.0, 25.0   # quadrant midlines + required clearance
MIN_PIXEL_GAP = 45.0          # official reference layouts measure 45.2 px; a pair below
                              # this is not safely separated in the image
QUADRANT_OF = {'LT': 'LT', 'ML': 'LT', 'RT': 'RT', 'MR': 'RT',
               'LB': 'LB', 'RB': 'RB', 'C': 'RB'}


def pixel_xy(x, y):
    return U0 + UX * x, V0 + VY * y


def x_for_u(u):
    return (u - U0) / UX


def y_for_v(v):
    return (v - V0) / VY


def quadrant_slots(quadrant, inner, inset):
    """Three slots per quadrant: the basket-corner slot, one that slides along the
    quadrant's own horizontal edge (toward the vertical midline) and one that slides
    along its vertical edge.  Two same-quadrant objects placed corner+edge separate by
    ~50-106 px, which is what the packing check needs.  Sliding *along an edge* rather
    than toward the centre matters: the centre-ward slots of two neighbouring quadrants
    sit only ~50 px apart (barely one box width), so a case that doubles two quadrants
    (scramble_06 does) cannot use them.  The edge slots stay clear of both midlines."""
    xmin, xmax, ymin, ymax = inner
    xL, xR = xmax - inset, xmin + inset            # image-left / image-right
    yT, yB = ymin + inset, ymax - inset            # image-top / image-bottom
    x_in_left = x_for_u(MID_U - MID_MARGIN)        # smallest x still left of midline
    x_in_right = x_for_u(MID_U + MID_MARGIN)       # largest x still right of midline
    y_in_top = y_for_v(MID_V - MID_MARGIN)         # largest y still above midline
    y_in_bottom = y_for_v(MID_V + MID_MARGIN)      # smallest y still below midline
    return {
        'LT': [(xL, yT), (x_in_left, yT), (xL, y_in_top)],
        'RT': [(xR, yT), (x_in_right, yT), (xR, y_in_top)],
        'LB': [(xL, yB), (x_in_left, yB), (xL, y_in_bottom)],
        'RB': [(xR, yB), (x_in_right, yB), (xR, y_in_bottom)],
    }[quadrant]


def predicted_box(point_xy, span_xy):
    u, v = pixel_xy(*point_xy)
    du, dv = abs(UX) * span_xy[0], abs(VY) * span_xy[1]
    return [u - du / 2, v - dv / 2, u + du / 2, v + dv / 2]


def box_gap(a, b):
    """Axis-aligned clearance: separation along either axis counts."""
    return max(max(a[0] - b[2], b[0] - a[2]), max(a[1] - b[3], b[1] - a[3]))


def min_pixel_gap(assignment, spans):
    ids = list(assignment)
    boxes = {i: predicted_box(assignment[i], spans[i]) for i in ids}
    worst = None
    for n, i in enumerate(ids):
        for j in ids[n + 1:]:
            gap = box_gap(boxes[i], boxes[j])
            if worst is None or gap < worst:
                worst = gap
    return worst


INSTRUCTION_CLASSES = {'烟雾弹': ('smoke_bomb', 'smoke_bomb_01'), '弹夹': ('Magazines',),
                       '军用手电筒': ('Flashlight',), '手雷': ('hand_grenade',)}
REGION_WORDS = {'左上方': '左上', '右上方': '右上', '左下方': '左下', '右下方': '右下'}
EXTREME_WORDS = ('最左方', '最右方')


def quadrant_word(xy):
    u, v = pixel_xy(*xy)
    return ('左' if u < MID_U else '右') + ('上' if v < MID_V else '下')


def assert_instructions_satisfiable(case_id, instructions, placed):
    """Every instruction must be satisfiable by exactly one object.

    The planner validates with implicit_unique: for a region word it filters the named
    class by region and then demands exactly one survivor (semantics.py: "implicit
    quantity requires one unambiguous matching object"); for a bare class word it demands
    exactly one instance in the whole scene.  scramble_08 shipped with BOTH smoke bombs in
    the lower-right quadrant, which made "抓取右下方的烟雾弹" unsatisfiable by construction
    -- a guaranteed rejection that never surfaced because rounds stopped earlier.  Extremes
    ('最左方') are satisfied by picking the class-relative extreme, so they only need the
    named class to exist."""
    where = {o['object_id']: quadrant_word(o['reference_xy']) for o in placed}
    for instruction in instructions:
        named = next((c for c in INSTRUCTION_CLASSES if c in instruction), None)
        if named is None:
            continue                                  # '抓取剩余的物品' names no class
        ids = [i for i in INSTRUCTION_CLASSES[named] if i in where]
        region = next((w for w in REGION_WORDS if w in instruction), None)
        if region is not None:
            inside = [i for i in ids if where[i] == REGION_WORDS[region]]
            assert len(inside) == 1, (case_id, instruction, 'region+class must be unique', inside)
        elif any(w in instruction for w in EXTREME_WORDS):
            assert ids, (case_id, instruction, 'named class missing')
        else:
            assert len(ids) == 1, (case_id, instruction, 'bare class must be unique', ids)


def main():
    template_path, poses_path, out_dir, register_path = sys.argv[1:5]
    template = json.load(open(template_path, encoding='utf-8'))
    poses = json.load(open(poses_path, encoding='utf-8'))['objects']
    inner = template['inner_xy']  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = inner

    cases, rows = [], []
    for n, spec in enumerate(SCENES, 1):
        case = copy.deepcopy(template)
        case['case_id'] = spec['id']
        case['case_index'] = n
        case['seed'] = 2026101100 + n
        case['planner_seed'] = 2026102100 + n
        case['instructions'] = spec['instructions']
        case['expected_sides'] = spec['sides']
        inset = template.get('wall_margin_m', 0.035) + 0.055
        # 姿态只在模板上加一次绕世界垂直轴的偏航；先算出投影跨度，再在"各自象限的
        # 两个极端槽位"里挑一组让图像内最小框间距最大——这直接决定深度分割会不会把
        # 两件物体并成一个 multiple_body_cores 区域（并了就会被判 ambiguous 并丢弃）。
        spans, slot_options = {}, []
        for obj in case['objects']:
            span = [obj['bbox_max'][k] - obj['bbox_min'][k] for k in range(3)]
            spans[obj['object_id']] = [span[1], span[0], span[2]] if span[1] > span[0] else span
            slot_options.append(quadrant_slots(QUADRANT_OF[spec['cells'][obj['object_id']]], inner, inset))
        best_score, chosen = None, None
        for combo in itertools.product(*slot_options):
            assignment = {obj['object_id']: combo[i] for i, obj in enumerate(case['objects'])}
            score = min_pixel_gap(assignment, spans)
            if best_score is None or score > best_score:
                best_score, chosen = score, assignment
        placed = []
        for obj in case['objects']:
            tx, ty = chosen[obj['object_id']]
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
            u, v = pixel_xy(*o['reference_xy'])
            assert abs(u - MID_U) >= MID_MARGIN and abs(v - MID_V) >= MID_MARGIN, (
                spec['id'], o['object_id'], 'midline', round(u), round(v))
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                gap = math.dist(a['reference_xy'], b['reference_xy']) - a['footprint_radius_m'] - b['footprint_radius_m']
                assert gap >= 0.005, (spec['id'], a['object_id'], b['object_id'], round(gap, 4))
        # 图像内最小框间距：这才是预测"深度分割会不会并核"的判据。官方参考布局量到
        # 45.2 px，所以低于 MIN_PIXEL_GAP 的场景一律拒绝生成。
        placed_boxes = {o['object_id']: predicted_box(o['reference_xy'],
                        [o['bbox_max'][k] - o['bbox_min'][k] for k in (0, 1)]) for o in placed}
        worst, worst_pair = None, None
        for n, a in enumerate(placed):
            for b in placed[n + 1:]:
                g = box_gap(placed_boxes[a['object_id']], placed_boxes[b['object_id']])
                if worst is None or g < worst:
                    worst, worst_pair = g, (a['object_id'], b['object_id'])
        assert worst >= MIN_PIXEL_GAP, (spec['id'], 'pixel gap', round(worst, 1), worst_pair)
        # 最后一道：每条指令都必须"唯一可满足"（否则计划校验必然拒单）
        assert_instructions_satisfiable(spec['id'], spec['instructions'], placed)
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
