"""Measured alternate grasp points and bounded per-object failure memory.

Development thresholds below are not competition rules or validated recovery
performance. This module never moves an object, changes identity, chooses a new
semantic target or reads simulator state. The original candidate is kept apart
from a pose-only copy: the latter must never be fed back to object tracking.
"""
import copy
import math
from numbers import Real


DEFAULTS = {
    'offsets_m': [0., .005, -.005, .010, -.010],
    'max_candidates': 5,
    'core_margin_m': .0015,
    'nearest_along_tolerance_m': .003,
    'cross_axis_tolerance_m': .002,
    'max_depth_difference_m': .008,
    'local_depth_spread_m': .004,
    'local_support_radius_m': .006,
    'local_support_min_points': 3,
    'local_support_min_span_m': .004,
    'min_core_area_m2': .000012,
    'minimum_axis_ratio': 1.20,
    'round_max_axis_ratio': 1.20,
    'round_max_core_radial_ratio': 1.70,
    'round_offset_m': .005,
    'distinct_point_m': .002,
}
FAILED_OUTCOMES = {'planning_rejected', 'empty_grasp', 'no_effective_lift', 'holding_feedback_lost', 'true_drop'}
OUTCOMES = FAILED_OUTCOMES | {'success', 'system_aborted','released_pending_verification'}


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, count):
    try:
        result = list(value)
    except TypeError:
        raise ValueError('missing_geometry_vector')
    if len(result) != count or not all(_finite(v) for v in result):
        raise ValueError('invalid_geometry_vector')
    return [float(v) for v in result]


def _rotation(value):
    try:
        rows = [_vector(row, 3) for row in value]
    except TypeError:
        raise ValueError('camera_to_world_rotation_required')
    if len(rows) != 3:
        raise ValueError('camera_to_world_rotation_required')
    for i in range(3):
        for j in range(3):
            if abs(sum(rows[i][k] * rows[j][k] for k in range(3)) - (1. if i == j else 0.)) > .002:
                raise ValueError('camera_to_world_not_a_rotation')
    determinant = sum(rows[0][i] * (rows[1][(i + 1) % 3] * rows[2][(i + 2) % 3]
                                    - rows[1][(i + 2) % 3] * rows[2][(i + 1) % 3]) for i in range(3))
    if abs(determinant - 1.) > .002:
        raise ValueError('camera_to_world_not_a_proper_rotation')
    return rows


def _intrinsics(candidate):
    meta = candidate.get('surface_reference_meta', {})
    k = _vector(meta.get('intrinsics'), 9)
    if min(k[0], k[4]) <= 0 or abs(k[8] - 1.) > 1e-8 or max(abs(k[i]) for i in (1, 3, 6, 7)) > 1e-8:
        raise ValueError('unsupported_camera_intrinsics')
    if meta.get('coordinate_system', 'optical_pixels_and_depth_m') != 'optical_pixels_and_depth_m':
        raise ValueError('surface_reference_coordinate_system_mismatch')
    if meta.get('source', 'current_rgbd_body_surface') != 'current_rgbd_body_surface':
        raise ValueError('surface_reference_not_camera_measurement')
    return k


def _points(value, minimum):
    if not isinstance(value, (list, tuple)):
        raise ValueError('missing_measured_surface')
    points = {}
    for point in value:
        try:
            u, v, z = _vector(point, 3)
        except (TypeError, ValueError):
            continue
        if not .1 < z < 5.:
            continue
        key = (round(u, 5), round(v, 5))
        if key in points and abs(points[key][2] - z) > .001:
            raise ValueError('conflicting_depth_at_same_pixel')
        points[key] = [u, v, z]
    if len(points) < minimum:
        raise ValueError('insufficient_unique_measured_surface')
    return list(points.values())


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull(points):
    ordered = sorted(set(tuple(point) for point in points))
    if len(ordered) < 3:
        raise ValueError('body_core_is_degenerate')
    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    result = lower[:-1] + upper[:-1]
    if len(result) < 3:
        raise ValueError('body_core_is_degenerate')
    return result


def _hull_area(hull):
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(hull, hull[1:] + hull[:1]))) / 2.


def _inside_margin(point, hull, margin):
    return all(_cross(a, b, point) / math.dist(a, b) >= margin - 1e-12
               for a, b in zip(hull, hull[1:] + hull[:1]))


def _optical(point, k):
    u, v, z = point
    return [(u - k[2]) * z / k[0], (v - k[5]) * z / k[4], z]


def _settings(config):
    result = copy.deepcopy(DEFAULTS)
    if config:
        if set(config) - set(result):
            raise ValueError('unsupported_grasp_candidate_setting')
        result.update(copy.deepcopy(config))
    offsets = result['offsets_m']
    if (not isinstance(offsets, (list, tuple)) or not offsets or offsets[0] != 0.
            or len(offsets) > 5 or any(not _finite(x) or abs(x) > .010 for x in offsets)
            or len(set(offsets)) != len(offsets)):
        raise ValueError('offsets_m_must_be_bounded_center_first')
    if type(result['max_candidates']) is not int or not 1 <= result['max_candidates'] <= 5:
        raise ValueError('max_candidates_out_of_scope')
    for key, value in result.items():
        if key not in ('offsets_m', 'max_candidates') and (not _finite(value) or value <= 0):
            raise ValueError('invalid_grasp_candidate_threshold:' + key)
    if type(result['local_support_min_points']) is not int or result['local_support_min_points'] < 3:
        raise ValueError('invalid_local_support_count')
    if (result['round_max_axis_ratio'] < 1. or result['round_max_axis_ratio'] > result['minimum_axis_ratio']
            or result['round_max_core_radial_ratio'] < 1. or result['round_offset_m'] > .005):
        raise ValueError('round_body_settings_out_of_bounded_scope')
    return result


def generate_grasp_candidates(object_candidate, camera_to_world, config=None):
    """Return centre then up to four surface-measured pose-only alternatives.

    camera_to_world is the proper 3x3 optical-to-world rotation matching
    object_candidate.world_position. Translation is anchored at that measured
    centre: alternative_world = centre_world + R*(sample_optical-centre_optical).
    No surface point is interpolated or invented. A missing/ambiguous geometry
    returns only the unmodified legacy centre and a fallback reason; that return
    does not certify the centre for motion or bypass normal complete-path guards.
    """
    cfg = _settings(config)
    original = copy.deepcopy(object_candidate)
    centre = {'candidate_id': 'centre', 'pose_candidate': copy.deepcopy(original),
              'nominal_offset_m': 0., 'actual_offset_m': 0., 'source': 'original_measured_centre',
              'geometry_verified': False, 'reason': 'original_centre_requires_normal_path_and_grasp_guards'}
    result = {'object_candidate': original, 'candidates': [centre], 'rejected': [],
              'fallback_reason': None, 'thresholds': cfg, 'threshold_status': 'development_initial_values'}
    try:
        if not isinstance(original.get('stable_id'), str) or not original['stable_id']:
            raise ValueError('stable_identity_required_for_alternate_grasps')
        if original.get('grasp_ready') is False:
            raise ValueError('object_geometry_not_grasp_ready')
        pixel = _vector(original.get('pixel'), 2)
        depth = original.get('depth')
        if not _finite(depth) or not .1 < depth < 5.:
            raise ValueError('invalid_centre_depth')
        world = _vector(original.get('world_position'), 3)
        rotation = _rotation(camera_to_world)
        k = _intrinsics(original)
        core = _points(original.get('depth_reference'), 6)
        samples = _points(original.get('surface_reference'), 8)
        axis = original.get('body_axis_deg')
        axis_source = 'measured_body_axis'
        if not _finite(axis):
            angle = original.get('angle_deg')
            if not _finite(angle):
                raise ValueError('body_axis_missing')
            axis = angle + 90.
            axis_source = 'body_axis_from_calibrated_grasp_angle'
        ratio = original.get('axis_ratio')
        if ratio is not None and (not _finite(ratio) or ratio < 1.):
            raise ValueError('body_axis_ratio_invalid')
        round_body = ratio is not None and ratio <= cfg['round_max_axis_ratio']
        if ratio is not None and ratio < cfg['minimum_axis_ratio'] and not round_body:
            raise ValueError('body_long_axis_ambiguous')
        if round_body and not _finite(original.get('angle_deg')):
            raise ValueError('round_body_wrist_orientation_missing')
        # Constant-centre-depth metric coordinates preserve core image topology;
        # actual optical depth is used separately for the returned 3-D point.
        def plane(point):
            return [(point[0] - pixel[0]) * depth / k[0], (point[1] - pixel[1]) * depth / k[4]]
        if any(abs(point[2] - depth) > cfg['max_depth_difference_m'] for point in core):
            raise ValueError('body_core_depth_inconsistent_with_centre')
        hull = _hull([plane(point) for point in core])
        if _hull_area(hull) < cfg['min_core_area_m2']:
            raise ValueError('body_core_area_too_small')
        if not _inside_margin([0., 0.], hull, cfg['core_margin_m']):
            raise ValueError('centre_not_inside_measured_core_with_margin')
        if round_body:
            # A near-one measured axis ratio is necessary but not sufficient:
            # verify a two-dimensional, reasonably isotropic core around the
            # original centre. Category names never determine this branch.
            inner = min(_cross(a, b, [0., 0.]) / math.dist(a, b)
                        for a, b in zip(hull, hull[1:] + hull[:1]))
            outer = max(math.hypot(*point) for point in hull)
            if outer / inner > cfg['round_max_core_radial_ratio']:
                raise ValueError('round_body_core_not_sufficiently_isotropic')
            if inner < cfg['round_offset_m'] + cfg['core_margin_m']:
                raise ValueError('round_body_core_too_small_for_offsets')
            axis_source = 'explicit_camera_plane_axes_for_measured_round_body'
            specifications = [('camera_x', sign * cfg['round_offset_m'], [1., 0.]) for sign in (1, -1)]
            specifications += [('camera_y', sign * cfg['round_offset_m'], [0., 1.]) for sign in (1, -1)]
            result['shape_evidence'] = {'mode': 'measured_round_body', 'axis_ratio': ratio,
                                        'core_outer_inner_ratio': outer / inner}
        else:
            rad = math.radians(axis)
            direction = [math.cos(rad) * depth / k[0], math.sin(rad) * depth / k[4]]
            norm = math.hypot(*direction)
            direction = [x / norm for x in direction]
            specifications = [('body_axis', offset, direction) for offset in cfg['offsets_m'][1:]]
            result['shape_evidence'] = {'mode': 'measured_long_axis', 'axis_ratio': ratio}
        basis = _optical([pixel[0], pixel[1], depth], k)
        prepared = []
        for sample in samples:
            xy = plane(sample)
            if not _inside_margin(xy, hull, cfg['core_margin_m']):
                continue
            if abs(sample[2] - depth) > cfg['max_depth_difference_m']:
                continue
            prepared.append((sample, xy))
        if not prepared:
            raise ValueError('no_surface_samples_inside_body_core')
        centre.update(geometry_verified=True, axis_source=axis_source,
                      reason='centre_inside_measured_core_full_path_still_required')
        chosen_positions = [world]
        for direction_name, offset, direction in specifications:
            if len(result['candidates']) >= cfg['max_candidates']:
                break
            transverse = [-direction[1], direction[0]]
            possible = []
            reject_reasons = set()
            for sample, xy in prepared:
                along = sum(x * d for x, d in zip(xy, direction))
                cross = sum(x * d for x, d in zip(xy, transverse))
                if abs(along - offset) > cfg['nearest_along_tolerance_m'] or abs(cross) > cfg['cross_axis_tolerance_m']:
                    continue
                neighbours = [(other, q) for other, q in prepared if math.dist(xy, q) <= cfg['local_support_radius_m']]
                if len(neighbours) < cfg['local_support_min_points']:
                    reject_reasons.add('insufficient_local_measured_support'); continue
                if max(p[2] for p, _ in neighbours) - min(p[2] for p, _ in neighbours) > cfg['local_depth_spread_m']:
                    reject_reasons.add('local_depth_discontinuity'); continue
                projections = [[sum(q[i] * d[i] for i in (0, 1)) for _, q in neighbours]
                               for d in (direction, transverse)]
                if any(max(values) - min(values) < cfg['local_support_min_span_m'] for values in projections):
                    reject_reasons.add('thin_or_one_dimensional_surface_support'); continue
                try:
                    local_hull = _hull([q for _, q in neighbours])
                except ValueError:
                    reject_reasons.add('degenerate_local_surface_support'); continue
                if _hull_area(local_hull) < cfg['local_support_min_span_m'] ** 2 / 4.:
                    reject_reasons.add('degenerate_local_surface_support'); continue
                optical = _optical(sample, k)
                delta = [a - b for a, b in zip(optical, basis)]
                position = [world[i] + sum(rotation[i][j] * delta[j] for j in range(3)) for i in range(3)]
                if any(math.dist(position, previous) < cfg['distinct_point_m'] for previous in chosen_positions):
                    reject_reasons.add('duplicate_or_near_centre_point'); continue
                score = abs(along - offset) + abs(cross) + .25 * abs(sample[2] - depth)
                possible.append((score, sample[0], sample[1], sample, position, along, cross))
            if not possible:
                result['rejected'].append({'nominal_offset_m': offset, 'offset_direction': direction_name,
                    'reason': ','.join(sorted(reject_reasons)) or 'no_measured_point_near_requested_core_offset'})
                continue
            _, _, _, sample, position, along, cross = min(possible)
            pose = copy.deepcopy(original)
            source = 'measured_round_body_offset' if round_body else 'surface_reference_measured_point'
            pose.update(pixel=list(sample[:2]), depth=sample[2], world_position=position,
                        grasp_point_method='measured_round_body_offset' if round_body else 'measured_core_surface_alternative')
            # position is a legacy optical-to-robot calibration, not the world
            # coordinate used by calibrated_grasp; do not silently rebase it.
            pose['pose_only_candidate'] = True
            pose['source_object_pixel'] = list(pixel)
            item = {'candidate_id': ('round_' + direction_name + '_%+dmm' if round_body else 'surface_%+dmm') % round(offset * 1000),
                    'pose_candidate': pose, 'nominal_offset_m': offset,
                    'actual_offset_m': math.dist(world, position), 'actual_along_m': along,
                    'actual_cross_m': cross, 'source': source, 'offset_direction': direction_name,
                    'geometry_verified': True, 'axis_source': axis_source,
                    'reason': 'measured_core_point_full_path_still_required'}
            result['candidates'].append(item)
            chosen_positions.append(position)
        if len(result['candidates']) == 1:
            result['fallback_reason'] = 'no_verified_alternate_core_surface_point'
    except (ValueError, TypeError, KeyError) as error:
        result['fallback_reason'] = str(error)
        result['candidates'] = [centre]
    return result


def _angle_distance(a, b, period):
    return abs((a - b + period / 2.) % period - period / 2.)


class GraspAttemptHistory:
    """Bounded failed-grasp memory survives reacquisition/re-numbering.

    begin_attempt/record_outcome surround one complete grasp candidate, not each
    IK seed or approach height. Record planning_rejected only after that grasp's
    bounded path options are exhausted. Limits count all candidate attempts;
    recovery duration starts at the first failure and never resets on motion.
    """
    def __init__(self, max_attempts_per_object=20, max_recovery_seconds=240.,
                 scene_change_m=.004, scene_change_deg=8., point_reuse_m=.002,
                 wrist_reuse_deg=3., max_depth_layers=2):
        if type(max_attempts_per_object) is not int or max_attempts_per_object < 1:
            raise ValueError('invalid_attempt_limit')
        if type(max_depth_layers) is not int or max_depth_layers < 1:
            raise ValueError('invalid_depth_layer_limit')
        for value in (max_recovery_seconds, scene_change_m, scene_change_deg, point_reuse_m, wrist_reuse_deg):
            if not _finite(value) or value <= 0:
                raise ValueError('invalid_history_threshold')
        self.limits = {'max_attempts_per_object': max_attempts_per_object,
                       'max_recovery_seconds': float(max_recovery_seconds), 'scene_change_m': scene_change_m,
                       'scene_change_deg': scene_change_deg, 'point_reuse_m': point_reuse_m,
                       'wrist_reuse_deg': wrist_reuse_deg, 'max_depth_layers': max_depth_layers,
                       'threshold_status': 'development_initial_values'}
        self.records = []
        self.recovery_started = {}
        self.next_id = 1

    def _descriptor(self, object_candidate, pose_candidate, wrist_variant, depth_layer,tilt_deg=0.):
        stable = object_candidate.get('stable_id')
        if not isinstance(stable, str) or not stable:
            raise ValueError('stable_identity_required')
        if pose_candidate.get('stable_id') != stable or pose_candidate.get('class') != object_candidate.get('class'):
            raise ValueError('pose_candidate_changed_object_identity')
        if wrist_variant not in (0, 180) or isinstance(wrist_variant, bool):
            raise ValueError('unsupported_wrist_symmetry')
        if not _finite(tilt_deg) or tilt_deg not in (0.,-10.,10.):raise ValueError('unsupported_bounded_tilt')
        if type(depth_layer) is not int or not 0 <= depth_layer < self.limits['max_depth_layers']:
            raise ValueError('unsupported_depth_layer')
        centre = _vector(object_candidate.get('world_position'), 3)
        point = _vector(pose_candidate.get('world_position'), 3)
        angle = object_candidate.get('angle_deg')
        pose_angle = pose_candidate.get('angle_deg')
        if not _finite(angle) or not _finite(pose_angle):
            raise ValueError('measured_grasp_angle_required')
        normalized_pose_angle = (pose_angle + 90.) % 180. - 90.
        return {'stable_id': stable, 'class': object_candidate.get('class'), 'object_world': centre,
                'object_angle_deg': angle % 180., 'grasp_world': point,
                'wrist_angle_deg': (normalized_pose_angle + wrist_variant) % 360., 'depth_layer': depth_layer,
                'wrist_variant': wrist_variant,'tilt_deg':float(tilt_deg)}

    def _same_scene(self, a, b):
        return (math.dist(a['object_world'], b['object_world']) <= self.limits['scene_change_m']
                and _angle_distance(a['object_angle_deg'], b['object_angle_deg'], 180.) <= self.limits['scene_change_deg'])

    def check(self, object_candidate, pose_candidate, wrist_variant=0, depth_layer=0, *,
              now, deadline, minimum_budget=10., stop_reserve=5.,tilt_deg=0.):
        """Return {allowed,reason}; explicit time arguments make audits replayable."""
        if any(not _finite(v) for v in (now, deadline, minimum_budget, stop_reserve)) or min(minimum_budget, stop_reserve) < 0:
            return {'allowed': False, 'reason': 'invalid_time_budget'}
        if deadline <= now or deadline - now < minimum_budget + stop_reserve:
            return {'allowed': False, 'reason': 'global_budget_insufficient'}
        try:
            descriptor = self._descriptor(object_candidate, pose_candidate, wrist_variant, depth_layer,tilt_deg)
        except (ValueError, TypeError) as error:
            return {'allowed': False, 'reason': str(error)}
        stable = descriptor['stable_id']
        records = [r for r in self.records if r['stable_id'] == stable]
        if any(now < (r['finished_at'] if r['finished_at'] is not None else r['started_at']) for r in records):
            return {'allowed': False, 'reason': 'history_time_moved_backwards'}
        if len(records) >= self.limits['max_attempts_per_object']:
            return {'allowed': False, 'reason': 'object_attempt_limit_reached'}
        if stable in self.recovery_started and now - self.recovery_started[stable] >= self.limits['max_recovery_seconds']:
            return {'allowed': False, 'reason': 'object_recovery_budget_exhausted'}
        if any(r['outcome'] is None for r in records):
            return {'allowed': False, 'reason': 'previous_attempt_not_finalized'}
        for record in records:
            if record['outcome'] not in FAILED_OUTCOMES or not self._same_scene(record, descriptor):
                continue
            if (record['depth_layer'] == descriptor['depth_layer']
                    and abs(record.get('tilt_deg',0.)-descriptor['tilt_deg'])<=self.limits['wrist_reuse_deg']
                    and math.dist(record['grasp_world'], descriptor['grasp_world']) <= self.limits['point_reuse_m']
                    and _angle_distance(record['wrist_angle_deg'], descriptor['wrist_angle_deg'], 360.) <= self.limits['wrist_reuse_deg']):
                return {'allowed': False, 'reason': 'unchanged_failed_grasp',
                        'previous_attempt_id': record['attempt_id'], 'previous_outcome': record['outcome']}
        return {'allowed': True, 'reason': 'new_or_materially_changed_grasp', 'descriptor': descriptor}

    def begin_attempt(self, object_candidate, pose_candidate, wrist_variant=0, depth_layer=0, *,
                      now, deadline, minimum_budget=10., stop_reserve=5.,tilt_deg=0.):
        decision = self.check(object_candidate, pose_candidate, wrist_variant, depth_layer,
                              now=now, deadline=deadline, minimum_budget=minimum_budget, stop_reserve=stop_reserve,tilt_deg=tilt_deg)
        if not decision['allowed']:
            raise ValueError(decision['reason'])
        ident = 'grasp-attempt-%d' % self.next_id
        self.next_id += 1
        self.records.append({**decision['descriptor'], 'attempt_id': ident, 'started_at': now,
                             'finished_at': None, 'outcome': None, 'details': {}})
        return ident

    def record_outcome(self, attempt_id, outcome, *, now, details=None):
        if outcome not in OUTCOMES:
            raise ValueError('unsupported_grasp_outcome')
        found = [r for r in self.records if r['attempt_id'] == attempt_id]
        if len(found) != 1:
            raise ValueError('unknown_attempt_id')
        record = found[0]
        if record['outcome'] is not None:
            raise ValueError('attempt_already_finalized')
        if not _finite(now) or now < record['started_at']:
            raise ValueError('invalid_outcome_time')
        record.update(outcome=outcome, finished_at=now, details=copy.deepcopy(details or {}))
        if outcome in FAILED_OUTCOMES:
            self.recovery_started.setdefault(record['stable_id'], now)
        return copy.deepcopy(record)

    def snapshot(self):
        return {'limits': copy.deepcopy(self.limits), 'records': copy.deepcopy(self.records),
                'recovery_started': dict(self.recovery_started)}
