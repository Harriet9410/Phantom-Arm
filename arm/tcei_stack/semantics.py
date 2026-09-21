"""Independent rejection guards for Nine's selections, never a target planner.

The instruction auditor recognises a deliberately bounded Chinese task grammar.
It only checks the model's claim; callers must obtain every plan from Nine. New
qualifiers fail closed rather than silently reducing the request to a category.
Camera quadrants are an explicit engineering convention, not a referee ruling.
"""
import copy
import json
import math
import re


CLASSES = ('Magazine', 'Torch', 'Grenade', 'Smokegrenade', 'CompressedFood')
ALIASES = {
    'Magazine': ('弹匣', '弹夹'),
    'Torch': ('军用手电筒', '手电筒'),
    'Grenade': ('手榴弹', '手雷'),
    'Smokegrenade': ('烟雾弹',),
    'CompressedFood': ('压缩干粮', '压缩食品', '压缩饼干'),
}
SPATIAL_WORDS = {
    'upper_left': ('左上方', '左上角', '左上'),
    'upper_right': ('右上方', '右上角', '右上'),
    'lower_left': ('左下方', '左下角', '左下'),
    'lower_right': ('右下方', '右下角', '右下'),
    'leftmost': ('最左方', '最左边', '最靠左', '最左侧', '最左'),
    'rightmost': ('最右方', '最右边', '最靠右', '最右侧', '最右'),
    'topmost': ('最上方', '最靠上', '最上边', '最上'),
    'bottommost': ('最下方', '最靠下', '最下边', '最下'),
    'left': ('左方', '左侧', '左边'),
    'right': ('右方', '右侧', '右边'),
    'upper': ('上方', '上边'),
    'lower': ('下方', '下边'),
}
EXTREMES = {'leftmost': (0, -1), 'rightmost': (0, 1),
            'topmost': (1, -1), 'bottommost': (1, 1)}
REGIONS = {'upper_left': (-1, -1), 'upper_right': (1, -1),
           'lower_left': (-1, 1), 'lower_right': (1, 1),
           'left': (-1, 0), 'right': (1, 0),
           'upper': (0, -1), 'lower': (0, 1)}
V2_KEYS = {'schema_version', 'class', 'side', 'count', 'ids', 'intent',
           'spatial', 'reference', 'quantity', 'status', 'reason'}
MODEL_SELECTION_PROTOCOL = 'nine.selection.compact.v1'
SELECTION_KEYS = {'class', 'side', 'count', 'ids'}


class ObservationRequired(ValueError):
    """A new camera observation/context is needed; re-answering cannot fix it."""


class ModelNeedsConfirmation(ValueError):
    """Recognized non-executable model decision, not invalid JSON to regenerate."""
    def __init__(self,response):
        self.model_response=copy.deepcopy(response)
        super().__init__('model_needs_confirmation: '+response['reason'])


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _consume_words(text, mapping):
    found = []
    # Longest first prevents 军用手电筒 and 最左方 being partly consumed.
    for word, meaning in sorted(((w, k) for k, words in mapping.items() for w in words),
                                key=lambda item: len(item[0]), reverse=True):
        if word in text:
            found.extend([meaning] * text.count(word))
            text = text.replace(word, '')
    return text, found


def audit_instruction(instruction):
    """Return constraints for comparing a supplied answer; never choose IDs."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('empty instruction')
    text = re.sub(r'[\s，,。.!！?？；;：:]', '', instruction)
    parts = re.split(r'放置到|放置在|放到|放在|送到|送往|放入|放至|放', text)
    if len(parts) != 2:
        raise ValueError('unsupported instruction: exactly one destination clause required')
    source, destination = parts
    side_match = re.fullmatch(r'(左侧|左边|左|右侧|右边|右)(?:的)?(?:传送带|输送带)?(?:上)?', destination)
    if side_match is None:
        raise ValueError('unsupported or ambiguous destination clause')
    side = 'left' if side_match.group(1).startswith('左') else 'right'
    source, categories = _consume_words(source, ALIASES)
    source, relations = _consume_words(source, SPATIAL_WORDS)
    if len(categories) > 1:
        raise ValueError('multiple categories require separate explicit requests')
    if len(relations) > 1:
        raise ValueError('conflicting or repeated spatial conditions')
    remaining = '剩余' in source or '剩下' in source
    if remaining and categories:
        raise ValueError('category-qualified remaining instruction not supported')
    if not remaining and not categories:
        raise ValueError('unsupported or missing category')
    if remaining and relations:
        raise ValueError('spatially qualified remaining instruction not supported')
    explicit = re.findall(r'([0-9]+|[一二两三四五六七八九十零〇]+)(?:个|件|枚|支|只|块|盒)', source)
    if len(explicit) > 1:
        raise ValueError('multiple quantities require separate explicit requests')
    numbers = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
               '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '零': 0, '〇': 0}
    all_words = any(word in source for word in ('全部', '所有'))
    if explicit and (all_words or remaining):
        raise ValueError('conflicting exact and all/remaining quantity')
    if explicit:
        token = explicit[0]
        if not token.isdigit() and token not in numbers:
            raise ValueError('unsupported exact quantity expression')
        count = int(token) if token.isdigit() else numbers[token]
        if not 1 <= count <= 10:
            raise ValueError('unsupported exact quantity')
        quantity = 'exact'
        source = re.sub(r'([0-9]+|[一二两三四五六七八九十零〇]+)(?:个|件|枚|支|只|块|盒)', '', source)
    else:
        count = None
        quantity = 'all' if all_words or remaining else 'implicit_unique'
    # This allow-list intentionally excludes ordinal, nearest/farthest, colour,
    # negation, ordering, bounds and conditionals until separately implemented.
    source = re.sub(r'抓取|拿取|拿起|取出|抓|请|将|把|全部|所有|剩余|剩下|物品|物资|东西|工具篮|篮子|篮内|框内|的', '', source)
    if source:
        raise ValueError('unsupported qualifier or instruction fragment: ' + source)
    relation = relations[0] if relations else 'none'
    if relation in EXTREMES and (quantity == 'all' or (count is not None and count != 1)):
        raise ValueError('extreme selection requires one uniquely ranked instance')
    return {'class': 'Remaining' if remaining else categories[0], 'side': side,
            'intent': 'remaining' if remaining else 'category', 'spatial': relation,
            'quantity': quantity, 'exact_count': count}


def spatial_config(scene):
    context = scene.get('spatial_context', {}) if scene else {}
    if context.get('reference') != 'camera_image' or context.get('definition') != 'basket_quadrants_v1':
        raise ValueError('spatial reference is unconfigured')
    if context.get('transform_declared') is not True:
        raise ValueError('image transform or mirror convention undeclared')
    matrix = context.get('pixel_to_reference')
    if (not isinstance(matrix, list) or len(matrix) != 3 or
            any(not isinstance(row, list) or len(row) != 3 for row in matrix) or
            not all(_finite(v) for row in matrix for v in row)):
        raise ValueError('invalid pixel-to-reference transform')
    determinant = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                   - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                   + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    if abs(determinant) < 1e-9:
        raise ValueError('singular pixel-to-reference transform')
    roi = context.get('basket_roi')
    if (not isinstance(roi, list) or len(roi) != 4 or not all(_finite(v) for v in roi)
            or roi[0] >= roi[2] or roi[1] >= roi[3]):
        raise ValueError('invalid calibrated basket ROI')
    tolerance = context.get('uncertainty_px', 5.)
    if not _finite(tolerance) or tolerance <= 0:
        raise ValueError('invalid spatial uncertainty')
    return context


def reference_pixel(pixel, context):
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2 or not all(_finite(v) for v in pixel):
        raise ValueError('invalid image position')
    m = context['pixel_to_reference']
    p = [sum(row[j] * (pixel[0], pixel[1], 1.)[j] for j in range(3)) for row in m]
    if abs(p[2]) < 1e-9:
        raise ValueError('image position maps to infinity')
    return [p[0] / p[2], p[1] / p[2]]


def _uncertainty(candidate, context):
    result = candidate.get('pixel_uncertainty_px', context['uncertainty_px'])
    if not _finite(result) or result < 0:
        raise ValueError('invalid candidate position uncertainty')
    # Additional transformed covariance is not supplied by the current camera;
    # its canonical-pixel engineering tolerance must never be reduced by input.
    return max(float(result), context['uncertainty_px'])


def _region_membership(pixel, relation, context, uncertainty):
    x1, y1, x2, y2 = context['basket_roi']
    if not (x1 <= pixel[0] <= x2 and y1 <= pixel[1] <= y2):
        return False
    axes = REGIONS[relation]
    mid = ((x1 + x2) / 2., (y1 + y2) / 2.)
    return all(not direction or direction * (pixel[axis] - mid[axis]) > uncertainty
               for axis, direction in enumerate(axes))


def _unknown_may_affect(region, category, relation, selected, context):
    possible = region.get('possible_classes')
    if isinstance(possible, list) and possible and category not in possible:
        return False
    if (region.get('reason') == 'unobserved_unverified_object'
            or region.get('bbox_is_current') is False
            or region.get('position_evidence') in ('historical', 'last_seen')):
        # A missing object can have moved anywhere. A right-side *last seen*
        # box cannot establish that the object is still right of the target.
        return True
    bbox = region.get('bbox')
    if context is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return True
    corners = [reference_pixel([x, y], context)
               for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])]
    limits = [[min(p[i] for p in corners), max(p[i] for p in corners)] for i in (0, 1)]
    tol = context['uncertainty_px']
    if relation in EXTREMES:
        axis, direction = EXTREMES[relation]
        selected_value = reference_pixel(selected['pixel'], context)[axis]
        return (limits[axis][0] <= selected_value + tol if direction < 0
                else limits[axis][1] >= selected_value - tol)
    if relation in REGIONS:
        roi = context['basket_roi']; mid = [(roi[0] + roi[2]) / 2., (roi[1] + roi[3]) / 2.]
        # A region might cross the boundary even when its centre does not.
        return all(not direction or (limits[axis][0] <= mid[axis] + tol if direction < 0
                                      else limits[axis][1] >= mid[axis] - tol)
                   for axis, direction in enumerate(REGIONS[relation]))
    return True


def _validate_scene_metadata(scene):
    if not scene or type(scene.get('frame_id')) is not int or not _finite(scene.get('stamp')):
        raise ValueError('missing source scene provenance')
    if not _finite(scene.get('observed_at')):
        raise ValueError('missing source observation time')
    if not isinstance(scene.get('unknown_regions', []), list):
        raise ValueError('invalid unknown-region metadata')


def observation_readiness(scene, instruction):
    """Wait for transient identity/fresh-scene conditions without selecting IDs.

    Geometry still validates Nine's eventual selection. This gate does not
    produce a plan or replace model reasoning, and intentionally leaves absent
    categories/unsupported language for the actual model to report.
    """
    try:
        _validate_scene_metadata(scene)
    except ValueError as error:
        return {'ready': False, 'reason': str(error)}
    try:
        constraints = audit_instruction(instruction)
    except ValueError:
        constraints = None
    candidates = scene.get('candidates', [])
    if constraints is None:
        relevant = candidates
    elif constraints['intent'] == 'remaining':
        relevant = candidates
    else:
        relevant = [c for c in candidates if c.get('class') == constraints['class']]
    confirmed = [c for c in relevant if c.get('identity_status') == 'confirmed'
                 and isinstance(c.get('stable_id'), str) and c['stable_id']]
    explicit_subset = (constraints is not None and constraints['intent'] == 'category'
                       and constraints['quantity'] == 'exact' and constraints['spatial'] == 'none')
    if explicit_subset and len(confirmed) >= constraints['exact_count']:
        return {'ready': True, 'reason': 'enough_confirmed_instances_for_model_selection'}
    if any(c not in confirmed for c in relevant):
        return {'ready': False, 'reason': 'candidate_identity_not_yet_confirmed'}
    complete = scene.get('coverage_complete', scene.get('scene_complete', False))
    if not complete and not scene.get('unknown_regions'):
        return {'ready': False, 'reason': 'unaccounted_scene_coverage_not_ready'}
    return {'ready': True, 'reason': 'fresh_identity_evidence_ready_for_model'}


def bind_task_context(task_context, scene):
    """Resolve trusted persistent ledger identities to this exact camera frame."""
    task = copy.deepcopy(task_context or {})
    original_frame = task.get('frame_id', task.get('scene_frame_id'))
    if 'request_scene_binding' not in task and any(key in task for key in (
            'frame_id', 'scene_frame_id', 'candidate_stable_ids', 'remaining_ids', 'reserved_ids')):
        task['request_scene_binding'] = {
            'frame_id': original_frame,
            **{key: task[key] for key in ('candidate_stable_ids', 'stable_ids', 'remaining_ids', 'reserved_ids')
               if key in task}}
    by_stable = {}
    for candidate in scene['candidates']:
        stable = candidate.get('stable_id')
        if stable:
            if stable in by_stable:
                raise ValueError('duplicate stable identity in camera scene')
            by_stable[stable] = candidate['id']
    for kind in ('remaining', 'reserved'):
        stable_key, id_key = kind + '_stable_ids', kind + '_ids'
        if stable_key in task:
            stable_ids = task[stable_key]
            if (not isinstance(stable_ids, list) or any(not isinstance(s, str) for s in stable_ids)
                    or len(stable_ids) != len(set(stable_ids))):
                raise ValueError('invalid stable task identities')
            missing = [s for s in stable_ids if s not in by_stable]
            if missing:
                # A missing reserved target stays reserved by stable identity;
                # an invisible remaining target prevents claiming a full set.
                task['remaining_complete'] = False
            task[id_key] = [by_stable[s] for s in stable_ids if s in by_stable]
        elif id_key in task and original_frame != scene['frame_id']:
            raise ObservationRequired('temporary task IDs belong to a different frame; refresh task context')
    task['frame_id'] = scene['frame_id']
    task['scene_frame_id'] = scene['frame_id']
    task['candidate_stable_ids'] = {ident: stable for stable, ident in by_stable.items()}
    task['stable_ids'] = sorted(by_stable)
    return task


def validate_selection(value, candidates, constraints, scene=None, task_context=None):
    """Reject incorrect model IDs using measured geometry and task ledger."""
    ids = value['ids']; count = value['count']; category = constraints['class']
    capabilities = scene.get('model_capabilities', {}) if scene else {}
    declared = capabilities.get('declared_classes')
    if isinstance(declared, list) and category != 'Remaining' and category not in declared:
        raise ValueError('recognition capability unavailable for requested class: ' + category)
    by_id = {}
    for candidate in candidates:
        ident = candidate.get('id')
        if not isinstance(ident, str) or ident in by_id:
            raise ValueError('duplicate or invalid scene candidate ID')
        by_id[ident] = candidate
    chosen = []
    for ident in ids:
        if ident not in by_id:
            raise ValueError('unknown candidate ID')
        candidate = by_id[ident]
        if candidate.get('class') not in CLASSES or (category != 'Remaining' and candidate['class'] != category):
            raise ValueError('vision/model category disagreement')
        if isinstance(declared, list) and candidate['class'] not in declared:
            raise ValueError('recognition capability unavailable for selected class')
        if not _finite(candidate.get('depth')) or not .1 < candidate['depth'] < 5.:
            raise ValueError('invalid depth')
        if not _finite(candidate.get('confidence')) or candidate['confidence'] < .6:
            raise ValueError('low-confidence candidate')
        if candidate.get('grasp_ready') is False:
            raise ObservationRequired('candidate requires geometry recovery before grasp')
        if candidate.get('identity_status') in ('ambiguous', 'lost', 'unresolved'):
            raise ObservationRequired('candidate identity is ambiguous')
        if value.get('schema_version') == 2 and (candidate.get('identity_status') != 'confirmed'
                or not isinstance(candidate.get('stable_id'), str) or not candidate['stable_id']):
            raise ObservationRequired('candidate identity not yet confirmed; obtain a new observation')
        if candidate.get('legal_source') is False:
            raise ValueError('candidate is outside legal source region')
        chosen.append(candidate)
    if constraints['intent'] == 'remaining':
        task = task_context or {}
        if (task.get('dependencies_satisfied') is not True or task.get('remaining_complete') is not True
                or scene is None or not scene.get('coverage_complete', scene.get('scene_complete', False))
                or scene.get('unknown_regions')):
            raise ObservationRequired('remaining set is incomplete or predecessors unresolved; refresh observation/task context')
        remaining = task.get('remaining_ids'); reserved = task.get('reserved_ids', [])
        if (not isinstance(remaining, list) or any(not isinstance(i, str) for i in remaining)
                or len(remaining) != len(set(remaining))):
            raise ValueError('remaining ledger missing or invalid')
        if not isinstance(reserved, list) or any(not isinstance(i, str) for i in reserved):
            raise ValueError('reserved ledger invalid')
        if reserved or task.get('reserved_stable_ids'):
            raise ValueError('remaining task ledger still contains unresolved reserved targets')
        if set(remaining) != set(by_id):
            raise ValueError('remaining task ledger omits or invents visible source objects')
        if set(ids) != set(remaining) or set(ids) & set(reserved):
            raise ValueError('model remaining selection contradicts task ledger')
        return
    matching = [c for c in candidates if c.get('class') == category]
    relation = constraints['spatial']
    context = spatial_config(scene) if relation != 'none' else None
    if context is not None:
        roi = context['basket_roi']
        for c in chosen:
            x, y = reference_pixel(c.get('pixel'), context)
            if not (roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]):
                raise ValueError('selected target outside calibrated source ROI')
    if relation in REGIONS:
        if constraints['quantity'] in ('all', 'implicit_unique'):
            roi = context['basket_roi']; mid = [(roi[0] + roi[2]) / 2., (roi[1] + roi[3]) / 2.]
            for candidate in matching:
                pixel = reference_pixel(candidate.get('pixel'), context)
                uncertainty = _uncertainty(candidate, context)
                possible = all(not direction or direction * (pixel[axis] - mid[axis]) >= -uncertainty
                               for axis, direction in enumerate(REGIONS[relation]))
                if possible and not _region_membership(pixel, relation, context, uncertainty):
                    raise ValueError('same-class candidate overlaps spatial boundary uncertainty')
        matching = [c for c in matching if _region_membership(reference_pixel(c.get('pixel'), context),
                                                             relation, context, _uncertainty(c, context))]
        if any(c not in matching for c in chosen):
            raise ValueError('selected target violates spatial region or boundary uncertainty')
    elif relation in EXTREMES:
        if count != 1:
            raise ValueError('extreme selection must contain exactly one object')
        axis, direction = EXTREMES[relation]
        selected = chosen[0]
        a = reference_pixel(selected.get('pixel'), context)[axis]
        for other in matching:
            if other['id'] == selected['id']:
                continue
            b = reference_pixel(other.get('pixel'), context)[axis]
            if direction * (a - b) <= _uncertainty(selected, context) + _uncertainty(other, context):
                raise ValueError('selected target is not the unique class-relative extreme')
        matching = chosen
    quantity = constraints['quantity']
    if quantity == 'exact' and count != constraints['exact_count']:
        raise ValueError('model quantity contradicts explicit instruction')
    if quantity == 'implicit_unique' and (count != 1 or len(matching) != 1):
        raise ValueError('implicit quantity requires one unambiguous matching object')
    if quantity == 'all' and set(ids) != {c['id'] for c in matching}:
        raise ValueError('all quantity must include every matching candidate')
    needs_complete = quantity in ('all', 'implicit_unique') or relation in EXTREMES
    if scene is not None and needs_complete:
        unknown = scene.get('unknown_regions', [])
        complete = scene.get('coverage_complete', scene.get('scene_complete', False))
        if not complete and not unknown:
            raise ObservationRequired('unaccounted occlusion prevents a unique/complete selection')
        if any(_unknown_may_affect(region, category, relation, chosen[0] if chosen else {}, context)
               for region in unknown):
            raise ObservationRequired('unknown foreground may change the requested selection')


def parse_model_plan(text, candidates, instruction, scene=None, task_context=None, allow_legacy=True):
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('invalid model JSON: ' + str(error)) from error
    if isinstance(value, dict) and value.get('status') == 'needs_confirmation':
        reason = value.get('reason')
        raise ValueError('model_needs_confirmation: ' + (reason if isinstance(reason, str) and reason else 'unspecified'))
    if not isinstance(value, dict):
        raise ValueError('schema: model answer must be an object')
    constraints = audit_instruction(instruction)
    legacy = set(value) == {'class', 'side', 'count', 'ids'}
    if legacy:
        if (not allow_legacy or constraints['spatial'] != 'none' or constraints['intent'] != 'category'
                or constraints['quantity'] != 'exact'):
            raise ValueError('schema v2 required for formal, spatial or set-valued instructions')
    else:
        if set(value) != V2_KEYS or type(value.get('schema_version')) is not int or value['schema_version'] != 2:
            raise ValueError('schema: expected version 2 fields ' + ','.join(sorted(V2_KEYS)))
        if value['status'] != 'ready' or value['reason'] != '':
            raise ValueError('model is not ready to execute')
        if value['reference'] != 'camera_image':
            raise ValueError('model uses unsupported spatial reference')
        _validate_scene_metadata(scene)
        for field in ('intent', 'spatial', 'quantity'):
            if value[field] != constraints[field]:
                raise ValueError('model ' + field + ' contradicts explicit instruction')
    for field in ('class', 'side'):
        if value[field] != constraints[field]:
            raise ValueError('model ' + field + ' contradicts explicit instruction')
    count = value['count']; ids = value['ids']
    minimum = 0 if constraints['intent'] == 'remaining' else 1
    if type(count) is not int or not minimum <= count <= 10:
        raise ValueError('invalid count')
    if not isinstance(ids, list) or len(ids) != count or any(type(i) is not str for i in ids):
        raise ValueError('count does not match IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate candidate')
    validate_selection(value, candidates, constraints, scene, task_context)
    return value


def parse_compact_selection(text,candidates,instruction,scene,task_context=None):
    """New explicit selection protocol; never a permissive legacy fallback.

    Model choices remain untouched. The existing strict v2 parser validates
    all instruction/scene constraints before we expose an internal v2 envelope.
    No object ID, class, side or count is computed or corrected on Nine's behalf.
    """
    text=text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text=text[7:-3].strip()
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    try:raw=json.loads(text,object_pairs_hook=unique_keys)
    except (TypeError,json.JSONDecodeError) as error:
        raise ValueError('invalid model JSON: '+str(error)) from error
    if not isinstance(raw,dict):raise ValueError('compact selection must be a JSON object')
    if set(raw)=={'status','reason'}:
        if raw['status']!='needs_confirmation' or not isinstance(raw['reason'],str) or not raw['reason'].strip():
            raise ValueError('invalid needs_confirmation response')
        raise ModelNeedsConfirmation(raw)
    if set(raw)!=SELECTION_KEYS:
        raise ValueError('compact selection requires exactly class, side, count, ids; extra fields are not discarded')
    constraints=audit_instruction(instruction)
    reference=spatial_config(scene)['reference']
    internal={'schema_version':2,'intent':constraints['intent'],'spatial':constraints['spatial'],
              'reference':reference,'quantity':constraints['quantity'],'status':'ready','reason':'',
              **copy.deepcopy(raw)}
    # This includes strict class/side agreement, types, ID uniqueness, full
    # geometry, unknown-region and remaining-ledger validation. Calling only
    # validate_selection would omit the outer class/side/schema checks.
    validated=parse_model_plan(json.dumps(internal,ensure_ascii=False),candidates,instruction,
                               scene,task_context,allow_legacy=False)
    provenance={key:'nine_raw_selection' for key in SELECTION_KEYS}
    provenance.update(intent='audit_instruction',spatial='audit_instruction',quantity='audit_instruction',
        reference='reference_scene.spatial_context.reference',schema_version='internal_protocol',
        status='successful_constraint_validation',reason='successful_constraint_validation')
    return {'model_protocol':MODEL_SELECTION_PROTOCOL,'model_selection':copy.deepcopy(raw),
            'internal_v2':validated,'audited_constraints':copy.deepcopy(constraints),
            'semantic_provenance':provenance}


def scene_unchanged(original, current, rebind, max_pixels=3., max_depth=.005):
    """Validate the entire inference scene, not just the model's chosen object.

    This prevents a moving unselected peer from invalidating an extreme. It
    returns the measured identity rebinding, never a different semantic target.
    """
    if len(original['candidates']) != len(current['candidates']):
        raise ObservationRequired('scene membership changed during inference; reobserve')
    if original.get('spatial_context') != current.get('spatial_context'):
        raise ObservationRequired('spatial calibration changed during inference')
    if (original.get('coverage_complete', original.get('scene_complete')) !=
            current.get('coverage_complete', current.get('scene_complete'))):
        raise ObservationRequired('unknown foreground changed during inference; reobserve')
    old_unknown = original.get('unknown_regions', [])
    new_unknown = current.get('unknown_regions', [])
    if len(old_unknown) != len(new_unknown):
        raise ObservationRequired('unknown foreground changed during inference; reobserve')
    for before, after in zip(old_unknown, new_unknown):
        if before == after:
            continue
        before_box, after_box = before.get('bbox'), after.get('bbox')
        if (not isinstance(before_box, list) or not isinstance(after_box, list)
                or len(before_box) != 4 or len(after_box) != 4
                or any(abs(a - b) > max_pixels for a, b in zip(before_box, after_box))
                or any(before.get(key) != after.get(key) for key in ('reason', 'class', 'possible_classes'))):
            raise ObservationRequired('unknown foreground changed during inference; reobserve')
    rebound = {}; used = set()
    for candidate in original['candidates']:
        try:
            target = rebind(candidate, current['candidates'], max_pixels=max_pixels, max_depth=max_depth)
        except ValueError as error:
            raise ObservationRequired('scene association changed during inference; reobserve: ' + str(error)) from error
        if target['id'] in used:
            raise ObservationRequired('non-unique scene association after inference')
        if candidate.get('stable_id') and target.get('stable_id') != candidate['stable_id']:
            raise ObservationRequired('stable identity changed during inference')
        rebound[candidate['id']] = target
        used.add(target['id'])
    return rebound
