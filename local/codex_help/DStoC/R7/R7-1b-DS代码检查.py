"""Codex 撰写：DS 执行的无仿真检查；不加载模型、不连接 ROS、不驱动机械臂。

python3 R7-1b-DS代码检查.py --code /root/tcei_final_v2_23/tcei_260920v2/tcei_stack
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import types
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', required=True)
    code = Path(parser.parse_args().code).resolve()
    sys.path.insert(0, str(code))
    from perception_tracking import CandidateTracker
    from mission_ledger import _validated_reclassification

    # 提取实际改动的方法，避免导入 nine_classify 的 ROS/视觉依赖。
    tree = ast.parse((code / 'nine_classify.py').read_text(encoding='utf-8'))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'NineRotationDetector')
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_classify_proposals', '_scan_worker')]
    namespace = {'json': json, 'uuid': uuid,
                 'time': types.SimpleNamespace(monotonic=lambda: 20.),
                 'rospy': types.SimpleNamespace(loginfo=lambda *a: None, logwarn=lambda *a: None),
                 '_centre': lambda b: ((b[0]+b[2])/2, (b[1]+b[3])/2),
                 '_numbered_view': lambda rgb, props: rgb,
                 'QUERY_CLASSES': [('烟雾弹', 'Smokegrenade'), ('弹夹', 'Magazine'),
                                   ('军用手电筒', 'Torch'), ('手雷', 'Grenade')],
                 'FULL_TTL': 300., 'ASSOC_MAX_CENTRE_PX': 20., 'RETAIN_SECONDS': 90.}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(code / 'nine_classify.py'), 'exec'), namespace)
    props = [{'bbox': [90, 90, 110, 110], 'pixel': [100., 100.]}]

    def classify(final, full='Torch', crop='Grenade', gs='Smokegrenade', gt='Grenade'):
        calls = []
        fake = types.SimpleNamespace(fastpath=False,
            capabilities={'declared_classes': ['Grenade', 'Smokegrenade', 'Torch', 'Magazine']})
        fake._ask = lambda *a, **kw: json.dumps({'1': full})
        def answer(rgb, bbox, prompt, tag):
            stage = tag.rsplit(':', 1)[-1]
            calls.append(stage)
            return {'crop_en': crop, 'crop_zh': crop, 'recheck_gt': gt,
                    'recheck_gs': gs, 'final': final}[stage]
        fake._classify_view_crop = answer
        result = namespace['_classify_proposals'](fake, None, props, scan_id='check')
        return result, calls

    (assign, cheap, unresolved), calls = classify('Torch')
    assert not assign and unresolved[0]['pre_final'] == 'Smokegrenade'
    assert calls.index('recheck_gt') < calls.index('recheck_gs')
    assert classify('Smokegrenade')[0][0] == {0: 'Smokegrenade'}
    assert classify(None)[0][2][0]['reason'] == 'final_unconfirmed'
    # 二选一被迫误报手雷时，两个裁剪支持的弹夹终审仍可纠正它。
    assert classify('Magazine', full='Grenade', crop='Magazine', gs='Grenade')[0][0] == {0: 'Magazine'}

    fake = types.SimpleNamespace(_objects=[],
                                 _evidence_dir=None, _conflict_captures=0, _scan_inflight=True)
    fake._objects = [dict(props[0], **{'class': 'Smokegrenade', 'full': True,
        'verified_at': 10., 'seen_at': 10., 'pending': 'Torch', 'pending_scan_id': 'old-t',
        'pending_full': True, 'class_scan_id': 'old-s', 'class_support_scan_ids': ['old-s']})]
    fake._classify_proposals = lambda *a, **kw: ({}, set(), {0: {'reason': 'final_conflict'}})
    namespace['_scan_worker'](fake, None, props)
    memory = fake._objects[0]
    assert memory['class'] == 'Smokegrenade' and memory['verified_at'] == 10.
    assert 'pending' not in memory and memory['class_support_scan_ids'] == ['old-s']
    assert memory['classification_status'] == 'retained_conflict'
    fake._objects = []
    namespace['_scan_worker'](fake, None, props)
    assert not fake._objects, '首次未决不能创建已知类别'
    fake._classify_proposals = lambda *a, **kw: ({0: 'Smokegrenade'}, set(), {})
    namespace['_scan_worker'](fake, None, props)
    namespace['_scan_worker'](fake, None, props)
    support = list(fake._objects[0]['class_support_scan_ids'])
    assert len(set(support)) == 2
    fake._classify_proposals = lambda *a, **kw: ({0: 'Smokegrenade'}, {0}, {})
    namespace['_scan_worker'](fake, None, props)
    assert fake._objects[0]['class_support_scan_ids'] == support, '廉价扫描不能冒充完整扫描'
    fake._classify_proposals = lambda *a, **kw: ({0: 'Torch'}, set(), {})
    namespace['_scan_worker'](fake, None, props)
    assert fake._objects[0]['class'] == 'Smokegrenade'
    namespace['_scan_worker'](fake, None, props)
    assert fake._objects[0]['class'] == 'Torch'
    assert len(set(fake._objects[0]['class_support_scan_ids'])) == 2
    assert not set(fake._objects[0]['class_support_scan_ids']) & set(support)

    tracker = CandidateTracker()
    def row(category, scans):
        return {'id': '1', 'class': category, 'raw_class': category,
                'pixel': [100., 100.], 'bbox': [90, 90, 110, 110], 'depth': 1.,
                'class_support_scan_ids': scans, 'classification_status': 'accepted',
                'grasp_ready': True}
    tracker.update([row('Smokegrenade', ['s0'])], 1, 100.)
    tracker.update([row('Smokegrenade', ['s0'])], 2, 100.1)
    for frame in range(3, 23):
        published, _ = tracker.update([row('Torch', ['t1'])], frame, 100.+frame/10)
        assert published[0]['class'] == 'Smokegrenade', '缓存重复帧不得触发改判'
    published, _ = tracker.update([row('Torch', ['t1', 't2'])], 23, 102.3)
    assert published[0]['class'] == 'Torch'
    published, _ = tracker.update([row('Torch', ['t1', 't2'])], 24, 102.4)
    candidate = published[0]
    assert candidate['class_source'] == 'recheck_majority' and candidate['reclass_audit']['frame'] == 23
    prior = {'class': 'Smokegrenade', 'last_frame': 2}
    assert _validated_reclassification(candidate, prior, 24)
    for field, value in [('stable_id', 'other'), ('scan_ids', ['t1', 't1']),
                         ('frame', 25), ('from', 'Magazine')]:
        broken = copy.deepcopy(candidate)
        broken['reclass_audit'][field] = value
        assert not _validated_reclassification(broken, prior, 24), field
    legacy = dict(candidate)
    legacy.pop('reclass_audit')
    assert not _validated_reclassification(legacy, prior, 24), '来源字符串不构成证据'
    published, _ = tracker.update([row('Smokegrenade', ['s3', 's4'])], 25, 102.5)
    assert published[0]['class'] == 'Smokegrenade', '后续有新证据时仍可纠错'
    guarded = CandidateTracker()
    guarded.update([row('Smokegrenade', ['s0'])], 1, 100.)
    guarded.update([row('Smokegrenade', ['s0'])], 2, 100.1)
    published, _ = guarded.update([row('Torch', ['t1', 't2'])], 3, 100.2,
        unknown_regions=[{'bbox': [90, 90, 110, 110], 'reason': 'occluded'}])
    assert published[0]['class'] == 'Smokegrenade', '遮挡期间不得用几何关联授权类别改判'
    print('R7-1b static behavior checks passed; no model, ROS or simulation used')


if __name__ == '__main__':
    main()
