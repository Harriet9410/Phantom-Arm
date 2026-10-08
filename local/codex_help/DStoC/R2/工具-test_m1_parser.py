#!/usr/bin/env python3
"""M1 offline validation (review R2-1 adoption gate: no new obvious misjudgments).

Part 1 - unit tests for the alias-aware parser (`_parse_class_answer`):
  English substrings preserved, Chinese aliases, conflict -> uncertain,
  negation handling, longer-match containment.
Part 2 - replay every saved raw answer of the baseline batch's five cases
  (`<stack>/logs/nine.log` classify_answered) through OLD vs NEW parsing and
  report every outcome change, so a human can check none is a misjudgment.

Usage (no ROS needed):
  python test_tools/test_m1_parser.py                # part 1 only (local)
  python test_tools/test_m1_parser.py --replay <batch_dir>   # both parts
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tcei_stack'))

import types


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _Base:
    def __init__(self, *a, **k):
        pass


for name in ('numpy', 'cv2', 'rospy'):
    _stub(name, get_param=lambda *a, **k: None, loginfo=lambda *a, **k: None,
          logwarn=lambda *a, **k: None, logwarn_throttle=lambda *a, **k: None)
_stub('std_msgs')
_stub('std_msgs.msg', String=object)
_stub('rotation_perception', RotationDetector=_Base,
      source_components=lambda *a, **k: [], associate=lambda *a, **k: {})

import importlib.util  # noqa: E402

MODULE = pathlib.Path(__file__).resolve().parent.parent / 'tcei_stack' / 'nine_classify.py'
spec = importlib.util.spec_from_file_location('nine_classify', MODULE)
nc = importlib.util.module_from_spec(spec)
sys.modules['nine_classify'] = nc
spec.loader.exec_module(nc)

DECLARED = ('Magazine', 'Torch', 'Grenade', 'Smokegrenade', 'CompressedFood')


def make_parser():
    det = object.__new__(nc.NineRotationDetector)
    det.capabilities = {'declared_classes': list(DECLARED)}
    return det


def old_parse(text):
    """Parser as shipped in dev_v7_t1_41 (English names only, longest first)."""
    low = str(text or '').lower()
    for cls in sorted(DECLARED, key=len, reverse=True):
        if cls.lower() in low:
            return cls
    return None


failures = []


def check(label, got, want):
    ok = got == want
    print('%-64s got=%-16s want=%-14s %s' % (label, got, want, 'OK' if ok else '<<< FAIL'))
    if not ok:
        failures.append(label)


def main():
    det = make_parser()
    new = det._parse_class_answer

    print('== Part 1: 解析器单测 ==')
    # 英文行为保持（长名优先）
    check("EN 'Smokegrenade' 不受 'Grenade' 子串影响",
          new('Smokegrenade'), 'Smokegrenade')
    check("EN 'Grenade（手雷）' -> Grenade", new('Grenade（手雷）'), 'Grenade')
    check("EN 句子带解释 -> 取类名", new('这个物体是 Torch（军用手电筒）。'), 'Torch')
    check("EN 'Magazine' -> Magazine", new('Magazine'), 'Magazine')
    # 中文别名（M1 新能力）
    check("CN '这个物资是手榴弹。' -> Grenade（旧解析 None）", new('这个物资是手榴弹。'), 'Grenade')
    check("CN '弹夹' -> Magazine", new('弹夹'), 'Magazine')
    check("CN '军用手电筒' -> Torch", new('军用手电筒'), 'Torch')
    check("CN '烟雾弹' -> Smokegrenade", new('烟雾弹'), 'Smokegrenade')
    check("CN '手雷' -> Grenade", new('手雷'), 'Grenade')
    check("CN '压缩干粮' -> CompressedFood", new('压缩干粮'), 'CompressedFood')
    # 冲突 -> 不确定
    check("冲突 '手雷和烟雾弹' -> None", new('手雷和烟雾弹'), None)
    check("冲突 'Grenade or Torch' -> None", new('Grenade or Torch'), None)
    # 否定就近否决：被否决的类弃权，剩余一类才返回（永不返回被否决的类）
    check("否定 '不是手雷，是烟雾弹' -> Smokegrenade（否决手雷）", new('不是手雷，是烟雾弹'), 'Smokegrenade')
    check("否定 'It is a Smokegrenade, not Grenade' -> Smokegrenade",
          new('It is a Smokegrenade, not Grenade'), 'Smokegrenade')
    check("否定 '这不是手雷' -> None（唯一类被否决）", new('这不是手雷'), None)
    check("否定 '并非烟雾弹' -> None", new('并非烟雾弹'), None)
    check("否定 '手雷？不是。' -> None（后窗否定）", new('手雷？不是。'), None)
    # 无类名 / 空答
    check("无类名 '无法判断' -> None", new('无法判断'), None)
    check("空 -> None", new(''), None)
    check("None -> None", new(None), None)
    # 长名包含短名（英文子串）不产生冲突
    check("包含 'It is a Smokegrenade, not Grenade' -> Smokegrenade",
          new('It is a Smokegrenade, not Grenade'), 'Smokegrenade')

    print()
    if '--replay' not in sys.argv:
        print('RESULT:', 'ALL PASS' if not failures else 'FAILURES: %s' % failures)
        return 1 if failures else 0

    print('== Part 2: 基线五案全量答案重放（旧 vs 新） ==')
    batch = pathlib.Path(sys.argv[sys.argv.index('--replay') + 1])
    changed, total = [], 0
    for stack in sorted(batch.parent.glob(batch.name + '_scramble_*_stack')):
        log = stack / 'logs' / 'nine.log'
        if not log.exists():
            continue
        case = stack.name[-2:]
        for line in log.read_text(encoding='utf-8', errors='ignore').splitlines():
            if 'classify_answered' not in line:
                continue
            i = line.find('{')
            try:
                d = json.loads(line[i:])
            except Exception:
                continue
            answer = str(d.get('answer') or '')
            total += 1
            o, n = old_parse(answer), new(answer)
            if o != n:
                changed.append({'case': case, 'old': o, 'new': n, 'answer': answer[:120]})
    from collections import Counter
    sig = Counter((c['old'], c['new']) for c in changed)
    print('答案总数 %d | 解析结果变化 %d' % (total, len(changed)))
    for (o, n), cnt in sig.most_common():
        print('  %-14s -> %-14s x%d' % (o, n, cnt))
    print()
    print('逐条（供人工核对是否有新增误判）：')
    for c in changed:
        print('  [%s] %-13s -> %-13s | %s' % (c['case'], c['old'], c['new'], c['answer'].replace('\n', ' ')))
    print()
    # 人工核对提示：old=None→new=类名 的句子必须语义正确；old=类名→new=None 只允许发生在
    # 多类/否定句（保守化）；old=A→new=B 属于解析修正，但需逐条确认。
    print('RESULT:', 'ALL PASS' if not failures else 'FAILURES: %s' % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
