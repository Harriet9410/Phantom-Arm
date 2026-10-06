"""Logic test for the nine_classify "settled object" cheap path (dev_v7_t1_39).

Loads the REAL tcei_stack/nine_classify.py with stubbed heavy deps (numpy/cv2/
rospy/base detector) and a counting fake model, then asserts:
  A cold scan  -> full battery for every object (unchanged behaviour)
  A warm scan  -> settled objects skip crops/re-checks (crop calls drop to 0)
  A disagreement -> that object alone runs the full battery (correction kept)
  FULL_TTL expiry -> settled hints are ignored (re-verified)
Run: python local/1003/v1/_test_fastpath_gate.py
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

MODULE = Path(__file__).resolve().parents[3] / 'arm' / 'tcei_stack' / 'nine_classify.py'


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _Base:
    def __init__(self, *a, **k):
        pass


_stub('numpy')
_stub('cv2')
_stub('rospy', loginfo=lambda *a, **k: None, logwarn=lambda *a, **k: None,
      get_param=lambda *a, **k: None, logwarn_throttle=lambda *a, **k: None)
_stub('perception_tracking', canonical_class=lambda v: v, CLASS_ALIASES={})
_stub('std_msgs')
_stub('std_msgs.msg', String=object)
_stub('rotation_perception', RotationDetector=_Base,
      source_components=lambda *a, **k: [], associate=lambda *a, **k: {})

spec = importlib.util.spec_from_file_location('nine_classify', MODULE)
nc = importlib.util.module_from_spec(spec)
sys.modules['nine_classify'] = nc
spec.loader.exec_module(nc)

CLASSES = ['Magazine', 'Torch', 'Grenade', 'Smokegrenade', 'CompressedFood']
BOXES = [[100, 100, 150, 130], [400, 100, 450, 130], [700, 100, 750, 130]]
# view A (whole frame) answers, per scenario
VIEW_A = {
    'cold': '{"1":"Magazine","2":"Torch","3":"Grenade"}',
    'flip': '{"1":"Magazine","2":"Torch","3":"Torch"}',
}
CROP_TRUTH = ['Magazine', 'Torch', 'Grenade']


class Counter:
    def __init__(self):
        self.ask = 0
        self.crop = 0

    def reset(self):
        self.ask = self.crop = 0


def make_detector(counter, view_a_key):
    det = object.__new__(nc.NineRotationDetector)
    det._objects = []
    det._cache = None
    det._scan_inflight = False
    det.fastpath = True
    det.timeout = 1.0
    det.capabilities = {'declared_classes': CLASSES}

    def _ask(rgb, prompt):
        counter.ask += 1
        return VIEW_A[view_a_key]

    def _crop(rgb, bbox, prompt):
        counter.crop += 1
        cx = (bbox[0] + bbox[2]) / 2.0
        for box, cls in zip(BOXES, CROP_TRUTH):
            if abs((box[0] + box[2]) / 2.0 - cx) < 30:
                # GS/GT re-check prompts must be answered consistently with truth
                return cls
        return None

    det._ask = _ask
    det._classify_view_crop = _crop
    return det


def props():
    return [{'bbox': list(b), 'pixel': [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]} for b in BOXES]


def classes_of(det):
    return [entry.get('class') for entry in det._objects]


failures = []


def check(label, got, want):
    ok = got == want
    print('%-46s got=%-28s want=%s %s' % (label, got, want, 'OK' if ok else '<<< FAIL'))
    if not ok:
        failures.append(label)


counter = Counter()
det = make_detector(counter, 'cold')

# --- A: cold scan -> full battery everywhere, nothing cheap
det._scan_worker(None, props())
cold_asks, cold_crops = counter.ask, counter.crop
print('   (cold scan cost: view-A=%d crop=%d)' % (cold_asks, cold_crops))
check('A cold scan: one whole-frame call', cold_asks, 1)
check('A cold scan: full battery ran (crop calls > 0)', cold_crops > 0, True)
check('A cold scan: classes', classes_of(det), CROP_TRUTH)
check('A all memory entries marked fully verified',
      all(e.get('full') and e.get('verified_at', 0) > 0 for e in det._objects), True)

# --- B: warm scan (nothing changed) -> settled, no crop calls at all
counter.reset()
det._scan_worker(None, props())
print('   (warm scan cost: view-A=%d crop=%d)' % (counter.ask, counter.crop))
check('B warm scan: one whole-frame call', counter.ask, 1)
check('B warm scan: crop calls dropped to zero', counter.crop, 0)
check('B warm scan: classes unchanged', classes_of(det), CROP_TRUTH)
check('B verification retained across the cheap scan',
      all(e.get('full') for e in det._objects), True)

# --- C: whole-frame view disagrees for one object -> only that one re-verified
det2 = make_detector(counter, 'cold')
counter.reset()
det2._scan_worker(None, props())          # cold: establish all three
counter.reset()
flipped = make_detector(counter, 'flip')
det2._classify_view_crop, det2._ask = flipped._classify_view_crop, flipped._ask
det2._scan_worker(None, props())          # warm, with object 3 disagreeing
print('   (disagreement scan cost: view-A=%d crop=%d, cold was %d)'
      % (counter.ask, counter.crop, cold_crops))
check('C disagreement: only the doubtful object re-verified',
      0 < counter.crop < cold_crops, True)
check('C disagreement: anchor error still corrected by the battery',
      classes_of(det2), CROP_TRUTH)

# --- D: FULL_TTL expiry -> hints ignored, full battery again
counter.reset()
det3 = make_detector(counter, 'cold')
det3._objects = [{'bbox': list(b), 'pixel': None, 'class': c, 'full': True,
                  'verified_at': time.monotonic() - nc.FULL_TTL - 1.0}
                 for b, c in zip(BOXES, CROP_TRUTH)]
det3._scan_worker(None, props())
print('   (expired scan cost: crop=%d)' % counter.crop)
check('D expired verification: full battery restored', counter.crop, cold_crops)

print()
print('FULL_TTL =', nc.FULL_TTL, '| ASSOC_MAX_CENTRE_PX =', nc.ASSOC_MAX_CENTRE_PX)
print('RESULT:', 'ALL PASS' if not failures else 'FAILURES: %s' % failures)
sys.exit(1 if failures else 0)
