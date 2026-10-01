#!/usr/bin/env python3
"""Nine-grid classification detector: per-class localization queries.

Same detect() contract as RotationDetector.  Depth segmentation (geometry)
proposes the blocks; classification is done the way the 9/28 live test PROVED
(sft_v2/outputs/live_jiuge/answers.txt): for each official class, ask the
nine-grid model once -- "请框出图中的 X。只输出一个 <box>。" -- parse the
<box>, convert 0-1000 -> pixels, then associate the box to depth proposals
exactly like the YOLO path does.  Rotation stays with the geometry stage.

The model call goes through the /tcei/classify_request -> /tcei/classify_result
service pair hosted by nine_node (the process that already owns the model --
never load a second model copy).  No robot commands here.
"""
import json
import re
import threading
import time
import uuid

import cv2
import numpy as np
import rospy
from PIL import Image as PILImage
from std_msgs.msg import String

from perception_tracking import canonical_class, CLASS_ALIASES
from rotation_perception import RotationDetector, source_components, associate

BOX_RE = re.compile(r'<box>\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*')

# (查询用名, 官方类名) —— 查询名与 9/28 live 实测一致（逐类单目标查询，已验证准确）
QUERY_CLASSES = [('烟雾弹', 'Smokegrenade'), ('弹夹', 'Magazine'),
                 ('军用手电筒', 'Torch'), ('手雷', 'Grenade')]

# 定位结果的有效期：扫描一轮约 15s，留足余量让语义端 40s 等待窗口内
# 一定能看到带类别的候选。
CACHE_TTL = 45.0


class NineRotationDetector(RotationDetector):
    """RotationDetector contract; classification by per-class localization
    queries to the nine-grid model (nine_node classify service)."""

    def __init__(self, model=None, timeout=8.0):
        if model is not None:
            RotationDetector.__init__(self, model)
        else:
            # 纯九格合规形态：YOLO 权重不加载，类别能力来自官方类目表。
            self.model = None
            self.preferred = []
            supported = sorted(CLASS_ALIASES)
            self.capabilities = {'declared_classes': supported, 'raw_names': supported,
                'missing_official_classes': [], 'source': 'nine_grid_localization',
                'accuracy_verified': False}
        self.timeout = float(timeout)
        self.request_pub = rospy.Publisher('/tcei/classify_request', String, queue_size=1)
        self.result_sub = rospy.Subscriber('/tcei/classify_result', String,
                                           self._on_result, queue_size=4)
        self._results = {}
        self._cache = None
        self._scan_inflight = False
        self.pending_scan = True   # 首帧扫描一次，让候选尽快带上类别
        rospy.Subscriber('/tcei/prepare_classification', String, self._on_prepare, queue_size=1)

    def _on_prepare(self, msg):
        """nine_node 语义请求到达时触发一次按需扫描（按需分类）。"""
        self.pending_scan = True

    def _on_result(self, msg):
        try:
            data = json.loads(msg.data)
            request_id = data.get('request_id')
            if request_id:
                self._results[request_id] = data
        except Exception:
            pass

    @staticmethod
    def _to_pixels(box, width, height):
        x1, y1, x2, y2 = [max(0.0, min(1000.0, float(v))) for v in box]
        return [round(x1 / 1000 * width), round(y1 / 1000 * height),
                round(x2 / 1000 * width), round(y2 / 1000 * height)]

    def _ask(self, rgb, prompt):
        """Full frame + prompt -> raw model answer via the nine_node service."""
        request_id = 'locate-' + uuid.uuid4().hex[:8]
        self._results.pop(request_id, None)
        pil = PILImage.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        self.request_pub.publish(String(json.dumps(
            {'request_id': request_id, 'prompt': prompt,
             'image_jpeg_b64': self._jpeg_b64(rgb)}, ensure_ascii=False)))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = self._results.pop(request_id, None)
            if data is not None:
                return '' if data.get('error') else str(data.get('raw_answer') or '')
            time.sleep(0.05)
        rospy.logwarn_throttle(5, 'nine locate timeout: %.20s', prompt)
        return ''

    def _locate_classes(self, rgb):
        """Per-class localization queries -> {official_class: {'bbox_px'}}.

        Runs on a background thread (see detect): the classify service answers
        in ~2s per class and the 4-8 query cycle takes ~15s, which must never
        block the streaming detect callback (that made every candidate stale).
        The adapter's boxes drift between samples, so each class gets up to two
        samples and keeps the first parseable box; association to depth
        proposals happens per detect cycle against the current geometry."""
        h, w = rgb.shape[:2]
        located = {}
        for zh, cls in QUERY_CLASSES:
            for _ in (1, 2):
                answer = self._ask(rgb, '请框出图中的 %s。只输出一个 <box>。' % zh)
                m = BOX_RE.search(answer or '')
                if not m:
                    continue
                box1000 = [float(m.group(i)) for i in range(1, 5)]
                if box1000[0] >= box1000[2] or box1000[1] >= box1000[3]:
                    continue
                located[cls] = {'bbox_px': self._to_pixels(box1000, w, h)}
                break
        return located

    def _scan_worker(self, rgb):
        try:
            located = self._locate_classes(rgb)
        except Exception as error:
            rospy.logwarn('nine scan failed: %s', error)
            located = {}
        self._cache = {'at': time.monotonic(), 'located': located}
        self._scan_inflight = False

    def detect(self, rgb, depth, k):
        began = time.monotonic()
        proposals, support, metadata = source_components(depth, k, return_metadata=True)
        found = [{} for _ in proposals]
        base = []
        if self.pending_scan and not self._scan_inflight:
            # 按需扫描转入后台线程：detect 绝不阻塞，候选持续按帧新鲜发布，
            # 扫描完成后类别写入缓存并在 TTL 内持续套用。
            self.pending_scan = False
            self._scan_inflight = True
            threading.Thread(target=self._scan_worker, args=(rgb.copy(),), daemon=True).start()
        classified = {}
        if self._cache is not None and time.monotonic() - self._cache['at'] < CACHE_TTL:
            classified = self._cache['located']
        # 框-提案匹配：对当前帧几何重新关联（缓存里只存像素框）
        for cls, loc in classified.items():
            index = associate(loc['bbox_px'], proposals)
            if index is None:
                continue
            row = {'class': cls, 'raw_class': cls, 'confidence': 0.90,
                   'bbox': [float(v) for v in loc['bbox_px']],
                   'recognition_rotation_deg': 0.0}
            found[index][cls] = row
            base.append(row)
        # 未分类的提案以 unknown 候选发布：编号与标注图保持稳定，
        # 语义请求到达触发扫描后即补齐真实类别。
        for index, proposal in enumerate(proposals):
            if found[index]:
                continue
            x1, y1, x2, y2 = proposal['bbox']
            row = {'class': 'unknown', 'raw_class': 'unknown', 'confidence': 0.90,
                   'bbox': [float(x1), float(y1), float(x2), float(y2)],
                   'recognition_rotation_deg': 0.0}
            found[index][row['class']] = row
            base.append(row)
        source = []
        usefulness = {}
        unknown = metadata['unknown_regions']
        for proposal, classes in zip(proposals, found):
            ordered = sorted(classes.values(), key=lambda b: b['confidence'], reverse=True)
            reason = None
            if not ordered:
                reason = 'unrecognized_foreground'
            elif ordered[0]['class'] is None:
                reason = 'unsupported_model_class'
            elif ordered[0]['confidence'] < .60:
                reason = 'low_class_confidence'
            elif len(ordered) > 1 and ordered[0]['confidence'] - ordered[1]['confidence'] < .10:
                reason = 'class_conflict'
            if reason:
                unknown.append({key: value for key, value in proposal.items() if key != 'body_mask'})
                unknown[-1].update({'reason': reason,
                    'class_hypotheses': [{'class': row['class'], 'raw_class': row['raw_class'],
                                          'confidence': row['confidence']} for row in ordered],
                    'grasp_ready': False})
                continue
            chosen = dict(ordered[0])
            angle = chosen['recognition_rotation_deg']
            usefulness[angle] = usefulness.get(angle, 0) + 1
            chosen.update({key: value for key, value in proposal.items() if key != 'body_mask'})
            if proposal['grasp_uncertainty']:
                unknown.append({'bbox': proposal['bbox'], 'pixel': proposal['pixel'],
                    'normalized_xy': proposal['normalized_xy'], 'class': chosen['class'],
                    'reason': proposal['grasp_uncertainty'][0],
                    'reasons': proposal['grasp_uncertainty'], 'recognized_region': True})
            source.append(chosen)
        self.preferred = sorted(usefulness, key=lambda a: (-usefulness[a], 0))
        return source, base, {'views': [0.0], 'seconds': time.monotonic() - began,
            'foreground_count': len(proposals), 'recognized_count': len(source),
            'support_z': support, 'unknown_regions': unknown,
            'visibility': metadata['visibility'], 'model_capabilities': self.capabilities,
            'coverage_complete': not unknown and metadata['visibility']['clear'],
            'coverage_scope': ('visible_foreground_only; hidden fully occluded objects cannot be excluded; '
                               'classification by FM9G4B-V per-class localization via nine_node service')}

    @staticmethod
    def _jpeg_b64(image):
        import base64
        ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


def query_prompt(name):
    return '请框出图中的 %s。只输出一个 <box>。' % name
