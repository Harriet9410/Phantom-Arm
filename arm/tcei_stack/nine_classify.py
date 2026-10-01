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
        """Image + prompt -> raw model answer via the nine_node service.

        grounding:false 始终走基座（选择权重）模型——分类/定位查询都不加载
        框选适配器（实测基座的类别判断远准于适配器）。"""
        request_id = 'locate-' + uuid.uuid4().hex[:8]
        self._results.pop(request_id, None)
        pil = PILImage.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        self.request_pub.publish(String(json.dumps(
            {'request_id': request_id, 'prompt': prompt, 'grounding': False,
             'image_jpeg_b64': self._jpeg_b64(rgb)}, ensure_ascii=False)))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = self._results.pop(request_id, None)
            if data is not None:
                return '' if data.get('error') else str(data.get('raw_answer') or '')
            time.sleep(0.05)
        rospy.logwarn_throttle(5, 'nine locate timeout: %.20s', prompt)
        return ''

    def _classify_view_crop(self, rgb, bbox, zh_prompt):
        """One crop-classification query: proposal crop (margin, upscaled) ->
        official class name parsed from the raw answer, or None."""
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        mx, my = int((x2 - x1) * 0.6), int((y2 - y1) * 0.6)
        cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        crop = rgb[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return None
        scale = max(2, 160 // max(1, max(ch, cw)))
        up = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
        answer = self._ask(up, zh_prompt)
        low = (answer or '').lower()
        for cls in self.capabilities['declared_classes']:
            if cls.lower() in low:
                return cls
        return None

    def _classify_proposals(self, rgb, props):
        """Three-view majority classification -> {proposal_index: class}.

        Views: full-frame numbered query + per-proposal crop (EN prompt) +
        per-proposal crop (CN prompt, wider margin).  View errors land on
        different objects (measured 10/1: 3/5 each, union 5/5), so a majority
        vote plus full-frame tie-break recovers most of them.  Runs on the
        background scan thread; never blocks detect."""
        ids = [str(i + 1) for i in range(len(props))]
        votes = {i: [] for i in range(len(props))}
        # 视角 A：全图 + 编号清单（与语义选择看到的编号一致）
        names = '、'.join('%s（%s）' % (cls, zh) for zh, cls in QUERY_CLASSES)
        prompt = (u'图中篮筐内有编号%s的物体。请分别判断每个编号物体的物资类别。'
                  u'类别只能是：%s。只输出一个 JSON 对象，形如 {%s}，不要输出其他文字。'
                  % ('、'.join(ids), names, ','.join('"%s":"类名"' % i for i in ids)))
        import re as _re
        raw = self._ask(rgb, prompt)
        numbered = dict(_re.findall(r'"(\d+)"\s*:\s*"([A-Za-z]+)"', raw or ''))
        declared = self.capabilities['declared_classes']
        for i in range(len(props)):
            cls = numbered.get(ids[i])
            if cls in declared:
                votes[i].append(cls)
        # 视角 B/C：单块裁剪（英文提示 / 中文提示）
        for i, p in enumerate(props):
            bbox = [int(v) for v in p['bbox']]
            a = self._classify_view_crop(rgb, bbox,
                u'这个物资是什么类别？只回答类名：Grenade、Magazine、Smokegrenade、Torch。')
            if a:
                votes[i].append(a)
            b = self._classify_view_crop(rgb, bbox,
                u'这个物资是手雷、弹夹、烟雾弹、军用手电筒中的哪一种？只回答类名：Grenade、Magazine、Smokegrenade、Torch。')
            if b:
                votes[i].append(b)
        assign = {}
        for i in range(len(props)):
            vs = votes[i]
            if not vs:
                continue
            best, best_n = None, 0
            for cls in set(vs):
                n = vs.count(cls)
                if n > best_n:
                    best, best_n = cls, n
            if best_n >= 2:
                assign[i] = best
            elif numbered.get(ids[i]) in declared:
                # 无多数（三票各异）：取全图视角（有上下文，实测在无多数场景正确）
                assign[i] = numbered[ids[i]]
            else:
                assign[i] = vs[0]
        return assign

    def _scan_worker(self, rgb, props):
        try:
            light = [{'bbox': [int(v) for v in p['bbox']], 'pixel': [float(v) for v in p['pixel']]} for p in props]
            assign = self._classify_proposals(rgb, light)
        except Exception as error:
            rospy.logwarn('nine scan failed: %s', error)
            assign = {}
        self._cache = {'at': time.monotonic(), 'assign': assign}
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
            threading.Thread(target=self._scan_worker, args=(rgb.copy(), proposals), daemon=True).start()
        assign = {}
        if self._cache is not None and time.monotonic() - self._cache['at'] < CACHE_TTL:
            assign = self._cache['assign']
        # 逐物体分类：类别按扫描时的提案序号直接挂载（编号=发布 id，无匹配环节）
        for index, cls in assign.items():
            if index >= len(proposals):
                continue
            x1, y1, x2, y2 = proposals[index]['bbox']
            row = {'class': cls, 'raw_class': cls, 'confidence': 0.90,
                   'bbox': [float(x1), float(y1), float(x2), float(y2)],
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
