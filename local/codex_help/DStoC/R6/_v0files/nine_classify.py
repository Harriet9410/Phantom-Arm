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
import os
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
# 后台自动续扫间隔：类别持续套用不回落，靠周期刷新保持新鲜（身份确认依赖连续性）。
RESCAN_INTERVAL = 60.0
# 类别↔物体绑定（10/4 根因修复）：扫描结果按【物体几何】记忆与匹配，绝不按候选下标。
# 依据：下标 = 每帧按 bbox.x 重排的序号，抓走一件就整体偏移、剩余物体左右顺序还会
# 翻转（实测同一批物体 t=1737 编号 {1:烟雾弹,2:手雷} → t=1860 {1:手雷,2:烟雾弹}）；
# 且扫描异步（60s 周期/按需 20-40s 落地）。按下标挂类别会把旧编号的类名贴到新编号的
# 物体上——实测"弹夹已被抓走 5 分钟后，新候选仍带 Magazine 标签"，模型据此拒绝执行。
ASSOC_MAX_CENTRE_PX = 20.0   # 静止画面中同一物体的中心位移远小于此值；物体间距≥40px
CACHE_MAX_AGE = 150.0        # 单条记忆的最长保鲜期：超过即不再用于贴类别
RETAIN_SECONDS = 90.0        # 本轮未观测到的记忆保留时长（空/残缺扫描不得清空记忆）
LABEL_RESCAN_AFTER = 10.0    # 真实物体覆盖不足且缓存超龄这么久：立刻补扫（不等 60s 周期）
REAL_BOX_MIN_AREA_PX = 500.  # 真实物体的最小成像面积（碎片/遮挡残片低于此值）
FULL_TTL = 300.0             # 一次"完整鉴定"（裁剪+配对复核+四选一）的有效期。静止物体的
                             # 类别不会变，期内只要全图视角与记忆一致就跳过重活；期满自动
                             # 重跑一次完整鉴定，保证错误类别最多存活这么长时间


def _centre(box):
    return ((float(box[0]) + float(box[2])) / 2., (float(box[1]) + float(box[3])) / 2.)


def _numbered_view(rgb, props):
    """Mark the proposal numbers used by the full-frame classification prompt."""
    view = rgb.copy()
    height, width = view.shape[:2]
    for index, prop in enumerate(props, 1):
        x1, y1, x2, y2 = [int(v) for v in prop['bbox']]
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 255), 2)
        position = (x1, y1 - 6 if y1 >= 30 else min(height - 5, y2 + 25))
        cv2.putText(view, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4)
        cv2.putText(view, str(index), position, cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    return view


def _is_real_box(box):
    try:
        return abs(float(box[2]) - float(box[0])) * abs(float(box[3]) - float(box[1])) >= REAL_BOX_MIN_AREA_PX
    except (TypeError, ValueError, IndexError):
        return False


def _associate(proposals, objects, max_centre=ASSOC_MAX_CENTRE_PX, now=None, max_age=CACHE_MAX_AGE):
    """One-to-one nearest-match from current proposals to remembered objects.

    Returns {proposal_index: class}.  A remembered object that no longer matches
    anything simply drops out, and a proposal with no remembered partner stays
    unknown -- so a class can never migrate onto a different object.  Stale
    entries (older than max_age) are never used."""
    if now is None:
        now = time.monotonic()
    pairs = []
    for pi, proposal in enumerate(proposals):
        pc = _centre(proposal['bbox'])
        for oi, obj in enumerate(objects):
            if now - obj.get('seen_at', 0.) > max_age:
                continue
            oc = _centre(obj['bbox'])
            distance = ((pc[0] - oc[0]) ** 2 + (pc[1] - oc[1]) ** 2) ** .5
            if distance <= max_centre:
                pairs.append((distance, pi, oi))
    pairs.sort(key=lambda row: row[0])
    used_proposal, used_object, matched = set(), set(), {}
    for _, pi, oi in pairs:
        if pi in used_proposal or oi in used_object:
            continue
        used_proposal.add(pi); used_object.add(oi)
        matched[pi] = objects[oi]['class']
    return matched


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
        # Unanimity fast path (default on, `~classify_fastpath:=false` restores the
        # strict mode): the final four-way adjudication costs two model calls per
        # object (~40% of a scan).  When the full-frame double vote and both crop
        # views already agree, that re-check only confirms what every view said.
        self.fastpath = bool(rospy.get_param('~classify_fastpath', True))
        self.request_pub = rospy.Publisher('/tcei/classify_request', String, queue_size=1)
        self.result_sub = rospy.Subscriber('/tcei/classify_result', String,
                                           self._on_result, queue_size=4)
        self._results = {}
        self._cache = None
        self._scan_inflight = False
        # 物体级类别记忆（几何键）；旧实现是下标键的 _stable_assign，见文件头说明
        self._objects = []
        log_dir = rospy.get_param('~log_dir', '')
        self._evidence_dir = os.path.join(log_dir, 'classification_conflicts') if log_dir else None
        self._conflict_captures = 0
        self._empty_streak = 0
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

    def _ask(self, rgb, prompt, tag='crop'):
        """Image + prompt -> raw model answer via the nine_node service.

        grounding:false 始终走基座（选择权重）模型——分类/定位查询都不加载
        框选适配器（实测基座的类别判断远准于适配器）。tag 只用于日志阶段
        标记（R2-0 要求）：nine_node 会把它原样带回 classify_answered，使
        每一票可按阶段回放。"""
        request_id = 'locate-' + uuid.uuid4().hex[:8]
        self._results.pop(request_id, None)
        pil = PILImage.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        self.request_pub.publish(String(json.dumps(
            {'request_id': request_id, 'prompt': prompt, 'grounding': False, 'tag': tag,
             'image_jpeg_b64': self._jpeg_b64(rgb)}, ensure_ascii=False)))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = self._results.pop(request_id, None)
            if data is not None:
                answer = '' if data.get('error') else str(data.get('raw_answer') or '')
                break
            time.sleep(0.05)
        else:
            rospy.logwarn_throttle(5, 'nine locate timeout: %.20s', prompt)
            answer = ''
        # 连续空答 = 分类服务尚未就绪（九格模型 8.5G 加载期、或服务重启）。
        # 立刻中止本轮扫描，而不是让 5 个物体 × 3 视角逐个耗满超时：一次残缺
        # 扫描可能拖几分钟，还会堵住补扫（_scan_inflight）。中止后 10s 内自适应
        # 补扫重试，模型就绪即成功。
        self._empty_streak = self._empty_streak + 1 if not answer.strip() else 0
        if self._empty_streak >= 3:
            raise RuntimeError('nine classify unavailable (3 consecutive empty answers)')
        return answer

    def _parse_class_answer(self, text):
        """M1（R2-1）：裁剪答案的类别解析——中英别名 + 冲突/否定保守判不确定。

        别名单一来源 = perception_tracking.CLASS_ALIASES（不在此重复定义）。
        规则：
        - 英文子串沿用长名优先：'Grenade' 在 'Smokegrenade' 内部不算第二类；
        - 中文别名做子串匹配（别名表里的中文项）；
        - 否定词（前后 6 字窗内）就近否决该类名：被否决的类弃权，剩余恰好
          一类才返回（'不是手雷，是烟雾弹'→Smokegrenade；'这不是手雷'→None）
          ——永不返回被否决的类；
        - 其余情形命中 ≠1 类（含 0 类与真冲突）→ None（不确定票）。"""
        if not text:
            return None
        raw = str(text)
        low = raw.lower()
        declared = self.capabilities.get('declared_classes') or ()
        spans = []
        for cls in sorted(declared, key=len, reverse=True):
            for m in re.finditer(re.escape(cls.lower()), low):
                spans.append([m.start(), m.end(), cls])
        for canonical, aliases in CLASS_ALIASES.items():
            if canonical not in declared:
                continue
            for alias in aliases:
                if not any('\u4e00' <= ch <= '\u9fff' for ch in alias):
                    continue                      # 英文别名已由 declared 分支覆盖
                for m in re.finditer(re.escape(alias), raw):
                    spans.append([m.start(), m.end(), canonical])
        if not spans:
            return None
        negations = ('不是', '并不是', '并非', '不算', '没有', '非', 'not', 'non-')
        hits = []
        for neg in negations:
            for m in re.finditer(re.escape(neg.lower()), low):
                hits.append((m.start(), m.end()))
        kept = []
        for s in spans:
            if any(o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0])
                   for o in spans):
                continue                          # 丢弃被更长匹配包含的子串
            kept.append(s)
        # 否定词就近否决：绑定与它间隔最近的类名跨度（并列才同标），
        # 被否决的类弃权——'不是手雷，是烟雾弹' 只弃权手雷。
        for ns, ne in hits:
            best, best_gap = [], None
            for s in kept:
                gap = max(0, s[0] - ne) + max(0, ns - s[1])
                if best_gap is None or gap < best_gap:
                    best, best_gap = [id(s)], gap
                elif gap == best_gap:
                    best.append(id(s))
            if best_gap is not None and best_gap <= 6:
                kept = [s for s in kept if id(s) not in best]
        classes = {s[2] for s in kept}
        if len(classes) != 1:
            return None
        return kept[0][2]

    def _classify_view_crop(self, rgb, bbox, zh_prompt, tag='crop'):
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
        answer = self._ask(up, zh_prompt, tag=tag)
        return self._parse_class_answer(answer)

    def _classify_proposals(self, rgb, props, settled=None, scan_id=''):
        """Three-view majority classification -> ({proposal_index: class}, cheap).

        Views: full-frame numbered query + per-proposal crop (EN prompt) +
        per-proposal crop (CN prompt, wider margin).  View errors land on
        different objects (measured 10/1: 3/5 each, union 5/5), so a majority
        vote plus full-frame tie-break recovers most of them.  Runs on the
        background scan thread; never blocks detect.

        ``settled`` maps a proposal index to a class already established for that
        object by a *full* battery within FULL_TTL.  A stationary object's class
        does not need re-deriving on every 60 s scan, and the per-object battery
        costs ~3-6 model calls that queue ahead of planning requests on a
        single-threaded model (measured: 180 classification answers inside one
        240 s round).  When the cheap full-frame view agrees with the settled
        class, the battery is skipped; when it disagrees, the full battery runs
        and the usual two-round hysteresis decides, so a wrong class is still
        correctable.  Returns the indices that took the cheap path so the caller
        can keep their verification timestamp current."""
        settled = settled or {}
        cheap = set()
        ids = [str(i + 1) for i in range(len(props))]
        votes = {i: [] for i in range(len(props))}
        # 视角 A：按本次 proposals 顺序在全图画编号；原相机帧没有这些编号。
        names = '、'.join('%s（%s）' % (cls, zh) for zh, cls in QUERY_CLASSES)
        prompt = (u'图中篮筐内有编号%s的物体。请分别判断每个编号物体的物资类别。'
                  u'类别只能是：%s。只输出一个 JSON 对象，形如 {%s}，不要输出其他文字。'
                  % ('、'.join(ids), names, ','.join('"%s":"类名"' % i for i in ids)))
        import re as _re
        raw = self._ask(_numbered_view(rgb, props), prompt, tag='%s:full' % scan_id)
        numbered = dict(_re.findall(r'"(\d+)"\s*:\s*"([A-Za-z]+)"', raw or ''))
        declared = self.capabilities['declared_classes']
        for i in range(len(props)):
            cls = numbered.get(ids[i])
            if cls in declared:
                # 全图视角计双票：有全局上下文，作为类别锚（裁剪视角在扫描间摆动时
                # 2:1 会翻转多数，双票后无多数回落全图，类别稳定）
                votes[i].append(cls)
                votes[i].append(cls)
        # 视角 B/C：单块裁剪（英文提示 / 中文提示）
        for i, p in enumerate(props):
            hint = settled.get(i)
            if hint and numbered.get(ids[i]) == hint:
                # 已完整鉴定过，且全图锚点与记忆一致：四票同值直接采用，
                # 省下 2 次裁剪 + 后续配对复核/四选一（每物体 3~6 次调用）
                votes[i] = [hint, hint, hint, hint]
                cheap.add(i)
                continue
            bbox = [int(v) for v in p['bbox']]
            a = self._classify_view_crop(rgb, bbox,
                u'这个物资是什么类别？只回答类名：Grenade、Magazine、Smokegrenade、Torch。',
                tag='%s:p%d:crop_en' % (scan_id, i + 1))
            if a:
                votes[i].append(a)
            b = self._classify_view_crop(rgb, bbox,
                u'这个物资是手雷、弹夹、烟雾弹、军用手电筒中的哪一种？只回答类名：Grenade、Magazine、Smokegrenade、Torch。',
                tag='%s:p%d:crop_zh' % (scan_id, i + 1))
            if b:
                votes[i].append(b)
        assign = {}
        for i in range(len(props)):
            vs = votes[i]
            if not vs:
                continue
            anchor = numbered.get(ids[i])
            best, best_n = None, 0
            for cls in sorted(set(vs)):          # 排序：set 迭代序随进程哈希变化，
                n = vs.count(cls)                # 平票时会让结果在不同进程间漂移
                if n > best_n or (n == best_n and cls == anchor):
                    best, best_n = cls, n        # 平票取全图锚点（有全局上下文且被双票加权）
            if best_n >= 2:
                assign[i] = best
            elif anchor in declared:
                # 无多数（三票各异）：取全图视角（有上下文，实测在无多数场景正确）
                assign[i] = anchor
            else:
                assign[i] = vs[0]
        # Grenade/Smokegrenade 俯视混淆严重（三视角 2/5 错误集中于此）：加特征
        # 提示的二选一复核，实测 6/6 一致，直接覆写复核结果。
        def _recheck_agree(bbox, prompt, tag='recheck'):
            """复核采两票：两票一致才改判。单票在模糊裁剪上是掷硬币，
            会引发类别振荡（实测 id2 Torch↔Grenade 反复横跳）。"""
            answers = []
            for _ in range(2):
                a = self._classify_view_crop(rgb, bbox, prompt, tag=tag)
                if a is not None:
                    answers.append(a)
            return answers[0] if len(answers) == 2 and answers[0] == answers[1] else None

        GS_PROMPT = (u'仔细看这个物体的形状：烟雾弹（Smokegrenade）是圆柱形容器，'
                     u'常带绿色环带；手雷（Grenade）是小型椭球体。'
                     u'这个物体是哪一类？只回答 Smokegrenade 或 Grenade。')
        for i, cls in list(assign.items()):
            if i in cheap:
                continue                     # 已完整鉴定过：无需再复核
            if cls in ('Grenade', 'Smokegrenade') and i < len(props):
                ans = _recheck_agree([int(v) for v in props[i]['bbox']], GS_PROMPT,
                                     tag='%s:p%d:recheck_gs' % (scan_id, i + 1))
                if ans in ('Grenade', 'Smokegrenade'):
                    assign[i] = ans
        # 手雷/手电筒混淆（实机 id5 被锁 Torch）：同一手法二次复核。
        GT_PROMPT = (u'仔细看这个物体的形状：手雷（Grenade）是小型椭球体，表面常有'
                     u'网格状防滑纹；军用手电筒（Torch）是细长圆柱形，一端有尾盖或按钮。'
                     u'这个物体是哪一类？只回答 Grenade 或 Torch。')
        for i, cls in list(assign.items()):
            if i in cheap:
                continue                     # 已完整鉴定过：无需再复核
            if cls in ('Grenade', 'Torch') and i < len(props):
                ans = _recheck_agree([int(v) for v in props[i]['bbox']], GT_PROMPT,
                                     tag='%s:p%d:recheck_gt' % (scan_id, i + 1))
                if ans in ('Grenade', 'Torch'):
                    assign[i] = ans
        # 终审（10/3 修订）：配对复核（GS/GT）只覆盖两对且无逃生口，跨类错误与
        # 非对内物体（实测：首扫 Magazine→Grenade 两票一致地错；第二颗烟雾弹被
        # M↔G 对强制误判）都兜不住。最终裁决由特征描述拉齐的强四选一承担：
        # 各类描述等强、置于所有配对复核之后、两票一致才改判。
        FIVE_PROMPT = (u'这是军用物资，四选一：Smokegrenade（烟雾弹，圆柱形容器，'
                       u'常带绿色环带）、Grenade（手雷，小型椭球体，整体圆润带网格状'
                       u'防滑纹）、Torch（军用手电筒，细长圆柱形，一端有尾盖或按钮）、'
                       u'Magazine（弹夹，扁平长条形弹匣，一侧平直，常可见排列的弹壳'
                       u'或供弹口）。这个物体是哪一类？只回答类名。')
        for i, cls in list(assign.items()):
            if i in cheap:
                continue                     # 已完整鉴定过：无需再复核
            if i < len(props):
                votes_i = votes.get(i, [])
                if self.fastpath and len(votes_i) >= 4 and len(set(votes_i)) == 1:
                    continue          # all views agreed; skip the two re-check calls
                ans = _recheck_agree([int(v) for v in props[i]['bbox']], FIVE_PROMPT,
                                     tag='%s:p%d:final' % (scan_id, i + 1))
                if ans in declared and ans != cls:
                    assign[i] = ans
        return assign, cheap

    def _scan_worker(self, rgb, props):
        scan_id = uuid.uuid4().hex[:10]
        light = [{'bbox': [int(v) for v in p['bbox']], 'pixel': [float(v) for v in p['pixel']]} for p in props]
        # 把"已完整鉴定且在有效期内"的记忆类别作为提示交给分类器，让全图视角一致的
        # 物体走廉价路径（见 _classify_proposals 文档）。几何匹配与发布路径同一把尺子。
        began = time.monotonic()
        settled, hint_used = {}, set()
        for pi, prop in enumerate(light):
            centre = _centre(prop['bbox'])
            best_index, best_distance = None, None
            for mi, mem in enumerate(self._objects):
                if mi in hint_used or not mem.get('full') or not mem.get('class'):
                    continue
                if began - mem.get('verified_at', 0.) > FULL_TTL:
                    continue
                mem_centre = _centre(mem['bbox'])
                distance = ((centre[0] - mem_centre[0]) ** 2 + (centre[1] - mem_centre[1]) ** 2) ** .5
                if distance <= ASSOC_MAX_CENTRE_PX and (best_distance is None or distance < best_distance):
                    best_index, best_distance = mi, distance
            if best_index is not None:
                hint_used.add(best_index)
                settled[pi] = self._objects[best_index]['class']
        try:
            assign, cheap = self._classify_proposals(rgb, light, settled=settled, scan_id=scan_id)
        except Exception as error:
            rospy.logwarn('nine scan failed: %s', error)
            assign, cheap = {}, set()
        # 物体级类别记忆：把本轮观测按几何对到既有对象上，类名只跟着物体走。
        # 滞回规则保持"连续两轮一致才改判"，但比对发生在【同一物体】之间，
        # 而不是旧实现的"同一下标"之间——下标会随重排漂移，物体不会。
        observed = [(i, {'bbox': light[i]['bbox'], 'pixel': light[i]['pixel'], 'class': assign[i],
                     'full': i not in cheap})
                    for i in sorted(assign) if 0 <= i < len(light) and assign.get(i)]
        now = time.monotonic()
        used_memory = set()
        merged = []
        scan_audit = []
        for proposal_index, obs in observed:
            centre = _centre(obs['bbox'])
            best_index, best_distance = None, None
            for mi, mem in enumerate(self._objects):
                if mi in used_memory:
                    continue
                mem_centre = _centre(mem['bbox'])
                distance = ((centre[0] - mem_centre[0]) ** 2 + (centre[1] - mem_centre[1]) ** 2) ** .5
                if distance <= ASSOC_MAX_CENTRE_PX and (best_distance is None or distance < best_distance):
                    best_index, best_distance = mi, distance
            if best_index is None:
                # 新物体：本轮扫描结果（只有走了完整鉴定才算"已鉴定"）
                merged.append(dict(obs, seen_at=now, full=bool(obs['full']),
                                   verified_at=now if obs['full'] else 0.))
                scan_audit.append({'scan_id':scan_id,'proposal_index':proposal_index + 1,
                                   'pixel':obs['pixel'],'scan_class':obs['class'],
                                   'old_class':None,'stored_class':obs['class'],
                                   'full':bool(obs['full']),'decision':'new'})
                continue
            used_memory.add(best_index)
            previous = self._objects[best_index]
            if previous['class'] == obs['class']:
                # 一致 -> 采用；廉价路径不刷新鉴定时刻，完整鉴定才刷新
                merged.append(dict(obs, seen_at=now,
                                   full=previous.get('full') or obs['full'],
                                   verified_at=now if obs['full'] else previous.get('verified_at', 0.)))
                decision='same'
            elif previous.get('pending') == obs['class']:
                # 第二轮确认 -> 采用新类别；它需要自己的完整鉴定
                merged.append(dict(obs, seen_at=now, full=bool(obs['full']),
                                   verified_at=now if obs['full'] else 0.))
                decision='confirmed_reclass'
            else:
                kept = dict(previous)
                kept['pending'] = obs['class']               # 单轮翻转：保留旧类名等确认
                kept['bbox'] = obs['bbox']; kept['pixel'] = obs['pixel']; kept['seen_at'] = now
                merged.append(kept)
                decision='pending_conflict'
            scan_audit.append({'scan_id':scan_id,'proposal_index':proposal_index + 1,
                               'pixel':obs['pixel'],'scan_class':obs['class'],
                               'old_class':previous['class'],'old_pending':previous.get('pending'),
                               'stored_class':merged[-1]['class'],'full':bool(obs['full']),
                               'decision':decision})
        # 关键：本轮没观测到的记忆【保留】一段时间。
        # 扫描会被模型加载/超时打断而产出为空；若像初版那样"整体替换"，
        # 一次空扫描就会清空全部类别 -> 所有候选变 unknown -> 计划必被拒。
        for mi, mem in enumerate(self._objects):
            if mi in used_memory:
                continue
            if now - mem.get('seen_at', 0.) <= RETAIN_SECONDS:
                merged.append(mem)
        self._objects = merged
        self._cache = {'at': now, 'objects': [dict(o) for o in merged]}
        if (self._evidence_dir and self._conflict_captures < 24 and
                any(row['decision'] in ('pending_conflict', 'confirmed_reclass') for row in scan_audit)):
            try:
                os.makedirs(self._evidence_dir, exist_ok=True)
                saved = []
                for kind, frame in (('raw', rgb), ('numbered', _numbered_view(rgb, light))):
                    name = '%s_%s.jpg' % (scan_id, kind)
                    path = os.path.join(self._evidence_dir, name)
                    if not cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                        raise IOError('could not save ' + path)
                    saved.append(name)
                self._conflict_captures += 1
                for row in scan_audit:
                    row['evidence_images'] = saved
                rospy.loginfo('nine scan conflict images: %s', json.dumps({'scan_id':scan_id,
                    'directory':self._evidence_dir,'images':saved,'proposals':light}))
            except Exception as error:
                rospy.logwarn('nine scan evidence capture failed: %s', error)
        rospy.loginfo('nine scan class audit: %s', json.dumps(scan_audit,ensure_ascii=False))
        rospy.loginfo('nine scan: classified %d/%d proposals, memory %d (settled %d, cheap %d)',
                      len(observed), len(light), len(merged), len(settled), len(cheap))
        self._scan_inflight = False

    def detect(self, rgb, depth, k):
        began = time.monotonic()
        proposals, support, metadata = source_components(depth, k, return_metadata=True)
        found = [{} for _ in proposals]
        base = []
        # 真实物体的匹配覆盖数决定是否需要补扫：模型加载期间/超时导致的残缺扫描
        # 会让部分候选长期没有类别，仅在 60s 周期里等会让计划一直拿不到标签。
        real_indices = [i for i, proposal in enumerate(proposals) if _is_real_box(proposal['bbox'])]
        matched = {}
        if self._cache is not None:
            matched = _associate(proposals, self._cache['objects'])
        covered = sum(1 for i in matched if i in set(real_indices))
        now = time.monotonic()
        cache_age = None if self._cache is None else now - self._cache['at']
        poor_coverage = bool(real_indices) and covered < len(real_indices)
        if (not self._scan_inflight and
                (self.pending_scan or
                 (cache_age is None and bool(real_indices)) or
                 (poor_coverage and cache_age is not None and cache_age > LABEL_RESCAN_AFTER) or
                 (cache_age is not None and cache_age > RESCAN_INTERVAL))):
            self.pending_scan = False
            self._scan_inflight = True
            threading.Thread(target=self._scan_worker, args=(rgb.copy(), proposals), daemon=True).start()
        # 几何匹配挂载：只有与记忆中【同一位置】的物体配对成功，才继承它的类别；
        # 配对失败（新物体/被遮挡/位置变化）一律发 unknown，等新一轮扫描。
        for index, cls in matched.items():
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
