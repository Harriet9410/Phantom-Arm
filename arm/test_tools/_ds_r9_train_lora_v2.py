#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9-P0 修正版：FM9G4B-V 分类 LoRA 训练 + 留出评估（**本轮不运行**）。

本文件按 R9《验收与 V0 收敛决定》§"训练脚本的阻断项"五项重写；接口全部依据
`/root/inference/FM9G4B-V` 的真实源码取证（见 R9-1 报告 §2），不复用旧脚本的猜测。

取证到的关键接口事实（源码位置）
  * 模型是 InternVL 系 `FM9GV`（`modeling_fm9gv.py`）：
      - `FM9GV.forward(self, data, **kwargs)` → 构造 `data` 后调用
        `self.llm(input_ids=None, position_ids=..., inputs_embeds=vllm_embedding, **kwargs)`
      - `FM9GV.get_vllm_embedding(data)` 读取 `data['input_ids'] / data['image_bound'] /
        data['pixel_values'] / data['tgt_sizes']`（或 `data['vision_hidden_states']`）
      - `FM9GV.chat()` 的正确前置是 **processor 一次性产出** 全部输入：
        `inputs = processor(prompts, images, return_tensors='pt', max_length=...)`，
        再 `inputs.pop('image_sizes')` 后交给网络。**不是** `tokenizer(text)` + `batch['images']`。
  * 图像占位：文本占位串 `"(<image>./</image>)"`，由
    `image_processor.get_slice_image_placeholder(...)` 展开
    （`im_start='<image>'`、`im_end='</image>'`、`image_feature_size=64`、`use_image_id=true`、
    `max_slice_nums=9`、`scale_resolution=448`、`patch_size=14`）。
  * `processing_fm9gv.py::_convert_images_texts_to_inputs` 产出并返回：
    `input_ids`（**padding_side='left'**）、`attention_mask`（bool，左侧 pad=False）、
    `pixel_values`、`image_sizes`、`image_bound`（每图 [start,end) 的 token 区间）、`tgt_sizes`。
  * 视觉侧（**必须排除在 LoRA 之外**）：`vpm`（SiglipVisionTransformer）、`resampler`。
  * LLM 侧注意力为 **MLA**（`modeling_fm9g.py::FM9GAttention`）：
    `q_a_proj`、`q_b_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`、`o_proj`；
    MLP：`gate_proj`、`up_proj`、`down_proj`。**不存在** `q_proj/k_proj/v_proj/Wqkv`。
    另需排除 `lm_head`、`score`。

用法
  python3 ds_r9_train_lora_v2.py --dry-run          # 不加载任何模型组件：数据/划分/提示自检
  python3 ds_r9_train_lora_v2.py --check-labels     # 只加载 tokenizer+processor（不加载模型）：
                                                    # 核对真实 prompt 展开与答案 token 跨
  python3 ds_r9_train_lora_v2.py --smoke            # R10：单样本真实前向+反向（冒烟）
  python3 ds_r9_train_lora_v2.py --train            # R10：完整训练
  python3 ds_r9_train_lora_v2.py --eval-holdout     # R10：留出评估（冻结口径）

本脚本**默认不训练**（必须显式给 --train）。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path('/root/gpufree-data/r9_dataset')
MODEL = '/root/inference/FM9G4B-V'
ADAPTER_OUT = Path('/root/sft_v2/outputs/r9_cls_lora_v2')
OLD_ADAPTER = '/root/sft_v2/outputs/qlora_crop/step_27000'   # 旧定位适配器（只读，不覆盖）
PROTOCOL = Path('/root/ds_r9_eval_protocol_v2.json')      # P0-3 冻结的评估协议

# ---- V0 口径：四类（用户明确 V0 不考虑 CompressedFood）----
V0_CLASSES = ['Grenade', 'Magazine', 'Smokegrenade', 'Torch']
CLASS_ZH = {'Smokegrenade': u'烟雾弹', 'Magazine': u'弹夹', 'Torch': u'军用手电筒',
            'Grenade': u'手雷', 'CompressedFood': u'压缩干粮'}
# 五类历史声明：仅用于「旧结果对照」，不用于 V0 训练/评估口径
DECLARED_5 = ['CompressedFood', 'Grenade', 'Magazine', 'Smokegrenade', 'Torch']

# ---- LoRA 目标：按源码取证的 MLA + MLP 名（只做语言侧）----
TARGET_ATTN = ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj']
TARGET_MLP = ['gate_proj', 'up_proj', 'down_proj']
TARGET_MODULES = TARGET_ATTN + TARGET_MLP
# 必须排除的路径片段（视觉侧 / 输出头）
EXCLUDE_PATHS = ('vpm.', 'resampler.', 'vision_model.', 'mlp1.', 'lm_head', 'score')
# 语言侧路径前缀（LoRA 只允许命中这些）
LLM_PREFIX = 'llm.'

SPLIT_TRAIN = ('scramble_01', 'scramble_02')     # 冻结划分
SPLIT_HOLDOUT = ('scramble_03', 'scramble_04')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def build_prompt(classes=V0_CLASSES):
    """V0 四类提示；与运行时分类分支使用的解析口径配套（见冻结协议）。"""
    return (u'这是军用物资分类任务。可选类别只有：%s。'
            u'只输出一个 JSON 对象，形如 {"class":"类名"}，类名必须是上面之一；'
            u'实在无法确定时输出 {"class":"unknown"}。'
            u'不要输出任何其他文字、解释或标点。'
            % u'、'.join(u'%s（%s）' % (c, CLASS_ZH[c]) for c in classes))


PROMPT = build_prompt()


def load_manifest():
    man = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for row in man:
        for c in row['crops']:
            p = DATA / c['path']
            if p.exists():
                rows.append({'path': str(p), 'truth': row['truth'], 'split': row['split'],
                             'case': row['case'], 'scan': row['scan'], 'object_id': row['object_id'],
                             'margin': c['margin'], 'sha16': sha256(p)[:16]})
    return rows


# ---------------------------------------------------------------- P0-1 台账
def data_ledger(rows):
    """四层计数（场景 / 物体 ID / 原图=观测 / 裁剪），每层同时给观测级与裁剪级。"""
    scenes = sorted({r['case'] for r in rows})
    obs_key = lambda r: (r['case'], r['scan'], r['object_id'])
    img_key = lambda r: (r['case'], r['scan'])
    obs, imgs = {obs_key(r) for r in rows}, {img_key(r) for r in rows}
    obj_ids = sorted({r['object_id'] for r in rows})
    by_obj_obs = {o: len({obs_key(r) for r in rows if r['object_id'] == o}) for o in obj_ids}
    by_obj_crops = {o: sum(1 for r in rows if r['object_id'] == o) for o in obj_ids}
    shared = {o: sorted({r['case'] for r in rows if r['object_id'] == o}) for o in obj_ids
              if len({r['case'] for r in rows if r['object_id'] == o}) > 1}
    cross = sorted(o for o in obj_ids
                   if {r['case'] for r in rows if r['object_id'] == o} & set(SPLIT_TRAIN)
                   and {r['case'] for r in rows if r['object_id'] == o} & set(SPLIT_HOLDOUT))
    margins = Counter(r['margin'] for r in rows)
    return {
        'layer1_scenes': {'n': len(scenes), 'ids': scenes,
                          'obs_per_scene': {s: len({obs_key(r) for r in rows if r['case'] == s}) for s in scenes},
                          'crops_per_scene': {s: sum(1 for r in rows if r['case'] == s) for s in scenes}},
        'layer2_object_ids': {'n': len(obj_ids), 'ids': obj_ids,
                              'obs_per_id': by_obj_obs, 'crops_per_id': by_obj_crops,
                              'ids_spanning_scenes': shared,
                              'ids_shared_train_and_holdout': cross},
        'layer3_observations': {'n': len(obs),
                                'per_scene': {s: len({obs_key(r) for r in rows if r['case'] == s}) for s in scenes},
                                'definition': '(case, scan, object_id) 唯一计数'},
        'layer4_crops': {'n': len(rows), 'per_margin': dict(sorted(margins.items())),
                         'per_scene': {s: sum(1 for r in rows if r['case'] == s) for s in scenes},
                         'per_split': {'train': sum(1 for r in rows if r['split'] == 'train'),
                                       'holdout': sum(1 for r in rows if r['split'] == 'holdout')}},
        'note_object_identity_leak': (
            '场景划分无泄漏（同一案只属一个划分）；但**物体身份划分不成立**：'
            'layer2.ids_shared_train_and_holdout 中的 5 个 object_id 同时出现在训练与留出场景。'
            '留出集只验证「同一资产换摆位」，不验证陌生实物。'),
        'note_two_smoke_bombs': (
            '场景有五件实物，其中 smoke_bomb 与 smoke_bomb_01 都是**同一个 Smokegrenade 类别的两个实例**，'
            '不是两个类别。'),
    }


def class_counts(rows):
    """类别分布：分别给观测级与裁剪级，避免把裁剪数当成观测数。"""
    out = {}
    for sp in ('train', 'holdout'):
        sub = [r for r in rows if r['split'] == sp]
        oc = Counter(r['truth'] for r in sub)                       # 裁剪级
        observations = {(r['case'], r['scan'], r['object_id']): r['truth'] for r in sub}
        oo = Counter(observations.values())                         # 观测级
        assert set(oc) == set(oo), (set(oc) ^ set(oo))
        out[sp] = {'n_observations': len(observations), 'n_crops': len(sub),
                   'class_observations': dict(sorted(oo.items())),
                   'class_crops': dict(sorted(oc.items())),
                   'v0_classes_present': sorted(set(oc) & set(V0_CLASSES)),
                   'non_v0_classes': sorted(set(oc) - set(V0_CLASSES))}
    return out


# ---------------------------------------------------------------- P0-2 编码
def make_processor():
    """只加载 tokenizer + image processor（**不加载模型权重**）。"""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)


def _to_msgs_text(processor, image, prompt, answer):
    """复刻 chat()：PIL 图像 → 占位串，再 apply_chat_template。"""
    from PIL import Image as PILImage
    content = [image, prompt] if answer is None else [image, prompt]
    msgs = [{'role': 'user', 'content': content}]
    if answer is not None:
        msgs.append({'role': 'assistant', 'content': answer})
    copy = []
    for m in msgs:
        parts = []
        for c in (m['content'] if isinstance(m['content'], list) else [m['content']]):
            parts.append('(<image>./</image>)' if isinstance(c, PILImage.Image) else c)
        copy.append({'role': m['role'], 'content': '\n'.join(parts)})
    return copy


def _process(processor, image_path, prompt, answer, max_length=8192):
    from PIL import Image as PILImage
    pil = PILImage.open(image_path).convert('RGB')
    msgs = _to_msgs_text(processor, pil, prompt, answer)
    text = processor.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=(answer is None))
    d = dict(processor([text], [[pil]], return_tensors='pt', max_length=max_length))
    d.pop('image_sizes', None)
    return d


def answer_fmt(truth_class):
    """训练/评估共用的答案串（与运行时协议一致：单键 JSON）。"""
    return json.dumps({'class': truth_class}, ensure_ascii=False)


def answer_span(processor, image_path, prompt, truth_class):
    """定位答案 token 跨：用「仅提示（add_generation_prompt=True）」的 token 长度作为起点。

    模板为 `<user>…</user><assistant>ANSWER<|im_end|>`；仅提示版本以 `<assistant>` 结尾，
    因此 `answer_start = len(prompt_only_ids)`。带图时占位符展开在两版一致，长度可比。
    终点由从起点起解码首次出现 `}` 的 token 决定，从而只覆盖 `{"class":"…"}` 本体。
    自检：被保留区段解码必须**逐字等于**答案串。
    """
    wanted = answer_fmt(truth_class)
    d_full = _process(processor, image_path, prompt, wanted)
    d_prompt = _process(processor, image_path, prompt, None)          # add_generation_prompt=True
    n_prompt = int(d_prompt['attention_mask'][0].bool().sum())
    ids = d_full['input_ids'][0].tolist()
    n_real = int(d_full['attention_mask'][0].bool().sum())
    k = n_prompt
    tk = processor.tokenizer
    end = None
    for j in range(k, n_real):
        if '}' in tk.decode(ids[k:j + 1]):
            end = j + 1
            break
    if end is None:
        end = n_real
    kept = tk.decode(ids[k:end])
    return {'answer_start': k, 'answer_end': end, 'n_real': n_real,
            'kept_text': kept, 'expected': wanted,
            'kept_equals_expected': kept.strip() == wanted.strip(),
            'n_kept': end - k, 'n_prompt_tokens': n_prompt}, d_full


def encode_batch(processor, image_path, prompt, truth_class=None, max_length=8192):
    """返回 processor 输出 + 答案 token 跨。

    `truth_class` 传**类别名**（如 'Smokegrenade'）；None 表示只构提示（评估前向用）。
    """
    if truth_class is None:
        return _process(processor, image_path, prompt, None, max_length), None
    span, d = answer_span(processor, image_path, prompt, truth_class)
    return d, span


def build_labels(input_ids, attention_mask, span):
    """只对助手答案 token `{"class":"..."}` 计损失；提示/图像占位/padding/终止符置 -100。"""
    import torch
    labels = torch.full_like(input_ids, -100)
    a, b = span['answer_start'], span['answer_end']
    assert 0 <= a < b <= span['n_real'], span
    labels[0, a:b] = input_ids[0, a:b]
    kept = int((labels != -100).sum())
    assert kept == span['n_kept'] and kept > 0, ('masked span mismatch', kept, span)
    return labels


def position_ids_from(attention_mask):
    """InternVL 惯例：左 padding，position 从 0 连续编号。"""
    import torch
    pos = attention_mask.long().cumsum(-1) - 1
    pos = pos.masked_fill(attention_mask == 0, 1)
    return pos


def make_data_dict(inputs_d, with_labels=True):
    """构造 FM9GV.forward(data, **kwargs) 需要的 data（+ 可选 labels）。"""
    d = {
        'input_ids': inputs_d['input_ids'],
        'image_bound': inputs_d['image_bound'],
        'position_ids': position_ids_from(inputs_d['attention_mask']),
        # get_vllm_embedding(data) 从 data 读视觉输入
        'pixel_values': inputs_d['pixel_values'],
        'tgt_sizes': inputs_d['tgt_sizes'],
    }
    extra = {'attention_mask': inputs_d['attention_mask']}
    if with_labels:
        span = inputs_d['_span']
        extra['labels'] = build_labels(inputs_d['input_ids'], inputs_d['attention_mask'], span)
    return d, extra


# ---------------------------------------------------------------- 目标模块
def discover_targets(model):
    """按**完整路径**筛选；打印并固定命中路径；排除视觉侧与输出头。"""
    import torch
    hits, excluded = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if name.split('.')[-1] not in TARGET_MODULES:
            continue
        if any(x in name for x in EXCLUDE_PATHS) or not name.startswith(LLM_PREFIX):
            excluded.append(name)
        else:
            hits.append(name)
    if not hits:
        linears = sorted(n for n, m in model.named_modules()
                         if isinstance(m, torch.nn.Linear))
        raise SystemExit('no LLM-side MLA/MLP projection matched under %r; '
                         'actual Linear paths: %s' % (LLM_PREFIX, linears[:40]))
    return sorted(hits), sorted(excluded)


def assert_vision_excluded(model, target_paths):
    """证据：命中集合必须全部在 llm.* 且不含 vpm/resampler。"""
    bad = [p for p in target_paths
           if not p.startswith(LLM_PREFIX) or any(x in p for x in EXCLUDE_PATHS)]
    assert not bad, 'vision/head module leaked into LoRA targets: %s' % bad
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    leak = [n for n in trainable if any(x in n for x in EXCLUDE_PATHS)]
    assert not leak, 'trainable params leak into vision/head: %s' % leak[:10]
    return {'n_trainable_tensors': len(trainable),
            'sample': trainable[:8],
            'all_under_llm': all(n.startswith(LLM_PREFIX) for n in trainable)}


# ---------------------------------------------------------------- 解析/评估
class Parse:
    """把模型输出分成四类：正确已知 / 错误已知 / 显式 unknown / **协议失败**。"""
    OK, WRONG, UNKNOWN, PROTO = 'correct_known', 'wrong_known', 'unknown_explicit', 'protocol_failure'


def parse_strict(text, allowed=V0_CLASSES):
    raw = str(text or '').strip()
    if raw.startswith('```'):
        raw = raw.strip('`').split('\n', 1)[-1].strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return Parse.PROTO, None
    if not isinstance(obj, dict) or list(obj) != ['class']:
        return Parse.PROTO, None                      # 额外字段/缺字段 → 协议失败
    v = obj['class']
    if not isinstance(v, str):
        return Parse.PROTO, None
    if v.lower() == 'unknown':
        return Parse.UNKNOWN, None                    # 显式 unknown 单独计数
    if v not in allowed:
        return Parse.PROTO, None                      # 越界类名 → 协议失败（不是错认）
    return Parse.OK, v


def score(rows_eval, answers, allowed=V0_CLASSES):
    conf = Counter()
    agg = defaultdict(lambda: {'n': 0, Parse.OK: 0, Parse.WRONG: 0, Parse.UNKNOWN: 0, Parse.PROTO: 0})
    for r, ans in zip(rows_eval, answers):
        kind, got = parse_strict(ans, allowed)
        if kind == Parse.OK and got != r['truth']:
            kind = Parse.WRONG
        conf['%s -> %s' % (r['truth'], got if got else '<' + kind + '>')] += 1
        a = agg[r['object_id']]
        a['n'] += 1
        a[kind] += 1
    return {'n': len(rows_eval), 'confusion': dict(sorted(conf.items())),
            'per_object': {k: v for k, v in sorted(agg.items())}}


# ---------------------------------------------------------------- 模式
def mode_dry_run(rows):
    led = data_ledger(rows)
    cc = class_counts(rows)
    out = {'prompt': PROMPT, 'v0_classes': V0_CLASSES,
           'declared_5_history_only': DECLARED_5,
           'lora_targets': TARGET_MODULES, 'exclude_paths': EXCLUDE_PATHS,
           'ledger': led, 'classes': cc,
           'dataset_dir': str(DATA), 'dataset_manifest_sha256':
               sha256(DATA / 'manifest.json') if (DATA / 'manifest.json').exists() else None}
    # 冻结划分自检：训练/留出场景集合必须与常量逐字一致
    assert set(SPLIT_TRAIN) | set(SPLIT_HOLDOUT) == set(led['layer1_scenes']['ids']), led['layer1_scenes']
    assert (led['layer3_observations']['n'], led['layer4_crops']['n']) == (145, 259), 'dataset size drifted'
    for sp, b in cc.items():
        assert not b['non_v0_classes'], 'non-V0 class present in %s: %s' % (sp, b['non_v0_classes'])
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print('DRY-RUN OK: ledger + splits consistent; dataset 145 obs / 259 crops; no model component loaded.')


def mode_check_labels(rows, limit=2):
    """只加载 tokenizer+processor，核对真实 prompt 展开与答案 token 跨（不加载模型）。"""
    proc = make_processor()
    rep = []
    for r in rows[:limit]:
        d, span = encode_batch(proc, r['path'], PROMPT, r['truth'])
        rep.append({'path': os.path.basename(r['path']), 'truth': r['truth'],
                    'seq_len': int(d['input_ids'].shape[1]),
                    'n_real_tokens': span['n_real'],
                    'image_token_span': [int(x) for x in (d['image_bound'][0][0].tolist()
                                                          if len(d['image_bound'][0]) else [])],
                    'answer_start': span['answer_start'], 'answer_end': span['answer_end'],
                    'n_kept_tokens': span['n_kept'], 'kept_text': span['kept_text'],
                    'expected': span['expected'],
                    'kept_equals_expected': span['kept_equals_expected']})
    print(json.dumps({'note': 'tokenizer+processor only; model NOT loaded', 'samples': rep},
                     ensure_ascii=False, indent=1))


def mode_smoke(rows):
    """R10 单样本真实多模态前向+反向；换图必须改变计算（R9 阻断项 2 的证明）。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model.to(dev)                                    # 阻断项 1：显式设备
    paths, excl = discover_targets(model)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
                     task_type='CAUSAL_LM', target_modules=[p.split('.')[-1] for p in paths])
    model = get_peft_model(model, cfg)
    model.train()
    ev = assert_vision_excluded(model, paths)
    proc = make_processor()
    r = rows[0]
    d, span = encode_batch(proc, r['path'], PROMPT, r['truth'])
    d['_span'] = span
    data, extra = make_data_dict(d)
    data = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in data.items()}
    extra = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in extra.items()}
    out = model(data=data, **extra)
    loss = out.loss if hasattr(out, 'loss') else out[0]
    loss.backward()
    n_updated = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None
                    and float(p.grad.abs().sum()) > 0)
    # 换图不变文本：logits 必须变化（否则图像未进入计算）
    d2, span2 = encode_batch(proc, rows[1]['path'], PROMPT, rows[1]['truth'])
    d2['_span'] = span2
    data2, extra2 = make_data_dict(d2)
    data2 = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in data2.items()}
    extra2 = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in extra2.items()}
    with torch.no_grad():
        out2 = model(data=data2, **extra2)
    l2 = out2.loss if hasattr(out2, 'loss') else out2[0]
    print(json.dumps({'device': dev, 'loss_sample0': float(loss), 'loss_sample1_same_text': float(l2),
                      'loss_differs_with_other_image': abs(float(loss) - float(l2)) > 1e-6,
                      'n_trainable_tensors_with_grad': n_updated,
                      'targets': paths, 'excluded_matched': excl, 'trainable_audit': ev,
                      'answer_span': span}, ensure_ascii=False, indent=1))
    print('SMOKE OK' if n_updated > 0 and abs(float(loss) - float(l2)) > 1e-6 else 'SMOKE FAILED')


def mode_train(rows, args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model.to(dev)
    paths, excl = discover_targets(model)
    print('LoRA target paths (%d):' % len(paths))
    for p in paths:
        print('   ', p)
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias='none',
                     task_type='CAUSAL_LM', target_modules=[p.split('.')[-1] for p in paths])
    model = get_peft_model(model, cfg)
    audit = assert_vision_excluded(model, paths)
    model.print_trainable_parameters()
    model.train()
    proc = make_processor()
    train = [r for r in rows if r['split'] == 'train']
    # 类别加权（烟雾弹约占 42%）
    w = Counter(r['truth'] for r in train)
    weights = torch.tensor([max(w.values()) / w[r['truth']] for r in train], dtype=torch.float32).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for i, r in enumerate(train):
            d, span = encode_batch(proc, r['path'], PROMPT, r['truth'])
            d['_span'] = span
            data, extra = make_data_dict(d)
            data = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in data.items()}
            extra = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in extra.items()}
            out = model(data=data, **extra)
            loss = (out.loss if hasattr(out, 'loss') else out[0]) * weights[i]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1; step += 1
        print('epoch %d loss %.4f' % (ep + 1, tot / max(1, n)), flush=True)
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    ckpt = ADAPTER_OUT / ('step_%d' % step)
    model.save_pretrained(str(ckpt))
    cfgout = {'base': MODEL, 'base_is_new_lora_on_base_model': True,
              'not_continuing_old_adapter': '/root/sft_v2/outputs/qlora_crop/step_27000',
              'target_paths': paths, 'trainable_audit': audit, 'epochs': args.epochs,
              'lr': args.lr, 'rank': args.rank, 'v0_classes': V0_CLASSES,
              'prompt': PROMPT, 'train_scenes': list(SPLIT_TRAIN),
              'holdout_scenes': list(SPLIT_HOLDOUT),
              'protocol': str(PROTOCOL), 'created_at': time.time()}
    (ckpt / 'train_config.json').write_text(json.dumps(cfgout, ensure_ascii=False, indent=1),
                                            encoding='utf-8')
    print('adapter saved', ckpt)
    print('checkpoint=' + ckpt.name)


def mode_eval(rows, adapter=None, baseline_old=None):
    """冻结口径的留出评估。

    - adapter=None 且 baseline_old=None：纯基座（无任何适配器）
    - baseline_old=<旧 adapter 目录>：**复现在线路径**——包装旧适配器后**禁用它**
      （peft 的 disable_adapter 上下文），等价于运行时的 `grounding=False`。
    - adapter=<新 adapter>：启用新分类适配器（R10 通过门后才可接入在线）。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModel
    from PIL import Image as PILImage
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proto = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model.to(dev)
    state = 'base_only'
    if baseline_old:
        model = PeftModel.from_pretrained(model, baseline_old)
        state = 'old_adapter_wrapped_and_disabled'
    elif adapter:
        model = PeftModel.from_pretrained(model, adapter)
        state = 'new_adapter_ENABLED'
    model.eval()
    proc = make_processor()
    hold = [r for r in rows if r['split'] == 'holdout']
    answers = []
    ctx = model.disable_adapter() if state == 'old_adapter_wrapped_and_disabled' else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for r in hold:
            pil = PILImage.open(r['path']).convert('RGB')
            ans = model.chat(image=None, msgs=[{'role': 'user', 'content': [pil, PROMPT]}],
                             tokenizer=proc.tokenizer, max_new_tokens=proto['max_new_tokens'],
                             sampling=False, num_beams=proto['num_beams'])
            answers.append(ans)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    res = score(hold, answers, proto['classes'])
    res.update({'adapter_state': state, 'old_adapter': baseline_old, 'new_adapter': adapter,
                'classes': proto['classes'], 'protocol_sha256': sha256(PROTOCOL)})
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check-labels', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--eval-holdout', action='store_true')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--baseline', action='store_true',
                    help='R10 训练前：复现在线旧路径（包装旧适配器并禁用）的四类基线')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=16)
    args = ap.parse_args()
    rows = load_manifest()
    if not rows:
        raise SystemExit('empty dataset at %s' % DATA)
    if args.dry_run:
        mode_dry_run(rows)
    elif args.check_labels:
        mode_check_labels(rows)
    elif args.smoke:
        mode_smoke(rows)
    elif args.train:
        mode_train(rows, args)
    elif args.eval_holdout:
        mode_eval(rows, args.adapter,
                  baseline_old=OLD_ADAPTER if args.baseline else None)
    else:
        print('no mode given; see --help. Default is to do NOTHING (training is opt-in).')


if __name__ == '__main__':
    main()
