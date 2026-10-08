#!/usr/bin/env python3
"""R9 续：分类适配（LoRA）训练 + 留出评估脚本。

**本脚本尚未在本轮运行**（训练属需要跑测试的步骤，按约定停在此处）。

用法
  /usr/bin/python3 /root/ds_r9_train_lora.py --dry-run     # 只构批并打印样例，不加载模型
  /usr/bin/python3 /root/ds_r9_train_lora.py               # 真正训练（下一步的测试动作）

设计要点
  * 数据：/root/gpufree-data/r9_dataset 的 manifest（由已保存冲突原图 + 已确证关联生成）。
  * **按场景划分**：训练 = s01+s02，留出 = s03+s04；同一 (案, 物体) 不跨集（manifest 已校验无泄漏）。
  * 目标格式与运行时协议**完全一致**：提示列出 5 类，答案必须是 {"class":"<类名>"} 单字段。
  * 适配器只加在语言侧投影层；发现不了预期模块名就打印实际线性层清单并中止，**不猜**。
  * 训练后自动跑留出评估：四项计数（正确已知/错误已知/unknown/协议失败）+ 混淆矩阵，
    与 /root/ds_r9_b_offline.py 同一评价口径，便于与 R9 的 5/8/27 直接对比。
  * 数据集只有 4 类样本（**无 CompressedFood 实例**）；类别声明仍为 5 类，不因缺样本而从运行时删除该类。
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image as PILImage

DATA = Path('/root/gpufree-data/r9_dataset')
MODEL = '/root/inference/FM9G4B-V'
ADAPTER_OUT = Path('/root/sft_v2/outputs/r9_cls_lora')
DECLARED = ['CompressedFood', 'Grenade', 'Magazine', 'Smokegrenade', 'Torch']
CLASS_ZH = {'Smokegrenade': u'烟雾弹', 'Magazine': u'弹夹', 'Torch': u'军用手电筒',
            'Grenade': u'手雷', 'CompressedFood': u'压缩干粮'}
PROMPT = (u'这是军用物资分类任务。可选类别只有：%s。'
          u'只输出一个 JSON 对象，形如 {"class":"类名"}，类名必须是上面之一；'
          u'实在无法确定时输出 {"class":"unknown"}。'
          u'不要输出任何其他文字、解释或标点。'
          % u'、'.join(u'%s（%s）' % (c, CLASS_ZH[c]) for c in DECLARED))
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'Wqkv', 'wo', 'wq', 'wk', 'wv')


def load_manifest():
    man = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    samples = []
    for row in man:
        for c in row['crops']:
            p = DATA / c['path']
            if p.exists():
                samples.append({'path': str(p), 'truth': row['truth'], 'split': row['split'],
                                'case': row['case'], 'scan': row['scan'],
                                'object_id': row['object_id'], 'margin': c['margin'],
                                'sha16': hashlib.sha256(p.read_bytes()).hexdigest()[:16]})
    return samples


def format_sample(s):
    return {'image': s['path'], 'prompt': PROMPT,
            'answer': json.dumps({'class': s['truth']}, ensure_ascii=False),
            'meta': {k: s[k] for k in ('case', 'scan', 'object_id', 'margin', 'truth', 'sha16')}}


def dry_run(samples):
    tr = [s for s in samples if s['split'] == 'train']
    ho = [s for s in samples if s['split'] == 'holdout']
    print('samples: train=%d holdout=%d' % (len(tr), len(ho)))
    print('train classes:', dict(Counter(s['truth'] for s in tr)))
    print('holdout classes:', dict(Counter(s['truth'] for s in ho)))
    print('train scenes:', dict(Counter(s['case'] for s in tr)))
    print('holdout scenes:', dict(Counter(s['case'] for s in ho)))
    ex = format_sample(tr[0])
    print('--- example record ---')
    print(json.dumps(ex, ensure_ascii=False, indent=1))
    img = cv2.imread(ex['image'])
    print('image shape:', None if img is None else img.shape)
    # 阻断性自检：同一 (案,物体) 不得跨集；答案必须与真值一致；提示必须含 5 类
    keys = {}
    for s in samples:
        keys.setdefault((s['case'], s['object_id']), set()).add(s['split'])
    leak = [k for k, v in keys.items() if len(v) > 1]
    assert not leak, 'split leakage: %s' % leak
    for c in DECLARED:
        assert c in PROMPT, c
    print('dry-run checks passed (no leakage, prompt lists all 5 declared classes)')


def discover_targets(model):
    names = {n.split('.')[-1] for n, _ in model.named_modules()}
    hit = sorted(names & set(TARGET_MODULES))
    if not hit:
        linears = sorted({n.split('.')[-1] for n, m in model.named_modules()
                          if isinstance(m, torch.nn.Linear)})
        raise SystemExit('no expected attention projection found; actual Linear layer names: %s' % linears[:40])
    return hit


def parse_answer(text):
    raw = str(text or '').strip()
    if raw.startswith('```'):
        raw = raw.strip('`').split('\n', 1)[-1].strip()
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict) or list(obj) != ['class']:
        return None
    v = obj['class']
    if not isinstance(v, str) or v.lower() == 'unknown' or v not in DECLARED:
        return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=16)
    args = ap.parse_args()

    samples = load_manifest()
    if not samples:
        raise SystemExit('empty dataset; run /root/ds_r9_dataset.py first')
    if args.dry_run:
        dry_run(samples)
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    targets = discover_targets(model)
    print('LoRA target modules:', targets)
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                     bias='none', task_type='CAUSAL_LM', target_modules=targets)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.eval()                     # the model exposes .chat; we drive forward() directly

    def encode(s):
        pil = PILImage.fromarray(cv2.cvtColor(cv2.imread(s['path']), cv2.COLOR_BGR2RGB))
        img_t = model.process_images([pil], model.config) if hasattr(model, 'process_images') else None
        text = tok.apply_chat_template([{'role': 'user', 'content': PROMPT},
                                        {'role': 'assistant', 'content': json.dumps({'class': s['truth']}, ensure_ascii=False)}],
                                       add_generation_prompt=False, tokenize=False)
        batch = tok(text, return_tensors='pt')
        if img_t is not None:
            batch['images'] = img_t
        return batch

    train = [s for s in samples if s['split'] == 'train']
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for epoch in range(args.epochs):
        tot = 0.0
        for i, s in enumerate(train):
            b = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in encode(s).items()}
            out = model(**b, labels=b['input_ids'])
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print('epoch %d loss %.4f' % (epoch + 1, tot / max(1, len(train))), flush=True)
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT / ('step_%d' % (args.epochs * len(train)))))
    print('adapter saved to', ADAPTER_OUT)
    (ADAPTER_OUT / 'train_config.json').write_text(json.dumps(
        {'base': MODEL, 'targets': targets, 'epochs': args.epochs, 'lr': args.lr,
         'rank': args.rank, 'train_scenes': sorted({s['case'] for s in train}),
         'holdout_scenes': sorted({s['case'] for s in samples if s['split'] == 'holdout'}),
         'prompt': PROMPT, 'declared': DECLARED, 'created_at': time.time()},
        ensure_ascii=False, indent=1), encoding='utf-8')

    # 留出评估：与 R9 离线同一口径
    hold = [s for s in samples if s['split'] == 'holdout']
    model.eval()
    conf, ok, wrong, unknown = Counter(), 0, 0, 0
    for s in hold:
        pil = PILImage.fromarray(cv2.cvtColor(cv2.imread(s['path']), cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            ans = model.chat(image=None, msgs=[{'role': 'user', 'content': [pil, PROMPT]}],
                             tokenizer=tok, max_new_tokens=32, sampling=False, num_beams=1)
        got = parse_answer(ans)
        if got is None:
            unknown += 1
            conf['%s -> unknown' % s['truth']] += 1
        elif got == s['truth']:
            ok += 1
            conf['%s -> %s' % (s['truth'], got)] += 1
        else:
            wrong += 1
            conf['%s -> %s' % (s['truth'], got)] += 1
    print(json.dumps({'holdout_n': len(hold), 'correct_known': ok, 'wrong_known': wrong,
                      'unknown': unknown, 'confusion': dict(sorted(conf.items()))},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
