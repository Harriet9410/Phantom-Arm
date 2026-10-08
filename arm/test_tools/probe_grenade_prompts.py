#!/usr/bin/env python3
"""Controlled comparison for the "grenade read as smoke bomb" failure (review §六 step 2).

Neutral evidence change only: the SAME object, the SAME runtime prompts, and either the
crop the runtime actually fed the model or a larger/upscaled crop — plus optionally a
two-object image for a purely neutral two-class question.  No prompt ever states which
class is expected, so the model's output stays an independent measurement.

Fidelity to the runtime (nine_node.on_classify_request + nine_classify._ask):
  * model: AutoModel.from_pretrained(FM9G4B-V, trust_remote_code, sdpa, bfloat16)
  * the grounding LoRA adapter is LOADED and DISABLED (classification never uses it)
  * model.chat(image=None, msgs=[{role:user, content:[PIL, prompt]}],
               max_new_tokens=400, sampling=False, num_beams=1)   ← launch defaults
  * num_beams=1 ⇒ greedy ⇒ the same input MUST give the same answer; variant 1 is run
    twice to demonstrate exactly that (why "ask it again the same way" cannot help).

Run AFTER the baseline batch has finished (the model is single-threaded and GPU-shared).

Usage:
  /opt/conda/envs/inference/bin/python probe_grenade_prompts.py \
      --current <crop_as_seen.png> --wide <larger_crop.png> [--pair <two_object.png>] \
      [--expect Grenade] [--out result.json]
"""
import argparse
import json
import pathlib
import sys
import time

MODEL_PATH = '/root/inference/FM9G4B-V'
ADAPTER = '/root/sft_v2/outputs/qlora_crop/step_27000'

# --- prompts copied verbatim from tcei_stack/nine_classify.py -------------------
CN_CROP = u'这个物资是什么类别？只回答类名：Grenade、Magazine、Smokegrenade、Torch。'
GS = (u'仔细看这个物体的形状：烟雾弹（Smokegrenade）是圆柱形容器，'
      u'常带绿色环带；手雷（Grenade）是小型椭球体。'
      u'这个物体是哪一类？只回答 Smokegrenade 或 Grenade。')
FIVE = (u'这是军用物资，四选一：Smokegrenade（烟雾弹，圆柱形容器，'
        u'常带绿色环带）、Grenade（手雷，小型椭球体，整体圆润带网格状'
        u'防滑纹）、Torch（军用手电筒，细长圆柱形，一端有尾盖或按钮）、'
        u'Magazine（弹夹，扁平长条形弹匣，一侧平直，常可见排列的弹壳'
        u'或供弹口）。这个物体是哪一类？只回答类名。')
PAIR_NEUTRAL = (u'图中左右各有一件物体。请分别判断它们的类别，'
                u'类别只能是 Grenade、Magazine、Smokegrenade、Torch。'
                u'只输出两行：左边=类名，右边=类名。')
PAIR_ELIM = (u'图中左右各有一件物体，它们不是同一类。'
             u'其中一件不是烟雾弹（Smokegrenade）。请指出哪一件不是烟雾弹，并给出它的类别。'
             u'只输出：左边/右边 + 类名。')

CLASSES = ('Smokegrenade', 'Grenade', 'Torch', 'Magazine')


def parse_class(answer):
    low = str(answer or '').lower()
    for cls in sorted(CLASSES, key=len, reverse=True):
        if cls.lower() in low:
            return cls
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--current', required=True, help='crop exactly as the runtime fed it')
    ap.add_argument('--wide', required=True, help='larger/upscaled crop of the same object')
    ap.add_argument('--pair', help='optional two-object image (neutral comparison)')
    ap.add_argument('--expect', default='', help='expected class, for reporting ONLY (never sent)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    began = time.time()
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                     attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().to('cuda')
    model.config.vision_batch_size = 1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    load_seconds = time.time() - began

    def ask(path, prompt):
        image = Image.open(path).convert('RGB')
        with model.disable_adapter():                     # classification never uses the adapter
            answer = model.chat(image=None, msgs=[{'role': 'user', 'content': [image, prompt]}],
                                tokenizer=tokenizer, max_new_tokens=400, sampling=False, num_beams=1)
        return str(answer)

    plan = [
        ('1_当前裁剪+GS提示（现状，第1次）', args.current, GS),
        ('1b_当前裁剪+GS提示（现状，第2次·验证确定性）', args.current, GS),
        ('2_当前裁剪+中文裁剪提示（现状问法2）', args.current, CN_CROP),
        ('3_当前裁剪+四选一提示（现状问法3）', args.current, FIVE),
        ('4_放大的裁剪+GS提示（换证据）', args.wide, GS),
        ('5_放大的裁剪+四选一提示（换证据）', args.wide, FIVE),
    ]
    if args.pair:
        plan.append(('6_两件同图+中性两问（换问法）', args.pair, PAIR_NEUTRAL))
        plan.append(('7_两件同图+排除问法（换问法·半引导，仅参考）', args.pair, PAIR_ELIM))

    rows = []
    for label, path, prompt in plan:
        try:
            raw = ask(path, prompt)
            parsed = parse_class(raw)
        except Exception as error:                        # keep going: one failure is data
            raw, parsed = 'ERROR: ' + str(error)[:200], None
        rows.append({'variant': label, 'image': pathlib.Path(path).name, 'prompt': prompt,
                     'raw': raw, 'parsed': parsed})
        print('%-46s -> %-14s | %s' % (label, parsed, raw.replace('\n', ' ')[:110]))

    print()
    print('模型加载 %.1fs | 期望类别（仅记录，未进提示）: %s' % (load_seconds, args.expect))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {'model': MODEL_PATH, 'adapter_disabled': True, 'num_beams': 1,
             'expect_reporting_only': args.expect, 'load_seconds': round(load_seconds, 1),
             'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('written:', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
