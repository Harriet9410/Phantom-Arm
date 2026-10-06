# -*- coding: utf-8 -*-
"""单图推理：加载 FM9G4b 基座 + 自训 adapter，出 bbox 并画回原图。

手册第 9 章要求五步：加载基座与 adapter -> 调 chat -> 解析 box -> 还原像素 -> 画框。
边界框是结构化任务，评测和比赛都保持 sampling=False、num_beams=1，保证同样输入
能稳定复现，也避免随机采样破坏 <box> 格式。

用法
----
  python3 infer_bbox_lora.py \
      --model-dir /root/inference/FM9G4B-V \
      --adapter-dir /root/sft_v2/outputs/smoke \
      --image /root/sft_v2/raw_images/batch_0000/000000.jpg \
      --query '请框出图中的 烟雾弹。只输出一个 <box>。' \
      --draw-output /tmp/demo_bbox.jpg
"""
import os
import re
import sys
import argparse

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

BOX_RE = re.compile(
    r"<box>\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]?\s*</box>"
)


def parse_box(text):
    m = BOX_RE.search(text or "")
    if not m:
        return None
    box = [float(m.group(i)) for i in range(1, 5)]
    x1, y1, x2, y2 = box
    return box if x1 < x2 and y1 < y2 else None


def to_pixels(box, width, height):
    x1, y1, x2, y2 = [max(0.0, min(1000.0, float(v))) for v in box]
    return [round(x1 / 1000 * width), round(y1 / 1000 * height),
            round(x2 / 1000 * width), round(y2 / 1000 * height)]


def load_model(model_dir, adapter_dir, device):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    base = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16).to(device)
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        return model.get_base_model(), processor
    base.eval()
    return base, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--adapter-dir", default="")
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", default="请框出图中的目标。只输出一个 <box>。")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-slice-nums", type=int, default=9)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--draw-output", default="")
    a = ap.parse_args()

    device = torch.device(a.device)
    model, processor = load_model(a.model_dir, a.adapter_dir, device)
    image = Image.open(a.image).convert("RGB")

    with torch.inference_mode():
        answer = model.chat(
            image=image,
            msgs=[{"role": "user", "content": a.query}],
            tokenizer=processor.tokenizer,
            processor=processor,
            sampling=False,
            max_new_tokens=a.max_new_tokens,
            max_slice_nums=a.max_slice_nums,
            use_image_id=True,
            num_beams=1,
        )

    box = parse_box(answer)
    if box is None:
        print("answer: %s" % answer)
        print("无法解析有效边界框")
        return 1
    pixel = to_pixels(box, *image.size)
    print("answer: %s" % answer)
    print("normalized_box_0_1000: %s" % box)
    print("pixel_box_xyxy: %s" % pixel)
    if a.draw_output:
        os.makedirs(os.path.dirname(a.draw_output) or ".", exist_ok=True)
        ImageDraw.Draw(image).rectangle(pixel, outline="red", width=4)
        image.save(a.draw_output)
        print("draw_output: %s" % a.draw_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
