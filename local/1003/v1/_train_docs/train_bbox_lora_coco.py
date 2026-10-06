# -*- coding: utf-8 -*-
"""FM9G4b 边界框 QLoRA 微调（COCO + RefCOCO 混合）

严格按《九格大模型的量化和训练》第 5 章实现，关键点逐条对齐：
  · 任务是"给定图 + 一条指令 -> 一个 bbox"，框写成文本，不是检测头微调
  · 答案固定格式 <box>[x1,y1,x2,y2]</box>，坐标为 0-1000 归一化
  · prompt 里必须保留 (<image>./</image>) 占位符，Processor 靠它插视觉 token
  · FM9G4b 用**左填充**，算 prompt_len 时必须计入 pad_len
  · 4-bit 量化要**跳过 Resampler**（它的 MultiheadAttention 不能吃 bnb 权重），并保持 BF16
  · LoRA 目标层是 MLA 专用名，不能照搬 LLaMA 的 q_proj/k_proj/v_proj
  · 只对非 -100 的目标 token 求 loss

先用 --max-samples 64 冒烟，确认 4-bit 能加载、前后向能走、日志无 nan/inf，
再改大跑正式训练。显存不够时按手册给的顺序降：先确认 4bit+checkpointing，
再降 max_slice_nums（9->4->1），再降 max_length（2048->1024）。

用法
----
  bash run_bbox_lora.sh                    # 见启动脚本
  # 或直接
  torchrun --standalone --nproc_per_node=1 train_bbox_lora_coco.py \
      --model-dir /root/inference/FM9G4B-V \
      --coco-root /root/sft_v2/converted/coco \
      --output-dir /root/sft_v2/outputs/smoke --max-samples 64 ...
"""
import os
import sys
import json
import math
import random
import argparse
import dataclasses
from dataclasses import dataclass, asdict, field

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from transformers import AutoProcessor, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import BitsAndBytesConfig

# 手册 4.6「随机使用多种问句模板」。目标名由 category 填入。
PROMPT_TEMPLATES = [
    "请框出图中的 {name}。只输出一个 <box>。",
    "框出图中的 {name}，只输出一个 <box>。",
    "在图中定位 {name}，只输出一个 <box>。",
    "请找出图中的 {name} 并用 <box> 标出。",
    "图中 {name} 在哪里？只输出一个 <box>。",
]


def normalize_box_xywh(box, width, height):
    """COCO 像素 xywh -> 0-1000 归一化 xyxy（带截断）"""
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(width, x))
    y1 = max(0.0, min(height, y))
    x2 = max(0.0, min(width, x + w))
    y2 = max(0.0, min(height, y + h))
    return [round(x1 / width * 1000), round(y1 / height * 1000),
            round(x2 / width * 1000), round(y2 / height * 1000)]


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_dir: str = "."
    coco_root: str = ""
    refcoco_root: str = ""
    refcoco_datasets: str = "refcoco"
    refcoco_splits: str = "train"
    output_dir: str = ""
    init_adapter_dir: str = ""
    split: str = "train"
    max_samples: int = 0
    refcoco_max_samples: int = 0
    min_area: int = 256
    seed: int = 42
    batch_size: int = 1
    grad_accum: int = 4
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    # 必须是 0：处理器的输出是远程代码里的自定义类（FM9GVBatchFeature），
    # DataLoader 子进程回传时无法 pickle，num_workers>0 会直接报 PicklingError。
    num_workers: int = 0
    max_length: int = 2048
    max_slice_nums: int = 9
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    log_every: int = 10
    save_every: int = 500
    load_in_4bit: bool = False
    gradient_checkpointing: bool = False
    train_resampler: bool = False


def parse_args():
    d = TrainConfig()
    ap = argparse.ArgumentParser(description="FM9G4b bbox QLoRA")
    for f in dataclasses.fields(d):
        name = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(name, action="store_true", default=f.default)
        else:
            ap.add_argument(name, type=type(f.default), default=f.default)
    cfg = TrainConfig(**{k: v for k, v in vars(ap.parse_args()).items()})
    if not cfg.coco_root:
        ap.error("--coco-root 必填")
    if not cfg.output_dir:
        ap.error("--output-dir 必填")
    return cfg


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
class CocoBBoxDataset(Dataset):
    """类别级定位。会丢弃「同图同类别多实例」的样本 —— 只给类别名时无法确定要哪个框。"""

    def __init__(self, root, split, min_area=256, max_samples=0, seed=0):
        self.root = root
        self.items = []
        ann_path = os.path.join(root, "annotations", "instances_%s2014.json" % split)
        data = json.loads(open(ann_path, encoding="utf-8").read())
        images = {i["id"]: i for i in data["images"]}
        cats = {c["id"]: c["name"] for c in data["categories"]}
        per_img_cat = {}
        for ann in data["annotations"]:
            per_img_cat.setdefault((ann["image_id"], ann["category_id"]), 0)
            per_img_cat[(ann["image_id"], ann["category_id"])] += 1
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w * h < min_area:
                continue
            if per_img_cat[(ann["image_id"], ann["category_id"])] > 1:
                continue                      # 歧义样本，交给 RefCOCO 侧
            im = images[ann["image_id"]]
            self.items.append({
                "image_path": os.path.join(root, split + "2014", im["file_name"]),
                "width": im["width"], "height": im["height"],
                "category": cats[ann["category_id"]],
                "box": normalize_box_xywh(ann["bbox"], im["width"], im["height"]),
            })
        random.Random(seed).shuffle(self.items)
        if max_samples > 0:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        prompt = random.choice(PROMPT_TEMPLATES).format(name=s["category"])
        box = ",".join(str(v) for v in s["box"])
        return {"image": image, "prompt": prompt, "answer": "<box>[%s]</box>" % box}


class RefCocoBBoxDataset(Dataset):
    """指代级定位：expression 唯一指向一个框。多目标同框图靠这里保住。"""

    def __init__(self, root, split, max_samples=0, seed=0):
        path = os.path.join(root, "refcoco_%s.json" % split)
        recs = json.loads(open(path, encoding="utf-8").read())
        self.root = root
        self.items = []
        for r in recs:
            objs = r.get("objects") or []
            if len(objs) != 1:
                continue
            x1, y1, x2, y2 = [float(v) for v in objs[0]["bbox"]]
            if not (x1 < x2 and y1 < y2):
                continue
            img = r["image"].replace("\\", "/")
            self.items.append({
                "image_path": os.path.join(root, img),
                "expression": r.get("expression") or "",
                "box_xyxy": [x1, y1, x2, y2],
            })
        random.Random(seed).shuffle(self.items)
        if max_samples > 0:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        W, H = image.size
        x1, y1, x2, y2 = s["box_xyxy"]
        box = [round(max(0.0, min(W, x1)) / W * 1000), round(max(0.0, min(H, y1)) / H * 1000),
               round(max(0.0, min(W, x2)) / W * 1000), round(max(0.0, min(H, y2)) / H * 1000)]
        box = [max(0, min(1000, v)) for v in box]
        return {"image": image,
                "prompt": "请框出图中的 %s。只输出一个 <box>。" % s["expression"],
                "answer": "<box>[%s]</box>" % ",".join(str(v) for v in box)}


class MixView(Dataset):
    """把若干个数据集的 __getitem__ 拼起来。"""

    def __init__(self, datasets):
        self.datasets = datasets
        self.index = []
        for di, d in enumerate(datasets):
            for i in range(len(d)):
                self.index.append((di, i))
        random.Random(0).shuffle(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        di, i = self.index[idx]
        return self.datasets[di][i]


# --------------------------------------------------------------------------
# Collator：对话、图像、答案标签
# --------------------------------------------------------------------------
class BBoxCollator:
    def __init__(self, processor, cfg):
        self.processor = processor
        self.cfg = cfg
        self.tokenizer = processor.tokenizer

    def __call__(self, batch):
        images, texts, prompts = [], [], []
        for item in batch:
            user_content = "(<image>./</image>)\n" + item["prompt"]
            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = prompt_messages + [{"role": "assistant", "content": item["answer"]}]
            texts.append(self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False))
            prompts.append(self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True))
            images.append(item["image"])

        inputs = self.processor(
            texts, images,
            max_slice_nums=self.cfg.max_slice_nums,
            use_image_id=True,
            return_tensors="pt",
            max_length=self.cfg.max_length,
        )
        labels = inputs["input_ids"].clone().long()
        labels[~inputs["attention_mask"]] = -100
        for i, ptext in enumerate(prompts):
            pin = self.processor(
                ptext, images[i],
                max_slice_nums=self.cfg.max_slice_nums,
                use_image_id=True,
                return_tensors="pt",
                max_length=self.cfg.max_length,
            )
            prompt_len = int(pin["attention_mask"][0].sum())
            full_len = int(inputs["attention_mask"][i].sum())
            pad_len = labels.shape[1] - full_len      # FM9G4b 左填充，必须计入
            labels[i, :pad_len + prompt_len] = -100
        position_ids = inputs["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(~inputs["attention_mask"], 0)
        inputs["labels"] = labels
        inputs["position_ids"] = position_ids
        inputs.pop("image_sizes", None)
        return inputs


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
def build_model(cfg, device):
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            # 关键：这两个模块**不能**量化。
            #   resampler —— 内部 MultiheadAttention 吃不了 bnb 权重（手册已写明）
            #   lm_head   —— 量化后 bnb 会在 forward 里断言 module.weight.shape[1] == 1
            #                失败（bitsandbytes/nn/modules.py 的 fix_4bit_weight_quant_state），
            #                手册只写了跳过 resampler，但输出层同样必须跳过。
            llm_int8_skip_modules=["lm_head", "resampler"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if quant is not None:
        kwargs["quantization_config"] = quant
        # 这里**不要**传 device_map。手册写的是 device_map={"": device.index}，
        # 那是按 accelerate 0.33 写的；新版 accelerate（本机 1.2.1）遇到单设备映射
        # 会走 model.to(device) 的捷径，而 bitsandbytes 的 4-bit 模型明确禁止 .to()，
        # 会直接抛 "`.to` is not supported for `4-bit` ... models"。
        # 正确做法：先把当前 CUDA 设备切过去，让 bnb 自己把权重放到这张卡上。
        if torch.cuda.is_available():
            torch.cuda.set_device(device.index)
    model = AutoModel.from_pretrained(cfg.model_dir, **kwargs)
    if quant is None:
        model = model.to(device)
    model.config.use_cache = False

    if cfg.init_adapter_dir:
        model = PeftModel.from_pretrained(model, cfg.init_adapter_dir, is_trainable=True)
    else:
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing)
        elif cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if hasattr(model, "resampler"):
            model.resampler.to(dtype=torch.bfloat16)
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=[                            # FM9G4b 的 MLA 专用层名
                "q_a_proj", "q_b_proj",
                "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=(["resampler"] if cfg.train_resampler else None),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    # FM9GV.forward 把 input_ids / position_ids 硬写死后再把 **kwargs 透传给内部 LLM。
    # 量化模型会挂 accelerate 的 AlignDevicesHook，它按签名重建参数时会把 batch 里的
    # input_ids 重新塞进 kwargs，于是内部 LLM 收到重复的 input_ids 而报：
    #   TypeError: FM9GForCausalLM(...) got multiple values for keyword argument 'input_ids'
    # 这里把这三个键从 kwargs 里摘掉 —— 它们本来就该由 FM9GV 依据 data 自己生成，
    # 不该从外部传进来（input_ids 恒为 None，模型走的是 inputs_embeds）。
    _base = model.get_base_model() if hasattr(model, "get_base_model") else model
    _orig_fwd = type(_base).forward

    def _safe_fwd(self, data, **kwargs):
        for _k in ("input_ids", "position_ids", "inputs_embeds"):
            kwargs.pop(_k, None)
        return _orig_fwd(self, data, **kwargs)

    type(_base).forward = _safe_fwd
    model.print_trainable_parameters()
    return model


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def save_checkpoint(model, processor, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    m.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def main():
    cfg = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank if torch.cuda.is_available() else 0)
    is_main = local_rank == 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    print("模型目录：%s" % cfg.model_dir)
    print("COCO 根目录：%s" % cfg.coco_root)
    print("输出目录：%s" % cfg.output_dir)
    print("GPU：%s 号卡；4bit=%s；grad_ckpt=%s；train_resampler=%s"
          % (local_rank, cfg.load_in_4bit, cfg.gradient_checkpointing, cfg.train_resampler))

    processor = AutoProcessor.from_pretrained(cfg.model_dir, trust_remote_code=True)

    dsets = [CocoBBoxDataset(cfg.coco_root, cfg.split, cfg.min_area,
                             cfg.max_samples, cfg.seed)]
    if cfg.refcoco_root:
        for name in [x for x in cfg.refcoco_datasets.split(",") if x]:
            for sp in [x for x in cfg.refcoco_splits.split(",") if x]:
                p = os.path.join(cfg.refcoco_root, "refcoco_%s.json" % sp)
                if os.path.exists(p):
                    dsets.append(RefCocoBBoxDataset(cfg.refcoco_root, sp,
                                                    cfg.refcoco_max_samples, cfg.seed))
    dataset = MixView(dsets)
    print("Training samples: %d" % len(dataset))
    for d in dsets:
        print("  %-10s %d" % (type(d).__name__, len(d)))
    if len(dataset) == 0:
        raise SystemExit("没有训练样本")

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=cfg.num_workers, collate_fn=BBoxCollator(processor, cfg),
                        pin_memory=False)

    model = build_model(cfg, device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps)
    print("GPUs: %d, effective batch size: %d"
          % (world_size, cfg.batch_size * cfg.grad_accum * world_size))
    print("Optimizer steps: %d, warmup steps: %d"
          % (total_steps, int(total_steps * cfg.warmup_ratio)))

    global_step = 0
    optimizer_steps = 0
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(loader):
            labels = batch.pop("labels").to(device, dtype=torch.long)
            batch = move_to_device(batch, device)
            # FM9GV.forward(self, data, **kwargs) —— data 是处理器输出的那个 dict，
            # 其余关键字会透传给内部的 FM9GForCausalLM.forward
            # （它接受 attention_mask / labels / use_cache / return_dict）。
            outputs = model(
                data=batch,
                attention_mask=batch["attention_mask"],
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            shift = labels[..., 1:].contiguous().view(-1)
            active = shift != -100
            loss = outputs.loss[active].mean() / cfg.grad_accum
            loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps >= total_steps:
                    break

            if is_main and (global_step % cfg.log_every == 0):
                print("epoch=%d step=%d/%d loss=%.4f lr=%.3e"
                      % (epoch, global_step, len(loader), float(loss) * cfg.grad_accum,
                         sched.get_last_lr()[0]), flush=True)
            global_step += 1

            if cfg.save_every > 0 and global_step % cfg.save_every == 0:
                if is_main:
                    d = os.path.join(cfg.output_dir, "step_%d" % global_step)
                    save_checkpoint(model, processor, cfg, d)
                    print("中途保存 -> %s" % d, flush=True)

    if is_main:
        save_checkpoint(model, processor, cfg, cfg.output_dir)
        print("Saved LoRA adapter to: %s" % cfg.output_dir)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
