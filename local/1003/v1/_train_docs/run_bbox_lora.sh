#!/usr/bin/env bash
# FM9G4b bbox QLoRA 启动脚本 —— 只负责读环境变量、组装参数、调 torchrun。
# 所有训练逻辑都在 train_bbox_lora_coco.py 里，这里不重复实现。
set -euo pipefail
cd "$(dirname "$0")"

args=(
  --model-dir "${MODEL_DIR:-.}"
  --coco-root "${COCO_ROOT:?请设置 COCO_ROOT}"
  --output-dir "${OUTPUT_DIR:?请设置 OUTPUT_DIR}"
  --split "${SPLIT:-train}"
  --max-samples "${MAX_SAMPLES:-0}"
  --min-area "${MIN_AREA:-256}"
  --seed "${SEED:-42}"
  --batch-size "${BATCH_SIZE:-1}"
  --grad-accum "${GRAD_ACCUM:-4}"
  --epochs "${EPOCHS:-1}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --num-workers "${NUM_WORKERS:-2}"
  --max-length "${MAX_LENGTH:-2048}"
  --max-slice-nums "${MAX_SLICE_NUMS:-9}"
  --lora-rank "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --log-every "${LOG_EVERY:-10}"
  --save-every "${SAVE_EVERY:-500}"
)
[[ "${LOAD_IN_4BIT:-1}" == 1 ]]         && args+=(--load-in-4bit)
[[ "${GRADIENT_CHECKPOINTING:-1}" == 1 ]] && args+=(--gradient-checkpointing)
[[ "${TRAIN_RESAMPLER:-0}" == 1 ]]      && args+=(--train-resampler)

if [[ -n "${REFCOCO_ROOT:-}" ]]; then
  args+=(--refcoco-root "$REFCOCO_ROOT")
  args+=(--refcoco-datasets "${REFCOCO_DATASETS:-refcoco}")
  args+=(--refcoco-splits "${REFCOCO_SPLITS:-train}")
  args+=(--refcoco-max-samples "${REFCOCO_MAX_SAMPLES:-0}")
fi
[[ -n "${INIT_ADAPTER_DIR:-}" ]] && args+=(--init-adapter-dir "$INIT_ADAPTER_DIR")

torchrun --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  train_bbox_lora_coco.py "${args[@]}"
