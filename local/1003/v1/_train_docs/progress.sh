#!/usr/bin/env bash
# 看九格 QLoRA 训练进度
#
# 自动识别最近一轮训练（找 /root/sft_v2/outputs/ 下最新的 train_*.log），
# 所以换轮次不用改脚本。
#
# 用法：  bash /root/sft_v2/progress.sh
#         bash /root/sft_v2/progress.sh /root/sft_v2/outputs/train_crop.log   # 指定日志
set -u

OUT=/root/sft_v2/outputs

if [ $# -ge 1 ]; then
  LOG="$1"
else
  LOG=$(ls -t $OUT/train*.log 2>/dev/null | head -1)
fi

echo "====================================="
if [ -z "${LOG:-}" ] || [ ! -f "$LOG" ]; then
  echo "找不到训练日志（$OUT/train*.log）"
  exit 1
fi
echo "日志: $(basename "$LOG")"

# 从日志里推断输出目录（run_bbox_lora 会把 OUTPUT_DIR 打在命令行里）
CKPT_DIR=$(grep -oE '\-\-output-dir [^ ]+' "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
[ -z "${CKPT_DIR:-}" ] && CKPT_DIR="$OUT/qlora_crop"

# 总前向次数 = (最大 epoch + 1) x 每轮批次数
# 每轮批次数从日志的 "step=X/N" 里取 N；最大 epoch 从 "epoch=K" 里取 K。
# 不依赖日志里有 --epochs（它不一定被打印出来）。
DENOM=$(grep -oE 'step=[0-9]+/[0-9]+' "$LOG" | tail -1 | sed 's|.*/||')
MAXEP=$(grep -oE 'epoch=[0-9]+' "$LOG" | tail -1 | sed 's|epoch=||')
TOTAL=$(( ${DENOM:-0} * (${MAXEP:-0} + 1) ))

LAST=$(grep -E "epoch=.*loss=" "$LOG" | tail -1)
if [ -z "$LAST" ]; then
  echo "还没有训练输出，日志尾部："
  tail -5 "$LOG"
else
  CUR=$(echo "$LAST" | sed -n 's/.*step=\([0-9]*\)\/.*/\1/p')
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  echo "最新     : $LAST"
  if [ "$TOTAL" -gt 0 ]; then
    echo "进度     : $CUR / $TOTAL  ($(( CUR * 100 / TOTAL ))%)"
  else
    echo "进度     : step $CUR"
  fi
  if [ "$AGE" -lt 300 ]; then
    echo "活跃度   : 日志 ${AGE} 秒前刚写过 —— 正常"
  else
    echo "活跃度   : 日志已 ${AGE} 秒没动 —— 可能卡住或已结束"
  fi
fi

echo "错误数   : $(grep -c -e Error -e Traceback "$LOG" 2>/dev/null)"
echo "检查点   : $(ls -d $CKPT_DIR/step_* 2>/dev/null | wc -l) 个   ($CKPT_DIR)"
echo "最终适配器: $(ls -l $CKPT_DIR/adapter_model.safetensors 2>/dev/null | awk '{print $5" 字节"}' || echo 未生成)"
echo "进程数   : $(pgrep -c -f train_bbox_lora) 个（大于 0 表示在跑）"
echo "GPU      : $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"
echo "磁盘可用 : $(df -h / | tail -1 | awk '{print $4}')"
echo "当前时间 : $(date '+%Y-%m-%d %H:%M:%S')"

# 各 epoch 平均 loss（判断是否还在下降 / 是否过拟合）
if grep -q "epoch=.*loss=" "$LOG"; then
  echo "-------------------------------------"
  echo "各 epoch 平均 loss:"
  grep -E "epoch=.*loss=" "$LOG" | sed "s/epoch=\([0-9]*\).*loss=\([0-9.]*\).*/\1 \2/" \
    | awk '{s[$1]+=$2;n[$1]++;if($2>mx[$1])mx[$1]=$2} END {for(e in s) printf "  epoch %s: 平均=%.4f 最高=%.3f 批次=%d\n", e, s[e]/n[e], mx[e], n[e]}' | sort
fi
echo "====================================="
