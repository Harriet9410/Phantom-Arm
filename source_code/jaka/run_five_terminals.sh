#!/usr/bin/env bash
# 五终端启动参考（云端；路径以实例实际为准）
# 用法：bash run_five_terminals.sh  或  分终端复制各段
# 阶段5：仅整理，不改变官方启动顺序。

set -e

ROOT_DATA="${ROOT_DATA:-/root/gpufree-data}"
# 兼容 /root 布局
if [[ ! -d "$ROOT_DATA/jaka" && -d /root/jaka ]]; then
  ROOT_DATA=/root
fi
if [[ ! -d "$ROOT_DATA/EAICON" && -d /root/EAICON ]]; then
  EAICON_DIR=/root/EAICON
else
  EAICON_DIR="$ROOT_DATA/EAICON"
fi
JAKA_WS="${JAKA_WS:-$ROOT_DATA/jaka}"

echo "=== TCEI 五终端启动参考 ==="
echo "EAICON=$EAICON_DIR"
echo "JAKA_WS=$JAKA_WS"
echo ""
echo "--- 终端1 ---"
echo "roscore"
echo ""
echo "--- 终端2（仿真，等机械臂出现）---"
echo "cd $EAICON_DIR && sh run_jaka_sim.sh"
echo ""
echo "--- 终端3（YOLO）---"
echo "cd $JAKA_WS && conda activate yolov8 && rosrun large_scale_model_arm isaac_yolov8.py"
echo ""
echo "--- 终端4（大模型 + 指令）---"
echo "cd $JAKA_WS && conda activate inference && rosrun large_scale_model_arm isaac_scale.py"
echo ""
echo "--- 终端5（决策+抓取；可带参数）---"
echo "cd $JAKA_WS && conda deactivate && roslaunch large_scale_model_arm isaac_jaka.launch"
echo ""
echo "可选 grasp 参数示例："
echo "  max_retries:=2 approach_height:=0.12 approach_dx:=0.03"
echo ""
echo "示例指令：抓取烟雾弹"
