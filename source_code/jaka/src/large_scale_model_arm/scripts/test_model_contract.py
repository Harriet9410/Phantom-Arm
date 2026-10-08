# -*- coding: utf-8 -*-
"""阶段1本地单测：不依赖 ROS/GPU，只验证输出契约与解析正则。"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# 与 isaac_scale.py.build_user_prompt 同步（独立拷贝，避免 import rospy）
def build_user_prompt(user_command):
    cmd = (user_command or "").strip() or "识别图中所有可抓取物资"
    return (
        "### 背景 ###\n"
        "你需要识别图中目标物资在相机像素系下的位置，并给出抓取所需的深度与倾斜角。\n"
        f"### 任务 ###\n{cmd}\n"
        "### 输出契约（必须严格遵守） ###\n"
        "只输出一行或多行元组，每个目标一行，格式：\n"
        "(u, v, depth_m, angle_deg)\n"
        "说明：u,v 为像素坐标整数；depth_m 为米制深度，带单位 m；angle_deg 为0~180度倾斜角。\n"
        "若需要分拣侧，单独另起一行写：左侧 或 右侧。\n"
        "不要输出 JSON、markdown、解释文字或额外符号。\n"
        "### 示例 ###\n"
        "(312, 245, 0.82m, 12.5)\n"
        "(401, 260, 0.95m, 87.0)\n"
        "左侧\n"
    )


# 与 isaac_yolov8.cpp 一致
RE = re.compile(
    r"[\(\[\（]\s*([-+]?\d+(?:\.\d+)?)\s*[,，]\s*([-+]?\d+(?:\.\d+)?)\s*[,，]\s*([-+]?\d+(?:\.\d+)?)\s*m?\s*[,，]\s*([-+]?\d+(?:\.\d+)?)\s*[\)\]\）]"
)


def parse(s):
    return [(float(m[1]), float(m[2]), float(m[3]), float(m[4])) for m in RE.finditer(s)]


def side(s):
    if "左侧" in s or "left" in s:
        return "left"
    if "右侧" in s or "right" in s:
        return "right"
    return "unknown"


def main():
    p = build_user_prompt("抓取烟雾弹")
    assert "抓取烟雾弹" in p
    assert "(u, v, depth_m, angle_deg)" in p

    p2 = build_user_prompt("   ")
    assert "识别图中所有可抓取物资" in p2

    cases = [
        ("(312, 245, 0.82m, 12.5)", [(312, 245, 0.82, 12.5)]),
        ("(312,245,0.82,12.5)", [(312, 245, 0.82, 12.5)]),
        ("[312, 245, 0.82m, 12.5]", [(312, 245, 0.82, 12.5)]),
        ("（312，245，0.82m，12.5）", [(312, 245, 0.82, 12.5)]),
        ("(10, 20, 1.0m, 0)\n(30, 40, 1.2m, 90.0)", [(10, 20, 1.0, 0), (30, 40, 1.2, 90.0)]),
        ("no tuple here", []),
    ]
    for text, expect in cases:
        got = parse(text)
        assert got == expect, (text, got, expect)

    assert side("目标在左侧") == "left"
    assert side("put right") == "right"
    assert side("ok") == "unknown"

    print("test_model_contract: ALL PASS")


if __name__ == "__main__":
    main()
