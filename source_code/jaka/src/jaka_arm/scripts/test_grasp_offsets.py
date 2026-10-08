# -*- coding: utf-8 -*-
"""阶段5本地单测：抓取接近偏移默认值（与 isaac_grasp.cpp 参数名对齐）。"""
import re
from pathlib import Path

CPP = Path(__file__).resolve().parents[1] / "src" / "isaac_grasp.cpp"
text = CPP.read_text(encoding="utf-8", errors="ignore")

required_params = [
    "approach_dx",
    "approach_dy",
    "descend_dx",
    "descend_dy",
    "place_lower_dz",
    "place_raise_dz",
    "settle_s_after_descend",
    "settle_s_after_open",
]
for p in required_params:
    assert f'pnh.param("{p}"' in text, p

# 不应再出现硬编码接近偏移
assert "position.x += 0.03" not in text
assert "position.y += 0.0075" not in text
assert "position.x += 0.01" not in text

# 默认值与官方基线一致
assert re.search(r'pnh\.param\("approach_dx", approach_dx_, 0\.03\)', text)
assert re.search(r'pnh\.param\("approach_height", approach_height_, 0\.1\)', text)

print("test_grasp_offsets: ALL PASS")
