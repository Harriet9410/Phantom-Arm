# -*- coding: utf-8 -*-
"""阶段2本地单测：深度邻域采样（不依赖 ROS/YOLO/相机）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# 避免 import rospy：只加载 sample_depth_robust 逻辑（与 isaac_yolov8.py 同步）
DEPTH_MIN = 0.05
DEPTH_MAX = 5.0


def sample_depth_robust(depth_image, u, v, window=5, min_valid=3):
    if depth_image is None:
        return None
    h, w = depth_image.shape[:2]
    cx_i, cy_i = int(round(u)), int(round(v))
    if not (0 <= cx_i < w and 0 <= cy_i < h):
        return None
    half = max(1, int(window) // 2)
    x0, x1 = max(0, cx_i - half), min(w, cx_i + half + 1)
    y0, y1 = max(0, cy_i - half), min(h, cy_i + half + 1)
    patch = np.asarray(depth_image[y0:y1, x0:x1], dtype=np.float64).ravel()
    patch = patch[np.isfinite(patch)]
    patch = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    if patch.size < min_valid:
        return None
    return float(np.median(patch))


def main():
    d = np.full((20, 20), 1.0, dtype=np.float32)
    assert abs(sample_depth_robust(d, 10, 10) - 1.0) < 1e-6

    d2 = np.full((20, 20), 1.0, dtype=np.float32)
    d2[10, 10] = float("nan")
    d2[10, 11] = 0.0
    d2[11, 10] = 99.0
    assert abs(sample_depth_robust(d2, 10, 10) - 1.0) < 1e-6

    d3 = np.zeros((20, 20), dtype=np.float32)
    assert sample_depth_robust(d3, 10, 10) is None

    d4 = np.full((20, 20), float("inf"), dtype=np.float32)
    assert sample_depth_robust(d4, 10, 10) is None

    assert sample_depth_robust(None, 1, 1) is None
    assert sample_depth_robust(d, -5, 10) is None
    assert sample_depth_robust(d, 10, 99) is None

    # 中心 3x3 为 0.8、窗口其余为 1.2：5x5 中位数应偏向多数点 1.2
    d5 = np.full((20, 20), 1.2, dtype=np.float32)
    d5[9:12, 9:12] = 0.8
    got5 = sample_depth_robust(d5, 10, 10, window=5)
    assert abs(got5 - 1.2) < 1e-6

    # window=3 时中心 3x3 全为 0.8，中位数为 0.8
    got3 = sample_depth_robust(d5, 10, 10, window=3)
    assert abs(got3 - 0.8) < 1e-6

    print("test_depth_sample: ALL PASS")


if __name__ == "__main__":
    main()
