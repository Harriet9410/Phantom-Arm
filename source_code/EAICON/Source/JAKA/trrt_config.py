# -*- coding: utf-8 -*-
"""TRRT* / 轨迹默认参数（阶段4集中配置，云端扫参只改这里）。"""

# 与 trrt_star_optimized 默认值对齐；plan_and_execute 原先只传了 step_size/radius
DEFAULT_TRTT = {
    "max_iter": 5000,
    "step_size": 0.01,   # 关节空间步长（rad），偏小更稳、更慢
    "radius": 0.5,       # rewire 邻域
    "init_temp": 1.0,
    "k0": 0.1,
    "alpha": 0.95,
    "beta": 1.1,
}

# 轨迹跟踪时长（秒）
DEFAULT_DURATION_S = 5.0

# 云端可选预设（只改 dict，不改算法逻辑）
PRESETS = {
    "baseline": dict(DEFAULT_TRTT),
    # 更快：加大 step、略减 max_iter
    "fast": {
        "max_iter": 3000,
        "step_size": 0.03,
        "radius": 0.5,
        "init_temp": 1.0,
        "k0": 0.15,
        "alpha": 0.95,
        "beta": 1.1,
    },
    # 更稳：更小 step、更多迭代
    "stable": {
        "max_iter": 8000,
        "step_size": 0.008,
        "radius": 0.6,
        "init_temp": 1.0,
        "k0": 0.08,
        "alpha": 0.97,
        "beta": 1.05,
    },
}


def get_trrt_params(preset: str = "baseline") -> dict:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset: {preset}, choose from {list(PRESETS)}")
    return dict(PRESETS[preset])
