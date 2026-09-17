# -*- coding: utf-8 -*-
"""阶段4本地单测：TRRT* 配置模块（不依赖 Isaac）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trrt_config import DEFAULT_TRTT, DEFAULT_DURATION_S, PRESETS, get_trrt_params


def main():
    required = {"max_iter", "step_size", "radius", "init_temp", "k0", "alpha", "beta"}
    assert required.issubset(DEFAULT_TRTT.keys())
    assert DEFAULT_TRTT["step_size"] > 0
    assert DEFAULT_TRTT["max_iter"] > 0
    assert DEFAULT_DURATION_S > 0

    for name, cfg in PRESETS.items():
        assert required.issubset(cfg.keys()), name
        assert cfg["step_size"] > 0

    # baseline 与 DEFAULT 一致
    assert get_trrt_params("baseline") == DEFAULT_TRTT
    # 返回拷贝，改 preset 不影响源 dict
    p = get_trrt_params("fast")
    p["step_size"] = 999
    assert get_trrt_params("fast")["step_size"] != 999

    try:
        get_trrt_params("nope")
        raise SystemExit("expected KeyError")
    except KeyError:
        pass

    print("test_trrt_config: ALL PASS")


if __name__ == "__main__":
    main()
