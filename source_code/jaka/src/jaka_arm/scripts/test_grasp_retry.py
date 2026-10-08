# -*- coding: utf-8 -*-
"""阶段3本地单测：抓取重试决策逻辑（不依赖 ROS）。"""
import sys


def decide_retry(attempt, ok, max_retries):
    """返回 (should_continue, next_attempt_or_none, final_ok)。"""
    if ok:
        return False, None, True
    total = max(1, max_retries + 1)
    if attempt + 1 < total:
        return True, attempt + 1, False
    return False, None, False


def failure_blocks_grasp(reason):
    """是否应在闭合/放置前中止。"""
    return reason in (
        "APPROACH_TIMEOUT",
        "DESCEND_TIMEOUT",
        "NOT_IN_TOLERANCE",
        "GRIPPER_TIMEOUT",
    )


def main():
    # 一次成功
    cont, nxt, fin = decide_retry(0, True, 2)
    assert cont is False and fin is True

    # 第一次失败、还有次数
    cont, nxt, fin = decide_retry(0, False, 2)
    assert cont is True and nxt == 1 and fin is False

    # 最后一次仍失败
    cont, nxt, fin = decide_retry(2, False, 2)
    assert cont is False and fin is False

    # max_retries=0 → 只试一次
    cont, nxt, fin = decide_retry(0, False, 0)
    assert cont is False and fin is False

    for r in ("APPROACH_TIMEOUT", "DESCEND_TIMEOUT", "NOT_IN_TOLERANCE", "GRIPPER_TIMEOUT"):
        assert failure_blocks_grasp(r)
    assert not failure_blocks_grasp("SUCCESS")

    print("test_grasp_retry: ALL PASS")


if __name__ == "__main__":
    main()
