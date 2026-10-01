"""Try a deterministic joint path only when every sampled state passes guards."""
import numpy as np


def guarded_joint_line(start, goal, state_valid, max_joint_step=.025):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.ndim != 1 or start.shape != goal.shape or not len(start):
        raise ValueError('invalid joint vectors')
    if not np.isfinite(start).all() or not np.isfinite(goal).all() or max_joint_step <= 0:
        raise ValueError('invalid joint values')
    count = max(61, int(np.ceil(np.max(np.abs(goal - start)) / max_joint_step)) + 1)
    path = start + np.linspace(0., 1., count)[:, None] * (goal - start)
    path[0] = start
    path[-1] = goal
    for state in path:
        if not state_valid(state):
            return None
    return path
