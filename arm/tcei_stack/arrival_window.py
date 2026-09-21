"""Require fresh, slow, in-tolerance observations before switching motion stages."""
import math
from core import quaternion_error


class ArrivalWindow:
    def __init__(self, seconds=.2, min_samples=3, linear_speed=.02, angular_speed=.1):
        if not .15 <= seconds <= 1. or min_samples < 3:
            raise ValueError('invalid arrival window')
        self.seconds = seconds
        self.min_samples = min_samples
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.previous = None
        self.since = None
        self.samples = 0

    def reset(self):
        self.since = None
        self.samples = 0

    def observe(self, now, stamp, position, quaternion, eligible):
        if not eligible:
            self.reset()
        if self.previous is not None and stamp <= self.previous[0]:
            return False
        slow = False
        if self.previous is not None:
            old_stamp, old_position, old_quaternion = self.previous
            dt = stamp - old_stamp
            slow = (0 < dt <= .3 and
                    math.dist(position, old_position) / dt <= self.linear_speed and
                    quaternion_error(quaternion, old_quaternion) / dt <= self.angular_speed)
        self.previous = (stamp, list(position), list(quaternion))
        if not eligible or not slow:
            self.reset()
            return False
        if self.since is None:
            self.since = now
        self.samples += 1
        return self.samples >= self.min_samples and now - self.since >= self.seconds
