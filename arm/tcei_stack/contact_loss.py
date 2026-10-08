"""Fresh contact-loss evidence persists across short trajectory segments."""
import math


class ContactLossWindow:
    def __init__(self,seconds=.6,max_age=1.5,min_samples=3):
        self.seconds=seconds;self.max_age=max_age;self.min_samples=min_samples
        self.reset()

    def reset(self):
        self.since=None;self.last=None;self.samples=0

    def observe(self,now,sample_at,captured):
        if not all(math.isfinite(t) for t in (now,sample_at)) or not 0<=now-sample_at<=self.max_age:
            self.reset();return False
        if self.last is not None and sample_at<=self.last:return False
        if self.last is not None and sample_at-self.last>self.max_age:self.reset()
        self.last=sample_at
        if captured:
            self.since=None;self.samples=0;return False
        if self.since is None:self.since=sample_at
        self.samples+=1
        return self.samples>=self.min_samples and sample_at-self.since>=self.seconds
