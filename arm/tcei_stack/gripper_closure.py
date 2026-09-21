"""Callback-driven contact evidence; no ROS, actuator, or sensor modification.

Effort/capture messages have no acquisition stamp in the supplied ROS protocol.
Their stamps here are the latest observed simulation clock at callback receipt,
not a claim of atomic sensor synchronization. Every received interruption counts.
"""
import math


class ContactConfirmationWindow:
    def __init__(self, threshold=.2, seconds=.45, samples=3, max_age=1.5):
        if (not all(math.isfinite(v) for v in (threshold, seconds, max_age)) or
                threshold < .2 or seconds < .45 or samples < 3 or max_age <= 0):
            raise ValueError('contact requirements cannot be weakened')
        self.threshold=threshold;self.seconds=seconds;self.samples=samples;self.max_age=max_age
        self.error=None;self.sim=None;self.clock_seen=None;self.clock_progress=None
        self.efforts=None;self.effort_seen=None;self.effort_sim=None
        self.captured=None;self.capture_seen=None;self.capture_sim=None
        self.fingers=None;self.joint_seen=None;self.joint_stamp=None;self.joint_progress=None;self.joint_sim=None
        self._break()

    def fail(self, reason):
        if self.error is None:self.error=reason
        self._break()

    def _break(self):
        self.start=None;self.count=0;self.counted_effort_sim=None

    def restart_confirmation(self):
        """A new load command requires a new stable contact window."""
        self._break()

    def observe_clock(self, sim, wall):
        if self.clock_progress is not None and wall-self.clock_progress>self.max_age:
            self.fail('simulation clock frozen')
        if not isinstance(sim,(int,float)) or not math.isfinite(sim):
            self.fail('simulation clock invalid');return
        if self.sim is not None and sim < self.sim:
            self.fail('simulation clock moved backwards');return
        if self.sim is None or sim > self.sim:self.clock_progress=wall
        self.sim=float(sim);self.clock_seen=wall

    def observe_efforts(self, values, wall):
        if self.effort_seen is not None and wall-self.effort_seen>self.max_age:
            self.fail('fresh two-finger contact feedback unavailable')
        try:valid=len(values)==2 and all(math.isfinite(v) for v in values)
        except (TypeError,ValueError):valid=False
        if not valid:self.fail('fresh two-finger contact feedback unavailable');return
        self.efforts=list(values);self.effort_seen=wall;self.effort_sim=self.sim
        self._update(wall, effort=True)

    def observe_capture(self, captured, wall):
        if self.capture_seen is not None and wall-self.capture_seen>self.max_age:
            self.fail('capture feedback missing or stale')
        if not isinstance(captured,bool):self.fail('capture feedback invalid');return
        self.captured=captured;self.capture_seen=wall;self.capture_sim=self.sim
        self._update(wall)

    def observe_joints(self, names, positions, stamp, wall):
        if self.joint_progress is not None and wall-self.joint_progress>self.max_age:
            self.fail('measured finger feedback frozen')
        try:
            if (len(names)!=len(positions) or len(set(names))!=len(names) or
                    not all(math.isfinite(v) for v in positions) or not math.isfinite(stamp)):
                raise ValueError()
            values=[float(positions[names.index(n)]) for n in ('finger1_joint','finger2_joint')]
        except (ValueError,TypeError,AttributeError):
            self.fail('measured finger joint feedback invalid');return
        if self.joint_stamp is not None and stamp < self.joint_stamp:
            self.fail('finger joint stamp moved backwards');return
        if self.joint_stamp is None or stamp > self.joint_stamp:
            self.joint_progress=wall
            # /JointState.header may be wall time when /use_sim_time is unset.
            # Use it only as an increasing source marker. Associate a NEW
            # marker with the latest received /clock, never mix the domains.
            self.joint_sim=self.sim
        self.fingers=values;self.joint_stamp=stamp;self.joint_seen=wall

    def _fresh(self, seen, wall):
        return seen is not None and 0 <= wall-seen <= self.max_age

    def _positive(self, wall):
        return (self.error is None and self.sim is not None and self.captured is True and
                self.efforts is not None and min(self.efforts)>=self.threshold and
                self._fresh(self.effort_seen,wall) and self._fresh(self.capture_seen,wall))

    def _update(self, wall, effort=False):
        if not self._positive(wall):self._break();return
        if self.start is None:self.start=self.sim
        if effort and (self.counted_effort_sim is None or self.sim>self.counted_effort_sim):
            self.count+=1;self.counted_effort_sim=self.sim

    def snapshot(self, wall):
        for seen,reason in ((self.clock_seen,'simulation clock missing or stale'),
                            (self.clock_progress,'simulation clock frozen'),
                            (self.effort_seen,'fresh two-finger contact feedback unavailable'),
                            (self.capture_seen,'capture feedback missing or stale'),
                            (self.joint_seen,'measured finger feedback missing or stale'),
                            (self.joint_progress,'measured finger feedback frozen')):
            if not self._fresh(seen,wall):self.fail(reason)
        if self.error is not None:raise RuntimeError(self.error)
        positive=self._positive(wall)
        # Clock ticks alone cannot extend contact evidence or make it pass.
        end=min(self.effort_sim,self.capture_sim,self.joint_sim) if positive else None
        span=max(0.,end-self.start) if positive and self.start is not None else 0.
        return {'contact_candidate':positive,'confirmed':positive and span>=self.seconds and self.count>=self.samples,
                'stable_sim_seconds':span,'fresh_contact_samples':self.count,
                'contact_efforts':list(self.efforts),'finger_positions':list(self.fingers),
                'simulation_time':self.sim,'joint_stamp':self.joint_stamp,
                'feedback_simulation_time':min(self.effort_sim,self.capture_sim,self.joint_sim)}


def next_closure_target(command, measured_fingers, step=.00025, maximum=.04, max_lead=.002):
    """Bound steps and loading separately; None means tracking loss/empty grasp.

    The 2mm lead is a pending native-validation engineering cap, inherited from
    the old preload amount. It is not a calibrated force or contact guarantee.
    """
    if (not all(math.isfinite(v) for v in [command,step,maximum,max_lead]+list(measured_fingers)) or
            len(measured_fingers)!=2 or step<=0 or step>.00025 or not 0<max_lead<=.002 or
            not 0<=command<=maximum<=.04):
        raise ValueError('invalid bounded closure input')
    if command-min(measured_fingers)>max_lead+1e-6:
        # Internal opening can leave an older, larger external target latched.
        # Do not silently keep increasing/reissue it, or abruptly command open.
        return None
    return max(command,min(maximum,command+step,min(measured_fingers)+max_lead))
