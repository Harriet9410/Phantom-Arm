"""Measured stop acknowledgement; no simulator, clock or joint-state mutation.

The actuator layer must continuously hold ``hold_position`` with its position
drive and zero target velocity.  This object never declares success merely
because a trajectory was removed.  Thresholds are engineering defaults pending
physical stop-distance/latency validation, not a certified emergency stop.
"""
import numpy as np


class StopLatch:
    def __init__(self, velocity_limit=.03, drift_limit=.0015,
                 stable_seconds=.25, timeout_seconds=5.):
        values=(velocity_limit,drift_limit,stable_seconds,timeout_seconds)
        if any(not np.isfinite(v) or v<=0 for v in values):
            raise ValueError('stop thresholds must be finite and positive')
        self.velocity_limit=velocity_limit;self.drift_limit=drift_limit
        self.stable_seconds=stable_seconds;self.timeout_seconds=timeout_seconds
        self.active=False;self.request_id=None;self.state='ready'
        self.hold_position=None;self.first_tcp=None;self.last_sim=None
        self.stable_since=None;self.stable_anchor=None;self.requested_at=None
        self.last_snapshot=None;self.fault_reason=None
        self.solver_type=None;self.physics_dt=None;self.velocity_source='reported_joint_velocity'
        self.previous_position=None;self.stable_tcp_anchor=None

    def configure_solver(self,solver_type,physics_dt):
        """Read-only SDK configuration supplied on the physics thread.

        TGS reports the last solver substep velocity, not necessarily the
        full-step displacement velocity (PhysX, Implicit Spring Joint Drives,
        2024, p5). Keep the raw value but measure actual per-physics-step motion.
        Unknown/PGS solvers retain the original reported-velocity requirement.
        """
        if solver_type is not None and solver_type not in ('TGS','PGS'):
            raise ValueError('unsupported stop feedback solver')
        if physics_dt is not None and (not np.isfinite(physics_dt) or physics_dt<=0):
            raise ValueError('invalid physics timestep for stop evidence')
        if solver_type=='TGS' and (physics_dt is None or physics_dt>.02):
            raise ValueError('TGS stop evidence requires physics sampling of at least50Hz')
        if self.hold_position is not None and (solver_type!=self.solver_type or physics_dt!=self.physics_dt):
            raise ValueError('solver or timestep changed during stop')
        self.solver_type=solver_type;self.physics_dt=physics_dt
        self.velocity_source=('consecutive_physics_step_position_difference' if solver_type=='TGS'
                              else 'reported_joint_velocity')

    def request(self, request_id, reason, wall_time):
        if not isinstance(request_id,str) or not request_id or len(request_id)>128:
            raise ValueError('stop id must be a nonempty string up to 128 characters')
        if not np.isfinite(wall_time):raise ValueError('invalid stop request time')
        if self.active:
            if request_id!=self.request_id:raise ValueError('another stop id is already latched')
            return self.state
        self.active=True;self.request_id=request_id;self.reason=str(reason)[:300]
        self.state='requested';self.requested_at=float(wall_time)
        self.hold_position=None;self.first_tcp=None;self.last_sim=None
        self.stable_since=None;self.stable_anchor=None;self.fault_reason=None
        self.last_snapshot=None
        self.previous_position=None;self.stable_tcp_anchor=None
        return self.state

    def sample(self, positions, velocities, tcp, simulation_time, wall_time):
        if not self.active:raise ValueError('no active stop')
        q=np.asarray(positions,dtype=float);v=np.asarray(velocities,dtype=float)
        p=np.asarray(tcp,dtype=float)
        valid=(q.ndim==1 and len(q)>0 and v.shape==q.shape and p.shape==(3,)
               and all(np.isfinite(x).all() for x in (q,v,p))
               and np.isfinite(simulation_time) and np.isfinite(wall_time))
        if not valid:
            self.fault_reason='invalid joint or TCP feedback';self.state='fault'
        else:
            if self.hold_position is None:
                self.hold_position=q.copy();self.first_tcp=p.copy()
            if self.hold_position.shape!=q.shape:
                self.fault_reason='joint feedback dimension changed';self.state='fault'
            elif self.last_sim is not None and simulation_time<self.last_sim:
                self.fault_reason='simulation time moved backwards';self.state='fault'
            elif self.last_sim is None or simulation_time>self.last_sim:
                # Repeated reads during the same physics frame cannot satisfy
                # the confirmation interval.  Use a fixed anchor, not drift
                # between adjacent frames (which could hide steady creep).
                dt=None if self.last_sim is None else float(simulation_time-self.last_sim)
                reported_speed=float(np.max(np.abs(v)));speed=reported_speed
                if self.velocity_source=='consecutive_physics_step_position_difference':
                    speed=None
                    if dt is not None:
                        if abs(dt-self.physics_dt)>max(1e-7,self.physics_dt*.01):
                            self.fault_reason='missing or changed physics step in stop evidence';self.state='fault'
                        else:speed=float(np.max(np.abs(q-self.previous_position))/dt)
                self.last_sim=float(simulation_time);self.previous_position=q.copy()
                if self.stable_anchor is None:
                    self.stable_anchor=q.copy();self.stable_tcp_anchor=p.copy()
                drift=float(np.max(np.abs(q-self.stable_anchor)))
                tcp_drift=float(np.linalg.norm(p-self.stable_tcp_anchor))
                if speed is None or speed>self.velocity_limit or drift>self.drift_limit or tcp_drift>self.drift_limit:
                    self.stable_since=None;self.stable_anchor=q.copy();self.stable_tcp_anchor=p.copy()
                    if not self.fault_reason:self.state='holding'
                else:
                    if self.stable_since is None:self.stable_since=float(simulation_time)
                    if not self.fault_reason:
                        self.state=('stopped' if simulation_time-self.stable_since>=self.stable_seconds
                                    else 'holding')
                self.last_snapshot={'max_joint_speed':speed,
                    'reported_max_joint_speed':reported_speed,'velocity_source':self.velocity_source,
                    'solver_type':self.solver_type,'physics_dt':self.physics_dt,'sample_dt':dt,
                    'joint_distance':float(np.max(np.abs(q-self.hold_position))),
                    'tcp_distance':float(np.linalg.norm(p-self.first_tcp)),
                    'stable_window_tcp_drift':tcp_drift,
                    'simulation_time':float(simulation_time)}
        elapsed=float(wall_time-self.requested_at)
        if elapsed<0:
            self.fault_reason='monotonic time moved backwards';self.state='fault'
        if self.state!='stopped' and elapsed>self.timeout_seconds:
            self.fault_reason=self.fault_reason or 'stop confirmation timed out';self.state='fault'
        return {'id':self.request_id,'state':self.state,'reason':self.reason,
                'seconds':elapsed,'fault_reason':self.fault_reason,
                **(self.last_snapshot or {})}

    def reset(self, request_id):
        if not self.active or request_id!=self.request_id:
            raise ValueError('stop reset id does not match active stop')
        if self.state!='stopped' or self.fault_reason:
            raise ValueError('measured stop has not been confirmed')
        self.active=False;self.state='ready'
        return {'id':request_id,'state':'reset','reason':self.reason}
