"""Hysteresis for two stamped, validity-aware finger effort readings.

The contact monitor reports support, not target identity or score. The separate
ContactLiftWindow combines that support with aperture and actual robot motion
for operational grasp confirmation; it does not claim a seen object trajectory.
Arm only after the separate closure/contact confirmation. Sensor invalidity must
not be encoded as zero: it is a feedback fault, not evidence of a dropped object.
Thresholds below are provisional engineering settings for native validation.
"""
import math
from numbers import Real


class ContactLiftWindow:
    """Operational grasp confirmation from contact, aperture and actual lift.

    Camera visibility is not required. This is a sensor-based control decision,
    not a claim that the object trajectory was independently seen by a camera.
    """
    def __init__(self,reference_stamp):
        self.last_stamp=float(reference_stamp);self.frames=[]

    def add(self,stamp,translation,rotation_delta,contact):
        reason=None
        if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<=self.last_stamp:
            return {'verified':False,'reason':'stale_or_duplicate_robot_sample','frames':len(self.frames)}
        gap=stamp-self.last_stamp;self.last_stamp=stamp
        if gap>.35:self.frames=[]
        if (not isinstance(translation,(list,tuple)) or len(translation)!=3 or
                not all(type(v) in (int,float) and math.isfinite(v) for v in translation)):
            reason='invalid_measured_lift'
        elif (not isinstance(rotation_delta,(int,float)) or not math.isfinite(rotation_delta) or
              abs(rotation_delta)>.015 or math.hypot(*translation[:2])>.005):
            reason='trial_motion_not_vertical_and_fixed_attitude'
        elif not isinstance(contact,dict) or not contact.get('stable_contact') or not contact.get('nonempty_aperture'):
            reason='stable_bilateral_nonempty_contact_required'
        elif contact.get('state')!='holding' or contact.get('contact_lost') or contact.get('fault'):
            reason='contact_not_supported'
        else:
            times=contact.get('sensor_stamps',[])
            if len(times)!=2 or not all(type(t) in (int,float) and math.isfinite(t) and abs(t-stamp)<=.05 for t in times):
                reason='contact_and_robot_feedback_not_synchronized'
        if reason is None and self.frames and translation[2]<self.frames[-1]['z']-.002:
            reason='measured_lift_reversed'
        if reason is not None:
            self.frames=[]
            return {'verified':False,'reason':reason,'frames':0}
        self.frames.append({'stamp':stamp,'z':translation[2],'sensor_stamps':tuple(contact['sensor_stamps'])})
        span=stamp-self.frames[0]['stamp'];baseline=translation[2]-self.frames[0]['z']
        acquisitions=len({row['sensor_stamps'] for row in self.frames})
        verified=(len(self.frames)>=5 and span>=.25-1e-9 and baseline>=.008-1e-9
                  and translation[2]>=.025-1e-9 and acquisitions>=3)
        return {'verified':verified,'reason':'contact_supported_trial_lift' if verified else 'collecting_contact_lift',
                'frames':len(self.frames),'span_seconds':span,'vertical_baseline_m':baseline,
                'measured_total_lift_m':translation[2],'distinct_contact_acquisitions':acquisitions,
                'evidence':'stamped_bilateral_contact_nonempty_aperture_and_measured_trial_lift',
                'visual_confirmation':False,'scope':'operational grasp confirmation; object trajectory not visually verified'}


EFFORT_PROTOCOL='tcei.finger_effort.v1'


def effort_payload(left,right,simulation_time,published_wall):
    """Keep native validity and acquisition time, including invalid samples."""
    samples=[]
    for name,reading in (('left',left),('right',right)):
        valid=getattr(reading,'is_valid',False) is True
        value=getattr(reading,'value',None);stamp=getattr(reading,'time',None)
        finite=lambda x:isinstance(x,Real) and not isinstance(x,bool) and math.isfinite(x)
        valid=valid and finite(value) and finite(stamp) and stamp>=0
        samples.append({'finger':name,'valid':valid,'value':float(value) if finite(value) else None,
                        'stamp':float(stamp) if finite(stamp) else None})
    return {'protocol':EFFORT_PROTOCOL,'samples':samples,'simulation_time':float(simulation_time),
            'published_wall':float(published_wall),'source':'native_EffortSensor_use_latest_data'}


class GraspContactMonitor:
    def __init__(self,on_threshold=.2,off_threshold=.05,loss_seconds=.6,
                 min_samples=3,max_age=1.5,confirm_seconds=.45):
        values=(on_threshold,off_threshold,loss_seconds,max_age,confirm_seconds)
        if (not all(math.isfinite(v) for v in values) or
                not 0<off_threshold<on_threshold or loss_seconds<=0 or
                max_age<=0 or confirm_seconds<=0 or type(min_samples) is not int or min_samples<3):
            raise ValueError('invalid contact hysteresis settings')
        self.on=on_threshold;self.off=off_threshold;self.loss_seconds=loss_seconds
        self.min_samples=min_samples;self.max_age=max_age;self.confirm_seconds=confirm_seconds
        self.reset()

    def reset(self):
        self.armed=False;self.lost=False;self.fault=None;self.last_stamps=None
        self.progress_at=None;self.efforts=None;self.low_since=None;self.low_samples=0
        self.high_since=None;self.high_samples=0;self.state='unarmed'

    def observe(self,now,stamps,efforts,valid):
        if self.fault is not None:
            if self.armed:return
            self.reset()  # Startup/open-gripper invalidity is not a latched held-object fault.
        try:
            if (not math.isfinite(now) or len(stamps)!=2 or len(efforts)!=2 or len(valid)!=2 or
                    any(type(v) is not bool or not v for v in valid) or
                    not all(math.isfinite(v) for v in tuple(stamps)+tuple(efforts)) or
                    min(stamps)<0 or abs(stamps[0]-stamps[1])>.05):
                raise ValueError('invalid or unsynchronized finger sensor reading')
            if self.last_stamps is not None:
                if any(t<old for t,old in zip(stamps,self.last_stamps)):
                    raise ValueError('finger sensor clock moved backwards')
                if not all(t>old for t,old in zip(stamps,self.last_stamps)):
                    return  # Repeated publications are not new sensor evidence.
                if self.armed and (self.progress_at is None or not 0<=now-self.progress_at<=self.max_age):
                    raise ValueError('finger sensor feedback gap')
            self.last_stamps=tuple(stamps);self.progress_at=now
            self.efforts=tuple(float(v) for v in efforts);stamp=min(stamps)
            if not self.armed or self.lost:return
            magnitudes=[abs(v) for v in efforts]
            if min(efforts)>=self.on:
                self.low_since=None;self.low_samples=0;self.state='holding'
                if self.high_since is None:self.high_since=stamp
                self.high_samples+=1
            else:
                self.high_since=None;self.high_samples=0;self.state='uncertain'
                if max(magnitudes)<self.off:
                    if self.low_since is None:self.low_since=stamp
                    self.low_samples+=1
                    if self.low_samples>=self.min_samples and stamp-self.low_since>=self.loss_seconds-1e-9:
                        self.lost=True;self.state='contact_lost'
                else:
                    self.low_since=None;self.low_samples=0
        except (TypeError,ValueError,OverflowError) as error:
            self.fault=str(error);self.state='feedback_fault'
            self.low_since=None;self.low_samples=0

    def arm(self,now):
        current=self.snapshot(now)
        if current['state']=='feedback_fault' or self.efforts is None or min(self.efforts)<self.on:
            raise ValueError('fresh two-finger contact required before arming')
        self.armed=True;self.lost=False;self.state='holding'
        self.low_since=None;self.low_samples=0
        self.high_since=min(self.last_stamps);self.high_samples=1

    def snapshot(self,now):
        if (not math.isfinite(now) or self.progress_at is None or
                not 0<=now-self.progress_at<=self.max_age):
            self.fault=self.fault or 'finger sensor feedback missing, frozen or stale'
        state='feedback_fault' if self.fault is not None else self.state
        stamp=min(self.last_stamps) if self.last_stamps else None
        high_span=0. if self.high_since is None else max(0.,stamp-self.high_since)
        low_span=0. if self.low_since is None else max(0.,stamp-self.low_since)
        return {'state':state,'contact_lost':self.lost and self.fault is None,
                'retained_support':self.armed and not self.lost and self.fault is None,
                'stable_contact':state=='holding' and high_span>=self.confirm_seconds-1e-9 and self.high_samples>=self.min_samples,
                'efforts':self.efforts,'sensor_stamps':self.last_stamps,
                'high_samples':self.high_samples,'high_seconds':high_span,
                'low_samples':self.low_samples,'low_seconds':low_span,
                'on_threshold':self.on,'off_threshold':self.off,'fault':self.fault,
                'scope':'contact support only; not object identity or lift verification'}
