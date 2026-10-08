"""Read live V7 readiness only; no observation, joint, gripper or task publish."""
import argparse
import json
import math
from pathlib import Path
import threading
import time


def evaluate(samples,now,wall_now=None):
    required=('scene','joints','pose','depth','info','capture','controller','nine')
    missing=[name for name in required if name not in samples]
    stale=[name for name,(value,at) in samples.items() if name in required and now-at>2.]
    errors=['missing:'+name for name in missing]+['stale:'+name for name in stale if name not in ('controller','nine')]
    for name in ('controller','nine'):
        if name in samples and samples[name][0].get('status')!='ready':errors.append(name+'_has_not_reported_ready')
    if 'scene' in samples and wall_now is not None:
        observed=samples['scene'][0].get('observed_at')
        if type(observed) not in (int,float) or not 0<=wall_now-observed<2.:errors.append('source_scene_stale')
    if 'joints' in samples:
        joints=samples['joints'][0];names=joints.get('all_joint_names',[]);values=joints.get('all_joint_positions',[])
        if len(names)<8 or len(values)!=len(names) or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            errors.append('complete_arm_and_finger_feedback_unavailable')
        if joints.get('tracking'):errors.append('unexpected_active_trajectory_during_no_action_warmup')
    if 'capture' in samples and samples['capture'][0]:errors.append('gripper_occupied')
    # Candidate count, complete visibility and parking pose are observations,
    # never readiness gates for this no-action phase.
    return {'ready_for_read_only':not errors,'errors':errors}


def main():
    import rospy
    from std_msgs.msg import String,Bool
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image,CameraInfo
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--timeout',type=float,default=30.);args=parser.parse_args()
    rospy.init_node('tcei_no_action_probe',anonymous=True)
    samples={};lock=threading.Lock()
    def store(name,value):
        with lock:samples[name]=(value,time.monotonic())
    for topic,name in (('/tcei/candidates','scene'),('/tcei/joint_diagnostics','joints'),
                       ('/tcei/task_status','controller'),('/tcei/nine_status','nine')):
        rospy.Subscriber(topic,String,lambda msg,name=name:store(name,json.loads(msg.data)),queue_size=1)
    rospy.Subscriber('/Jaka/get_end_effector_pose',PoseStamped,lambda msg:store('pose',True),queue_size=1)
    rospy.Subscriber('/Jaka/camera/depth',Image,lambda msg:store('depth',{'stamp':msg.header.stamp.to_sec(),'width':msg.width,'height':msg.height}),queue_size=1)
    rospy.Subscriber('/Jaka/camera/camera_info',CameraInfo,lambda msg:store('info',{'K':list(msg.K),'width':msg.width,'height':msg.height}),queue_size=1)
    rospy.Subscriber('/Jaka/gripper_is_captured',Bool,lambda msg:store('capture',bool(msg.data)),queue_size=1)
    until=time.monotonic()+args.timeout;result={}
    while time.monotonic()<until and not rospy.is_shutdown():
        disabled=all(rospy.get_param(name,False) is False for name in ('/tcei_controller/execute','/tcei_nine/execute'))
        with lock:result=evaluate(dict(samples),time.monotonic(),time.time())
        if not disabled:result={'ready_for_read_only':False,'errors':['execution_switch_not_disabled']};break
        if result['ready_for_read_only']:break
        time.sleep(.1)
    with lock:data={name:value for name,(value,_) in samples.items()}
    result.update(time=time.time(),mode='read_only_no_action',observations=data,
                  candidate_count=len(data.get('scene',{}).get('candidates',[])),
                  grasp_capability=data.get('controller',{}).get('lift_evidence_capable',False),
                  count_is_not_a_gate=True)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));return 0 if result.get('ready_for_read_only') else 1


if __name__=='__main__':raise SystemExit(main())
