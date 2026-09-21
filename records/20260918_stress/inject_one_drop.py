#!/usr/bin/env python3
"""Test-only: one real release through the normal gripper topic, never fake feedback."""
import argparse,json,time
from pathlib import Path
import rospy
from std_msgs.msg import String,Float32

p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
rospy.init_node('tcei_test_one_drop',anonymous=True)
publisher=rospy.Publisher('/Jaka/set_gripper_value',Float32,queue_size=1)
marker=rospy.Publisher('/tcei/test_fault',String,queue_size=1,latch=True)
target=None;injected=False


def event(message):
    global target,injected
    value=json.loads(message.data)
    if value['status']=='attempt' and value['candidate']['class']=='Torch' and target is None:target=value['request_id']
    if not injected and target and value.get('request_id')==target and value['status']=='arrived' and str(value.get('phase','')).startswith('lift_01_'):
        if not publisher.get_num_connections():raise RuntimeError('no simulator gripper subscriber')
        injected=True
        fault={'time':time.time(),'request_id':target,'kind':'single_deliberate_gripper_open',
               'at_phase':value['phase'],'command':0.,'feedback_not_modified':True,'object_pose_not_modified':True}
        args.output.write_text(json.dumps(fault,indent=2));marker.publish(String(json.dumps(fault)))
        publisher.publish(Float32(0.));print(json.dumps(fault),flush=True)


subscriber=rospy.Subscriber('/tcei/task_status',String,event,queue_size=30)
deadline=time.monotonic()+600
while not rospy.is_shutdown() and not injected and time.monotonic()<deadline:time.sleep(.1)
if not injected:raise RuntimeError('test never reached the selected lift; no drop was injected')
time.sleep(.3)
