#!/usr/bin/python3
"""Read-only native ROS evidence; no control publishers or truth feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import rospy
from std_msgs.msg import String,Bool,Float32,Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock
from probe_idle_stop import TOPICS,plain

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
stream=(out/'events.jsonl').open('x',encoding='utf-8',buffering=1)
lock=threading.RLock();seen=set();last_clock=None;count=0
types={t._type:t for t in (String,Bool,Float32,Float32MultiArray,JointState,PoseStamped,Clock)}
topics={**TOPICS,'/tcei/nine_status':'std_msgs/String'}
required={'/clock','/Jaka/get_jointstate','/Jaka/get_end_effector_pose',
          '/Jaka/get_gripper_efforts','/Jaka/gripper_is_captured','/tcei/joint_diagnostics'}
rospy.init_node('tcei_scalar_evidence',anonymous=True)

def receive(topic,message):
    global last_clock,count
    value=plain(message)
    if topics[topic]=='std_msgs/String':
        try:value=json.loads(message.data)
        except (ValueError,TypeError):pass
    with lock:
        if topic=='/clock':last_clock=message.clock.to_sec()
        row={'topic':topic,'received_wall':time.time(),'received_monotonic':time.monotonic(),
             'last_received_simulation_time':last_clock,'value':value,
             'callerid':getattr(message,'_connection_header',{}).get('callerid')}
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        count+=1;seen.add(topic)
        if required<=seen and not (out/'ready.json').exists():
            (out/'ready.json').write_text(json.dumps({'ready_at':time.time(),'topics_seen':sorted(seen),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'clock_note':'Latest received /clock is a receipt association, not an atomic sensor timestamp.'},indent=2))

subscriptions=[rospy.Subscriber(topic,types[kind],lambda m,t=topic:receive(t,m),queue_size=1000)
               for topic,kind in topics.items()]
try:rospy.spin()
finally:
    for sub in subscriptions:sub.unregister()
    with lock:
        stream.close()
        (out/'closed.json').write_text(json.dumps({'closed_at':time.time(),'recorded_rows':count,'topics_seen':sorted(seen)},indent=2))
