#!/usr/bin/python3
"""Reconstruct one preview from a recorded grasp attempt, never publish pose/gripper/execution."""
from pathlib import Path
import copy
import json
import os
import signal
import sys
import time
import traceback
import uuid
import xmlrpc.client
import rospy
from std_msgs.msg import String,Float32,Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

ROOT=Path('/root/tcei_competition_20260919')
RUNS=Path('/root/gpufree-data/tcei_competition_20260919')
CODE=ROOT/'releases/dev_v7_t1_11/tcei_stack'
sys.path.insert(0,str(CODE))
from ros_endpoints import BoundedXmlRpcTransport,bus_rows,peers


def main():
    out=ROOT/'preview11_D_01';out.mkdir(exist_ok=False)
    result={'status':'failed','motion_commands':[],'scope':'reconstructed recorded attempt, no execution'}
    received=[];subscribers=[];publisher=None
    def alarm(_signum,_frame):raise TimeoutError('35second preview diagnostic wall guard')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,35.)
    try:
        assert json.loads((ROOT/'full11_launcher_result.json').read_text())['returncode']==0
        source=RUNS/'no_action09_01/events/control_events.jsonl';attempt=None
        with source.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('request_id')=='episode_f73a717efbd04e3aa6f6c8a270e2cb93' and row.get('status')=='attempt':
                    attempt=copy.deepcopy(row);break
        assert attempt is not None,'recorded grasp attempt missing'
        from core import calibrated_grasp,grasp_approach_heights
        from observation_view import TRANSFER_POSITION
        p,q=calibrated_grasp(attempt['grasp_candidate']['pose_candidate'],attempt['depth_layer'],attempt['yaw_variant'])
        request={'grasp':p,'above':[p[0],p[1],grasp_approach_heights(p)[0]],'quaternion':q,
                 'transfer':list(TRANSFER_POSITION),'side':'left','place_y':.15}
        result.update(source_attempt=attempt,reconstruction='same recorded attempt and frozen core; place_y .15 from original default; original preview request was not recorded')
        request['id']='diagnostic_'+uuid.uuid4().hex
        rospy.init_node('tcei_preview11_diagnostic',anonymous=True)
        def disabled():
            assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
        disabled()
        subscribers.append(rospy.Subscriber('/tcei/grasp_preview_response',String,
            lambda msg:received.append({'received_monotonic':time.monotonic(),'value':json.loads(msg.data)}),queue_size=10))
        for topic,typ in [('/Jaka/set_end_effector_pose',PoseStamped),('/Jaka/set_gripper_value',Float32),('/tcei/execute_preview',String)]:
            subscribers.append(rospy.Subscriber(topic,typ,lambda msg,t=topic:result['motion_commands'].append(t),queue_size=10))
        publisher=rospy.Publisher('/tcei/grasp_preview_request',String,queue_size=1)
        code,_,state=rospy.get_master().getSystemState();assert code==1
        publishers=dict(state[0]);native=publishers['/Jaka/get_jointstate'];assert len(native)==1
        assert publishers['/tcei/grasp_preview_response']==native
        deadline=time.monotonic()+8.
        while time.monotonic()<deadline:
            with xmlrpc.client.ServerProxy(rospy.get_node_uri(),transport=BoundedXmlRpcTransport(.5)) as api:
                rows=bus_rows(api.getBusInfo(rospy.get_name()))
            if native[0] in peers(rows,'/tcei/grasp_preview_request','o') and peers(rows,'/tcei/grasp_preview_response','i'):break
            time.sleep(.05)
        else:raise TimeoutError('native preview endpoint not connected')
        before=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=3)
        assert rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=3).data is False
        disabled();result['request']=request;result['published_monotonic']=time.monotonic()
        publisher.publish(String(json.dumps(request)))
        while time.monotonic()-result['published_monotonic']<10.:
            matching=[row for row in received if row['value'].get('id')==request['id']]
            if matching:result['reply']=matching[-1];break
            time.sleep(.05)
        else:raise TimeoutError('matching preview response missing')
        after=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=3)
        assert before.name==after.name and len(before.position)==8 and len(after.position)==8
        result['sampled_joint_change']=[abs(a-b) for a,b in zip(before.position,after.position)]
        assert max(result['sampled_joint_change'][:6])<.0015
        assert max(result['sampled_joint_change'][6:])<.0002
        disabled();assert not result['motion_commands'];result['status']='diagnostic_reply_captured_no_execution'
    except BaseException as error:result.update(error=type(error).__name__+': '+str(error),error_traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        for sub in subscribers:sub.unregister()
        if publisher is not None:publisher.unregister()
        rospy.signal_shutdown('single preview diagnostic finished')
        result['finished_at']=time.time();(out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
