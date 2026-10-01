"""Native complete-route preview of measured Magazine poses; no motion publisher."""
from pathlib import Path
import argparse,copy,json,sys,time,uuid,xmlrpc.client
import rospy
from std_msgs.msg import String,Float32,Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--category',choices=('Magazine','Torch','Grenade','Smokegrenade'),default='Magazine')
    parser.add_argument('--side',choices=('left','right'),default='left');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.package/'tcei_stack'))
    from core import calibrated_grasp,grasp_approach_heights,empty_finger_lowest_z
    from observation_view import TRANSFER_POSITION
    from ros_endpoints import BoundedXmlRpcTransport,bus_rows,peers
    rospy.init_node('tcei_tilt23_no_motion',anonymous=True)
    assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
    replies=[];motions=[];subs=[]
    subs.append(rospy.Subscriber('/tcei/grasp_preview_response',String,
        lambda m:replies.append(json.loads(m.data)),queue_size=50))
    for topic,typ in [('/Jaka/set_end_effector_pose',PoseStamped),('/Jaka/set_gripper_value',Float32),('/tcei/execute_preview',String)]:
        subs.append(rospy.Subscriber(topic,typ,lambda m,t=topic:motions.append(t),queue_size=20))
    pub=rospy.Publisher('/tcei/grasp_preview_request',String,queue_size=1)
    code,_,state=rospy.get_master().getSystemState();assert code==1
    publishers=dict(state[0]);native=publishers['/Jaka/get_jointstate'];assert len(native)==1
    assert publishers['/tcei/grasp_preview_response']==native
    until=time.monotonic()+10
    while time.monotonic()<until:
        with xmlrpc.client.ServerProxy(rospy.get_node_uri(),transport=BoundedXmlRpcTransport(.5)) as api:
            rows=bus_rows(api.getBusInfo(rospy.get_name()))
        if native[0] in peers(rows,'/tcei/grasp_preview_request','o') and peers(rows,'/tcei/grasp_preview_response','i'):break
        time.sleep(.05)
    else:raise TimeoutError('native preview subscriber not ready')
    scene=json.loads(rospy.wait_for_message('/tcei/candidates',String,timeout=5).data)
    candidates=[c for c in scene['candidates'] if c['class']==args.category and c.get('identity_status')=='confirmed']
    assert len(candidates)==1 and 0<=time.time()-scene['observed_at']<2.
    candidate=copy.deepcopy(candidates[0]);before=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
    assert not rospy.wait_for_message('/Jaka/gripper_is_captured',Bool,timeout=5).data
    result={'scope':'actual current camera candidate, native full-route preview only, no task or motion execution',
        'candidate':candidate,'trials':[],'motion_commands':motions,'started_at':time.time()}
    deadline=time.monotonic()+100.
    try:
        for tilt in (0.,10.,-10.):
            family_valid=False
            for yaw in (0,180):
                p,q=calibrated_grasp(candidate,0,yaw,tilt)
                for height in grasp_approach_heights(p,q if tilt else None):
                    assert time.monotonic()<deadline
                    assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
                    ident='tilt_probe_'+uuid.uuid4().hex
                    request={'id':ident,'grasp':p,'above':[p[0],p[1],height],'quaternion':q,
                             'transfer':list(TRANSFER_POSITION),'side':args.side,'place_y':.15}
                    pub.publish(String(json.dumps(request)));sent=time.monotonic()
                    while time.monotonic()-sent<10.:
                        matching=[r for r in replies if r.get('id')==ident]
                        if matching:break
                        time.sleep(.03)
                    else:raise TimeoutError('native preview response missing')
                    reply=matching[-1]
                    result['trials'].append({'tilt_deg':tilt,'yaw_variant':yaw,'request':request,'reply':reply,
                        'nominal_open_finger_rim_clearance':empty_finger_lowest_z(request['above'],q)-2.485238,
                        'nominal_support_clearance':empty_finger_lowest_z(p,q)-candidate['support_z']})
                    (args.output/'progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
                    if reply.get('valid'):family_valid=True;break
                if family_valid:break
        after=rospy.wait_for_message('/Jaka/get_jointstate',JointState,timeout=5)
        result['joint_change']=[abs(a-b) for a,b in zip(before.position,after.position)]
        assert len(result['joint_change'])==8 and max(result['joint_change'][:6])<.0015 and max(result['joint_change'][6:])<.0002
        assert not motions
        result['execution_flags_false']=all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
        assert result['execution_flags_false'];result['status']='native_preview_only_completed'
    except BaseException as error:result.update(status='failed',error=repr(error))
    finally:
        for sub in subs:sub.unregister()
        pub.unregister();rospy.signal_shutdown('preview only finished')
        result['finished_at']=time.time();(args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'status':result['status'],'trials':len(result['trials']),
                      'valid':[(r['tilt_deg'],r['yaw_variant'],r['request']['above'][2]) for r in result['trials'] if r['reply'].get('valid')]}),flush=True)
    return 0 if result['status']=='native_preview_only_completed' else 1

if __name__=='__main__':raise SystemExit(main())
