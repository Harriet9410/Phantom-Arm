"""Request controller-owned cancellation and verify its matching stop acknowledgement."""
import argparse,json,threading,time,uuid
from common import ROOT,stack_path,owned,write_new


def cancel_and_confirm(reason,timeout=10.):
    import rospy
    from std_msgs.msg import String
    from ros_endpoints import CANCEL_PROTOCOL,BoundedXmlRpcTransport,bus_rows,missing_peers
    import xmlrpc.client
    if not rospy.core.is_initialized():rospy.init_node('tcei_package_cancel',anonymous=True)
    lock=threading.Lock();events=[]
    def receive(topic,msg):
        try:value=json.loads(msg.data)
        except ValueError:return
        with lock:events.append((time.monotonic(),topic,value))
    subs=[rospy.Subscriber(t,String,lambda m,t=t:receive(t,m),queue_size=50)
          for t in ('/tcei/cancel_ack','/tcei/stop_ack')]
    pub=rospy.Publisher('/tcei/cancel_request',String,queue_size=1)
    request={'protocol':CANCEL_PROTOCOL,'cancel_id':uuid.uuid4().hex,
        'round_id':'package_'+uuid.uuid4().hex,'request_id':'package_cancel','reason':reason[:1000]}
    began=time.monotonic();sent=None;mapping=None
    try:
        while time.monotonic()-began<3.:
            with xmlrpc.client.ServerProxy(rospy.get_node_uri(),transport=BoundedXmlRpcTransport(.5)) as api:
                missing=missing_peers(bus_rows(api.getBusInfo(rospy.get_name())),
                    '/tcei/cancel_request',require_response=False)
            if not missing:
                sent=time.monotonic();pub.publish(String(json.dumps(request)));break
            time.sleep(.05)
        if sent is None:return {'state':'unconfirmed','reason':'controller cancellation receiver not connected'}
        while time.monotonic()-sent<timeout:
            with lock:rows=list(events)
            for at,topic,value in rows:
                if at<sent:continue
                if topic=='/tcei/cancel_ack' and value.get('cancel_id')==request['cancel_id']:
                    if value.get('state')=='rejected':return {'state':'unconfirmed','reason':value.get('reason')}
                    if value.get('state')=='accepted':mapping=value
            if mapping:
                matches=[(at,v) for at,t,v in rows if at>=sent and t=='/tcei/stop_ack'
                         and v.get('id')==mapping.get('stop_id')]
                if matches:
                    at,proof=matches[-1]
                    if time.monotonic()-at<1. and proof.get('state') in ('stopped','fault'):
                        return {'state':proof['state'],'proof':proof,'mapping':mapping,
                                'seconds':time.monotonic()-sent,'request':request}
            time.sleep(.03)
        return {'state':'unconfirmed','mapping':mapping,'request':request,'reason':'stop acknowledgement timeout'}
    finally:
        for sub in subs:sub.unregister()
        pub.unregister()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stack');p.add_argument('--reason',default='operator package cancel')
    args=p.parse_args();stack=stack_path(args.stack)
    if not owned(stack,('controller',))['controller']:raise RuntimeError('当前部署没有存活的控制器')
    result=cancel_and_confirm(args.reason)
    import rospy
    for key in ('/tcei_controller/execute','/tcei_nine/execute'):rospy.set_param(key,False)
    write_new(stack/('cancel_'+str(time.time_ns())+'.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['state']=='stopped' else 1

if __name__=='__main__':raise SystemExit(main())
