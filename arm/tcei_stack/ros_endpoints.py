"""Actual TCPROS peer checks; observers are never command receivers."""
import http.client
import xmlrpc.client


CANCEL_PROTOCOL='tcei.cancel.v1'
OBSERVATION_PROTOCOL='tcei.observation.v1'


class BoundedXmlRpcTransport(xmlrpc.client.Transport):
    """Bound this diagnostic RPC only; do not change global ROS socket timeouts."""
    def __init__(self,timeout=.5):
        super().__init__();self.timeout=timeout

    def make_connection(self,host):
        if self._connection and host==self._connection[0]:return self._connection[1]
        chost,self._extra_headers,_=self.get_host_info(host)
        connection=http.client.HTTPConnection(chost,timeout=self.timeout)
        self._connection=(host,connection)
        return connection


def bus_rows(reply):
    if not isinstance(reply,(list,tuple)) or len(reply)!=3 or reply[0]!=1 or not isinstance(reply[2],list):
        raise ValueError('invalid node getBusInfo response')
    if any(not isinstance(row,(list,tuple)) or len(row)<6 for row in reply[2]):
        raise ValueError('invalid TCPROS connection row')
    return reply[2]


def peers(rows,topic,direction):
    return {row[1] for row in rows if row[2]==direction and row[3]=='TCPROS'
            and row[4]==topic and row[5] is True and isinstance(row[1],str)}


OUTGOING={'/tcei/request':'/tcei_nine','/tcei/prepare_observation':'/tcei_controller',
          '/tcei/cancel_request':'/tcei_controller'}
INCOMING={'/tcei/nine_status':'/tcei_nine','/tcei/task_status':'/tcei_controller',
          '/tcei/cancel_ack':'/tcei_controller','/tcei/candidates':'/tcei_perception'}


def missing_peers(rows,command=None,require_response=True,node_uris=None):
    outgoing=OUTGOING if command is None else {command:OUTGOING[command]}
    incoming=INCOMING if command is None else {
        '/tcei/request':{'/tcei/nine_status':'/tcei_nine'},
        '/tcei/prepare_observation':{'/tcei/task_status':'/tcei_controller'},
        '/tcei/cancel_request':{'/tcei/cancel_ack':'/tcei_controller'}}[command]
    missing=[]
    for topic,node in outgoing.items():
        if node not in peers(rows,topic,'o'):missing.append('outgoing '+topic+' -> '+node)
    if require_response:
        for topic,node in incoming.items():
            # Noetic reports incoming peers by their XML-RPC URI, whereas
            # outgoing peers use caller IDs. Bind URIs through lookupNode.
            allowed={node}
            uri=(node_uris or {}).get(node)
            if uri:allowed.add(uri)
            actual=peers(rows,topic,'i')
            if len(actual)!=1 or not actual<=allowed:missing.append('incoming '+topic+' <- '+node)
    if command is None:
        capture=peers(rows,'/Jaka/gripper_is_captured','i')
        stop=peers(rows,'/tcei/stop_ack','i')
        if len(capture)!=1 or stop!=capture:
            missing.append('capture and stop feedback from the same single native simulator')
    return missing


def cancel_payload(value):
    if not isinstance(value,dict) or value.get('protocol')!=CANCEL_PROTOCOL:
        raise ValueError('unsupported cancellation protocol')
    for key in ('cancel_id','round_id','request_id'):
        if not isinstance(value.get(key),str) or not 0<len(value[key])<=128:
            raise ValueError('cancellation requires '+key)
    if not isinstance(value.get('reason'),str) or len(value['reason'])>1000:
        raise ValueError('invalid cancellation reason')
    return {key:value[key] for key in ('protocol','cancel_id','round_id','request_id','reason')}
