"""Explicit engineering fault fixture; NEVER use this as the normal entry."""
from pathlib import Path
import hashlib,json,os,sys,threading,time
import rospy
from std_msgs.msg import String

code=Path(os.environ['TCEI_CODE']).resolve()
assert code.name=='tcei_stack' and code.parent.name=='dev_v7_t1_21'
sys.path.insert(0,str(code))
import controller as production
from delivery_evidence_gap import DeliveryEvidenceGap

class EvidenceGapController(production.Controller):
    def __init__(self):
        assert rospy.get_param('~fault_mode',None)=='hide_first_delivery_observations'
        self.gap=DeliveryEvidenceGap();self.gap_lock=threading.RLock()
        self.gap_file=Path(rospy.get_param('~fault_log')).open('x',encoding='utf-8',buffering=1)
        self.record_gap('fixture_started',mode='hide_first_delivery_observations',
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            production_controller_sha256=hashlib.sha256((code/'controller.py').read_bytes()).hexdigest(),
            raw_images_changed=False,robot_commands_injected=False)
        super().__init__()
    def record_gap(self,status,**fields):
        self.gap_file.write(json.dumps({'time':time.time(),'monotonic':time.monotonic(),'status':status,**fields},
                                       ensure_ascii=False,allow_nan=False)+'\n')
    def event(self,status,**fields):
        if status=='release_started':
            with self.gap_lock:
                if self.gap.select_first_release(fields['release_id']):
                    self.record_gap('gap_armed',release_id=self.gap.release_id,request_id=self.current_request,
                                    task_id=self.current_task,side=fields.get('side'),category=fields.get('category'))
        return super().event(status,**fields)
    def on_candidates(self,message):
        scene=json.loads(message.data)
        with self.gap_lock:
            filtered,removed=self.gap.filter_scene(scene)
            if removed:self.record_gap('observations_withheld',release_id=self.gap.release_id,
                frame_id=scene['frame_id'],image_stamp=scene.get('stamp'),observed_at=scene.get('observed_at'),
                original_observations=removed,source_candidates_unchanged=True)
        return super().on_candidates(String(json.dumps(filtered,ensure_ascii=False)))

if __name__=='__main__':
    rospy.init_node('tcei_controller');node=EvidenceGapController()
    try:node.run()
    finally:node.gap_file.close()
