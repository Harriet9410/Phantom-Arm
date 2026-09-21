from pathlib import Path
import json,subprocess,sys,os,time
p=Path('/root/tcei_package_validation01/tcei_260920v2');r=Path('/root/tcei_competition_20260919');old=Path('/root/gpufree-data/tcei_competition_20260919/pending21_stack01');finished=old.parent/'pending21_A01/supervisor_finished.json'
sys.path.insert(0,str(p/'scripts'))
from common import current,read_record,guard,write_new
from cancel import cancel_and_confirm
import rospy
rospy.init_node('tcei_package_native_validation',anonymous=True)
assert json.loads(finished.read_text())['evidence_complete'] is True
assert json.loads(finished.read_text())['episode_status']=='completed_with_unverified_placements'
assert all(rospy.get_param(k) is False for k in ('/tcei_controller/execute','/tcei_nine/execute'))
assert all(current(read_record(old/'pids'/(n+'.json'))) for n in ('nine','controller','perception','sim'))
proof=cancel_and_confirm('closed pending21 fixture; begin normal package entry validation');write_new(r/'package_candidate01_prior_stop.json',proof);assert proof['state']=='stopped',proof
stops=[]
for n in ('nine','controller','perception','sim'):
 result=guard('stop',old/'pids'/(n+'.json'),'--timeout','20');stops.append(dict(name=n,code=result.returncode,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0
write_new(r/'package_candidate01_prior_retired.json',stops);rospy.set_param('/tcei_controller/observation_ready',False);rospy.signal_shutdown('old fixture closed')
steps=[['bash','robot.sh','start','package_smoke_stack01'],['bash','robot.sh','run','package_smoke_round01','--official-example'],['bash','robot.sh','stop']]
reports=[]
for command in steps:
 with (p/('native_'+command[2]+'.log')).open('xb') as log:result=subprocess.run(command,cwd=str(p),env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=750)
 reports.append(dict(command=command,returncode=result.returncode,finished_at=time.time()));(r/'package_candidate01_native_status.json').write_text(json.dumps(reports,indent=2))
 if result.returncode and command[2]!='run':break
print('PACKAGE_NATIVE_FINISHED',reports,flush=True)
