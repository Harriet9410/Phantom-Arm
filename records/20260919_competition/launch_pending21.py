"""Use the frozen19 services with an explicit observation-gap test wrapper."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('/root/tcei_competition_20260919');REL=ROOT/'releases/dev_v7_t1_21'
sys.path.insert(0,str(REL/'harness'))
import launch_stack
normal_commands=launch_stack.commands

def fault_commands(config,run_dir,case_file=None):
    assert Path(config['code']).resolve()==(REL/'tcei_stack').resolve() and case_file is None
    spec=normal_commands(config,run_dir,case_file)
    wrapper=ROOT/'fault_controller_entry21.py';gap=ROOT/'delivery_evidence_gap.py'
    altered=[]
    for name,cwd,command in spec:
        if name=='controller':
            command=list(command);assert command[2]==str(REL/'tcei_stack/controller.py')
            command[2]=str(wrapper)
            command+=['_fault_mode:=hide_first_delivery_observations','_fault_log:='+str(Path(run_dir)/'fault_injection.jsonl')]
        altered.append((name,cwd,command))
    (Path(run_dir)/'FAULT_TEST_ONLY.json').open('x').write(json.dumps({'normal_competition_run':False,
        'fault':'withhold only first release-bound belt observations from controller; preserve raw camera/candidates and all other releases',
        'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        'gap_sha256':hashlib.sha256(gap.read_bytes()).hexdigest()},indent=2))
    return altered

launch_stack.commands=fault_commands
if __name__=='__main__':launch_stack.main()
