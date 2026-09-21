"""One registered scene startup/shutdown check; never starts an episode."""
import argparse,hashlib,json,re,subprocess,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--case',required=True,type=Path);parser.add_argument('--name',required=True)
    args=parser.parse_args();package=args.package.resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,45}',args.name):raise ValueError('invalid name')
    data=Path(json.loads((package/'state/configuration.json').read_text())['runs'])
    out=data/args.name;out.mkdir(exist_ok=False);stack=data/(args.name+'_stack')
    case=args.case.resolve();result={'scope':'initialization_only_no_task_execution',
        'started_at':time.time(),'case':str(case),'case_sha256':hashlib.sha256(case.read_bytes()).hexdigest()}
    started=False
    try:
        command=['bash','robot.sh','start',stack.name,'--case',str(case)]
        with (out/'startup.log').open('xb') as log:
            done=subprocess.run(command,cwd=package,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        result['start_code']=done.returncode
        if done.returncode:raise RuntimeError('initialization failed; preserve stack setup diagnostics')
        started=True
        result['scene']=json.loads((stack/'scene_initialized.json').read_text())
        result['readiness']=json.loads((stack/'read_only_ready.json').read_text())
        assert result['scene']['valid'] is True and len(result['scene']['objects'])==5
        assert result['scene']['case_sha256']==result['case_sha256']
        assert result['readiness']['mode']=='read_only_no_action' and result['readiness']['ready_for_read_only'] is True
        assert not (stack/'round_started.lock').exists()
        result['status']='passed_initialization_only'
    except BaseException as error:result.update(status='failed',error=repr(error))
    finally:
        if started:
            with (out/'shutdown.log').open('xb') as log:
                stopped=subprocess.run(['bash','robot.sh','stop','--stack',stack.name],cwd=package,
                    stdout=log,stderr=subprocess.STDOUT,timeout=120)
            result['stop_code']=stopped.returncode
            if stopped.returncode:result['status']='shutdown_unconfirmed'
        result['finished_at']=time.time();(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('scene','readiness')},ensure_ascii=False),flush=True)
    return 0 if result['status']=='passed_initialization_only' else 1

if __name__=='__main__':raise SystemExit(main())
