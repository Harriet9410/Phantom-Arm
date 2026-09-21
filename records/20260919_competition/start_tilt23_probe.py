"""Install isolated candidate23 and run a bounded preview-only old24 check."""
from pathlib import Path
import hashlib,json,os,subprocess,time,zipfile

ROOT=Path('/root/tcei_competition_20260919')
DEST=Path('/root/tcei_tilt23')
PACKAGE=DEST/'tcei_260920v2'
ARCHIVE=ROOT/'260920v2-candidate03b.zip'

def main():
    campaign=Path('/root/gpufree-data/tcei_260920v2/old_failed_v2_01/campaign.json')
    assert json.loads(campaign.read_text())['status']=='completed'
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()=='03b2631c53c32fa5c64be871242a404643eda4e7d74300703a318cf7ff620575'
    assert not DEST.exists();DEST.mkdir()
    with zipfile.ZipFile(ARCHIVE) as z:
        assert all(n.startswith('tcei_260920v2/') and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(DEST)
    steps=[['/usr/bin/python3','scripts/verify_package.py'],
           ['bash','robot.sh','init','--display',':20','--data-root','/root/gpufree-data/tcei_tilt23'],
           ['bash','robot.sh','doctor'],
           ['bash','robot.sh','start','tilt23_old24_probe_stack01','--case','cases_b01/random_24.json']]
    reports=[]
    for index,command in enumerate(steps):
        with (PACKAGE/('probe_setup_%d.log'%index)).open('xb') as log:
            result=subprocess.run(command,cwd=PACKAGE,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        reports.append({'command':command,'returncode':result.returncode,'time':time.time()})
        (PACKAGE/'probe_setup_results.json').write_text(json.dumps(reports,indent=2))
        if result.returncode:raise RuntimeError('preview stack preparation failed')
    try:
        command=['bash','-c','source scripts/env.sh\nexec /usr/bin/python3 evaluation/probe_tilt23.py --package "$TCEI_PACKAGE_ROOT" --output "$TCEI_RUNS/tilt23_old24_preview01"']
        with (PACKAGE/'probe24_native.log').open('xb') as log:
            result=subprocess.run(command,cwd=PACKAGE,stdout=log,stderr=subprocess.STDOUT,timeout=150)
        reports.append({'phase':'preview_only','returncode':result.returncode,'time':time.time()})
    finally:
        with (PACKAGE/'probe24_stop.log').open('xb') as log:
            stopped=subprocess.run(['bash','robot.sh','stop'],cwd=PACKAGE,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        reports.append({'phase':'stop','returncode':stopped.returncode,'time':time.time()})
        (PACKAGE/'probe_setup_results.json').write_text(json.dumps(reports,indent=2))
    print('TILT23_PROBE_FINISHED',flush=True)

if __name__=='__main__':main()
