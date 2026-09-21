from pathlib import Path
import subprocess,json,time,hashlib
r=Path('/root/tcei_competition_20260919');assert json.loads((r/'full07_launcher_result.json').read_text())['returncode']==0;checks=[]
def run(label,cmd,result_path,expected):
 with (r/(label+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=150)
 report=json.loads(result_path.read_text());checks.append({'label':label,'returncode':p.returncode,'status':report.get('status'),'result':str(result_path)});(r/'full07_preparation_checks.json').write_text(json.dumps(checks,indent=2));assert p.returncode==0 and report['status']==expected
assert hashlib.sha256((r/'probe_idle_stop.py').read_bytes()).hexdigest()=='4a804eca7d637eff9e829a22a39b047b7cac431d300e36db392c63a1e81d2b2a';assert hashlib.sha256((r/'probe_dynamic_stop.py').read_bytes()).hexdigest()=='d9d968cf08c4e833736d93778ccf3702400b5d4d74fbd1815b9dc56a6e84e0a7'
run('H01_idle_A_03',['/usr/bin/python3','-u',str(r/'probe_idle_stop.py'),'--outdir',str(r/'H01_idle_A_03')],r/'H01_idle_A_03/summary.json','passed_idle_api')
run('H01_dynamic_B_02',['/usr/bin/python3','-u',str(r/'probe_dynamic_stop.py'),'--idle-summary',str(r/'H01_idle_A_03/summary.json'),'--outdir',str(r/'H01_dynamic_B_02')],r/'H01_dynamic_B_02/summary.json','passed_dynamic_short_move')
p=r/'probe_observation07_C_01.py';source=(r/'probe_observation04_C_01.py').read_text().replace('H01_dynamic_B_01','H01_dynamic_B_02').replace('observation04_C_01','observation07_C_01');p.open('x').write(source);run('observation07_C_01',['/usr/bin/python3','-u',str(p)],r/'observation07_C_01/summary.json','observation_completed')
p=r/'probe_capture07_clear01.py';source=(r/'probe_capture06_clear.py').read_text().replace('t1_capture06_clear01','t1_capture07_clear01').replace('tcei_capture06_clear','tcei_capture07_clear').replace("if time.time()-s['observed_at']<2:","if time.time()-s['observed_at']<2 and s['scene_complete'] is True and all(c['identity_status']=='confirmed' for c in s['candidates']):");p.open('x').write(source)
with (r/'capture07_clear01.log').open('x') as log:p2=subprocess.run(['/usr/bin/python3','-u',str(p)],stdout=log,stderr=subprocess.STDOUT,timeout=25)
assert p2.returncode==0;print('FULL07_PREPARATION_AND_CAPTURE_COMPLETE',flush=True)
