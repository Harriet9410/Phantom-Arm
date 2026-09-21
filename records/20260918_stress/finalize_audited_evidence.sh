#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/ros/noetic/setup.bash
/usr/bin/python3 /root/tcei_stress_20260918/analysis_tools_final/analyze_multiversion.py --root /root/tcei_stress_20260918 --output /root/tcei_stress_20260918/analysis_final > /root/tcei_stress_20260918/analysis_verified.log
/usr/bin/python3 /root/tcei_stress_20260918/analysis_tools_final/completion_audit.py > /root/tcei_stress_20260918/COMPLETION_AUDIT.log
/usr/bin/python3 /root/tcei_stress_20260918/analysis_tools_final/collect_evidence.py --root /root/tcei_stress_20260918 --output /root/Pictures/tcei_stress_final_verified_20260918.zip > /root/tcei_stress_20260918/evidence_export_verified.log
/usr/bin/python3 - <<'PYINNER'
from pathlib import Path
import zipfile,json,hashlib
r=Path('/root/tcei_stress_20260918');out=Path('/root/Pictures/tcei_stress_results_essentials_20260918.zip');assert not out.exists()
files=set(p for base in (r/'analysis_final',r/'holdout_cases',r/'frozen_v6/tcei_stack') for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.py','.txt'))
for n in ('COMPLETION_AUDIT.json','COMPLETION_AUDIT.log','V6_FREEZE_AND_RESUME.json','initialization_retries.json','v6_tests.log','planner_cases_08_09.json','planning_probe_08_09.json','preflight_case08_v6.json'):files.add(r/n)
meta=json.loads((r/'analysis_final/first_attempts_campaign.json').read_text())
for row in meta['results']:
 run=r/'deploy_runs'/row['run']
 for n in ('episode/summary.json','episode/events.jsonl','events/control_events.jsonl','scene_initialized.json','preflight_before_round.json','observation_prepared.json'):
  p=run/n
  if p.exists():files.add(p)
manifest=[]
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):
  data=p.read_bytes();name=p.relative_to(r).as_posix();z.writestr(name,data);manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 z.writestr('ESSENTIALS_MANIFEST.json',json.dumps(manifest,indent=2))
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(files)}))
PYINNER
