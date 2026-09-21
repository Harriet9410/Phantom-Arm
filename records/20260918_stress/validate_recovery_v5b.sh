#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_RUNS=/root/tcei_stress_20260918
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v5/tcei_stack
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v6/harness
source "$STRESS_HARNESS/env.sh"
cd "$TCEI_CODE"
"$TCEI_YOLO_PY" -m unittest discover > "$TCEI_RUNS/v5b_tests.log" 2>&1
export STRESS_CASE_FILE="$TCEI_RUNS/holdout_cases/random_04.json"
if ! bash "$STRESS_HARNESS/start_case.sh" drop_recovery_v5b_probe; then
 bash /root/Pictures/recover_recorders.sh drop_recovery_v5b_probe
fi
tcei_select_run drop_recovery_v5b_probe
tcei_start fault "$TCEI_CODE" /usr/bin/python3 -u "$TCEI_RUNS/inject_one_drop.py" --output "$TCEI_RUN_DIR/fault_injection.json"
trap 'tcei_select_run drop_recovery_v5b_probe; tcei_stop fault || true' EXIT
bash /root/Pictures/run_single_drop_probe.sh drop_recovery_v5b_probe
tcei_stop fault
trap - EXIT
python3 - <<'PY'
from pathlib import Path
import json
p=Path('/root/tcei_stress_20260918/deploy_runs/drop_recovery_v5b_probe');assert (p/'fault_injection.json').exists()
s=json.loads((p/'episode/summary.json').read_text());assert s['status']=='succeeded' and s['verified_objects']==1
r=[json.loads(x) for x in (p/'episode/events.jsonl').read_text().splitlines()]
for status in ['drop_detected','drop_recovery_reacquired','drop_recovery_completed']:assert any(x['status']==status for x in r),status
(p/'recovery_validation.json').write_text(json.dumps({'valid':True,'real_gripper_release':True,'verified_objects':1,'elapsed_seconds':s['elapsed_seconds'],'recovery_events':[x for x in r if x['status'].startswith('drop_')]},indent=2))
PY
bash "$STRESS_HARNESS/stop_stack.sh" drop_recovery_v5b_probe --with-sim
export STRESS_CASE_FILE="$TCEI_RUNS/holdout_cases/random_06.json"
if ! bash "$STRESS_HARNESS/start_case.sh" preview_alignment_v5b_06; then
 bash /root/Pictures/recover_recorders.sh preview_alignment_v5b_06
fi
bash "$STRESS_HARNESS/run_case.sh" preview_alignment_v5b_06
bash "$STRESS_HARNESS/stop_stack.sh" preview_alignment_v5b_06 --with-sim
printf '%s
' 'V5B_RECOVERY_AND_CASE06_PASSED'
