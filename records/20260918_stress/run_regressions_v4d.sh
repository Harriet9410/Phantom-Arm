#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_RUNS=/root/tcei_stress_20260918
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v4/tcei_stack
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v5/harness
source "$STRESS_HARNESS/env.sh"
bash "$STRESS_HARNESS/stop_stack.sh" regression_v4c_04 --with-sim
python3 - <<'PY'
from pathlib import Path
import tarfile,hashlib,shutil
root=Path('/root/tcei_stress_20260918');shutil.copytree(root/'candidate_v4',root/'candidate_v4_before_path_priority_fix')
p=Path('/root/Pictures/planning_v4d.tar.gz');assert hashlib.sha256(p.read_bytes()).hexdigest()=='c2a3a309f62f39af9acc21c261d889c7c70d0c081d78e1cdf207a4d26aa372ce'
with tarfile.open(p) as t:
 assert all(not n.name.startswith('/') and '..' not in n.name.split('/') and n.isfile() for n in t.getmembers());t.extractall(root)
PY
cd "$TCEI_CODE"
"$TCEI_YOLO_PY" -m unittest discover > "$TCEI_RUNS/v4d_tests.log" 2>&1
bash -n /root/Pictures/recover_recorders.sh
for index in 04 02; do
 export STRESS_CASE_FILE="$TCEI_RUNS/holdout_cases/random_$index.json"
 run="regression_v4d_$index"
 if ! bash "$STRESS_HARNESS/start_case.sh" "$run"; then
  bash /root/Pictures/recover_recorders.sh "$run"
 fi
 bash "$STRESS_HARNESS/run_case.sh" "$run"
 bash "$STRESS_HARNESS/stop_stack.sh" "$run" --with-sim
done
printf '%s
' 'REGRESSIONS_V4D_04_02_PASSED'
