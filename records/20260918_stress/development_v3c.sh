#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v3/tcei_stack
export TCEI_RUNS=/root/tcei_stress_20260918
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v4/harness
source "$STRESS_HARNESS/env.sh"
bash "$STRESS_HARNESS/stop_stack.sh" development_pose_02 --with-sim
python3 /root/tcei_stress_20260918/prepare_v3c.py
cd "$TCEI_CODE"
"$TCEI_YOLO_PY" -m unittest discover > "$TCEI_RUNS/v3c_tests.log" 2>&1
for index in 02 03; do
 export STRESS_CASE_FILE="$STRESS_HARNESS/cases/random_$index.json"
 run="development_pose_c_$index"
 bash "$STRESS_HARNESS/start_case.sh" "$run"
 tcei_select_run "$run"
 /usr/bin/python3 "$STRESS_HARNESS/stress_preflight.py" --timeout 30 --output "$TCEI_RUN_DIR/development_gate.json"
 python3 -c 'import json,os;from pathlib import Path;d=json.loads((Path(os.environ["TCEI_RUN_DIR"])/"development_gate.json").read_text());assert d["observations"]["initial_detection_complete"],"development scene incomplete detection"'
 bash "$STRESS_HARNESS/run_case.sh" "$run"
 bash "$STRESS_HARNESS/stop_stack.sh" "$run" --with-sim
done
printf '%s
' 'DEVELOPMENT_ANGLE_CHECKS_PASSED'
