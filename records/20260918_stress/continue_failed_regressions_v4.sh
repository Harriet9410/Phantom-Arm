#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_RUNS=/root/tcei_stress_20260918
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v4/tcei_stack
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v5/harness
source "$STRESS_HARNESS/env.sh"
tcei_select_run regression_v4_02
python3 -c 'import json,os;from pathlib import Path;d=json.loads((Path(os.environ["TCEI_RUN_DIR"])/"preflight_after_recorder_recovery.json").read_text());assert d["ok"]'
date -Is > "$TCEI_RUN_DIR/stack_ready.txt"
bash "$STRESS_HARNESS/run_case.sh" regression_v4_02
bash "$STRESS_HARNESS/stop_stack.sh" regression_v4_02 --with-sim
export STRESS_CASE_FILE="$TCEI_RUNS/holdout_cases/random_04.json"
bash "$STRESS_HARNESS/start_case.sh" regression_v4_04
bash "$STRESS_HARNESS/run_case.sh" regression_v4_04
bash "$STRESS_HARNESS/stop_stack.sh" regression_v4_04 --with-sim
printf '%s
' 'FAILED_ROUNDS_02_04_RETEST_PASSED'
