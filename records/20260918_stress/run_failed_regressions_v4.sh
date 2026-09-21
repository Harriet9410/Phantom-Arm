#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_RUNS=/root/tcei_stress_20260918
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v4/tcei_stack
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v5/harness
source "$STRESS_HARNESS/env.sh"
python3 - <<'PY'
from pathlib import Path
import json,hashlib
root=Path('/root/tcei_stress_20260918');proof=json.loads((root/'planning_preview_regression.json').read_text());assert all(any(c['valid'] for c in r['checks']) for r in proof)
assert 'Ran 116 tests' in (root/'v4b_tests.log').read_text() and (root/'v4b_tests.log').read_text().rstrip().endswith('OK')
code=root/'candidate_v4/tcei_stack';manifest=json.loads((code/'BUILD_MANIFEST.json').read_text());assert all(hashlib.sha256((code/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
PY
for index in 02 04; do
 export STRESS_CASE_FILE="$TCEI_RUNS/holdout_cases/random_$index.json"
 run="regression_v4_$index"
 bash "$STRESS_HARNESS/start_case.sh" "$run"
 bash "$STRESS_HARNESS/run_case.sh" "$run"
 bash "$STRESS_HARNESS/stop_stack.sh" "$run" --with-sim
done
printf '%s
' 'FAILED_ROUNDS_02_04_RETEST_PASSED'
