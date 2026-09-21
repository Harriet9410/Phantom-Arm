#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v3/tcei_stack
export TCEI_RUNS=/root/tcei_stress_20260918
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v2/harness
export STRESS_CASE_FILE="$STRESS_HARNESS/cases/random_01.json"
source "$STRESS_HARNESS/env.sh"
bash "$STRESS_HARNESS/stop_stack.sh" pilot_setup02 --with-sim
cp -a /root/tcei_stress_20260918/candidate_v3 /root/tcei_stress_20260918/candidate_v3_before_frame_fix
python3 - <<'PY'
import tarfile,hashlib
from pathlib import Path
p=Path('/root/Pictures/recognition_pose_v3b.tar.gz');assert hashlib.sha256(p.read_bytes()).hexdigest()=='618b7d5f71fc07fe6da6211a09a6181cf77c6659bf053e67c8a81d8b84607c41'
with tarfile.open(p) as t:
 assert all(not n.name.startswith('/') and '..' not in n.name.split('/') and n.isfile() for n in t.getmembers());t.extractall('/root/tcei_stress_20260918')
PY
cd "$TCEI_CODE"
"$TCEI_YOLO_PY" -m unittest discover > "$TCEI_RUNS/v3b_tests.log" 2>&1
bash "$STRESS_HARNESS/start_case.sh" pilot_pose03
bash "$STRESS_HARNESS/run_case.sh" pilot_pose03
