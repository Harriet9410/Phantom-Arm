#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_RUNS=/root/tcei_stress_20260918
export TCEI_CODE=/root/tcei_stress_20260918/frozen_v3/tcei_stack
source /root/tcei_stress_20260918/harness_v5/harness/env.sh
bash "$TCEI_SCRIPTS/stop_stack.sh" formal30_v3_05 --with-sim
python3 - <<'PY'
from pathlib import Path
import tarfile,hashlib
p=Path('/root/Pictures/planning_v4.tar.gz');assert hashlib.sha256(p.read_bytes()).hexdigest()=='015bc68218ddf10da299aa4cb0f5e0c71f3d0b6186cf8d588de4aea5f0696119'
with tarfile.open(p) as t:
 assert all(not n.name.startswith('/') and '..' not in n.name.split('/') and n.isfile() for n in t.getmembers());t.extractall('/root/tcei_stress_20260918')
PY
cd /root/tcei_stress_20260918/candidate_v4/tcei_stack
"$TCEI_YOLO_PY" -m unittest discover > /root/tcei_stress_20260918/v4_tests.log 2>&1
cd /root/EAICON
"$ISAACSIM_PYTHON_EXE" /root/tcei_stress_20260918/planning_probe_v4.py
