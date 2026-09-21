#!/usr/bin/env bash
set -Eeuo pipefail
export TCEI_CODE=/root/tcei_stress_20260918/candidate_v3/tcei_stack
export TCEI_RUNS=/root/tcei_stress_20260918
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v5/harness
source "$STRESS_HARNESS/env.sh"
cd "$TCEI_CODE"
"$TCEI_YOLO_PY" -m unittest discover > "$TCEI_RUNS/v3d_tests.log" 2>&1
tcei_select_run development_pose_c_03
/usr/bin/python3 - <<'PY'
import rospy,json,os
from pathlib import Path
from std_msgs.msg import String
rospy.init_node('tcei_save_clear_observation',anonymous=True)
d=json.loads(rospy.wait_for_message('/tcei/task_status',String,timeout=5).data)
assert d.get('observation_prepared') and d['visibility']['clear']
(Path(os.environ['TCEI_RUN_DIR'])/'observation_prepared.json').write_text(json.dumps(d,indent=2))
PY
bash "$STRESS_HARNESS/run_case.sh" development_pose_c_03
bash "$STRESS_HARNESS/stop_stack.sh" development_pose_c_03 --with-sim
export STRESS_CASE_FILE="$STRESS_HARNESS/cases/random_02.json"
bash "$STRESS_HARNESS/start_case.sh" development_pose_d_02
tcei_select_run development_pose_d_02
/usr/bin/python3 "$STRESS_HARNESS/stress_preflight.py" --timeout 30 --output "$TCEI_RUN_DIR/development_gate.json"
python3 -c 'import json,os;from pathlib import Path;d=json.loads((Path(os.environ["TCEI_RUN_DIR"])/"development_gate.json").read_text());assert d["observations"]["initial_detection_complete"]'
bash "$STRESS_HARNESS/run_case.sh" development_pose_d_02
bash "$STRESS_HARNESS/stop_stack.sh" development_pose_d_02 --with-sim
printf '%s
' 'DEVELOPMENT_V3D_CHECKS_PASSED'
