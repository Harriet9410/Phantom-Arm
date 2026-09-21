export TCEI_RUNS=/root/tcei_stress_20260918
export TCEI_CODE=/root/tcei_stress_20260918/frozen_v5/tcei_stack
export STRESS_HARNESS=/root/tcei_stress_20260918/harness_v6/harness
source "$STRESS_HARNESS/env.sh"
cd /root/EAICON
exec /root/isaacsim/python.sh /root/Pictures/planning_probe_08_09.py
