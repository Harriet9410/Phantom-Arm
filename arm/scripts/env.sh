#!/usr/bin/env bash
# Only environment setup: no robot, ROS master, model or desktop is started.
export TCEI_PACKAGE_ROOT
TCEI_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$TCEI_PACKAGE_ROOT/state/instance.env" ]] || { printf '%s\n' '请先运行 bash robot.sh init'; return 1; }
source "$TCEI_PACKAGE_ROOT/state/instance.env"
export TCEI_HARNESS="$TCEI_PACKAGE_ROOT/harness"
_tcei_nounset=0
case $- in *u*) _tcei_nounset=1; set +u ;; esac
source /opt/ros/noetic/setup.bash
[[ ! -f /root/jaka/devel/setup.bash ]] || source /root/jaka/devel/setup.bash
[[ $_tcei_nounset != 1 ]] || set -u
unset _tcei_nounset
for _tcei_library in /opt/conda/envs/inference/lib/python3.10/site-packages/nvidia/cudnn/lib /usr/local/cuda/lib64; do
    [[ ! -d $_tcei_library ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_tcei_library"
done
unset _tcei_library
