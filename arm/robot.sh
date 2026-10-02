#!/usr/bin/env bash
set -Eeuo pipefail
_tcei_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_tcei_command="${1:-help}"
[[ $# == 0 ]] || shift
case "$_tcei_command" in
  init) exec /usr/bin/python3 "$_tcei_root/scripts/setup.py" "$@" ;;
  desktop) exec /usr/bin/python3 "$_tcei_root/scripts/desktop.py" "$@" ;;
  help) printf '%s\n' '用法：bash robot.sh init|doctor|start|run|status|cancel|stop|panel|desktop [参数]'; exit 0 ;;
esac
source "$_tcei_root/scripts/env.sh"
case "$_tcei_command" in
  doctor) exec /usr/bin/python3 "$_tcei_root/scripts/doctor.py" "$@" ;;
  start) exec /usr/bin/python3 "$_tcei_root/scripts/start.py" "$@" ;;
  run) exec /usr/bin/python3 "$_tcei_root/scripts/run_round.py" "$@" ;;
  status) exec /usr/bin/python3 "$_tcei_root/scripts/status.py" "$@" ;;
  cancel) exec /usr/bin/python3 "$_tcei_root/scripts/cancel.py" "$@" ;;
  stop) exec /usr/bin/python3 "$_tcei_root/scripts/stop.py" "$@" ;;
  panel) exec /usr/bin/python3 "$_tcei_root/scripts/panel.py" "$@" ;;
  console) exec /usr/bin/python3 "$_tcei_root/evaluation/instruction_console.py" "$@" ;;
  drill) exec /usr/bin/python3 "$_tcei_root/scripts/drill.py" "$@" ;;
  *) printf '%s\n' "未知命令：$_tcei_command" >&2; exit 2 ;;
esac
