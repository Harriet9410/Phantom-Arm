#!/usr/bin/env bash
set -Eeuo pipefail
# Use the image's original desktop and original authentication configuration.
# The vendor mode script may be a readable but non-executable read-only mount.
[[ -r /etc/.mode/mode.sh && -x /etc/gpufree ]] || { printf '%s\n' '镜像自带桌面入口缺失'; exit 1; }
bash /etc/.mode/mode.sh
[[ ! -f /usr/bin/xfsettingsd ]] || chmod u+x /usr/bin/xfsettingsd
exec /etc/gpufree
