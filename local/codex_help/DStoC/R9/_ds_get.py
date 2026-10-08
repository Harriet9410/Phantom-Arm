#!/usr/bin/env python3
"""DS 本地工具：从服务器下载单个文件（Jupyter Contents API）。
用法： python _ds_get.py <服务器相对 /root 路径> <本地目标路径>
"""
import json
import os
import ssl
import sys
import urllib.request
from pathlib import Path

HOST = "https://zj02.jupyter.gpufree.cn:8443"
BASE = "/notebook/jupyter-tigm5a6o-xmc7rdge"
COOKIE = os.environ.get("GPUFREE_COOKIE", "")
if not COOKIE:
    raise SystemExit("请通过 GPUFREE_COOKIE 环境变量提供当前会话凭据；勿写入 Git。")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def main():
    remote, local = sys.argv[1], sys.argv[2]
    url = "%s%s/api/contents/%s?content=1&format=base64" % (HOST, BASE, remote)
    req = urllib.request.Request(url)
    req.add_header("Cookie", COOKIE)
    with urllib.request.urlopen(req, context=CTX, timeout=180) as r:
        model = json.loads(r.read().decode())
    import base64
    blob = base64.b64decode(model["content"])
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    Path(local).write_bytes(blob)
    print("OK remote=%s bytes=%d -> %s" % (remote, len(blob), local))


if __name__ == "__main__":
    main()
