#!/usr/bin/env python3
"""DS 本地工具：通用单文件上传到服务器（Jupyter Contents API，base64，字节精确）。

用法： python _ds_put.py <本地文件> <服务器相对 /root 的路径>
例：   python _ds_put.py ../../arm/test_tools/_ds_r7cand_export.py ds_r7cand_export.py
"""
import base64
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
XSRF = next((v.split("=", 1)[1] for v in COOKIE.split(";") if v.strip().startswith("_xsrf=")), "").strip()
if not XSRF:
    raise SystemExit("GPUFREE_COOKIE 中缺少 _xsrf。")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def main():
    local, remote = sys.argv[1], sys.argv[2]
    blob = Path(local).read_bytes()
    url = "%s%s/api/contents/%s?content=1" % (HOST, BASE, remote)
    body = json.dumps({"type": "file", "format": "base64",
                       "content": base64.b64encode(blob).decode()}).encode()
    req = urllib.request.Request(url, data=body, method="PUT")
    req.add_header("Cookie", COOKIE)
    req.add_header("X-XSRFToken", XSRF)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        print("OK http=%s remote=%s bytes=%d" % (r.status, remote, len(blob)))


if __name__ == "__main__":
    main()
