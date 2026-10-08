#!/usr/bin/env python3
"""DS 本地工具：从服务器下载单个文件（Jupyter Contents API）。
用法： python _ds_get.py <服务器相对 /root 路径> <本地目标路径>
"""
import json
import ssl
import sys
import urllib.request
from pathlib import Path

HOST = "https://zj02.jupyter.gpufree.cn:8443"
BASE = "/notebook/jupyter-tigm5a6o-xmc7rdge"
COOKIE = ("_xsrf=2|895f9d93|7921a5222198ec3266b1470a0b34d26b|1791342490; "
          "access_token=eyJhbGciOiJSUzI1NiIsImtpZCI6Ijk4MmFmNWE1LTc2ZTAtNDZmMy1iOGEyLTdiZjZlYmIyNzdlNiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiMm5UMUZBelViQWFVVlZtbXRNOXQ4dDNrZktxIl0sImNsaWVudF9pZCI6IjJuVDFGQXpVYkFhVVZWbW10TTl0OHQza2ZLcSIsImV4cCI6MTc5MTQ2NjU5NSwiaWF0IjoxNzkwODYyNzYzLCJpc3MiOiJodHRwczovL3d3dy5ncHVmcmVlLmNuIiwianRpIjoiOTkwYWRlOWEtOWRmMy00NDcwLTkyZWYtMjhjY2Q4YWMxZTVkIiwibmJmIjoxNzkwODYyNzYzLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIG9mZmxpbmVfYWNjZXNzIGVtYWlsIHBob25lIHVzZXJfaW5mbyIsInN1YiI6IjEwMjg1NDkyMTQwOTI2NTY2NSIsInVzZXJfaW5mbyI6eyJpZF9jaGVja2VkIjp0cnVlLCJvcGVuaWQiOiJvNHY1ZTY1QkpXN2pRdWVSdVJfeF9ndWVfR1FnIiwid3hfYm91bmQiOnRydWV9fQ.J-jQ-VqRYXe8N17bHLWSNu_eLsix4zStp_0ZLNe7uLGuRYrGWjX2mVGOO8LSIz8QqkzB8r3sLvOdD54Om1aSK7NlXrrJARoCPtP6RXbq8-MsEKV1lHYoCfPZ5wLAWNVns1fCx7QhlyAzoI8KLbVFwVH5_NkRkKSnAufZRcVGQIXlSjXs_X7OQLf_Lsbhg8GKehjGlu7htWDp7pXKrSvdFtJU0q7ZUe8TaHnKuJfWQLLuvAhUsWyy9960NZDF0NVo5gur5NwN-fIBSgiFII8X6PJPq_aDViO4R_Nz6vc5RyKgI6C-RWkCKIWpxM-WXLLgkjrSd73XFzetYPgUJDkyvA; "
          "refresh_token=312a0db0-a922-4f09-9132-3e01b7352083")
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
