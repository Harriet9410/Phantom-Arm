#!/usr/bin/env bash
# 云服务器网络体检（算力自由容器内执行）
# 用法: bash net_check.sh

echo "==== DNS / ICMP ===="
ping -c 3 gitee.com || echo "gitee ping FAIL"
nslookup gitee.com || true
ping -c 3 github.com || echo "github ping FAIL"
nslookup github.com || true

echo ""
echo "==== HTTPS ===="
echo -n "gitee:   "
curl -I --connect-timeout 8 -s -o /dev/null -w "%{http_code}\n" https://gitee.com || echo "FAIL"
echo -n "github:  "
curl -I --connect-timeout 8 -s -o /dev/null -w "%{http_code}\n" https://github.com || echo "FAIL"

echo ""
echo "==== TCP 端口 ===="
timeout 5 bash -c 'echo >/dev/tcp/gitee.com/443' && echo "gitee:443 OK" || echo "gitee:443 FAIL"
timeout 5 bash -c 'echo >/dev/tcp/github.com/443' && echo "github:443 OK" || echo "github:443 FAIL"
timeout 5 bash -c 'echo >/dev/tcp/github.com/22' && echo "github:22 OK" || echo "github:22 FAIL"

echo ""
echo "==== 结论建议 ===="
echo "gitee 通 → git 用 https://gitee.com/<user>/<repo>.git"
echo "github 443 不通 → 不要死磕 clone，用 Gitee 导入或镜像"
