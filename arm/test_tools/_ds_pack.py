#!/usr/bin/env python3
"""DS 打包：把 B1 批次的派生证据 + campaign/run 日志 + 九格冲突图像目录打成一个 tar.gz。
用法：/usr/bin/python3 /root/ds_pack.py <batch> <case...>
"""
import glob
import hashlib
import os
import re
import sys
import tarfile

RUNS = '/root/gpufree-data/tcei_260920v2'
EXP = '/root/gpufree-data/r7cand_export'
OUT = '/root/gpufree-data/ds_r7b_evidence.tar.gz'
DIRPAT = re.compile(r'"directory":\s*"([^"]+)"')


def main():
    batch = sys.argv[1]
    cases = sys.argv[2:]
    members = []
    members += sorted(glob.glob(EXP + '/*'))

    bdir = '%s/%s' % (RUNS, batch)
    for name in ('campaign.json',):
        p = os.path.join(bdir, name)
        if os.path.exists(p):
            members.append(p)
    for case in cases:
        for suf in ('_run.log', '_stop.log', '_start.log', '_instructions.json'):
            p = os.path.join(bdir, case + suf)
            if os.path.exists(p):
                members.append(p)
    # 冲突图像目录（从导出脚本产出的 conflict 行里取 directory）
    dirs = set()
    for case in cases:
        for l in open('%s/conflict_lines_%s.txt' % (EXP, case), errors='ignore'):
            m = DIRPAT.search(l)
            if m:
                dirs.add(m.group(1))
    for d in sorted(dirs):
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for fn in files:
                    members.append(os.path.join(root, fn))
        else:
            print('warn: conflict dir missing:', d)

    print('members:', len(members))
    with tarfile.open(OUT, 'w:gz') as tf:
        for p in members:
            tf.add(p, arcname=os.path.relpath(p, '/root/gpufree-data'))
    h = hashlib.sha256()
    with open(OUT, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    print('out:', OUT, os.path.getsize(OUT), 'bytes')
    print('sha256:', h.hexdigest())
    print('conflict dirs:', len(dirs))


if __name__ == '__main__':
    main()
