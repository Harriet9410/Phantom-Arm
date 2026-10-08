#!/usr/bin/env python3
"""DS 检视：打印某时间窗内的 frame / lineage / audit 原始行。
用法：/usr/bin/python3 /root/ds_window.py <case> <lo_epoch> <hi_epoch>
"""
import re
import sys

EXP = '/root/gpufree-data/r7cand_export'
TSPAT = re.compile(r'\[(\d+\.\d+)\]')


def main():
    case, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    for name in ('frame_lines_%s.txt', 'lineage_lines_%s.txt', 'audit_lines_%s.txt'):
        p = '%s/%s' % (EXP, name % case)
        print('=== %s ===' % (name % case))
        try:
            for l in open(p, errors='ignore'):
                m = TSPAT.search(l)
                if m and lo <= float(m.group(1)) <= hi:
                    print(l.rstrip()[:300])
        except FileNotFoundError:
            print('(missing)')


if __name__ == '__main__':
    main()
