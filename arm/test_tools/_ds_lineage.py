#!/usr/bin/env python3
"""DS 检视：输出某案的逐帧类别时间线（lineage）与九格存储类别（audit）。
用法：/usr/bin/python3 /root/ds_lineage.py <case> [--full]
"""
import json
import re
import sys

EXP = '/root/gpufree-data/r7cand_export'
TSPAT = re.compile(r'\[(\d+\.\d+)\]')


def main():
    case = sys.argv[1]
    full = '--full' in sys.argv

    print('===== lineage (published class timeline) =====')
    seen = {}
    for l in open('%s/lineage_lines_%s.txt' % (EXP, case), errors='ignore'):
        m = TSPAT.search(l)
        ts = float(m.group(1)) if m else None
        body = l.split('candidate class lineage:', 1)[-1].strip()
        try:
            r = json.loads(body)
        except Exception:
            mm = re.search(r'\{.*\}', body)
            r = json.loads(mm.group(0)) if mm else {}
        sid = r.get('stable_id')
        row = (r.get('frame_id'), r.get('pretracker_class'), r.get('published_class'),
               r.get('class_source'), r.get('class_scan_id'), r.get('reclass_audit'))
        if full or seen.get(sid) != row:
            print('%s sid=%s frame=%s pre=%s pub=%s src=%s scan=%s audit=%s' % (
                ('%.1f' % ts) if ts else '?', sid, row[0], row[1], row[2], row[3], row[4], row[5]))
            seen[sid] = row

    print('===== nine scan audit (stored class) =====')
    for l in open('%s/audit_lines_%s.txt' % (EXP, case), errors='ignore'):
        m = TSPAT.search(l)
        ts = '%.1f' % float(m.group(1)) if m else '?'
        try:
            arr = json.loads(l.split('nine scan class audit:', 1)[1].strip())
        except Exception:
            continue
        sid = arr[0].get('scan_id') if arr else None
        for a in arr:
            print('%s scan=%s p%s px=%s scan_class=%s old=%s pend=%s stored=%s full=%s dec=%s' % (
                ts, a.get('scan_id', sid), a.get('proposal_index'), a.get('pixel'),
                a.get('scan_class'), a.get('old_class'), a.get('old_pending'),
                a.get('stored_class'), a.get('full'), a.get('decision')))


if __name__ == '__main__':
    main()
