#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 semantic probe: send official instructions to /tcei/request.
Read-only: execution switches are false; nine_node answers dry_run_validated."""
import json
import sys
import time
import uuid
import rospy
from std_msgs.msg import String

INSTRUCTIONS = [
    '抓取左上方的烟雾弹，放到左侧传送带',
    '抓取右下方的弹夹，放到右侧传送带',
    '抓取最左方的军用手电筒，放到左侧传送带',
    '抓取手雷，放到右侧传送带',
    '抓取剩余的物品，放到左侧传送带',
]

def main():
    gap = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rospy.init_node('semantic_probe', anonymous=True, disable_signals=True)
    pub = rospy.Publisher('/tcei/request', String, queue_size=1)
    time.sleep(2.0)
    ids = []
    for r in range(rounds):
        for i, instr in enumerate(INSTRUCTIONS, 1):
            rid = 'probe-r%d-t%d-%s' % (r + 1, i, uuid.uuid4().hex[:6])
            pub.publish(String(json.dumps({'instruction': instr, 'request_id': rid},
                                          ensure_ascii=False)))
            ids.append(rid)
            print('SENT %s | %s' % (rid, instr), flush=True)
            time.sleep(gap)
    print('ALL_SENT ' + json.dumps(ids, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
