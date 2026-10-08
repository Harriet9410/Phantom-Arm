# -*- coding: utf-8 -*-
"""R4-2 修复单测：F1a 锚定类别改判投票 + F1b/F1c 消息反馈。

场景复刻 b02_r3b r09：物体被首扫误锚成 Torch，后续每帧 VLM 原始标签
Magazine（被 established_track 覆写）。F1a 要求连续 6 帧冲突后改判一次。
反向：VLM 与锚定一致（持续错标）不触发改判；偶发闪变不清零后立即改判。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tcei_stack'))
from perception_tracking import CandidateTracker

fails=[]
def check(name,cond,detail=''):
    print(('PASS' if cond else 'FAIL'),name,detail)
    if not cond:fails.append(name)

_seq=[0]
def cand(px,depth=1.0,cls='Torch'):
    _seq[0]+=1;x,y=float(px[0]),float(px[1])
    return {'id':'c%d'%_seq[0],'pixel':[x,y],'bbox':[x-8,y-8,x+8,y+8],
            'depth':float(depth),'class':cls,'normalized_xy':[x/640.,y/480.],'area':256.}

# --- 场景 A：锚定 Torch，连续 6 帧 raw=Magazine → 第 6 帧改判 ---
t=CandidateTracker()
stable=None
for i in range(3):
    rows,_=t.update([cand([100.,100.],cls='Torch')],frame_id=i+1,observed_at=1.+i*.1)
    if rows:stable=rows[0]['stable_id']
cls_seq=None
for i in range(6):
    rows,_=t.update([cand([100.,100.],cls='Magazine')],frame_id=10+i,observed_at=6.+i*.1)
    cls_seq=rows[0]['class'] if rows else None
    if i<5:
        check('A%d 帧冲突期间仍锚定 Torch'%(i+1),cls_seq=='Torch',cls_seq)
check('A 第 6 帧改判为 Magazine',cls_seq=='Magazine',cls_seq)
prior=t.tracks[stable]['candidate']
check('A 锚定已更新',prior['class']=='Magazine',prior['class'])
check('A 改判审计存 track 级',t.tracks[stable].get('reclass_audit')=={'from':'Torch','to':'Magazine','frame':15},
      t.tracks[stable].get('reclass_audit'))
check('A 只改判一次（alt_reclass_done）',t.tracks[stable].get('alt_reclass_done') is True)
# 再来 6 帧冲突不再改判（已 done，锚定已是 Magazine，无冲突路径）
rows,_=t.update([cand([100.,100.],cls='Magazine')],frame_id=30,observed_at=20.)
check('A 后续稳定为 Magazine',rows[0]['class']=='Magazine')

# --- 场景 B：VLM 持续错标（raw 与锚定一致）→ 不改判 ---
t2=CandidateTracker()
sb=None
for i in range(3):
    rows,_=t2.update([cand([200.,200.],cls='Torch')],frame_id=i+1,observed_at=1.+i*.1)
    if rows:sb=rows[0]['stable_id']
for i in range(10):
    rows,_=t2.update([cand([200.,200.],cls='Torch')],frame_id=10+i,observed_at=6.+i*.1)
check('B VLM 一致错标不触发改判（保持 Torch）',t2.tracks[sb]['candidate']['class']=='Torch')

# --- 场景 C：冲突被打断（闪变）→ 计数清零，不改判 ---
t3=CandidateTracker()
sc=None
for i in range(3):
    rows,_=t3.update([cand([300.,300.],cls='Torch')],frame_id=i+1,observed_at=1.+i*.1)
    if rows:sc=rows[0]['stable_id']
for i in range(5):
    rows,_=t3.update([cand([300.,300.],cls='Grenade')],frame_id=10+i,observed_at=6.+i*.1)
rows,_=t3.update([cand([300.,300.],cls='Torch')],frame_id=20,observed_at=12.)  # 打断
for i in range(4):
    rows,_=t3.update([cand([300.,300.],cls='Grenade')],frame_id=21+i,observed_at=13.+i*.1)
check('C 闪变打断后不满阈值不改判',t3.tracks[sc]['candidate']['class']=='Torch',
      t3.tracks[sc]['candidate']['class'])

# --- 场景 D：F1b/F1c 消息反馈 ---
from core import parse_selection
import json as _json
hints=[{'id':'1','class':'Torch','pixel':[100.,100.],'bbox':[90,90,110,110],
        'normalized_xy':[.2,.2],'depth':2.0,'confidence':0.9,'grasp_ready':True,
        'identity_status':'confirmed','stable_id':'s-obj-0001'},
       {'id':'2','class':'Magazine','pixel':[500.,300.],'bbox':[490,290,510,310],
        'normalized_xy':[.8,.6],'depth':2.0,'confidence':0.9,'grasp_ready':True,
        'identity_status':'confirmed','stable_id':'s-obj-0002'},
       {'id':'3','class':'Grenade','pixel':[400.,200.],'bbox':[390,190,410,210],
        'normalized_xy':[.6,.4],'depth':2.0,'confidence':0.9,'grasp_ready':True,
        'identity_status':'confirmed','stable_id':'s-obj-0003'}]
scene={'frame_id':1,'stamp':10.0,'observed_at':10.0,'image_size':[1280,720],
       'spatial_context':{'reference':'camera_image','definition':'basket_quadrants_v1',
                          'transform_declared':True,'pixel_to_reference':[[1,0,0],[0,1,0],[0,0,1]],
                          'basket_roi':[0.,0.,1280.,720.],'uncertainty_px':5.0},
       'unknown_regions':[],'coverage_complete':True,'scene_complete':True}
answer=_json.dumps({'class':'Torch','side':'right','count':1,'ids':['1']})
try:
    parse_selection(answer,hints,'抓取手电筒，放到左侧传送带',scene)
    check('D1 side 拒单应触发',False)
except ValueError as e:
    msg=str(e)
    check('D1 side 消息带期望侧','left conveyor belt' in msg and 'answer said right' in msg,msg)
from planning_prompt import feedback_principle_zh
fb=feedback_principle_zh(msg)
check('D2 反馈点名具体侧','left侧传送带' in fb and '必须等于' in fb,fb[:150])

# F1c：remaining 多选漏件的数量对比消息
tc={'dependencies_satisfied':True,'allow_reorder':False,
    'remaining_ids':['1','2','3'],'reserved_ids':[],
    'remaining_stable_ids':['s-obj-0001','s-obj-0002','s-obj-0003'],
    'remaining_complete':True,
    'released_pending_stable_ids':[],'stable_ids':['s-obj-0001','s-obj-0002','s-obj-0003'],
    'reserved_stable_ids':[],'released_pending_count':0}
ans3=_json.dumps({'class':'Remaining','side':'left','count':2,'ids':['1','2']})
try:
    parse_selection(ans3,hints,'抓取剩余的物品，放到左侧传送带',scene,tc)
    check('D3 多选漏件应拒单',False)
except ValueError as e:
    check('D4 消息带数量对比','ledger has 3 remaining objects; answer selected 2' in str(e),str(e)[:140])


# --- 场景 E：F1a 配套——改判后 class_source=recheck_majority 传递到候选 ---
t5=CandidateTracker()
se=None
for i in range(3):
    rows,_=t5.update([cand([400.,400.],cls='Torch')],frame_id=i+1,observed_at=1.+i*.1)
    if rows:se=rows[0]['stable_id']
for i in range(6):
    rows,_=t5.update([cand([400.,400.],cls='Smokegrenade')],frame_id=10+i,observed_at=6.+i*.1)
check('E 改判帧起 class_source=recheck_majority',rows[0].get('class_source')=='recheck_majority',
      rows[0].get('class_source'))

print('RESULT:','ALL PASS' if not fails else 'FAILED: %s'%fails)
sys.exit(1 if fails else 0)
