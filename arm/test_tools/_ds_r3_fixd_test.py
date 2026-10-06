# -*- coding: utf-8 -*-
"""Fix D（R3-1 幻影抑制）单元验证：复刻满十案取证的幻影时序。

时序：物体 A 在槽位被确认 → placement_verified 退休 → 抓取期间槽位附近出现
手臂深度团块（比桌面深，走 revealed_after_delivery 路径）建立独立重复轨迹
→ 团块消失 → 修复前该轨迹永报 unobserved_unverified_object 阻断第五项；
修复后按"位置重合退休身份建轨位+互斥 ID"抑制。
反向保护：远离退休槽位的真实失踪物体照常报警。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tcei_stack'))
from perception_tracking import CandidateTracker

_seq=[0]
def cand(px,depth=1.0,cls='Grenade'):
    _seq[0]+=1;x,y=float(px[0]),float(px[1])
    return {'id':'c%d'%_seq[0],'pixel':[x,y],'bbox':[x-8,y-8,x+8,y+8],
            'depth':float(depth),'class':cls,'normalized_xy':[x/640.,y/480.],'area':256.}

def regions(unk):
    return [r for r in unk if r.get('reason')=='unobserved_unverified_object']

fails=[]
def check(name,cond,detail=''):
    print(('PASS' if cond else 'FAIL'),name,detail)
    if not cond:fails.append(name)

# --- 场景1：交付后槽位幻影 ---
t=CandidateTracker()
stable_a=None
for i in range(3):
    rows,unk=t.update([cand([100.,100.])],frame_id=i+1,observed_at=1.+i*.1)
    if rows: stable_a=rows[0].get('stable_id')
check('s1 A 确认',stable_a is not None and stable_a.endswith('-obj-0001'),stable_a)
check('s1 交付前无未知区域',regions(unk)==[],unk)
ack=t.on_event({'status':'placement_verified','stable_id':stable_a,'time':5.})
check('s1 退休 ack',ack is None)
# 抓取期间：槽位附近手臂团块（depth 更大=更近相机下方表面，走 revealed 路径）
phantom=None
for i in range(6):
    rows,unk=t.update([cand([105.,102.],depth=1.25)],frame_id=10+i,observed_at=6.+i*.1)
    if rows: phantom=rows[0].get('stable_id')
check('s1 幻影建独立轨迹',phantom is not None and phantom!=stable_a,(stable_a,phantom))
# 幻影消失 → 修复前会永报 unobserved；修复后应被抑制
rows,unk=t.update([],frame_id=30,observed_at=15.)
check('s1 幻影不报失踪',regions(unk)==[],regions(unk))
check('s1 无其它未知区域',unk==[],unk)
# 轨迹本体保留供审计
check('s1 幻影轨迹保留',phantom in t.tracks)

# --- 场景2：远离退休槽位的真实失踪物体照常报警 ---
t2=CandidateTracker()
stable_b=None
for i in range(3):
    rows,unk=t2.update([cand([200.,200.],cls='Torch')],frame_id=i+1,observed_at=1.+i*.1)
    if rows: stable_b=rows[0].get('stable_id')
rows,unk=t2.update([],frame_id=10,observed_at=5.)
rs=regions(unk)
check('s2 真实失踪仍报警',len(rs)==1 and rs[0].get('stable_id')==stable_b,rs)

# --- 场景3：无退休身份时不抑制任何区域 ---
t3=CandidateTracker()
stable_c=None
for i in range(3):
    rows,unk=t3.update([cand([300.,300.],cls='Magazine')],frame_id=i+1,observed_at=1.+i*.1)
    if rows: stable_c=rows[0].get('stable_id')
# 活跃轨迹旁 6px 的团块会被正常关联（不建新轨）；45px 外同类团块被远距重现
# 歧义守卫拦下（也不建新轨）。独立轨迹只能由异类团块形成——用 Torch 团块
# 在 (345,300) 建独立轨迹，消失后必须照常报警（证明不是全局抑制）。
p3=None
for i in range(6):
    rows,unk=t3.update([cand([345.,300.],depth=1.25,cls='Torch')],frame_id=10+i,observed_at=6.+i*.1)
    if rows: p3=rows[0].get('stable_id')
rows,unk=t3.update([],frame_id=30,observed_at=15.)
rs=regions(unk)
ids={r.get('stable_id') for r in rs}
check('s3 未退休槽位幻影照常报警',p3 in ids,(sorted(ids),p3))

# --- 场景4：双交付（两件同批交付后各自槽位幻影都抑制）---
t4=CandidateTracker()
sa=sb=None
for i in range(3):
    rows,unk=t4.update([cand([100.,100.]),cand([160.,100.])],frame_id=i+1,observed_at=1.+i*.1)
    for r in rows:
        if r['pixel'][0]<130: sa=r['stable_id']
        else: sb=r['stable_id']
t4.on_event({'status':'placement_verified','stable_id':sa,'time':5.})
t4.on_event({'status':'placement_verified','stable_id':sb,'time':5.})
for i in range(6):
    t4.update([cand([103.,101.],depth=1.25),cand([158.,99.],depth=1.25)],frame_id=10+i,observed_at=6.+i*.1)
rows,unk=t4.update([],frame_id=30,observed_at=15.)
check('s4 双交付槽位幻影均抑制',regions(unk)==[],regions(unk))

# --- 场景5：Fix D2 在位确认目标邻位的幻影（先验激活流程）---
SLOTS=[(100.,100.),(160.,100.),(220.,100.),(671.,350.),(280.,100.)]
t5=CandidateTracker(layout_prior=[{'pixel':list(s),'class':'Grenade'} for s in SLOTS])
sa5=sb5=None
# 首扫 4+ 命中先验 → prior_active 激活
for i in range(3):
    rows,unk=t5.update([cand(SLOTS[0]),cand(SLOTS[1]),cand(SLOTS[2]),cand(SLOTS[3],cls='Smokegrenade'),cand(SLOTS[4])],
                       frame_id=i+1,observed_at=1.+i*.1)
    for r in rows:
        if abs(r['pixel'][0]-671)<5: sa5=r['stable_id']
# 活体邻位幻影：obj-0003(671,350) 右侧 22px 的竖条伪影，先建立轨迹
p5=None
for i in range(6):
    rows,unk=t5.update([cand(SLOTS[0]),cand(SLOTS[1]),cand(SLOTS[2]),cand(SLOTS[3],cls='Smokegrenade'),cand(SLOTS[4]),
                        cand([693.,345.],depth=1.25,cls='Smokegrenade')],frame_id=10+i,observed_at=6.+i*.1)
    for r in rows:
        if abs(r['pixel'][0]-693)<3: p5=r['stable_id']
check('s5 邻位幻影成独立轨迹',p5 is not None and p5!=sa5,(p5,sa5))
# 幻影消失、真物仍在且 confirmed → 幻影不报失踪
rows,unk=t5.update([cand(SLOTS[0]),cand(SLOTS[1]),cand(SLOTS[2]),cand(SLOTS[3],cls='Smokegrenade'),cand(SLOTS[4])],
                   frame_id=30,observed_at=15.)
rs=regions(unk)
ids5={r.get('stable_id') for r in rs}
check('s5 邻位幻影被抑制',p5 not in ids5,sorted(ids5))

# --- 场景6：Fix D2 反例——远离所有确认候选的幻影照常报警 ---
t6=CandidateTracker(layout_prior=[{'pixel':list(s),'class':'Grenade'} for s in SLOTS])
for i in range(3):
    t6.update([cand(SLOTS[0]),cand(SLOTS[1]),cand(SLOTS[2]),cand(SLOTS[3],cls='Smokegrenade'),cand(SLOTS[4])],
              frame_id=i+1,observed_at=1.+i*.1)
p6=None
for i in range(6):
    rows,unk=t6.update([cand(SLOTS[0]),cand(SLOTS[1]),cand(SLOTS[2]),cand(SLOTS[3],cls='Smokegrenade'),cand(SLOTS[4]),
                        cand([500.,300.],depth=1.25,cls='Torch')],frame_id=10+i,observed_at=6.+i*.1)
    for r in rows:
        if abs(r['pixel'][0]-500)<3: p6=r['stable_id']
rows,unk=t6.update([cand(SLOTS[0]),cand(SLOTS[1]),cand(SLOTS[2]),cand(SLOTS[3],cls='Smokegrenade'),cand(SLOTS[4])],
                   frame_id=30,observed_at=15.)
rs=regions(unk)
ids6={r.get('stable_id') for r in rs}
check('s6 远离确认候选的幻影照常报警',p6 in ids6,(sorted(ids6),p6))

# --- 场景7：Fix D3 无先验时邻位幻影仍被抑制（B03 打乱布局的真实条件）---
t7=CandidateTracker()  # 无 layout_prior → prior_active 恒为 False
for i in range(3):
    rows,unk=t7.update([cand([671.,349.],cls='Smokegrenade')],frame_id=i+1,observed_at=1.+i*.1)
p7=None
for i in range(6):
    rows,unk=t7.update([cand([671.,349.],cls='Smokegrenade'),
                        cand([650.,349.],depth=1.25,cls='Smokegrenade')],frame_id=10+i,observed_at=6.+i*.1)
    for r in rows:
        if abs(r['pixel'][0]-650)<3: p7=r['stable_id']
rows,unk=t7.update([cand([671.,349.],cls='Smokegrenade')],frame_id=30,observed_at=15.)
rs=regions(unk)
ids7={r.get('stable_id') for r in rs}
check('s7 无先验时 21px 邻位幻影被抑制',p7 not in ids7,(sorted(ids7),p7))

print('RESULT:','ALL PASS' if not fails else 'FAILED: %s'%fails)
sys.exit(1 if fails else 0)
