"""Read archived closure and stop feedback. No ROS publishers or simulator truth."""
from pathlib import Path
import collections
import hashlib
import json
import math
import shutil

ROOT=Path('/root/tcei_competition_20260919')
RUN=Path('/root/gpufree-data/tcei_competition_20260919/official17_round01')
events=[json.loads(x) for x in (RUN/'episode/events.jsonl').read_text().splitlines()]
preload=next(e for e in events if e.get('status')=='grasp_preload_started')
failure=next(e for e in events if e.get('status')=='empty_grasp')
stop=next(e for e in events if e.get('status')=='controlled_stop_requested')
rows=collections.defaultdict(list)
for line in (RUN/'scalars/events.jsonl').open():
    row=json.loads(line)
    if preload['time']-2.<=row['received_wall']<=stop['time']+6.5:
        rows[row['topic']].append(row)

def nearest(topic,when):
    return min(rows[topic],key=lambda r:abs(r['received_wall']-when))

def contact_at(when):
    row=nearest('/tcei/gripper_contact',when)
    value=row['value']
    return {'dt':row['received_wall']-when,
            'values':[s['value'] for s in value['samples']],
            'valid':[s['valid'] for s in value['samples']]}

commands=[]
for r in rows['/Jaka/set_gripper_value']:
    when=r['received_wall'];before=[j for j in rows['/Jaka/get_jointstate'] if j['received_wall']<=when][-1]
    after=nearest('/Jaka/get_jointstate',when+.04)
    commands.append({'time':when,'offset_preload':when-preload['time'],'command':r['value'],
        'before_fingers':before['value']['position'][-2:],
        'after_fingers':after['value']['position'][-2:],
        'contact_at_command':contact_at(when),'contact_after_40ms':contact_at(when+.04)})

bins=[]
for tenth in range(-10,12):
    start=preload['time']+tenth*.1;end=start+.1
    group=[r for r in rows['/tcei/gripper_contact'] if start<=r['received_wall']<end]
    joints=[r for r in rows['/Jaka/get_jointstate'] if start<=r['received_wall']<end]
    if not group or not joints:continue
    force=[[r['value']['samples'][i]['value'] for r in group] for i in (0,1)]
    bins.append({'offset':tenth*.1,'n':len(group),'force_min':[min(v) for v in force],
        'force_max':[max(v) for v in force],
        'fingers_min':[min(j['value']['position'][-2+i] for j in joints) for i in (0,1)],
        'fingers_max':[max(j['value']['position'][-2+i] for j in joints) for i in (0,1)]})

transitions=[];previous=None
for r in rows['/Jaka/gripper_is_captured']:
    if r['value']!=previous:
        transitions.append({'offset':r['received_wall']-preload['time'],'value':r['value']})
        previous=r['value']

stops=[r for r in rows['/tcei/stop_ack'] if r['value'].get('id')==stop['stop_id']]
samples=[r for r in rows['/Jaka/get_jointstate'] if r['received_wall']>=stop['time']]
origin=samples[0]['value']['position'][:6]
speeds=[max(abs(v) for v in r['value']['velocity'][:6]) for r in samples]
drifts=[max(abs(a-b) for a,b in zip(origin,r['value']['position'][:6])) for r in samples]
low_since=None;longest=0.
for r,v in zip(samples,speeds):
    t=r['last_received_simulation_time']
    if v>.03:low_since=None
    elif low_since is None:low_since=t
    else:longest=max(longest,t-low_since)
high=[{'offset':r['received_wall']-stop['time'],'simulation_time':r['last_received_simulation_time'],
       'positions':r['value']['position'][:6],'velocities':r['value']['velocity'][:6]}
      for r,v in zip(samples,speeds) if v>.03]
stop_summary={'samples':len(samples),'speed_min':min(speeds),'speed_max':max(speeds),
    'count_over_velocity_limit':len(high),'joint_excursion_max':max(drifts),
    'longest_contiguous_low_velocity_sim_seconds':longest,
    'per_joint_velocity_max':[max(abs(r['value']['velocity'][i]) for r in samples) for i in range(6)],
    'high_speed_examples':high[:12],
    'ack_states':dict(collections.Counter(r['value'].get('state') for r in stops)),
    'ack_examples':[r['value'] for r in stops[:3]+stops[-3:]]}

review=RUN/'closure_stop_review17';review.mkdir(exist_ok=False)
index=[json.loads(x) for x in (RUN/'rgbd/index.jsonl').read_text().splitlines()]
frames=[]
for name,when in [('before_preload',preload['time']-.2),('during_preload',preload['time']+.15),
                  ('closure_rejected',failure['time']),('stop_latched',stop['time']+1.)]:
    pair=min(index,key=lambda p:abs(p['captured_at']-when));source=RUN/'rgbd'/(pair['stem']+'.jpg')
    destination=review/(name+'.jpg');shutil.copyfile(source,destination)
    frames.append({'name':name,'source':str(source),'display':str(destination),'time':pair['captured_at'],
                   'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})

report={'scope':'Archived native feedback only. No physical-drop inference from effort loss alone.',
    'preload':preload,'closure_failure':failure,'stop_request':stop,'commands':commands,'bins':bins,
    'capture_transitions':transitions,'stop_feedback':stop_summary,'frames':frames,
    'key_events':[e for e in events if e.get('status') in ('arrived','empty_grasp','grasp_attempt_outcome',
        'controlled_stop_requested','controlled_stop_fault','grasp_orientation_selected')]}
(ROOT/'official17_closure_stop_analysis.txt').open('x').write(json.dumps(report,ensure_ascii=False,indent=2))
stream={topic:values for topic,values in rows.items() if topic in ('/Jaka/get_jointstate','/Jaka/get_gripper_efforts',
    '/Jaka/gripper_is_captured','/clock','/tcei/stop_ack','/Jaka/set_gripper_value')}
(review/'native_feedback_window.json').open('x').write(json.dumps(stream))
(review/'evidence_manifest.json').open('x').write(json.dumps({'frames':frames,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print('OFFICIAL17_FAILURE_ANALYZED',len(commands),len(samples),flush=True)
