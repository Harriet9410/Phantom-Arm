"""Read-only analysis of archived task1 contact/lift; never controls the robot."""
from pathlib import Path
import collections
import hashlib
import json
import shutil

ROOT = Path('/root/tcei_competition_20260919')
RUN = Path('/root/gpufree-data/tcei_competition_20260919/official13_round01')
events = [json.loads(x) for x in (RUN/'episode/events.jsonl').read_text().splitlines()]
contact = next(e for e in events if e.get('status') == 'grasp_contact_confirmed')
lost = next(e for e in events if e.get('status') == 'holding_feedback_lost')
start = contact['time']-1.
end = lost['time']+1.
rows = collections.defaultdict(list)
commands=[]
for line in (RUN/'scalars/events.jsonl').open():
    row=json.loads(line)
    if row['topic']=='/Jaka/set_gripper_value':commands.append(row)
    if start<=row['received_wall']<=end:rows[row['topic']].append(row)

samples=rows['/tcei/gripper_contact']
bins={}
for row in samples:
    value=row['value'];pair=value['samples']
    offset=round(row['received_wall']-contact['time'],3)
    key=round(int(offset*10)/10,1)
    bins.setdefault(key,[]).append(row)
compact=[]
for key,group in sorted(bins.items()):
    forces=[[r['value']['samples'][i]['value'] for r in group] for i in range(2)]
    row=group[-1];v=row['value'];stamp=v.get('simulation_time')
    d=min(rows['/tcei/joint_diagnostics'],key=lambda r:abs(r['value']['simulation_time']-stamp))['value']
    compact.append({'wall_offset':key,'samples':len(group),'sim':stamp,
                    'force_min':[min(x) for x in forces],'force_max':[max(x) for x in forces],
                    'all_valid':all(s['valid'] for r in group for s in r['value']['samples']),
                    'fingers':d['all_joint_positions'][-2:],'tcp':d['tcp_actual']})
capture=[];previous=None
for row in rows['/Jaka/gripper_is_captured']:
    if row['value']!=previous:
        capture.append(row);previous=row['value']
index=[json.loads(x) for x in (RUN/'rgbd/index.jsonl').read_text().splitlines()]
targets=[('before_contact',contact['time']-.4),('contact',contact['time']),
         ('lift_0_25',contact['time']+.25),('lift_0_5',contact['time']+.5),
         ('lift_0_75',contact['time']+.75),('lift_1_0',contact['time']+1.),
         ('lift_1_5',contact['time']+1.5),('loss_report',lost['time']),
         ('after_stop',lost['time']+1.)]
review=RUN/'contact_review';review.mkdir(exist_ok=False)
frames=[]
for name,when in targets:
    m=min(index,key=lambda r:abs(r['captured_at']-when))
    source=RUN/'rgbd'/(m['stem']+'.jpg');destination=review/(name+'.jpg')
    shutil.copyfile(source,destination)
    frames.append({'name':name,'source':str(source),'display':str(destination),
                   'metadata':m,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
selected=[e for e in events if e.get('status') in ('attempt','grasp_orientation_selected','descend',
          'grasp_contact_confirmed','trial_lift_started','trial_lift','holding_feedback_lost')]
lift=[e for e in events if e.get('status')=='trial_lift_evidence']
report={'scope':'Recorded evidence only; force loss is not alone proof of physical drop.',
        'contact':contact,'lost':lost,'bins':compact,'capture_transitions':capture,
        'gripper_commands':commands,'frames':frames,'motion_events':selected,'lift_events':lift}
(ROOT/'official13_contact_analysis.txt').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(review/'evidence_manifest.json').write_text(json.dumps({'frames':frames,'contact_at':contact['time'],
    'feedback_lost_at':lost['time'],'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print('CONTACT_ANALYSIS_DONE',len(samples),len(frames),flush=True)
