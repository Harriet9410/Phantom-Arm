"""Read-only replay of archived RGB-D/events, never publishes robot commands."""
from pathlib import Path
import collections,copy,json,sys,time
import numpy as np
ROOT=Path('/root/tcei_competition_20260919')
RUN=Path('/root/gpufree-data/tcei_competition_20260919/official16_round01')
sys.path.insert(0,str(ROOT/'releases/dev_v7_t1_16/tcei_stack'))
from conveyor_observer17 import DepthConveyorTracker
from core import verify_transport
events=sorted([json.loads(x) for x in (RUN/'episode/events.jsonl').read_text().splitlines()],key=lambda x:x.get('time',0.))
index=[json.loads(x) for x in (RUN/'rgbd/index.jsonl').read_text().splitlines()]
tracker=DepthConveyorTracker();trace=[];observations=[];next_event=0;latest_event=None
original=tracker._preexisting_exclusions
def audit(components,stamp,height):
    state=tracker.active or tracker.preexisting_state;before=copy.deepcopy(state)
    result=original(components,stamp,height)
    if state and (state.get('provenance_uncertain') and not (before or {}).get('provenance_uncertain')):
        trace.append({'kind':'provenance_became_uncertain','stamp':stamp,'event_context':latest_event,
                      'before':before,'after':copy.deepcopy(state),'components':copy.deepcopy(components),
                      'excluded':copy.deepcopy(result)})
    return result
tracker._preexisting_exclusions=audit
previous=None;counts=collections.Counter()
for number,frame in enumerate(index):
    while next_event<len(events) and events[next_event].get('time',0.)<=frame['captured_at']:
        event=events[next_event];next_event+=1
        tracker.on_event(event)
        if event.get('status') in ('attempt','grasp_verified','release_started','released','holding_feedback_lost'):
            latest_event={k:event.get(k) for k in ('status','request_id','time','category','release_id')}
            if event.get('candidate'):latest_event['candidate']={k:event['candidate'].get(k) for k in ('class','stable_id','pixel')}
    with np.load(RUN/'rgbd'/(frame['stem']+'.npz')) as archive:depth=archive['depth']
    rows=tracker.process(depth,frame['camera']['K'],frame['captured_at'],frame['stamp'])
    state=tracker.diagnostics.get('state');counts[state]+=1
    if state!=previous:
        trace.append({'kind':'tracker_state','frame':frame['stem'],'wall':frame['captured_at'],
                      'stamp':frame['stamp'],'diagnostic':copy.deepcopy(tracker.diagnostics),'event_context':latest_event})
        previous=state
    if rows:observations.append({'observed_at':frame['captured_at'],'stamp':frame['stamp'],
                                'frame':frame['stem'],'image_size':[frame['camera']['width'],frame['camera']['height']],'observations':rows})
    if number%100==0:
        (ROOT/'replay_transport17_progress.json').write_text(json.dumps({'frames':number,'total':len(index),'time':time.time()}))
releases=[e for e in events if e.get('status')=='released'];checks=[]
for e in releases:
    start=next(x for x in events if x.get('status')=='release_started' and x.get('release_id')==e['release_id'])
    proof=verify_transport(observations,e['category'],e['side'],e['released_at'],start['drop_y'],e['request_id'],start['drop_x'],
        stable_id=e['stable_id'],release_id=e['release_id'])
    checks.append({'class':e['category'],'side':e['side'],'release_id':e['release_id'],'verified':proof is not None,
                   'observations':sum(any(o['release_id']==e['release_id'] for o in row['observations']) for row in observations),
                   'proof':proof})
out={'finished_at':time.time(),'frames':len(index),'counts':dict(counts),'trace':trace,'checks':checks,
     'scope':'chronological replay of archived pairs/events; original callback scheduling may differ'}
(ROOT/'replay_transport17_result.txt').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(ROOT/'replay_transport17_observations.json').write_text(json.dumps(observations,ensure_ascii=False))
print('REPLAY_TRANSPORT17_COMPLETE',len(index),flush=True)
