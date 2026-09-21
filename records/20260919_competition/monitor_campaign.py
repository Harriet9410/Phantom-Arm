"""Compact read-only batch status; optionally point the review UI at its campaign."""
import argparse,collections,hashlib,json,shutil,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--update-panel',action='store_true');args=parser.parse_args()
    package=args.package.resolve();base=Path(json.loads((package/'state/configuration.json').read_text())['runs'])
    path=base/args.batch/'campaign.json';raw=path.read_bytes();data=json.loads(raw)
    result={'reported_at':time.time(),'batch':data['batch'],'status':data['status'],
        'planned_rounds':data['planned_rounds'],'completed_rounds':len(data['results']),
        'active_case':data.get('active_case'),'runtime':data.get('runtime_version'),
        'runtime_manifest_sha256':data.get('runtime_manifest_sha256'),
        'campaign_sha256_at_read':hashlib.sha256(raw).hexdigest(),'free_bytes':shutil.disk_usage(base).free,
        'instructions':data.get('instructions'),'results':[]}
    for row in data['results']:
        counts=row.get('event_counts',{})
        small={k:row.get(k) for k in ('case_id','seed','status','verified_objects','independent_placements',
            'physical_completed','elapsed_seconds','reason','commands','code_changed_during_episode')}
        small['counts']={k:counts.get(k,0) for k in ('attempt','descend','grasp_verified','empty_grasp',
            'holding_feedback_lost','drop_detected','drop_recovery_completed','model_answer','repair_started',
            'grasp_orientation_rejected','grasp_tilt_fallback_started','placement_pending_verification')}
        small['evidence_complete']=row.get('supervisor',{}).get('evidence_complete')
        result['results'].append(small)
    active=data.get('active_case')
    if active:
        run=base/(args.batch+'_'+active+'_round');stack=base/(args.batch+'_'+active+'_stack')
        events=[];path_events=run/'episode/events.jsonl'
        if path_events.exists():
            for line in path_events.read_text().splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:continue
        result['active_event_counts']=dict(collections.Counter(e.get('status') for e in events))
        result['active_recent']=[{k:e.get(k) for k in ('status','reason','time','tilt_deg')}
            for e in events[-5:]]
        for name in ('scene_setup_error.json','scene_initialized.json','read_only_ready.json'):
            if (stack/name).exists():
                value=json.loads((stack/name).read_text())
                result[name]={k:value.get(k) for k in ('valid','case_id','seed','settle_frames','ready_for_read_only','error','errors')}
    if args.update_panel:
        config=package/'state/panel_config.json'
        if config.exists():
            value=json.loads(config.read_text());value['campaign_file']=str(path)
            temp=config.with_name(config.name+'.monitor.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(config)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','completed_rounds','planned_rounds','active_case','free_bytes')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
