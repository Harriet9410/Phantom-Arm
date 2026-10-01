"""Run a preregistered batch through the deployment entry, preserving every attempt.

Scene truth is read only AFTER each episode for independent grading; it is
never used to choose a grasp, modify model answers or publish robot commands.
"""
import argparse,collections,hashlib,json,math,os,signal,statistics,subprocess,time
from pathlib import Path


def save(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []


def audit(run,stack,case,instructions):
    summary=json.loads((run/'episode/summary.json').read_text())
    events=rows(run/'episode/events.jsonl');truth=rows(stack/'ground_truth.jsonl')
    counts=collections.Counter(e.get('status') for e in events)
    identities={};identity_conflicts=[]
    for event in events:
        if event.get('status')!='attempt' or not truth:continue
        candidate=event['candidate'];sample=min(truth,key=lambda s:abs(s['observed_at']-event['time']))
        distances=sorted((math.dist(candidate['world_position'][:2],obj['reference_world'][:2]),obj['object_id'],obj['class']) for obj in sample['objects'])
        if distances and distances[0][0]<.06 and abs(sample['observed_at']-event['time'])<.5 and (len(distances)==1 or distances[1][0]-distances[0][0]>.015):
            distance,ident,category=distances[0];stable=candidate['stable_id']
            if stable in identities and identities[stable]['object_id']!=ident:identity_conflicts.append(stable)
            identities[stable]={'object_id':ident,'actual_class':category,'model_class':candidate['class'],
                                'match_distance_m':distance,'request_id':event['request_id'],'task_id':event['task_id']}
    release_checks=[];used=set();task_order={x['task_id']:i for i,x in enumerate(summary['tasks'])}
    for event in events:
        if event.get('status')!='released':continue
        match=identities.get(event.get('stable_id'));check={'release_id':event.get('release_id'),
            'task_id':event.get('task_id'),'side':event.get('side'),'category':event.get('category'),
            'identity_match':match,'confirmed':False,'samples':[]}
        index=task_order.get(event.get('task_id'))
        instruction=instructions[index] if index is not None and index<len(instructions) else ''
        expected_side='left' if '左侧传送带' in instruction else 'right' if '右侧传送带' in instruction else None
        aliases={'Smokegrenade':('烟雾弹',),'Magazine':('弹匣','弹夹'),'Torch':('手电筒',),
                 'Grenade':('手榴弹','手雷'),'CompressedFood':('压缩干粮','压缩食品')}
        expected_class=next((name for name,words in aliases.items() if any(w in instruction for w in words)),None)
        semantic_ok=(event.get('side')==expected_side and
            (event.get('category')==expected_class or (expected_class is None and '剩余' in instruction)))
        check['instruction_side_and_class_match']=semantic_ok
        if match and semantic_ok and match['actual_class']==event['category'] and match['object_id'] not in used:
            sign=1 if event['side']=='left' else -1
            for sample in truth:
                delta=sample['observed_at']-event['time']
                if not 1.0<=delta<=5.:continue
                obj=next((o for o in sample['objects'] if o['object_id']==match['object_id']),None)
                if not obj:continue
                pos=obj['reference_world'];matrix=obj.get('usd_world_matrix')
                consistency=math.dist(obj['position'],matrix[3][:3]) if matrix else math.inf
                on_belt=(.43<sign*pos[0]<.95 and -.65<pos[1]<1.5 and 2.33<pos[2]<2.62 and consistency<.02)
                if on_belt:check['samples'].append({'time':sample['observed_at'],'delta':delta,
                    'position':pos,'physics_usd_delta_m':consistency})
            check['confirmed']=len(check['samples'])>=3 and check['samples'][-1]['time']-check['samples'][0]['time']>=.35
            if case.get('expected_object_ids') and index is not None:
                check['expected_object_id']=case['expected_object_ids'][index]
                check['confirmed']=check['confirmed'] and match['object_id']==check['expected_object_id'] and event['side']==case['expected_sides'][index]
            if check['confirmed']:used.add(match['object_id'])
        release_checks.append(check)
    result={'status':summary['status'],'verified_objects':summary.get('verified_objects',0),
        'elapsed_seconds':summary.get('elapsed_seconds'),'reason':summary.get('reason'),
        'safe_stop':summary.get('safe_stop'),'event_counts':dict(counts),'identity_conflicts':identity_conflicts,
        'independent_placements':sum(x['confirmed'] for x in release_checks),'release_checks':release_checks,
        'physical_completed':len(used)==len(case['objects'])==5 and not identity_conflicts and summary.get('elapsed_seconds',math.inf)<=600
            and [x['instruction'] for x in summary['tasks']]==instructions,
        'code_changed_during_episode':summary.get('code_changed_during_episode',[]),
        'task_results':[x['result'].get('status') for x in summary['tasks']],
        'counts_scope':'machine verification and independent physical outcomes are distinct',
        'source_summary_sha256':hashlib.sha256((run/'episode/summary.json').read_bytes()).hexdigest()}
    save(run/'independent_audit.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True,type=Path);p.add_argument('--register',required=True,type=Path)
    p.add_argument('--case-root',required=True,type=Path)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--case-ids',nargs='+');selection.add_argument('--all-registered',action='store_true')
    p.add_argument('--batch',required=True);p.add_argument('--instructions',required=True,type=Path)
    p.add_argument('--task-source',required=True);args=p.parse_args()
    package=args.package.resolve();register=json.loads(args.register.read_text(encoding='utf-8-sig'))
    manifest_path=package/'tcei_stack/BUILD_MANIFEST.json'
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if register.get('runtime_manifest_sha256') not in (None,manifest_sha):raise ValueError('registered runtime differs')
    cfg=json.loads((package/'state/configuration.json').read_text());data=Path(cfg['runs']);out=data/args.batch
    if out.exists():raise RuntimeError('batch already exists; do not replace first-attempt results')
    selected=[];case_ids=args.case_ids if args.case_ids is not None else [x['case_id'] for x in register['cases']]
    if len(case_ids)!=len(set(case_ids)):raise ValueError('duplicate case identifiers')
    for ident in case_ids:
        found=[x for x in register['cases'] if x['case_id']==ident]
        if len(found)!=1:raise ValueError('case not uniquely registered: '+ident)
        row=found[0];path=args.case_root/row['case_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('registered case digest differs')
        selected.append((row,path,json.loads(path.read_text(encoding='utf-8-sig'))))
    instructions=json.loads(args.instructions.read_text(encoding='utf-8-sig'))
    out.mkdir();state={'batch':args.batch,'planned_rounds':len(selected),'status':'running','started_at':time.time(),
        'register_sha256':hashlib.sha256(args.register.read_bytes()).hexdigest(),'instructions':instructions,
        'task_source':args.task_source,'package':str(package),'runtime_manifest_sha256':manifest_sha,
        'runtime_version':json.loads(manifest_path.read_text())['version'],'results':[]};save(out/'campaign.json',state)
    for row,path,case in selected:
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=manifest_sha:
            state['status']='runtime_changed_between_rounds';save(out/'campaign.json',state);return 1
        name=args.batch+'_'+row['case_id'];stack=data/(name+'_stack');run=data/(name+'_round')
        result={'case_id':row['case_id'],'seed':row['seed'],'case_sha256':row['sha256'],
                'stack':str(stack),'run':str(run),'started_at':time.time(),'commands':[]}
        state['active_case']=row['case_id'];save(out/'campaign.json',state)
        commands=[['bash','robot.sh','start',stack.name,'--case',str(path.resolve()),'--case-register',str(args.register.resolve())],
                  ['bash','robot.sh','run',run.name,'--instructions',str(args.instructions.resolve()),'--task-source',args.task_source],
                  ['bash','robot.sh','stop','--stack',stack.name]]
        for command in commands:
            phase=command[2]
            try:
                with (out/(row['case_id']+'_'+phase+'.log')).open('xb') as log:
                    child=subprocess.Popen(command,cwd=package,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=child.wait(timeout=750)
            except subprocess.TimeoutExpired:
                # Do not let subprocess.run kill a motion supervisor before
                # controller cancellation and evidence closure have a chance.
                result.update(status='supervision_timeout',failed_phase=phase,supervisor_pid=child.pid)
                if phase=='run':
                    with (out/(row['case_id']+'_watchdog_cancel.log')).open('xb') as log:
                        cancellation=subprocess.run(['bash','robot.sh','cancel','--stack',stack.name,
                            '--reason','batch watchdog'],cwd=package,stdout=log,stderr=subprocess.STDOUT,timeout=30)
                    result['watchdog_cancel_code']=cancellation.returncode
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:result['supervisor_still_running']=True
                code=-1
            result['commands'].append({'phase':phase,'returncode':code,'finished_at':time.time()})
            if phase=='run' and (run/'supervisor_finished.json').exists():
                result['supervisor']=json.loads((run/'supervisor_finished.json').read_text())
                if (run/'episode/summary.json').exists():result.update(audit(run,stack,case,instructions))
            if code!=0 and phase!='run':
                result.setdefault('status','initialization_failed' if phase=='start' else 'shutdown_failed');break
        result['finished_at']=time.time();state['results'].append(result)
        state['automatic_verified_rounds']=sum(x.get('status')=='succeeded' and x.get('verified_objects')==5 for x in state['results'])
        state['independent_physical_rounds']=sum(x.get('physical_completed') is True for x in state['results'])
        state['elapsed_seconds']=[x['elapsed_seconds'] for x in state['results'] if x.get('elapsed_seconds') is not None]
        save(out/'campaign.json',state)
        # Stop on infrastructure/stop problems, preserving the scene for review.
        if (result['commands'][-1]['phase']!='stop' or result['commands'][-1]['returncode']!=0 or
                result.get('code_changed_during_episode') or result.get('supervisor_still_running') or
                result.get('status')=='supervision_timeout'):
            state['status']='attention_required';save(out/'campaign.json',state);return 1
    state.update(status='completed',finished_at=time.time(),active_case=None)
    save(out/'campaign.json',state);print(json.dumps(state,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
