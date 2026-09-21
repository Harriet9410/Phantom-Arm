"""Read-only re-audit of A01. Preserves the original harness status error."""
from pathlib import Path
import hashlib,json,math
from loaded_stop_checks import J,P,C,E,K,position,held_stationary

def validate_interrupted_closeout(finished,summary,stop_id):
    status=finished.get('episode_status')
    if status not in ('stopped','failed') or summary.get('status')!=status:
        raise ValueError('not a matching interrupted episode')
    if finished.get('execution_disabled') is not True or finished.get('evidence_complete') is not True:
        raise ValueError('execution or evidence not closed')
    if finished.get('returncode')!=1 or summary.get('verified_objects')!=0 or summary.get('code_changed_during_episode'):
        raise ValueError('unexpected completion or code mutation')
    stop=summary.get('safe_stop',{});proof=stop.get('proof',{})
    if (stop.get('state')!='stopped' or proof.get('state')!='stopped' or
            proof.get('id')!=stop_id or proof.get('fault_reason') is not None):
        raise ValueError('no matching measured stop in driver closeout')
    speed=proof.get('max_joint_speed')
    if type(speed) not in (int,float) or not math.isfinite(speed) or not 0<=speed<=.03:
        raise ValueError('invalid measured stop speed')
    return True

def main():
    root=Path('/root/tcei_competition_20260919')
    run=Path('/root/gpufree-data/tcei_competition_20260919/H01_loaded18_A01')
    probe=json.loads((run/'probe/summary.json').read_text())
    finished=json.loads((run/'supervisor_finished.json').read_text())
    summary=json.loads((run/'episode/summary.json').read_text())
    assert probe['status']=='failed' and probe['error']=="AssertionError('a deliberate cancellation cannot be a normal completed round')"
    assert not probe['callback_errors'] and probe['cancel_sent'] and not probe['reset_sent']
    rows=[json.loads(line) for line in (run/'probe/events.jsonl').read_text().splitlines()]
    mappings=[r for r in rows if r['topic']=='/tcei/cancel_ack' and r['mono']>=probe['cancel_at']
              and r['value'].get('cancel_id')==probe['cancel_request']['cancel_id'] and r['value'].get('state')=='accepted']
    assert mappings;mapping=mappings[0];sid=mapping['value']['stop_id']
    validate_interrupted_closeout(finished,summary,sid)
    acks=[r for r in rows if r['topic']=='/tcei/stop_ack' and r['mono']>=probe['cancel_at'] and r['value'].get('id')==sid]
    assert not any(r['value'].get('state')=='fault' for r in acks)
    stopped=next(r for r in acks if r['value'].get('state')=='stopped' and r['value'].get('fault_reason') is None)
    independent=held_stationary(rows,stopped['mono'],rows[-1]['mono'])
    assert independent is not None,'full post-stop window failed'
    prior=[r for r in rows if r['topic']==P and r['mono']<=probe['cancel_at']]
    start=position(prior[-1]['value'])
    post=[position(r['value']) for r in rows if r['topic']==P and r['mono']>=probe['cancel_at']]
    travel=max(math.dist(start,p) for p in post);assert travel<=.02
    commands=[r for r in rows if r['topic'] in ('/Jaka/set_end_effector_pose','/Jaka/set_gripper_value','/tcei/execute_preview')
              and r['mono']>mapping['mono']+.1]
    assert not commands
    echo=[r for r in rows if r['topic']=='/tcei/cancel_request' and r['value'].get('cancel_id')==probe['cancel_request']['cancel_id']]
    assert len(echo)==1,'own cancellation must occur exactly once'
    files=('probe/summary.json','probe/events.jsonl','episode/summary.json','supervisor_finished.json')
    out={'status':'passed_recorded_loaded_dynamic_stop_subcase','physical_rerun':False,
        'original_probe_status':probe['status'],'original_probe_error':probe['error'],
        'original_summary_preserved':True,'driver_status':summary['status'],'stop_id':sid,
        'cancellation_echo_count':len(echo),'stop_confirmation_seconds':stopped['mono']-probe['cancel_at'],
        'max_tcp_travel_after_cancel_m':travel,'late_action_command_count':len(commands),
        'independent_post_stop':independent,'execution_disabled':finished['execution_disabled'],
        'evidence_complete':finished['evidence_complete'],
        'raw_sha256':{n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in files},
        'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'One measured upward loaded-stop subcase. Does not settle17 contact-at-source stop failure, test reset, or certify complete H01/H06.'}
    (run/'probe/review_after_status_fix.json').open('x').write(json.dumps(out,ensure_ascii=False,indent=2))
    (root/'h01_loaded18_audited.txt').open('x').write(json.dumps(out,ensure_ascii=False,indent=2))
    print('H01_LOADED18_RECORDED_AUDIT',out['status'],travel,flush=True)

if __name__=='__main__':main()
