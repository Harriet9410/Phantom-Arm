#!/usr/bin/env python3
"""Finalize an aborted batch's campaign.json so it stops misreporting status='running'.

The round-2 review flagged that a batch aborted mid-run (runner killed by SIGTERM)
leaves campaign.json at status='running' forever, which misstates the evidence.
This tool:
  1. backs up the original to campaign.original.json (once)
  2. sets status='aborted_by_operator' plus aborted_at / aborted_note
  3. adds 'case_roll': for every registered case, one of
       completed_or_stopped / aborted_mid_run / not_initiated
  4. never touches per-case results (they stay exactly as recorded)

Usage:
  python3 campaign_finalize.py <batch_dir> --register <register.json> [--note "..."]
"""
import json
import pathlib
import sys
import time


def read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    batch = pathlib.Path(sys.argv[1]).resolve()
    args = sys.argv[2:]
    register_path = args[args.index('--register') + 1] if '--register' in args else None
    note = args[args.index('--note') + 1] if '--note' in args else 'runner process terminated by operator; per-case records preserved as-is'
    campaign_path = batch / 'campaign.json'
    campaign = read_json(campaign_path)
    if campaign is None:
        print('campaign.json not readable:', campaign_path)
        return 1
    register = read_json(register_path) if register_path else None
    backup = batch / 'campaign.original.json'
    if not backup.exists():
        backup.write_text(json.dumps(campaign, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    planned_ids = [r['case_id'] for r in (register or {}).get('cases', [])]
    done_ids = [r.get('case_id') for r in campaign.get('results', [])]
    roll = []
    for cid in planned_ids or done_ids:
        if cid in done_ids:
            roll.append({'case_id': cid, 'roll': 'completed_or_stopped'})
        elif cid == campaign.get('active_case'):
            roll.append({'case_id': cid, 'roll': 'aborted_mid_run'})
        else:
            roll.append({'case_id': cid, 'roll': 'not_initiated'})
    if campaign.get('status') == 'running':
        campaign['status'] = 'aborted_by_operator'
    campaign['aborted_at'] = time.time()
    campaign['aborted_note'] = note
    campaign['case_roll'] = roll
    campaign['case_roll_summary'] = {
        'completed_or_stopped': sum(1 for r in roll if r['roll'] == 'completed_or_stopped'),
        'aborted_mid_run': sum(1 for r in roll if r['roll'] == 'aborted_mid_run'),
        'not_initiated': sum(1 for r in roll if r['roll'] == 'not_initiated'),
    }
    campaign_path.write_text(json.dumps(campaign, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('finalized:', campaign_path)
    print('  status =', campaign['status'])
    print('  roll   =', ', '.join('%s=%s' % (r['case_id'], r['roll']) for r in roll))
    print('  backup =', backup)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
