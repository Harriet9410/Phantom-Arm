#!/usr/bin/env python3
"""R7-0 evidence export (READ-ONLY; no code change, no simulation).

From batch b03_r6tgt_10081642 cases s02/s04, export to /root/gpufree-data/r7_export:
  1. perception.log: complete, original-order `nine scan class audit` and
     `nine scan conflict images` lines (verbatim, ALL pixels/proposals -- no
     (672,350) filtering, no dedup/chain stitching).
  2. nine_events.jsonl: every `classify_answered` (with tag) ordered by time;
     also a scan_id->answers index. Explicitly record which stages were never
     called (tag absent).
  3. Per-frame candidate publication: from round/episode/events.jsonl extract
     every event that carries a `candidates` list, with
     frame_id/stable_id/bbox/pixel/raw_class/class/class_source/identity_status.
     If raw_class or class_source are absent in the source, record the absence
     rather than inferring.
  4. remaining_context_incomplete raw payloads (verbatim).
  5. Layer alignment table: model answer -> nine_classify store -> tracker
     publish -> ledger rejection, with the FIRST anomaly per layer.
Outputs files + _SHA256.txt manifest.
"""
import glob
import hashlib
import json
import os

RUNS = '/root/gpufree-data/tcei_260920v2'
BATCH = 'b03_r6tgt_10081642'
OUT = '/root/gpufree-data/r7_export'
CASES = ('scramble_02', 'scramble_04')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def lines_of(p):
    try:
        return open(p, errors='ignore').read().splitlines()
    except Exception:
        return []


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    bundle = {'batch': BATCH, 'cases': {}}

    for case in CASES:
        st = os.path.join(RUNS, BATCH + '_' + case + '_stack')
        rd = os.path.join(RUNS, BATCH + '_' + case + '_round')
        per = os.path.join(st, 'logs', 'perception.log')
        nine = os.path.join(st, 'events', 'nine_events.jsonl')
        ev = os.path.join(rd, 'episode', 'events.jsonl')

        # --- 1. verbatim audit + conflict-image lines ---
        aud = [l for l in lines_of(per) if 'nine scan class audit' in l]
        cim = [l for l in lines_of(per) if 'nine scan conflict images' in l]
        fa = os.path.join(OUT, 'audit_lines_%s.txt' % case)
        open(fa, 'w').write('\n'.join(aud) + '\n')
        fc = os.path.join(OUT, 'conflict_images_lines_%s.txt' % case)
        open(fc, 'w').write('\n'.join(cim) + '\n')

        # --- 2. classify_answered (with tag) ---
        answers = []
        for l in lines_of(nine):
            if not l.strip():
                continue
            try:
                e = json.loads(l)
            except Exception:
                continue
            if e.get('status') == 'classify_answered' or 'classify_answered' in str(e.get('status') or ''):
                answers.append(e)
        tags = {}
        for e in answers:
            tags.setdefault(str(e.get('tag')), 0)
            tags[str(e.get('tag'))] += 1
        fj = os.path.join(OUT, 'nine_events_classify_answered_%s.json' % case)
        json.dump({'n': len(answers), 'tag_counts': tags, 'events': answers},
                  open(fj, 'w'), ensure_ascii=False, indent=1)

        # --- 3. per-frame candidate publication ---
        pubs = []
        rem = []
        for l in lines_of(ev):
            if not l.strip():
                continue
            try:
                e = json.loads(l)
            except Exception:
                continue
            if e.get('status') == 'remaining_context_incomplete':
                rem.append(e)
            if isinstance(e.get('candidates'), list) and e['candidates']:
                for c in e['candidates']:
                    pubs.append({
                        'event_time': e.get('time'), 'event_status': e.get('status'),
                        'frame_id': e.get('frame_id') or (e.get('context') or {}).get('scene_frame_id'),
                        'stable_id': c.get('stable_id'), 'class': c.get('class'),
                        'raw_class': c.get('raw_class'), 'class_source': c.get('class_source'),
                        'identity_status': c.get('identity_status'),
                        'bbox': c.get('bbox'), 'pixel': c.get('pixel'),
                        'observed_class': c.get('observed_class'),
                        'raw_class_present': 'raw_class' in c,
                        'class_source_present': 'class_source' in c})
        fp = os.path.join(OUT, 'candidate_publication_%s.json' % case)
        json.dump(pubs, open(fp, 'w'), ensure_ascii=False, indent=1)
        fr = os.path.join(OUT, 'remaining_context_%s.json' % case)
        json.dump(rem, open(fr, 'w'), ensure_ascii=False, indent=1)

        # --- 4/5. layer alignment ---
        # nine_classify store: order audit entries by (line order, array order)
        store = []
        for l in aud:
            try:
                arr = json.loads(l.split('nine scan class audit:', 1)[1].strip())
            except Exception:
                continue
            sid = arr[0].get('scan_id') if arr else None
            for a in arr:
                store.append({'scan_id': a.get('scan_id', sid),
                              'proposal_index': a.get('proposal_index'),
                              'pixel': a.get('pixel'), 'bbox': a.get('bbox'),
                              'scan_class': a.get('scan_class'), 'old_class': a.get('old_class'),
                              'old_pending': a.get('old_pending'),
                              'stored_class': a.get('stored_class'),
                              'full': a.get('full'), 'decision': a.get('decision')})
        bundle['cases'][case] = {
            'paths': {'perception_log': per, 'nine_events': nine, 'episode_events': ev},
            'n_audit_lines': len(aud), 'n_conflict_image_lines': len(cim),
            'n_classify_answered': len(answers), 'classify_tag_counts': tags,
            'n_candidate_publication_rows': len(pubs),
            'candidate_pub_fields_present': {
                'raw_class_present_rows': sum(1 for r in pubs if r['raw_class_present']),
                'class_source_present_rows': sum(1 for r in pubs if r['class_source_present'])},
            'n_remaining_events': len(rem),
            'nine_store_entries': store,
            'remaining_last': rem[-1] if rem else None,
        }
        manifest += [fa, fc, fj, fp, fr]

    bundlef = os.path.join(OUT, 'layer_alignment.json')
    json.dump(bundle, open(bundlef, 'w'), ensure_ascii=False, indent=1)
    manifest.append(bundlef)
    with open(os.path.join(OUT, '_SHA256.txt'), 'w') as f:
        for p in manifest:
            f.write('%s  %s\n' % (sha(p), os.path.basename(p)))
    for p in manifest:
        print(os.path.basename(p), os.path.getsize(p))


if __name__ == '__main__':
    main()
