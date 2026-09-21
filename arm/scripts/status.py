"""Read-only deployment/worker/latest-result status."""
import argparse,json,shutil
from common import ROOT,RUNS,stack_path,owned

def main():
    p=argparse.ArgumentParser();p.add_argument('--stack');args=p.parse_args();stack=stack_path(args.stack)
    out={'stack':str(stack),'workers':owned(stack),'free_bytes':shutil.disk_usage(RUNS).free}
    active=ROOT/'state/active_episode.json'
    if active.exists():
        out['episode']=json.loads(active.read_text());episode=RUNS/out['episode']['name']
        for name in ('episode/summary.json','supervisor_finished.json','supervisor_aborted.json'):
            if (episode/name).exists():out[name]=json.loads((episode/name).read_text())
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
