"""Check the exact ZIP payload manifest without importing robotics dependencies."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    bad=[]
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve()
        if ROOT not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            bad.append(name)
    print(json.dumps({'package':manifest['package'],'files':len(manifest['files']),
                      'ok':not bad,'mismatches':bad},ensure_ascii=False,indent=2))
    return 1 if bad else 0

if __name__=='__main__':raise SystemExit(main())
