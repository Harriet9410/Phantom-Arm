"""Package-local ownership and path helpers; importing sends no commands."""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=Path(os.environ.get('TCEI_RUNS',str(ROOT/'runs'))).resolve()
sys.path[:0]=[str(ROOT/'tcei_stack'),str(ROOT/'harness')]
from process_guard import current, read_record


def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',name):
        raise ValueError('轮次名称只允许字母、数字、下划线、连字符和点，最长64字符')
    return RUNS/name


def stack_path(name=None):
    if name:return run_path(name)
    value=json.loads((ROOT/'state/active_stack.json').read_text())
    path=Path(value['stack_dir']).resolve()
    if path.parent!=RUNS:raise ValueError('记录不属于当前部署的数据目录')
    return path


def guard(action,record,*args):
    return subprocess.run(['/usr/bin/python3',str(ROOT/'harness/process_guard.py'),
        action,str(record),*map(str,args)],capture_output=True,text=True,timeout=35)


def owned(stack,names=('sim','controller','perception','nine')):
    return {name: (stack/'pids'/(name+'.json')).is_file() and
            current(read_record(stack/'pids'/(name+'.json'))) for name in names}


def start_owned(record,cwd,log,command):
    result=guard('start',record,cwd,log,'--',*command)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return read_record(record)


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+str(time.time_ns())+'.tmp')
    write_new(temp,value);temp.replace(path)
