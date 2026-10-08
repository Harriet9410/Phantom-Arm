#!/usr/bin/env python3
"""Own only recorded processes; protect against stale PID reuse. Linux only."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def identity(pid):
    # comm can contain spaces/parentheses; field 3 starts after the last ')'.
    text = Path('/proc/%d/stat' % pid).read_text()
    fields = text[text.rfind(')') + 2:].split()
    if fields[0] == 'Z':
        return None
    return {'pid': pid, 'pgid': int(fields[2]), 'start_ticks': int(fields[19])}


def current(record):
    try:
        got = identity(record['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    return got is not None and all(got[k] == record[k] for k in got)


def read_record(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    start = sub.add_parser('start')
    start.add_argument('record'); start.add_argument('cwd'); start.add_argument('log')
    start.add_argument('command', nargs=argparse.REMAINDER)
    for name in ('check', 'stop', 'owns-node'):
        q = sub.add_parser(name); q.add_argument('record')
        if name == 'stop': q.add_argument('--timeout', type=float, default=20)
        if name == 'owns-node': q.add_argument('node')
    sub.add_parser('conflicts')
    a = p.parse_args()
    if a.action == 'conflicts':
        found = []
        for item in Path('/proc').iterdir():
            if not item.name.isdigit(): continue
            try:
                args = (item/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except (OSError, PermissionError): continue
            for arg in args:
                name = Path(arg).name
                if name in ('randomized_sim.py', 'original_layout_sim.py', 'sim_with_feedback.py', 'jaka_sim.py', 'isaac_grasp', 'isaac_scale.py', 'isaac_yolov8.py', 'isaac_yolov8'):
                    found.append({'pid': int(item.name), 'command': args}); break
                if '/tcei_stack/' in arg and name in ('controller.py', 'perception.py', 'nine_node.py', 'evidence_recorder.py', 'record_transport_rgbd.py', 'run_episode.py'):
                    found.append({'pid': int(item.name), 'command': args}); break
        print(json.dumps({'conflicts': found}, ensure_ascii=False, indent=2))
        return 1 if found else 0
    if a.action == 'start':
        command = a.command[1:] if a.command[:1] == ['--'] else a.command
        if not command: raise ValueError('empty command')
        record_path = Path(a.record)
        # Exclusive creation avoids replacing a previous ownership record.
        with record_path.open('x') as handle:
            with open(a.log, 'ab', buffering=0) as log:
                child = subprocess.Popen(command, cwd=a.cwd, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, close_fds=True)
            record = identity(child.pid)
            if record is None: raise RuntimeError('process immediately exited')
            if record['pgid'] != child.pid:
                raise RuntimeError('process was not placed in a private group')
            record.update(command=command, cwd=a.cwd, log=a.log, started_at=time.time())
            json.dump(record, handle, ensure_ascii=False, indent=2)
        print(json.dumps(record, ensure_ascii=False))
        return 0
    if not Path(a.record).is_file():
        print('No ownership record: ' + a.record, file=sys.stderr)
        return 0 if a.action == 'stop' else 1
    record = read_record(a.record)
    if a.action == 'check': return 0 if current(record) else 1
    if a.action == 'owns-node':
        if not current(record): return 1
        socket.setdefaulttimeout(5)
        import rosgraph
        import xmlrpc.client
        master = rosgraph.Master('/tcei_deploy_owner_check')
        uri = master.lookupNode(a.node)
        code, message, pid = xmlrpc.client.ServerProxy(uri).getPid('/tcei_deploy_owner_check')
        return 0 if code == 1 and pid == record['pid'] else 1
    if not current(record):
        print('Already stopped or PID identity changed; no signal sent: ' + a.record)
        return 0
    if record['pgid'] != record['pid']:
        raise RuntimeError('refusing signal to a non-private process group')
    os.killpg(record['pgid'], signal.SIGINT)
    deadline = time.monotonic() + a.timeout
    while current(record) and time.monotonic() < deadline: time.sleep(.2)
    if current(record):
        print('Still exiting after SIGINT; not force-killed: ' + a.record, file=sys.stderr)
        return 1
    print('Stopped owned process: ' + a.record)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        raise SystemExit(1)
