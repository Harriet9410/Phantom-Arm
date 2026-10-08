"""Start the existing vendor desktop only when its port is not listening."""
from pathlib import Path
import json,socket,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main():
    with socket.socket() as s:
        s.settimeout(2.)
        if s.connect_ex(('127.0.0.1',3000))==0:
            print('平台桌面端口已运行，请从平台控制台打开该实例的远程桌面。');return 0
    sys.path.insert(0,str(ROOT/'harness'))
    from process_guard import current,read_record
    state=ROOT/'state';state.mkdir(exist_ok=True);record=state/'desktop.pid.json'
    if record.exists():
        if current(read_record(record)):raise RuntimeError('上次桌面进程仍在启动；请查看state/desktop.log，不重复启动')
        raise RuntimeError('已有桌面启动记录；保留日志后使用新的部署目录处理，不自动覆盖失败')
    return subprocess.run(['/usr/bin/python3',str(ROOT/'harness/process_guard.py'),'start',str(record),str(ROOT),
        str(state/'desktop.log'),'--','bash',str(ROOT/'scripts/desktop_boot.sh')]).returncode
if __name__=='__main__':raise SystemExit(main())
