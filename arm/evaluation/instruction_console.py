#!/usr/bin/env python3
"""入口 B · 指令控制台：手动输入自然语言指令 → 生成指令文件 → 一键启动回合。

与 --official-example 走完全相同的受控链路（run_round 入口、证据记录器、
托管校验全部保留），仅指令来源不同——操作员输入，非 truth feedback。
载体：GPUFree 远程桌面 Tkinter 窗口。仅 evaluation/ 新增文件，竞赛路径零改动。
"""
import json, os, queue, subprocess, threading, time
import tkinter as tk
from tkinter import ttk

ROOT = '/root/tcei_final_v2_23/tcei_260920v2'
RUNS = '/root/gpufree-data/tcei_260920v2'
ENV = {**os.environ, 'ROS_MASTER_URI': 'http://127.0.0.1:11311'}


def sh(cmd, timeout=40):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, executable='/bin/bash', env=ENV)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return str(e)


RESET_SCRIPT = r'''import json, time, os, subprocess, signal
import rospy
from std_msgs.msg import String
S = json.load(open('/root/tcei_final_v2_23/tcei_260920v2/state/active_stack.json'))['stack_dir']
rospy.init_node('entryb_reset', anonymous=True)
# 1) 停止锁存复位
try:
    m = rospy.wait_for_message('/tcei/stop_ack', String, timeout=4)
    d = json.loads(m.data); sid = d.get('id')
    if sid:
        pub = rospy.Publisher('/tcei/stop_reset', String, queue_size=1, latch=True)
        time.sleep(.5)
        pub.publish(String(json.dumps({'id': sid})))
        time.sleep(1.5)
        print('latch reset', sid[:8])
    else:
        print('latch clean')
except Exception as e:
    print('latch', e)
# 2) controller 重启（cancelled 锁存无 clear 路径，重启是规定动作）
subprocess.run('pkill -f "controller[.]py"', shell=True)
time.sleep(2)
env = {**os.environ}
r = subprocess.run(['bash', '-c', 'source /root/tcei_final_v2_23/tcei_260920v2/scripts/env.sh && cd /root/tcei_final_v2_23/tcei_260920v2/tcei_stack && setsid nohup /usr/bin/python3 -u controller.py _execute:=false _prepare_observation:=false _require_planner_feedback:=true _log_dir:=' + S + '/events _robot_projection_calibration:=/root/tcei_final_v2_23/tcei_260920v2/calibration/robot_projection.static_checked_v1.json >> ' + S + '/logs/controller.log 2>&1 < /dev/null & echo OK'], capture_output=True, text=True)
time.sleep(3)
pg = subprocess.run('pgrep -f "controller[.]py" | head -1', shell=True, capture_output=True, text=True).stdout.strip()
if pg:
    pid = int(pg.split()[0])
    st = open('/proc/%d/stat' % pid).read(); f = st[st.rfind(')') + 2:].split()
    rec = json.load(open(S + '/pids/controller.json'))
    rec['pid'] = pid; rec['pgid'] = int(f[2]); rec['start_ticks'] = int(f[19]); rec['started_at'] = time.time()
    json.dump(rec, open(S + '/pids/controller.json', 'w'), indent=2)
    print('controller restarted', pid)
else:
    print('controller FAIL')
# 3) 清栈锁
lock = S + '/round_started.lock'
if os.path.exists(lock):
    os.remove(lock)
    print('lock cleared')
else:
    print('lock clean')
'''


class Console:
    def __init__(self, root):
        self.root = root
        root.title('TCEI 入口 B · 指令控制台')
        root.geometry('740x580')
        self.round_name = tk.StringVar(value='（未启动）')
        self.status = tk.StringVar(value='空闲')
        self.queue = queue.Queue()
        self.current_round = None
        self.instr_file = None
        self._build()
        self.root.after(400, self._pump)
        self.root.after(1600, self._tick)

    def _build(self):
        pad = {'padx': 8, 'pady': 3}
        frm = ttk.Frame(self.root); frm.pack(fill='x', **pad)
        ttk.Label(frm, text='每行一条自然语言指令（留空即只执行填了的），默认预填官方五条').pack(anchor='w')
        self.entries = []
        examples = ['抓取左上方的烟雾弹，放到左侧传送带', '抓取右下方的弹夹，放到右侧传送带',
                    '抓取最左方的军用手电筒，放到左侧传送带', '抓取手雷，放到右侧传送带',
                    '抓取剩余的物品，放到左侧传送带']
        for i in range(5):
            row = ttk.Frame(frm); row.pack(fill='x', **pad)
            ttk.Label(row, text='指令%d' % (i + 1), width=6).pack(side='left')
            e = ttk.Entry(row)
            e.insert(0, examples[i])
            e.pack(side='left', fill='x', expand=True)
            self.entries.append(e)
        btns = ttk.Frame(self.root); btns.pack(fill='x', **pad)
        ttk.Button(btns, text='生成指令文件', command=self.on_generate).pack(side='left', padx=4)
        ttk.Button(btns, text='启动回合', command=self.on_launch).pack(side='left', padx=4)
        ttk.Button(btns, text='取消当前回合', command=self.on_cancel).pack(side='left', padx=4)
        ttk.Button(btns, text='回合间复位', command=self.on_reset).pack(side='left', padx=4)
        info = ttk.Frame(self.root); info.pack(fill='x', **pad)
        ttk.Label(info, text='当前回合：').pack(side='left')
        ttk.Label(info, textvariable=self.round_name).pack(side='left')
        ttk.Label(info, text='状态：').pack(side='left', padx=(18, 0))
        ttk.Label(info, textvariable=self.status).pack(side='left')
        logf = ttk.LabelFrame(self.root, text='回合日志'); logf.pack(fill='both', expand=True, **pad)
        self.log = tk.Text(logf, height=16, state='disabled', font=('Menlo', 10))
        self.log.pack(fill='both', expand=True)

    def _log(self, text):
        def _do():
            self.log.configure(state='normal')
            self.log.insert('end', text + '\n')
            self.log.see('end')
            self.log.configure(state='disabled')
        self.root.after(0, _do)

    def _bg(self, fn, done=None):
        def _w():
            try:
                result = fn()
            except Exception as e:
                result = 'ERR ' + str(e)
            self.queue.put((done, result))
        threading.Thread(target=_w, daemon=True).start()

    def _pump(self):
        try:
            while True:
                done, result = self.queue.get_nowait()
                self._log('[%s] %s' % (time.strftime('%H:%M:%S'), str(result)[:300]))
                if done:
                    done(result)
        except queue.Empty:
            pass
        self.root.after(400, self._pump)

    def _tick(self):
        def _read():
            if not self.current_round:
                return '空闲'
            sfp = os.path.join(RUNS, self.current_round, 'episode', 'summary.json')
            if not os.path.exists(sfp):
                return '回合 %s 执行中…' % self.current_round
            try:
                s = json.load(open(sfp))
            except Exception:
                return '回合 %s summary 解析中…' % self.current_round
            tasks = s.get('tasks', [])
            ok = sum(1 for t in tasks if (t.get('result') or {}).get('status') == 'task_succeeded')
            head = '回合 %s：成功 %d 条 | 状态 %s' % (self.current_round, ok, s.get('status'))
            tails = []
            for t in tasks[-3:]:
                r = t.get('result') or {}
                tails.append('%s → %s %s' % (t.get('instruction', '')[:18], r.get('status', '…'),
                                             str(r.get('reason', ''))[:36]))
            return head + '\n' + '\n'.join(tails)
        def _show(result):
            self.status.set(result.splitlines()[0])
            for line in result.splitlines()[1:]:
                self._log(line)
        self._bg(_read, _show)
        self.root.after(1600, self._tick)

    def on_generate(self):
        texts = [e.get().strip() for e in self.entries if e.get().strip()]
        if not texts:
            self._log('错误：至少填写一条指令')
            return
        stamp = time.strftime('%H%M%S')
        path = os.path.join(RUNS, 'manual_%s.json' % stamp)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=1)
        self.instr_file = path
        self.round_name.set('manual_%s（%d 条）' % (stamp, len(texts)))
        self._log('指令文件已生成：%s' % path)

    def on_launch(self):
        if not self.instr_file or not os.path.exists(self.instr_file):
            self._log('请先点击「生成指令文件」')
            return
        name = os.path.basename(self.instr_file).replace('.json', '')
        self.current_round = name
        cmd = ('cd %s && setsid nohup bash robot.sh run %s --instructions %s '
               '> /tmp/%s.out 2>&1 < /dev/null & echo LAUNCHED %s' % (ROOT, name, self.instr_file, name, name))
        self._log('启动回合 %s …' % name)
        self._bg(lambda: sh(cmd))

    def on_cancel(self):
        self._log('请求取消当前回合（cancel_and_confirm，含测量停止）…')
        self._bg(lambda: sh('cd %s && timeout 90 bash robot.sh cancel 2>&1 | tail -3' % ROOT, timeout=100))

    def on_reset(self):
        self._log('回合间复位：锁存复位 + controller 重启 + 清栈锁 …')
        def _go():
            open('/tmp/entryb_reset.py', 'w').write(RESET_SCRIPT)
            return sh('bash -c "source %s/scripts/env.sh && python3 /tmp/entryb_reset.py"' % ROOT, timeout=60)
        self._bg(_go)


def main():
    root = tk.Tk()
    Console(root)
    root.mainloop()


if __name__ == '__main__':
    main()
