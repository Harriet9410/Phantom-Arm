"""Replace only the owned review window; preserve geometry and old feedback."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/tcei_competition_20260919');OUT=ROOT/'review_ui_04'
SCRIPT=ROOT/'desktop_review_v4b.py';CONFIG=ROOT/'review_ui_02/config.json'
assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest()=='ca4ae738c6bd37b5e8fdaacd435261e1d6f2e8d5dfe26335f587100806e51085'
assert not OUT.exists()
old_pid=374772;command=Path('/proc/%d/cmdline'%old_pid).read_bytes().decode().split('\0')
assert str(ROOT/'desktop_review_v3.py') in command and str(ROOT/'review_ui_03') in command
env=os.environ.copy();env['DISPLAY']=':20'
rows=subprocess.run(['wmctrl','-lpG'],env=env,capture_output=True,text=True,check=True).stdout.splitlines()
matches=[x for x in rows if x.endswith('TCEI 联合核对：YOLO · 九格 · Isaac')]
assert len(matches)==1
parts=matches[0].split(None,8);assert parts[0]=='0x02e00049'
x,y,width,height=map(int,parts[3:7])
OUT.mkdir()
(OUT/'prior_window.json').write_text(json.dumps({'window':matches[0],'pid':old_pid,'command':command},ensure_ascii=False,indent=2))
(OUT/'previous_config.json').write_text(CONFIG.read_text())
config=json.loads(CONFIG.read_text());config['review_image']='/root/gpufree-data/tcei_competition_20260919/official16_round01/delivery_review16_01/task02_release_plus_3000ms.jpg'
CONFIG.write_text(json.dumps(config,ensure_ascii=False,indent=2))
subprocess.run(['wmctrl','-ic',parts[0]],env=env,check=True)
until=time.monotonic()+15.
while True:
    proc=Path('/proc/%d/stat'%old_pid)
    if not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[0]=='Z':break
    assert time.monotonic()<until,'old review did not finish closing; no second panel started'
    time.sleep(.1)
env['TCEI_REVIEW_GEOMETRY']='%dx%d+%d+%d'%(width,height,x,y)
sys.path.insert(0,str(ROOT/'releases/dev_v7_t1_16/harness'))
from process_guard import identity
with (OUT/'panel.log').open('xb') as log:
    child=subprocess.Popen(['/usr/bin/python3','-u',str(SCRIPT),'--config',str(CONFIG),'--output',str(OUT)],
        cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=identity(child.pid);assert record is not None
record.update(command=[str(SCRIPT),'--config',str(CONFIG),'--output',str(OUT)],started_at=time.time(),
              script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),geometry=env['TCEI_REVIEW_GEOMETRY'])
(OUT/'panel.pid.json').write_text(json.dumps(record,indent=2))
print('REVIEW_V4B_STARTED',child.pid,flush=True)
