"""Configure this extracted package on a new OFFICIAL competition image."""
import argparse,hashlib,json,os,shlex,shutil,socket,subprocess,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--display');p.add_argument('--data-root',type=Path)
    args=p.parse_args();state=ROOT/'state';state.mkdir(exist_ok=True)
    required=('/root/EAICON/Content/JAKA/scene.usd','/root/EAICON/Source/JAKA/jaka_sim.py',
        '/root/EAICON/Source/JAKA/jaka_env.py','/root/jaka/best.pt','/root/isaacsim/python.sh',
        '/opt/ros/noetic/setup.bash','/opt/conda/envs/yolov8/bin/python','/opt/conda/envs/inference/bin/python')
    missing=[s for s in required if not Path(s).is_file()]
    if not Path('/root/inference/FM9G4B-V').is_dir():missing.append('/root/inference/FM9G4B-V')
    if missing:raise RuntimeError('官方镜像资源不完整：'+', '.join(missing))
    display=args.display or os.environ.get('DISPLAY')
    if not display:
        choices=[p.name[1:] for p in Path('/tmp/.X11-unix').glob('X*') if p.name[1:].isdigit()]
        if len(choices)==1:display=':'+choices[0]
    if not display:raise RuntimeError('未找到唯一桌面显示号。先启动平台桌面，再用 --display :实际显示号 初始化。')
    subprocess.run(['xdpyinfo','-display',display],check=True,timeout=10,stdout=subprocess.DEVNULL)
    code=ROOT/'tcei_stack';manifest=json.loads((code/'BUILD_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((code/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('运行代码校验不通过：'+name)
    marker=state/'INSTANCE_ID.txt'
    if marker.exists():ident=marker.read_text().strip()
    else:
        ident=socket.gethostname()+'-'+uuid.uuid4().hex[:12]
        marker.open('x').write(ident+'\n')
    data=(args.data_root or (Path('/root/gpufree-data/tcei_260920v2') if Path('/root/gpufree-data').is_dir() else ROOT/'runs')).resolve()
    data.mkdir(parents=True,exist_ok=True)
    values={'DISPLAY':display,'TCEI_CODE':str(code),'TCEI_V7_CODE':str(code),'TCEI_RUNS':str(data),
        'TCEI_V7_RUNS':str(data),'TCEI_SCENE':'/root/EAICON','TCEI_WEIGHTS':'/root/jaka/best.pt',
        'TCEI_MODEL':'/root/inference/FM9G4B-V','TCEI_ISAAC_PY':'/root/isaacsim/python.sh',
        'TCEI_YOLO_PY':'/opt/conda/envs/yolov8/bin/python','TCEI_NINE_PY':'/opt/conda/envs/inference/bin/python',
        'TCEI_ROBOT_CALIBRATION':str(ROOT/'calibration/robot_projection.static_checked_v1.json'),
        'TCEI_INSTANCE_FILE':str(marker),'TCEI_INSTANCE_ID':ident,'TCEI_PACKAGE_ROOT':str(ROOT),
        'ROS_MASTER_URI':'http://127.0.0.1:11311'}
    config=state/'instance.env'
    config.write_text('\n'.join('export '+k+'='+shlex.quote(v) for k,v in values.items())+'\n')
    os.chmod(config,0o600)
    (state/'configuration.json').write_text(json.dumps({'package':str(ROOT),'runtime':manifest['version'],
        'display':display,'runs':str(data),'instance_id':ident,'free_bytes':shutil.disk_usage(data).free},ensure_ascii=False,indent=2))
    print('初始化完成。下一步：bash robot.sh doctor')

if __name__=='__main__':main()
