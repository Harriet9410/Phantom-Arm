import hashlib, pathlib, shutil, time
p = pathlib.Path('/root/EAICON/Source/JAKA/jaka_sim.py')
src = p.read_text(encoding='utf-8')
old = 'create.render_product("/World/Cameras/top", (320, 240))'
new = 'create.render_product("/World/Cameras/top", (1280, 720))'
print('occurrences of old:', src.count(old))
assert src.count(old) == 1
bak = p.with_name('jaka_sim.py.bak_320x240')
shutil.copy2(p, bak)
p.write_text(src.replace(old, new), encoding='utf-8')
print('backup:', bak)
for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
    if 'Cameras/top' in line and ('render_product' in line or 'create_viewport_window' in line or 'width=' in line or 'height=' in line):
        print('  L%d: %s' % (i, line.strip()))
print('sha256:', hashlib.sha256(p.read_bytes()).hexdigest())