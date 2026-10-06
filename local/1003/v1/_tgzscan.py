import zipfile, io, tarfile, collections
Z = r'D:\MYCODE\TCEI\9.29存档.zip'
z = zipfile.ZipFile(Z)
with z.open('存档/TCEI_jiuge_archive_20260929.tgz') as fh:
    data = fh.read()
tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')
mem = [m for m in tf.getmembers() if m.isfile()]

print('=== 根层文件（3 个）===')
for m in mem:
    p = m.name.lstrip('./')
    if '/' not in p:
        print('  %-40s %10.1f KB' % (p, m.size/1024))

print()
print('=== scripts/（33 个）===')
for m in sorted([m for m in mem if m.name.lstrip('./').startswith('scripts/')], key=lambda x: x.name):
    print('  %-52s %8.1f KB' % (m.name.lstrip('./')[len('scripts/'):][:52], m.size/1024))

print()
print('=== outputs/（56 个）===')
outs = sorted([m for m in mem if m.name.lstrip('./').startswith('outputs/')], key=lambda x: -x.size)
for m in outs[:40]:
    print('  %-64s %9.2f MB' % (m.name.lstrip('./')[len('outputs/'):][:64], m.size/1e6))
print('  ...' if len(outs) > 40 else '')
print()
print('outputs/ 子目录聚合:')
agg = collections.Counter(); sz = collections.Counter()
for m in outs:
    rel = m.name.lstrip('./')[len('outputs/'):]
    k = '/'.join(rel.split('/')[:2]) if '/' in rel else '(直属)'
    agg[k] += 1; sz[k] += m.size
for k, v in sorted(sz.items(), key=lambda x: -x[1]):
    print('  %-56s %3d 个 %9.2f MB' % (k[:56], agg[k], v/1e6))

print()
print('=== 关键判据：是否含 adapter / lora / merged 权重 ===')
keys = ('adapter', 'lora', 'merges', 'pytorch_model.bin', 'model.safetensors', 'optimizer', 'scheduler', 'trainer_state')
for m in mem:
    p = m.name.lstrip('./').lower()
    if any(k in p for k in keys):
        print('  %-70s %8.2f MB' % (m.name.lstrip('./')[:70], m.size/1e6))
