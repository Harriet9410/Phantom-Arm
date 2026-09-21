from pathlib import Path
import tarfile,hashlib,shutil
root=Path('/root/tcei_stress_20260918')
shutil.copytree(root/'candidate_v3',root/'candidate_v3_before_partial_occlusion')
for name,digest in [('recognition_pose_v3c.tar.gz','1e2f9e5e7dfc8ebcc3dd32554d3ae77a1ee646ea91bbc2adc3ac4ac639a001aa'),('holdout_cases_30.tar.gz','cfdbb3edd4a94d9e7474ffc3a8eabf68f66ac33e90adde9bf89fa8746aa5cd71')]:
 p=Path('/root/Pictures')/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 with tarfile.open(p) as t:
  assert all(not n.name.startswith('/') and '..' not in n.name.split('/') and n.isfile() for n in t.getmembers());t.extractall(root)
