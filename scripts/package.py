"""Create a relocatable archive. ZIP frames and weights become ordinary archive files.

Example: python scripts/package.py --output D:/transfer/AHA-v2.tar
Does not create another 22GB archive unless explicitly run.
"""
import argparse
import os
import shutil
import tarfile
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
root=Path(__file__).resolve().parents[1]; output=args.output.resolve()
if output==root or root in output.parents:
    raise ValueError('Archive must be outside AHA to avoid recursive self-inclusion')
exclude={'.deps','.venv','__pycache__','.pytest_cache','runs'}
files=[]
for directory,dirs,names in os.walk(root):
    dirs[:]=[d for d in dirs if d not in exclude and not d.startswith('pytest_tmp') and d!='partial_features']
    files.extend(Path(directory)/n for n in names if not n.endswith('.partial'))
size=sum(x.stat().st_size for x in files)
output.parent.mkdir(parents=True,exist_ok=True)
if output.exists():
    raise FileExistsError(output)
if shutil.disk_usage(output.parent).free<size*1.02:
    raise OSError(f'Need about {size/1e9:.2f} GB free for materialized TAR')
with tarfile.open(output,'w',dereference=True) as tar:
    for path in files:
        tar.add(path,arcname='AHA/'+path.relative_to(root).as_posix(),recursive=False)
print(output)
