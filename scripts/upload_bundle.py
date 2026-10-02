"""Upload everything the server needs that is NOT in git and NOT a frame ZIP to the private
Hugging Face dataset, under bundle/<protocol>/ with the same relative paths as this folder.

That is: labels, questions/targets, held-out gold, manifests, SurgFormer checkpoint and vendored
source, MiniLM weights, references, proposal.md and FREEZE.json. Run from the machine that holds
the full AHA folder:   python scripts/upload_bundle.py --repo NhatHuy1110/AHA
"""
import argparse
import json
import subprocess
from pathlib import Path
from huggingface_hub import CommitOperationAdd, HfApi

p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--dry-run',action='store_true')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
frozen=json.loads((root/'FREEZE.json').read_text(encoding='utf-8'))
tracked=set(subprocess.run(['git','ls-files'],cwd=root,capture_output=True,text=True,check=True).stdout.split('\n'))
files=sorted(f for f in list(frozen['files'])+['FREEZE.json']
             if f not in tracked and not (f.startswith('data/frames/') and f.endswith('.zip')))
size=sum((root/f).stat().st_size for f in files)
prefix=f'bundle/{frozen["protocol"]}'
print(f'{len(files)} files, {size/1e6:.1f} MB -> {args.repo}:{prefix}/')
if not args.dry_run:
    api=HfApi()
    if not api.repo_info(args.repo,repo_type='dataset').private:
        raise SystemExit('Refusing to upload held-out gold and third-party data to a public dataset')
    api.create_commit(args.repo,repo_type='dataset',commit_message=f'AHA bundle {frozen["protocol"]}',
        operations=[CommitOperationAdd(path_in_repo=f'{prefix}/{f}',path_or_fileobj=str(root/f)) for f in files])
    print('uploaded')
