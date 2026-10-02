"""Server side: assemble the full AHA folder from the git checkout + the private Hugging Face dataset.

  hf auth login            # token with read access to the private dataset
  python scripts/fetch_data.py --repo NhatHuy1110/AHA --protocol AHA-M1-v2.0
  python -m aha verify     # every frozen file must match its SHA-256

Frame ZIPs come from part_*/NNN.zip in the dataset; everything else from bundle/<protocol>/.
"""
import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--protocol',required=True)
args=p.parse_args()
root=Path(__file__).resolve().parents[1]; stage=root/'.hf_download'
snapshot_download(args.repo,repo_type='dataset',local_dir=stage,
                  allow_patterns=['part_*/*.zip',f'bundle/{args.protocol}/**'])
bundle=stage/'bundle'/args.protocol
if not (bundle/'FREEZE.json').exists():
    raise SystemExit(f'No bundle for {args.protocol} in {args.repo}: run scripts/upload_bundle.py first')
moved=0
for source in sorted(bundle.rglob('*')):
    if source.is_file():
        target=root/source.relative_to(bundle)
        if target.exists():
            continue  # git-tracked or already placed; verify decides whether it is right
        target.parent.mkdir(parents=True,exist_ok=True); shutil.move(source,target); moved+=1
for source in sorted(stage.glob('part_*/*.zip')):
    target=root/'data/frames'/source.name
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True); shutil.move(source,target); moved+=1
shutil.rmtree(stage)
print(f'placed {moved} files; now run: python -m aha verify')
