"""Download a frozen image encoder at an immutable revision into assets/encoders/<name>/.

Example:
  python scripts/fetch_encoder.py --id facebook/dinov2-large --name dinov2_vitl14
"""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p=argparse.ArgumentParser(); p.add_argument('--id',required=True); p.add_argument('--name',required=True)
p.add_argument('--revision',help='40-character commit; default resolves the current main once and pins it')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]; target=root/'assets/encoders'/args.name
if (target/'source.json').exists():
    raise FileExistsError(f'{target} already staged; encoders are immutable once used')
revision=args.revision or HfApi().model_info(args.id).sha
if len(revision)!=40:
    raise ValueError('Need a full commit SHA')
snapshot_download(args.id,revision=revision,local_dir=target,allow_patterns=['*.json','*.safetensors'])
(target/'source.json').write_text(json.dumps(dict(id=args.id,revision=revision),indent=2)+'\n')
print(json.dumps(dict(staged=str(target),id=args.id,revision=revision)))
