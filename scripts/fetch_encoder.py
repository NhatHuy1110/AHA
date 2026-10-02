"""Download a frozen model at an immutable revision into assets/<kind>/<name>/.

Examples:
  python scripts/fetch_encoder.py --id facebook/dinov2-large --name dinov2_vitl14
  python scripts/fetch_encoder.py --kind vlm --family qwen3vl --id Qwen/Qwen3-VL-8B-Instruct --name qwen3vl_8b
  python scripts/fetch_encoder.py --kind vlm --family hulumed --id ZJU-AI4H/Hulu-Med-7B --name hulumed_7b
"""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p=argparse.ArgumentParser(); p.add_argument('--id',required=True); p.add_argument('--name',required=True)
p.add_argument('--kind',choices=['encoders','vlm'],default='encoders')
p.add_argument('--family',choices=['qwen3vl','hulumed'],help='required for --kind vlm: which loader src/aha/vlm.py uses')
p.add_argument('--revision',help='40-character commit; default resolves the current main once and pins it')
args=p.parse_args()
if (args.kind=='vlm')!=bool(args.family):
    raise SystemExit('--family is required for, and only valid with, --kind vlm')
root=Path(__file__).resolve().parents[1]; target=root/'assets'/args.kind/args.name
if (target/'source.json').exists():
    raise FileExistsError(f'{target} already staged; models are immutable once used')
revision=args.revision or HfApi().model_info(args.id).sha
if len(revision)!=40:
    raise ValueError('Need a full commit SHA')
patterns=dict(allow_patterns=['*.json','*.safetensors']) if args.kind=='encoders' else dict(ignore_patterns=['*.md','.gitattributes'])
snapshot_download(args.id,revision=revision,local_dir=target,**patterns)
source=dict(id=args.id,revision=revision)
if args.family:
    source['family']=args.family
(target/'source.json').write_text(json.dumps(source,indent=2)+'\n')
print(json.dumps(dict(staged=str(target),**source)))
