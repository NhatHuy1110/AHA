"""Aggregate ALL prespecified seeds of every evaluated variant; never pick the best test seed."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
KEYS=('supported_joint@5','joint@5','answer','grounding@5','grounding@10','onset_mae_valid')
result={}
for directory in sorted((root/'runs').iterdir()):
    if not (root/f'configs/{directory.name}.json').exists():
        continue
    for name in ('test_original','test_derived','test_original_cascade','test_derived_cascade',
                 'test_original_zero_visual','test_original_shuffle_visual'):
        paths=[directory/f'seed_{seed}/{name}_metrics/metrics.json' for seed in (17,29,43)]
        if not any(p.exists() for p in paths):
            continue
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f'{directory.name}/{name}: all three seeds are required')
        metrics=[json.loads(p.read_text())['primary'] for p in paths]
        result[f'{directory.name}/{name}']={k:dict(mean=float(np.mean([m[k] for m in metrics])),
            std=float(np.std([m[k] for m in metrics],ddof=1)),seeds=[m[k] for m in metrics])
            for k in KEYS if all(m[k] is not None for m in metrics)}
for method in ('surgformer_index','prior'):
    for qa in ('original','derived'):
        path=root/f'runs/{method}_test_{qa}/metrics.json'
        if path.exists():
            m=json.loads(path.read_text())['primary']
            result[f'{method}/test_{qa}']={k:m[k] for k in KEYS}
# Multimodal-LLM comparators: zero-shot runs are single deterministic runs; LoRA has seeds.
vlm=root/'runs/vlm'
if vlm.exists():
    for directory in sorted(vlm.iterdir()):
        for qa in ('original','derived'):
            single=directory/f'test_{qa}_metrics/metrics.json'
            seeds=sorted(directory.glob(f'seed_*/test_{qa}_metrics/metrics.json'))
            if single.exists():
                m=json.loads(single.read_text())['primary']
                result[f'vlm/{directory.name}/test_{qa}']={k:m[k] for k in KEYS}
            elif seeds:
                metrics=[json.loads(p.read_text())['primary'] for p in seeds]
                result[f'vlm/{directory.name}/test_{qa}']={k:dict(mean=float(np.mean([m[k] for m in metrics])),
                    std=float(np.std([m[k] for m in metrics],ddof=1)) if len(metrics)>1 else None,
                    seeds=[m[k] for m in metrics]) for k in KEYS if all(m[k] is not None for m in metrics)}
(root/'runs/seed_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
