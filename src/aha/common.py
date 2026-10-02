import hashlib
import json
import os
from pathlib import Path

PHASES = ['Preparation', 'SuspendPericardium', 'DissociateVein',
          'SuturePerfusionNeedleSpacer', 'InsertPerfusionNeedle', 'BlockAorta',
          'LeftHeartDrain', 'DissectLeftAtrium', 'ExposeMitralValve',
          'MitralValvuloplasty', 'SuturetLeftAtrium', 'HemostasisPericardiumSuture']
TOOLS = ['Needle_holder', 'Aspirator', 'Endotherm_knife', 'Needle_holder_small',
         'Occlusion_forceps', 'Knife', 'Atrial_retractor', 'Scissor']

# Display names follow the MedHorizon M1 wording; the two phases that never occur in
# an original question use the same naming pattern.
PHASE_NAMES = dict(zip(PHASES, [
    'Preparation', 'Pericardial Suspension', 'Vein Dissection',
    'Perfusion Needle Spacer Suturing', 'Perfusion Needle Insertion', 'Aortic Clamping',
    'Left Heart Drain Placement', 'Left Atrium Dissection', 'Mitral Valve Exposure',
    'Mitral Valve Repair (Mitral Valvuloplasty)', 'Left Atrium Suturing',
    'Pericardial Hemostasis Suturing']))
TOOL_NAMES = dict(zip(TOOLS, [
    'Needle Holder', 'Aspirator', 'Endotherm Knife', 'Small Needle Holder',
    'Occlusion Forceps', 'Knife', 'Atrial Retractor', 'Scissor']))
PROTOCOL = 'AHA-M1-v2.2'

def qa_files(root, split, qa='original'):
    """(questions, gold) paths. Test gold of either kind lives under data/private."""
    root = Path(root)
    name = split if qa == 'original' else f'{split}_derived_v2'
    if qa not in ('original', 'derived'):
        raise ValueError(qa)
    gold = root/f'data/private/{name}_gold.json' if split == 'test' else root/f'data/targets/{name}.json'
    return root/f'data/questions/{name}.json', gold

def label_dir(root, split):
    return Path(root)/('data/private/test_labels' if split == 'test' else f'data/labels/{split}')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
