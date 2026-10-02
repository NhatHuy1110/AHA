"""SurgFormer loader adapted from the local IPL implementation.

The official code (github.com/xmed-lab/SurgFormer, pinned commit below) is
vendored under AHA/vendor; ``repo`` points at that snapshot. The compiled MultiScaleDeformableAttention
CUDA op is optional: without it we substitute the repository's own pure-PyTorch
reference ``ms_deform_attn_core_pytorch`` (same math, no compilation).
Inference follows the official ``inference_offline.py``: contiguous clips of 64
frames at 1 FPS, Resize((320, 320)), ImageNet normalisation, sigmoid scores.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np

from .common import sha

PINNED_COMMIT = '73c0a931f6caf8eecf82aea1ec6803a2f59b57c8'
CLIP = 64  # official offline inference clip length (im2col_step limit)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _patch_imports(repo):
    import torch
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import MultiScaleDeformableAttention as msda
        compiled = not getattr(msda, '_ipl_stub', False)
    except ImportError:
        compiled = False
        stub = types.ModuleType('MultiScaleDeformableAttention')
        stub._ipl_stub = True
        sys.modules['MultiScaleDeformableAttention'] = stub
    import models.ops.functions.ms_deform_attn_func as func
    import models.ops.modules.ms_deform_attn as module
    if not compiled:
        class TorchMSDeformAttn:
            @staticmethod
            def apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
                return func.ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)
        module.MSDeformAttnFunction = TorchMSDeformAttn
    import models.backbone as backbone
    import torchvision

    class _Models:
        def __getattr__(self, name):
            constructor = getattr(torchvision.models, name)

            def build(**kwargs):
                kwargs.pop('pretrained', None)  # weights come from the checkpoint; never download
                return constructor(weights=None, **kwargs)
            return build
    backbone.torchvision = types.SimpleNamespace(models=_Models())
    return compiled


def load(repo, checkpoint, device='cuda'):
    import torch
    compiled = _patch_imports(repo)
    from models import build_model
    import models.SurgFormer as surgformer_module
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)  # local trusted file (argparse args inside)
    args = state['args']
    args.device = device
    if args.dataset_file != 'mvp' or args.num_queries != 4:
        raise ValueError('Checkpoint is not the 4-task TMVP SurgFormer')
    # Official __init__ references an undefined global ``num_pred`` (one shared head
    # per decoder layer, as in Deformable DETR); define it from the checkpoint args.
    surgformer_module.num_pred = args.dec_layers + (1 if args.two_stage else 0)
    model, _ = build_model(args)
    missing, unexpected = model.load_state_dict(state['model'], strict=False)
    unexpected = [k for k in unexpected if not k.endswith(('total_params', 'total_ops'))]
    if missing or unexpected:
        raise ValueError(f'Checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    model.to(device).eval()
    info = {'epoch': int(state.get('epoch', -1)), 'output_dir': getattr(args, 'output_dir', ''),
            'sequence_length': args.sequence_length, 'compiled_msda': compiled,
            'checkpoint_sha256': sha(checkpoint)}
    return model, info


def _tensor(images, device):
    import torch
    x = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in images])  # T,H,W,3
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)

