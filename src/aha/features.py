"""Frozen feature extraction: label-free image encoder (main input) and SurgFormer (comparators)."""
from pathlib import Path
import numpy as np
from .common import read_json, write_json, atomic_npz, sha, fingerprint, qa_files
from PIL import Image
from .frames import FrameStore

COMMIT = '73c0a931f6caf8eecf82aea1ec6803a2f59b57c8'
TEXT_ID = 'sentence-transformers/all-MiniLM-L6-v2'
TEXT_REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'

def _run(root, name, spec, encode, extra, ids, limit, review_dir, device):
    """Shared resumable per-video cache writer. encode(store, seconds) -> dict of [n,*] arrays."""
    import torch
    splits = read_json(root/'data/manifests/splits.json')
    ids = ids or sorted({v for vs in splits.values() for v in vs})
    if review_dir is not None and limit is None:
        raise ValueError('--review-dir is only for a limited smoke extraction')
    outdir = Path(review_dir) if review_dir is not None else (
        root/'artifacts/features'/name if limit is None else root/'review/partial_features'/name)
    outdir.mkdir(parents=True,exist_ok=True)
    existing = outdir/'spec.json'
    if existing.exists() and read_json(existing)!=spec:
        raise ValueError('Feature directory has different provenance')
    write_json(existing,spec)
    for vid in ids:
        path = outdir/f'{vid}.npz'
        fm = read_json(root/f'data/frames/{vid}.json')
        signature = fingerprint(dict(spec=spec,frames=fm['zip_sha256'],limit=limit))
        if path.exists() and path.with_suffix('.json').exists():
            meta = read_json(path.with_suffix('.json'))
            if meta['signature'] != signature or meta['sha256'] != sha(path):
                raise ValueError(f'Stale features {vid}')
            continue
        arrays = {}
        with FrameStore(root/'data/frames',vid,verify=True) as store:
            n = min(store.count,limit) if limit is not None else store.count
            for start in range(0,n,spec['clip']):
                ts = range(start,min(start+spec['clip'],n))
                for key,value in encode(store,ts).items():
                    arrays.setdefault(key,[]).append(value)
                print(f'features {name}/{vid}: {ts[-1]+1}/{n}',flush=True)
        arrays = {k:np.concatenate(v) for k,v in arrays.items()}
        assert all(len(v)==n for v in arrays.values())
        atomic_npz(path,**arrays,times=np.arange(n,dtype=np.int32))
        write_json(path.with_suffix('.json'),dict(signature=signature,sha256=sha(path),
                   frames=n,complete=limit is None,hidden_dim=arrays['hidden'].shape[1],
                   execution_device=device,torch_version=torch.__version__,**extra))

def extract(root, device='cuda', ids=None, limit=None, review_dir=None):
    """SurgFormer recognition queries + scores. Used by the index/VLM comparators and by
    the seen-backbone ablation; NOT the main model input (it was trained on our train cases)."""
    import torch
    from .surgformer import load, _tensor
    root = Path(root)
    repo = root/f'vendor/SurgFormer-{COMMIT}'
    model, info = load(repo, root/'assets/surgformer_checkpoint.pth', device)
    for p in model.parameters():
        p.requires_grad_(False)
    captured = {}
    def hook(name):
        def capture(_module, inputs):
            captured[name] = inputs[0].detach()
        return capture
    handles = [model.phase_embed[-1].register_forward_pre_hook(hook('phase')),
               model.tool_embed[-1].register_forward_pre_hook(hook('tool'))]
    spec = dict(schema=2,backbone='SurgFormer',commit=COMMIT,clip=64,
                hidden='concat final phase and tool recognition query, before classifiers',
                transform='RGB; bilinear 320x320; ImageNet normalization',**info)
    def encode(store,ts):
        x = _tensor([store.surgformer_input(t) for t in ts],device)
        with torch.inference_mode():
            result = model([x])
        hidden = torch.cat([captured['phase'],captured['tool']],-1)
        assert hidden.ndim==2 and hidden.shape[0]==len(ts)
        out = dict(hidden=hidden.cpu().numpy().astype(np.float16))
        for key in ('phase','tool'):
            out[key] = result[f'pred_{key}'][0].sigmoid().float().cpu().numpy()
        return out
    try:
        _run(root,'surgformer',spec,encode,{},ids,limit,review_dir,device)
    finally:
        for h in handles:
            h.remove()

def encoder(root, name, device='cuda', ids=None, limit=None, review_dir=None, batch=64, height=256, width=448):
    """Frozen, label-free image encoder (e.g. DINOv3/DINOv2) applied to every second.

    Weights are read offline from assets/encoders/<name>/, which must also hold
    source.json = {"id": ..., "revision": <40-char commit>} written at download time.
    hidden[t] = concat(CLS token, mean patch token). No TMVP label ever touched these weights,
    so train and held-out cases have the same feature distribution.
    """
    import torch
    from transformers import AutoModel
    from .surgformer import MEAN, STD
    root = Path(root); source = root/'assets/encoders'/name
    origin = read_json(source/'source.json')
    if len(origin['revision'])!=40:
        raise ValueError('source.json needs an immutable 40-character model revision')
    model = AutoModel.from_pretrained(source,local_files_only=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    patch = model.config.patch_size
    size = (round(height/patch)*patch, round(width/patch)*patch)
    skip = 1 + getattr(model.config,'num_register_tokens',0)
    mean = torch.tensor(MEAN,device=device).view(1,3,1,1); std = torch.tensor(STD,device=device).view(1,3,1,1)
    half = device.startswith('cuda')
    spec = dict(schema=2,backbone=origin['id'],revision=origin['revision'],clip=batch,
                hidden='concat CLS token and mean patch token of the last layer',
                transform=f'RGB; bilinear {size[0]}x{size[1]} (HxW); ImageNet normalization',
                autocast_fp16=half,weights={p.name:sha(p) for p in sorted(source.iterdir()) if p.is_file()})
    def encode(store,ts):
        frames = np.stack([np.asarray(store.image(t).resize((size[1],size[0]),Image.Resampling.BILINEAR)) for t in ts])
        x = torch.from_numpy(frames).to(device).permute(0,3,1,2).float().div(255)
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16,enabled=half):
            tokens = model(pixel_values=(x-mean)/std).last_hidden_state
        hidden = torch.cat([tokens[:,0],tokens[:,skip:].mean(1)],-1).float()
        if not torch.isfinite(hidden).all():
            raise FloatingPointError('Nonfinite encoder features')
        return dict(hidden=hidden.cpu().numpy().astype(np.float16))
    _run(root,name,spec,encode,dict(encoder=origin['id']),ids,limit,review_dir,device)

def text(root, device='cpu', model_path=None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    root = Path(root)
    source = str(model_path or root/'assets/text_encoder')
    tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=True)
    encoder = AutoModel.from_pretrained(source,local_files_only=True).to(device).eval()
    rows=[]
    for split in ('train','val','test'):
        for kind in ('original','derived'):
            rows += read_json(qa_files(root,split,kind)[0])
    texts = sorted({q['question'] for q in rows} | {o[3:] for q in rows for o in q['options']})
    values=[]
    for start in range(0,len(texts),64):
        inputs = tokenizer(texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
        with torch.inference_mode():
            tokens=encoder(**inputs).last_hidden_state
            mask=inputs.attention_mask.unsqueeze(-1)
            v=F.normalize((tokens*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
        values.append(v.cpu().numpy())
    path=root/'artifacts/text/embeddings.npz'
    atomic_npz(path,embeddings=np.concatenate(values))
    write_json(path.with_suffix('.json'),dict(texts=texts,model=TEXT_ID,revision=TEXT_REVISION,
               dim=int(values[0].shape[1]),sha256=sha(path),
               encoder_files={p.name:sha(p) for p in Path(source).iterdir() if p.is_file()}))
