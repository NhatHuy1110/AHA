import datetime
import importlib.util
import platform
import zipfile
from pathlib import Path
from .common import read_json,write_json,sha,qa_files,PROTOCOL

def audit(root,full=False):
    root=Path(root); splits=read_json(root/'data/manifests/splits.json')
    sets=[set(v) for v in splits.values()]
    assert not any(a&b for i,a in enumerate(sets) for b in sets[i+1:]),'Split leakage'
    result=dict(protocol=PROTOCOL,python=platform.python_version(),full_hash_check=full,splits={})
    for split,ids in splits.items():
        result['splits'][split]=dict(cases=len(ids))
        for kind in ('original','derived'):
            qp,gp=qa_files(root,split,kind)
            q=read_json(qp); g=read_json(gp)
            assert len({r['qid'] for r in q})==len(q)
            assert all(r['video_id'] in ids for r in q)
            assert all(not ({'answer','onsets','supported_onsets','per_onset'}&set(r)) for r in q)
            assert {r['qid'] for r in q}=={r['qid'] for r in g}
            result['splits'][split][kind]=dict(qa=len(q),primary=sum(r['primary_eligible'] for r in g))
        for vid in ids:
            meta=read_json(root/f'data/frames/{vid}.json')
            with zipfile.ZipFile(root/f'data/frames/{vid}.zip') as z:
                names=z.namelist()
                assert len(names)==meta['frames']
                assert all(f'f/{t:06d}.jpg' in z.NameToInfo for t in (0,meta['frames']-1))
    assets=read_json(root/'data/manifests/assets.json')
    for a in assets:
        p=root/a['path']
        assert p.is_file() and p.stat().st_size==a['bytes'],str(p)
        if full:
            assert sha(p)==a['sha256'],f'Hash mismatch: {p}'
    result['assets']=len(assets)
    feature_root=root/'artifacts/features'
    result['feature_caches']={d.name:sum((d/f'{v}.json').exists() for ids in splits.values() for v in ids)
                              for d in sorted(feature_root.iterdir()) if d.is_dir()} if feature_root.exists() else {}
    result['encoders_staged']=sorted(d.name for d in (root/'assets/encoders').iterdir() if (d/'source.json').exists()) if (root/'assets/encoders').exists() else []
    result['text_embeddings_ready']=(root/'artifacts/text/embeddings.npz').exists()
    result['dependencies']={m:importlib.util.find_spec(m) is not None for m in ('torch','torchvision','transformers','einops','scipy','peft')}
    import torch
    result['torch']=torch.__version__; result['cuda']=torch.cuda.is_available()
    write_json(root/'review/preflight.json',result)
    return result

def freeze(root):
    root=Path(root)
    files={}
    for directory in ('src','configs','scripts','tests','data','assets','references','vendor'):
        for p in sorted((root/directory).rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ('__pycache__','.cache','encoders','vlm')):
                continue
            rel=p.relative_to(root).as_posix()
            files[rel]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for p in sorted(root.glob('*')):
        if p.is_file() and p.name!='FREEZE.json':
            files[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    write_json(root/'FREEZE.json',dict(protocol=PROTOCOL,date=datetime.date.today().isoformat(),files=files,
        excludes=['artifacts (generated caches)','assets/encoders and assets/vlm (downloaded, pinned by revision)','runs','review (test reports)','.deps','Python caches'],
        note='Research direction and initial implementation freeze; amendments require a new version before test use.'))
    return dict(files=len(files))

def verify(root):
    root=Path(root); frozen=read_json(root/'FREEZE.json')
    for rel,expected in frozen['files'].items():
        p=root/rel
        if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:
            raise ValueError(f'Frozen artifact missing/changed: {rel}')
    return dict(verified=len(frozen['files']))
