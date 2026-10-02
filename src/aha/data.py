from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .common import PHASES, TOOLS, read_json, sha, fingerprint, qa_files

def smooth_onsets(onset, sigma):
    """Gaussian-smoothed per-phase onset targets with peak 1 at each annotated onset."""
    radius = int(3*sigma)
    kernel = np.exp(-np.arange(-radius, radius+1)**2/(2*sigma**2)).astype(np.float32)
    out = np.zeros_like(onset, dtype=np.float32)
    for t, p in zip(*np.nonzero(onset)):
        lo, hi = max(0, t-radius), min(len(onset), t+radius+1)
        out[lo:hi, p] = np.maximum(out[lo:hi, p], kernel[lo-t+radius:hi-t+radius])
    return out

class Study:
    """One split, one QA set. Only a training study can read labels or gold."""
    def __init__(self,root,config,split,training=False,qa='original'):
        if training and split!='train':
            raise ValueError('Training may only open train labels')
        self.root,self.config,self.split,self.training=Path(root),config,split,training
        self.ids=read_json(self.root/'data/manifests/splits.json')[split]
        self.feature_dir=self.root/'artifacts/features'/config['features']['name']
        self.feature_spec=read_json(self.feature_dir/'spec.json')
        self.gold={}
        if training:
            self.questions=[]; gs=[]
            for kind in ('original','derived') if config['train']['use_derived'] else ('original',):
                qp,gp=qa_files(self.root,'train',kind)
                self.questions+=read_json(qp); gs+=read_json(gp)
            self.gold={g['qid']:g for g in gs if g['primary_eligible'] and g['supported_onsets']}
            self.questions=[q for q in self.questions if q['qid'] in self.gold]
        else:
            self.questions=read_json(qa_files(self.root,split,qa)[0])
        assert all(q['video_id'] in self.ids for q in self.questions)
        self.by_video={v:[q for q in self.questions if q['video_id']==v] for v in self.ids}
        text_path=self.root/'artifacts/text/embeddings.npz'
        tm=read_json(text_path.with_suffix('.json'))
        if sha(text_path)!=tm['sha256']:
            raise ValueError('Text features changed')
        with np.load(text_path) as z:
            emb=z['embeddings']
        self.text=dict(zip(tm['texts'],emb))

    @lru_cache(maxsize=64)
    def features(self,vid):
        if vid not in self.ids:
            raise ValueError('Video outside selected split')
        path=self.feature_dir/f'{vid}.npz'
        meta=read_json(path.with_suffix('.json'))
        fm=read_json(self.root/f'data/frames/{vid}.json')
        expected=fingerprint(dict(spec=self.feature_spec,frames=fm['zip_sha256'],limit=None))
        if meta.get('signature')!=expected:
            raise ValueError(f'Feature provenance mismatch: {vid}')
        if not meta['complete'] or meta['frames']!=fm['frames'] or sha(path)!=meta['sha256']:
            raise ValueError(f'Incomplete/corrupt features: {vid}')
        with np.load(path) as z:
            hidden=z['hidden']
        assert hidden.shape==(fm['frames'],self.config['model']['visual_dim'])
        return hidden

    @lru_cache(maxsize=64)
    def labels(self,vid):
        if not self.training or vid not in self.ids:
            raise ValueError('Inference dataset cannot read supervision')
        with np.load(self.root/f'data/labels/train/{vid}.npz') as z:
            lab={k:z[k] for k in z.files}
        lab['onset_soft']=smooth_onsets(lab['onset'],self.config['train']['onset_sigma'])
        return lab

    def video(self,vid,questions=None,rng=None,diagnostic=None):
        """All tensors for one video and a list of its questions."""
        qs=self.by_video[vid] if questions is None else questions
        hidden=np.asarray(self.features(vid),np.float32)
        n=len(hidden)
        if diagnostic=='zero_visual':
            hidden=np.zeros_like(hidden)
        elif diagnostic=='shuffle_visual':
            hidden=hidden[np.random.default_rng(0).permutation(n)]
        elif diagnostic is not None:
            raise ValueError(diagnostic)
        orders=[]
        for q in qs:
            order=np.arange(len(q['options']))
            if self.training and rng is not None and self.config['train']['permute_options']:
                rng.shuffle(order)
            orders.append(order)
        tensors=dict(x=torch.from_numpy(hidden),
                     position=torch.linspace(0,1,n).unsqueeze(-1),
                     question=torch.from_numpy(np.stack([self.text[q['question']] for q in qs])),
                     options=torch.from_numpy(np.stack([[self.text[q['options'][i][3:]] for i in o] for q,o in zip(qs,orders)])))
        target=None
        if self.training:
            target={k:torch.from_numpy(v) for k,v in self.labels(vid).items()}
            gold=[self.gold[q['qid']] for q in qs]
            target.update(
                answer=[int(np.flatnonzero(o==ord(g['answer'])-65)[0]) for g,o in zip(gold,orders)],
                onsets=[g['supported_onsets'] for g in gold],all_onsets=[g['onsets'] for g in gold],
                question_phase=torch.tensor([PHASES.index(q['target_phase']) for q in qs]),
                option_tool=torch.tensor([[TOOLS.index(q['option_tools'][i]) for i in o] for q,o in zip(qs,orders)]))
        return tensors,target,dict(questions=qs,orders=orders,frames=n)

def to_device(tensors,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in tensors.items()}
