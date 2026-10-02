import math
import os
import platform
from pathlib import Path
import numpy as np
import torch
from .common import read_json,write_json,seed_all,sha,fingerprint,qa_files
from .data import Study,to_device
from .model import AnswerOnsetDecoder
from .losses import qa_loss,dense_loss
from .evaluate import score_rows,aggregate

SELECTION='mean of val supported_joint@5 on original primary QA and on derived QA; earliest tie'

def checkpoint(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    torch.save(payload,tmp); os.replace(tmp,path)

def data_signature(root,config):
    root=Path(root); fdir=root/'artifacts/features'/config['features']['name']
    paths=[root/'data/manifests/splits.json',root/'artifacts/text/embeddings.json',fdir/'spec.json']
    for split in ('train','val'):
        for kind in ('original','derived'):
            paths.extend(qa_files(root,split,kind))
    splits=read_json(root/'data/manifests/splits.json')
    for vid in splits['train']+splits['val']:
        paths.append(fdir/f'{vid}.json')
    for vid in splits['train']:
        paths.append(root/f'data/labels/train/{vid}.npz')
    return fingerprint(dict(config=config,files={p.relative_to(root).as_posix():sha(p) for p in paths},
                            source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))

@torch.inference_mode()
def predictions(model,study,device,diagnostic=None,decode='joint'):
    """decode='joint': argmax over (second, option). 'cascade': localize first, then answer there."""
    model.eval(); rows=[]
    for vid in study.ids:
        if not study.by_video[vid]:
            continue
        x,_,meta=study.video(vid,diagnostic=diagnostic)
        out=model(**to_device(x,device))
        for i,(q,order) in enumerate(zip(meta['questions'],meta['orders'])):
            s=out['logits'][i]
            if decode=='joint':
                index=int(s.argmax()); ti,ai=divmod(index,s.shape[1])
            elif decode=='cascade':
                ti=int(out['onset_evidence'][i].argmax()); ai=int(out['tool_evidence_options'][i,ti].argmax())
            else:
                raise ValueError(decode)
            rows.append(dict(qid=q['qid'],video_id=vid,answer=chr(65+int(order[ai])),second=ti,
                             observed='dense',diagnostic=diagnostic,decode=decode,
                             confidence=float(s.flatten().softmax(0)[ti*s.shape[1]+ai])))
    return rows

def phase_weights(study,device):
    counts=np.zeros(12)
    for vid in study.ids:
        p=study.labels(vid)['phase']; counts+=np.bincount(p[p>=0],minlength=12)
    w=1/np.sqrt(np.maximum(counts,1)); w[counts==0]=0
    return torch.tensor(w/w[counts>0].mean(),dtype=torch.float32,device=device)

def learning_rate(step,total,config):
    warm=config['warmup_steps']
    if step<warm:
        return config['lr']*(step+1)/warm
    return config['lr']*(0.05+0.95*0.5*(1+math.cos(math.pi*(step-warm)/max(total-warm,1))))

def validate(model,root,studies,device):
    scores={}; preds={}
    for kind,study in studies.items():
        preds[kind]=predictions(model,study,device)
        rows=score_rows(root,study.questions,read_json(qa_files(root,'val',kind)[1]),preds[kind])
        scores[kind]=aggregate([r for r in rows if r['primary']])
    selection=float(np.mean([scores[k]['supported_joint@5'] for k in scores]))
    return selection,scores,preds

def train(root,config_path,out,seed=17,device='cuda',resume=False):
    root,out=Path(root),Path(out)
    config=read_json(config_path); seed_all(seed); tc=config['train']
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError('Run directory is nonempty: choose a new directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    signature=data_signature(root,config)
    study=Study(root,config,'train',training=True)
    val_studies={kind:Study(root,config,'val',qa=kind) for kind in ('original','derived')}
    model=AnswerOnsetDecoder(config['model']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    rng=np.random.default_rng(seed)
    start=0; best=-1.; stale=0; history=[]
    if resume:
        state=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if state['signature']!=signature or state['seed']!=seed:
            raise ValueError('Cannot resume with changed data/code/config/seed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start=state['epoch']+1; best=state['best']; stale=state['stale']; history=state['history']
        rng.bit_generator.state=state['numpy_rng']; torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    else:
        write_json(out/'config.json',config)
        write_json(out/'run.json',dict(signature=signature,seed=seed,python=platform.python_version(),
                   torch=torch.__version__,device=device,trainable_parameters=sum(p.numel() for p in model.parameters()),
                   training_qa=len(study.questions),selection=SELECTION))
    pw=phase_weights(study,device)
    # One optimizer step per training video: equal case mass, all of its questions together.
    cases=[v for v in study.ids if study.by_video[v]]
    total_steps=tc['epochs']*len(cases)
    for epoch in range(start,tc['epochs']):
        if stale>=tc['patience']:
            break
        model.train(); totals={}
        for i,ci in enumerate(rng.permutation(len(cases))):
            vid=cases[ci]; qs=study.by_video[vid]
            if len(qs)>tc['max_questions']:
                qs=[qs[j] for j in sorted(rng.choice(len(qs),tc['max_questions'],replace=False))]
            x,y,_=study.video(vid,qs,rng); x,y=to_device(x,device),to_device(y,device)
            for group in optimizer.param_groups:
                group['lr']=learning_rate(epoch*len(cases)+i,total_steps,tc)
            optimizer.zero_grad(set_to_none=True)
            result=model(**x)
            lq,stats=qa_loss(result,y,config)
            ld,terms=dense_loss(model,result,y,config['loss'],pw)
            loss=lq+ld
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at video {vid}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),tc['clip_grad'])
            optimizer.step()
            for k,v in dict(loss=loss,qa=lq,**terms).items():
                totals[k]=totals.get(k,0)+float(v.detach())/len(cases)
        score,val,pred=validate(model,root,val_studies,device)
        improved=score>best
        if improved:
            best=score; stale=0
        else:
            stale+=1
        history.append(dict(epoch=epoch,train=totals,selection=score,val=val))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,signature=signature,
                   seed=seed,epoch=epoch,best=best,stale=stale,history=history,numpy_rng=rng.bit_generator.state,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        checkpoint(out/'last.pt',state)
        if improved:
            checkpoint(out/'best.pt',state); write_json(out/'best_val_predictions.json',pred)
        write_json(out/'history.json',history)
        print(f'epoch={epoch+1} loss={totals["loss"]:.4f} val_original={val["original"]["supported_joint@5"]:.4f} '
              f'val_derived={val["derived"]["supported_joint@5"]:.4f} selection={score:.4f} best={best:.4f}',flush=True)
    return dict(best_val_selection=best,epochs=len(history),checkpoint=str(out/'best.pt'))

def predict(root,checkpoint_path,split,out,device='cuda',diagnostic=None,qa='original',decode='joint'):
    state=torch.load(checkpoint_path,map_location=device,weights_only=False)
    if data_signature(root,state['config'])!=state['signature']:
        raise ValueError('Checkpoint data/code provenance no longer matches this checkout')
    model=AnswerOnsetDecoder(state['config']['model']).to(device)
    model.load_state_dict(state['model'])
    study=Study(root,state['config'],split,qa=qa)
    result=predictions(model,study,device,diagnostic,decode)
    write_json(out,result)
    write_json(Path(out).with_suffix('.meta.json'),dict(checkpoint_sha256=sha(checkpoint_path),split=split,qa=qa,
               decode=decode,signature=state['signature'],seed=state['seed'],diagnostic=diagnostic))
    return dict(predictions=len(result))
