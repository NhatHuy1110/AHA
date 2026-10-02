"""Multimodal LLM comparators: zero-shot inference and supervised LoRA on the same train QA.

Models live under assets/vlm/<name>/ with source.json = {id, revision, family}
(scripts/fetch_encoder.py --kind vlm). Two label-free input protocols:
  uniform   : `budget` frames evenly spaced over the whole video (the MedHorizon protocol)
  retrieval : SurgFormer phase-change proposals, centres first then +/-4 s context
Every frame is shown with its timestamp; the model must return an answer and one of the
supplied timestamps. No timestamp snapping at inference.
"""
import json
import re
import tempfile
from pathlib import Path
import numpy as np
import torch
from .common import PHASES,read_json,write_json,seed_all,qa_files
from .proposals import select_times
from .frames import FrameStore
from .evaluate import score_rows,aggregate

# Longest image side per family. Hulu-Med's vision encoder attends over all frames' patches at
# once, which runs out of memory on a 24 GB GPU above ~10k patches (and its processor caps the
# prompt at 16,384 tokens); 224 px gives 144 visual tokens per frame (Qwen3-VL: about 110).
IMAGE_SIDE=dict(qwen3vl=448,hulumed=224)
INSTRUCTION=('\nChoose one answer and the onset frame. Return only JSON with keys '
             '"answer" (A, B, C, or D) and "second" (one of the supplied integer timestamps).')

def sampled_seconds(phase,phase_index,config,budget=48):
    times,centers,_=select_times(phase,phase_index,config['proposals'])
    # Centers first; then +/-4 second context. All chosen without supervision.
    ranked=list(centers)+[c+d for c in centers for d in (-4,4)]
    keep=[]; available=set(times.tolist())
    for t in ranked:
        if t in available and t not in keep:
            keep.append(t)
        if len(keep)>=budget:
            break
    return sorted(keep)

def frame_times(root,q,config,budget,frames):
    root=Path(root)
    if frames=='uniform':
        n=read_json(root/f'data/frames/{q["video_id"]}.json')['frames']
        return sorted(set(np.linspace(0,n-1,budget,dtype=int).tolist()))
    if frames=='retrieval':
        with np.load(root/f'artifacts/features/surgformer/{q["video_id"]}.npz') as f:
            return sampled_seconds(f['phase'],PHASES.index(q['target_phase']),config,budget)
    raise ValueError(frames)

def load_images(root,q,times,side=448):
    images=[]
    with FrameStore(Path(root)/'data/frames',q['video_id']) as store:
        for t in times:
            image=store.image(t); image.thumbnail((side,side)); images.append(image)
    return images

def prompt_text(q):
    return q['question']+'\n'+'\n'.join(q['options'])+INSTRUCTION

def messages_and_images(root,q,config,budget,frames='retrieval'):
    times=frame_times(root,q,config,budget,frames)
    content=[]
    for t in times:
        content.extend([dict(type='text',text=f'Frame at {t} seconds:'),dict(type='image')])
    content.append(dict(type='text',text=prompt_text(q)))
    return [dict(role='user',content=content)],load_images(root,q,times),times

def encode(processor,messages,images,device,target=None):
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[prefix],images=images,return_tensors='pt')
    if target is not None:
        full=messages+[dict(role='assistant',content=[dict(type='text',text=json.dumps(target))])]
        text=processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
        complete=processor(text=[text],images=images,return_tensors='pt')
        n=inputs['input_ids'].shape[1]
        if not torch.equal(complete['input_ids'][:,:n],inputs['input_ids']):
            raise ValueError('Assistant prefix mismatch; refusing incorrect SFT loss mask')
        labels=complete['input_ids'].clone(); labels[:,:n]=-100
        if not (labels!=-100).any():
            raise ValueError('No supervised response tokens')
        complete['labels']=labels; inputs=complete
    return inputs.to(device)

def parse(raw):
    """Lenient: a JSON object anywhere in the reply, else the first option letter / integer."""
    answer=second=None
    match=re.search(r'\{[^{}]*\}',raw,re.S)
    if match:
        try:
            parsed=json.loads(match.group(0)); answer=parsed.get('answer'); second=parsed.get('second')
        except (ValueError,AttributeError):
            pass
    if answer is None:
        m=re.search(r'"?answer"?\s*[:=]\s*"?([ABCD])\b',raw) or re.search(r'\b([ABCD])\b',raw)
        answer=m.group(1) if m else None
    if second is None:
        m=re.search(r'"?second"?\s*[:=]\s*"?(\d+)',raw)
        second=int(m.group(1)) if m else None
    if isinstance(answer,str):
        answer=answer.strip()[:1].upper()
    if isinstance(second,str) and second.strip().isdigit():
        second=int(second)
    return (answer if answer in ('A','B','C','D') else None,
            second if isinstance(second,int) and not isinstance(second,bool) else None)

def chunk_vision_encoder(encoder,images=8):
    """Hulu-Med's SDPA vision attention builds one dense mask over every frame's patches even though
    attention never crosses frames. Encoding a few frames at a time is equivalent and fits in memory."""
    inner=encoder.forward
    def forward(pixel_values,grid_sizes,merge_sizes=None):
        sizes=grid_sizes.prod(dim=1).tolist(); outputs=[]; start=0
        for i in range(0,len(sizes),images):
            n=sum(sizes[i:i+images])
            outputs.append(inner(pixel_values[start:start+n],grid_sizes[i:i+images],
                                 None if merge_sizes is None else merge_sizes[i:i+images]))
            start+=n
        return torch.cat(outputs,0)
    encoder.forward=forward

def load(root,name,device,adapter=None):
    source=Path(root)/'assets/vlm'/name
    origin=read_json(source/'source.json')
    if len(origin['revision'])!=40:
        raise ValueError('source.json needs an immutable 40-character model revision')
    if origin['family']=='qwen3vl':
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        processor=AutoProcessor.from_pretrained(source,local_files_only=True,min_pixels=128*32*32,max_pixels=256*32*32)
        model=Qwen3VLForConditionalGeneration.from_pretrained(source,local_files_only=True,
                  torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    elif origin['family']=='hulumed':
        from transformers import AutoModelForCausalLM,AutoProcessor
        processor=AutoProcessor.from_pretrained(source,local_files_only=True,trust_remote_code=True)
        model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=True,
                  torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
        chunk_vision_encoder(model.get_model().get_vision_encoder())
    else:
        raise ValueError(origin['family'])
    if adapter:
        from peft import PeftModel
        spec=read_json(Path(adapter)/'aha_adapter.json')
        if spec['model']!=origin:
            raise ValueError('Adapter/base model mismatch')
        model=PeftModel.from_pretrained(model,adapter)
    return model,processor,origin

def generate(model,processor,origin,root,q,config,device,budget,frames):
    if origin['family']=='qwen3vl':
        messages,images,times=messages_and_images(root,q,config,budget,frames)
        inputs=encode(processor,messages,images,device)
        output=model.generate(**inputs,max_new_tokens=64,do_sample=False)
        return processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True),times
    # Hulu-Med: its processor takes an interleaved conversation with image paths.
    times=frame_times(root,q,config,budget,frames)
    with tempfile.TemporaryDirectory() as tmp:
        content=[]
        for t,image in zip(times,load_images(root,q,times,IMAGE_SIDE['hulumed'])):
            path=str(Path(tmp)/f'{t}.jpg'); image.save(path,quality=95)
            content.extend([dict(type='text',text=f'Frame at {t} seconds:'),dict(type='image',image=dict(image_path=path))])
        content.append(dict(type='text',text=prompt_text(q)))
        inputs=processor(conversation=[dict(role='user',content=content)],add_system_prompt=True,
                         add_generation_prompt=True,return_tensors='pt')
    inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
    if 'pixel_values' in inputs:
        inputs['pixel_values']=inputs['pixel_values'].to(torch.bfloat16)
    output=model.generate(**inputs,max_new_tokens=64,do_sample=False)
    return processor.batch_decode(output,skip_special_tokens=True,use_think=False)[0].strip(),times

@torch.inference_mode()
def infer_rows(model,processor,origin,root,questions,config,device,budget,frames):
    model.eval(); result=[]
    for q in questions:
        raw,times=generate(model,processor,origin,root,q,config,device,budget,frames)
        answer,second=parse(raw)
        result.append(dict(qid=q['qid'],video_id=q['video_id'],answer=answer,second=second,
                           observed_seconds=times,candidate_seconds=times,raw=raw))
        print(f'vlm {q["qid"]} {answer} {second}',flush=True)
    return result

def infer(root,config_path,model,split,out,device='cuda',budget=48,frames='retrieval',qa='original',adapter=None):
    config=read_json(config_path)
    net,processor,origin=load(root,model,device,adapter)
    qs=read_json(qa_files(root,split,qa)[0])
    if adapter:
        spec=read_json(Path(adapter)/'aha_adapter.json')
        if spec['proposals']!=config['proposals'] or spec['budget']!=budget or spec['frames']!=frames:
            raise ValueError('Adapter preprocessing differs')
    result=infer_rows(net,processor,origin,root,qs,config,device,budget,frames)
    write_json(out,result)
    write_json(Path(out).with_suffix('.meta.json'),dict(model=origin,adapter=str(adapter) if adapter else None,
               budget=budget,frames=frames,split=split,qa=qa,proposals=config['proposals'],
               image_side=IMAGE_SIDE[origin['family']]))
    return dict(predictions=len(result),unparsed=sum(r['answer'] is None or r['second'] is None for r in result))

def sft(root,config_path,model,out,device='cuda',seed=17,budget=48,frames='retrieval',epochs=3):
    from peft import LoraConfig,get_peft_model
    root,out=Path(root),Path(out); config=read_json(config_path); seed_all(seed)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty SFT output directory')
    out.mkdir(parents=True,exist_ok=True)
    net,processor,origin=load(root,model,device)
    if origin['family']!='qwen3vl':
        raise ValueError('LoRA SFT is implemented for the Qwen3-VL family only')
    net=get_peft_model(net,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
          target_modules=r'.*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)',
          task_type='CAUSAL_LM',bias='none'))
    net.config.use_cache=False
    net.gradient_checkpointing_enable(); net.enable_input_require_grads()
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=2e-5)
    qs=[]; gs=[]
    for kind in ('original','derived') if config['train']['use_derived'] else ('original',):
        qp,gp=qa_files(root,'train',kind); qs+=read_json(qp); gs+=read_json(gp)
    gold={g['qid']:g for g in gs if g['primary_eligible'] and g['supported_onsets']}
    vals={kind:(read_json(qa_files(root,'val',kind)[0]),read_json(qa_files(root,'val',kind)[1])) for kind in ('original','derived')}
    usable=[]; skipped=[]
    # No supervision is used to choose input images; only to validate SFT target support.
    for q in qs:
        if q['qid'] not in gold:
            continue
        g=gold[q['qid']]
        ts=frame_times(root,q,config,budget,frames)
        t=min(ts,key=lambda t:min(abs(t-o) for o in g['supported_onsets']))
        if min(abs(t-o) for o in g['supported_onsets'])>5:
            skipped.append(q['qid']); continue
        usable.append((q,dict(answer=g['answer'],second=int(t))))
    if not usable:
        raise ValueError('No supported SFT examples under frame budget')
    provenance=dict(model=origin,budget=budget,frames=frames,proposals=config['proposals'],seed=seed,
                    train_examples=len(usable),skipped_retrieval_misses=skipped,
                    target_modules='language_model self_attn q/k/v/o projections',lr=2e-5,epochs=epochs,
                    selection='mean of val supported_joint@5 on original primary QA and on derived QA')
    write_json(out/'sft_data_report.json',provenance)
    rng=np.random.default_rng(seed); best=-1; history=[]
    for epoch in range(epochs):
        net.train(); optimizer.zero_grad(set_to_none=True); total=0
        for i,index in enumerate(rng.permutation(len(usable))):
            q,target=usable[index]
            messages,images,_=messages_and_images(root,q,config,budget,frames)
            inputs=encode(processor,messages,images,device,target)
            loss=net(**inputs).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(q['qid'])
            group=min(4,len(usable)-(i//4)*4)
            (loss/group).backward(); total+=float(loss.detach())
            if (i+1)%4==0 or i+1==len(usable):
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if i%20==0:
                print(f'sft epoch={epoch+1} step={i+1}/{len(usable)} loss={total/(i+1):.4f}',flush=True)
        metrics={}; preds={}
        for kind,(vq,vg) in vals.items():
            preds[kind]=infer_rows(net,processor,origin,root,vq,config,device,budget,frames)
            metrics[kind]=aggregate([r for r in score_rows(root,vq,vg,preds[kind]) if r['primary']])
        score=float(np.mean([m['supported_joint@5'] for m in metrics.values()]))
        history.append(dict(epoch=epoch,loss=total/len(usable),selection=score,val=metrics))
        if score>best:
            best=score; net.save_pretrained(out/'best')
            write_json(out/'best/aha_adapter.json',provenance)
            write_json(out/'best_val_predictions.json',preds)
        write_json(out/'history.json',history)
        print(f'sft epoch={epoch+1} loss={total/len(usable):.4f} selection={score:.4f} best={best:.4f}',flush=True)
    return dict(best_val_selection=best,examples=len(usable),skipped=len(skipped))
