"""Optional Qwen3-VL multi-image timestamp baseline: zero-shot and supervised LoRA.

The inputs always use label-free proposals. No timestamp snapping at inference.
Requires server extras and an explicit immutable Hugging Face revision.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .common import PHASES,read_json,write_json,seed_all,sha,qa_files
from .proposals import select_times
from .frames import FrameStore
from .evaluate import score_rows,aggregate

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

def messages_and_images(root,q,config,budget):
    root=Path(root)
    with np.load(root/f'artifacts/features/surgformer/{q["video_id"]}.npz') as f:
        times=sampled_seconds(f['phase'],PHASES.index(q['target_phase']),config,budget)
    content=[]; images=[]
    with FrameStore(root/'data/frames',q['video_id']) as frames:
        for t in times:
            content.extend([dict(type='text',text=f'Frame at {t} seconds:'),dict(type='image')])
            image=frames.image(t); image.thumbnail((448,448)); images.append(image)
    prompt=(q['question']+'\n'+'\n'.join(q['options'])+
            '\nChoose one answer and the onset frame. Return only JSON with keys '
            '"answer" (A, B, C, or D) and "second" (one of the supplied integer timestamps).')
    content.append(dict(type='text',text=prompt))
    return [dict(role='user',content=content)],images,times

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

def load(root,revision,device,adapter=None):
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Supply immutable 40-character Qwen model commit SHA')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    source=Path(root)/'assets/vlm'/revision
    processor=AutoProcessor.from_pretrained(source,local_files_only=True,min_pixels=128*32*32,max_pixels=256*32*32)
    model=Qwen3VLForConditionalGeneration.from_pretrained(source,local_files_only=True,
              torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    if adapter:
        from peft import PeftModel
        spec=read_json(Path(adapter)/'aha_adapter.json')
        if spec['revision']!=revision:
            raise ValueError('Adapter/base revision mismatch')
        model=PeftModel.from_pretrained(model,adapter)
    return model,processor

@torch.inference_mode()
def infer_rows(model,processor,root,questions,config,device,budget):
    model.eval(); result=[]
    for q in questions:
        messages,images,times=messages_and_images(root,q,config,budget)
        inputs=encode(processor,messages,images,device)
        output=model.generate(**inputs,max_new_tokens=64,do_sample=False)
        raw=processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        try:
            parsed=json.loads(raw)
            answer=parsed.get('answer'); second=parsed.get('second')
        except (ValueError,AttributeError):
            answer=second=None
        result.append(dict(qid=q['qid'],video_id=q['video_id'],answer=answer,second=second,
                           observed_seconds=times,candidate_seconds=times,raw=raw))
        print(f'vlm {q["qid"]}',flush=True)
    return result

def infer(root,config_path,revision,split,out,device='cuda',budget=48,adapter=None):
    config=read_json(config_path)
    model,processor=load(root,revision,device,adapter)
    qs=read_json(qa_files(root,split)[0])
    if adapter:
        spec=read_json(Path(adapter)/'aha_adapter.json')
        if spec['config']!=config or spec['budget']!=budget:
            raise ValueError('Adapter preprocessing differs')
    result=infer_rows(model,processor,root,qs,config,device,budget)
    write_json(out,result)
    write_json(Path(out).with_suffix('.meta.json'),dict(model='Qwen/Qwen3-VL-4B-Instruct',revision=revision,
        adapter=str(adapter) if adapter else None,budget=budget,config=config))
    return dict(predictions=len(result))

def sft(root,config_path,revision,out,device='cuda',seed=17,budget=48,epochs=3):
    from peft import LoraConfig,get_peft_model
    root,out=Path(root),Path(out); config=read_json(config_path); seed_all(seed)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty SFT output directory')
    out.mkdir(parents=True,exist_ok=True)
    model,processor=load(root,revision,device)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
          target_modules=r'.*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)',
          task_type='CAUSAL_LM',bias='none'))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    qs=read_json(root/'data/questions/train.json'); gs=read_json(root/'data/targets/train.json')
    if config['train']['use_derived']:
        qp,gp=qa_files(root,'train','derived'); qs+=read_json(qp); gs+=read_json(gp)
    gold={g['qid']:g for g in gs if g['primary_eligible'] and g['supported_onsets']}
    val=read_json(root/'data/questions/val.json'); vg=read_json(root/'data/targets/val.json')
    usable=[]; skipped=[]
    # No supervision is used to choose input images; only to validate SFT target support.
    for q in qs:
        if q['qid'] not in gold:
            continue
        g=gold[q['qid']]
        with np.load(root/f'artifacts/features/surgformer/{q["video_id"]}.npz') as f:
            ts=sampled_seconds(f['phase'],PHASES.index(q['target_phase']),config,budget)
        t=min(ts,key=lambda t:min(abs(t-o) for o in g['supported_onsets']))
        if min(abs(t-o) for o in g['supported_onsets'])>5:
            skipped.append(q['qid']); continue
        usable.append((q,dict(answer=g['answer'],second=int(t))))
    if not usable:
        raise ValueError('No supported SFT examples under frame budget')
    provenance=dict(revision=revision,budget=budget,config=config,seed=seed,
                    train_examples=len(usable),skipped_retrieval_misses=skipped,
                    target_modules='language_model self_attn q/k/v/o projections',lr=2e-5,epochs=epochs)
    write_json(out/'sft_data_report.json',provenance)
    rng=np.random.default_rng(seed); best=-1; history=[]
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); total=0
        for i,index in enumerate(rng.permutation(len(usable))):
            q,target=usable[index]
            messages,images,_=messages_and_images(root,q,config,budget)
            inputs=encode(processor,messages,images,device,target)
            loss=model(**inputs).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(q['qid'])
            group=min(4,len(usable)-(i//4)*4)
            (loss/group).backward(); total+=float(loss.detach())
            if (i+1)%4==0 or i+1==len(usable):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        pred=infer_rows(model,processor,root,val,config,device,budget)
        metrics=aggregate([r for r in score_rows(root,val,vg,pred) if r['primary']])
        history.append(dict(epoch=epoch,loss=total/len(usable),val=metrics))
        if metrics['supported_joint@5']>best:
            best=metrics['supported_joint@5']; model.save_pretrained(out/'best'); processor.save_pretrained(out/'best')
            write_json(out/'best/aha_adapter.json',provenance)
            write_json(out/'best_val_predictions.json',pred)
        write_json(out/'history.json',history)
    return dict(best_val_supported_joint5=best,examples=len(usable))
