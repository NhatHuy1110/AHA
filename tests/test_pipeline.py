"""Actual optimizer/checkpoint/inference/evaluation plumbing on explicitly synthetic data."""
from pathlib import Path
import numpy as np
import pytest
import torch
from aha.common import read_json,write_json,atomic_npz,sha,fingerprint,qa_files
from aha.data import smooth_onsets
from aha.engine import train,predict
from aha.evaluate import evaluate

def fixture(root):
    config=read_json(Path(__file__).resolve().parents[1]/'configs/full.json')
    config['features']['name']='synthetic'
    config['model'].update(visual_dim=8,text_dim=6,width=16,temporal_layers=3,fusion_layers=1,heads=2,dropout=0.,feature_dropout=0.)
    config['train'].update(epochs=2,warmup_steps=1)
    write_json(root/'configs/test.json',config)
    write_json(root/'data/manifests/splits.json',dict(train=['001'],val=['051'],test=['053']))
    rng=np.random.default_rng(1)
    texts=['What tool?', 'tool0','tool1','tool2','tool3']
    tp=root/'artifacts/text/embeddings.npz'
    atomic_npz(tp,embeddings=rng.normal(size=(5,6)).astype(np.float32))
    write_json(tp.with_suffix('.json'),dict(texts=texts,sha256=sha(tp),synthetic=True))
    spec=dict(synthetic=True,warning='SOFTWARE TEST ONLY')
    write_json(root/'artifacts/features/synthetic/spec.json',spec)
    for split,vid in [('train','001'),('val','051'),('test','053')]:
        n=40
        fp=root/f'artifacts/features/synthetic/{vid}.npz'
        atomic_npz(fp,hidden=rng.normal(size=(n,8)).astype(np.float16),times=np.arange(n))
        write_json(fp.with_suffix('.json'),dict(complete=True,frames=n,sha256=sha(fp),synthetic=True,
                   signature=fingerprint(dict(spec=spec,frames='synthetic',limit=None))))
        write_json(root/f'data/frames/{vid}.json',dict(frames=n,synthetic=True,zip_sha256='synthetic'))
        q=dict(qid=vid,video_id=vid,question='What tool?',options=[f'{chr(65+i)}. tool{i}' for i in range(4)],
               target_phase='SuspendPericardium',option_tools=['Needle_holder','Aspirator','Endotherm_knife','Knife'])
        g=dict(qid=vid,video_id=vid,answer='A',onsets=[20],supported_onsets=[20],primary_eligible=True,flags=[])
        for kind in ('original','derived'):
            qp,gp=qa_files(root,split,kind)
            write_json(qp,[dict(q,qid=f'{vid}_{kind}')]); write_json(gp,[dict(g,qid=f'{vid}_{kind}')])
        if split=='train':
            onset=np.zeros((n,12),np.float32); onset[20,1]=1
            phase=np.zeros(n,np.int64); phase[20:]=1
            atomic_npz(root/f'data/labels/train/{vid}.npz',phase=phase,tool=np.zeros((n,8),np.float32),
                       tool_mask=np.ones((n,8),bool),onset=onset,boundary_mask=np.ones(n,bool))
    return config

def test_end_to_end_checkpoint_resume_and_test_isolation(tmp_path):
    torch.set_num_threads(2)
    root=tmp_path/'synthetic'; config=fixture(root)
    # Hide both kinds of test gold during training/inference: those paths must not require it.
    held=[]
    for kind in ('original','derived'):
        gp=qa_files(root,'test',kind)[1]; gp.rename(gp.with_suffix('.held')); held.append(gp)
    result=train(root,root/'configs/test.json',root/'runs/test',device='cpu')
    assert result['epochs']==2 and (root/'runs/test/best.pt').exists()
    for decode in ('joint','cascade'):
        predict(root,root/'runs/test/best.pt','test',root/f'runs/test/{decode}.json',device='cpu',decode=decode)
    predict(root,root/'runs/test/best.pt','test',root/'runs/test/derived.json',device='cpu',qa='derived')
    predict(root,root/'runs/test/best.pt','val',root/'runs/test/zero.json',device='cpu',diagnostic='zero_visual')
    for gp in held:
        gp.with_suffix('.held').rename(gp)
    report=evaluate(root,'test',root/'runs/test/joint.json',root/'runs/test/eval',paired=root/'runs/test/cascade.json')
    assert report['primary']['n']==1 and report['primary']['citation_valid']==1
    assert 'paired_delta_ci supported_joint@5' in report
    assert evaluate(root,'test',root/'runs/test/derived.json',root/'runs/test/eval_d',qa='derived')['primary']['n']==1
    resumed=train(root,root/'configs/test.json',root/'runs/test',device='cpu',resume=True)
    assert resumed['epochs']==2
    config['train']['lr']*=2; write_json(root/'configs/test.json',config)
    with pytest.raises(ValueError,match='changed data/code/config/seed'):
        train(root,root/'configs/test.json',root/'runs/test',device='cpu',resume=True)

@pytest.mark.parametrize('variant',[dict(),dict(structured=False),dict(residual=False),dict(workflow='head')])
def test_can_overfit_answer_and_onset_on_tiny_example(variant):
    from aha.model import AnswerOnsetDecoder
    from aha.losses import qa_loss,dense_loss
    from test_core import config,inputs,target
    torch.manual_seed(7); torch.set_num_threads(2)
    c=config(**variant); m=AnswerOnsetDecoder(c['model']); x=inputs(q=1); y=target(q=1)
    y['tool'][12:,2]=1  # option index 2 maps to tool 2, which enters at the onset
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    losses=[]
    for _ in range(150):
        optimizer.zero_grad(); out=m(**x)
        loss=qa_loss(out,y,c)[0]+dense_loss(m,out,y,c['loss'])[0]; losses.append(float(loss.detach()))
        loss.backward(); optimizer.step()
    assert losses[-1]<losses[0]*.3
    m.eval(); out=m(**x)['logits'][0]; t,a=divmod(int(out.argmax()),4)
    assert a==2 and abs(t-12)<=2
