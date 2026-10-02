import json
import numpy as np
import pytest
import torch
from aha.model import AnswerOnsetDecoder
from aha.losses import qa_loss,dense_loss,masked_mean
from aha.data import smooth_onsets
from aha.proposals import select_times,context_features
from aha.evaluate import score_rows,case_bootstrap
from aha.prepare import labels
from aha.derive import derive_video

def config(**model):
    m=dict(visual_dim=8,text_dim=6,width=16,temporal_layers=3,fusion_layers=1,heads=2,dropout=0.,feature_dropout=0.,
           tool_window=3,evidence_floor=-9.2,variant='joint',workflow='crf',structured=True,residual=True)
    m.update(model)
    return dict(model=m,train=dict(positive_tolerance=2),
                loss=dict(joint=1.,answer=.5,hard_negative=.1,text=.5,phase=.5,crf=.5,tool=1.,onset=.5))

def inputs(n=30,q=2):
    return dict(x=torch.randn(n,8),position=torch.linspace(0,1,n)[:,None],question=torch.randn(q,6),options=torch.randn(q,4,6))

def target(n=30,q=2,onsets=(12,)):
    onset=np.zeros((n,12),np.float32); onset[list(onsets),1]=1
    phase=torch.zeros(n,dtype=torch.long); phase[onsets[0]:]=1
    return dict(answer=[2]*q,onsets=[list(onsets)]*q,all_onsets=[list(onsets)]*q,
                question_phase=torch.ones(q,dtype=torch.long),option_tool=torch.arange(4)[None].repeat(q,1),
                phase=phase,tool=torch.zeros(n,8),tool_mask=torch.ones(n,8,dtype=torch.bool),
                onset=torch.from_numpy(onset),onset_soft=torch.from_numpy(smooth_onsets(onset,2.)),
                boundary_mask=torch.ones(n,dtype=torch.bool))

VARIANTS=[dict(),dict(workflow='head'),dict(residual=False),dict(structured=False),dict(variant='independent')]

@pytest.mark.parametrize('variant',VARIANTS)
def test_option_permutation_equivariance(variant):
    torch.manual_seed(1); m=AnswerOnsetDecoder(config(**variant)['model']).eval(); x=inputs()
    first=m(**x)['logits']; perm=torch.tensor([3,0,2,1]); x['options']=x['options'][:,perm]
    assert torch.allclose(m(**x)['logits'],first[:,:,perm],atol=1e-5)

@pytest.mark.parametrize('variant',VARIANTS)
def test_losses_finite_and_reach_every_used_module(variant):
    torch.manual_seed(2); c=config(**variant); m=AnswerOnsetDecoder(c['model']); out=m(**inputs())
    lq,stats=qa_loss(out,target(),c); ld,_=dense_loss(m,out,target(),c['loss'])
    (lq+ld).backward()
    assert stats['positive_hit']==1 and torch.isfinite(lq+ld)
    grads={n:p.grad for n,p in m.named_parameters()}
    for name in ('proj.weight','encoder.0.conv.weight','text.weight','question_phase.weight','phase.weight','tool.weight','onset.weight'):
        assert grads[name] is not None and grads[name].abs().sum()>0,name
    if c['model']['workflow']=='crf':
        assert grads['crf.transition'].abs().sum()>0

def test_independent_distribution_factorizes():
    m=AnswerOnsetDecoder(config(variant='independent')['model']).eval()
    s=m(**inputs())['logits'][0]; p=s.flatten().softmax(0).reshape_as(s)
    assert torch.allclose(p,p.sum(1)[:,None]*p.sum(0)[None],atol=1e-6)

def test_joint_distribution_does_not_factorize():
    torch.manual_seed(3); m=AnswerOnsetDecoder(config()['model']).eval()
    s=m(**inputs())['logits'][0]; p=s.flatten().softmax(0).reshape_as(s)
    assert not torch.allclose(p,p.sum(1)[:,None]*p.sum(0)[None],atol=1e-6)

def test_questions_of_one_video_are_scored_independently():
    torch.manual_seed(4); m=AnswerOnsetDecoder(config()['model']).eval(); x=inputs(q=3)
    full=m(**x)['logits']; x['question']=x['question'][1:2]; x['options']=x['options'][1:2]
    assert torch.allclose(m(**x)['logits'][0],full[1],atol=1e-5)

def test_onset_outside_video_is_finite_and_counted_as_miss():
    c=config(); m=AnswerOnsetDecoder(c['model']); y=target(); y['onsets']=[[500]]*2
    loss,stats=qa_loss(m(**inputs()),y,c)
    assert stats['positive_hit']==0 and torch.isfinite(loss)

def test_empty_masks_finite():
    c=config(); m=AnswerOnsetDecoder(c['model']); y=target()
    y['tool_mask'].zero_(); y['boundary_mask'].zero_(); y['phase'].fill_(-1)
    out=m(**inputs()); loss=qa_loss(out,y,c)[0]+dense_loss(m,out,y,c['loss'])[0]
    assert torch.isfinite(loss)

def test_smoothed_onsets_peak_and_support():
    onset=np.zeros((50,12),np.float32); onset[[10,30],4]=1; onset[0,2]=1
    soft=smooth_onsets(onset,2.)
    assert soft[10,4]==1 and soft[30,4]==1 and soft[0,2]==1 and soft[20,4]==0
    assert 0<soft[12,4]<1 and soft[:,0].sum()==0

def test_proposals_unique_in_range_and_label_free():
    p=np.zeros((300,12)); p[:,0]=.3; p[100:140,1]=.8
    times,_,_=select_times(p,1,dict(topk=16,nms_seconds=30,coverage=8,radius=32))
    assert np.array_equal(times,np.unique(times)) and times.min()==0 and times.max()==299

def test_context_uses_contiguous_seconds_not_sparse_neighbors():
    f=np.arange(100,dtype=np.float32)[:,None]
    x=context_features(f,np.array([20,80]),radius=2)
    assert np.allclose(x,[[20,18.5,21.5],[80,78.5,81.5]])

def test_unknown_label_mask_and_offset(tmp_path):
    p=tmp_path/'p.txt'; t=tmp_path/'t.txt'
    p.write_text('Frame Phase\n100 Preparation\n101 BlockAorta\n')
    t.write_text('header\n100 1 0 -1 0 0 0 0 0\n101 0 1 0 0 0 0 0 0\n')
    lab=labels(p,t,3,5)
    assert np.all(lab['phase'][:3]==-1) and not lab['tool_mask'][3,2]
    assert lab['onset'][4,5]==1 and not lab['boundary_mask'][3]
    v=torch.tensor([2.,1000.],requires_grad=True)
    loss=masked_mean(v,torch.tensor([True,False])); loss.backward()
    assert loss==2 and v.grad[1]==0

@pytest.fixture
def scoring(tmp_path):
    (tmp_path/'data/frames').mkdir(parents=True)
    (tmp_path/'data/frames/001.json').write_text(json.dumps(dict(frames=1000)))
    q=dict(qid='x',video_id='001',target_phase='Preparation')
    g=dict(qid='x',answer='B',onsets=[100,200],supported_onsets=[200],primary_eligible=True,flags=['repeated_phase'])
    return tmp_path,q,g

def test_historical_and_supported_metrics_distinct(scoring):
    root,q,g=scoring
    for p in (dict(qid='x',video_id='001',answer='B',second=100,observed_seconds=[100]),
              dict(qid='x',video_id='001',answer='B',second=100,observed='dense')):
        r=score_rows(root,[q],[g],[p])[0]
        assert r['joint@5']==1 and r['supported_joint@5']==0 and r['citation_valid']==1

def test_missing_unobserved_and_out_of_video_citations_fail(scoring):
    root,q,g=scoring
    for pred in ([],[dict(qid='x',video_id='001',answer='B',second=200,observed_seconds=[100])],
                 [dict(qid='x',video_id='001',answer='B',second=1000,observed='dense')],
                 [dict(qid='x',video_id='001',answer='B',second=200.5,observed='dense')],
                 [dict(qid='x',video_id='002',answer='B',second=200,observed='dense')]):
        r=score_rows(root,[q],[g],pred)[0]
        assert r['joint@5']==0 and r['citation_valid']==0

def test_duplicate_prediction_rejected(scoring):
    root,q,g=scoring; p=dict(qid='x')
    with pytest.raises(ValueError):
        score_rows(root,[q],[g],[p,p])

def test_paired_cluster_bootstrap_identical_predictions_zero(scoring):
    root,q,g=scoring; r=score_rows(root,[q],[g],[])
    for metric in ('joint@5','supported_joint@5'):
        ci=case_bootstrap(r,r,repeats=10,metric=metric)
        assert ci['low']==ci['high']==0

def test_derived_questions_single_choice_across_repeated_onsets():
    n=20; onset=np.zeros((n,12)); onset[[5,15],0]=1
    tool=np.zeros((n,8)); tool[[5,15],2]=1; tool[5,1]=1
    lab=dict(onset=onset,tool=tool,tool_mask=np.ones((n,8),bool))
    qs,gs=derive_video('001',lab)
    assert len(qs)==1 and gs[0]['answer_tool']=='Endotherm_knife' and gs[0]['flags']==['repeated_phase']
    assert 'Aspirator' not in qs[0]['option_tools'] and 'Preparation' in qs[0]['question']
    assert qs[0]['options'][ord(gs[0]['answer'])-65].endswith('Endotherm Knife')
