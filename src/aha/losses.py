import torch
import torch.nn.functional as F

def masked_mean(values,mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)

def dense_loss(model,out,target,weights,phase_weight=None):
    """Recognition supervision on every labelled second of one training video."""
    terms={}
    terms['tool']=masked_mean(F.binary_cross_entropy_with_logits(out['tool'],target['tool'].float(),reduction='none'),target['tool_mask'])
    labels=target['phase'].long(); mask=labels>=0
    terms['phase']=F.cross_entropy(out['phase'][mask],labels[mask],weight=phase_weight) if mask.any() else out['phase'].sum()*0
    if model.crf is not None:
        terms['crf']=model.crf.nll(out['emit'],labels)
    # Quality focal loss on Gaussian-smoothed onset targets; left-censored starts masked.
    truth=target['onset_soft'].float()
    b=target['boundary_mask'][:,None].expand_as(truth)
    bce=F.binary_cross_entropy_with_logits(out['onset'],truth,reduction='none')
    focal=(truth-out['onset'].sigmoid()).abs().pow(2)*bce
    terms['onset']=(focal*b).sum()/((truth>0.99)&b).sum().clamp_min(1)
    return sum(weights.get(k,0)*v for k,v in terms.items()),terms

def qa_loss(out,target,config):
    """Set-likelihood over (second, option) pairs for every question of one video."""
    s=out['logits']; w=config['loss']; tol=config['train']['positive_tolerance']
    ts=torch.arange(s.shape[1],device=s.device)
    labelled=target['phase'].to(s.device)>=0
    total=s.sum()*0; hits=0
    for i,(a,ons,all_ons) in enumerate(zip(target['answer'],target['onsets'],target['all_onsets'])):
        si=s[i]
        marginal=torch.logsumexp(si,0)
        total=total+w['answer']*F.cross_entropy(marginal[None],torch.tensor([a],device=s.device))
        positive=(ts[:,None]-torch.as_tensor(ons,device=s.device)[None]).abs().min(1).values<=tol
        if positive.any():
            hits+=1
            total=total+w['joint']*(torch.logsumexp(si.flatten(),0)-torch.logsumexp(si[positive,a],0))
            # A repeated valid onset is never a hard negative, supported or not.
            negative=(ts[:,None]-torch.as_tensor(all_ons,device=s.device)[None]).abs().min(1).values>5
            negative&=labelled
            if negative.any() and w['hard_negative']:
                total=total+w['hard_negative']*F.relu(0.2+si[negative,a].max()-si[positive,a].max())
    n=len(target['answer'])
    text=F.nll_loss(out['log_pi'],target['question_phase'])+F.nll_loss(out['log_rho'].flatten(0,1),target['option_tool'].flatten())
    return total/n+w['text']*text,dict(positive_hit=hits/n)
