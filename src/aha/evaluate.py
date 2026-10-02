"""Retrospective historical onset metric + explicit tool-supported sensitivity."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read_json, write_json, qa_files

def score_rows(root,questions,gold,predictions,tolerances=(3,5,10,15)):
    gm={g['qid']:g for g in gold}; pm={}
    frames={}
    def duration(vid):
        if vid not in frames:
            frames[vid]=read_json(Path(root)/f'data/frames/{vid}.json')['frames']
        return frames[vid]
    expected={q['qid'] for q in questions}
    for p in predictions:
        if p['qid'] in pm or p['qid'] not in expected:
            raise ValueError('Duplicate or unexpected prediction qid')
        pm[p['qid']]=p
    rows=[]
    for q in questions:
        g=gm[q['qid']]; p=pm.get(q['qid'],{})
        t=p.get('second'); dense=p.get('observed')=='dense'
        observed=p.get('observed_seconds',[])
        # Dense models processed every second: a citation is any real frame index of that video.
        valid=(isinstance(t,(int,float)) and not isinstance(t,bool) and np.isfinite(t)
               and (float(t).is_integer() if dense else t in observed) and 0<=t<duration(q['video_id'])
               and p.get('video_id')==q['video_id'] and p.get('answer') in ('A','B','C','D')) if p.get('answer') else False
        correct=bool(p.get('answer')==g['answer'])
        error=min((abs(t-o) for o in g['onsets']),default=float('inf')) if valid else float('inf')
        strict=min((abs(t-o) for o in g['supported_onsets']),default=float('inf')) if valid else float('inf')
        retrieved=p.get('candidate_seconds',observed)
        recall=bool(p) and (dense or any(abs(t0-o)<=5 for t0 in retrieved for o in g['onsets']))
        row=dict(qid=q['qid'],video_id=q['video_id'],phase=q['target_phase'],
                 primary=g['primary_eligible'],flags=g['flags'],answer=int(correct),
                 citation_valid=int(valid),onset_error=error if np.isfinite(error) else None,
                 retrieval_recall5=int(recall))
        for tau in tolerances:
            row[f'grounding@{tau}']=int(error<=tau)
            row[f'joint@{tau}']=int(correct and error<=tau)
            row[f'supported_joint@{tau}']=int(correct and strict<=tau)
        rows.append(row)
    return rows

def aggregate(rows):
    if not rows:
        return dict(n=0)
    metrics=['answer','citation_valid','retrieval_recall5']+[k for k in rows[0] if '@' in k]
    out=dict(n=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in metrics})
    errors=[r['onset_error'] for r in rows if r['onset_error'] is not None]
    out.update(onset_mae_valid=float(np.mean(errors)) if errors else None,
               onset_error_n=len(errors),invalid_citations=len(rows)-len(errors))
    return out

def case_bootstrap(rows,other=None,repeats=2000,seed=20261002,metric='joint@5'):
    groups=defaultdict(list)
    other_map={r['qid']:r for r in other} if other is not None else None
    if other_map is not None and set(other_map)!={r['qid'] for r in rows}:
        raise ValueError('Paired bootstrap needs identical qids')
    for r in rows:
        v=r[metric]-(other_map[r['qid']][metric] if other_map is not None else 0)
        groups[r['video_id']].append(v)
    if not groups:
        return None
    values=list(groups.values()); rng=np.random.default_rng(seed); estimates=[]
    for _ in range(repeats):
        sample=rng.integers(len(values),size=len(values))
        estimates.append(float(np.mean([v for i in sample for v in values[i]])))
    return dict(metric=metric,method='percentile case-cluster bootstrap; question-weighted',replicates=repeats,
                low=float(np.quantile(estimates,.025)),high=float(np.quantile(estimates,.975)),
                estimate=float(np.mean([v for group in values for v in group])))

def summarize(rows):
    primary=[r for r in rows if r['primary']]
    out=dict(primary=aggregate(primary),all=aggregate(rows),joint5_ci=case_bootstrap(primary),
             supported_joint5_ci=case_bootstrap(primary,metric='supported_joint@5'),
             answer_ci=case_bootstrap(primary,metric='answer'),grounding5_ci=case_bootstrap(primary,metric='grounding@5'))
    for field in ('video_id','phase'):
        out[f'by_{field}']={k:aggregate([r for r in primary if r[field]==k]) for k in sorted({r[field] for r in primary})}
    out['sensitivity']={
        'without_left_censored':aggregate([r for r in primary if 'left_censored' not in r['flags']]),
        'repeated_phase':aggregate([r for r in primary if 'repeated_phase' in r['flags']]),
        'single_phase':aggregate([r for r in primary if 'repeated_phase' not in r['flags']])}
    return out

def evaluate(root,split,predictions,out,paired=None,qa='original'):
    root=Path(root)
    qp,gp=qa_files(root,split,qa)
    qs=read_json(qp); gold=read_json(gp)
    rows=score_rows(root,qs,gold,read_json(predictions))
    report=summarize(rows)
    if paired:
        other=score_rows(root,qs,gold,read_json(paired))
        a,b=[r for r in rows if r['primary']],[r for r in other if r['primary']]
        for metric in ('joint@5','supported_joint@5','answer','grounding@5'):
            report[f'paired_delta_ci {metric}']=case_bootstrap(a,b,metric=metric)
    out=Path(out)
    write_json(out/'per_question.json',rows)
    write_json(out/'metrics.json',report)
    return report
