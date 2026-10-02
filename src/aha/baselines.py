"""New-split deterministic comparators; these are NOT renamed historical PCJD."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .common import PHASES,TOOLS,read_json,write_json,qa_files
from .proposals import select_times

def run(root,config_path,split,method,out,qa='original'):
    root=Path(root); config=read_json(config_path)
    qs=read_json(qa_files(root,split,qa)[0])
    counts=defaultdict(Counter)
    for g in read_json(root/'data/targets/train.json'):
        if g['primary_eligible']:
            counts[g['target_phase']][g['answer_tool']]+=1
    result=[]
    for q in qs:
        with np.load(root/f'artifacts/features/surgformer/{q["video_id"]}.npz') as f:
            times,centers,curve=select_times(f['phase'],PHASES.index(q['target_phase']),config['proposals'])
            ti=int(times[np.argmax(curve[times])])
            if method=='prior':
                scores=[counts[q['target_phase']][t] for t in q['option_tools']]
                ti=0  # Explicit fixed timestamp: no claim that a prior can ground.
            elif method=='surgformer_index':
                scores=f['tool'][ti:min(ti+6,len(f['tool']))].mean(0)[[TOOLS.index(t) for t in q['option_tools']]]
            else:
                raise ValueError(method)
        result.append(dict(qid=q['qid'],video_id=q['video_id'],answer=chr(65+int(np.argmax(scores))),
                           second=ti,observed_seconds=times.tolist(),candidate_seconds=times.tolist(),method=method))
    write_json(out,result)
    return dict(predictions=len(result))
