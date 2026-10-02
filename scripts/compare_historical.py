"""Paired case-bootstrap comparison with frozen PCJD v3 test rows."""
import argparse
import csv
from pathlib import Path
from aha.common import read_json,write_json
from aha.evaluate import case_bootstrap

p=argparse.ArgumentParser(); p.add_argument('--rows',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
args=p.parse_args(); root=Path(__file__).resolve().parents[1]
new=[r for r in read_json(args.rows) if r['primary']]
with (root/'references/historical_pcjd_v3/test_per_question.csv').open() as f:
    old=[dict(qid=r['qid'],video_id=r['video_id'],**{'joint@5':int(r['joint_correct'])})
         for r in csv.DictReader(f) if r['method']=='PCJD' and int(r['tol'])==5 and int(r['primary_eligible'])]
write_json(args.out,dict(comparison='AHA minus historical PCJD v3; retrospective same primary qids',
                        paired_joint5_delta=case_bootstrap(new,old,metric='joint@5')))
