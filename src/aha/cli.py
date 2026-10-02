import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='AHA: joint answer and surgical onset grounding')
    p.add_argument('--root',type=Path,default=Path('.'),help='AHA root; all runtime data stays here')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--workspace',required=True); a.add_argument('--copy',action='store_true')
    a=sub.add_parser('audit'); a.add_argument('--full',action='store_true')
    sub.add_parser('freeze'); sub.add_parser('verify'); sub.add_parser('derive')
    for command in ('extract','encode'):
        a=sub.add_parser(command); a.add_argument('--device',default='cuda'); a.add_argument('--ids',nargs='+'); a.add_argument('--limit',type=int); a.add_argument('--review-dir')
        if command=='encode':
            a.add_argument('--name',required=True,help='folder under assets/encoders and artifacts/features'); a.add_argument('--batch',type=int,default=64)
    a=sub.add_parser('text'); a.add_argument('--device',default='cpu'); a.add_argument('--model-path')
    a=sub.add_parser('train'); a.add_argument('--config',default='configs/full.json'); a.add_argument('--out',required=True)
    a.add_argument('--seed',type=int,default=17); a.add_argument('--device',default='cuda'); a.add_argument('--resume',action='store_true')
    a=sub.add_parser('predict'); a.add_argument('--checkpoint',required=True); a.add_argument('--split',choices=['val','test'],default='val')
    a.add_argument('--out',required=True); a.add_argument('--device',default='cuda'); a.add_argument('--diagnostic',choices=['zero_visual','shuffle_visual'])
    a.add_argument('--qa',choices=['original','derived'],default='original'); a.add_argument('--decode',choices=['joint','cascade'],default='joint')
    a=sub.add_parser('evaluate'); a.add_argument('--split',choices=['val','test'],default='val'); a.add_argument('--predictions',required=True)
    a.add_argument('--out',required=True); a.add_argument('--paired'); a.add_argument('--qa',choices=['original','derived'],default='original')
    a=sub.add_parser('baseline'); a.add_argument('--config',default='configs/full.json'); a.add_argument('--split',choices=['val','test'],default='val')
    a.add_argument('--method',choices=['prior','surgformer_index'],required=True); a.add_argument('--out',required=True)
    a.add_argument('--qa',choices=['original','derived'],default='original')
    for command in ('vlm-predict','vlm-train'):
        a=sub.add_parser(command); a.add_argument('--config',default='configs/full.json'); a.add_argument('--revision',required=True)
        a.add_argument('--out',required=True); a.add_argument('--device',default='cuda'); a.add_argument('--budget',type=int,default=48)
        if command=='vlm-predict':
            a.add_argument('--adapter'); a.add_argument('--split',choices=['val','test'],default='val')
        else:
            a.add_argument('--seed',type=int,default=17); a.add_argument('--epochs',type=int,default=3)
    args=vars(p.parse_args()); cmd=args.pop('command'); root=args.pop('root').resolve()
    # Relative user paths resolve within AHA, never relative to the original workspace.
    for key in ('config','out','checkpoint','predictions','paired','adapter','model_path','review_dir'):
        if args.get(key) and not Path(args[key]).is_absolute():
            args[key]=root/args[key]
    if cmd=='prepare':
        from .prepare import prepare
        result=prepare(args['workspace'],root,hardlink=not args['copy'])
    elif cmd in ('audit','freeze','verify'):
        from . import audit
        result=getattr(audit,cmd)(root,**args)
    elif cmd=='derive':
        from .derive import derive
        result=derive(root)
    elif cmd in ('extract','text','encode'):
        from . import features
        result=getattr(features,'encoder' if cmd=='encode' else cmd)(root,**args)
    elif cmd=='train':
        from .engine import train
        args['config_path']=args.pop('config'); result=train(root,**args)
    elif cmd=='predict':
        from .engine import predict
        args['checkpoint_path']=args.pop('checkpoint'); result=predict(root,**args)
    elif cmd=='evaluate':
        from .evaluate import evaluate
        result=evaluate(root,**args)
    elif cmd=='baseline':
        from .baselines import run
        args['config_path']=args.pop('config'); result=run(root,**args)
    else:
        from . import vlm
        args['config_path']=args.pop('config')
        result=(vlm.infer if cmd=='vlm-predict' else vlm.sft)(root,**args)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
