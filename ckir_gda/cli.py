"""Explicit local commands; nothing trains or downloads on import."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .io import load_config,load_episode,digest
from .pipeline import setup,train_stage,infer
from .prepare import prepare_episode
from .evaluate import evaluate_saved

PACKAGE=Path(__file__).resolve().parent.parent


def demo(out,device):
    out=Path(out)
    if out.exists():raise FileExistsError('choose a fresh demo output directory')
    out.mkdir(parents=True)
    rng=np.random.default_rng(20261024)
    n,d,c,nq=24,12,4,8
    unit=lambda x:x/np.linalg.norm(x,axis=1,keepdims=True)
    arrays={'support':unit(rng.normal(size=(n,d))), 'query':unit(rng.normal(size=(nq,d))),
        'observed_labels':np.arange(n,dtype=np.int64)%c,
        'template':unit(rng.normal(size=(c,d))).T.astype(np.float16),
        'cupl':unit(rng.normal(size=(c,d))).T.astype(np.float16),
        'gda_text':unit(rng.normal(size=(c,d))).T.astype(np.float32)}
    np.savez_compressed(out/'episode.npz',**arrays)
    manifest={'format':'ckir-gda-episode-v1','episode_id':'synthetic-demo','kind':'synthetic','phase':'demo',
        'support_seed':1,'label_version':1,'class_names':[f'synthetic-{i}' for i in range(c)],
        'support_ids':[f's-{i}' for i in range(n)],'query_ids':[f'q-{i}' for i in range(nq)],
        'arrays':{'path':'episode.npz','sha256':digest(out/'episode.npz')}}
    (out/'episode.json').write_text(json.dumps(manifest,indent=2)+'\n','utf-8')
    config=load_config(PACKAGE/'configs/retained.json')
    config['scope']='Synthetic software demonstration; not a paper experiment'
    config['text_training'].update(epochs=2,retained_epoch=2,batch_size=8)
    config['head_training']['steps']=1
    config['joint_training']['steps']=1
    (out/'demo-config.json').write_text(json.dumps(config,indent=2)+'\n','utf-8')
    manifest,arrays=load_episode(out/'episode.json')
    initial=None
    for stage in ('text','head','joint'):
        target=out/f'{stage}.npz'
        train_stage(stage,arrays,manifest,config,device,target,initial)
        initial=target
    scores,meta=infer(arrays,manifest,config,device,initial)
    np.savez_compressed(out/'scores.npz',**scores)
    result={'status':'PASS_SYNTHETIC_THREE_STAGE_AND_GDA_DEMO','synthetic':True,'paper_accuracy_evaluated':False,
            'device':device,'support':n,'query':nq,'classes':c,'feature_dim':d,'inference':meta}
    (out/'demo-result.json').write_text(json.dumps(result,indent=2)+'\n','utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='CKIR-GDA release candidate; explicit cached-feature inputs')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('train','infer'):
        a=sub.add_parser(name)
        a.add_argument('--manifest',required=True,type=Path)
        a.add_argument('--data-root',type=Path)
        a.add_argument('--config',type=Path,default=PACKAGE/'configs/retained.json')
        a.add_argument('--device',default='auto')
        a.add_argument('--output',type=Path,required=True)
    sub.choices['train'].add_argument('--stage',choices=('text','head','joint','all'),required=True)
    sub.choices['train'].add_argument('--init',type=Path)
    sub.choices['infer'].add_argument('--checkpoint',type=Path,required=True)
    d=sub.add_parser('demo')
    d.add_argument('--output',type=Path,required=True)
    d.add_argument('--device',default='cpu')
    a=sub.add_parser('prepare')
    a.add_argument('--index',type=Path,default=PACKAGE/'data/episodes.json')
    a.add_argument('--feature-cache',type=Path,required=True)
    a.add_argument('--template',type=Path,required=True)
    a.add_argument('--cupl',type=Path,required=True)
    a.add_argument('--phase',choices=('development','pilot'),required=True)
    a.add_argument('--seed',type=int,required=True)
    a.add_argument('--label-version',type=int,choices=range(4),required=True)
    a.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate')
    e.add_argument('--scores',type=Path,required=True)
    e.add_argument('--labels',type=Path,default=PACKAGE/'data/evaluation-labels.json')
    args=p.parse_args(argv)
    if args.command=='evaluate':
        print(json.dumps(evaluate_saved(args.scores,args.labels),ensure_ascii=False,indent=2))
        return
    if args.command=='prepare':
        print(json.dumps(prepare_episode(args.index,args.feature_cache,args.template,args.cupl,
            phase=args.phase,seed=args.seed,label_version=args.label_version,output=args.output),ensure_ascii=False,indent=2))
        return
    device=setup(args.device)
    if args.command=='demo':
        result=demo(args.output,device)
    else:
        config=load_config(args.config)
        manifest,arrays=load_episode(args.manifest,data_root=args.data_root)
        if args.command=='train':
            if args.stage=='all':
                if args.init:raise ValueError('--stage all starts from zero, so --init is not applicable')
                if args.output.exists():raise FileExistsError('choose a fresh stage output directory')
                args.output.mkdir(parents=True)
                initial=None
                for stage in ('text','head','joint'):
                    target=args.output/f'{stage}.npz'
                    train_stage(stage,arrays,manifest,config,device,target,initial)
                    initial=target
                result={'status':'COMPLETE_THREE_STAGE_TRAINING','checkpoint':str(initial),'device':device}
            else:
                _,metadata=train_stage(args.stage,arrays,manifest,config,device,args.output,args.init)
                result={'status':'COMPLETE_STAGE','stage':args.stage,'checkpoint':str(args.output),'audit':metadata['audit']}
        else:
            if args.output.suffix!='.npz':raise ValueError('--output must end in .npz')
            if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError('choose a fresh score output path')
            scores,metadata=infer(arrays,manifest,config,device,args.checkpoint)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,**scores)
            metadata.update(sha256=digest(args.output),episode_id=manifest['episode_id'],query_ids=manifest['query_ids'],
                            phase=manifest.get('phase'),support_seed=manifest['support_seed'],label_version=manifest['label_version'],
                            config_sha256=digest(args.config),checkpoint_sha256=digest(args.checkpoint))
            args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n','utf-8')
            result={'status':'COMPLETE_INFERENCE','scores':str(args.output),'query_labels_used':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))
