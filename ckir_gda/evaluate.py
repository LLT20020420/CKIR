"""Explicit evaluation of saved predictions, separate from training/inference."""
from pathlib import Path
import json
import numpy as np
from .io import digest


def counts(predictions,labels,zero):
    p,y,z=map(np.asarray,(predictions,labels,zero))
    if p.ndim!=1 or p.shape!=y.shape or p.shape!=z.shape or len(y)==0:
        raise ValueError('prediction/label/baseline shapes must match')
    correct=int(np.sum(p==y));baseline=int(np.sum(z==y))
    benefit=int(np.sum((z!=y)&(p==y)));harm=int(np.sum((z==y)&(p!=y)))
    if correct-baseline!=benefit-harm:
        raise RuntimeError('paired count identity failed')
    return {'correct':correct,'denominator':len(y),'accuracy_percent':100.*correct/len(y),
        'baseline_correct':baseline,'benefit':benefit,'harm':harm,'net_correct':correct-baseline}


def evaluate_saved(scores_path,labels_path):
    p=Path(scores_path)
    metadata=json.loads(p.with_suffix('.json').read_text('utf-8'))
    labels=json.loads(Path(labels_path).read_text('utf-8'))
    if metadata['sha256']!=digest(p) or metadata['query_ids']!=labels['query_ids']:
        raise ValueError('saved scores hash or evaluation query order differs')
    with np.load(p,allow_pickle=False) as z:
        result={name:counts(z[name+'_predictions'],labels['labels'],labels['zero_predictions'])
                for name in ('ckir','gda','fused')}
    return {'episode_id':metadata['episode_id'],'methods':result,'reference':'recorded original-prompt zero-shot CLIP'}
