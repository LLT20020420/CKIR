"""Explicit relative episode assets, numeric NPZ files and checkpoint validation."""
from pathlib import Path
import hashlib
import json
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    c = json.loads(Path(path).read_text('utf-8'))
    for stage in ('text', 'final'):
        k = c['kernels'][stage]
        if set(k) != {'beta','lambda','tau','alpha'} or not all(np.isfinite(v) for v in k.values()):
            raise ValueError('each kernel requires finite beta/lambda/tau/alpha')
        if k['beta'] <= 0 or k['lambda'] <= 0 or k['tau'] < 0:
            raise ValueError('invalid kernel parameters')
    if not 0 <= c['fusion']['rho'] <= 1:
        raise ValueError('invalid fusion rho')
    t,h,j=c['text_training'],c['head_training'],c['joint_training']
    if not (1<=t['retained_epoch']<=t['epochs'] and t['batch_size']>=1 and h['steps']>=1 and j['steps']>=1):
        raise ValueError('invalid training budget or retained epoch')
    if h.get('gamma')!=100:
        raise ValueError('this retained recipe fixes gamma at 100')
    for value in (t['lr'],t['eps'],j['head_lr_factor'],j['text_lr'],j['text_eps']):
        if not np.isfinite(value) or value<=0:
            raise ValueError('learning rates and epsilons must be positive finite values')
    if any(not np.isfinite(v) or v<0 for v in (t['weight_decay'],j['text_weight_decay'],c['fusion']['gda_alpha'])):
        raise ValueError('invalid weight decay or GDA alpha')
    return c


def load_episode(path, *, data_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text('utf-8'))
    if manifest.get('format') != 'ckir-gda-episode-v1':
        raise ValueError('unsupported episode manifest format')
    root = Path(data_root).resolve() if data_root else path.parent
    asset = manifest['arrays']
    relative = Path(asset['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('array asset path must be relative to the explicit data root')
    p = root / relative
    if digest(p) != asset['sha256']:
        raise ValueError('episode arrays SHA-256 mismatch')
    with np.load(p, allow_pickle=False) as z:
        arrays = {n:z[n].copy(order='K') for n in z.files}
    required = ('support','query','observed_labels','template','cupl','gda_text')
    if any(n not in arrays for n in required):
        raise ValueError('episode arrays miss a required field')
    s,q = arrays['support'],arrays['query']
    if s.ndim != 2 or q.ndim != 2 or s.shape[1] != q.shape[1] or min(len(s),len(q)) == 0:
        raise ValueError('support/query must be non-empty [rows,feature] arrays')
    d,c = s.shape[1],len(manifest['class_names'])
    if any(arrays[n].shape != (d,c) for n in ('template','cupl','gda_text')):
        raise ValueError('text columns must match manifest class order')
    o = arrays['observed_labels']
    if o.shape != (len(s),) or not np.issubdtype(o.dtype,np.integer) or np.any(o<0) or np.any(o>=c):
        raise ValueError('invalid observed labels')
    for name,x in arrays.items():
        if not np.issubdtype(x.dtype,np.number) or not np.isfinite(x).all():
            raise ValueError(f'{name} must be a finite numeric array')
    si,qi = manifest['support_ids'],manifest['query_ids']
    if len(si)!=len(s) or len(qi)!=len(q) or len(set(si))!=len(si) or len(set(qi))!=len(qi) or set(si)&set(qi):
        raise ValueError('support/query identifiers must be unique, disjoint and match array rows')
    if len(set(manifest['class_names']))!=c or c<2:
        raise ValueError('at least two unique ordered class names are required')
    fingerprint=hashlib.sha256()
    for name in ('support','observed_labels','template','cupl'):
        a=np.ascontiguousarray(arrays[name])
        fingerprint.update(name.encode())
        fingerprint.update(str((a.dtype.str,a.shape)).encode())
        fingerprint.update(a.tobytes())
    manifest['support_fingerprint']=fingerprint.hexdigest()
    return manifest,arrays


def save_checkpoint(path, arrays, metadata):
    path = Path(path)
    if path.suffix!='.npz':
        raise ValueError('checkpoint output must end in .npz')
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f'refusing to overwrite checkpoint: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    meta=dict(metadata,sha256=digest(path),format='ckir-gda-checkpoint-v1')
    path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n','utf-8')


def load_checkpoint(path, manifest, config):
    path=Path(path)
    metadata=json.loads(path.with_suffix('.json').read_text('utf-8'))
    if metadata.get('format')!='ckir-gda-checkpoint-v1' or metadata['sha256']!=digest(path):
        raise ValueError('checkpoint format/hash mismatch')
    for key in ('class_names','support_ids','episode_id','support_fingerprint'):
        if metadata[key]!=manifest[key]:
            raise ValueError(f'checkpoint/episode mismatch: {key}')
    if metadata['kernels']!=config['kernels']:
        raise ValueError('checkpoint kernel/alpha settings differ from selected configuration')
    with np.load(path,allow_pickle=False) as z:
        state={n:z[n].copy(order='K') for n in z.files}
    if any(not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all() for a in state.values()):
        raise ValueError('checkpoint parameters must be finite numeric arrays')
    return state,metadata
