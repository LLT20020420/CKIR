"""Package-local retained three-stage training and fixed score inference.

Training receives observed support labels only. Query labels are not inputs.
"""
from copy import deepcopy
import numpy as np
from scipy.linalg import eigvalsh
import torch
from torch.nn import functional as F
from .models import CKIRText, LPHead
from .losses import SmoothCrossEntropy, adaptive_reranking_loss
from .geometry import coefficients
from .gda import fit_gda, fuse_scores
from .io import load_checkpoint, save_checkpoint


def setup(device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available; choose --device cpu for portable checks')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def tensor(x, device, dtype=torch.float64):
    return torch.as_tensor(x, device=device, dtype=dtype)


def weights(arrays, manifest, config, *, stage, target):
    key = f'{stage}_{target}_weights'
    k = config['kernels'][stage]
    if key in arrays:
        expected = {n:k[n] for n in ('beta','lambda','tau')}
        if manifest.get('cached_geometry',{}).get(stage) != expected:
            raise ValueError(f'{key} lacks matching beta/lambda/tau cache metadata')
        w = np.asarray(arrays[key], dtype=np.float64)
        count = len(arrays['support']) if target == 'support' else len(arrays['query'])
        if w.shape != (len(arrays['support']),count) or not np.isfinite(w).all():
            raise ValueError(f'invalid {key} shape/values')
        if np.max(abs(w), initial=0.) > k['tau'] + 1e-10:
            raise ValueError(f'{key} violates the configured influence bound')
        return w, {'source':'episode cache', 'stage':stage}
    q = arrays['support'] if target == 'support' else arrays['query']
    return coefficients(arrays['support'], q, k, stage=stage, support_targets=target=='support')


def text_model(arrays, device, state=None):
    model = CKIRText(tensor(arrays['template'],device,torch.float16), tensor(arrays['cupl'],device,torch.float16))
    if state is not None:
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name not in state or tuple(state[name].shape)!=tuple(p.shape):
                    raise ValueError(f'missing/wrong checkpoint field: {name}')
                p.copy_(tensor(state[name],device,p.dtype))
    return model


def head_model(arrays, device, state=None):
    d,c = arrays['template'].shape
    model = LPHead(c,d,device=device)
    if state is not None:
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name not in state or tuple(state[name].shape)!=tuple(p.shape):
                    raise ValueError(f'missing/wrong checkpoint field: {name}')
                p.copy_(tensor(state[name],device,p.dtype))
    if not torch.all(model.gamma==100):
        raise ValueError('the retained recipe keeps gamma fixed at 100')
    return model


def snapshot(text, head=None):
    state = {n:p.detach().cpu().numpy().copy() for n,p in text.named_parameters()}
    if head is not None:
        state.update({n:p.detach().cpu().numpy().copy() for n,p in head.named_parameters()})
    with torch.no_grad():
        state['text_features'] = text.positive().t().cpu().numpy().copy()
    return state


def learning_rate(support):
    s = np.asarray(support,dtype=np.float64)
    largest = float(eigvalsh(s @ s.T,subset_by_index=[len(s)-1,len(s)-1])[0])
    if largest<=0:
        raise ValueError('support Gram has no positive largest eigenvalue')
    return 4*len(s)/largest


def train_text(arrays, manifest, config, device):
    recipe = config['text_training']
    epochs,retained = recipe['epochs'],recipe['retained_epoch']
    if not 1<=retained<=epochs:
        raise ValueError('retained text epoch must be within the training budget')
    seed = 234000+100*int(manifest['support_seed'])+int(manifest['label_version'])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    model = text_model(arrays,device)
    support = tensor(arrays['support'],device)
    labels = tensor(arrays['observed_labels'],device,torch.long)
    one = F.one_hot(labels,arrays['template'].shape[1]).double()
    w,audit = weights(arrays,manifest,config,stage='text',target='support')
    w = tensor(w,device)
    # Historical DataLoader shuffles the stable label-sorted row-ID array.
    order = np.argsort(arrays['observed_labels'],kind='stable')
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(order,dtype=torch.long)),
        batch_size=recipe['batch_size'],shuffle=True,drop_last=False)
    # Original cached image view: fp16 conversion, then one row normalization.
    sx = tensor(arrays['support'],device,torch.float16)
    sx = sx/sx.norm(dim=-1,keepdim=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=recipe['lr'],eps=recipe['eps'],weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs*len(loader))
    ce = SmoothCrossEntropy()
    trace=[]
    for epoch in range(1,epochs+1):
        total = np.zeros(4)
        for (ids,) in loader:
            b=ids.to(device)
            ps=model.probability(support)
            score=ps[b]+config['kernels']['text']['alpha']*w[:,b].t()@(one-ps)
            cls=ce(score,labels[b])
            rank=adaptive_reranking_loss(sx[b],model.positive().t(),labels[b],scale=4.,knn=3)
            cons=model.consistency()
            loss=cls+rank+8*cons
            optimizer.zero_grad();loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('non-finite text-stage loss/gradient')
            optimizer.step();scheduler.step()
            total+=np.array([float(x.detach()) for x in (loss,cls,rank,cons)])*len(b)
        trace.append({'epoch':epoch,'loss':float(total[0]/len(support)),'CE':float(total[1]/len(support)),
                      'rerank':float(total[2]/len(support)),'consistency':float(total[3]/len(support))})
        if epoch==retained:
            retained_state=snapshot(model)
    return retained_state, {'trace':trace,'retained_epoch':retained,'fit_seed':seed,'geometry':audit,
                            'optimizer_updates':epochs*len(loader),'query_labels_used':False}


def train_head(arrays,manifest,config,device,initial):
    text=text_model(arrays,device,initial)
    for p in text.parameters():p.requires_grad_(False)
    head=head_model(arrays,device)
    support=tensor(arrays['support'],device)
    labels=tensor(arrays['observed_labels'],device,torch.long)
    one=F.one_hot(labels,arrays['template'].shape[1]).double()
    w,audit=weights(arrays,manifest,config,stage='final',target='support');w=tensor(w,device)
    with torch.no_grad(): T=text.positive().double();sl=support@T
    lr=learning_rate(arrays['support'])
    optimizer=torch.optim.SGD([head.weight,head.bias],lr=lr,momentum=.9)
    trace=[]
    for step in range(1,config['head_training']['steps']+1):
        ps=head.probability(support,sl,mode='aligned')
        loss=F.cross_entropy(ps+config['kernels']['final']['alpha']*w.t()@(one-ps),labels)
        optimizer.zero_grad();loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in (head.weight,head.bias)):
            raise RuntimeError('non-finite aligned-head loss/gradient')
        optimizer.step();trace.append({'step':step,'CE_before_update':float(loss.detach())})
    return snapshot(text,head),{'trace':trace,'lr_W':lr,'geometry':audit,'query_labels_used':False}


def train_joint(arrays,manifest,config,device,initial):
    text=text_model(arrays,device,initial);head=head_model(arrays,device,initial)
    support=tensor(arrays['support'],device)
    labels=tensor(arrays['observed_labels'],device,torch.long)
    one=F.one_hot(labels,arrays['template'].shape[1]).double()
    w,audit=weights(arrays,manifest,config,stage='final',target='support');w=tensor(w,device)
    recipe=config['joint_training']
    lr=recipe['head_lr_factor']*learning_rate(arrays['support'])
    oh=torch.optim.SGD([head.weight,head.bias],lr=lr,momentum=.9)
    ot=torch.optim.AdamW(text.parameters(),lr=recipe['text_lr'],eps=recipe['text_eps'],weight_decay=recipe['text_weight_decay'])
    trace=[]
    for step in range(1,recipe['steps']+1):
        T=text.positive().double();ps=head.probability(support,support@T,mode='joint')
        loss=F.cross_entropy(ps+config['kernels']['final']['alpha']*w.t()@(one-ps),labels)
        oh.zero_grad();ot.zero_grad();loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in [head.weight,head.bias,*text.parameters()]):
            raise RuntimeError('non-finite joint-stage loss/gradient')
        oh.step();ot.step()
        trace.append({'step':step,'CE_before_update':float(loss.detach())})
    return snapshot(text,head),{'trace':trace,'head_lr':lr,'text_lr':recipe['text_lr'],'geometry':audit,
                               'shared_pre_update_backward':True,'query_labels_used':False}


def train_stage(stage,arrays,manifest,config,device,output,initial_path=None):
    if stage not in ('text','head','joint'):
        raise ValueError('invalid training stage')
    if stage=='text':
        if initial_path is not None:raise ValueError('text stage starts from zero residuals')
        state,audit=train_text(arrays,manifest,config,device)
    else:
        if initial_path is None:raise ValueError(f'{stage} stage requires --init')
        initial,meta=load_checkpoint(initial_path,manifest,config)
        expected='text' if stage=='head' else 'head'
        if meta['stage']!=expected:raise ValueError(f'{stage} must start from a {expected} checkpoint')
        fn=train_head if stage=='head' else train_joint
        state,audit=fn(arrays,manifest,config,device,initial)
    metadata={k:deepcopy(manifest[k]) for k in ('episode_id','class_names','support_ids','support_fingerprint')}
    metadata.update(stage=stage,kernels=deepcopy(config['kernels']),recipe=deepcopy(config),
                    episode_arrays_sha256=manifest['arrays']['sha256'],audit=audit,
                    head_mode='aligned' if stage=='head' else 'joint',
                    synthetic=manifest.get('kind')=='synthetic')
    save_checkpoint(output,state,metadata)
    return state,metadata


def infer(arrays,manifest,config,device,checkpoint):
    state,meta=load_checkpoint(checkpoint,manifest,config)
    support,query=tensor(arrays['support'],device),tensor(arrays['query'],device)
    c=arrays['template'].shape[1]
    if state['text_features'].shape!=(c,support.shape[1]):
        raise ValueError('stored learned-text shape mismatch')
    T=tensor(state['text_features'].T,device)
    one=F.one_hot(tensor(arrays['observed_labels'],device,torch.long),c).double()
    stage='text' if meta['stage']=='text' else 'final'
    w,audit=weights(arrays,manifest,config,stage=stage,target='query');w=tensor(w,device)
    with torch.no_grad():
        if meta['stage']=='text':
            # Stage-one CKIR probability multiplies features before the text dot product.
            ps=torch.softmax((100.*support)@T,dim=-1)
            pq=torch.softmax((100.*query)@T,dim=-1)
        else:
            head=head_model(arrays,device,state)
            ps=head.probability(support,support@T,mode=meta['head_mode'])
            pq=head.probability(query,query@T,mode=meta['head_mode'])
        cs=pq+config['kernels'][stage]['alpha']*w.t()@(one-ps)
        gda=fit_gda(arrays['support'],arrays['observed_labels'],arrays['gda_text'],alpha=config['fusion']['gda_alpha'])
        gs=gda.scores(arrays['query'])
        fused=fuse_scores(cs.cpu(),gs,rho=config['fusion']['rho'])
    result={'ckir_scores':cs.cpu().numpy(),'gda_scores':gs.numpy(),'fused_scores':fused.numpy(),
            'ckir_predictions':cs.argmax(1).cpu().numpy(),'gda_predictions':gs.argmax(1).numpy(),
            'fused_predictions':fused.argmax(1).numpy()}
    return result,{'geometry':audit,'kernel':config['kernels'][stage], 'fusion':config['fusion'],
                  'gda_native_dtype':'float32','fusion_dtype':'float64','query_labels_used':False}
