"""Short joint continuation from aligned CKIR, with a head-only continuation control."""
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('D:/CodexResearch/Projects/RobustCLIP');HERE=Path(__file__).parent;OUT=HERE/'results'
BASE=ROOT/'analysis/v05-ckir-aligned-head-training-20261022'
TEXT=ROOT/'analysis/v05-ckir-simnl-training-20261021'
PARENT=ROOT/'analysis/v05-ckir-lpp-training-20261021'
TABLE=ROOT/'analysis/mixture-comparison-20261021'
sys.path.insert(0,str(PARENT))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
parent=load('joint_text_head_parent',PARENT/'run.py')
tm=load('joint_text_head_text',TEXT/'model.py')
u=parent.u;read,save,sha=u.read,u.save,u.sha
STEPS=7;TEXT_LR=1e-5;HEAD_LR_FACTOR=.1;ALPHA=2.51


def prepare():
    OUT.mkdir()
    old=read(BASE/'results/frozen-plan.json');paths=dict(old['protected_hashes'])
    # Current table and weights supersede the archived parent-plan table, not historical rows.
    paths[str(TABLE/'results/完整原始记录.json')]=sha(TABLE/'results/完整原始记录.json')
    for p in (Path(__file__),BASE/'results/findings.json',BASE/'results/verification.json',TEXT/'model.py',PARENT/'model.py'):
        paths[str(p)]=sha(p)
    geometries={}
    for phase in ('development','pilot'):
        for e in old['episodes'][phase]:
            seed=e['seed'];g=read(BASE/f'results/{phase}-support-geometry-{seed}.json')
            assert sha(g['path'])==g['sha256'];geometries[f'{phase}/{seed}']=g['path']
            paths[g['path']]=g['sha256']
            for k in range(4):
                for p in (BASE/f'results/{phase}-fit-{seed}-{k}.npz',TEXT/f'results/{phase}-weights-{seed}-{k}.npz'):
                    paths[str(p)]=sha(p)
    raw={}
    for name in ('template','cupl','negative_template'):
        p=ROOT/f'analysis/v05-simnl-mixture-20261021/results/text_weights_{name}.pt'
        paths[str(p)]=sha(p);raw[name]=str(p)
    p={'created_hk':u.now(),'authorization':'User: use the new version as starting point for option2, joint text-residual and W/b training',
       'episodes':old['episodes'],'feature_path':old['feature_path'],'scope':old['scope'],
       'query_weights':old['query_weights'],'support_weights':geometries,'raw_text':raw,
       'start':'Exact aligned-head epoch7 W/b/gamma and original SimNL-trained epoch25 template/CuPL residuals; zero updates must reproduce87.6013514%',
       'arms':['head-only continuation control','joint text residuals and W/b'],
       'steps':STEPS,'head_optimizer':'New SGD state,momentum.9,constant lr=.1*same fit data-derived original lr',
       'text_optimizer':'Joint only,new AdamW state,constant lr1e-5,eps1e-3,weight_decay.1',
       'loss':'Same single full-support CE through CKIR16/1/.03/2.51 for both arms; no auxiliary losses,MAE,OT or added regularization',
       'joint_update':'Compute both parameter-group gradients from same pre-update state,then step both optimizers once',
       'fixed':'CLIP encoders,image geometry,coupled bounded QPs,alpha2.51,gamma100,text mix.45/.55,all296 supports',
       'selection':'Fixed continuation step7 and learning rates before outcomes; no new checkpoint/grid/pilot choice',
       'protected_hashes':paths,'official_test_used':False,'new_QP_encoding_sampling':0,'BGSR':'paused'}
    save(OUT/'frozen-plan.json',p);return p


def fit(s,q,observed,wt,wq,raw,phase,seed,k,arm):
    text=tm.CKIRText(*raw).cuda();head=parent.LPHead().cuda()
    with np.load(TEXT/f'results/{phase}-weights-{seed}-{k}.npz') as z:
        for name,param in text.named_parameters():param.data.copy_(torch.tensor(z[name][25],device='cuda'))
    with np.load(BASE/f'results/{phase}-fit-{seed}-{k}.npz') as z:
        for name,param in head.named_parameters():param.data.copy_(torch.tensor(z[name][-1],device='cuda'))
        baseline=z['predictions'][-1].copy()
    for param in text.parameters():param.requires_grad_(bool(arm))
    head.gamma.requires_grad_(False)
    old_audit=read(BASE/f'results/{phase}-training-{seed}-{k}.json')['audit']
    lr=HEAD_LR_FACTOR*old_audit['lr_W']
    oh=torch.optim.SGD([head.weight,head.bias],lr=lr,momentum=.9)
    ot=torch.optim.AdamW(text.parameters(),lr=TEXT_LR,eps=1e-3,weight_decay=.1) if arm else None
    support=torch.tensor(s,device='cuda');query=torch.tensor(q,device='cuda')
    labels=torch.tensor(observed,device='cuda');one=F.one_hot(labels,37).double()
    tw,qw=torch.tensor(wt,device='cuda'),torch.tensor(wq,device='cuda')
    states=[];preds=[];texts=[];trace=[];audit={}
    def prob(features,T):return torch.softmax(features@head.weight.double().t()+head.bias.double()+head.gamma*(features@T),dim=-1)
    for step in range(STEPS+1):
        if step:
            T=text.positive().double();ps=prob(support,T)
            score=ps+ALPHA*tw.t()@(one-ps);loss=F.cross_entropy(score,labels)
            oh.zero_grad()
            if ot:ot.zero_grad()
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in (head.weight,head.bias))
            if step==1:
                audit['first_head_gradient_norm']=float(head.weight.grad.norm())
                if arm:
                    audit['first_text_gradient_norms']={n:float(p.grad.float().norm()) for n,p in text.named_parameters()}
                    assert all(x>0 and np.isfinite(x) for x in audit['first_text_gradient_norms'].values())
            if ot:assert all(torch.isfinite(p.grad).all() for p in text.parameters())
            oh.step()
            if ot:ot.step()
            trace.append({'step':step,'CE_before_update':float(loss.detach())})
        with torch.no_grad():
            T=text.positive();ps=prob(support,T.double());pq=prob(query,T.double())
            scores=pq+ALPHA*qw.t()@(one-ps);assert torch.isfinite(scores).all()
            pred=scores.argmax(-1).cpu().numpy().astype(np.uint8)
            if step==0:assert np.array_equal(pred,baseline),(phase,seed,k,arm)
            preds.append(pred);texts.append(T.t().cpu().numpy().copy())
            states.append({**{n:p.detach().cpu().numpy().copy() for n,p in text.named_parameters()},
                           **{n:p.detach().cpu().numpy().copy() for n,p in head.named_parameters()}})
    np.savez_compressed(OUT/f'{phase}-fit-{seed}-{k}-arm{arm}.npz',predictions=np.stack(preds),
        text_features=np.stack(texts),steps=np.arange(STEPS+1),**{n:np.stack([v[n] for v in states]) for n in states[0]})
    audit.update({'start_predictions_exactly_match87_6014':True,'steps':STEPS,'head_lr':lr,
        'text_lr':TEXT_LR if arm else 0.,'text_parameters_updated':bool(arm),'gamma_fixed100':True,
        'query_labels_used_in_optimizer':False,
        'text_residual_change_norm':float(sum(np.linalg.norm(states[-1][n].astype(float)-states[0][n].astype(float)) for n in ('res_template','res_cupl'))),
        'head_weight_change_norm':float(np.linalg.norm(states[-1]['weight']-states[0]['weight']))})
    save(OUT/f'{phase}-training-{seed}-{k}-arm{arm}.json',{'trace':trace,'audit':audit})
    return np.stack(preds)


def main():
    parent.setup();p=prepare();started=time.perf_counter()
    images,idx,qids,q,y,zero=parent.data(p);good=zero==y
    raw=[torch.load(p['raw_text'][n],weights_only=True,map_location='cuda').detach() for n in ('template','cupl','negative_template')]
    phase_metrics={};pairs_baseline=[];pairs_control=[];fits=0
    for phase in ('development','pilot'):
        counts=[]
        for e in p['episodes'][phase]:
            seed=e['seed'];s,obs,_,_,_=parent.inputs(e,phase,images,idx,qids)
            with np.load(p['support_weights'][f'{phase}/{seed}']) as z:wt=z['wstar'].copy()
            with np.load(p['query_weights'][f'{phase}/{seed}']) as z:wq=z['wstar'].copy()
            byarm=[]
            for arm in range(2):
                pred=np.stack([fit(s,q,o,wt,wq,raw,phase,seed,k,arm) for k,o in enumerate(obs)],axis=1)
                byarm.append(pred);fits+=4
            pred=np.stack(byarm);assert np.array_equal(pred[0,0],pred[1,0])
            cc=np.stack([[u.count(v,y,good) for v in bystep] for bystep in pred]);counts.append(cc)
            with np.load(BASE/f'results/{phase}-predictions-{seed}.npz') as z:assert np.array_equal(pred[0,0],z['predictions'][-1])
            np.savez_compressed(OUT/f'{phase}-predictions-{seed}.npz',predictions=pred,counts=cc,steps=np.arange(STEPS+1),y=y,zero=zero)
            if phase=='pilot':
                for target,change in ((pairs_baseline,cc[1,-1]-cc[1,0]),(pairs_control,cc[1,-1]-cc[0,-1])):
                    target.append(dict(zip(('seed','fixed_change','harmed_change','net_correct_change','clean_correct_change'),[seed,*map(int,change)])))
            save(OUT/'runtime-state.json',{'stage':phase,'completed_arm_fits':fits,'total_arm_fits':176,
                'last_seed':seed,'seconds':time.perf_counter()-started})
            print(f'{phase} seed{seed}:head-only {cc[0,-1,2]}/1776,joint {cc[1,-1,2]}/1776;{fits}/176 arm fits',flush=True)
        arr=np.stack(counts);np.savez_compressed(OUT/f'{phase}-counts.npz',counts=arr)
        phase_metrics[phase]=[u.metrics(arr[:,a,-1].sum(0),len(arr)) for a in range(2)]
        rows=[{'arm':arm,'step':step,**u.metrics(arr[:,arm,step].sum(0),len(arr))} for arm in range(2) for step in range(STEPS+1)]
        u.csv_out(OUT/f'{phase}-all-steps.csv',rows)
    control,joint=phase_metrics['pilot'];baseline=read(BASE/'results/findings.json')['pilot_aligned_ckir']
    def diff(reference,pairs):
        return {'net_correct_change':joint['noisy_correct']-reference['noisy_correct'],
            'accuracy_delta_pp':100*(joint['noisy_accuracy']-reference['noisy_accuracy']),
            'fixed_change':joint['fixed']-reference['fixed'],'harmed_change':joint['harmed']-reference['harmed'],
            'clean_correct_change':joint['clean_correct']-reference['clean_correct'],
            'positive_blocks':sum(v['net_correct_change']>0 for v in pairs),'negative_blocks':sum(v['net_correct_change']<0 for v in pairs),
            'tied_blocks':sum(v['net_correct_change']==0 for v in pairs),'per_support_draw':pairs}
    assert all(sha(path)==h for path,h in p['protected_hashes'].items())
    f={'status':'COMPLETE_JOINT_TEXT_HEAD_CONTINUATION','completed_hk':u.now(),'seconds_main':time.perf_counter()-started,
       'pilot_joint_ckir':joint,'pilot_head_only_control':control,'pilot_previous_best_ckir':baseline,
       'development_joint_ckir':phase_metrics['development'][1],'development_head_only_control':phase_metrics['development'][0],
       'paired_pilot_difference':diff(baseline,pairs_baseline),'paired_joint_vs_head_only':diff(control,pairs_control),
       'continuation_steps':STEPS,'head_fits':176,'joint_text_fits':88,'head_updates':1232,'text_updates':616,
       'protected_files_unchanged':len(p['protected_hashes']),'scope':p['scope'],'official_test_used':False,'new_QP_encoding_sampling':0}
    save(OUT/'findings.json',f);save(OUT/'runtime-state.json',{'stage':'COMPLETE','completed_hk':u.now()})
    print(json.dumps(f,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
