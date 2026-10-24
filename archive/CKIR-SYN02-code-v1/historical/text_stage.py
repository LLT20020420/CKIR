"""CKIR-trained positive text residuals, one checkpoint and two inference heads."""
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.linalg import cho_factor,cho_solve
from model import CKIRText,source_losses,author_frozen_components,author_score

ROOT=Path('D:/CodexResearch/Projects/RobustCLIP');HERE=Path(__file__).parent;OUT=HERE/'results'
MATCH=ROOT/'analysis/v05-ckir-matched-text-20261021/results'
EXPLAIN=ROOT/'analysis/v05-simnl-explanation-20261021/results'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
sim=load('source_simnl_runner',ROOT/'analysis/v05-simnl-mixture-20261021/run_mix.py')
u=sim.u;read,save,sha,now=u.read,u.save,u.sha,u.now
core=load('ckir_core',ROOT/'core.py')
NAMES=['trained_ckir_head','trained_simnl_head']

def setup():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def support_weights(s,path):
    start=time.perf_counter();g=s.astype(np.float64);g=g/np.linalg.norm(g,axis=1,keepdims=True)
    gram,jitter=core.regularized_kernel(g,2.);a=gram+np.eye(296)
    w0=cho_solve(cho_factor(a,lower=True),gram);solves=[core.solve_box(a,gram[:,j],.08,w0[:,j]) for j in range(296)]
    # Near a support sample, the original objective-change stopping criterion
    # can precede its own projected-gradient check. Solve the identified free
    # block exactly, then coordinate polish if necessary, without relaxing it.
    weights=[];pgs=[];refined=0
    for j,v in enumerate(solves):
        w=v.w.copy();k=gram[:,j]
        if not v.accepted:
            refined+=1;free=abs(w)<.08-1e-10;fixed=~free
            cand=w.copy();cand[free]=np.linalg.solve(a[np.ix_(free,free)],k[free]-a[np.ix_(free,fixed)]@w[fixed])
            if abs(cand).max()<=.08 and core.projected_gradient(cand,a,k,.08)<=v.pg_tolerance:w=cand
            else:
                for sweep in range(10000):
                    for i in range(296):w[i]=np.clip(w[i]+(k[i]-a[i]@w)/a[i,i],-.08,.08)
                    if core.projected_gradient(w,a,k,.08)<=v.pg_tolerance:break
        pg=core.projected_gradient(w,a,k,.08)
        assert pg<=v.pg_tolerance and abs(w).max()<=.08+1e-10
        assert core.objective(w,a,k)<=v.clip_objective+1e-8*(1+abs(v.clip_objective))
        weights.append(w);pgs.append(pg)
    w=np.stack(weights,axis=1)
    np.savez_compressed(path,wstar=w,projected_gradient=pgs,tolerance=[v.pg_tolerance for v in solves],jitter=jitter,refined_count=refined)
    return w,{'seconds':time.perf_counter()-start,'QP_count':296,'free_block_refinements':refined,'max_projected_gradient_ratio':max(p/v.pg_tolerance for p,v in zip(pgs,solves)),'jitter':jitter,'max_abs_weight':float(abs(w).max())}

def fit(gl,classes,capture,s,q,o,text,wt,wq,phase,seed,k):
    tseed=sim.seed_for(seed,k);torch.manual_seed(tseed);torch.cuda.manual_seed_all(tseed);np.random.seed(tseed)
    model=CKIRText(*text).cuda();ce,rerank=source_losses()
    sd=torch.as_tensor(s,device='cuda',dtype=torch.float64);qd=torch.as_tensor(q,device='cuda',dtype=torch.float64)
    one=F.one_hot(torch.as_tensor(o,device='cuda'),num_classes=37).double()
    trainw=torch.as_tensor(wt,device='cuda');queryw=torch.as_tensor(wq,device='cuda')
    comp=author_frozen_components(sim,gl,classes,capture,s,q,o,text)
    sx,order=comp[-2:];inverse=np.argsort(order);sx=sx[torch.as_tensor(inverse,device='cuda')]
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(order,dtype=torch.long)),batch_size=256,shuffle=True,drop_last=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,eps=1e-3,weight_decay=.1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,30*len(loader))
    states=[];preds=[];lossrows=[];audit={};start=time.perf_counter()
    for epoch in range(31):
        if epoch:
            totals=np.zeros(4);lrs=[]
            for (b_cpu,) in loader:
                b=b_cpu.cuda();lab=torch.as_tensor(o,device='cuda')[b]
                ps=model.probability(sd);scores=ps[b]+2.44*trainw[:,b].t()@(one-ps)
                cls=ce(scores,lab);rank=rerank(sx[b],model.positive().t(),lab);cons=model.consistency()
                loss=cls+rank+8*cons
                assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward()
                if epoch==1 and not audit:
                    grads={n:float(p.grad.float().norm()) for n,p in model.named_parameters()}
                    assert all(np.isfinite(v) and v>0 for v in grads.values())
                    audit={'gradient_norms_first_update':grads,'CE_vs_cross_entropy_error':float(abs(cls-F.cross_entropy(scores,lab))),
                           'train_rows':len(s),'trainable_parameters':sum(p.numel() for p in model.parameters()),'query_labels_used_in_optimizer':False}
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                lrs.append(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
                totals+=np.array([float(loss.detach()),float(cls.detach()),float(rank.detach()),float(cons.detach())])*len(b)
            lossrows.append({'epoch':epoch,'loss':totals[0]/296,'classification':totals[1]/296,'reranking':totals[2]/296,'consistency':totals[3]/296,'lr_steps':lrs})
        with torch.no_grad():
            a=model.ckir(sd,qd,queryw,one);b=author_score(model,comp)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            preds.append(np.stack([a.argmax(-1).cpu().numpy(),b.argmax(-1).cpu().numpy()]).astype(np.uint8))
            states.append({n:p.detach().cpu().numpy().copy() for n,p in model.named_parameters()})
    audit['final_parameter_norms']={n:float(p.float().norm()) for n,p in model.named_parameters()}
    assert all(v>0 for v in audit['final_parameter_norms'].values())
    audit['seconds']=time.perf_counter()-start
    np.savez_compressed(OUT/f'{phase}-weights-{seed}-{k}.npz',**{n:np.stack([st[n] for st in states]) for n in states[0]})
    save(OUT/f'{phase}-training-{seed}-{k}.json',{'phase':phase,'seed':seed,'label_version':k,'torch_seed':tseed,'trace':lossrows,'audit':audit})
    return np.stack(preds),audit

def main():
    if OUT.exists():raise RuntimeError('Fresh immutable run required')
    OUT.mkdir();setup();started=time.perf_counter()
    previous=read(MATCH/'frozen-plan-before-scoring.json');protected=dict(previous['protected_hashes'])
    files=[HERE/'model.py',Path(__file__),ROOT/'core.py',MATCH/'findings.json',EXPLAIN/'findings.json',sim.SNAP/'main.py',sim.SNAP/'utils.py',sim.HERE/'run_mix.py']
    for phase in ('development','pilot'):
        for e in previous['episodes'][phase]:files += [MATCH/f'{phase}-seed-{e["seed"]}.npz',EXPLAIN/f'{phase}-diagnostic-{e["seed"]}.npz']
    for name in ('template','cupl','negative_template'):files.append(sim.OUT/f'text_weights_{name}.pt')
    for p in files:protected[str(p)]=sha(p)
    for p,h in protected.items():assert sha(p)==h,p
    plan={'created_hk':now(),'authorization':'Train CKIR following SimNL; user explicitly selects same learned weights scored by CKIR and author SimNL formulas',
          'episodes':previous['episodes'],'feature_path':previous['feature_path'],'ckir_parameters':previous['ckir_parameters'],'protected_hashes':protected,
          'trainable':'only positive template and CuPL residuals,75776 parameters; frozen CLIP RN50, visual and negative residuals zero',
          'loss':'CE(CKIR raw residual-probability scores, observed support labels)+source adaptive reranking+8*source text consistency',
          'optimizer':{'AdamW':{'lr':1e-4,'eps':1e-3,'weight_decay':.1},'cosine_steps':60,'epochs':30,'batch_size':256,'drop_last':False},
          'training_geometry':'support-to-support bounded QP once per support draw, same beta2/lambda1/tau.08, includes self as native cache training; no derivative needed through frozen geometry',
          'solver_refinement':'original solve_box, then exact free-block/coordinate refinement only when original projected-gradient criterion fails; unchanged tolerance, objective and box constraints',
          'inference':'same learned positive text in both heads. CKIR2/1/.08/2.44; native fp16 SimNL6.72/1.05/.76; original author negative and visual branches initialized, fixed',
          'selection':'one common epoch1..30 by CKIR aggregate development noisy accuracy, tie fewer harmed then earliest; freeze before pilot fits. Both heads use that exact epoch. No head-specific grid search.',
          'scope':previous['scope']+'; training extension, not complete original SimNL retraining or new independent evaluation',
          'BGSR':'paused','official_test_encoded_or_scored':False}
    save(OUT/'frozen-plan-before-training.json',plan)
    text=[torch.load(sim.OUT/f'text_weights_{name}.pt',weights_only=True,map_location='cuda').detach() for name in ('template','cupl','negative_template')]
    gl,classes,capture=sim.sources()
    with np.load(plan['feature_path']) as z:ids=[str(v) for v in z['ids']];image=z['image_features'].astype(np.float64)
    idx={v:i for i,v in enumerate(ids)};manifest=read(ROOT/'runs/E0-20261019-v03/data-manifest.json');qr=manifest['split']['Qdev']
    qids=[r['id'] for r in qr];q=image[[idx[v] for v in qids]];y=np.array([r['label'] for r in qr])
    with np.load(MATCH/'pilot-seed-2000.npz') as z:zero=z['zero'].copy();assert np.array_equal(y,z['y'])
    good=zero==y;assert good.sum()==477
    save(OUT/'runtime-state.json',{'started_hk':now(),'stage':'development','y':y.tolist(),'zero':zero.tolist()})
    phasecounts={};tims=[];qpaudit=[];selected=None;zero_events=0
    for phase in ('development','pilot'):
        allcounts=[]
        for e in plan['episodes'][phase]:
            receipt=read(e['receipt']);lab=read(e['labels']);assert receipt['support_ids']==lab['support_ids'] and not set(qids)&set(receipt['support_ids'])
            obs=np.array([lab['clean_labels']]+[v['observed_labels'] for v in lab['noise']]);s=image[[idx[v] for v in receipt['support_ids']]]
            assert obs.shape==(4,296) and all((o!=obs[0]).sum()==74 for o in obs[1:])
            path=OUT/f'{phase}-support-weights-{e["seed"]}.npz';wt,qa=support_weights(s,path)
            qpaudit.append({'phase':phase,'seed':e['seed'],**qa})
            print(f'{phase} {e["seed"]}:296 training QPs {qa["seconds"]:.2f}s; training four label versions',flush=True)
            with np.load(e['qp_weights']) as z:wq=z['wstar'].copy()
            preds=[]
            for k,o in enumerate(obs):
                pred,audit=fit(gl,classes,capture,s,q,o,text,wt,wq,phase,e['seed'],k);preds.append(pred)
                tims.append({'phase':phase,'seed':e['seed'],'version':k,**audit})
            pred=np.stack(preds,axis=2) # epoch0..30,head,label_version,query
            with np.load(MATCH/f'{phase}-seed-{e["seed"]}.npz') as z:assert np.array_equal(pred[0,0],z['predictions'][1])
            with np.load(EXPLAIN/f'{phase}-diagnostic-{e["seed"]}.npz') as z:
                i=list(z['names']).index('untrained_full_fixed_hp');assert np.array_equal(pred[0,1],z['predictions'][i])
            zero_events+=pred[0].size
            cc=np.stack([np.stack([u.count(pred[ep,h],y,good) for h in range(2)]) for ep in range(31)])
            allcounts.append(cc)
            np.savez_compressed(OUT/f'{phase}-predictions-{e["seed"]}.npz',predictions=pred,counts=cc,y=y,zero=zero,names=NAMES)
            save(OUT/'runtime-state.json',{'updated_hk':now(),'stage':phase,'last_completed_seed':e['seed'],'elapsed_seconds':time.perf_counter()-started,'selected_epoch':selected})
            print(f'{phase} {e["seed"]}:four30-epoch fits complete',flush=True)
        arr=np.stack(allcounts);phasecounts[phase]=arr;rows=[]
        for epoch in range(31):
            for h,name in enumerate(NAMES):
                rates=arr[:,epoch,h,2]/1776
                rows.append({'epoch':epoch,'method':name,**u.metrics(arr[:,epoch,h].sum(0),len(arr)),
                             'accuracy_se':float(rates.std(ddof=1)/np.sqrt(len(arr)))})
        u.csv_out(OUT/f'{phase}-epoch-curve.csv',rows);np.savez_compressed(OUT/f'{phase}-counts.npz',counts=arr)
        if phase=='development':
            selected=min(range(1,31),key=lambda ep:(-arr[:,ep,0,2].sum(),arr[:,ep,0,1].sum(),ep))
            save(OUT/'selected-checkpoint-before-pilot.json',{'saved_hk':now(),'epoch':selected,'rule':plan['selection'],
                 'development_metrics':[r for r in rows if r['epoch']==selected]})
            print(f'COMMON CHECKPOINT FROZEN:epoch{selected}; pilot training begins',flush=True)
    results={}
    for phase,arr in phasecounts.items():
        results[phase]=[]
        for h,name in enumerate(NAMES):
            rates=arr[:,selected,h,2]/1776
            results[phase].append({'method':name,**u.metrics(arr[:,selected,h].sum(0),len(arr)),
                                  'accuracy_se':float(rates.std(ddof=1)/np.sqrt(len(arr)))})
    for p,h in protected.items():assert sha(p)==h,p
    old=read(MATCH/'findings.json');full=read(sim.OUT/'findings.json')
    f={'status':'COMPLETE_CKIR_TRAINING_TWO_SHARED_WEIGHT_HEADS','completed_hk':now(),'selected_epoch':selected,'seconds':time.perf_counter()-started,
       'development':results['development'],'pilot_trained_ckir':results['pilot'][0],'pilot_trained_simnl_head':results['pilot'][1],
       'pilot_matched_ckir_before_training':old['pilot_matched_ckir'],'pilot_original_ckir':old['pilot_original_ckir'],
       'pilot_full_original_SimNL_reference':full['pilot_simnl'],'QP_audit':qpaudit,'training_fits':88,'epochs_trained':2640,'optimizer_steps':5280,
       'new_training_QPs':6512,'trainable_parameters_per_fit':75776,'timings_and_gradient_audits':tims,
       'zero_update_prediction_events_reproduced':zero_events,'protected_files_unchanged':len(protected),
       'same_weights_two_heads':True,'head_hyperparameters_retuned':False,'scope':plan['scope'],'BGSR':'paused'}
    save(OUT/'findings.json',f);save(OUT/'runtime-state.json',{'stage':'TRAINING_COMPLETE','completed_hk':now(),'selected_epoch':selected})
    print(json.dumps({k:f[k] for k in ('status','selected_epoch','seconds','pilot_trained_ckir','pilot_trained_simnl_head')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
