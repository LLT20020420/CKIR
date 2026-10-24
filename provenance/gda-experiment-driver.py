"""Source-faithful GDA selection and unchanged CKIR/native-GDA score mixing."""
import ast
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch

ROOT=Path('D:/CodexResearch/Projects/RobustCLIP')
HERE=Path(__file__).parent
OUT=HERE/'results'
ORIGINAL=ROOT/'analysis/v05-strong-mixture-20261021/results'
PREVIOUS=ROOT/'analysis/v05-tip-mixture-20261021/results'
PRO=ROOT/'analysis/v05-proker-mixture-20261021/results'
UP=ROOT/'third_party/ProKeR/trainers/gda.py'
ALPHAS=[.0001,.001,.01,.1,1.,10.,100.]
RATIOS=np.arange(101)/100

spec=importlib.util.spec_from_file_location('mix_common',ORIGINAL.parent/'run_mix.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
read,save,sha,now=u.read,u.save,u.sha,u.now


def source_check():
    author=ast.parse((HERE/'source-snapshot/gda-author-main_few_shots.py').read_text())
    local=ast.parse(UP.read_text())
    names=('mus','center_vecs','cov_inv','ps','W','b')
    def assignments(tree):
        return {n.targets[0].id:ast.unparse(n.value) for n in ast.walk(tree)
                if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in names}
    a,b=assignments(author),assignments(local)
    for k in names:assert a[k].replace('.cuda()', '.to(device)')==b[k],k
    (HERE/'source-snapshot/gda-proker-unmodified.py').write_bytes(UP.read_bytes())
    save(HERE/'source-snapshot/formula-parity.json',{'status':'PASS','six_estimator_assignments_identical':list(names),
        'only_device_difference':'original .cuda() versus supplied-tensor .to(device)',
        'local_function_modified':False,'author_commit':read(HERE/'source-snapshot/upstream-commit.json')})
    capture={'scores':[], 'selected':0}
    def cls_acc(scores,labels):
        i=len(capture['scores']);capture['scores'].append(scores.detach().cpu().numpy().copy())
        return 100. if i==capture['selected'] else 0.
    gl={'torch':torch,'cls_acc':cls_acc}
    node=next(n for n in local.body if isinstance(n,ast.FunctionDef) and n.name=='GDA')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(UP),'exec'),gl)
    return gl['GDA'],capture


def call_native(fn,capture,s,q,text,labels,selected=0,development=False):
    capture['scores']=[];capture['selected']=selected
    v=q if development else q[:1]
    with torch.no_grad(),contextlib.redirect_stdout(io.StringIO()):
        ans=fn(torch.from_numpy(s.astype(np.float32)),torch.from_numpy(labels.astype(np.int64)),
               torch.from_numpy(v.astype(np.float32)),torch.zeros(len(v),dtype=torch.long),
               torch.from_numpy(q.astype(np.float32)),torch.from_numpy(text.T.astype(np.float32)),
               'oxford_pets',8,0,'fixed')
    assert len(capture['scores'])==7
    assert np.isfinite(ans.numpy()).all()
    return np.stack(capture['scores']) if development else ans.numpy().copy()


def numpy_reference(s,q,text,labels,alpha):
    mus=np.stack([s[labels==i].mean(0) for i in range(37)])
    center=np.concatenate([s[labels==i]-mus[i] for i in range(37)])
    cov=np.cov(center,rowvar=False,ddof=1)
    precision_times_mu=1024*cho_solve(cho_factor((len(s)-1)*cov+np.trace(cov)*np.eye(1024),lower=True),mus.T)
    bias=-np.log(37)-.5*np.einsum('ij,ji->i',mus,precision_times_mu)
    return (100*q)@text.T+alpha*(q@precision_times_mu+bias)


def main():
    started=time.perf_counter();torch.set_num_threads(1)
    if OUT.exists():raise RuntimeError('Preserve existing results')
    fn,capture=source_check()
    prior=read(ORIGINAL/'frozen-plan-before-mixing.json')
    protected=dict(prior['protected_hashes'])
    extras=[Path(__file__),HERE/'figure-contract.md',UP,ORIGINAL/'frozen-plan-before-mixing.json',
        PREVIOUS/'findings.json',PRO/'findings.json']+list((HERE/'source-snapshot').iterdir())
    extras += [PREVIOUS/f'{phase}-seed-{e["seed"]}-all-predictions.npz'
               for phase in ('development','pilot') for e in prior['episodes'][phase]]
    for p in extras:protected[str(p)]=sha(p)
    for p,h in protected.items():assert sha(p)==h,p
    OUT.mkdir()
    plan={'frozen_hk':now(),'authorization':'用户同意先补GDA单独成绩，再按原101比例方式缝合CKIR与GDA',
        'ckir_parameters':prior['ckir_parameters'],'gda_native_alpha_grid':ALPHAS,'zero_adaptation_control':True,
        'gda_selection':'12 development blocks, maximum noisy_correct; ties fewer harmed then lower alpha. Native nonzero choice retained separately from zero-inclusive choice.',
        'selection':'Development noisy_correct highest; ties fewer harmed then smaller CKIR percentage. Freeze before pilot; report seen-pilot peak separately.',
        'formula':'rho*CKIR_native_score+(1-rho)*GDA_native_score','percentages':list(range(101)),
        'gda_formula':'100*(q@text.T)+alpha*(q@W+b); W=precision@mu, b=log(1/37)-mu.T@precision@mu/2; KS precision as author source',
        'scaling':'Author native torch.float32 GDA scores promoted to float64, unchanged CKIR float64 scores; no new normalization, temperature or softmax.',
        'scope':'Original12 development/10 seen pilot support draws, same592query/frozenRN50/8shot/25%noise/3realizations and clean; same single-view inputs, not original multi-augmentation benchmark reproduction; exploratory.',
        'episodes':prior['episodes'],'feature_path':prior['feature_path'],'protected_hashes':protected,
        'new_QP_training_encoding_sampling_test':0,'new_GDA_estimator_fits':88,'BGSR':'paused'}
    save(OUT/'frozen-plan-before-gda-selection.json',plan)
    sys.path.insert(0,str(ROOT));from e1_analysis import probabilities
    with np.load(plan['feature_path'],allow_pickle=False) as z:
        ids=[str(x) for x in z['ids']];image=z['image_features'].astype(np.float64);text=z['text_features'].astype(np.float64)
        prob=probabilities(image,text,float(z['logit_scale']))
    idx={s:i for i,s in enumerate(ids)}
    qrows=read(ROOT/'runs/E0-20261019-v03/data-manifest.json')['split']['Qdev']
    qids=[r['id'] for r in qrows];qi=[idx[s] for s in qids];q=image[qi];y=np.array([r['label'] for r in qrows])
    zero=prob[qi].argmax(1);good=zero==y
    assert len(y)==592 and good.sum()==477
    zero_scores=((100*torch.from_numpy(q.astype(np.float32)))@torch.from_numpy(text.T.astype(np.float32))).numpy()
    assert np.array_equal(zero_scores.argmax(1),zero)

    def inputs(e):
        rec=read(e['receipt']);lab=read(e['labels'])
        assert rec['support_ids']==lab['support_ids'] and not set(qids)&set(rec['support_ids'])
        observed=np.array([lab['clean_labels']]+[n['observed_labels'] for n in lab['noise']])
        assert observed.shape==(4,296) and all((o!=observed[0]).sum()==74 for o in observed[1:])
        assert all(np.bincount(o,minlength=37).min()>0 for o in observed)
        s=image[[idx[v] for v in rec['support_ids']]]
        return s,observed

    gridcounts=[]
    for e in plan['episodes']['development']:
        s,observed=inputs(e)
        native=np.stack([call_native(fn,capture,s,q,text,o,development=True) for o in observed],axis=1)
        scores=np.concatenate([np.broadcast_to(zero_scores,(1,4,592,37)),native],axis=0)
        predictions=scores.argmax(-1).astype(np.uint8)
        counts=np.stack([u.count(p,y,good) for p in predictions]);gridcounts.append(counts)
        np.savez_compressed(OUT/f'development-gda-grid-{e["seed"]}.npz',scores=scores,predictions=predictions,
                            alphas=np.array([0.]+ALPHAS),counts=counts)
        print(f'development {e["seed"]}: 7 native GDA parameters + zero control',flush=True)
    gc=np.stack(gridcounts);gridrows=[]
    for i,alpha in enumerate([0.]+ALPHAS):
        gridrows.append({'alpha':alpha,**u.metrics(gc[:,i].sum(0),12)})
    native_idx=min(range(1,8),key=lambda i:(-gridrows[i]['noisy_correct'],gridrows[i]['harmed'],i))
    inclusive=u.choose(gridrows);selected_alpha=gridrows[native_idx]['alpha']
    u.csv_out(OUT/'development-gda-native-grid.csv',gridrows)
    np.savez_compressed(OUT/'development-gda-grid-counts.npz',counts=gc,alphas=[0.]+ALPHAS)
    gda_selection={'saved_hk':now(),'selected_native':gridrows[native_idx],
        'selected_zero_inclusive':gridrows[inclusive],'rule':plan['gda_selection']}
    save(OUT/'gda-configuration-before-mixing-and-pilot.json',gda_selection)
    plan['gda_parameters']={'alpha':selected_alpha,'native_rule_index':native_idx}
    plan['frozen_gda_hk']=now()
    save(OUT/'frozen-plan-before-mixing.json',plan)

    results={};arrays={};dev_idx=None;events=0;maxerr=0.;refdiffs=0;refevents=0;referr=0.
    for phase in ('development','pilot'):
        blocks=[];blockrows=[]
        for e in plan['episodes'][phase]:
            s,observed=inputs(e)
            with np.load(PREVIOUS/f'{phase}-seed-{e["seed"]}-all-predictions.npz',allow_pickle=False) as z:
                qs=z['ckir_native_scores'].copy();assert np.array_equal(z['y'],y) and np.array_equal(z['zero'],zero)
                assert np.array_equal(qs.argmax(-1),z['predictions'][100])
            if phase=='development':
                with np.load(OUT/f'development-gda-grid-{e["seed"]}.npz',allow_pickle=False) as z:score=z['scores'][native_idx].astype(np.float64)
            else:
                score=np.stack([call_native(fn,capture,s,q,text,o,selected=native_idx-1) for o in observed]).astype(np.float64)
            reference=np.stack([numpy_reference(s,q,text,o,selected_alpha) for o in observed])
            refdiffs+=int((reference.argmax(-1)!=score.argmax(-1)).sum());refevents+=4*592
            referr=max(referr,float(np.max(np.abs(reference-score))))
            predictions=[];cc=[]
            for rho in RATIOS:
                mixed=rho*qs+(1-rho)*score;other=score+rho*(qs-score)
                yp=mixed.argmax(-1).astype(np.uint8)
                assert np.array_equal(yp,other.argmax(-1))
                maxerr=max(maxerr,float(np.max(np.abs(mixed-other))));events+=yp.size
                predictions.append(yp);cc.append(u.count(yp,y,good))
            cc=np.stack(cc);blocks.append(cc)
            np.savez_compressed(OUT/f'{phase}-seed-{e["seed"]}-all-predictions.npz',predictions=np.stack(predictions),
                ckir_percent=np.arange(101),y=y,zero=zero,ckir_native_scores=qs,gda_native_scores=score)
            for i in range(101):blockrows.append({'phase':phase,'seed':e['seed'],'ckir_percent':i,**u.metrics(cc[i],1)})
            print(f'{phase} {e["seed"]}: 101 mixtures; GDA/source check complete',flush=True)
        arr=np.stack(blocks);arrays[phase]=arr;n=len(blocks);rows=[]
        for i in range(101):
            rates=arr[:,i,2]/1776
            rows.append({'phase':phase,'ckir_percent':i,'gda_percent':100-i,**u.metrics(arr[:,i].sum(0),n),
                'accuracy_sd':float(rates.std(ddof=1)),'accuracy_se':float(rates.std(ddof=1)/np.sqrt(n))})
        results[phase]=rows
        u.csv_out(OUT/f'{phase}-mixture-curve.csv',rows);u.csv_out(OUT/f'{phase}-per-block.csv',blockrows)
        np.savez_compressed(OUT/f'{phase}-per-block-counts.npz',counts=arr,seeds=[e['seed'] for e in plan['episodes'][phase]])
        if phase=='development':assert all(rows[0][k]==gridrows[native_idx][k] for k in u.FIELDS)
        ckref=read(ROOT/'runs/E1-v05-joint-grid-20261020/selected-configurations-before-pilot.json')['choices']['qp'] if phase=='development' else read(PREVIOUS/'findings.json')['pilot_ckir']
        assert all(rows[-1][k]==ckref[k] for k in u.FIELDS)
        if phase=='development':
            dev_idx=u.choose(rows)
            save(OUT/'selected-ratio-before-pilot.json',{'saved_hk':now(),'selected':rows[dev_idx],'selection':plan['selection']})
    pilot=results['pilot'];best=u.choose(pilot);diff={}
    for name,i in [('pilot_peak',best),('development_selected',dev_idx)]:
        diff[name]={}
        for method,j in [('gda',0),('ckir',100)]:
            d=arrays['pilot'][:,i,2]-arrays['pilot'][:,j,2]
            diff[name][method]={'accuracy_delta_pp':100*(pilot[i]['noisy_accuracy']-pilot[j]['noisy_accuracy']),
                'correct_event_difference':int(d.sum()),'fixed_difference':pilot[i]['fixed']-pilot[j]['fixed'],
                'harmed_difference':pilot[i]['harmed']-pilot[j]['harmed'],
                'positive_blocks':int((d>0).sum()),'tie_blocks':int((d==0).sum()),'negative_blocks':int((d<0).sum())}
    for p,h in protected.items():assert sha(p)==h,p
    assert maxerr<1e-10
    f={'status':'COMPLETE_EXPLORATORY_GDA_AND_101_MIXTURES','completed_hk':now(),'seconds':time.perf_counter()-started,
        'ckir_parameters':plan['ckir_parameters'],'gda_parameters':plan['gda_parameters'],'gda_development_selection':gda_selection,
        'development_selected':results['development'][dev_idx],'pilot_at_development_selected':pilot[dev_idx],
        'pilot_peak_descriptive':pilot[best],
        'pilot_tied_maximum_percentages':[r['ckir_percent'] for r in pilot if r['noisy_correct']==pilot[best]['noisy_correct']],
        'pilot_gda':pilot[0],'pilot_ckir':pilot[100],'differences':diff,
        'previous_proker_mixture_peak':read(PRO/'findings.json')['pilot_peak_descriptive'],
        'previous_tip_mixture_peak':read(PREVIOUS/'findings.json')['pilot_peak_descriptive'],
        'prediction_events_checked_two_score_expressions':int(events),'maximum_score_expression_error':maxerr,
        'native_float32_vs_independent_float64':{'prediction_events':refevents,'prediction_disagreements':refdiffs,'maximum_score_error':referr,
            'authoritative_precision':'Unchanged official GDA torch.float32, regardless of float64 rounding differences'},
        'protected_files_unchanged':len(protected),'scaling':plan['scaling'],'scope':plan['scope'],
        'new_QP_training_encoding_sampling_test':0,'new_GDA_estimator_fits':88,'BGSR':'paused'}
    save(OUT/'findings.json',f);print(json.dumps(f,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
