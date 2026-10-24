"""Scan cached scores of pure-CKIR-trained weights and fixed GDA at101 ratios."""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path('D:/CodexResearch/Projects/RobustCLIP');HERE=Path(__file__).parent;OUT=HERE/'results'
PRE=ROOT/'analysis/v05-ckir-gda-training-20261022/results'
JOINT=ROOT/'analysis/v05-ckir-joint-text-head-20261022/results'
TABLE=ROOT/'analysis/mixture-comparison-20261021'
spec=importlib.util.spec_from_file_location('trained_mix_common',ROOT/'analysis/v05-strong-mixture-20261021/run_mix.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
read,save,sha,now=u.read,u.save,u.sha,u.now
RATIOS=np.array([i/100 for i in range(101)])


def main():
    if OUT.exists():raise RuntimeError('Preserve existing results')
    started=time.perf_counter();previous=read(PRE/'frozen-plan.json');provenance=read(PRE/'artifact-manifest.json')
    assert read(PRE/'verification.json')['status']=='PASS' and read(PRE/'delivery-audit.json')['status']=='PASS'
    protected={}
    for p in [Path(__file__),HERE/'figure-contract.md',PRE/'frozen-plan.json',PRE/'findings.json',PRE/'verification.json',
              PRE/'delivery-audit.json',PRE/'artifact-manifest.json',JOINT/'findings.json',
              TABLE/'build.py',TABLE/'results/完整原始记录.json']:
        protected[str(p)]=sha(p)
    for phase in ('development','pilot'):
        for e in previous['episodes'][phase]:
            p=PRE/f'{phase}-predictions-{e["seed"]}.npz';h=sha(p)
            assert h==provenance[str(p)],p
            protected[str(p)]=h
    OUT.mkdir()
    p={'frozen_hk':now(),'authorization':'用户要求先训练CKIR再混合，重新尝试不同比例并作图',
       'CKIR_weights':'Existing pure-CKIR CE-trained positive text+W/b, joint step7 (87.6126126%); all weights fixed',
       'CKIR_score_source':'Verified control_ckir_scores from v05-ckir-gda-training-20261022, not mixed-loss CKIR weights',
       'gda_alpha':.1,'gda':'Original single-prompt nativefloat32 GDA, cached scores promotedfloat64, fixed statistics per support/noise condition',
       'formula':'rho*CKIR+(1-rho)*GDA','percentages':list(range(101)),
       'selection':'Development most noisy correct, ties fewer harmed then smaller CKIR percent; save before pilot. Pilot full curve and descriptive peak separately retained.',
       'scope':previous['scope'],'episodes':previous['episodes'],'protected_hashes':protected,
       'new_training_GDA_fitting_QP_encoding_sampling':0,'official_test_used':False,'BGSR':'paused'}
    save(OUT/'frozen-plan-before-mixing.json',p)
    phase_rows={};phase_counts={};events=0;maxerr=0.;selected=None
    old=read(PRE/'findings.json');pairs={}
    for phase in ('development','pilot'):
        blocks=[];blockrows=[]
        for e in p['episodes'][phase]:
            seed=e['seed']
            with np.load(PRE/f'{phase}-predictions-{seed}.npz') as z:
                cs=z['control_ckir_scores'].copy();gs=z['gda_query_scores'].copy();y=z['y'].copy();zero=z['zero'].copy()
                cp=z['control_ckir_predictions'].copy();mp95=z['control_mix_predictions'].copy()
            assert cs.shape==gs.shape==(4,592,37) and np.array_equal(cs.argmax(-1),cp)
            preds=[];counts=[]
            for i,rho in enumerate(RATIOS):
                scores=rho*cs+(1-rho)*gs
                reference=gs+rho*(cs-gs)
                pred=scores.argmax(-1).astype(np.uint8)
                assert np.array_equal(pred,reference.argmax(-1)),(phase,seed,i)
                maxerr=max(maxerr,float(np.max(abs(scores-reference))));events+=pred.size
                if i==95:assert np.array_equal(pred,mp95)
                preds.append(pred);c=u.count(pred,y,zero==y);counts.append(c)
                blockrows.append({'seed':seed,'ckir_percent':i,**dict(zip(u.FIELDS,map(int,c)))})
            arr=np.stack(counts);blocks.append(arr)
            np.savez_compressed(OUT/f'{phase}-predictions-{seed}.npz',predictions=np.stack(preds),counts=arr,ckir_percent=np.arange(101),y=y,zero=zero)
        counts=np.stack(blocks);n=len(blocks);total=counts.sum(0)
        rows=[{'ckir_percent':i,'gda_percent':100-i,**u.metrics(c,n),
               'accuracy_se':float((counts[:,i,2]/1776).std(ddof=1)/np.sqrt(n))} for i,c in enumerate(total)]
        phase_rows[phase]=rows;phase_counts[phase]=counts
        u.csv_out(OUT/f'{phase}-mixture-curve.csv',rows);u.csv_out(OUT/f'{phase}-per-support-draw.csv',blockrows)
        np.savez_compressed(OUT/f'{phase}-counts.npz',counts=counts)
        assert all(rows[95][k]==old['all_phase_metrics'][phase]['posthoc_mix'][k] for k in u.FIELDS)
        assert all(rows[100][k]==old['all_phase_metrics'][phase]['previous_best_ckir'][k] for k in u.FIELDS)
        assert all(rows[0][k]==old['all_phase_metrics'][phase]['gda'][k] for k in u.FIELDS)
        if phase=='development':
            selected=u.choose(rows)
            save(OUT/'selected-ratio-before-pilot.json',{'saved_hk':now(),'ckir_percent':selected,'gda_percent':100-selected,'development_metrics':rows[selected],'selection':p['selection']})
            print(f'Development ratio saved: CKIR{selected}/GDA{100-selected}; now scanning seen pilot.',flush=True)
        print(f'{phase}:101 ratios,{n} support draws complete',flush=True)
    rows=phase_rows['pilot'];peak=u.choose(rows);ties=[r['ckir_percent'] for r in rows if r['noisy_correct']==rows[peak]['noisy_correct']]
    def compare(i,ref):
        delta=phase_counts['pilot'][:,i]-phase_counts['pilot'][:,ref]
        per=[dict(zip(('seed','fixed_change','harmed_change','net_correct_change','clean_correct_change'),[e['seed'],*map(int,d)])) for e,d in zip(p['episodes']['pilot'],delta)]
        total=delta.sum(0)
        return {'net_correct_change':int(total[2]),'accuracy_delta_pp':100*int(total[2])/17760,
          'fixed_change':int(total[0]),'harmed_change':int(total[1]),'clean_correct_change':int(total[3]),
          'positive_blocks':sum(d[2]>0 for d in delta),'negative_blocks':sum(d[2]<0 for d in delta),
          'tie_blocks':sum(d[2]==0 for d in delta),'per_support_draw':per}
    assert all(sha(path)==h for path,h in protected.items())
    f={'status':'COMPLETE_TRAINED_CKIR_GDA_RATIO_SCAN','completed_hk':now(),'seconds_main':time.perf_counter()-started,
       'development_selected':phase_rows['development'][selected],
       'pilot_at_development_selected':rows[selected],'pilot_peak_descriptive':rows[peak],
       'pilot_tied_maximum_percentages':ties,'pilot_previous95_5':rows[95],'pilot_ckir':rows[100],'pilot_gda':rows[0],
       'selected_vs_previous95_5':compare(selected,95),'peak_vs_previous95_5':compare(peak,95),
       'selected_vs_pure_ckir':compare(selected,100),'peak_vs_pure_ckir':compare(peak,100),
       'prediction_events_two_expressions_compared':events,'maximum_score_expression_error':maxerr,
       'protected_file_count':len(protected),'official_test_used':False,'new_training_GDA_fitting_QP_encoding_sampling':0}
    save(OUT/'findings.json',f);print(json.dumps(f,ensure_ascii=False,default=int),flush=True)


if __name__=='__main__':main()
