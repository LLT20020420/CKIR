"""Train W/b with the already-selected best CKIR inference geometry and alpha."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from scipy.linalg import eigvalsh

ROOT = Path('D:/CodexResearch/Projects/RobustCLIP')
HERE = Path(__file__).parent
OUT = HERE / 'results'
PARENT = ROOT / 'analysis/v05-ckir-lpp-training-20261021'
BEST = ROOT / 'analysis/v05-ckir-lpp-joint-grid-20261021'
TABLE = ROOT / 'analysis/mixture-comparison-20261021'
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location('aligned_parent', PARENT / 'run.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
LPHead, u = parent.LPHead, parent.u
read, save, sha = u.read, u.save, u.sha
EPOCH = 7
INFERENCE = {'beta': 16., 'lambda': 1., 'tau': .03, 'alpha': 2.51}


def freeze():
    OUT.mkdir()
    p = read(PARENT / 'results/frozen-plan-before-training.json')
    best = read(BEST / 'results/findings.json')
    assert best['selected_config'] == INFERENCE and best['checkpoint_epoch'] == EPOCH
    inventory = read(BEST / 'results/frozen-plan-before-search.json')['inventory']
    paths = {str(Path(__file__)), str(PARENT / 'run.py'), str(PARENT / 'model.py'),
             str(BEST / 'results/findings.json'), str(p['feature_path']),
             str(TABLE / 'results/完整原始记录.json'),str(ROOT/'core.py'),str(ROOT/'core_v04.py')}
    query_weights = {}
    for phase, episodes in p['episodes'].items():
        for e in episodes:
            seed = e['seed']
            if phase == 'development':
                rec = inventory[f'{phase}/{seed}/16|1|0.03']
            else:
                rec = read(BEST / f'results/pilot-jobs/seed-{seed}.json')['weight_records'][1]
            assert sha(rec['path']) == rec['sha256']
            query_weights[f'{phase}/{seed}'] = rec['path']
            paths.update((e['receipt'], e['labels'], rec['path'],
                str(parent.TEXT / f'results/{phase}-support-weights-{seed}.npz'),
                str(parent.BASE / f'{phase}-components-{seed}.npz')))
            for k in range(4):
                paths.add(str(PARENT / f'results/{phase}-head-weights-{seed}-{k}.npz'))
    plan = {'frozen_hk': u.now(), 'authorization': 'User: first do option1, retrain W/b with the final best inference parameters',
        'fixed_epoch': EPOCH, 'selection': 'Fixed epoch7, training/inference16/1/.03/2.51 before outcomes; no new checkpoint or parameter search',
        'training': 'Exact retained LP++-inspired recipe: start W/b0,gamma100 and frozen learned epoch25 text; 7 full-batch W/b SGD momentum0.9 updates, original learning rates; gamma first updates at10 so remains100',
        'training_geometry': INFERENCE,
        'inference': INFERENCE, 'loss': 'CE(p0(support)+2.51*wstar(16,1,.03).T@(onehot-p0(support)),observed labels); no MAE,anchor,temperature or new modules',
        'comparison': 'Same training start, steps, data and fixed best inference as historical87.5675676%; re-train W/b from the same zero initialization; only change training geometry/alpha to match final inference; no extra continuation',
        'scope': p['scope'], 'episodes': p['episodes'], 'feature_path': p['feature_path'],
        'query_weights': query_weights, 'protected_hashes': {f: sha(f) for f in sorted(paths)},
        'official_test_used': False, 'new_encoding_sampling': 0, 'planned_new_support_QP_cases': 6512, 'BGSR': 'paused'}
    save(OUT / 'frozen-plan.json', plan)
    return plan


def fit(s, q, observed, train_weights, query_weights, text, lr_w, phase, seed, version):
    model = LPHead().cuda()
    support = torch.tensor(s, device='cuda')
    query = torch.tensor(q, device='cuda')
    text = torch.tensor(text, device='cuda')
    sl, ql = support @ text.t(), query @ text.t()
    labels = torch.tensor(observed, device='cuda')
    onehot = F.one_hot(labels, 37).double()
    wt, wq = torch.tensor(train_weights, device='cuda'), torch.tensor(query_weights, device='cuda')
    optimizer = torch.optim.SGD([model.weight, model.bias], lr=lr_w, momentum=.9)
    states, predictions, traces, audit = [], [], [], {}
    with np.load(PARENT / f'results/{phase}-head-weights-{seed}-{version}.npz') as z:
        old_epochs = list(z['epochs'])
        old = {n: z[n].copy() for n in ('weight', 'bias', 'gamma')}
        assert 0 in old_epochs and EPOCH in old_epochs
    for epoch in range(EPOCH+1):
        if epoch:
            model.gamma.requires_grad_(False)
            ps = model.probability(support, sl)
            score = ps + INFERENCE['alpha'] * wt.t() @ (onehot-ps)
            loss = F.cross_entropy(score, labels)
            if epoch == 1 and phase == 'development' and seed == 1000 and version == 1:
                grad = torch.autograd.grad(loss,model.weight,retain_graph=True)[0]
                old_grad = -torch.as_tensor(old['weight'][old_epochs.index(1)],device='cuda')/lr_w
                audit['first_W_gradient_norm_aligned'] = float(grad.norm().detach())
                audit['first_W_gradient_norm_old_approx_from_update'] = float(old_grad.norm())
                audit['first_W_gradient_difference_norm'] = float((grad-old_grad).norm().detach())
                assert audit['first_W_gradient_difference_norm']>0
            optimizer.zero_grad()
            loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in (model.weight, model.bias))
            optimizer.step()
            traces.append({'epoch': epoch, 'classification': float(loss.detach()), 'loss': float(loss.detach()),
                           'weight_norm': float(model.weight.detach().norm())})
        with torch.no_grad():
            ps = model.probability(support, sl)
            scores = model.probability(query, ql) + INFERENCE['alpha'] * wq.t() @ (onehot-ps)
            assert torch.isfinite(scores).all()
            predictions.append(scores.argmax(-1).cpu().numpy().astype(np.uint8))
            state = {n: v.detach().cpu().numpy().copy() for n, v in model.named_parameters()}
            states.append(state)
            if epoch == 0:
                for n in state:
                    assert np.array_equal(state[n], old[n][old_epochs.index(epoch)]), (phase, seed, version, epoch, n)
            assert np.all(state['gamma'] == 100.)
    # Build baseline with the SAME inference kernel and historical epoch7 parameters.
    with torch.no_grad():
        for n, v in model.named_parameters():
            v.copy_(torch.as_tensor(old[n][old_epochs.index(EPOCH)], device='cuda'))
        ps = model.probability(support, sl)
        baseline = model.probability(query, ql)+INFERENCE['alpha']*wq.t()@(onehot-ps)
        baseline = baseline.argmax(-1).cpu().numpy().astype(np.uint8)
    path = OUT / f'{phase}-fit-{seed}-{version}.npz'
    np.savez_compressed(path, predictions=np.stack(predictions), baseline=baseline,
                        epochs=np.arange(EPOCH+1), **{n: np.stack([v[n] for v in states]) for n in states[0]})
    audit.update({'training_geometry': INFERENCE, 'lr_W': lr_w, 'steps': EPOCH, 'gamma_updates': 0,
                  'initial_exactly_matches_old': True,
                  'query_labels_used_in_optimizer': False,
                  'W_difference_from_old_epoch7': float(np.linalg.norm(states[-1]['weight']-old['weight'][old_epochs.index(EPOCH)]))})
    save(OUT/f'{phase}-training-{seed}-{version}.json', {'trace': traces, 'audit': audit})
    return np.stack(predictions), baseline


def aligned_support_weights(s,phase,seed):
    from scipy.linalg import cho_factor,cho_solve
    sys.path.insert(0,str(ROOT))
    from core import regularized_kernel
    from core_v04 import BoxWorkspace
    cache=Path('D:/CodexResearch/Cache/RobustCLIP/v05-ckir-aligned-head-training-20261022')
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{phase}-support-weights-{seed}.npz'
    if path.exists():
        with np.load(path) as z:return z['wstar'].copy()
    started=time.perf_counter()
    g=s/np.linalg.norm(s,axis=1,keepdims=True)
    gram,jitter=regularized_kernel(g,16.);A=gram+np.eye(len(g))
    w0=cho_solve(cho_factor(A,lower=True),gram);box=BoxWorkspace(A,.03)
    w=np.empty_like(w0);ratios=[]
    for j in range(len(g)):
        got=box.solve(gram[:,j],w0[:,j]);assert got.accepted,got.summary()
        w[:,j]=got.w;ratios.append(got.pg_inf/got.pg_tolerance)
    np.savez_compressed(path,wstar=w,pg_ratio=ratios,jitter=jitter)
    save(OUT/f'{phase}-support-geometry-{seed}.json',{'path':str(path),'sha256':sha(path),'QP_cases':len(g),
        'max_pg_ratio':float(max(ratios)),'seconds':time.perf_counter()-started,'jitter':jitter})
    return w


def main():
    parent.setup()
    p = freeze()
    started = time.perf_counter()
    images, idx, qids, q, y, zero = parent.data(p)
    good = zero == y
    phase_metrics, paired, count_arrays = {}, [], {}
    for phase in ('development', 'pilot'):
        all_counts, base_counts = [], []
        for e in p['episodes'][phase]:
            seed = e['seed']
            s, observed, _, _, comp = parent.inputs(e, phase, images, idx, qids)
            wt = aligned_support_weights(s,phase,seed)
            with np.load(p['query_weights'][f'{phase}/{seed}']) as z:
                wq = z['wstar'].copy()
            lr_w = 4*296/float(eigvalsh(s@s.T, subset_by_index=[295,295])[0])
            pairs = [fit(s, q, o, wt, wq, comp['learned_text'][k], lr_w, phase, seed, k)
                     for k, o in enumerate(observed)]
            pred = np.stack([a for a, _ in pairs], axis=1)
            base = np.stack([b for _, b in pairs])
            cc = np.stack([u.count(v, y, good) for v in pred])
            bc = u.count(base, y, good)
            if phase == 'pilot':
                with np.load(BEST/f'results/pilot-jobs/seed-{seed}.npz') as z:
                    assert np.array_equal(base, z['predictions'][1])
            else:
                with np.load(BEST/f'results/development-selected-predictions-{seed}.npz') as z:
                    assert np.array_equal(base, z['predictions'])
            np.savez_compressed(OUT/f'{phase}-predictions-{seed}.npz', predictions=pred,
                baseline=base, counts=cc, baseline_counts=bc, y=y, zero=zero, epochs=np.arange(EPOCH+1))
            all_counts.append(cc)
            base_counts.append(bc)
            if phase == 'pilot':
                change = cc[EPOCH]-bc
                paired.append(dict(zip(('seed','fixed_change','harmed_change','net_correct_change','clean_correct_change'), [seed,*map(int,change)])))
            save(OUT/'runtime-state.json', {'stage': phase, 'last_seed': seed,
                'completed_fits': 4*(len(all_counts)+(12 if phase=='pilot' else 0)),
                'elapsed_seconds': time.perf_counter()-started})
            print(f'{phase} seed{seed}: new {cc[EPOCH,2]}/1776 vs best {bc[2]}/1776', flush=True)
        arr, base = np.stack(all_counts), np.stack(base_counts)
        count_arrays[phase] = arr
        np.savez_compressed(OUT/f'{phase}-counts.npz', counts=arr, baseline_counts=base, epochs=np.arange(EPOCH+1))
        rows = [{'epoch': j, **u.metrics(arr[:,j].sum(0),len(arr))} for j in range(EPOCH+1)]
        u.csv_out(OUT/f'{phase}-all-epochs.csv', rows)
        phase_metrics[phase] = {'new': rows[EPOCH], 'baseline': u.metrics(base.sum(0),len(base))}
    old = read(BEST/'results/findings.json')['pilot_joint_ckir']
    m = phase_metrics['pilot']['new']
    assert m['noisy_correct'] == 14310+m['fixed']-m['harmed']
    assert all(phase_metrics['pilot']['baseline'][n] == old[n] for n in ('fixed','harmed','noisy_correct','clean_correct','noisy_accuracy'))
    d = {'net_correct_change': m['noisy_correct']-old['noisy_correct'],
        'accuracy_delta_pp': 100*(m['noisy_accuracy']-old['noisy_accuracy']),
        'fixed_change': m['fixed']-old['fixed'], 'harmed_change': m['harmed']-old['harmed'],
        'clean_correct_change': m['clean_correct']-old['clean_correct'],
        'positive_blocks': sum(r['net_correct_change']>0 for r in paired),
        'negative_blocks': sum(r['net_correct_change']<0 for r in paired),
        'tied_blocks': sum(r['net_correct_change']==0 for r in paired), 'per_support_draw': paired}
    assert all(sha(path)==digest for path,digest in p['protected_hashes'].items())
    findings = {'status': 'COMPLETE_FIXED_EPOCH_ALIGNED_HEAD_TRAINING', 'completed_hk': u.now(),
        'seconds_main': time.perf_counter()-started, 'fixed_epoch': EPOCH,
        'pilot_aligned_ckir': m, 'pilot_best_previous_ckir': old,
        'development_aligned_ckir': phase_metrics['development']['new'],
        'development_baseline': phase_metrics['development']['baseline'],
        'paired_pilot_difference': d, 'training_fits': 88, 'W_updates': 616,
        'protected_files_unchanged': len(p['protected_hashes']), 'scope': p['scope'],
        'official_test_used': False, 'new_encoding_sampling': 0,'new_support_QP_cases': 6512}
    save(OUT/'findings.json', findings)
    save(OUT/'runtime-state.json', {'stage': 'COMPLETE', 'completed_hk': u.now()})
    print(json.dumps(findings,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
