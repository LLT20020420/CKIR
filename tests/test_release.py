"""Small software checks against archived implementations; no dataset accuracy."""
import ast,contextlib,io,json,os,sys,tempfile,types,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ckir_gda.models import CKIRText,LPHead
from ckir_gda.gda import fit_gda,fuse_scores
from ckir_gda.pipeline import train_head,train_joint,snapshot,setup,learning_rate
from ckir_gda.io import load_config,load_episode,load_checkpoint,save_checkpoint
from ckir_gda.io import digest
from ckir_gda.pipeline import weights
from ckir_gda.evaluate import counts
from ckir_gda.solvers.core import enumerate_box,kernel
from ckir_gda.solvers.core_v04 import BoxWorkspace


def archived_declaration(path,name,namespace):
    module=ast.parse(path.read_text('utf-8'))
    node=next(n for n in module.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class ReleaseChecks(unittest.TestCase):
    def test_paired_counts(self):
        result=counts([0,1,1,0],[0,0,1,1],[1,0,0,1])
        self.assertEqual((result['benefit'],result['harm'],result['net_correct']),(2,2,0))
        self.assertEqual(result['accuracy_percent'],50.)
    def test_manifest_checkpoint_and_cache_identity(self):
        scratch=Path(os.environ.get('CKIR_TEST_TMP',Path.cwd()/'.ckir-tests')).resolve()
        scratch.mkdir(parents=True,exist_ok=True)
        config=load_config(ROOT/'configs/retained.json')
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            base=Path(tmp)
            arrays={'support':np.ones((4,8)), 'query':np.ones((2,8)),
                'template':np.ones((8,2),dtype=np.float16), 'cupl':np.ones((8,2),dtype=np.float16),
                'gda_text':np.ones((8,2),dtype=np.float32), 'observed_labels':np.array([0,1,0,1])}
            np.savez_compressed(base/'episode.npz',**arrays)
            m={'format':'ckir-gda-episode-v1','episode_id':'contract','support_seed':1,'label_version':0,
               'class_names':['a','b'],'support_ids':['s0','s1','s2','s3'],'query_ids':['q0','q1'],
               'arrays':{'path':'episode.npz','sha256':digest(base/'episode.npz')}}
            (base/'episode.json').write_text(json.dumps(m),'utf-8')
            manifest,loaded=load_episode(base/'episode.json')
            metadata={k:manifest[k] for k in ('episode_id','class_names','support_ids','support_fingerprint')}
            metadata.update(stage='text',kernels=config['kernels'])
            save_checkpoint(base/'checkpoint.npz',{'res_template':np.zeros((2,8))},metadata)
            load_checkpoint(base/'checkpoint.npz',manifest,config)
            loaded['final_query_weights']=np.zeros((4,2))
            with self.assertRaises(ValueError):weights(loaded,manifest,config,stage='final',target='query')
            changed=dict(config);changed['kernels']={'text':dict(config['kernels']['text']),'final':dict(config['kernels']['final'],alpha=1.19)}
            with self.assertRaises(ValueError):load_checkpoint(base/'checkpoint.npz',manifest,changed)
            arrays['observed_labels']=np.array([1,0,0,1])
            np.savez_compressed(base/'episode.npz',**arrays)
            with self.assertRaises(ValueError):load_episode(base/'episode.json')
            m['arrays']['sha256']=digest(base/'episode.npz');(base/'episode.json').write_text(json.dumps(m),'utf-8')
            changed_manifest,_=load_episode(base/'episode.json')
            with self.assertRaises(ValueError):load_checkpoint(base/'checkpoint.npz',changed_manifest,config)

    def test_solver_and_native_gda(self):
        rng=np.random.default_rng(41)
        s=rng.normal(size=(4,3));s/=np.linalg.norm(s,axis=1,keepdims=True)
        A=kernel(s,s,2)+.1*np.eye(4);k=kernel(s,s,2)[:,0]
        result=BoxWorkspace(A,.05).solve(k)
        self.assertTrue(result.accepted)
        np.testing.assert_allclose(result.w,enumerate_box(A,k,.05),rtol=0,atol=1e-8)
        x=rng.normal(size=(12,8)).astype('float32');y=np.arange(12)%3
        q=rng.normal(size=(4,8)).astype('float32');t=rng.normal(size=(8,3)).astype('float32')
        count=[0]
        def select_fixed_alpha(scores,labels):
            i=count[0];count[0]+=1
            return 100. if i==3 else 0.
        old=archived_declaration(ROOT/'provenance/gda-proker-unmodified.py','GDA',{'torch':torch,'cls_acc':select_fixed_alpha})
        with contextlib.redirect_stdout(io.StringIO()):
            expected=old(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(q),torch.zeros(4,dtype=torch.long),torch.from_numpy(q),torch.from_numpy(t),'synthetic',4,0,'fixed')
        actual=fit_gda(x,y,t,alpha=.1).scores(q)
        self.assertTrue(torch.equal(actual,expected))
        c=torch.from_numpy(q[:,:3]).double()
        self.assertTrue(torch.equal(fuse_scores(c,actual,rho=.95),.95*c+(1-.95)*expected.double()))

    def test_alpha_is_required(self):
        model=LPHead(3,8,device='cpu')
        with self.assertRaises(TypeError):
            model.scores(None,None,None,None,None)
        text=CKIRText(torch.ones(8,3).half(),torch.ones(8,3).half())
        with self.assertRaises(TypeError):
            text.ckir(None,None,None,None)

    @unittest.skipUnless(torch.cuda.is_available(),'archived declarations require CUDA')
    def test_frozen_text_gradients_and_two_seven_step_stages(self):
        device=setup('cuda');torch.manual_seed(4024)
        c,d,n,nq=37,1024,12,5
        raw=[F.normalize(torch.randn(d,c,device=device),dim=0).half() for _ in range(3)]
        namespace={'torch':torch,'nn':nn,'F':F}
        archive=ROOT/'archive/CKIR-SYN02-code-v1'
        OldText=archived_declaration(archive/'historical/text_model.py','CKIRText',dict(namespace))
        OldHead=archived_declaration(archive/'historical/head_model.py','LPHead',dict(namespace))
        old=OldText(*raw);new=CKIRText(*raw)
        s=F.normalize(torch.randn(n,d,device=device,dtype=torch.float64),dim=1)
        q=F.normalize(torch.randn(nq,d,device=device,dtype=torch.float64),dim=1)
        labels=torch.arange(n,device=device)%c;one=F.one_hot(labels,c).double()
        w=torch.randn(n,n,device=device,dtype=torch.float64)*.001
        rank_ns=dict(namespace)
        rank=archived_declaration(archive/'upstream/simnl/main.py','adaptive_reranking_loss',rank_ns)
        # Compare the actual archived loss graph, including fp16 view normalization.
        sx=s.half();sx=sx/sx.norm(dim=-1,keepdim=True)
        for model in (old,new):
            ps=model.probability(s)
            score=ps+2.44*w.t()@(one-ps)
            loss=F.cross_entropy(score,labels)+rank(sx,model.positive().t(),labels)+8*model.consistency()
            loss.backward()
        self.assertTrue(torch.equal(old.positive(),new.positive()))
        for a,b in zip(old.parameters(),new.parameters()):self.assertTrue(torch.equal(a.grad,b.grad))
        arrays={'support':s.cpu().numpy(),'query':q.cpu().numpy(),'observed_labels':labels.cpu().numpy(),
                'template':raw[0].cpu().numpy(),'cupl':raw[1].cpu().numpy(),
                'final_support_weights':w.cpu().numpy()}
        config=load_config(ROOT/'configs/retained.json')
        manifest={'cached_geometry':{'final':{'beta':16.,'lambda':1.,'tau':.03}}}
        initial=snapshot(new)
        scratch=Path(os.environ.get('CKIR_TEST_TMP',Path.cwd()/'.ckir-tests')).resolve()
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            base=Path(tmp);parent=base/'parent';textdir=base/'text'
            (parent/'results').mkdir(parents=True);(textdir/'results').mkdir(parents=True)
            old_out=base/'aligned/results';old_out.mkdir(parents=True)
            h=OldHead()
            np.savez_compressed(parent/'results/synthetic-head-weights-1-1.npz',epochs=np.array([0,7]),
                **{name:np.stack([p.detach().cpu().numpy()]*2) for name,p in h.named_parameters()})
            def save(path,data):Path(path).write_text(json.dumps(data),'utf-8')
            common={'torch':torch,'np':np,'F':F,'LPHead':OldHead,'PARENT':parent,'OUT':old_out,
                    'EPOCH':7,'INFERENCE':config['kernels']['final'],'save':save}
            fit=archived_declaration(archive/'historical/aligned_head_stage.py','fit',common)
            lr=learning_rate(arrays['support']);T=new.positive().detach().t().double().cpu().numpy()
            qw=np.zeros((n,nq),dtype=np.float64)
            fit(arrays['support'],arrays['query'],arrays['observed_labels'],arrays['final_support_weights'],qw,T,lr,'synthetic',1,1)
            aligned,_=train_head(arrays,manifest,config,device,initial)
            with np.load(old_out/'synthetic-fit-1-1.npz') as z:
                for name in ('weight','bias','gamma'):np.testing.assert_array_equal(aligned[name],z[name][-1])
            # Feed both implementations the same recorded-form stage starts.
            np.savez_compressed(textdir/'results/synthetic-weights-1-1.npz',
                **{name:np.stack([initial[name]]*26) for name in ('res_template','res_cupl')})
            joint_out=base/'joint';joint_out.mkdir()
            import time
            joint_gl={'torch':torch,'np':np,'F':F,'tm':types.SimpleNamespace(CKIRText=OldText),
                'parent':types.SimpleNamespace(LPHead=OldHead),'TEXT':textdir,'BASE':old_out.parent,'OUT':joint_out,
                'read':lambda p:json.loads(Path(p).read_text()),'save':save,'STEPS':7,'ALPHA':2.51,
                'HEAD_LR_FACTOR':.1,'TEXT_LR':1e-5,'time':time}
            joint=archived_declaration(archive/'historical/joint_stage.py','fit',joint_gl)
            joint(arrays['support'],arrays['query'],arrays['observed_labels'],arrays['final_support_weights'],qw,raw,'synthetic',1,1,1)
            portable,_=train_joint(arrays,manifest,config,device,aligned)
            with np.load(joint_out/'synthetic-fit-1-1-arm1.npz') as z:
                for name in ('weight','bias','gamma','res_template','res_cupl','text_features'):
                    np.testing.assert_array_equal(portable[name],z[name][-1])


if __name__=='__main__':unittest.main()
