"""Train positive text prototypes through CKIR; cross-score one model two ways.

This borrows SimNL's optimizer/loss/epoch recipe. It is not full original SimNL
training: image geometry, visual/negative prototypes and CKIR hyperparameters
remain fixed. Both evaluation heads share the same learned positive prototypes.
"""
import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('D:/CodexResearch/Projects/RobustCLIP')
SIMNL=ROOT/'analysis/v05-simnl-mixture-20261021'


def source_losses():
    main=ast.parse((SIMNL/'source-snapshot/main.py').read_text())
    utils=ast.parse((SIMNL/'source-snapshot/utils.py').read_text())
    nodes=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='adaptive_reranking_loss']
    nodes += [n for n in utils.body if isinstance(n,ast.ClassDef) and n.name=='SmoothCrossEntropy']
    namespace={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SIMNL/'source-snapshot/main.py'),'exec'),namespace)
    return namespace['SmoothCrossEntropy'](),namespace['adaptive_reranking_loss']


class CKIRText(nn.Module):
    def __init__(self,template,cupl,negative):
        super().__init__()
        self.register_buffer('template',template.clone())
        self.register_buffer('cupl',cupl.clone())
        self.register_buffer('negative',negative.clone())
        self.res_template=nn.Parameter(torch.zeros((37,1024),device='cuda',dtype=torch.float16))
        self.res_cupl=nn.Parameter(torch.zeros((37,1024),device='cuda',dtype=torch.float16))

    def positive(self):
        tp=F.normalize(self.template+self.res_template.t(),dim=0).half()
        cp=F.normalize(self.cupl+self.res_cupl.t(),dim=0).half()
        # Preserve both normalizations performed by original positive/negative adapters.
        tp=F.normalize(tp,dim=0).half();cp=F.normalize(cp,dim=0).half()
        return .45*tp+.55*cp

    def probability(self,features):
        return torch.softmax((100.*features.double())@self.positive().double(),dim=-1)

    def ckir(self,support,query,weights,onehot):
        ps=self.probability(support)
        return self.probability(query)+2.44*weights.t()@(onehot-ps)

    def consistency(self):
        pos=self.positive()
        cos=nn.CosineSimilarity(dim=1,eps=1e-7)
        return (.45*(1-cos(pos,self.template))+.55*(1-cos(pos,self.cupl))).mean()


def author_frozen_components(sim,gl,classes,capture,s,q,o,text):
    capture['models']=[]
    sx,neg,soft,values,models,changed,order=sim.prepare(gl,classes,s,o,text)
    keys,nkeys,tp,cp,tn,_=changed
    query=torch.as_tensor(q,dtype=torch.float16,device='cuda')
    with torch.no_grad():
        pos=query@keys.half().t();negative=1-query@nkeys.half().t()
        pa=((-1)*(6.72-6.72*pos)).exp()@soft
        na=((-1)*(6.72-6.72*negative)).exp()@values
        nw=100.*(1-query@tn)*.15
    capture['models']=[]
    return query,pa,na,nw,sx,order


def author_score(model,components):
    query,pa,na,nw,_,_=components
    pw=100.*(query@model.positive())
    return .76*(pw+pa*1.05)+(1-.76)*(nw+na*1.05)
