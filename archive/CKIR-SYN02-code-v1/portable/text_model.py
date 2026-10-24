"""Verbatim CKIRText declaration from the frozen executed text-stage module."""
import torch
from torch import nn
from torch.nn import functional as F

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
