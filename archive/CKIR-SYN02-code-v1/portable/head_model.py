"""LP++-style linear image head and per-class text coefficient inside CKIR p0."""
import torch
from torch import nn
from torch.nn import functional as F

class LPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros((37,1024),device='cuda',dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(37,device='cuda',dtype=torch.float32))
        self.gamma=nn.Parameter(torch.full((37,),100.,device='cuda',dtype=torch.float64))

    def probability(self,features,text_logits):
        logits=F.linear(features,self.weight.double(),self.bias.double())+text_logits*self.gamma
        return torch.softmax(logits,dim=-1)

    def scores(self,features,text_logits,support_probability,weights,onehot):
        return self.probability(features,text_logits)+1.19*weights.t()@(onehot-support_probability)
