"""Portable packaging of the retained CKIR parameterization.

Derived from the verbatim declarations in archive/CKIR-SYN02-code-v1.
Dimensions/device are inferred or supplied; alpha is always explicit.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CKIRText(nn.Module):
    def __init__(self, template, cupl, negative=None):
        super().__init__()
        if template.ndim != 2 or template.shape != cupl.shape:
            raise ValueError('template and cupl must have matching [feature, class] shapes')
        if negative is not None and negative.shape != template.shape:
            raise ValueError('negative text shape differs from positive text')
        self.register_buffer('template', template.clone().half())
        self.register_buffer('cupl', cupl.clone().half())
        if negative is not None:
            self.register_buffer('negative', negative.clone().half())
        d, c = template.shape
        self.res_template = nn.Parameter(torch.zeros((c, d), device=template.device, dtype=torch.float16))
        self.res_cupl = nn.Parameter(torch.zeros((c, d), device=template.device, dtype=torch.float16))

    def positive(self):
        tp = F.normalize(self.template + self.res_template.t(), dim=0).half()
        cp = F.normalize(self.cupl + self.res_cupl.t(), dim=0).half()
        tp = F.normalize(tp, dim=0).half()
        cp = F.normalize(cp, dim=0).half()
        return .45 * tp + .55 * cp

    def probability(self, features):
        return torch.softmax((100. * features.double()) @ self.positive().double(), dim=-1)

    def ckir(self, support, query, weights, onehot, *, alpha):
        ps = self.probability(support)
        return self.probability(query) + float(alpha) * weights.t() @ (onehot - ps)

    def consistency(self):
        pos = self.positive()
        cos = nn.CosineSimilarity(dim=1, eps=1e-7)
        return (.45 * (1 - cos(pos, self.template)) + .55 * (1 - cos(pos, self.cupl))).mean()


class LPHead(nn.Module):
    def __init__(self, classes, feature_dim, *, device):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((classes, feature_dim), device=device, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(classes, device=device, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((classes,), 100., device=device, dtype=torch.float64), requires_grad=False)

    def probability(self, features, text_logits, *, mode='joint'):
        if mode == 'aligned':
            # Historical aligned-head stage used F.linear.
            logits = F.linear(features.double(), self.weight.double(), self.bias.double()) + text_logits * self.gamma
        elif mode == 'joint':
            # Keep the retained joint-stage operation order, including bias.
            logits = features.double() @ self.weight.double().t() + self.bias.double() + self.gamma * text_logits
        else:
            raise ValueError('head mode must be aligned or joint')
        return torch.softmax(logits, dim=-1)

    def scores(self, features, text_logits, support_probability, weights, onehot, *, alpha, mode='joint'):
        return self.probability(features, text_logits, mode=mode) + float(alpha) * weights.t() @ (onehot - support_probability)
