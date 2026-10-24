"""Fixed-alpha GDA using the recorded KS estimator and native float32 scores.

Adapted from ProKeR/trainers/gda.py and the ICLR24 GDA implementation.
See THIRD_PARTY_NOTICES.md and LICENSES/ProKeR-Apache-2.0.txt.
No validation-label search, softmax or rescaling is performed here.
"""
from dataclasses import dataclass
import torch


@dataclass
class GDA:
    weight: torch.Tensor
    bias: torch.Tensor
    text: torch.Tensor
    alpha: float

    def scores(self, features):
        q = torch.as_tensor(features, dtype=torch.float32, device='cpu')
        return 100. * q.float() @ self.text.float() + self.alpha * (q.float() @ self.weight + self.bias)


def fit_gda(support, observed_labels, text, *, alpha):
    vecs = torch.as_tensor(support, dtype=torch.float32, device='cpu')
    labels = torch.as_tensor(observed_labels, dtype=torch.long, device='cpu')
    clip_weights = torch.as_tensor(text, dtype=torch.float32, device='cpu')
    if vecs.ndim != 2 or clip_weights.ndim != 2 or vecs.shape[1] != clip_weights.shape[0]:
        raise ValueError('GDA expects support [N,d] and unadapted text [d,C]')
    classes = clip_weights.shape[1]
    if labels.shape != (len(vecs),) or torch.any(labels < 0) or torch.any(labels >= classes):
        raise ValueError('invalid observed support labels')
    if any(not torch.any(labels == i) for i in range(classes)):
        raise ValueError('GDA requires at least one observed support sample for every class')
    with torch.no_grad():
        mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(classes)])
        center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(classes)])
        # Preserve both covariance calls and the author's pseudoinverse semantics.
        cov_inv = center_vecs.shape[1] * torch.linalg.pinv(
            (center_vecs.shape[0] - 1) * center_vecs.T.cov()
            + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).to(vecs.device))
        ps = torch.ones(classes).to(vecs.device) * 1. / classes
        weight = torch.einsum('nd, dc -> cn', mus, cov_inv)
        bias = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
        if not all(torch.isfinite(x).all() for x in (weight, bias)):
            raise ValueError('GDA returned non-finite parameters')
    return GDA(weight, bias, clip_weights, float(alpha))


def fuse_scores(ckir_scores, gda_scores, *, rho):
    if not 0. <= rho <= 1.:
        raise ValueError('rho must be between zero and one')
    c = torch.as_tensor(ckir_scores, dtype=torch.float64, device='cpu')
    g = torch.as_tensor(gda_scores, dtype=torch.float64, device='cpu')
    if c.shape != g.shape or not torch.isfinite(c).all() or not torch.isfinite(g).all():
        raise ValueError('finite, matching CKIR and GDA score arrays are required')
    return float(rho) * c + (1. - float(rho)) * g
