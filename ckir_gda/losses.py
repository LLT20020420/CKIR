"""SimNL loss declarations, verbatim from upstream commit a0871bcf31a607a7ec8acc99aff82f41ed7eaff3."""
import torch
from torch import nn
from torch.nn import functional as F

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def adaptive_reranking_loss(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    scale: float = 4.0,
    knn: int = 3,
    **_: torch.Tensor,
) -> torch.Tensor:

    N = visual_features.shape[0]
    C = class_prototypes.shape[0]
    knn = min(knn, C)
    
    
    distances = torch.cdist(visual_features.float(), class_prototypes.float(), p=2)

    sorted_distances, sorted_indices = torch.sort(
        distances, dim=1, descending=False)
    anchor = (
        ((visual_features - class_prototypes[labels]) ** 2).sum(-1).sqrt().unsqueeze(1)
    )
    sorted_distances = sorted_distances[:, :knn]

    pos_cla_proto = class_prototypes[labels].unsqueeze(1)
    all_cls = class_prototypes[sorted_indices[:, :knn]]
    margins = (1.0 - (all_cls * pos_cla_proto).sum(-1)) / scale

    loss = torch.max(
        anchor + margins - sorted_distances,
        torch.zeros(N, knn).to(visual_features.device),
    )

    return loss.mean()
