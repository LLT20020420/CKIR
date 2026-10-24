import torch 
from utils import cls_acc

def GDA(vecs, labels, val_features, val_labels, test_features, clip_weights, dataset, shots, seed, hp_selection, backbone='RN50'):
    """
        A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation, ICLR 2024.
    """
    # normal distribution
    mus = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(clip_weights.shape[1])])
    device = vecs.device
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    # KS Estimator.  
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(clip_weights.shape[1])])
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).to(device))    

    ps = torch.ones(clip_weights.shape[1]).to(device) * 1. / clip_weights.shape[1]
    W = torch.einsum('nd, dc -> cn', mus, cov_inv)
    b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2
    # Evaluate
    # Grid search for hyper-parameter alpha
    best_val_acc = 0
    best_alpha = 0.1
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        val_logits = 100. * val_features.float() @ clip_weights.float() + alpha * (val_features.float() @ W + b)
    
        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = alpha
    
    print("best_val_alpha: %s \t best_val_acc: %s" % (best_alpha, best_val_acc))
    alpha = best_alpha
    test_features = test_features.to(device)
    test_logits = 100. * test_features.float() @ clip_weights.float() + alpha * (test_features.float() @ W + b)
    return test_logits.cpu()