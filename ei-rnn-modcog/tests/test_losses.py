import torch
from torch.testing import assert_close
from src.training.losses import ModCogLossCombined, decision_mask_from_inputs

def test_mask_and_loss_shapes():
    B,T,D,C = 4, 11, 8, 6
    X = torch.zeros(B,T,D); X[...,0] = 1.0; X[:,3:7,0] = 0.0
    y = torch.randint(0, C-1, (B,T))
    logits = torch.randn(B,T,C)

    mask = decision_mask_from_inputs(X, thresh=0.5)
    assert mask.shape == (B,T) and mask.dtype == torch.bool

    crit = ModCogLossCombined()
    total, lf, ld = crit(logits, y, mask)
    assert total.ndim == lf.ndim == ld.ndim == 0

def test_mask_monotonic():
    B,T,D,C = 2, 7, 5, 4
    X = torch.zeros(B,T,D); X[...,0] = 1.0; X[:,2:5,0] = 0.0
    y = torch.zeros(B,T, dtype=torch.long)
    mask = decision_mask_from_inputs(X, 0.5)

    torch.manual_seed(0)
    logits_a = torch.randn(B,T,C)
    logits_b = logits_a.clone()
    target_fix = torch.zeros(B,T,C); target_fix[...,0] = 1.0
    logits_b[~mask] = target_fix[~mask]

    crit = ModCogLossCombined()
    total_a, lf_a, _ = crit(logits_a, y, mask)
    total_b, lf_b, _ = crit(logits_b, y, mask)
    assert lf_b.item() <= lf_a.item() + 1e-6
