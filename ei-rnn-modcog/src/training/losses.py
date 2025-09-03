from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

def row_sum_penalty(W_hh: torch.Tensor, sign_vec: Optional[torch.Tensor] = None, *, per_sign: bool = True, reduction: str = "mean",) -> torch.Tensor:
    if (sign_vec is None) or (not per_sign):
        vals = W_hh.sum(dim=1).pow(2) 
    else:
        s = sign_vec.reshape(1, -1).to(dtype=W_hh.dtype, device=W_hh.device)
        e_mask = (s > 0).to(W_hh.dtype)
        i_mask = (s < 0).to(W_hh.dtype)
        row_e = (W_hh * e_mask).sum(dim=1)
        row_i = (W_hh * i_mask).sum(dim=1)
        vals = row_e.pow(2) + row_i.pow(2)

    if reduction == "mean":
        return vals.mean()
    if reduction == "sum":
        return vals.sum()
    return vals

@torch.no_grad()
def decision_mask_from_inputs(X: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (X[..., 0] < thresh)

class ModCogLossCombined(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, fixdown_weight: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.fixdown_weight = float(fixdown_weight)

    def forward(self, outputs, labels, dec_mask):
        B, T, C = outputs.shape
        target_fix = outputs.new_zeros(B, T, C); target_fix[..., 0] = 1.0

        loss_fix = self.mse(outputs[~dec_mask], target_fix[~dec_mask]) if (~dec_mask).any() else outputs.sum() * 0.0

        if dec_mask.any():
            loss_dec = self.ce(outputs[dec_mask], 1 + labels[dec_mask])
            fix_logits_dec = outputs[..., 0][dec_mask]
            loss_fixdown = (fix_logits_dec ** 2).mean() * self.fixdown_weight
        else:
            loss_dec = outputs.sum() * 0.0
            loss_fixdown = outputs.sum() * 0.0

        return loss_fix + loss_dec + loss_fixdown, loss_fix.detach(), (loss_dec + loss_fixdown).detach()
