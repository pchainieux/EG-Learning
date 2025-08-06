from __future__ import annotations
import torch
import torch.nn as nn

def decision_mask_from_inputs(X: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (X[..., 0] < thresh)

class ModCogLossCombined(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, fixdown_weight: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.fixdown_weight = float(fixdown_weight)

    def forward(
        self,
        outputs: torch.Tensor, 
        labels: torch.Tensor, 
        dec_mask: torch.Tensor 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = outputs.shape

        target_fix = outputs.new_zeros(B, T, C)
        target_fix[..., 0] = 1.0 
        loss_fix = self.mse(outputs[~dec_mask], target_fix[~dec_mask]) if (~dec_mask).any() else outputs.sum()*0.0

        if dec_mask.any():
            loss_dec = self.ce(outputs[dec_mask], 1 + labels[dec_mask])
            fix_logits_dec = outputs[..., 0][dec_mask]
            loss_fixdown = (fix_logits_dec ** 2).mean() * self.fixdown_weight
        else:
            loss_dec = outputs.sum()*0.0
            loss_fixdown = outputs.sum()*0.0

        total = loss_fix + loss_dec + loss_fixdown
        return total, loss_fix.detach(), (loss_dec + loss_fixdown).detach()