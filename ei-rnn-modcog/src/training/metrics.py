from __future__ import annotations
import torch

@torch.no_grad()
def accuracy_with_fixation(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    dec_mask: torch.Tensor 
) -> tuple[float, float]:
    pred = outputs.argmax(dim=-1)

    labels_shifted = outputs.new_zeros(labels.shape, dtype=torch.long)
    labels_shifted.copy_(labels + 1)
    targets = labels_shifted.clone()
    targets[~dec_mask] = 0

    acc_all = (pred == targets).float().mean().item()
    if dec_mask.any():
        acc_dec = (pred[dec_mask] == targets[dec_mask]).float().mean().item()
    else:
        acc_dec = float("nan")
    return acc_all, acc_dec
