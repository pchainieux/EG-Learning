from __future__ import annotations
import torch

@torch.no_grad()
def accuracy_with_fixation(outputs, labels, dec_mask):
    pred = outputs.argmax(dim=-1)
    labels_shifted = outputs.new_zeros(labels.shape, dtype=torch.long)
    labels_shifted.copy_(labels + 1)
    labels_full = labels_shifted.clone()
    labels_full[~dec_mask] = 0
    acc_all = (pred == labels_full).float().mean().item()
    acc_dec = (pred[dec_mask] == labels_full[dec_mask]).float().mean().item() if dec_mask.any() else float("nan")
    return acc_all, acc_dec