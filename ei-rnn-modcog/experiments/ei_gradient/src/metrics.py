from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import Tensor
import math

def split_by_type(x: Tensor, idx_E: Tensor, idx_I: Tensor) -> Tuple[Tensor, Tensor]:
    xE = x[..., idx_E]
    xI = x[..., idx_I]
    return xE, xI

def timecourse_l2(grad_h: Tensor, idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    gE, gI = split_by_type(grad_h, idx_E, idx_I)
    tcE = gE.norm(dim=-1).mean(dim=1)
    tcI = gI.norm(dim=-1).mean(dim=1)
    return {"tc_l2_E": tcE, "tc_l2_I": tcI}

def timecourse_l2_per_unit(grad_h: Tensor, idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    gE, gI = split_by_type(grad_h, idx_E, idx_I)
    tcE = (gE.pow(2).sum(dim=-1) / gE.shape[-1]).sqrt().mean(dim=1)
    tcI = (gI.pow(2).sum(dim=-1) / gI.shape[-1]).sqrt().mean(dim=1)
    return {"tc_l2_E_norm": tcE, "tc_l2_I_norm": tcI}

def timecourse_mean_abs(grad_h: Tensor, idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    gE, gI = split_by_type(grad_h, idx_E, idx_I)
    tcE = gE.abs().mean(dim=-1).mean(dim=1)
    tcI = gI.abs().mean(dim=-1).mean(dim=1)
    return {"tc_abs_E": tcE, "tc_abs_I": tcI}

def cosine_alignment(h_seq: Tensor, grad_h: Tensor, idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    def _cos(a, b):
        num = (a * b).sum(dim=-1)
        denom = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
        return (num / denom).mean(dim=1)
    hE, hI = split_by_type(h_seq, idx_E, idx_I)
    gE, gI = split_by_type(grad_h, idx_E, idx_I)
    return {"tc_cos_E": _cos(hE, gE), "tc_cos_I": _cos(hI, gI)}

def gate_adjusted_timecourse(grad_u: Tensor | None, grad_h: Tensor, phi_prime: Tensor | None,
                             idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    if grad_u is not None:
        g = grad_u
    elif phi_prime is not None:
        g = grad_h * phi_prime
    else:
        g = grad_h 
    return timecourse_mean_abs(g, idx_E, idx_I)

def cumulative_backprop(grad_h: Tensor) -> Tensor:
    Wbp_sum = grad_h.abs().sum(dim=(0, 1)) 
    Wbp_mean = grad_h.abs().mean(dim=(0, 1))
    return Wbp_sum, Wbp_mean

def fisher_like(grad_h: Tensor) -> Tensor:
    return grad_h.pow(2).mean(dim=(0, 1)) 

def gini_coefficient(x: Tensor) -> Tensor:
    x = x.flatten()
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    x_sorted = x.sort().values
    n = x_sorted.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return (2 * (idx * x_sorted).sum() / (n * x_sorted.sum()) - (n + 1) / n)

def summarize_ei_distributions(
    Wbp_sum: Tensor, Wbp_mean: Tensor, Fisher: Tensor, idx_E: Tensor, idx_I: Tensor
) -> Dict[str, float]:
    out = {}
    for name, vec in [("Wbp_sum", Wbp_sum), ("Wbp_mean", Wbp_mean), ("Fisher", Fisher)]:
        Evals = vec[idx_E]; Ivals = vec[idx_I]
        out[f"{name}_E_mean"] = float(Evals.mean().item())
        out[f"{name}_I_mean"] = float(Ivals.mean().item())
        out[f"{name}_E_gini"] = float(gini_coefficient(Evals).item())
        out[f"{name}_I_gini"] = float(gini_coefficient(Ivals).item())
    return out

