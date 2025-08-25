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


def timecourse_mean_abs(grad_h: Tensor, idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    gE, gI = split_by_type(grad_h, idx_E, idx_I)
    return {
        "tc_ma_E": gE.abs().mean(dim=(1, 2)),
        "tc_ma_I": gI.abs().mean(dim=(1, 2)),
    }


def cumulative_backprop(grad_h: Tensor) -> Tensor:
    return grad_h.abs().sum(dim=(0, 1))  # [N]


def fisher_like(grad_h: Tensor) -> Tensor:
    return (grad_h ** 2).mean(dim=(0, 1))  # [N]


def gini_coefficient(x: Tensor, eps: float = 1e-12) -> Tensor:
    x = x.flatten()
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    x = x.abs() + eps
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    cumx = torch.cumsum(x_sorted, dim=0)
    gini = (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n
    return gini


def cosine_alignment(h_seq: Tensor, grad_h: Tensor, idx_E: Tensor, idx_I: Tensor) -> Dict[str, Tensor]:
    def _cos(a: Tensor, b: Tensor) -> Tensor:
        num = (a * b).sum(dim=-1)
        den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
        return (num / den).mean(dim=1)  # [T]
    hE, hI = split_by_type(h_seq, idx_E, idx_I)
    gE, gI = split_by_type(grad_h, idx_E, idx_I)
    return {"cos_E": _cos(hE, gE), "cos_I": _cos(hI, gI)}


def gate_adjusted_timecourse(
    grad_h: Tensor,
    u_seq: Tensor,
    idx_E: Tensor,
    idx_I: Tensor,
    activation: str = "softplus"
) -> Dict[str, Tensor]:
    if activation == "softplus":
        sigma = torch.sigmoid(u_seq)
        g_adj = grad_h * sigma
    elif activation == "tanh":
        g_adj = grad_h * (1 - torch.tanh(u_seq) ** 2)
    else:
        g_adj = grad_h

    gE, gI = split_by_type(g_adj, idx_E, idx_I)
    return {
        "tc_l2_E_gated": gE.norm(dim=-1).mean(dim=1),
        "tc_l2_I_gated": gI.norm(dim=-1).mean(dim=1),
    }

def summarize_ei_distributions(
    Wbp: Tensor,
    fisher: Tensor,
    idx_E: Tensor,
    idx_I: Tensor
) -> Dict[str, float]:
    out = {}
    for name, vec in [("Wbp", Wbp), ("Fisher", fisher)]:
        Evals = vec[idx_E]
        Ivals = vec[idx_I]
        out[f"{name}_E_mean"] = float(Evals.mean().item())
        out[f"{name}_I_mean"] = float(Ivals.mean().item())
        out[f"{name}_E_gini"] = float(gini_coefficient(Evals).item())
        out[f"{name}_I_gini"] = float(gini_coefficient(Ivals).item())
    return out
