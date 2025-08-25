from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F


def total_variation(x: Tensor) -> Tensor:
    if x.dim() == 4:
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return (dx.abs().mean() + dy.abs().mean())
    elif x.dim() == 3:
        dt = x[1:, :, :] - x[:-1, :, :]
        dd = x[:, :, 1:] - x[:, :, :-1]
        return (dt.abs().mean() + dd.abs().mean())
    else:
        dlast = x[..., 1:] - x[..., :-1]
        return dlast.abs().mean()


def saliency(
    forward_score: Callable[[Tensor], Tensor],
    x: Tensor,
    retain_graph: bool = False
) -> Tensor:
    x = x.clone().detach().requires_grad_(True)
    score = forward_score(x)
    if score.dim() != 0:
        score = score.mean()
    score.backward(retain_graph=retain_graph)
    return x.grad.detach()


def optimize_inputs(
    forward_score: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    steps: int = 200,
    step_size: float = 0.05,
    epsilon: float = 0.05,
    bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
    reg_l2: float = 1e-4,
    reg_tv: float = 1e-4,
    freeze_mask: Optional[Tensor] = None
) -> Dict[str, Tensor]:
    x0 = x0.detach()
    delta = torch.zeros_like(x0, requires_grad=True)

    if freeze_mask is None:
        freeze_mask = torch.ones_like(x0)

    opt = torch.optim.Adam([delta], lr=step_size)

    with torch.no_grad():
        score_before = forward_score(x0).detach()

    for _ in range(steps):
        opt.zero_grad()
        x = x0 + delta * freeze_mask
        if bounds is not None:
            x = x.clamp(bounds[0], bounds[1])
        score = forward_score(x)
        if score.dim() != 0:
            score = score.mean()
        reg = reg_l2 * (delta * freeze_mask).pow(2).mean()
        reg = reg + reg_tv * total_variation(x)
        loss = -(score - reg)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data = (x0 + delta * freeze_mask).clamp(*(bounds if bounds else (-float("inf"), float("inf")))) - x0
            delta.data.clamp_(-epsilon, epsilon)
            delta.data = delta.data * freeze_mask

    with torch.no_grad():
        x_opt = (x0 + delta * freeze_mask)
        if bounds is not None:
            x_opt = x_opt.clamp(bounds[0], bounds[1])
        score_after = forward_score(x_opt).detach()

    return {
        "x_opt": x_opt.detach(),
        "delta": (x_opt - x0).detach(),
        "score_before": score_before,
        "score_after": score_after,
    }
