from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

"""
Saliency and constrained input optimisation (projected gradient ascent) utilities.

Notes:
- Optimisation supports L_infinity constraint (epsilon), L2 regularisation, and TV regularisation
  for smoothness. Channels can be frozen (e.g., fixation).
- Caller supplies `forward_score` which returns a scalar to maximize (e.g., negative loss for class c).
"""

def total_variation(x: Tensor) -> Tensor:
    """
    Isotropic total variation for sequences (T,B,D) or images (B,C,H,W).
    For 3D tensors (T,B,D), applies 1D TV over time and a simple neighbor penalty over D.
    """
    if x.dim() == 4:  # images
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return (dx.abs().mean() + dy.abs().mean())
    elif x.dim() == 3:  # [T,B,D]
        dt = x[1:, :, :] - x[:-1, :, :]
        dd = x[:, :, 1:] - x[:, :, :-1]
        return (dt.abs().mean() + dd.abs().mean())
    else:
        # fallback: L1 on finite differences along last dim
        dlast = x[..., 1:] - x[..., :-1]
        return dlast.abs().mean()


def saliency(
    forward_score: Callable[[Tensor], Tensor],
    x: Tensor,
    retain_graph: bool = False
) -> Tensor:
    """
    Compute ∂score/∂x, where 'score' is a scalar function of x (e.g., class target).
    Args:
        forward_score: callable that maps x -> scalar tensor to maximize.
        x: input tensor with shape compatible with the model.
    Returns:
        grad_x: same shape as x.
    """
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
    epsilon: float = 0.05,               # L∞ budget
    bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
    reg_l2: float = 1e-4,
    reg_tv: float = 1e-4,
    freeze_mask: Optional[Tensor] = None # same shape as x0, 1=free, 0=frozen
) -> Dict[str, Tensor]:
    """
    Projected gradient ascent on inputs to maximize 'forward_score(x)'.

    Args:
        forward_score: callable mapping x -> scalar to maximize (e.g., -loss(target=c)).
        x0: baseline input (will not be modified).
        steps, step_size: optimisation schedule.
        epsilon: L∞ constraint on delta.
        bounds: (min,max) clip range after each step.
        reg_l2, reg_tv: regularization weights.
        freeze_mask: binary mask (1: optimizable, 0: frozen channels/timesteps).

    Returns:
        dict with:
          - "x_opt": optimized input
          - "delta": perturbation x_opt - x0
          - "score_before": forward_score(x0).detach()
          - "score_after": forward_score(x_opt).detach()
    """
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
        # Regularization (penalize large deltas, encourage smoothness)
        reg = reg_l2 * (delta * freeze_mask).pow(2).mean()
        reg = reg + reg_tv * total_variation(x)
        loss = -(score - reg)   # maximize score - reg
        loss.backward()
        opt.step()
        # Project back to L∞ ball and bounds
        with torch.no_grad():
            delta.data = (x0 + delta * freeze_mask).clamp(*(bounds if bounds else (-float("inf"), float("inf")))) - x0
            delta.data.clamp_(-epsilon, epsilon)
            delta.data = delta.data * freeze_mask  # enforce freezing

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
