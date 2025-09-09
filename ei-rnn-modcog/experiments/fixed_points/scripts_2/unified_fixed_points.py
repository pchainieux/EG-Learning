from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class SolverCfg:
    lr: float = 1e-1
    tol: float = 1e-10
    max_iter: int = 5000
    line_beta: float = 0.5
    line_c: float = 1e-4
    lam_prox: float = 1e-3 

def step_F(model: torch.nn.Module, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
    if h.dim() == 1: h = h[None, :]
    if x.dim() == 1: x = x[None, :]
    pre = x @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    phi = F.softplus(beta * pre) / beta
    return (1.0 - leak) * h + leak * phi

def residual(model: torch.nn.Module, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
    return step_F(model, h, x, leak=leak, beta=beta) - h

def jacobian_at(model: torch.nn.Module, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> np.ndarray:
    with torch.no_grad():
        W_hh = model.W_hh.detach().to("cpu", torch.float64)
        b_h  = model.b_h.detach().to("cpu", torch.float64)
        x64  = x.detach().to("cpu", torch.float64)
        h64  = h.detach().to("cpu", torch.float64)
        pre = x64 @ model.W_xh.detach().to("cpu", torch.float64).T + h64 @ W_hh.T + b_h
        sig = torch.sigmoid(beta * pre).squeeze(0)
        I = torch.eye(W_hh.shape[0], dtype=torch.float64)
        J = (1.0 - leak) * I + leak * torch.diag(sig) @ W_hh
    return J.numpy()

def solve_one_fp(
    model: torch.nn.Module,
    x_bar: torch.Tensor,
    h0: torch.Tensor,
    leak: float,
    beta: float,
    cfg: SolverCfg,
) -> Tuple[torch.Tensor, float, int]:
    device, dtype = x_bar.device, x_bar.dtype
    h = h0.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
    h_seed = h0.detach().clone().to(device=device, dtype=dtype)   # <— for proximal term

    def loss_fn(h_):
        r = residual(model, h_, x_bar, leak=leak, beta=beta)
        q = (r ** 2).sum()
        if cfg.lam_prox > 0:
            q = q + cfg.lam_prox * ((h_ - h_seed) ** 2).sum()
        return q

    lr, last, t = cfg.lr, float("inf"), 0
    while t < cfg.max_iter:
        t += 1
        loss = loss_fn(h)
        resv = float(loss.detach().cpu().item())
        if resv < cfg.tol:
            return h.detach(), resv, t
        loss.backward()
        g = h.grad.detach().clone()
        with torch.no_grad():
            new_h = h - lr * g
            new_loss = float(loss_fn(new_h).detach().cpu().item())
            it_bt = 0
            while new_loss > resv - cfg.line_c * lr * (g ** 2).sum().item() and it_bt < 25:
                lr *= cfg.line_beta
                new_h = h - lr * g
                new_loss = float(loss_fn(new_h).detach().cpu().item())
                it_bt += 1
            h.copy_(new_h); h.grad.zero_()
        if abs(last - resv) < 1e-16: lr *= 0.5
        last = resv
    return h.detach(), last, t

def classify_from_eigs(eigs: np.ndarray) -> str:
    if eigs is None or len(eigs) == 0: return "unknown"
    mag = np.abs(eigs)
    any_gt1 = (mag > 1.0 + 1e-3).any()
    all_lt1 = (mag < 1.0 - 1e-3).all()
    near_one = (np.abs(mag - 1.0) < 0.02).any()
    if all_lt1: return "stable"
    if near_one and not any_gt1: return "continuous"
    if any_gt1 and not all_lt1:
        rot = np.iscomplex(eigs).any() and (np.abs(mag - 1.0) < 0.05).any()
        return "rotational" if rot else "saddle"
    return "unstable"
