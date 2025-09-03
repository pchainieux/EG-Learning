from __future__ import annotations


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd


from config import load_config
from model_io import rebuild_model_from_ckpt


# Optional imports (safe fallbacks if unavailable)
try:
from sklearn.cluster import DBSCAN # type: ignore
_HAS_SK = True
except Exception:
_HAS_SK = False




# ----------------------
# Utilities & dataclasses
# ----------------------
@dataclass
class SolverCfg:
max_iter: int = 500
tol: float = 1e-6
lr: float = 1.0 # step size for optimizing h
backtrack_beta: float = 0.5 # line search shrink
backtrack_c: float = 1e-4 # Armijo constant




def _device(name: str) -> torch.device:
if name == "auto":
return torch.device("cuda" if torch.cuda.is_available() else "cpu")
return torch.device(name)




# ----------------------
# Dynamics consistent with training (uses model's leak & nonlinearity params)
# ----------------------
@torch.no_grad()
def step_F(model, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
"""One-step update: (1-α) h + α φ(W_rec h + W_in x + b), φ=softplus_β.
All in float64 for numerical stability.
"""
W_hh = model.W_hh
W_xh = model.W_xh
b_h = model.b_h
pre = x @ W_xh.T + h @ W_hh.T + b_h
# Softplus with custom beta; derivative is sigmoid(beta*pre)
phi = F.softplus(beta * pre) / beta
return (1.0 - leak) * h + leak * phi




def residual(model, h: torch.Tensor, x_bar: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
return step_F(model, h, x_bar, leak=leak, beta=beta) - h




def jacobian_at(model, h: torch.Tensor, x_bar: torch.Tensor, leak: float, beta: float) -> np.ndarray:
"""J = (1-α)I + α diag(φ'(pre)) W_rec with φ' = sigmoid(beta*pre)."""
with torch.no_grad():
W_hh = model.W_hh.detach().to("cpu", torch.float64)
W_xh = model.W_xh.detach().to("cpu", torch.float64)
b_h = model.b_h.detach().to("cpu", torch.float64)


x64 = x_bar.detach().to("cpu", torch.float64)
h64 = h.detach().to("cpu", torch.float64)
pre = x64 @ W_xh.T + h64 @ W_hh.T + b_h
sig = torch.sigmoid(beta * pre).squeeze(0) # (H,)
I = torch.eye(W_hh.shape[0], dtype=torch.float64)
J = (1.0 - leak) * I + leak * torch.diag(sig) @ W_hh
return J.numpy()




# ----------------------
# Fixed-point solver with backtracking line search
# ----------------------
@torch.no_grad()
def _loss_from_residual(r: torch.Tensor) -> torch.Tensor:
return (r ** 2).sum()




def solve_fixed_point(
model: torch.nn.Module,
x_bar: torch.Tensor,