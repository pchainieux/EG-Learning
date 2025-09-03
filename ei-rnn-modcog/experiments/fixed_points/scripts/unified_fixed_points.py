# experiments/fixed_points/scripts/unified_fixed_points.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.fixed_points.src.config import load_config
from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt


@dataclass
class SolverCfg:
    max_iter: int = 500
    tol: float = 1e-6
    lr: float = 1.0               # step size for optimizing h
    line_beta: float = 0.5        # backtracking shrink
    line_c: float = 1e-4          # Armijo constant


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def _step_F(model, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
    # ensure same dtype as model parameters
    dtype = model.W_hh.dtype
    h = h.to(dtype)
    x = x.to(dtype)

    pre = x @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    phi = F.softplus(beta * pre) / beta
    return (1.0 - leak) * h + leak * phi


def _residual(model, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
    return _step_F(model, h, x, leak=leak, beta=beta) - h


def _jacobian(model, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> np.ndarray:
    with torch.no_grad():
        W_hh = model.W_hh.detach().to("cpu", torch.float64)
        W_xh = model.W_xh.detach().to("cpu", torch.float64)
        b_h = model.b_h.detach().to("cpu", torch.float64)

        x64 = x.detach().to("cpu", torch.float64)
        h64 = h.detach().to("cpu", torch.float64)
        pre = x64 @ W_xh.T + h64 @ W_hh.T + b_h
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
    """Minimize ||F(h) - h||^2 over h via gradient descent with backtracking."""
    h = h0.detach().clone().to(device=x_bar.device, dtype=x_bar.dtype).requires_grad_(True)

    def loss_fn(h_):
        r = _residual(model, h_, x_bar, leak=leak, beta=beta)
        return (r ** 2).sum()

    t = 0
    lr = cfg.lr
    last = float("inf")
    while t < cfg.max_iter:
        t += 1
        loss = loss_fn(h)
        res = float(loss.detach().cpu().item())
        if res < cfg.tol:
            return h.detach(), res, t

        # backtracking
        loss.backward()
        g = h.grad.detach().clone()
        with torch.no_grad():
            step = -lr * g
            new_h = h + step
            new_loss = float(loss_fn(new_h).detach().cpu().item())
            # Armijo condition
            iters_bt = 0
            while new_loss > res - cfg.line_c * lr * (g ** 2).sum().item() and iters_bt < 20:
                lr *= cfg.line_beta
                step = -lr * g
                new_h = h + step
                new_loss = float(loss_fn(new_h).detach().cpu().item())
                iters_bt += 1
            h[:] = new_h
            h.grad.zero_()
            last = new_loss

    return h.detach(), last, t


def classify_motif(eigs: np.ndarray) -> str:
    mag = np.abs(eigs)
    maxmag = float(mag.max()) if mag.size else 0.0

    near_one = (np.abs(mag - 1.0) < 0.02).any()
    any_gt1 = (mag > 1.0 + 1e-3).any()
    all_lt1 = (mag < 1.0 - 1e-3).all()

    if all_lt1:
        return "stable"
    if near_one and not any_gt1:
        return "continuous"
    if any_gt1 and not all_lt1:
        # check for complex/rotational near unit circle
        has_rot = np.iscomplex(eigs).any() and (np.abs(mag - 1.0) < 0.05).any()
        return "rotational" if has_rot else "saddle"
    return "unstable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="Path to <outdir>/run")
    ap.add_argument("--config", type=str, required=True, help="Training YAML used to recover model / eval dirs")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    cfg = load_config(args.config)

    # Where the orchestrator placed seeds & where to write results
    eval_dir = run_dir / cfg.get("fixed_points", {}).get("eval", {}).get("outdir", "eval/fixed_points")
    seeds = eval_dir / "rollout_seeds.npz"
    if not seeds.exists():
        raise FileNotFoundError(f"Missing seeds file: {seeds}")

    data = np.load(str(seeds), allow_pickle=True)
    X_np: np.ndarray = data["X"]
    H0_np: np.ndarray = data["H0"]

    # Recover model
    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    device = _device(cfg.get("device", "auto"))
    model, saved_cfg = rebuild_model_from_ckpt(ckpt, device=device)
    model = model.to(device).eval()
    mcfg = saved_cfg.get("model", {})
    leak = float(mcfg.get("leak", 1.0))
    beta = float(mcfg.get("softplus_beta", 8.0))

    solver = SolverCfg(max_iter=args.max_iter, tol=args.tol, lr=args.lr)

    # match model params
    param = next(model.parameters())
    dtype = param.dtype
    device = param.device

    H0    = torch.from_numpy(H0_np).to(device=device, dtype=dtype)
    X_last = torch.from_numpy(X_np[:, -1, :]).to(device=device, dtype=dtype)

    results: List[Dict[str, Any]] = []
    H_star_list: List[np.ndarray] = []
    Eigs_list: List[np.ndarray] = []

    for i in range(H0.shape[0]):
        x_bar = X_last[i].unsqueeze(0)     # (1, in_dim)
        h0 = H0[i].unsqueeze(0)            # (1, H)

        h_star, res, iters = solve_one_fp(model, x_bar, h0, leak, beta, solver)

        # Jacobian & spectrum at the fixed point
        J = _jacobian(model, h_star, x_bar, leak, beta)
        eigs = np.linalg.eigvals(J)
        rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
        margin = float(1.0 - rho)
        label = classify_motif(eigs)

        results.append(dict(
            index=i, residual=res, iters=iters, rho=rho, margin=margin, label=label
        ))
        H_star_list.append(h_star.squeeze(0).detach().cpu().numpy())
        Eigs_list.append(eigs.astype(np.complex128))

    eval_dir.mkdir(parents=True, exist_ok=True)
    # Save summary CSV
    df = pd.DataFrame(results)
    df.to_csv(eval_dir / "summary.csv", index=False)

    # Save raw FP tensors & eigenvalues
    np.savez_compressed(
        str(eval_dir / "fixed_points.npz"),
        H_star=np.stack(H_star_list, axis=0),
        eigvals=np.array(Eigs_list, dtype=object),
    )

    print(f"[unified_fixed_points] wrote {len(results)} entries → {eval_dir}")


if __name__ == "__main__":
    main()
