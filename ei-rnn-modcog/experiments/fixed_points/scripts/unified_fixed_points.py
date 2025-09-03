from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.fixed_points.src.config import load_config
from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt
from src.data import mod_cog_tasks as mct
from neurogym import Dataset
import pandas as pd


@dataclass
class SolverCfg:
    max_iter: int = 500
    tol: float = 1e-6
    alpha: float = 1.0


@torch.no_grad()
def step_leaky_softplus(model, h: torch.Tensor, x_t: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    pre = x_t @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    return (1 - alpha) * h + alpha * F.softplus(pre)


def residual(model, h: torch.Tensor, x_bar: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return step_leaky_softplus(model, h, x_bar, alpha=alpha) - h


def jacobian_at(model, h: torch.Tensor, x_bar: torch.Tensor, alpha: float = 1.0) -> np.ndarray:
    with torch.no_grad():
        W_hh = model.W_hh.detach().to("cpu", torch.float64)
        b_h = model.b_h.detach().to("cpu", torch.float64)
        W_xh = model.W_xh.detach().to("cpu", torch.float64)

        x64 = x_bar.detach().to("cpu", torch.float64)
        h64 = h.detach().to("cpu", torch.float64)
        pre = x64 @ W_xh.T + h64 @ W_hh.T + b_h
        sig = torch.sigmoid(pre)
        I = torch.eye(W_hh.shape[0], dtype=torch.float64)
        J = (1.0 - alpha) * I + alpha * torch.diag_embed(sig.mean(0)) @ W_hh  

    return J.numpy()


def solve_fixed_point(
    model,
    x_bar: torch.Tensor,
    h0: torch.Tensor,
    cfg: SolverCfg,
) -> Tuple[torch.Tensor, float]:
    h = h0.detach().clone().requires_grad_(True)
    opt = torch.optim.SGD([h], lr=cfg.alpha) 
    last = float("inf")

    for _ in range(cfg.max_iter):
        opt.zero_grad()
        r = residual(model, h, x_bar, alpha=cfg.alpha)
        loss = (r ** 2).sum()
        loss.backward()
        opt.step()
        val = float(loss.detach().cpu().item())
        if abs(last - val) < cfg.tol:
            break
        last = val
    with torch.no_grad():
        r = residual(model, h, x_bar, alpha=cfg.alpha)
        res = float(r.norm().item())
    return h.detach(), res

def stability_and_label(J: np.ndarray) -> Dict[str, Any]:
    vals = np.linalg.eigvals(J)
    mags = np.abs(vals)
    rho = float(np.max(mags)) if mags.size else 0.0
    n_lt = int(np.sum(mags < 1.0))
    has_c_lt = bool(np.any((mags < 1.0) & (np.abs(np.imag(vals)) > 1e-9)))
    if rho < 1.0:
        label = "stable"
    elif n_lt > 0 and n_lt < J.shape[0]:
        label = "saddle"
    elif has_c_lt:
        label = "rotational"
    else:
        label = "unstable"
    return {"rho": rho, "label": label}

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="Path to run dir that contains ckpt.pt")
    ap.add_argument("--config", type=str, required=True)
    return ap

def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    fp_cfg = cfg.get("fixed_points", {})
    solver_cfg = fp_cfg.get("solver", {})
    scfg = SolverCfg(
        max_iter=int(solver_cfg.get("max_iter", 500)),
        tol=float(solver_cfg.get("tol", 1e-6)),
        alpha=float(solver_cfg.get("alpha", 1.0)),
    )

    run_dir = Path(args.run)
    ckpt = run_dir / "ckpt.pt"
    eval_dir = run_dir / fp_cfg.get("eval", {}).get("outdir", "eval/fixed_points")
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved = rebuild_model_from_ckpt(ckpt, device=device)

    seeds_npz = eval_dir / "rollout_seeds.npz"
    data = np.load(str(seeds_npz))
    X_np = data["X"]               
    H0_np = data["H0"]    

    H = model.W_hh.shape[0]
    Din = model.W_xh.shape[1]

    if H0_np.shape[1] != H:
        X_bar_full = torch.from_numpy(X_np[:, -1, :]).to(device=device, dtype=torch.float32) 
        with torch.no_grad():
            pre = X_bar_full @ model.W_xh.T + model.b_h
            H0 = F.softplus(pre)
        H0_np = H0.detach().cpu().numpy()

    X_bar = torch.from_numpy(X_np[:, -1, :]).to(device=device, dtype=torch.float32) 
    H0 = torch.from_numpy(H0_np).to(device=device, dtype=torch.float32)

    records: List[Dict[str, Any]] = []
    B = X_bar.shape[0]
    for i in range(B):
        h_star, res = solve_fixed_point(model, X_bar[i : i + 1, :], H0[i : i + 1, :], scfg)
        J = jacobian_at(model, h_star, X_bar[i : i + 1, :], alpha=scfg.alpha)
        stab = stability_and_label(J)
        records.append(
            {
                "seed_idx": int(i),
                "residual": float(res),
                "rho": float(stab["rho"]),
                "label": str(stab["label"]),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(eval_dir / "summary.csv", index=False)

    metrics = {
        "n_fp": int((df["residual"] < 1e-3).sum()),
        "stability": {
            "stable": int((df["label"] == "stable").sum()),
            "saddle": int((df["label"] == "saddle").sum()),
            "rotational": int((df["label"] == "rotational").sum()),
            "unstable": int((df["label"] == "unstable").sum()),
        },
        "rho_stats": {
            "mean": float(df["rho"].mean()),
            "std": float(df["rho"].std(ddof=0)),
            "count": int(df.shape[0]),
        },
    }
    with (eval_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[unified_fixed_points] Wrote {eval_dir/'summary.csv'} and {eval_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
