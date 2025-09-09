# run_fixed_points.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from neurogym import Dataset

from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt

# Import the FP ops (try local, then package path fallback)
try:
    from unified_fixed_points import step_F, solve_one_fp, jacobian_at, classify_from_eigs
except ModuleNotFoundError:  # optional fallback if your file lives inside a package
    from experiments.fixed_points.scripts_2.unified_fixed_points import (  # type: ignore
        step_F, solve_one_fp, jacobian_at, classify_from_eigs
    )


# -----------------------
# Minimal config helpers
# -----------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


# -----------------------
# Checkpoint helpers
# -----------------------

def _find_ckpt(run_dir: Path) -> Path:
    """Prefer <run_dir>/ckpt.pt; else pick latest singlehead_epoch*.pt."""
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        return ckpt
    cands = sorted(run_dir.glob("singlehead_epoch*.pt"))
    if cands:
        return cands[-1]
    raise FileNotFoundError(
        f"Could not find a checkpoint in {run_dir} "
        "(expected ckpt.pt or singlehead_epoch*.pt)"
    )


# -----------------------
# Core helpers
# -----------------------

def _device_and_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(model.parameters())
    return p.device, p.dtype


def _as_tensor(x: np.ndarray | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _get_model_and_hparams(run_dir: Path, device: Optional[torch.device] = None):
    ckpt = _find_ckpt(run_dir)
    model, saved_cfg = rebuild_model_from_ckpt(ckpt, device=device or torch.device("cpu"))
    model.eval()
    mcfg = saved_cfg.get("model", {})
    leak = float(mcfg.get("leak", 1.0))
    beta = float(mcfg.get("softplus_beta", 8.0))
    print(f"[run_fixed_points] using checkpoint: {ckpt}")
    return model, leak, beta, saved_cfg


def _build_context_input(
    in_dim: int,
    ctx_cfg: Mapping[str, Any],
) -> np.ndarray:
    """
    Build a single context/memory input x_ctx (in_dim,) with:
      - fixation channel = 1 (ctx_cfg['fixation_idx'])
      - rule/index channels ON (ctx_cfg['rule_on_idxs'])
      - stimulus channels OFF (ctx_cfg['stimulus_off_idxs'])
    """
    x_ctx = np.zeros((in_dim,), dtype=np.float32)

    fix_idx = ctx_cfg.get("fixation_idx", ctx_cfg.get("memory", None))
    if fix_idx is not None and 0 <= int(fix_idx) < in_dim:
        x_ctx[int(fix_idx)] = 1.0

    rule_on = ctx_cfg.get("rule_on_idxs", None)
    if rule_on is None:
        for k in ("rule_mem", "rule", "context"):
            if k in ctx_cfg:
                rule_on = [int(ctx_cfg[k])]
                break
    if rule_on is not None:
        if isinstance(rule_on, (int, np.integer)):
            rule_on = [int(rule_on)]
        for idx in rule_on:
            if 0 <= int(idx) < in_dim:
                x_ctx[int(idx)] = 1.0

    # stimulus_off_idxs are already zero by default; we keep this for clarity
    return x_ctx.astype(np.float32)


def _simulate_sequence_hidden(
    model: torch.nn.Module,
    X: np.ndarray,
    leak: float,
    beta: float,
    h0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rollout with the SAME update as the solver:
        h_{t+1} = (1-leak) h_t + leak * softplus(beta*(W_xh x_t + W_hh h_t + b_h))/beta
    X: (T, in_dim) → returns H: (T, hid_dim) with H[t] after consuming x_t.
    """
    device, dtype = _device_and_dtype(model)
    W_xh = model.W_xh.to(device=device, dtype=dtype)
    W_hh = model.W_hh.to(device=device, dtype=dtype)
    b_h  = model.b_h.to(device=device, dtype=dtype)

    T, _ = X.shape
    hid_dim = W_hh.shape[0]
    h = torch.zeros(hid_dim, device=device, dtype=dtype) if h0 is None else _as_tensor(h0, device, dtype).view(-1)
    H = []
    with torch.no_grad():
        for t in range(T):
            xt = _as_tensor(X[t], device, dtype).view(1, -1)
            pre = xt @ W_xh.T + h.view(1, -1) @ W_hh.T + b_h
            phi = F.softplus(beta * pre) / beta
            h = (1.0 - leak) * h + leak * phi.view(-1)
            H.append(h.clone())
    return torch.stack(H, dim=0).detach().cpu().numpy()


def _find_memory_start_indices(
    X: np.ndarray,
    ctx_cfg: Mapping[str, Any],
) -> List[int]:
    """
    Heuristic: earliest t where fixation==1 and all stimulus_off_idxs==0; else T-1.
    X: (B, T, in_dim)
    """
    B, T, D = X.shape
    t_list: List[int] = []
    fix_idx = ctx_cfg.get("fixation_idx", ctx_cfg.get("memory", None))
    stim_off = ctx_cfg.get("stimulus_off_idxs", [])
    if isinstance(stim_off, (int, np.integer)):
        stim_off = [int(stim_off)]
    for b in range(B):
        t_found = None
        for t in range(T):
            cond_fix = True
            if fix_idx is not None and 0 <= int(fix_idx) < D:
                cond_fix = np.isclose(X[b, t, int(fix_idx)], 1.0, atol=1e-3)
            cond_stim = True
            if stim_off:
                cond_stim = all(np.isclose(X[b, t, int(k)], 0.0, atol=1e-3) for k in stim_off if 0 <= int(k) < D)
            if cond_fix and cond_stim:
                t_found = t
                break
        t_list.append(t_found if t_found is not None else (T - 1))
    return t_list


def _collect_rollout_and_seeds(
    model: torch.nn.Module,
    ds: Dataset,
    leak: float,
    beta: float,
    ctx_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X (B,T,D), H_seed (B,H), and x_ctx (D,).
    Seeds are actual hidden states at the detected start-of-memory per trial.
    """
    device, dtype = _device_and_dtype(model)
    X, _ = ds()  # numpy arrays
    X = torch.from_numpy(X).to(device=device, dtype=dtype).cpu().numpy()
    B, T, D = X.shape

    x_ctx = _build_context_input(D, ctx_cfg)

    H_seed = np.zeros((B, int(model.W_hh.shape[0])), dtype=np.float32)
    for b in range(B):
        H_traj = _simulate_sequence_hidden(model, X[b], leak=leak, beta=beta, h0=None)
        t_mem = _find_memory_start_indices(X[b][None, ...], ctx_cfg)[0]
        H_seed[b] = H_traj[t_mem]

    return X, H_seed, x_ctx


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--run", type=str, required=True, help="Path to the training run directory (contains ckpt.pt or singlehead_epoch*.pt)")
    ap.add_argument("--plot", action="store_true", help="Make ring panel after solving FPs")
    ap.add_argument("--overrides", type=str, default=None, help="JSON string of config overrides for quick tweaks")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        try:
            deep_update(cfg, json.loads(args.overrides))
        except Exception as e:
            raise ValueError(f"--overrides is not valid JSON: {e}")

    run_dir = Path(args.run).resolve()
    eval_dir = run_dir / "eval" / "fixed_points"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leak, beta, saved_cfg = _get_model_and_hparams(run_dir, device=device)
    device, dtype = _device_and_dtype(model)

    # Build a dataset for gathering diverse seeds
    fp_cfg = cfg.get("fixed_points", {})
    rollout_cfg = fp_cfg.get("rollout", {})
    batch = int(rollout_cfg.get("batch", 256))
    seq_len = int(rollout_cfg.get("seq_len", 80))
    env_name = rollout_cfg.get("env", saved_cfg.get("data", {}).get("env", "ModalityCuedChoice-v0"))

    # NeuroGym Dataset accepts env name string
    ds = Dataset(env_name, batch_size=batch, seq_len=seq_len, batch_first=True)

    # Context/memory channels
    ctx_cfg = rollout_cfg.get("context_channels", {})
    X, H0, x_ctx = _collect_rollout_and_seeds(model, ds, leak=leak, beta=beta, ctx_cfg=ctx_cfg)

    # Solve fixed points under constant x_ctx
    x_bar = torch.from_numpy(x_ctx[None, :]).to(device=device, dtype=dtype)  # (1, D)

    H_star_list: List[np.ndarray] = []
    Eigs_list: List[np.ndarray] = []
    Resid_list: List[float] = []

    solver_cfg = fp_cfg.get("solver", {})
    lr = float(solver_cfg.get("lr", 1e-1))
    tol = float(solver_cfg.get("tol", 1e-10))
    max_iter = int(solver_cfg.get("max_iter", 5000))
    line_beta = float(solver_cfg.get("line_beta", 0.5))
    line_c = float(solver_cfg.get("line_c", 1e-4))

    for i in range(H0.shape[0]):
        h0 = torch.from_numpy(H0[i][None, :]).to(device=device, dtype=dtype)
        h_star, res, iters = solve_one_fp(
            model, x_bar, h0, leak=leak, beta=beta,
            cfg=type("Cfg", (), dict(lr=lr, tol=tol, max_iter=max_iter, line_beta=line_beta, line_c=line_c)),
        )
        H_star_list.append(h_star.detach().cpu().numpy().squeeze(0))
        Resid_list.append(res)
        # eigenvalues of Jacobian at fixed point
        J = jacobian_at(model, h_star, x_bar, leak=leak, beta=beta)
        eigs = np.linalg.eigvals(J)
        Eigs_list.append(eigs)

    H_star = np.stack(H_star_list, axis=0)  # (N_fp, H)

    # Save artifacts
    np.savez_compressed(
        str(eval_dir / "fixed_points.npz"),
        H_star=H_star,
        eigvals=np.array(Eigs_list, dtype=object),
        H0=H0,
        x_ctx=x_ctx,
    )

    # Summary CSV
    import pandas as pd
    labels = [classify_from_eigs(e) for e in Eigs_list]
    rho = [np.max(np.abs(e)) if e is not None and len(e) > 0 else np.nan for e in Eigs_list]
    df = pd.DataFrame({
        "idx": np.arange(H_star.shape[0], dtype=int),
        "residual": np.array(Resid_list, dtype=float),
        "rho": np.array(rho, dtype=float),
        "label": labels,
    })
    df.to_csv(eval_dir / "summary.csv", index=False)

    print(f"[run_fixed_points] wrote {len(H_star_list)} fixed points -> {eval_dir}")

    if args.plot:
        # (plotter imports the model again; ensure its bias name uses W_out.bias in your plot script)
        try:
            from plot_ring_panels import plot_ring_panel_for_run
        except ModuleNotFoundError:
            from experiments.fixed_points.scripts_2.plot_ring_panels import plot_ring_panel_for_run  # type: ignore
        fig_path = eval_dir / "ring_panel.png"
        plot_ring_panel_for_run(run_dir, cfg, outfile=fig_path)
        print(f"[run_fixed_points] wrote figure -> {fig_path}")


if __name__ == "__main__":
    main()
