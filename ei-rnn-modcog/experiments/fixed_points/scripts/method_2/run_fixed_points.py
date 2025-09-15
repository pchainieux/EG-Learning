from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from neurogym import Dataset

from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt
from experiments.fixed_points.scripts.method_2.unified_fixed_points import ( 
        step_F, solve_one_fp, jacobian_at, classify_from_eigs
    )

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

def _find_ckpt(run_dir: Path) -> Path:
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        return ckpt
    cands = sorted(run_dir.glob("singlehead_epoch*.pt"))
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"Could not find a checkpoint in {run_dir} (ckpt.pt or singlehead_epoch*.pt)")

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

def _build_context_input(in_dim: int, ctx_cfg: Mapping[str, Any]) -> np.ndarray:
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
    return x_ctx

def _simulate_sequence_hidden(
    model: torch.nn.Module, X: np.ndarray, leak: float, beta: float, h0: Optional[np.ndarray] = None
) -> np.ndarray:
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

def _find_delay_middle_via_stim_off(X: np.ndarray, stim_idxs: List[int]) -> List[int]:
    B, T, D = X.shape
    t_list = []
    eps = 1e-6
    stim_idxs = [int(i) for i in stim_idxs] if stim_idxs else []

    for b in range(B):
        S = np.abs(np.take(X[b], stim_idxs, axis=-1)).sum(axis=-1) if stim_idxs else np.zeros(T, dtype=float)
        last_on = np.where(S > eps)[0]

        if len(last_on) == 0:
            t_list.append(int(S.argmin()))
            continue

        t0 = int(last_on[-1] + 1)
        
        if t0 >= T:
            t_list.append(T - 1)
            continue
            
        t1 = t0
        while t1 < T and S[t1] <= eps:
            t1 += 1

        mid = t0 + max(0, (t1 - t0 - 1) // 2)
        t_list.append(min(mid, T - 1))

    return t_list

def _collect_rollout_and_seeds(
    model: torch.nn.Module,
    ds: Dataset,
    leak: float,
    beta: float,
    ctx_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, _ = ds()
    B, T, D = X.shape
    x_ctx = _build_context_input(D, ctx_cfg)
    stim_off = ctx_cfg.get("stimulus_off_idxs", [])
    t_mems = _find_delay_middle_via_stim_off(X, stim_off)
    H_seed = np.zeros((B, int(model.W_hh.shape[0])), dtype=np.float32)
    for b in range(B):
        H_traj = _simulate_sequence_hidden(model, X[b], leak=leak, beta=beta, h0=None)
        H_seed[b] = H_traj[t_mems[b]]
    return X, H_seed, x_ctx

def _make_dataset_with_aliases(env_name: str, *, batch: int, seq_len: int):
    candidates = [env_name]
    if "." not in env_name and "/" not in env_name:
        candidates += [f"yang19.{env_name}", f"{env_name}-v0"]
    last_err = None
    for cand in candidates:
        try:
            return Dataset(cand, batch_size=batch, seq_len=seq_len, batch_first=True)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not instantiate Dataset. Tried: {candidates}. Last error: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--overrides", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        deep_update(cfg, json.loads(args.overrides))

    run_dir = Path(args.run).resolve()
    eval_dir = run_dir / "eval" / "fixed_points"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leak, beta, saved_cfg = _get_model_and_hparams(run_dir, device=device)
    device, dtype = _device_and_dtype(model)

    fp_cfg = cfg.get("fixed_points", {})
    rollout_cfg = fp_cfg.get("rollout", {})
    batch = int(rollout_cfg.get("batch", 256))
    seq_len = int(rollout_cfg.get("seq_len", 80))
    env_name = rollout_cfg.get("env", "yang19.dlygo")
    ds = _make_dataset_with_aliases(env_name, batch=batch, seq_len=seq_len)

    ctx_cfg = rollout_cfg.get("context_channels", {})
    X, H0, x_ctx = _collect_rollout_and_seeds(model, ds, leak=leak, beta=beta, ctx_cfg=ctx_cfg)
    x_bar = torch.from_numpy(x_ctx[None, :]).to(device=device, dtype=dtype)

    solver_cfg = fp_cfg.get("solver", {})
    lr = float(solver_cfg.get("lr", 1e-1))
    tol = float(solver_cfg.get("tol", 1e-10))
    max_iter = int(solver_cfg.get("max_iter", 5000))
    line_beta = float(solver_cfg.get("line_beta", 0.5))
    line_c = float(solver_cfg.get("line_c", 1e-4))
    lam_prox = float(solver_cfg.get("lam_prox", 1e-3))

    H_star_list: List[np.ndarray] = []
    Eigs_list: List[np.ndarray] = []
    Resid_list: List[float] = []

    for i in range(H0.shape[0]):
        h0 = torch.from_numpy(H0[i][None, :]).to(device=device, dtype=dtype)
        h_star, res, iters = solve_one_fp(
            model, x_bar, h0, leak=leak, beta=beta,
            cfg=type("Cfg", (), dict(
                lr=lr, tol=tol, max_iter=max_iter,
                line_beta=line_beta, line_c=line_c, lam_prox=lam_prox
            )),
        )
        H_star_list.append(h_star.detach().cpu().numpy().squeeze(0))
        Resid_list.append(res)
        J = jacobian_at(model, h_star, x_bar, leak=leak, beta=beta)
        Eigs_list.append(np.linalg.eigvals(J))

    H_star = np.stack(H_star_list, axis=0)

    np.savez_compressed(
        str(eval_dir / "fixed_points.npz"),
        H_star=H_star,
        eigvals=np.array(Eigs_list, dtype=object),
        H0=H0,
        x_ctx=x_ctx,
    )

    import pandas as pd
    labels = [classify_from_eigs(e) for e in Eigs_list]
    rho = [np.max(np.abs(e)) if e is not None and len(e) > 0 else np.nan for e in Eigs_list]
    pd.DataFrame({
        "idx": np.arange(H_star.shape[0], dtype=int),
        "residual": np.array(Resid_list, dtype=float),
        "rho": np.array(rho, dtype=float),
        "label": labels,
    }).to_csv(eval_dir / "summary.csv", index=False)

    print(f"[run_fixed_points] wrote {len(H_star_list)} fixed points -> {eval_dir}")

    if args.plot:
        from experiments.fixed_points.scripts.method_2.plot_ring_panels import plot_ring_panel_for_run
        fig_path = eval_dir / "ring_panel.png"
        plot_ring_panel_for_run(run_dir, cfg, outfile=fig_path)
        print(f"[run_fixed_points] wrote figure -> {fig_path}")

if __name__ == "__main__":
    main()
