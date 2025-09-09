# plot_ring_panels.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt

# Try local import first, then package path (so this works whether you run with -m or directly)
from experiments.fixed_points.scripts_2.unified_fixed_points import step_F  # type: ignore


# -----------------------
# Small helpers
# -----------------------

def _device_and_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(model.parameters())
    return p.device, p.dtype


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


def _get_model_and_hparams(run_dir: Path, device: torch.device):
    ckpt = _find_ckpt(run_dir)
    model, saved_cfg = rebuild_model_from_ckpt(ckpt, device=device)
    model.eval()
    mcfg = saved_cfg.get("model", {})
    leak = float(mcfg.get("leak", 1.0))
    beta = float(mcfg.get("softplus_beta", 8.0))
    print(f"[plot_ring_panels] using checkpoint: {ckpt}")
    return model, leak, beta, saved_cfg


def _load_fp_artifacts(run_dir: Path):
    npz = np.load(run_dir / "eval" / "fixed_points" / "fixed_points.npz", allow_pickle=True)
    H_star = npz["H_star"]
    eigs = list(npz["eigvals"])
    H0 = npz["H0"]
    x_ctx = npz["x_ctx"]
    return H_star, eigs, H0, x_ctx


# -----------------------
# PCA utilities
# -----------------------

def pca_fit(X: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Centered PCA basis via SVD.
    X: (N, H)
    Returns (P, mu, evr) where columns of P are principal axes.
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt.T[:, :k]  # (H, k)
    evr = (S ** 2) / (len(S) - 1)
    evr = evr / evr.sum()
    return P, mu.squeeze(0), evr[:k]


def pca_project(X: np.ndarray, P: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (X - mu) @ P


# -----------------------
# Readout-ring projection (optional)
# -----------------------

def ring_readout_coords(
    model: torch.nn.Module,
    H: torch.Tensor,
    ring_cfg: Mapping[str, Any] | None,
) -> Optional[np.ndarray]:
    """
    If ring_cfg supplies 'indices' and 'angles' (radians), compute:
        x = sum_i y_i cos(theta_i), y = sum_i y_i sin(theta_i),
    where y = W_out H + bias (logits). Returns (N, 2) or None if not configured.
    """
    if ring_cfg is None:
        return None
    idxs = ring_cfg.get("indices", None)
    angs = ring_cfg.get("angles", None)
    if idxs is None or angs is None:
        return None
    idxs = list(map(int, idxs))
    angs = np.asarray(angs, dtype=np.float64)
    if len(idxs) != len(angs):
        return None

    with torch.no_grad():
        logits = (H @ model.W_out.weight.T + (model.W_out.bias if model.W_out.bias is not None else 0.0)).detach().cpu().numpy()  # (N, out_dim)
    y = logits[:, idxs]  # (N, K)
    x = (y * np.cos(angs)).sum(axis=1)
    s = (y * np.sin(angs)).sum(axis=1)
    return np.stack([x, s], axis=1)  # (N, 2)


# -----------------------
# Plotting
# -----------------------

def _simulate_traj(model, h0: torch.Tensor, x_bar: torch.Tensor, steps: int, leak: float, beta: float) -> torch.Tensor:
    H = [h0.detach().clone()]
    h = h0.detach().clone()
    for _ in range(steps):
        h = step_F(model, h, x_bar, leak=leak, beta=beta)
        H.append(h.clone())
    return torch.stack(H, dim=0)


def _scatter_with_trajs(ax, fp_xy: np.ndarray, trajs_xy: List[np.ndarray], title: str, cval: Optional[np.ndarray] = None):
    if cval is None:
        ax.scatter(fp_xy[:, 0], fp_xy[:, 1], s=10, alpha=0.9)
    else:
        sc = ax.scatter(fp_xy[:, 0], fp_xy[:, 1], s=10, alpha=0.9, c=cval, cmap="viridis")
        cb = plt.colorbar(sc, ax=ax, shrink=0.9)
        cb.set_label("readout (logit)")
    for tr in trajs_xy:
        ax.plot(tr[:, 0], tr[:, 1], linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.axis("equal")


def plot_ring_panel_for_run(run_dir: Path, cfg: Mapping[str, Any], outfile: Path, steps: int = 50, n_traj: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leak, beta, saved_cfg = _get_model_and_hparams(run_dir, device=device)
    H_star, eigs, H0, x_ctx = _load_fp_artifacts(run_dir)

    dtype = next(model.parameters()).dtype
    x_bar = torch.from_numpy(x_ctx[None, :]).to(device=device, dtype=dtype)

    # Optional readout ring config
    ring_cfg = cfg.get("fixed_points", {}).get("ring_readout", None)

    # Readout at fixed points (for coloring)
    with torch.no_grad():
        Ht = torch.from_numpy(H_star).to(device=device, dtype=dtype)
        logits = (Ht @ model.W_out.weight.T + (model.W_out.bias if model.W_out.bias is not None else 0.0)).detach().cpu().numpy()  # (N_fp, out_dim)

    # PCA fit on short memory trajectories (under constant x_ctx) for subspace
    pick = np.linspace(0, H0.shape[0] - 1, num=min(n_traj, H0.shape[0]), dtype=int).tolist()
    trajs = []
    for i in pick:
        h0 = torch.from_numpy(H0[i][None, :]).to(device=device, dtype=dtype)
        H_traj = _simulate_traj(model, h0, x_bar, steps=steps, leak=leak, beta=beta).squeeze(1 if h0.ndim == 2 else 0)
        trajs.append(H_traj.detach().cpu().numpy())  # (steps+1, H)

    # Concatenate for PCA (fallback to FP set if degenerate)
    H_concat = np.concatenate(trajs, axis=0) if len(trajs) > 0 else H_star
    if H_concat.shape[0] < 3:
        H_concat = H_star
    P, mu, _ = pca_fit(H_concat, k=2)

    # Project fixed points and trajectories
    fp_pc = pca_project(H_star, P, mu)  # (N_fp, 2)
    trajs_pc = [pca_project(tr, P, mu) for tr in trajs]

    # Also try ring-readout projection if configured
    fp_rr = None
    if ring_cfg is not None:
        with torch.no_grad():
            Ht = torch.from_numpy(H_star).to(device=device, dtype=dtype)
        fp_rr = ring_readout_coords(model, Ht, ring_cfg)

    # Build figure
    fig, axs = plt.subplots(1, 2 if fp_rr is not None else 1, figsize=(10 if fp_rr is not None else 5, 4), constrained_layout=True)

    # Panel A: PCA space
    ax0 = axs if fp_rr is None else axs[0]
    cvals = None
    color_idx = None
    if ring_cfg is not None and "color_readout_idx" in ring_cfg:
        color_idx = int(ring_cfg["color_readout_idx"])
    if color_idx is not None and 0 <= color_idx < logits.shape[1]:
        cvals = logits[:, color_idx]
    _scatter_with_trajs(ax0, fp_pc, [pca_project(tr, P, mu) for tr in trajs], "Fixed points & flows in memory PCA space", cval=cvals)

    # Panel B: ring readout space (if available)
    if fp_rr is not None:
        ax1 = axs[1]
        _scatter_with_trajs(ax1, fp_rr, [], "Fixed points in ring readout space")
        # draw unit circle for reference
        t = np.linspace(0, 2*np.pi, 200)
        ax1.plot(np.cos(t), np.sin(t), linestyle="--", linewidth=0.8, alpha=0.6)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
