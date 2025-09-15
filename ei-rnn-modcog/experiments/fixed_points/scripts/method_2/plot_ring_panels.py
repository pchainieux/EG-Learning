from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt
from experiments.fixed_points.scripts.method_2.unified_fixed_points import step_F 


def _find_ckpt(run_dir: Path) -> Path:
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        return ckpt
    cands = sorted(run_dir.glob("singlehead_epoch*.pt"))
    if cands:
        return cands[-1]
    raise FileNotFoundError(
        f"Could not find a checkpoint in {run_dir} (ckpt.pt or singlehead_epoch*.pt)"
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

def _rho_from_eigs(eigs_list: List[np.ndarray]) -> np.ndarray:
    out = np.full(len(eigs_list), np.nan, dtype=float)
    for i, e in enumerate(eigs_list):
        if e is not None and len(e) > 0:
            out[i] = float(np.max(np.abs(e)))
    return out

def _device_and_dtype(model: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(model.parameters())
    return p.device, p.dtype

def pca_fit(X: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt.T[:, :k] 
    evr = (S ** 2) / (len(S) - 1)
    evr = evr / evr.sum()
    return P, mu.squeeze(0), evr[:k]

def pca_project(X: np.ndarray, P: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (X - mu) @ P

def ring_readout_coords(
    model: torch.nn.Module,
    H: torch.Tensor,
    ring_cfg: Mapping[str, Any] | None,
) -> Optional[np.ndarray]:
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
        logits = (H @ model.W_out.weight.T +
                  (model.W_out.bias if model.W_out.bias is not None else 0.0)).detach().cpu().numpy()
    y = logits[:, idxs]
    x = (y * np.cos(angs)).sum(axis=1)
    s = (y * np.sin(angs)).sum(axis=1)
    return np.stack([x, s], axis=1)


def _simulate_traj_clipped(model, h0: torch.Tensor, x_bar: torch.Tensor,
                           steps: int, leak: float, beta: float,
                           clip: float = 1e6) -> torch.Tensor:
    H = [h0.detach().clone()]
    h = h0.detach().clone()
    for _ in range(steps):
        h = step_F(model, h, x_bar, leak=leak, beta=beta)
        if torch.linalg.vector_norm(h).item() > clip:
            break
        H.append(h.clone())
    return torch.stack(H, dim=0)


def plot_ring_panel_for_run(run_dir: Path, cfg: Mapping[str, Any], outfile: Path,
                            steps: int = 10, n_traj: int = 16,
                            standardize: bool = True, use_ring_subset: bool = False,
                            ring_eps: float = 0.03):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leak, beta, saved_cfg = _get_model_and_hparams(run_dir, device=device)
    H_star, eigs, H0, x_ctx = _load_fp_artifacts(run_dir)

    rho = _rho_from_eigs(eigs)
    mask_ring = np.isfinite(rho) & (np.abs(rho - 1.0) < ring_eps)
    if use_ring_subset and mask_ring.any():
        H_plot = H_star[mask_ring]
    else:
        H_plot = H_star

    if standardize:
        mu_u = H_plot.mean(axis=0, keepdims=True)
        std_u = H_plot.std(axis=0, keepdims=True) + 1e-8
        H_for_pca = (H_plot - mu_u) / std_u
        P, mu, _ = pca_fit(H_for_pca, k=2)
        H_all_std = (H_star - mu_u) / std_u
        fp_pc = pca_project(H_all_std, P, mu)
    else:
        P, mu, _ = pca_fit(H_plot, k=2)
        fp_pc = pca_project(H_star, P, mu)

    ring_cfg = cfg.get("fixed_points", {}).get("ring_readout", None)
    with torch.no_grad():
        Ht = torch.from_numpy(H_star).to(device=device, dtype=next(model.parameters()).dtype)
        logits = (Ht @ model.W_out.weight.T +
                  (model.W_out.bias if model.W_out.bias is not None else 0.0)).detach().cpu().numpy()

    dtype = next(model.parameters()).dtype
    x_bar = torch.from_numpy(x_ctx[None, :]).to(device=device, dtype=dtype)
    pick = np.linspace(0, min(H0.shape[0]-1, 4*n_traj-1), num=min(n_traj, H0.shape[0]), dtype=int).tolist()
    trajs_pc: List[np.ndarray] = []
    for i in pick:
        h0 = torch.from_numpy(H0[i][None, :]).to(device=device, dtype=dtype)
        H_traj = _simulate_traj_clipped(model, h0, x_bar, steps=steps, leak=leak, beta=beta)
        if standardize:
            H_traj_np = H_traj.detach().cpu().numpy()
            H_traj_std = (H_traj_np - mu_u.squeeze(0)) / std_u.squeeze(0)
            trajs_pc.append(pca_project(H_traj_std, P, mu))
        else:
            trajs_pc.append(pca_project(H_traj.detach().cpu().numpy(), P, mu))

    fp_rr = None
    if ring_cfg is not None:
        with torch.no_grad():
            Ht = torch.from_numpy(H_star).to(device=device, dtype=dtype)
        fp_rr = ring_readout_coords(model, Ht, ring_cfg)

    ncols = 2 if fp_rr is not None else 1
    fig, axs = plt.subplots(1, ncols, figsize=(10 if ncols == 2 else 6, 4), constrained_layout=True)

    ax0 = axs if ncols == 1 else axs[0]
    cvals = None
    color_idx = None
    if ring_cfg is not None and "color_readout_idx" in ring_cfg:
        color_idx = int(ring_cfg["color_readout_idx"])
    if color_idx is not None and 0 <= color_idx < logits.shape[1]:
        cvals = logits[:, color_idx]
        sc = ax0.scatter(fp_pc[:, 0], fp_pc[:, 1], c=cvals, s=10, alpha=0.9, cmap="viridis")
        cb = plt.colorbar(sc, ax=ax0, shrink=0.9)
        cb.set_label("readout (logit)")
    else:
        ax0.scatter(fp_pc[:, 0], fp_pc[:, 1], s=10, alpha=0.9)

    for tr in trajs_pc:
        ax0.plot(tr[:, 0], tr[:, 1], linewidth=0.8, alpha=0.7)

    ax0.set_title("Fixed points & short flows (blank delay)")
    ax0.set_xlabel("PC 1")
    ax0.set_ylabel("PC 2")
    ax0.axis("equal")

    if fp_rr is not None:
        ax1 = axs[1]
        ax1.scatter(fp_rr[:, 0], fp_rr[:, 1], s=10, alpha=0.9)
        t = np.linspace(0, 2*np.pi, 200)
        ax1.plot(np.cos(t), np.sin(t), linestyle="--", linewidth=0.8, alpha=0.6)
        ax1.set_title("Fixed points in ring readout space")
        ax1.set_xlabel("dim 1")
        ax1.set_ylabel("dim 2")
        ax1.axis("equal")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
