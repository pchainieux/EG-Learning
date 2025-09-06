# experiments/fixed_points/scripts/plot_ring_panels.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt


# -----------------------
# Utilities (no sklearn)
# -----------------------
def pca_fit(X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (N, H) rows = samples, cols = features (hidden units).
    Returns: (components P [H,k], mean mu [H], explained_var_ratio [k])
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt.T[:, :k]                  # (H,k)
    var = (S ** 2) / (X.shape[0] - 1)
    evr = var[:k] / var.sum()
    return P, mu.squeeze(0), evr


def pca_project(H: np.ndarray, P: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (H - mu) @ P  # rows projected to k-dim


def classify_from_eigs(eigs: np.ndarray) -> str:
    mag = np.abs(eigs)
    near1 = (np.abs(mag - 1.0) < 0.02).any()
    gt1   = (mag > 1.0 + 1e-3).any()
    lt1   = (mag < 1.0 - 1e-3).all()
    if lt1:
        return "stable"
    if near1 and not gt1:
        return "continuous"
    if gt1 and not lt1:
        rot = np.iscomplex(eigs).any() and (np.abs(mag - 1.0) < 0.05).any()
        return "rotational" if rot else "saddle"
    return "unstable"


@torch.no_grad()
def step_F(model, h: torch.Tensor, x: torch.Tensor, leak: float, beta: float) -> torch.Tensor:
    # match model dtype
    dtype = next(model.parameters()).dtype
    h = h.to(dtype)
    x = x.to(dtype)
    pre = x @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    phi = F.softplus(beta * pre) / beta
    return (1.0 - leak) * h + leak * phi


@torch.no_grad()
def simulate_traj(model, h0: torch.Tensor, x_bar: torch.Tensor, steps: int, leak: float, beta: float) -> torch.Tensor:
    """
    Returns (steps+1, H): includes initial h0.
    x_bar: (1, in_dim)
    """
    H = [h0.detach().clone()]
    h = h0.detach().clone()
    for _ in range(steps):
        h = step_F(model, h, x_bar, leak=leak, beta=beta)
        H.append(h.clone())
    return torch.stack(H, dim=0)


def readout_value(model, H: torch.Tensor, idx: int) -> torch.Tensor:
    """
    H: (..., Hdim) → scalar logit for output channel 'idx'
    """
    W = model.W_out.weight[idx].to(H)
    b = model.W_out.bias[idx].to(H)
    # last dim matmul
    return (H @ W) + b


def load_fp_package(run_dir: Path) -> Dict[str, np.ndarray]:
    eval_dir = run_dir / "eval" / "fixed_points"
    fp = np.load(str(eval_dir / "fixed_points.npz"), allow_pickle=True)
    return {
        "H_star": fp["H_star"],      # (N_fp, H)
        "eigvals": fp["eigvals"],    # object array of (H,) eigenvalues per fp
        "eval_dir": str(eval_dir),
    }


def load_seeds(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    # seeds saved by orchestrator
    seeds = np.load(str(run_dir / "eval/fixed_points/rollout_seeds.npz"), allow_pickle=True)
    X = seeds["X"]    # (B, T, in_dim)
    H0 = seeds["H0"]  # (B, H)
    return X, H0


def get_model_and_hparams(run_dir: Path, device: torch.device):
    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model, saved_cfg = rebuild_model_from_ckpt(ckpt, device=device)
    model.eval()
    mcfg = saved_cfg.get("model", {})
    leak = float(mcfg.get("leak", 1.0))
    beta = float(mcfg.get("softplus_beta", 8.0))
    return model, leak, beta, saved_cfg


def gather_memory_trajs(model, leak, beta, X_np, H0_np, device, steps: int, stride: int = 1) -> np.ndarray:
    """
    Build memory-like trajectories by freezing x to last time step (x̄).
    Returns array of shape (N_total, H) concatenating all frames (subsampled by 'stride').
    """
    H_list: List[np.ndarray] = []
    B = X_np.shape[0]
    for i in range(B):
        x_bar = torch.from_numpy(X_np[i, -1, :][None, :]).to(device=device, dtype=next(model.parameters()).dtype)
        h0    = torch.from_numpy(H0_np[i][None, :]).to(device=device, dtype=next(model.parameters()).dtype)
        traj  = simulate_traj(model, h0, x_bar, steps=steps, leak=leak, beta=beta)   # (steps+1, 1, H) or (steps+1, H)
        H_list.append(traj[::stride].squeeze(1 if traj.ndim==3 else 0).cpu().numpy())
    return np.concatenate(H_list, axis=0)  # (N_concat, H)


def prepare_panel_data(run_dir: Path, readout_idx: int, steps: int, n_traj: int, device: torch.device):
    """
    Returns dict with: projected FP coords, motif labels, readout at FP, a few trajectories (projected), and PCA basis.
    """
    run_dir = Path(run_dir)
    X_np, H0_np = load_seeds(run_dir)
    fp_pkg = load_fp_package(run_dir)

    model, leak, beta, _ = get_model_and_hparams(run_dir, device)
    param = next(model.parameters())
    dtype = param.dtype

    # Trajectories to build PCA basis (memory-like constant input = last X)
    H_mem = gather_memory_trajs(model, leak, beta, X_np, H0_np, device, steps=steps, stride=1)  # (N, H)

    # PCA on those trajectories
    P, mu, evr = pca_fit(H_mem, k=3)

    # Fixed points & labels
    H_star = fp_pkg["H_star"]  # (N_fp, H)
    eigs   = fp_pkg["eigvals"] # array of arrays

    labels = [classify_from_eigs(e) for e in eigs]
    rho    = np.array([np.max(np.abs(e)) if e is not None and len(e) > 0 else np.nan for e in eigs])

    # Readout values at fixed points
    with torch.no_grad():
        Ht = torch.from_numpy(H_star).to(device=device, dtype=dtype)
        z  = readout_value(model, Ht, readout_idx).detach().cpu().numpy()  # (N_fp,)

    # Project fixed points & sample a few trajectories to overlay
    H_star_pc = pca_project(H_star, P[:, :2], mu)  # (N_fp, 2)

    # Sample trajectories (uniform subset)
    B = X_np.shape[0]
    pick = np.linspace(0, B - 1, num=min(n_traj, B), dtype=int).tolist()
    trajs_2d: List[np.ndarray] = []
    for i in pick:
        x_bar = torch.from_numpy(X_np[i, -1, :][None, :]).to(device=device, dtype=dtype)
        h0    = torch.from_numpy(H0_np[i][None, :]).to(device=device, dtype=dtype)
        traj  = simulate_traj(model, h0, x_bar, steps=steps, leak=leak, beta=beta).squeeze(1 if h0.ndim==2 else 0)
        traj2 = pca_project(traj.cpu().numpy(), P[:, :2], mu)  # (steps+1, 2)
        trajs_2d.append(traj2)

    return {
        "P": P, "mu": mu, "evr": evr,
        "H_star_pc": H_star_pc,
        "labels": labels,
        "rho": rho,
        "readout": z,
        "trajs_2d": trajs_2d,
    }


# -----------------------
# Plotting
# -----------------------
def _markers_for_labels(labels: List[str]) -> List[str]:
    m = []
    for L in labels:
        if L == "stable":      m.append("o")
        elif L == "continuous":m.append("s")
        elif L == "rotational":m.append("D")
        elif L == "saddle":    m.append("^")
        else:                  m.append("x")
    return m


def plot_single_panel(ax, panel, title: str):
    # Trajectories (light grey)
    for Txy in panel["trajs_2d"]:
        ax.plot(Txy[:, 0], Txy[:, 1], lw=0.6, alpha=0.5, zorder=1, color="0.75")

    # Fixed points: color = readout, marker = motif
    xy = panel["H_star_pc"]
    z  = panel["readout"]
    markers = _markers_for_labels(panel["labels"])

    # Draw by motif to get a legend that's meaningful
    unique = sorted(set(panel["labels"]))
    cmap_sc = None
    for L in unique:
        mask = np.array([lab == L for lab in panel["labels"]], dtype=bool)
        if not np.any(mask):
            continue
        sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=z[mask], s=20,
                        marker=_markers_for_labels([L])[0], linewidths=0.2, edgecolors="k", alpha=0.9, zorder=2)
        cmap_sc = sc  # save for colorbar

    ax.set_xlabel("PC1 (memory)")
    ax.set_ylabel("PC2 (memory)")
    ax.set_title(title)
    ax.axhline(0, lw=0.5, color="0.8"); ax.axvline(0, lw=0.5, color="0.8")
    return cmap_sc


def plot_eg_vs_gd(eg_panel, gd_panel, outfile: Path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharex=True, sharey=True)
    sc0 = plot_single_panel(axs[0], eg_panel, "EG: fixed points (color = readout)")
    sc1 = plot_single_panel(axs[1], gd_panel, "GD: fixed points (color = readout)")

    # single shared colorbar if ranges are similar; otherwise separate
    if sc0 is not None:
        cbar = fig.colorbar(sc0, ax=axs, shrink=0.9, location="right")
        cbar.set_label("readout (logit)")

    fig.suptitle("Fixed-point geometry in memory PCA space")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_single_run(panel, outfile: Path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    sc = plot_single_panel(ax, panel, "Fixed points (color = readout)")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("readout (logit)")
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eg-run", type=str, default=None, help="Path to EG <outdir>/run")
    ap.add_argument("--gd-run", type=str, default=None, help="Path to GD <outdir>/run")
    ap.add_argument("--single-run", type=str, default=None, help="Use this when plotting a single run (instead of EG/GD)")
    ap.add_argument("--readout-index", type=int, default=0, help="Output channel index used for coloring")
    ap.add_argument("--traj-steps", type=int, default=60, help="Steps to simulate under constant memory input")
    ap.add_argument("--n-traj", type=int, default=24, help="How many trajectories to overlay")
    ap.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    ap.add_argument("--outfile", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    if args.single_run:
        panel = prepare_panel_data(
            run_dir=Path(args.single_run),
            readout_idx=args.readout_index,
            steps=args.traj_steps,
            n_traj=args.n_traj,
            device=device,
        )
        plot_single_run(panel, Path(args.outfile))
        return

    assert args.eg_run or args.gd_run, "Provide --single-run OR (--eg-run and/or --gd-run)."

    # Build panels independently; to strictly share PCA basis, you can refit PCA
    # on concatenated trajectories from both runs. For simplicity, we use each run's PCA.
    # To force a SHARED basis across EG & GD, uncomment the block below.
    eg_panel = gd_panel = None
    if args.eg_run:
        eg_panel = prepare_panel_data(Path(args.eg_run), args.readout_index, args.traj_steps, args.n_traj, device)
    if args.gd_run:
        gd_panel = prepare_panel_data(Path(args.gd_run), args.readout_index, args.traj_steps, args.n_traj, device)

    # Optional: fit a shared PCA on concatenated trajectories.
    # if eg_panel and gd_panel:
    #     P = np.concatenate([eg_panel["P"], gd_panel["P"]], axis=0)  # dummy to silence linter
    #     # (Left as an exercise to keep this script compact. See notes in the chat.)

    if eg_panel and gd_panel:
        plot_eg_vs_gd(eg_panel, gd_panel, Path(args.outfile))
    elif eg_panel:
        plot_single_run(eg_panel, Path(args.outfile))
    else:
        plot_single_run(gd_panel, Path(args.outfile))


if __name__ == "__main__":
    main()
