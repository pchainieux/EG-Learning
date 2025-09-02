from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

EG_COLOR = "#1f77b4"
GD_COLOR = "#d62728"
TARGET_COLOR = "#6c757d" 

def _set_pub_style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

def _gaussian_smooth(y: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    if sigma_bins is None or sigma_bins <= 0:
        return y
    L = int(max(3, np.ceil(6 * sigma_bins)))
    if L % 2 == 0:
        L += 1
    xs = np.arange(L) - L//2
    ker = np.exp(-0.5 * (xs / float(sigma_bins))**2)
    ker /= ker.sum()
    return np.convolve(y, ker, mode="same")

def _common_edges(values: np.ndarray, bins: int, xlim: Optional[Tuple[float,float]]):
    if xlim is not None:
        lo, hi = map(float, xlim)
    else:
        vmax = np.percentile(np.abs(values), 99.5) if values.size else 1.0
        lo, hi = -float(vmax), float(vmax)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, int(bins) + 1)

def plot_whh_distribution(W_hh, sign_vec, outpath: str, *, bins: int = 150,
                          xlim: Optional[Tuple[float,float]] = None,
                          smooth_sigma_bins: float = 1.5,
                          density: bool = True,
                          reflect_inhibitory: bool = True,
                          draw_medians: bool = True,
                          draw_means: bool = False,
                          title: Optional[str] = "Recurrent weights distribution (columns)"):
    W = np.asarray(W_hh, dtype=float)
    H = W.shape[0]
    s = np.asarray(sign_vec, dtype=float).reshape(-1) if sign_vec is not None else np.ones(H)

    exc_cols = W[:, s > 0].ravel()
    inh_cols = W[:, s < 0].ravel()

    if reflect_inhibitory:
        e_vals = np.abs(exc_cols)
        i_vals = -np.abs(inh_cols)
        max_mag = float(np.max(np.abs(np.concatenate([e_vals, -i_vals]))) if (e_vals.size or i_vals.size) else 1.0)
        x_lo, x_hi = -max_mag, max_mag
        xlabel = "signed magnitude (I reflected)"
    else:
        e_vals = exc_cols
        i_vals = inh_cols
        if xlim is not None:
            x_lo, x_hi = map(float, xlim)
        else:
            vmax = float(np.percentile(np.abs(np.concatenate([e_vals, i_vals])), 99.5)) if (e_vals.size or i_vals.size) else 1.0
            x_lo, x_hi = -vmax, vmax
        xlabel = "weight"

    edges = np.linspace(x_lo, x_hi, int(bins) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    binw = float(edges[1] - edges[0])

    def _hist(vals):
        if vals.size == 0:
            return np.zeros_like(centers)
        h, _ = np.histogram(vals, bins=edges, density=density)
        h = _gaussian_smooth(h, smooth_sigma_bins)
        if reflect_inhibitory:
            if np.all(vals >= 0):
                h[centers < 0] = 0.0
            if np.all(vals <= 0):
                h[centers > 0] = 0.0
        if density and h.sum() > 0:
            h = h / (h.sum() * binw)
        return h

    h_e = _hist(e_vals)
    h_i = _hist(i_vals)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _set_pub_style(ax)

    ax.axvline(0.0, color=TARGET_COLOR, lw=1.0, ls=":")

    ax.plot(centers, h_e, lw=2.2, label="E columns", color=EG_COLOR)
    if inh_cols.size:
        ax.plot(centers, h_i, lw=2.2, label="I columns", color=GD_COLOR)

    def _stats(vals):
        if vals.size == 0:
            return np.nan, np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))

    mu_e, sd_e, med_e = _stats(e_vals)
    mu_i, sd_i, med_i = _stats(i_vals) if inh_cols.size else (np.nan, np.nan, np.nan)

    if draw_medians and not np.isnan(med_e):
        ax.axvline(med_e, color=EG_COLOR, ls="--", lw=1.2, alpha=0.9)
    if draw_medians and inh_cols.size and not np.isnan(med_i):
        ax.axvline(med_i, color=GD_COLOR, ls="--", lw=1.2, alpha=0.9)

    if draw_means and not np.isnan(mu_e):
        ax.axvline(mu_e, color=EG_COLOR, ls=":", lw=1.0, alpha=0.9)
    if draw_means and inh_cols.size and not np.isnan(mu_i):
        ax.axvline(mu_i, color=GD_COLOR, ls=":", lw=1.0, alpha=0.9)

    lines = []
    if not np.isnan(mu_e):
        lines.append(fr"E: $\mu={mu_e:.3f}$, $\sigma={sd_e:.3f}$, med={med_e:.3f}")
    if inh_cols.size and not np.isnan(mu_i):
        lines.append(fr"I: $\mu={mu_i:.3f}$, $\sigma={sd_i:.3f}$, med={med_i:.3f}")
    if lines:
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=11,
                bbox=dict(fc="white", ec="0.85", boxstyle="round,pad=0.35"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    if title:
        ax.set_title(title)

    ax.legend(frameon=False)
    ax.set_xlim(x_lo, x_hi) 
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_whh_heatmap_observed(
    W_hh,
    sign_vec,
    outpath: str,
    *,
    title: str = "$W_{hh}$ heatmap (observed order)",
    cmap: str = "RdBu_r",
    strip_size: str = "6%",
    strip_pad: float = 0.10, 
    cb_size: str = "3.5%", 
    cb_pad: float = 0.10, 
):
    import numpy as np
    import matplotlib.pyplot as plt

    W = np.asarray(W_hh, dtype=float)
    H = W.shape[0]
    s = np.asarray(sign_vec, dtype=float).reshape(-1) if sign_vec is not None else np.ones(H)

    vmax = np.percentile(np.abs(W), 99.0) if W.size else 1.0
    vmin = -float(vmax); vmax = float(vmax)

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    _set_pub_style(ax)

    im = ax.imshow(
        W, cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", origin="upper",
        extent=[-0.5, H-0.5, H-0.5, -0.5],
    )
    ax.set_xlabel("pre-synaptic (column)", labelpad=10, fontsize=12)
    ax.set_ylabel("post-synaptic (row)")
    if title:
        ax.set_title(title, pad=8)

    divider = make_axes_locatable(ax)

    ax_top = divider.append_axes("top", size=strip_size, pad=strip_pad, sharex=ax)
    col_vals = ((s > 0).astype(int))[None, :] 
    cm = ListedColormap([GD_COLOR, EG_COLOR])
    ax_top.imshow(
        col_vals, aspect="auto", interpolation="nearest", cmap=cm, vmin=0, vmax=1,
        origin="upper", extent=[-0.5, H-0.5, 0, 1],
    )

    ax_left = divider.append_axes("left", size=strip_size, pad=strip_pad, sharey=ax)
    row_vals = ((s > 0).astype(int))[:, None]
    ax_left.imshow(
        row_vals, aspect="auto", interpolation="nearest", cmap=cm, vmin=0, vmax=1,
        origin="upper", extent=[0, 1, H-0.5, -0.5],
    )

    for a in (ax_top, ax_left):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): 
            sp.set_visible(False)

    ax.set_ylabel("")  
    ax_left.set_ylabel("post-synaptic (row)", rotation=90, labelpad=10, fontsize=12)
    ax_left.yaxis.set_label_position("left")

    cax = divider.append_axes("right", size=cb_size, pad=cb_pad)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _split_by_sign(values_1d: np.ndarray, sign_vec: np.ndarray):
    e = values_1d[sign_vec > 0]
    i = values_1d[sign_vec < 0]
    return e, i

def plot_row_col_sums(W_hh, sign_vec, outpath: str, *, bins: int = 120,
                      smooth_sigma_bins: float = 1.0, density: bool = True,
                      title: Optional[str] = "Row vs Column sums of $W_{hh}$"):
    W = np.asarray(W_hh, dtype=float)
    s = np.asarray(sign_vec, dtype=float).reshape(-1)
    row_sums = W.sum(axis=1) 
    col_sums = W.sum(axis=0) 

    e_rows, i_rows = _split_by_sign(row_sums, s)
    e_cols, i_cols = _split_by_sign(col_sums, s)

    all_vals = np.concatenate([row_sums, col_sums]) if row_sums.size else np.array([0.0])
    edges = _common_edges(all_vals, bins=bins, xlim=None)
    centers = 0.5 * (edges[1:] + edges[:-1])

    def _h(vals):
        h, _ = np.histogram(vals, bins=edges, density=density)
        return _gaussian_smooth(h, smooth_sigma_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), sharey=True)
    for ax in axes:
        _set_pub_style(ax)
        ax.axvline(0.0, color=TARGET_COLOR, lw=1.0, ls=":")

    h_er = _h(e_rows); h_ir = _h(i_rows)
    axes[0].plot(centers, h_er, lw=2.2, color=EG_COLOR, label="E rows")
    axes[0].plot(centers, h_ir, lw=2.2, color=GD_COLOR, label="I rows")
    axes[0].set_xlabel("row sum (incoming)"); axes[0].set_ylabel("density" if density else "count")
    axes[0].legend(frameon=False)

    h_ec = _h(e_cols); h_ic = _h(i_cols)
    axes[1].plot(centers, h_ec, lw=2.2, color=EG_COLOR, label="E cols")
    axes[1].plot(centers, h_ic, lw=2.2, color=GD_COLOR, label="I cols")
    axes[1].set_xlabel("column sum (outgoing)"); axes[1].legend(frameon=False)

    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_spectrum(W_hh, outpath: str, *, title: Optional[str] = "Spectrum of $W_{hh}$"):
    W = np.asarray(W_hh, dtype=float)
    eigs = np.linalg.eigvals(W)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _set_pub_style(ax)

    ax.scatter(eigs.real, eigs.imag, s=14, lw=0, alpha=0.7, color=EG_COLOR, label="eigs")
    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), ls="--", lw=1.2, color=TARGET_COLOR, label="|z|=1")

    rad = np.abs(eigs).max()
    ax.add_artist(plt.Circle((0, 0), rad, fill=False, lw=1.2, color=GD_COLOR, ls=":"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_whh_heatmap(W_hh, sign_vec, outpath: str, *, title: Optional[str] = "$W_{hh}$ heatmap", reorder_by_sign: bool = True):
    W = np.asarray(W_hh, dtype=float)
    H = W.shape[0]
    s = np.asarray(sign_vec, dtype=float).reshape(-1) if sign_vec is not None else np.ones(H)
    n_exc = int((s > 0).sum())

    if reorder_by_sign and sign_vec is not None:
        order = np.concatenate([np.flatnonzero(s > 0), np.flatnonzero(s < 0)])
        W = W[np.ix_(order, order)]

    vmax = np.percentile(np.abs(W), 99.0) if W.size else 1.0
    vmin = -float(vmax); vmax = float(vmax)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    _set_pub_style(ax)
    im = ax.imshow(W, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    ax.set_xlabel("pre-synaptic (column)"); ax.set_ylabel("post-synaptic (row)")
    if title:
        ax.set_title(title)


    ax.axhline(n_exc-0.5, color=TARGET_COLOR, lw=1.0)
    ax.axvline(n_exc-0.5, color=TARGET_COLOR, lw=1.0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def _load_whh_from_ckpt(ckpt_path: str):
    import torch
    pkg = torch.load(ckpt_path, map_location="cpu")
    state = pkg.get("model", pkg)
    W = state["W_hh"].cpu().numpy() if hasattr(state["W_hh"], "cpu") else np.asarray(state["W_hh"])
    sign = state.get("sign_vec", None)
    if sign is not None and hasattr(sign, "cpu"):
        sign = sign.cpu().numpy()
    return W, sign

def plots_from_ckpt(ckpt_path: str, outdir: str):
    import os
    os.makedirs(outdir, exist_ok=True)
    W, s = _load_whh_from_ckpt(ckpt_path)
    plot_whh_distribution(W, s, os.path.join(outdir, "whh_distribution.png"))
    plot_row_col_sums(W, s, os.path.join(outdir, "whh_row_col_sums.png"))
    plot_spectrum(W, os.path.join(outdir, "whh_spectrum.png"))
    plot_whh_heatmap(W, s, os.path.join(outdir, "whh_heatmap.png"))
