from __future__ import annotations
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch 
except Exception: 
    torch = None

ArrayLike = Union[np.ndarray, "torch.Tensor", List[float]]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    window = int(window)
    if window > len(y):
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    return (c[window:] - c[:-window]) / float(window)

def _maybe_smooth(y: ArrayLike, smooth: int) -> np.ndarray:
    y = _to_numpy(y).astype(float)
    return _moving_average(y, smooth) if smooth and smooth > 1 else y

def _safe_series(y, smooth):
    y = _to_numpy(y).astype(float)
    if len(y) == 0:
        return y
    if smooth and smooth > 1 and len(y) >= smooth:
        y = _moving_average(y, smooth)
    return y

def save_loss_curve(losses, outpath, smooth=0, xlabel="Epoch", ylabel="Loss", title="Training loss"):
    _ensure_dir(outpath)
    y = _safe_series(losses, smooth)
    if len(y) == 0:
        return
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y, lw=1.6, marker="o", markersize=3)
    if len(x) == 1:
        plt.xlim(-0.5, 0.5)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()


def _mask_to_spans(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a boolean 1D mask (length T) into [ (start_idx, end_idx), ... ] spans
    where the mask is True. end_idx is exclusive.
    """
    m = np.asarray(mask_1d).astype(bool).reshape(-1)
    if m.size == 0:
        return []
    # pad to catch trailing run
    padded = np.concatenate([[False], m, [False]])
    starts = np.flatnonzero((~padded[:-1]) & padded[1:])
    ends   = np.flatnonzero(padded[:-1] & (~padded[1:]))
    return list(zip(starts, ends))


def _infer_stim_layout(input_dim: int, obs_name: dict | None) -> tuple[int, int]:
    """
    Infer (ring_dim, n_modalities) from input dimension and (optional) obs_name mapping.

    Convention in mod_cog_tasks.py:
      - channel 0 is fixation
      - channels 1..ring_dim are 'stimulus' for a single modality
      - if multiple modalities are used, 'stimulus' blocks are concatenated:
            [fix] [stim M0 (ring_dim)] [stim M1 (ring_dim)] ...

    Returns (ring_dim, n_modalities).
    """
    if obs_name and isinstance(obs_name, dict) and 'stimulus' in obs_name:
        ring_dim = len(obs_name['stimulus'])
        if ring_dim <= 0 or input_dim <= 1:
            return input_dim - 1, 1
        n_mod = max(1, (input_dim - 1) // ring_dim)
        return ring_dim, n_mod

    # Fallback: try common ring sizes
    candidates = [64, 48, 40, 36, 32, 24, 20, 18, 16, 12, 10, 9, 8]
    for rd in candidates:
        if rd > 0 and (input_dim - 1) % rd == 0:
            return rd, (input_dim - 1) // rd
    # Ultimate fallback: treat it as a single "ring"
    return input_dim - 1, 1


def save_task_trial_overview(
    X: ArrayLike,
    logits: ArrayLike,
    outpath: str,
    *,
    dec_mask: ArrayLike | None = None,
    obs_name: dict | None = None,
    sample_idx: int = 0,
    topk_logits: int = 3,
    title: str | None = None,
    cmap_stim: str = "Blues",
):
    _ensure_dir(outpath)

    X = _to_numpy(X)
    L = _to_numpy(logits)
    if X.ndim == 2:
        X = X[None, ...]
    if L.ndim == 2:
        L = L[None, ...]
    B, T, D = X.shape
    _, _, C = L.shape

    b = min(sample_idx, B - 1)
    Xb = X[b]
    Lb = L[b]

    # Decision mask handling
    M = None
    if dec_mask is not None:
        M = _to_numpy(dec_mask)
        if M.ndim == 2:
            M = M[b]
        M = M.astype(bool).reshape(-1)
        spans = _mask_to_spans(M)
    else:
        spans = []

    ring_dim, n_mod = _infer_stim_layout(D, obs_name)
    n_rows = 2 + n_mod  # fixation + modalities + outputs
    time = np.arange(T)

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })

    # ---- Use constrained layout; do NOT call tight_layout() later ----
    figsize = (10.0, 6.0 + 1.5 * max(0, n_mod - 1))
    try:
        fig = plt.figure(figsize=figsize, layout="constrained")  # mpl >= 3.6
    except TypeError:
        fig = plt.figure(figsize=figsize, constrained_layout=True)  # older mpl

    gs = fig.add_gridspec(n_rows, 1, height_ratios=[1] + [2]*n_mod + [2.5])

    # (1) Fixation input
    ax_fix = fig.add_subplot(gs[0, 0])
    ax_fix.plot(time, Xb[:, 0], lw=1.8)
    ax_fix.set_ylabel("Fixation\ninput")
    ax_fix.set_xlim(0, T - 1)
    ax_fix.set_ylim(-0.05, 1.05)
    for s0, s1 in spans:
        ax_fix.axvspan(s0, s1, color="0.85", alpha=0.6, lw=0)
    ax_fix.grid(alpha=0.2, linestyle=":")

    # (2) Stimulus heatmap(s)
    for m in range(n_mod):
        start = 1 + m * ring_dim
        stop  = start + ring_dim
        stim_block = Xb[:, start:stop]    # (T, ring_dim)
        ax_stim = fig.add_subplot(gs[1 + m, 0], sharex=ax_fix)
        im = ax_stim.imshow(
            stim_block.T, aspect="auto", origin="lower", interpolation="nearest",
            extent=[0, T - 1, 0, ring_dim], cmap=cmap_stim,
        )
        ax_stim.set_ylabel(f"Input\nmodality {m+1}")
        for s0, s1 in spans:
            ax_stim.axvspan(s0, s1, color="0.85", alpha=0.5, lw=0)
        tick_vals = [0, ring_dim//2, ring_dim-1]
        ax_stim.set_yticks(tick_vals)
        ax_stim.set_yticklabels([r"$0^\circ$", r"$180^\circ$", r"$360^\circ$"])
        # Use fig.colorbar so constrained layout knows about it
        cbar = fig.colorbar(im, ax=ax_stim, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    # (3) Network outputs
    ax_out = fig.add_subplot(gs[-1, 0], sharex=ax_fix)
    ax_out.plot(time, Lb[:, 0], lw=1.8, label="fixation logit (0)")

    if C > 1:
        choices = Lb[:, 1:]
        energy  = choices.var(axis=0)
        k = min(topk_logits, choices.shape[1])
        top_idx = np.argsort(energy)[-k:]
        for j in top_idx:
            cls = j + 1
            ax_out.plot(time, choices[:, j], lw=1.4, label=f"choice logit {cls}")

    for s0, s1 in spans:
        ax_out.axvspan(s0, s1, color="0.85", alpha=0.6, lw=0)

    ax_out.set_ylabel("logit")
    ax_out.set_xlabel("time (steps)")
    if title:
        ax_out.set_title(title)
    ax_out.legend(loc="upper right", frameon=False, ncol=1)
    ax_out.grid(alpha=0.2, linestyle=":")

    # No tight_layout() here
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_per_task_accuracy_curves(per_task_acc, outpath, smooth=0, xlabel="Epoch", ylabel="Accuracy",
                                  title="Per-task validation accuracy", legend_loc="best"):
    _ensure_dir(outpath)
    plt.figure()
    any_plotted = False
    for task, arr in per_task_acc.items():
        y = _safe_series(arr, smooth)
        if len(y) == 0:
            continue
        x = np.arange(len(y))
        plt.plot(x, y, lw=1.6, marker="o", markersize=3, label=str(task))
        any_plotted = True
    if not any_plotted:
        plt.close(); return
    max_len = max(len(v) for v in per_task_acc.values() if len(v) > 0)
    if max_len == 1:
        plt.xlim(-0.5, 0.5)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if title: plt.title(title)
    if len(per_task_acc) > 1:
        plt.legend(loc=legend_loc, frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()

def save_logits_time_trace(
    logits: ArrayLike,
    outpath: str,
    dec_mask: Optional[ArrayLike] = None,
    topk: int = 3,
    sample_idx: int = 0,
    title: Optional[str] = "Example trial logits",
    fixation_class: int = 0,
):
    _ensure_dir(outpath)

    L = _to_numpy(logits)
    if L.ndim == 2:
        L = L[None, ...]
    assert L.ndim == 3, f"logits must be (B,T,C) or (T,C); got {L.shape}"
    B, T, C = L.shape

    b = min(sample_idx, B - 1)
    Lb = L[b]

    fix = np.asarray(Lb[:, fixation_class])
    choices = np.delete(Lb, fixation_class, axis=1) if C > 1 else np.empty((T, 0))
    if choices.shape[1] > 0:
        energy = choices.var(axis=0)
        k = min(topk, choices.shape[1])
        top_idx = np.argsort(energy)[-k:]
    else:
        top_idx = np.array([], dtype=int)

    plt.figure()
    plt.plot(fix, label=f"fixation logit ({fixation_class})", lw=1.6)
    for j in top_idx:
        cls = j + (1 if fixation_class == 0 else 0)
        plt.plot(choices[:, j], label=f"choice logit {cls}", lw=1.2)

    if dec_mask is not None:
        M = _to_numpy(dec_mask)
        if M.ndim == 2:
            M = M[b]
        M = M.astype(float).reshape(-1)
        ylo, yhi = plt.ylim()
        plt.plot(M * (yhi - ylo) + ylo, ls="--", label="decision mask (scaled)")

    if title: plt.title(title)
    plt.xlabel("time"); plt.ylabel("logit")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_weight_hists_W_hh(
    W_hh: ArrayLike,
    sign_vec: Optional[ArrayLike],
    outpath: str,
    bins: int = 80,
    title: Optional[str] = "W_hh distribution (E vs I columns)",
):
    _ensure_dir(outpath)
    W = _to_numpy(W_hh)
    H = W.shape[0]
    s = _to_numpy(sign_vec) if sign_vec is not None else np.ones(H)
    exc_cols = W[:, s > 0].ravel()
    inh_cols = W[:, s < 0].ravel()

    plt.figure()
    if exc_cols.size:
        plt.hist(exc_cols, bins=bins, alpha=0.6, label="exc columns")
    if inh_cols.size:
        plt.hist(inh_cols, bins=bins, alpha=0.6, label="inh columns")
    if title: plt.title(title)
    plt.xlabel("weight"); plt.ylabel("count"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()


# ---------- Thesis-grade curve helpers ----------

def _ema(y: np.ndarray, beta: float) -> np.ndarray:
    """Exponential moving average, beta in [0,1)."""
    y = _to_numpy(y).astype(float).reshape(-1)
    if y.size == 0:
        return y
    out = np.empty_like(y)
    m = 0.0
    for i, v in enumerate(y):
        m = beta * m + (1.0 - beta) * v
        out[i] = m
    return out


def _rolling(y: np.ndarray, win: int) -> np.ndarray:
    """Simple rolling mean with window size win."""
    y = _to_numpy(y).astype(float).reshape(-1)
    return _moving_average(y, max(1, int(win)))


def _conf_band(y: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean ± 1 std over a centered window."""
    y = _to_numpy(y).astype(float).reshape(-1)
    if win <= 1 or y.size == 0:
        return y, np.zeros_like(y)
    k = int(win)
    # centered windows: pad at ends
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    means = np.convolve(yp, np.ones(k)/k, mode="valid")
    # rolling std
    sq = np.convolve(yp**2, np.ones(k)/k, mode="valid")
    std = np.sqrt(np.maximum(0.0, sq - means**2))
    return means, std


def save_training_curve(
    y_raw: ArrayLike,
    outprefix: str,
    *,
    x: ArrayLike | None = None,
    xlabel: str = "training steps",
    ylabel: str = "metric",
    title: str | None = None,
    smooth: dict | None = None,      # e.g. {"ema": 0.98} or {"window": 101}
    shade_window: int | None = 0,    # e.g. 101 for mean±std band
):
    """
    Save a single training metric curve with optional smoothing and shaded band.
    Writes both PNG and PDF with the same prefix.
    """
    _ensure_dir(outprefix + ".png")

    y = _to_numpy(y_raw).astype(float).reshape(-1)
    if x is None:
        x = np.arange(len(y))
    else:
        x = _to_numpy(x).astype(float).reshape(-1)
        assert len(x) == len(y), "x and y must have the same length"

    # smoothing
    y_sm = y.copy()
    if smooth:
        if "ema" in smooth and smooth["ema"] is not None:
            y_sm = _ema(y, float(smooth["ema"]))
        elif "window" in smooth and smooth["window"] and smooth["window"] > 1:
            y_sm = _rolling(y, int(smooth["window"]))

    # confidence shading (on raw series, rolled)
    mean_band = None
    std_band = None
    if shade_window and shade_window > 1 and len(y) >= shade_window:
        mean_band, std_band = _conf_band(y, int(shade_window))

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(8.0, 4.0))
    ax = fig.add_subplot(111)

    if mean_band is not None:
        ax.fill_between(x, mean_band - std_band, mean_band + std_band,
                        alpha=0.12, lw=0, label=f"±1 std (win={int(shade_window)})")

    ax.plot(x, y, lw=1.0, alpha=0.35, label="raw")
    ax.plot(x, y_sm, lw=2.0, label="smoothed")

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, ncol=2, loc="best")

    plt.tight_layout()
    plt.savefig(outprefix + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(outprefix + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_multitask_curves(
    curves: dict[str, ArrayLike],
    outprefix: str,
    *,
    x: ArrayLike | None = None,
    xlabel: str = "epochs",
    ylabel: str = "accuracy",
    title: str | None = None,
    smooth: dict | None = None,      # same format as above
):
    """
    Plot multiple task curves on one figure with consistent smoothing.
    Writes PNG and PDF.
    """
    _ensure_dir(outprefix + ".png")

    # compute smoothed versions
    def _apply_smooth(v):
        v = _to_numpy(v).astype(float).reshape(-1)
        if not smooth:
            return v
        if "ema" in smooth and smooth["ema"] is not None:
            return _ema(v, float(smooth["ema"]))
        if "window" in smooth and smooth["window"] and smooth["window"] > 1:
            return _rolling(v, int(smooth["window"]))
        return v

    keys = list(curves.keys())
    series = {k: _to_numpy(curves[k]).astype(float).reshape(-1) for k in keys}
    max_len = max(len(v) for v in series.values()) if keys else 0
    if x is None:
        x = np.arange(max_len)
    else:
        x = _to_numpy(x).astype(float).reshape(-1)

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(8.0, 4.5))
    ax = fig.add_subplot(111)
    any_plotted = False
    for k in keys:
        y = series[k]
        xs = x[:len(y)]
        ys = _apply_smooth(y)
        ax.plot(xs, ys, lw=2.0, marker="o", markersize=3, label=str(k))
        any_plotted = True

    if not any_plotted:
        plt.close(fig); return

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(keys) > 1:
        ax.legend(frameon=False, ncol=2, loc="best")
    ax.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(outprefix + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(outprefix + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
