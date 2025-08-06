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


