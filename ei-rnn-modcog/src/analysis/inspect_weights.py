# python -m scripts.inspect_from_ckpt

from __future__ import annotations
from typing import Dict, Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None

def _to_numpy(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def dale_sign_violations(W_hh, sign_vec) -> Dict[str, float]:
    W = _to_numpy(W_hh)
    s = _to_numpy(sign_vec).reshape(-1)
    assert W.shape[0] == W.shape[1] == s.shape[0], "W must be square HxH; sign_vec length H"

    col_signed = W * s 
    viol_mask = col_signed < 0.0

    n_total = W.size
    n_viol_total = int(viol_mask.sum())

    exc_cols = s > 0
    inh_cols = s < 0
    n_exc = int(np.sum(exc_cols)) * W.shape[0]
    n_inh = int(np.sum(inh_cols)) * W.shape[0]
    n_viol_exc = int(viol_mask[:, exc_cols].sum())
    n_viol_inh = int(viol_mask[:, inh_cols].sum())

    return {
        "viol_total": n_viol_total,
        "viol_frac_total": n_viol_total / float(n_total),
        "viol_exc": n_viol_exc,
        "viol_frac_exc": 0.0 if n_exc == 0 else n_viol_exc / float(n_exc),
        "viol_inh": n_viol_inh,
        "viol_frac_inh": 0.0 if n_inh == 0 else n_viol_inh / float(n_inh),
    }


def row_col_summaries(W_hh) -> Dict[str, np.ndarray]:
    W = _to_numpy(W_hh)
    row_sum = W.sum(axis=1)
    col_sum = W.sum(axis=0)
    row_l2 = np.linalg.norm(W, axis=1)
    col_l2 = np.linalg.norm(W, axis=0)
    return {
        "row_sum": row_sum,
        "col_sum": col_sum,
        "row_l2": row_l2,
        "col_l2": col_l2,
    }


def gram_singular_values(W_hh) -> np.ndarray:
    W = _to_numpy(W_hh)
    s = np.linalg.svd(W, compute_uv=False)
    return s

def save_weight_hist_by_group(W_hh, sign_vec, outpath: str, bins: int = 80,
                              title: Optional[str] = "W_hh distribution (E vs I columns)") -> None:
    _ensure_dir(outpath)
    W = _to_numpy(W_hh)
    s = _to_numpy(sign_vec).reshape(-1)
    assert W.shape[1] == s.shape[0], "sign_vec must match number of columns in W"

    exc_cols = W[:, s > 0].ravel()
    inh_cols = W[:, s < 0].ravel()

    plt.figure()
    if exc_cols.size:
        plt.hist(exc_cols, bins=bins, alpha=0.6, label="E columns")
    if inh_cols.size:
        plt.hist(inh_cols, bins=bins, alpha=0.6, label="I columns")
    if title:
        plt.title(title)
    plt.xlabel("weight"); plt.ylabel("count"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()


def save_row_col_sums(W_hh, outpath: str, title: Optional[str] = "Row/column sums") -> None:
    _ensure_dir(outpath)
    stats = row_col_summaries(W_hh)
    rsum, csum = stats["row_sum"], stats["col_sum"]

    plt.figure()
    plt.plot(rsum, label="row sum")
    plt.plot(csum, label="col sum")
    if title:
        plt.title(title)
    plt.xlabel("index"); plt.ylabel("sum")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()


def save_gram_spectrum(W_hh, outpath: str, title: Optional[str] = "Singular values of W_hh") -> None:
    _ensure_dir(outpath)
    s = gram_singular_values(W_hh)
    s_sorted = np.sort(s)[::-1]
    plt.figure()
    plt.plot(s_sorted, marker="o", markersize=2, lw=1)
    if title:
        plt.title(title)
    plt.xlabel("index (desc)"); plt.ylabel("singular value")
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()


def save_matrix_heatmap(W, outpath: str, title: Optional[str] = None,
                        vmin: Optional[float] = None, vmax: Optional[float] = None,
                        cmap: str = "viridis") -> None:
    _ensure_dir(outpath)
    M = _to_numpy(W)
    plt.figure()
    im = plt.imshow(M, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, fraction=0.046)
    if title:
        plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()


def save_dale_violation_report(W_hh, sign_vec, outpath_txt: str) -> None:
    _ensure_dir(outpath_txt)
    stats = dale_sign_violations(W_hh, sign_vec)
    lines = [f"{k}: {v}" for k, v in stats.items()]
    with open(outpath_txt, "w") as f:
        f.write("\n".join(lines))
