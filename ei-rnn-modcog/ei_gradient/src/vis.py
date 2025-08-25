# experiments/credit_assignment/creditlib/vis.py
from __future__ import annotations
from typing import Optional, Sequence, Dict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
Matplotlib helpers for consistent, publication-friendly figures.
No external style dependencies to keep things portable.
"""

_DEF_FIGSIZE_TC = (6.0, 3.0)
_DEF_FIGSIZE_DIST = (6.0, 3.0)
_DEF_FIGSIZE_GRID = (6.0, 4.5)

def _maybe_save(fig: plt.Figure, outpath: Optional[Path]) -> None:
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timecourses(
    t: np.ndarray,
    series: Dict[str, np.ndarray],
    title: str = "",
    ylabel: str = "",
    outpath: Optional[Path] = None
) -> None:
    """
    Plot multiple time-series (e.g., tc_l2_E vs tc_l2_I).
    'series' is a dict name -> y[t].
    """
    fig, ax = plt.subplots(1, 1, figsize=_DEF_FIGSIZE_TC)
    for name, y in series.items():
        ax.plot(t, y, label=name, linewidth=2)
    ax.set_xlabel("time (t)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    _maybe_save(fig, outpath)


def plot_distributions(
    data: Dict[str, np.ndarray],
    title: str = "",
    xlabel: str = "",
    outpath: Optional[Path] = None,
    bins: int = 50,
) -> None:
    """
    Overlay histograms for several named vectors (e.g., Wbp_E vs Wbp_I).
    """
    fig, ax = plt.subplots(1, 1, figsize=_DEF_FIGSIZE_DIST)
    for name, vec in data.items():
        ax.hist(vec, bins=bins, alpha=0.5, density=True, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    _maybe_save(fig, outpath)


def plot_saliency_grid(
    saliency_maps: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    outpath: Optional[Path] = None
) -> None:
    """
    Show a list of 2D maps (e.g., time × channel) as a grid.
    """
    K = len(saliency_maps)
    cols = min(3, K)
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(_DEF_FIGSIZE_GRID[0], _DEF_FIGSIZE_GRID[1] * rows / 1.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < K:
            im = ax.imshow(saliency_maps[i], aspect="auto", interpolation="nearest", cmap=cmap)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _maybe_save(fig, outpath)


def plot_before_after_inputs(
    x_before: np.ndarray,
    x_after: np.ndarray,
    titles: Sequence[str] = ("Before", "After"),
    cmap: str = "viridis",
    outpath: Optional[Path] = None
) -> None:
    """
    Side-by-side visualization of an input (e.g., time × channel) before and after optimisation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    ims = []
    ims.append(axes[0].imshow(x_before, aspect="auto", interpolation="nearest", cmap=cmap))
    axes[0].set_title(titles[0])
    ims.append(axes[1].imshow(x_after, aspect="auto", interpolation="nearest", cmap=cmap))
    axes[1].set_title(titles[1])
    for ax, im in zip(axes, ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("channel")
        ax.set_ylabel("time")
    _maybe_save(fig, outpath)
