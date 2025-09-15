from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.fixed_points.src.config import load_config


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_hist_rho(df: pd.DataFrame, out_png: Path):
    plt.figure()
    plt.hist(df["rho"].values, bins=30)
    plt.xlabel("spectral radius ρ(J)")
    plt.ylabel("count")
    plt.title("Distribution of spectral radii")
    _ensure_dir(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_stability_bar(df: pd.DataFrame, out_png: Path):
    plt.figure()
    order = ["stable", "saddle", "rotational", "continuous", "unstable"]
    counts = df["label"].value_counts().reindex(order).fillna(0)
    plt.bar(np.arange(len(order)), counts.values)
    plt.xticks(np.arange(len(order)), order, rotation=20)
    plt.ylabel("count")
    plt.title("Motif counts")
    _ensure_dir(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_margin(df: pd.DataFrame, out_png: Path):
    plt.figure()
    plt.hist(df["margin"].values, bins=30)
    plt.xlabel("1 − ρ(J)")
    plt.ylabel("count")
    plt.title("Margins")
    _ensure_dir(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_residuals(df: pd.DataFrame, out_png: Path):
    plt.figure()
    plt.hist(df["residual"].values, bins=30)
    plt.xlabel("||F(h*) − h*||²")
    plt.ylabel("count")
    plt.title("Residuals at fixed points")
    _ensure_dir(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_unit_proximity(eigvals_obj: np.ndarray, out_png: Path):
    mags = []
    for arr in eigvals_obj:
        if arr is None:
            continue
        mags.extend(np.abs(np.asarray(arr)).ravel().tolist())
    mags = np.asarray(mags, dtype=float)
    plt.figure()
    plt.hist(np.abs(mags) - 1.0, bins=50)
    plt.xlabel("|λ| - 1")
    plt.ylabel("count")
    plt.title("Eigenvalue distance to unit circle")
    _ensure_dir(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = load_config(args.config)
    eval_dir = run_dir / cfg.get("fixed_points", {}).get("eval", {}).get("outdir", "eval/fixed_points")

    df = pd.read_csv(eval_dir / "summary.csv")
    fp = np.load(str(eval_dir / "fixed_points.npz"), allow_pickle=True)

    plot_hist_rho(df, eval_dir / "rho_hist.png")
    plot_stability_bar(df, eval_dir / "stability_bar.png")
    plot_margin(df, eval_dir / "margin_hist.png")
    plot_residuals(df, eval_dir / "residual_hist.png")
    plot_unit_proximity(fp["eigvals"], eval_dir / "unit_proximity.png")

    print(f"[plot_fixed_points] wrote figures to {eval_dir}")


if __name__ == "__main__":
    main()
