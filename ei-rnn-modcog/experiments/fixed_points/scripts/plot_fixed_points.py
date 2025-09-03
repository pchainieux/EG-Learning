from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.fixed_points.src.config import load_config  


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="run dir (contains eval/fixed_points)")
    return ap

def plot_hist_rho(df: pd.DataFrame, out_png: Path):
    fig = plt.figure(figsize=(5, 3))
    plt.hist(df["rho"].values, bins=30)
    plt.xlabel("Spectral radius ρ(J)")
    plt.ylabel("Count")
    plt.title("Fixed-point spectral radii")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_stability_bar(df: pd.DataFrame, out_png: Path):
    counts = df["label"].value_counts().reindex(["stable", "saddle", "rotational", "unstable"]).fillna(0)
    fig = plt.figure(figsize=(5, 3))
    counts.plot(kind="bar")
    plt.ylabel("Count")
    plt.title("Fixed-point stability")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run)
    eval_dir = run_dir / "eval" / "fixed_points"

    df = pd.read_csv(eval_dir / "summary.csv")

    plot_hist_rho(df, eval_dir / "rho_hist.png")
    plot_stability_bar(df, eval_dir / "stability_bar.png")
    print(f"[plot_fixed_points] wrote figures to {eval_dir}")

if __name__ == "__main__":
    main()
