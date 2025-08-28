from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import sys

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (10, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.2,
    "grid.linewidth": 0.6,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12, 
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "axes.prop_cycle": mpl.cycler(color=[
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]),
})

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_tensors(path: str | Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unexpected tensor file format at {path}")

def _finalize_axes(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.tick_params(direction="out", length=4, width=0.8)

def plot_timecourses(data: dict[str, np.ndarray], ylabel: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, y in data.items():
        x = np.arange(len(y))
        ax.plot(x, y, label=label)
    _finalize_axes(ax, xlabel="Time", ylabel=ylabel)
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_distributions(series: dict[str, np.ndarray], xlabel: str, outpath: Path, bins=40):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, x in series.items():
        ax.hist(x, bins=bins, density=True, alpha=0.55, label=label)
    _finalize_axes(ax, xlabel=xlabel, ylabel="Density")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_saliency_single(
    sal_map: np.ndarray,
    outpath: Path,
    cbar_label: str = "abs(dLoss/dx)"
):
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    im = ax.imshow(
        sal_map,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
        rasterized=True,
    )
    _finalize_axes(ax, xlabel="Time", ylabel="Channel")

    cbar = fig.colorbar(
        im, ax=ax, location="right", orientation="vertical",
        fraction=0.046, pad=0.04, shrink=0.9
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(cbar_label, rotation=270, labelpad=10)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_two(
    left: np.ndarray,
    right: np.ndarray,
    outpath: Path,
    cbar_label: str = "abs(dLoss/dx)"
):
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharex=True, sharey=True,
        constrained_layout=True
    )

    vmin = min(np.min(left), np.min(right))
    vmax = max(np.max(left), np.max(right))

    ims = []
    for ax, arr in zip(axes, (left, right)):
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
            rasterized=True,
            vmin=vmin, vmax=vmax
        )
        _finalize_axes(ax, xlabel="Time", ylabel="Channel")
        ims.append(im)

    cbar = fig.colorbar(
        ims[1], ax=axes, location="right", orientation="vertical",
        fraction=0.046, pad=0.04, shrink=0.9
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(cbar_label, rotation=270, labelpad=10)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML config used for the run (same as for compute_metrics)")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    base_out = Path(cfg["out_dir"])
    grads_dir = base_out / "grads"
    metrics_dir = base_out / "metrics"
    figs_dir = ensure_dir(base_out / "figs")

    tc_csv = metrics_dir / "metrics_timecourses.csv"
    units_csv = metrics_dir / "metrics_units.csv"
    if not tc_csv.exists():
        print(f"[plot_all] Missing {tc_csv}. Did you run compute_metrics.py?", file=sys.stderr)
        sys.exit(1)
    if not units_csv.exists():
        print(f"[plot_all] Missing {units_csv}. Did you run compute_metrics.py?", file=sys.stderr)
        sys.exit(1)

    df_tc = pd.read_csv(tc_csv)
    df_units = pd.read_csv(units_csv)

    need_cols = ["tc_abs_E", "tc_abs_I"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"mean abs grad (E)": df_tc["tc_abs_E"].to_numpy(),
             "mean abs grad (I)": df_tc["tc_abs_I"].to_numpy()},
            ylabel="Mean absolute gradient",
            outpath=figs_dir / "tc_meanabs.png",
        )
    else:
        print("[plot_all] Skipping mean |g| timecourse (columns not found).", file=sys.stderr)

    need_cols = ["tc_l2_E", "tc_l2_I"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"L2 norm (E)": df_tc["tc_l2_E"].to_numpy(),
             "L2 norm (I)": df_tc["tc_l2_I"].to_numpy()},
            ylabel="Mean L2 norm",
            outpath=figs_dir / "tc_l2.png",
        )
    else:
        print("[plot_all] Skipping raw L2 timecourse (columns not found).", file=sys.stderr)

    need_cols = ["tc_l2_E_norm", "tc_l2_I_norm"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"L2 per unit (E)": df_tc["tc_l2_E_norm"].to_numpy(),
             "L2 per unit (I)": df_tc["tc_l2_I_norm"].to_numpy()},
            ylabel="Mean L2 per unit",
            outpath=figs_dir / "tc_l2_norm.png",
        )
    else:
        print("[plot_all] Skipping normalized L2 timecourse (columns not found).", file=sys.stderr)

    need_cols = ["tc_cos_E", "tc_cos_I"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"cosine(h, grad) E": df_tc["tc_cos_E"].to_numpy(),
             "cosine(h, grad) I": df_tc["tc_cos_I"].to_numpy()},
            ylabel="Cosine similarity",
            outpath=figs_dir / "tc_alignment.png",
        )
    else:
        print("[plot_all] Skipping alignment timecourse (columns not found).", file=sys.stderr)

    if set(["Wbp_sum", "Wbp_mean", "Fisher", "type"]).issubset(df_units.columns):
        e_mask = df_units["type"] == "E"
        i_mask = df_units["type"] == "I"

        plot_distributions(
            {"Wbp_sum E": df_units.loc[e_mask, "Wbp_sum"].to_numpy(),
             "Wbp_sum I": df_units.loc[i_mask, "Wbp_sum"].to_numpy()},
            xlabel="Sum abs(gradient) over time and batch",
            outpath=figs_dir / "dist_wbp_sum.png",
        )
        plot_distributions(
            {"Wbp_mean E": df_units.loc[e_mask, "Wbp_mean"].to_numpy(),
             "Wbp_mean I": df_units.loc[i_mask, "Wbp_mean"].to_numpy()},
            xlabel="Mean abs(gradient) over time and batch",
            outpath=figs_dir / "dist_wbp_mean.png",
        )
        plot_distributions(
            {"Fisher E": df_units.loc[e_mask, "Fisher"].to_numpy(),
             "Fisher I": df_units.loc[i_mask, "Fisher"].to_numpy()},
            xlabel="Expected squared gradient",
            outpath=figs_dir / "dist_fisher.png",
        )
    else:
        print("[plot_all] Skipping distributions (metrics_units.csv missing expected columns).", file=sys.stderr)

    grads_pt = grads_dir / "grads.pt"
    if grads_pt.exists():
        T = load_tensors(grads_pt)
        if "grad_x" in T and "x_seq" in T:
            gx = T["grad_x"].abs().mean(dim=1)
            sal = gx.numpy().T

            if "dec_mask" in T:
                dm = T["dec_mask"].bool()
                w_dec = dm.float().mean(dim=1, keepdim=True)
                w_fix = (~dm).float().mean(dim=1, keepdim=True)
                gx_dec = (T["grad_x"].abs() * dm.unsqueeze(-1)).sum(dim=1) / (w_dec + 1e-12)
                gx_fix = (T["grad_x"].abs() * (~dm).unsqueeze(-1)).sum(dim=1) / (w_fix + 1e-12)
                plot_saliency_two(
                    left=gx_fix.numpy().T,
                    right=gx_dec.numpy().T,
                    outpath=figs_dir / "saliency.png",
                )
            else:
                plot_saliency_single(
                    sal_map=sal,
                    outpath=figs_dir / "saliency.png",
                )
        else:
            print("[plot_all] grads.pt lacks grad_x/x_seq → skipping saliency", file=sys.stderr)
    else:
        print(f"[plot_all] Missing {grads_pt} → skipping saliency", file=sys.stderr)

    print(f"[plot_all] Wrote figures to {figs_dir}")

if __name__ == "__main__":
    main()
