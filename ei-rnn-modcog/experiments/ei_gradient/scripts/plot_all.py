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

def plot_timecourses(data: dict[str, np.ndarray], title: str, ylabel: str, outpath: Path):
    plt.figure(figsize=(10, 4))
    for label, y in data.items():
        x = np.arange(len(y))
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel("time (t)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_distributions(series: dict[str, np.ndarray], title: str, xlabel: str, outpath: Path, bins=40):
    plt.figure(figsize=(10, 4))
    for label, x in series.items():
        plt.hist(x, bins=bins, density=True, alpha=0.6, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_saliency_single(sal_map: np.ndarray, title: str, outpath: Path):
    plt.figure(figsize=(10, 4))
    im = plt.imshow(sal_map, aspect="auto", origin="upper")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("channel")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_saliency_two(left: np.ndarray, right: np.ndarray, titles: tuple[str, str], outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    ims = []
    for ax, arr, ttl in zip(axes, (left, right), titles):
        im = ax.imshow(arr, aspect="auto", origin="upper")
        ax.set_title(ttl)
        ax.set_xlabel("time")
        ax.set_ylabel("channel")
        ims.append(im)
    vmin = min(im.get_clim()[0] for im in ims)
    vmax = max(im.get_clim()[1] for im in ims)
    for im in ims:
        im.set_clim(vmin, vmax)
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist())
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

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
            {"mean |g| (E)": df_tc["tc_abs_E"].to_numpy(),
             "mean |g| (I)": df_tc["tc_abs_I"].to_numpy()},
            title="Mean |Gradient| Timecourses",
            ylabel="mean |g| over batch",
            outpath=figs_dir / "tc_meanabs.png",
        )
    else:
        print("[plot_all] Skipping mean |g| timecourse (columns not found).", file=sys.stderr)

    need_cols = ["tc_l2_E", "tc_l2_I"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"||g|| (E)": df_tc["tc_l2_E"].to_numpy(),
             "||g|| (I)": df_tc["tc_l2_I"].to_numpy()},
            title="Gradient L2 Timecourses",
            ylabel="mean L2 over batch",
            outpath=figs_dir / "tc_l2.png",
        )
    else:
        print("[plot_all] Skipping raw L2 timecourse (columns not found).", file=sys.stderr)

    need_cols = ["tc_l2_E_norm", "tc_l2_I_norm"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"||g||/√n (E)": df_tc["tc_l2_E_norm"].to_numpy(),
             "||g||/√n (I)": df_tc["tc_l2_I_norm"].to_numpy()},
            title="Per-Unit Normalized L2 Timecourses",
            ylabel="mean L2 per unit",
            outpath=figs_dir / "tc_l2_norm.png",
        )
    else:
        print("[plot_all] Skipping normalized L2 timecourse (columns not found).", file=sys.stderr)

    need_cols = ["tc_cos_E", "tc_cos_I"]
    if all(c in df_tc.columns for c in need_cols):
        plot_timecourses(
            {"cos(h,g) E": df_tc["tc_cos_E"].to_numpy(),
             "cos(h,g) I": df_tc["tc_cos_I"].to_numpy()},
            title="Alignment cos(h,g)",
            ylabel="cosine",
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
            title="Cumulative Backprop (Wbp_sum)",
            xlabel="sum_{t,b} |g_t|",
            outpath=figs_dir / "dist_wbp_sum.png",
        )
        plot_distributions(
            {"Wbp_mean E": df_units.loc[e_mask, "Wbp_mean"].to_numpy(),
             "Wbp_mean I": df_units.loc[i_mask, "Wbp_mean"].to_numpy()},
            title="Mean Backprop (Wbp_mean)",
            xlabel="mean_{t,b} |g_t|",
            outpath=figs_dir / "dist_wbp_mean.png",
        )
        plot_distributions(
            {"Fisher E": df_units.loc[e_mask, "Fisher"].to_numpy(),
             "Fisher I": df_units.loc[i_mask, "Fisher"].to_numpy()},
            title="Fisher-like Sensitivity",
            xlabel="E[(dℓ/dh)^2]",
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
                    titles=("|∂ℓ/∂x| (fixation)", "|∂ℓ/∂x| (decision)"),
                    outpath=figs_dir / "saliency.png",
                )
            else:
                plot_saliency_single(
                    sal_map=sal,
                    title="|∂ℓ/∂x| (avg over batch)",
                    outpath=figs_dir / "saliency.png",
                )
        else:
            print("[plot_all] grads.pt lacks grad_x/x_seq → skipping saliency", file=sys.stderr)
    else:
        print(f"[plot_all] Missing {grads_pt} → skipping saliency", file=sys.stderr)

    print(f"[plot_all] Wrote figures to {figs_dir}")


if __name__ == "__main__":
    main()
