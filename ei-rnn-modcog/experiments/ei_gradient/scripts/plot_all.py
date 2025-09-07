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
    "axes.labelsize": 20,
    "axes.titlesize": 20, 
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "axes.prop_cycle": mpl.cycler(color=[
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]),
})

YLIMS = {
    "cosine":      (-1.0, 1.0)
}

def plot_tc_band(df_ms: pd.DataFrame, keyE: str, keyI: str, ylabel: str, outpath: Path):
    mE = df_ms[f"{keyE}_mean"].to_numpy(); sE = df_ms[f"{keyE}_std"].to_numpy()
    mI = df_ms[f"{keyI}_mean"].to_numpy(); sI = df_ms[f"{keyI}_std"].to_numpy()
    x = np.arange(len(mE))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, mE, color="tab:red", label="E")
    ax.fill_between(x, mE - sE, mE + sE, color="tab:red", alpha=0.20, linewidth=0)
    ax.plot(x, mI, color="tab:blue", label="I")
    ax.fill_between(x, mI - sI, mI + sI, color="tab:blue", alpha=0.20, linewidth=0)

    _finalize_axes(ax, xlabel="Time", ylabel=ylabel)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_timecourses_ei(
    data: dict[str, np.ndarray],
    ylabel: str,
    outpath: Path,
    trim_edges: int = 0,
    ylim: tuple[float, float] | None = None,
):
    def _ei_color_from_label(label: str) -> str | None:
        s = label.strip().lower()
        if "(e" in s or s.endswith(" e") or "excitatory" in s:
            return "red"
        if "(i" in s or s.endswith(" i") or "inhibitory" in s:
            return "blue"
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, y in data.items():
        y = np.asarray(y)
        if trim_edges > 0:
            if y.shape[0] <= 2 * trim_edges:
                continue
            y = y[trim_edges:-trim_edges]
        x = np.arange(len(y))
        ax.plot(x, y, label=label, color=_ei_color_from_label(label))

    if ylim is not None:
        ax.set_ylim(*ylim)

    _finalize_axes(ax, xlabel="Time", ylabel=ylabel)
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


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
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.tick_params(direction="out", length=4, width=0.8)

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
    cbar_label: str = "abs(dLoss/dx)",
    left_title: str = "Fixation",
    right_title: str = "Decision",
):
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), sharex=True, sharey=True, constrained_layout=True
    )

    vmin = min(np.min(left), np.min(right))
    vmax = max(np.max(left), np.max(right))

    ims = []
    for ax, arr, title in zip(axes, (left, right), (left_title, right_title)):
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
            rasterized=True,
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title, pad=8, fontweight="bold")
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
    parser.add_argument("--cfg", required=True,
                        help="YAML config used for the run (same as for compute_metrics)")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    base_out   = Path(cfg["out_dir"])
    grads_dir  = base_out / "grads"
    metrics_dir= base_out / "metrics"
    figs_dir   = ensure_dir(base_out / "figs")

    tc_csv_single = metrics_dir / "metrics_timecourses.csv"
    tc_csv_ms     = metrics_dir / "metrics_timecourses_meanstd.csv" 
    units_csv     = metrics_dir / "metrics_units.csv"

    use_ms = tc_csv_ms.exists()
    if not (tc_csv_single.exists() or tc_csv_ms.exists()):
        print(f"[plot_all] Missing timecourse metrics. Did you run compute_metrics.py?", file=sys.stderr)
        sys.exit(1)
    if not units_csv.exists():
        print(f"[plot_all] Missing {units_csv}. Did you run compute_metrics.py?", file=sys.stderr)
        sys.exit(1)

    df_tc = pd.read_csv(tc_csv_single) if tc_csv_single.exists() else None
    df_ms = pd.read_csv(tc_csv_ms)     if use_ms else None
    df_units = pd.read_csv(units_csv)

    def _bandplot_from_ms(df, keyE_base: str, keyI_base: str, ylabel: str, outpath: Path):
        mE = df[f"{keyE_base}_mean"].to_numpy()
        sE = df[f"{keyE_base}_std"].to_numpy()
        mI = df[f"{keyI_base}_mean"].to_numpy()
        sI = df[f"{keyI_base}_std"].to_numpy()
        x = np.arange(len(mE))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, mE, color="tab:red", label="E")
        ax.fill_between(x, mE - sE, mE + sE, color="tab:red", alpha=0.20, linewidth=0)
        ax.plot(x, mI, color="tab:blue", label="I")
        ax.fill_between(x, mI - sI, mI + sI, color="tab:blue", alpha=0.20, linewidth=0)

        ax.set_xlabel("Time", fontsize=20); ax.set_ylabel(ylabel, fontsize=20)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    if use_ms:
        need = {"tc_abs_E_mean","tc_abs_E_std","tc_abs_I_mean","tc_abs_I_std"}
        if need.issubset(df_ms.columns):
            _bandplot_from_ms(df_ms, "tc_abs_E", "tc_abs_I",
                              ylabel="Mean absolute gradient",
                              outpath=figs_dir / "tc_meanabs.png")
        else:
            print("[plot_all] Skipping mean |g| band plot (columns not found).", file=sys.stderr)

        need = {"tc_l2_E_mean","tc_l2_E_std","tc_l2_I_mean","tc_l2_I_std"}
        if need.issubset(df_ms.columns):
            _bandplot_from_ms(df_ms, "tc_l2_E", "tc_l2_I",
                              ylabel="Mean L2 norm",
                              outpath=figs_dir / "tc_l2.png")
        else:
            print("[plot_all] Skipping L2 band plot (columns not found).", file=sys.stderr)

        need = {"tc_l2_E_norm_mean","tc_l2_E_norm_std","tc_l2_I_norm_mean","tc_l2_I_norm_std"}
        if need.issubset(df_ms.columns):
            _bandplot_from_ms(df_ms, "tc_l2_E_norm", "tc_l2_I_norm",
                              ylabel="Mean L2 per unit",
                              outpath=figs_dir / "tc_l2_norm.png")
        else:
            print("[plot_all] Skipping normalized L2 band plot (columns not found).", file=sys.stderr)

        need = {"tc_cos_E_mean","tc_cos_E_std","tc_cos_I_mean","tc_cos_I_std"}
        if need.issubset(df_ms.columns):
            _bandplot_from_ms(df_ms, "tc_cos_E", "tc_cos_I",
                              ylabel="Cosine similarity",
                              outpath=figs_dir / "tc_alignment.png")
        else:
            print("[plot_all] Skipping alignment band plot (columns not found).", file=sys.stderr)

    else:
        need = ["tc_abs_E", "tc_abs_I"]
        if all(c in df_tc.columns for c in need):
            plot_timecourses(
                {"mean abs grad (E)": df_tc["tc_abs_E"].to_numpy(),
                 "mean abs grad (I)": df_tc["tc_abs_I"].to_numpy()},
                ylabel="Mean absolute gradient",
                outpath=figs_dir / "tc_meanabs.png",
            )
        else:
            print("[plot_all] Skipping mean |g| timecourse (columns not found).", file=sys.stderr)

        need = ["tc_l2_E", "tc_l2_I"]
        if all(c in df_tc.columns for c in need):
            plot_timecourses(
                {"L2 norm (E)": df_tc["tc_l2_E"].to_numpy(),
                 "L2 norm (I)": df_tc["tc_l2_I"].to_numpy()},
                ylabel="Mean L2 norm",
                outpath=figs_dir / "tc_l2.png",
            )
        else:
            print("[plot_all] Skipping raw L2 timecourse (columns not found).", file=sys.stderr)

        need = ["tc_l2_E_norm", "tc_l2_I_norm"]
        if all(c in df_tc.columns for c in need):
            plot_timecourses(
                {"L2 per unit (E)": df_tc["tc_l2_E_norm"].to_numpy(),
                 "L2 per unit (I)": df_tc["tc_l2_I_norm"].to_numpy()},
                ylabel="Mean L2 per unit",
                outpath=figs_dir / "tc_l2_norm.png",
            )
        else:
            print("[plot_all] Skipping normalized L2 timecourse (columns not found).", file=sys.stderr)

        need = ["tc_cos_E", "tc_cos_I"]
        if all(c in df_tc.columns for c in need):
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

    grads_pt = list(grads_dir.glob("grads_seed000*.pt"))
    if grads_pt[0].exists():
        T = load_tensors(grads_pt[0])
        if "grad_x" in T and "x_seq" in T:
            G = T["grad_x"].abs()  
            if G.ndim != 3:
                raise ValueError(f"[plot_all] Expected grad_x to be 3D (T,B,C). Got: {tuple(G.shape)}")
            sal = G.mean(dim=1).transpose(0, 1).contiguous().cpu().numpy()
            if "dec_mask" in T:
                dm = T["dec_mask"].bool()
                if dm.shape[:2] != G.shape[:2]:
                    raise ValueError(f"[plot_all] dec_mask shape {tuple(dm.shape)} incompatible with grad_x {tuple(G.shape)}")
                w_dec = dm.sum(dim=1, keepdim=True).clamp_min(1)
                w_fix = (~dm).sum(dim=1, keepdim=True).clamp_min(1)
                gx_dec_T_C = (G * dm.unsqueeze(-1)).sum(dim=1) / w_dec
                gx_fix_T_C = (G * (~dm).unsqueeze(-1)).sum(dim=1) / w_fix
                left  = gx_fix_T_C.transpose(0, 1).contiguous().cpu().numpy()
                right = gx_dec_T_C.transpose(0, 1).contiguous().cpu().numpy()
                plot_saliency_two(left=left, right=right, outpath=figs_dir / "saliency.png")
            else:
                plot_saliency_single(sal_map=sal, outpath=figs_dir / "saliency.png")
        else:
            print("[plot_all] grads.pt lacks grad_x/x_seq → skipping saliency", file=sys.stderr)
    else:
        print(f"[plot_all] Missing {grads_pt} → skipping saliency", file=sys.stderr)

    print(f"[plot_all] Wrote figures to {figs_dir}")

if __name__ == "__main__":
    main()
