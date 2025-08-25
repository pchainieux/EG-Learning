from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from ei_gradient.src.io import load_yaml, ensure_outdir, load_tensors
from ei_gradient.src.vis import plot_timecourses, plot_distributions, plot_saliency_grid, plot_before_after_inputs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    exp_id = cfg.get("exp_id", "exp_ei_credit")
    out_dir = ensure_outdir(cfg.get("out_dir", "outputs"), exp_id)
    figs_dir = out_dir / "figs"

    # Load precomputed metrics (CSV) and tensors (optional extras for saliency)
    df_tc = pd.read_csv(out_dir / "metrics_timecourse.csv")
    df_units = pd.read_csv(out_dir / "metrics_units.csv")
    T = load_tensors(out_dir / "grads.pt")

    # ---- Timecourses ----
    t = df_tc["t"].to_numpy()
    series_tc = {
        "||g|| (E)": df_tc["tc_l2_E"].to_numpy(),
        "||g|| (I)": df_tc["tc_l2_I"].to_numpy(),
    }
    plot_timecourses(t, series_tc, title="Gradient L2 Timecourses", ylabel="mean L2 over batch", outpath=figs_dir / "tc_l2.png")

    series_ma = {
        "mean |g| (E)": df_tc["tc_ma_E"].to_numpy(),
        "mean |g| (I)": df_tc["tc_ma_I"].to_numpy(),
    }
    plot_timecourses(t, series_ma, title="Mean |Gradient| Timecourses", ylabel="mean |g| over batch", outpath=figs_dir / "tc_meanabs.png")

    series_cos = {
        "cos(h,g) E": df_tc["cos_E"].to_numpy(),
        "cos(h,g) I": df_tc["cos_I"].to_numpy(),
    }
    plot_timecourses(t, series_cos, title="Alignment cos(h,g)", ylabel="cosine", outpath=figs_dir / "tc_alignment.png")

    if "tc_l2_E_gated" in df_tc.columns and "tc_l2_I_gated" in df_tc.columns:
        series_gate = {
            "gated ||g|| (E)": df_tc["tc_l2_E_gated"].to_numpy(),
            "gated ||g|| (I)": df_tc["tc_l2_I_gated"].to_numpy(),
        }
        plot_timecourses(t, series_gate, title="Gate-adjusted Gradient L2", ylabel="mean L2 (gated)", outpath=figs_dir / "tc_l2_gated.png")

    # ---- Distributions ----
    Wbp_E = df_units.loc[df_units["type"] == "E", "Wbp"].to_numpy()
    Wbp_I = df_units.loc[df_units["type"] == "I", "Wbp"].to_numpy()
    plot_distributions({"Wbp E": Wbp_E, "Wbp I": Wbp_I}, title="Cumulative Backprop (Wbp)", xlabel="sum_t,b |g_t|", outpath=figs_dir / "dist_wbp.png")

    Fisher_E = df_units.loc[df_units["type"] == "E", "Fisher"].to_numpy()
    Fisher_I = df_units.loc[df_units["type"] == "I", "Fisher"].to_numpy()
    plot_distributions({"Fisher E": Fisher_E, "Fisher I": Fisher_I}, title="Fisher-like Sensitivity", xlabel="E[(dℓ/dh)^2]", outpath=figs_dir / "dist_fisher.png")

    # ---- Saliency / Input opt panels (if present) ----
    # If grad_x was saved during collection: show simple saliency heatmap of one batch item
    if "grad_x" in T and "x_seq" in T:
        gx = T["grad_x"].abs().mean(dim=1)  # [T,D] average over batch for display
        x = T["x_seq"].mean(dim=1)          # [T,D]
        sal_maps = [gx.numpy().T]           # channels × time
        plot_saliency_grid(sal_maps, titles=["|∂ℓ/∂x| (avg over batch)"], outpath=figs_dir / "saliency.png")
        # Before-after example (first channel map)
        plot_before_after_inputs(x.numpy().T, (x.numpy() + 0.0).T, outpath=figs_dir / "before_after_placeholder.png")

    print(f"[plot_all] Figures written to {figs_dir}")

if __name__ == "__main__":
    main()
