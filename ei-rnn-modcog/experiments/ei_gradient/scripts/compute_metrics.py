from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np

from experiments.ei_gradient.src.io_utils import load_yaml, ensure_outdir, save_csv, save_json, load_tensors
from experiments.ei_gradient.src.metrics import (
    timecourse_l2, timecourse_l2_per_unit, timecourse_mean_abs,
    cumulative_backprop, fisher_like, cosine_alignment,
    gate_adjusted_timecourse, summarize_ei_distributions
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    if "out_dir" not in cfg:
        raise KeyError("Your config must define 'out_dir'.")
    base_out = Path(cfg["out_dir"])
    out_dir = ensure_outdir(base_out, "metrics")

    seeds = cfg.get("seeds", None)
    if seeds is None:
        seeds = [int(cfg.get("seed", 12345))]
    else:
        seeds = [int(s) for s in seeds]

    def compute_from_pt(pt_path: Path):
        T = load_tensors(pt_path) 
        h_seq   = T["h_seq"]
        g_h     = T["grad_h"]
        idx_E   = T["idx_E"].bool()
        idx_I   = T["idx_I"].bool()
        decmask = T.get("dec_mask", None)
        g_u     = T.get("grad_u", None)
        phi_p   = T.get("phi_prime", None)

        tc = {}
        tc.update(timecourse_mean_abs(g_h, idx_E, idx_I))
        tc.update(timecourse_l2(g_h, idx_E, idx_I))
        tc.update(timecourse_l2_per_unit(g_h, idx_E, idx_I))
        tc.update(cosine_alignment(h_seq, g_h, idx_E, idx_I))
        tc.update({k.replace("tc_abs", "tc_abs_pre"): v
                   for k, v in gate_adjusted_timecourse(g_u, g_h, phi_p, idx_E, idx_I).items()})

        Wbp_sum, Wbp_mean = cumulative_backprop(g_h)
        Fisher = fisher_like(g_h)
        units_df = pd.DataFrame({
            "Wbp_sum": Wbp_sum.cpu().numpy(),
            "Wbp_mean": Wbp_mean.cpu().numpy(),
            "Fisher": Fisher.cpu().numpy(),
            "type": np.where(idx_E.cpu().numpy(), "E", "I"),
        })
        summaries = {}
        if decmask is not None:
            mask_dec = decmask.bool(); mask_fix = ~mask_dec
            def masked_mean_over_tb(x, m):
                w = m.float().unsqueeze(-1)
                num = (x * w).sum(dim=(0,1)); den = w.sum(dim=(0,1)).clamp_min(1e-12)
                return num / den
            mean_g_fix = masked_mean_over_tb(g_h.abs(), mask_fix)
            mean_g_dec = masked_mean_over_tb(g_h.abs(), mask_dec)
            summaries.update({
                "mean_abs_grad_fix_E": float(mean_g_fix[idx_E].mean().item()),
                "mean_abs_grad_fix_I": float(mean_g_fix[idx_I].mean().item()),
                "mean_abs_grad_dec_E": float(mean_g_dec[idx_E].mean().item()),
                "mean_abs_grad_dec_I": float(mean_g_dec[idx_I].mean().item()),
            })
        summaries.update(summarize_ei_distributions(Wbp_sum, Wbp_mean, Fisher, idx_E, idx_I))
        return tc, units_df, summaries

    if len(seeds) == 1:
        default_grads = base_out / "grads" / "grads.pt"
        grads_path = Path(cfg.get("grads_path", default_grads))
        if not grads_path.exists():
            raise FileNotFoundError(f"Could not find grads tensors at '{grads_path}'.")
        tc, units_df, summaries = compute_from_pt(grads_path)
        df_tc = pd.DataFrame({k: v.cpu().numpy() for k, v in tc.items()})
        save_csv(df_tc, out_dir / "metrics_timecourses.csv")
        save_csv(units_df, out_dir / "metrics_units.csv")
        save_json(summaries, out_dir / "metrics_summary.json")
    else:
        tc_keys = None
        tc_stack = {}
        units_all = []
        summaries_all = []
        for s in seeds:
            pt_path = base_out / "grads" / f"grads_seed{s:04d}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"Missing {pt_path}. Did you run collect for seed {s}?")
            tc, units_df, summaries = compute_from_pt(pt_path)
            if tc_keys is None: tc_keys = list(tc.keys())
            for k in tc_keys:
                tc_stack.setdefault(k, []).append(tc[k].cpu().numpy())
            units_df = units_df.copy()
            units_df["seed"] = s
            units_all.append(units_df)
            summaries_all.append({"seed": s, **summaries})

        meanstd = {}
        for k, arrs in tc_stack.items():
            A = np.stack(arrs, axis=0) 
            meanstd[f"{k}_mean"] = A.mean(axis=0)
            meanstd[f"{k}_std"]  = A.std(axis=0, ddof=1 if A.shape[0] > 1 else 0)
        df_ms = pd.DataFrame(meanstd)
        save_csv(df_ms, out_dir / "metrics_timecourses_meanstd.csv")

        df_units_all = pd.concat(units_all, axis=0, ignore_index=True)
        save_csv(df_units_all, out_dir / "metrics_units.csv")

        save_csv(pd.DataFrame(summaries_all), out_dir / "metrics_summary_by_seed.csv")

    print(f"[compute_metrics] Wrote metrics to {out_dir}")


if __name__ == "__main__":
    main()
