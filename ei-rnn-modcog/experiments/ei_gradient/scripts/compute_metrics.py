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

    default_grads = base_out / "grads" / "grads.pt"
    grads_path = Path(cfg.get("grads_path", default_grads))
    if not grads_path.exists():
        raise FileNotFoundError(
            f"Could not find grads tensors at '{grads_path}'. "
            f"Either run collect_grads first, or set 'grads_path' in your YAML."
        )

    out_dir = ensure_outdir(base_out, "metrics")

    T = load_tensors(grads_path) 

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
    tc.update({k.replace("tc_abs", "tc_abs_pre"): v for k, v in gate_adjusted_timecourse(g_u, g_h, phi_p, idx_E, idx_I).items()})

    df_tc = pd.DataFrame({k: v.cpu().numpy() for k, v in tc.items()})
    save_csv(df_tc, out_dir / "metrics_timecourses.csv")

    summaries = {}
    if decmask is not None:
        mask_dec = decmask.bool()
        mask_fix = ~mask_dec

        def masked_mean_over_tb(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3 or m.dim() != 2:
                raise ValueError(f"masked_mean_over_tb expects x (T,B,N) and m (T,B); got {tuple(x.shape)}, {tuple(m.shape)}")
            w = m.float().unsqueeze(-1)
            num = (x * w).sum(dim=(0,1))
            den = w.sum(dim=(0,1)).clamp_min(1e-12)
            return num / den

        mean_g_fix = masked_mean_over_tb(g_h.abs(), mask_fix)
        mean_g_dec = masked_mean_over_tb(g_h.abs(), mask_dec)

        summaries["mean_abs_grad_fix_E"] = float(mean_g_fix[idx_E].mean().item())
        summaries["mean_abs_grad_fix_I"] = float(mean_g_fix[idx_I].mean().item())
        summaries["mean_abs_grad_dec_E"] = float(mean_g_dec[idx_E].mean().item())
        summaries["mean_abs_grad_dec_I"] = float(mean_g_dec[idx_I].mean().item())

    Wbp_sum, Wbp_mean = cumulative_backprop(g_h) 
    Fisher = fisher_like(g_h) 

    df_units = pd.DataFrame({
        "Wbp_sum": Wbp_sum.cpu().numpy(),
        "Wbp_mean": Wbp_mean.cpu().numpy(),
        "Fisher": Fisher.cpu().numpy(),
        "type": np.where(idx_E.cpu().numpy(), "E", "I"),
    })
    save_csv(df_units, out_dir / "metrics_units.csv")

    summaries.update(summarize_ei_distributions(Wbp_sum, Wbp_mean, Fisher, idx_E, idx_I))
    save_json(summaries, out_dir / "metrics_summary.json")

    print(f"[compute_metrics] Wrote:\n - {out_dir/'metrics_timecourses.csv'}\n - {out_dir/'metrics_units.csv'}\n - {out_dir/'metrics_summary.json'}")

if __name__ == "__main__":
    main()
