from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np

from ei_gradient.src.io_utils import load_yaml, ensure_outdir, save_csv, save_json, load_tensors
from ei_gradient.src.metrics import (
    timecourse_l2, timecourse_mean_abs, cumulative_backprop, fisher_like,
    cosine_alignment, gate_adjusted_timecourse, summarize_ei_distributions
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    exp_id = cfg.get("exp_id", "exp_ei_credit")
    out_dir = ensure_outdir(cfg.get("out_dir", "outputs"), exp_id)

    T = load_tensors(out_dir / "grads.pt")
    h_seq: torch.Tensor = T["h_seq"]
    grad_h: torch.Tensor = T["grad_h"]
    idx_E: torch.Tensor = T["idx_E"].bool()
    idx_I: torch.Tensor = T["idx_I"].bool()
    u_seq: torch.Tensor | None = T.get("u_seq", None)

    tc_l2 = timecourse_l2(grad_h, idx_E, idx_I)
    tc_ma = timecourse_mean_abs(grad_h, idx_E, idx_I)
    cos = cosine_alignment(h_seq, grad_h, idx_E, idx_I)

    gated = {}
    if u_seq is not None:
        gated = gate_adjusted_timecourse(grad_h, u_seq, idx_E, idx_I, activation=cfg.get("activation", "softplus"))

    Wbp = cumulative_backprop(grad_h)
    Fisher = fisher_like(grad_h)
    summaries = summarize_ei_distributions(Wbp, Fisher, idx_E, idx_I)

    t_axis = np.arange(h_seq.shape[0])
    rows = []
    for t in range(len(t_axis)):
        row = {
            "t": int(t_axis[t]),
            "tc_l2_E": float(tc_l2["tc_l2_E"][t].item()),
            "tc_l2_I": float(tc_l2["tc_l2_I"][t].item()),
            "tc_ma_E": float(tc_ma["tc_ma_E"][t].item()),
            "tc_ma_I": float(tc_ma["tc_ma_I"][t].item()),
            "cos_E": float(cos["cos_E"][t].item()),
            "cos_I": float(cos["cos_I"][t].item()),
        }
        if gated:
            row["tc_l2_E_gated"] = float(gated["tc_l2_E_gated"][t].item())
            row["tc_l2_I_gated"] = float(gated["tc_l2_I_gated"][t].item())
        rows.append(row)
    df_tc = pd.DataFrame(rows)
    save_csv(df_tc, out_dir / "metrics_timecourse.csv")

    df_units = pd.DataFrame({
        "Wbp": Wbp.cpu().numpy(),
        "Fisher": Fisher.cpu().numpy(),
        "type": np.where(idx_E.cpu().numpy(), "E", "I"),
    })
    save_csv(df_units, out_dir / "metrics_units.csv")

    save_json(summaries, out_dir / "metrics_summary.json")

    print(f"[compute_metrics] Wrote:\n - {out_dir/'metrics_timecourse.csv'}\n - {out_dir/'metrics_units.csv'}\n - {out_dir/'metrics_summary.json'}")

if __name__ == "__main__":
    main()
