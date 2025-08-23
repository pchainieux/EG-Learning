# python -m fixed_points.plot_reports --config configs/fixed-points/plot.yaml

import os, pandas as pd, matplotlib.pyplot as plt

def _parse_with_config(build_parser):
    import argparse, sys, os
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=str, default=None)
    cfg_args, remaining = cfg_parser.parse_known_args()
    cfg = {}
    if cfg_args.config:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML is required for --config support. Install with `pip install pyyaml`.") from e
        if not os.path.isfile(cfg_args.config):
            raise FileNotFoundError(f"Config file not found: {cfg_args.config}")
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    parser = build_parser(cfg)
    args = parser.parse_args(remaining)
    return args

def build_parser(cfg):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--summary", required=not bool(cfg), default=cfg.get("summary", None))
    ap.add_argument("--outdir", required=not bool(cfg), default=cfg.get("outdir", None))
    return ap

def line_with_ci(ax, df, x, y, hue, ci="sem", title=None, ylabel=None):
    for key, grp in df.groupby(hue):
        stats = grp.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
        stats["sem"] = stats["std"] / stats["count"].clip(lower=1).pow(0.5)
        ax.plot(stats[x], stats["mean"], label=str(key), marker="o")
        err = stats["sem"] if ci == "sem" else stats["std"]
        ax.fill_between(stats[x], stats["mean"]-err, stats["mean"]+err, alpha=0.2)
    ax.set_title(title or y); ax.set_xlabel(x); ax.set_ylabel(ylabel or y); ax.legend()

def main():
    args = _parse_with_config(build_parser)
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.summary)

    fig1 = plt.figure(); ax1 = fig1.add_subplot(111)
    line_with_ci(ax1, df, x="pI", y="stable_mean_memory", hue="optimizer",
                 title="Stability margin (memory) vs pI", ylabel="1 - rho (mean)")
    fig1.savefig(os.path.join(args.outdir, "stability_vs_pI_memory.png"), dpi=160); plt.close(fig1)

    fig2 = plt.figure(); ax2 = fig2.add_subplot(111)
    line_with_ci(ax2, df, x="pI", y="stable_mean_anti", hue="optimizer",
                 title="Stability margin (anti) vs pI", ylabel="1 - rho (mean)")
    fig2.savefig(os.path.join(args.outdir, "stability_vs_pI_anti.png"), dpi=160); plt.close(fig2)

    fig3 = plt.figure(); ax3 = fig3.add_subplot(111)
    line_with_ci(ax3, df, x="pI", y="reuse_overlap", hue="optimizer",
                 title="Motif reuse overlap vs pI", ylabel="mean cos(principal angles)")
    fig3.savefig(os.path.join(args.outdir, "reuse_vs_pI.png"), dpi=160); plt.close(fig3)

    if "acc_alpha1" in df.columns:
        fig4 = plt.figure(); ax4 = fig4.add_subplot(111)
        line_with_ci(ax4, df, x="pI", y="acc_alpha1", hue="optimizer",
                     title="Validation accuracy (alphaI=1) vs pI", ylabel="acc")
        fig4.savefig(os.path.join(args.outdir, "acc_vs_pI.png"), dpi=160); plt.close(fig4)

    print(f"Saved figures in {args.outdir}")

if __name__ == "__main__":
    main()
