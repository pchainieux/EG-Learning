from __future__ import annotations
import argparse
from pathlib import Path

from src.analysis.viz_weights import (
    plots_from_ckpt,
    plot_whh_distribution, plot_row_col_sums, plot_spectrum,
    plot_whh_heatmap, plot_whh_heatmap_observed,
    _load_whh_from_ckpt,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    ap.add_argument("--outdir", default=None, help="Output directory (default: next to ckpt)")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=["dist", "sums", "spec", "heat", "heat_obs", "heat_observed", "heat2", "heat_2"],
        help="Subset of plots to generate",
    )
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    outdir = Path(args.outdir) if args.outdir else ckpt.parent / (ckpt.stem + "_figs")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.only is None:
        plots_from_ckpt(str(ckpt), str(outdir))
        return

    from src.analysis.viz_weights import _load_whh_from_ckpt
    W, s = _load_whh_from_ckpt(str(ckpt))

    if args.only is None:
        plots_from_ckpt(str(ckpt), str(outdir))
        return

    if "dist" in args.only:
        plot_whh_distribution(W, s, str(outdir / "whh_distribution.png"))

    if "sums" in args.only:
        plot_row_col_sums(W, s, str(outdir / "whh_row_col_sums.png"))

    if "spec" in args.only:
        plot_spectrum(W, str(outdir / "whh_spectrum.png"))

    if any(x in args.only for x in ("heat_obs", "heat_observed", "heat2", "heat_2")):
        plot_whh_heatmap_observed(
            W, s,
            str(outdir / "whh_heatmap_observed.png"),
            title="$W_{hh}$ heatmap (observed order)",
            strip_size="6%", 
            strip_pad=0.10, 
        )

    if "heat" in args.only:
        plot_whh_heatmap(W, s, str(outdir / "whh_heatmap.png"))


if __name__ == "__main__":
    main()
