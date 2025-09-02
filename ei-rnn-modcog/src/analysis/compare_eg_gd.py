from __future__ import annotations
import argparse
from pathlib import Path

from src.analysis.viz_compare import (
    plot_accuracy_steps, plot_loss_steps, plot_val_accuracy_epochs
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eg", required=True, help="EG run directory (contains metrics_*.npz)")
    ap.add_argument("--gd", required=True, help="GD run directory (contains metrics_*.npz)")
    ap.add_argument("--outdir", default=None, help="Where to write the comparison figures")
    ap.add_argument("--ema-beta", type=float, default=0.98, help="EMA smoothing for step curves")
    ap.add_argument("--epoch-window", type=int, default=5, help="Rolling window for epoch curves")
    ap.add_argument("--target_acc", type=float, default=None, help="Optional horizontal target accuracy line")
    ap.add_argument("--target-loss", type=float, default=None, help="Optional horizontal target loss line")
    ap.add_argument("--steps-file", default="metrics_steps.npz")
    ap.add_argument("--epoch-file", default="metrics_epoch.npz")
    args = ap.parse_args()

    eg_dir = Path(args.eg); gd_dir = Path(args.gd)
    outdir = Path(args.outdir) if args.outdir else eg_dir.parent / "compare_eg_gd"
    outdir.mkdir(parents=True, exist_ok=True)

    eg_steps = eg_dir / args.steps_file
    gd_steps = gd_dir / args.steps_file
    eg_epoch = eg_dir / args.epoch_file
    gd_epoch = gd_dir / args.epoch_file

    plot_accuracy_steps(str(eg_steps), str(gd_steps),
                        str(outdir / "compare_acc_steps.png"),
                        ema_beta=args.ema_beta, target=args.target_acc)

    plot_loss_steps(str(eg_steps), str(gd_steps),
                    str(outdir / "compare_loss_steps.png"),
                    ema_beta=args.ema_beta, target=args.target_loss)

    plot_val_accuracy_epochs(str(eg_epoch), str(gd_epoch),
                             str(outdir / "compare_val_acc_epoch.png"),
                             window=args.epoch_window, target=args.target_acc if hasattr(args, "target_acc") else None)

if __name__ == "__main__":
    main()
