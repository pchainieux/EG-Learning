# python -m fixed_points.aggregate_results --config configs/fixed-points/aggregate.yaml

import os, json, glob
import pandas as pd

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
    ap.add_argument("--root", required=not bool(cfg), default=cfg.get("root", None))
    ap.add_argument("--out", type=str, default=cfg.get("out", None))
    return ap

def main():
    args = _parse_with_config(build_parser)
    rows = []
    for metrics_path in glob.glob(os.path.join(args.root, "*", "pI=*", "seed=*", "eval", "metrics.json")):
        run_dir = os.path.dirname(os.path.dirname(metrics_path))
        parts = run_dir.split(os.sep)
        optimizer = parts[-3]; pI = float(parts[-2].split("=")[-1]); seed = int(parts[-1].split("=")[-1])
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        rob_path = os.path.join(run_dir, "eval", "robustness.json")
        robustness = None
        if os.path.isfile(rob_path):
            with open(rob_path, "r") as f:
                robustness = json.load(f)
        row = {"run_dir": run_dir, "optimizer": optimizer, "pI": pI, "seed": seed,
                "n_fp_memory": metrics["per_context"].get("memory", {}).get("n", 0),
                "n_fp_anti": metrics["per_context"].get("anti", {}).get("n", 0),
                "stable_mean_memory": metrics["per_context"].get("memory", {}).get("stability", {}).get("mean", float("nan")),
                "stable_mean_anti": metrics["per_context"].get("anti", {}).get("stability", {}).get("mean", float("nan")),
                "reuse_overlap": metrics.get("reuse", {}).get("pair", {}).get("overlap", float("nan"))}
        if robustness:
            for r in robustness:
                if abs(r.get("alphaI", 0.0) - 1.0) < 1e-6:
                    row["acc_alpha1"] = r.get("acc", float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(args.root, "summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
