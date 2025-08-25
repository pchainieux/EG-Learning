#  python -m fixed_points.sweep --config configs/fixed-points/sweep.yaml

import os, itertools, subprocess, sys
from datetime import datetime
from pathlib import Path

def _parse_with_config(build_parser):
    import argparse, os
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=str, default=None)
    cfg_args, remaining = cfg_parser.parse_known_args()
    cfg = {}
    if cfg_args.config:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML is required for --config support. Install with pip install pyyaml.") from e
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
    ap.add_argument("--exp", default=cfg.get("exp", "ei_pI"))
    ap.add_argument("--optim", nargs="+", default=cfg.get("optim", ["eg", "gd"]))
    ap.add_argument("--pI", nargs="+", type=float, default=cfg.get("pI", [0.35]))
    ap.add_argument("--seeds", type=int, default=cfg.get("seeds", 5))
    ap.add_argument("--task", type=str, default=cfg.get("task", "dm1"))
    ap.add_argument("--hidden", type=int, default=cfg.get("hidden", 256))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 10))
    ap.add_argument("--steps", type=int, default=cfg.get("steps", 500))
    ap.add_argument("--jobs", type=int, default=cfg.get("jobs", 1))
    ap.add_argument("--lr_eg", type=float, default=cfg.get("lr_eg", 1.5))
    ap.add_argument("--lr_gd", type=float, default=cfg.get("lr_gd", 0.01))
    ap.add_argument("--device", type=str, default=cfg.get("device", "auto"))
    ap.add_argument("--outroot", type=str, default=cfg.get("outroot", "experiments/runs"))
    return ap


def main():
    args = _parse_with_config(build_parser)
    os.makedirs(args.outroot, exist_ok=True)
    base = os.path.join(args.outroot, args.exp); os.makedirs(base, exist_ok=True)

    jobs = []
    seed_list = list(range(args.seeds))

    repo_root = Path(__file__).resolve().parents[1]

    for optim, pI, seed in itertools.product(args.optim, args.pI, seed_list):
        run_dir = os.path.join(base, f"{optim}", f"pI={pI:.2f}", f"seed={seed:03d}")
        os.makedirs(run_dir, exist_ok=True)
        cmd = [
            sys.executable, "-m", "fixed_points.train",
            "--outdir", run_dir,
            "--optim", optim,
            "--pI", str(pI),
            "--seed", str(seed),
            "--task", args.task,
            "--hidden", str(args.hidden),
            "--epochs", str(args.epochs),
            "--steps", str(args.steps),
            "--lr_eg", str(args.lr_eg),
            "--lr_gd", str(args.lr_gd),
            "--device", args.device,
        ]
        jobs.append((cmd, repo_root))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run(cmd, cwd):
        start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{start}] START: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if proc.returncode != 0:
            print(f"[{end}] FAIL: {' '.join(cmd)}")
            print(proc.stdout); print(proc.stderr)
        else:
            print(f"[{end}] DONE: {' '.join(cmd)}")
        return proc.returncode

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [ex.submit(run, cmd, cwd) for cmd, cwd in jobs]
        for _ in as_completed(futures):
            pass

if __name__ == "__main__":
    main()
