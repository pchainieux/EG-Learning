from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
import sys
import os

import numpy as np
import torch
import yaml
from neurogym import Dataset
import torch.nn.functional as F

from experiments.fixed_points.src.config import load_config, get_outdir
from experiments.fixed_points.src.model_io import rebuild_model_from_ckpt
from src.data import mod_cog_tasks as mct


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _collect_rollout(
    model: torch.nn.Module,
    env_fn,
    rollout_cfg: Mapping[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    batch = int(rollout_cfg.get("batch", 256))
    seq_len = int(rollout_cfg.get("seq_len", 60))
    ctx = rollout_cfg.get("context_channels", {})
    fix_ch = int(ctx.get("memory", 0))
    anti_ch = int(ctx.get("anti", -1))
    rule_mem_ch = int(ctx.get("rule_mem", -1))

    env = env_fn()
    ds = Dataset(env, batch_size=batch, seq_len=seq_len, batch_first=True)

    X, Y = ds()
    X = torch.from_numpy(X).to(device=device, dtype=torch.float32)

    if fix_ch >= 0 and X.shape[-1] > fix_ch:
        X[..., 0, fix_ch] = 1.0
    if anti_ch >= 0 and X.shape[-1] > anti_ch:
        X[..., 0, anti_ch] = 0.0
    if rule_mem_ch >= 0 and X.shape[-1] > rule_mem_ch:
        X[..., 0, rule_mem_ch] = 0.0

    with torch.no_grad():
        x_last = X[:, -1, :]
        pre = x_last @ model.W_xh.T + model.b_h
        H0 = F.softplus(pre)

    X_np = X.detach().cpu().numpy()
    H0_np = H0.detach().cpu().numpy()
    return X_np, H0_np


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config with optional fixed_points: block")
    ap.add_argument("--outdir", type=str, default=None, help="Override outdir for this run (parent of run/)")
    ap.add_argument("--plot", action="store_true", help="Produce figures after analysis")

    ap.add_argument("--inh", type=float, default=None, help="Inhibitory fraction; sets model.exc_frac = 1 - inh")
    ap.add_argument("--exc-frac", type=float, default=None, help="Direct set of model.exc_frac (overrides --inh)")
    ap.add_argument("--seed", type=int, default=None, help="Training seed override")
    ap.add_argument("--one-fp-seed", action="store_true",
                    help="Use a single fixed-point seed (fixed_points.rollout.batch=1)")
    ap.add_argument("--epochs", type=int, default=None, help="Override train.num_epochs")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="Override train.steps_per_epoch")
    return ap


def main():
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[3]
    TRAINER_PY = REPO_ROOT / "base_scripts" / "train_singlehead_modcog.py"

    args = build_argparser().parse_args()
    base_cfg = load_config(args.config)

    cfg: Dict[str, Any] = {k: v for k, v in base_cfg.items()}
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("fixed_points", {})
    cfg["fixed_points"].setdefault("rollout", {})

    if args.outdir:
        cfg["outdir"] = args.outdir

    if args.exc_frac is not None:
        cfg["model"]["exc_frac"] = float(args.exc_frac)
    elif args.inh is not None:
        cfg["model"]["exc_frac"] = 1.0 - float(args.inh)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    if args.one_fp_seed:
        cfg["fixed_points"]["rollout"]["batch"] = 1

    if args.epochs is not None:
        cfg["train"]["num_epochs"] = int(args.epochs)
    if args.steps_per_epoch is not None:
        cfg["train"]["steps_per_epoch"] = int(args.steps_per_epoch)

    outdir_cfg = cfg.get("outdir", "outputs/fp_runs/default")
    if not os.path.isabs(outdir_cfg):
        outdir_cfg = str((REPO_ROOT / outdir_cfg).resolve())
    cfg["outdir"] = outdir_cfg

    outdir = get_outdir(cfg)
    run_dir = Path(outdir) / "run"
    ckpt = run_dir / "ckpt.pt"

    print(f"[orchestrator] REPO_ROOT={REPO_ROOT}")
    print(f"[orchestrator] OUTDIR(abs)={outdir}")
    print(f"[orchestrator] RUN_DIR={run_dir}")
    print(f"[orchestrator] CKPT_EXPECTED={ckpt}")

    if not ckpt.exists():
        with tempfile.NamedTemporaryFile(prefix="fp_", suffix=".yaml",
                                         mode="w", encoding="utf-8", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            cfg_path_for_trainer = tmp.name

        print(f"[orchestrator] trainer={TRAINER_PY}")
        res = subprocess.run(
            [sys.executable, "-u", str(TRAINER_PY), "--config", cfg_path_for_trainer],
            cwd=str(REPO_ROOT),
        )
        if res.returncode != 0:
            raise RuntimeError(f"Trainer failed with exit code {res.returncode}")

    if not ckpt.exists():
        candidates = [
            run_dir / "checkpoint.pt",
            run_dir / "model.pt",
            run_dir / "best.pt",
            Path(outdir) / "ckpt.pt",
            Path(outdir) / "model.pt",
            Path(outdir) / "checkpoint.pt",
        ]
        candidates += sorted(Path(outdir).glob("singlehead_epoch*.pt"))
        for cand in candidates:
            if cand.exists():
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                if ckpt != cand:
                    import shutil
                    shutil.copy2(cand, ckpt)
                print(f"[orchestrator] normalized checkpoint → {ckpt}")
                break
        else:
            raise FileNotFoundError(
                "Checkpoint not found after training.\n"
                f"Expected: {ckpt}\n"
                "Verify the trainer saves to <outdir>/run/ckpt.pt or one of the common names."
            )

    device = _device(cfg.get("device", "auto"))
    fp_cfg = cfg.get("fixed_points", {})
    eval_subdir = fp_cfg.get("eval", {}).get("outdir", "eval/fixed_points")
    eval_dir = run_dir / eval_subdir
    seeds_file = eval_dir / "rollout_seeds.npz"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not seeds_file.exists():
        model, _saved = rebuild_model_from_ckpt(ckpt, device=device)
        task = cfg.get("tasks", ["dm1"])[0]
        env_fn = getattr(mct, task)
        X_np, H0_np = _collect_rollout(
            model=model,
            env_fn=env_fn,
            rollout_cfg=fp_cfg.get("rollout", {}),
            device=device,
        )
        np.savez_compressed(str(seeds_file), X=X_np, H0=H0_np)
        print(f"[orchestrator] wrote seeds → {seeds_file}")

    res = subprocess.run(
        [sys.executable, "-u", "-m",
         "experiments.fixed_points.scripts.unified_fixed_points",
         "--run", str(run_dir),
         "--config", args.config],
        cwd=str(REPO_ROOT),
    )
    if res.returncode != 0:
        raise RuntimeError(f"Analysis failed with exit code {res.returncode}")

    if args.plot:
        res = subprocess.run(
            [sys.executable, "-u", "-m",
             "experiments.fixed_points.scripts.plot_fixed_points",
             "--run", str(run_dir),
             "--config", args.config],
            cwd=str(REPO_ROOT),
        )
        if res.returncode != 0:
            raise RuntimeError(f"Plotting failed with exit code {res.returncode}")


if __name__ == "__main__":
    main()
