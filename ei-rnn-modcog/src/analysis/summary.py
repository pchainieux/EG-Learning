from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

def _load_npz_safe(path: Path) -> Optional[dict]:
    try:
        with np.load(str(path), allow_pickle=True) as f:
            return {k: f[k] for k in f.files}
    except Exception:
        return None

def _glob_runs(inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out.append(p)
        else:
            for q in sorted(Path().glob(s)):
                if q.is_dir():
                    out.append(q)
    seen = set()
    uniq: list[Path] = []
    for r in out:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.glob("singlehead_epoch*.pt"))
    if cands:
        return cands[-1]
    return cands[-1] if cands else None


def _load_w_hh_from_ckpt(ckpt_path: Path) -> Optional[torch.Tensor]:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        return None

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
        for k in ("W_hh", "core.W_hh"):
            if k in sd:
                return sd[k].detach().cpu()
        for k, v in sd.items():
            t = torch.as_tensor(v)
            if t.ndim == 2 and t.shape[0] == t.shape[1]:
                return t.detach().cpu()
        return None

    if "core" in ckpt and isinstance(ckpt["core"], dict):
        core_sd = ckpt["core"]
        if "W_hh" in core_sd:
            return core_sd["W_hh"].detach().cpu()
        for k, v in core_sd.items():
            t = torch.as_tensor(v)
            if t.ndim == 2 and t.shape[0] == t.shape[1]:
                return t.detach().cpu()

    return None


def _load_config_from_ckpt(ckpt_path: Path) -> dict:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    return {}



def _ema(x: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0.0:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    m = 0.0
    for i, v in enumerate(x.astype(float)):
        m = beta * m + (1.0 - beta) * v
        out[i] = m / (1.0 - beta ** (i + 1))
    return out


def _first_idx_ge(x: np.ndarray, thr: float) -> Optional[int]:
    idx = np.argmax(x >= thr)
    if x.size == 0:
        return None
    if x[idx] >= thr:
        return int(idx)
    return None


def _area_under_curve(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.mean(y))


def _spectral_gap(W: torch.Tensor) -> float:
    try:
        s = torch.linalg.svdvals(W).cpu().numpy()
        s = np.asarray(s, dtype=float)
        s.sort()
        s = s[::-1] 
        if s.size >= 2:
            return float(s[0] - s[1])
        if s.size == 1:
            return float(s[0])
        return float("nan")
    except Exception:
        return float("nan")


def _eff_nonzero_pct(W: torch.Tensor, tau_scale: float) -> float:
    A = W.detach().abs().cpu().numpy().astype(float)
    med = np.median(A)
    tau = max(1e-12, tau_scale * med)
    pct = 100.0 * float((A > tau).mean())
    return pct

@dataclass
class Row:
    run: str
    acc_10k: float
    steps_to_thr: float
    aulc_10k: float
    eff_nonzero_whh_pct: float
    spectral_gap: float
    algorithm: str
    hidden_size: Optional[int]
    num_tasks: Optional[int]
    seed: Optional[int]
    final_epoch: Optional[int]

    def as_csv_row(self) -> list[str | float | int]:
        return [
            self.run,
            self.acc_10k,
            self.steps_to_thr,
            self.aulc_10k,
            self.eff_nonzero_whh_pct,
            self.spectral_gap,
            self.algorithm,
            (self.hidden_size if self.hidden_size is not None else ""),
            (self.num_tasks if self.num_tasks is not None else ""),
            (self.seed if self.seed is not None else ""),
            (self.final_epoch if self.final_epoch is not None else ""),
        ]


CSV_HEADER = [
    "run",
    "acc_10k",
    "steps_to_thr",
    "aulc_10k",
    "eff_nonzero_whh_pct",
    "spectral_gap",
    "algorithm",
    "hidden_size",
    "num_tasks",
    "seed",
    "final_epoch",
]


def summarize_run(
    run_dir: Path,
    *,
    max_steps: int = 10_000,
    thr: float = 0.80,
    ema_beta: float = 0.98,
    tau_scale: float = 1e-3,
) -> Optional[Row]:
    steps_npz = _load_npz_safe(run_dir / "metrics_steps.npz")
    epoch_npz = _load_npz_safe(run_dir / "metrics_epoch.npz")
    ckpt_path = _find_latest_ckpt(run_dir)

    if (steps_npz is None) and (epoch_npz is None) and (ckpt_path is None):
        return None

    acc_10k = float("nan")
    final_epoch = None
    if epoch_npz is not None:
        val_mean = epoch_npz.get("val_acc_epoch_mean", None)
        epoch_idx = epoch_npz.get("epoch_idx", None)
        if val_mean is not None and val_mean.size > 0:
            acc_10k = float(val_mean[-1])
        if epoch_idx is not None and epoch_idx.size > 0:
            final_epoch = int(epoch_idx[-1])

    steps_to_thr = float("nan")
    aulc_10k = float("nan")
    if steps_npz is not None:
        step_idx = np.asarray(steps_npz.get("step_idx", []), dtype=float)
        acc_train = np.asarray(steps_npz.get("acc_train_step", []), dtype=float)

        if step_idx.size > 0 and acc_train.size == step_idx.size:
            if np.isfinite(max_steps):
                m = step_idx <= max_steps
                if m.any():
                    step_idx = step_idx[m]
                    acc_train = acc_train[m]
            acc_smooth = _ema(acc_train, beta=ema_beta)
            idx = _first_idx_ge(acc_smooth, thr)
            if idx is not None:
                steps_to_thr = float(step_idx[idx])
            else:
                steps_to_thr = float("inf")
            aulc_10k = _area_under_curve(acc_train)

        if not np.isfinite(acc_10k) and acc_train.size > 0:
            acc_10k = float(acc_train[-1])

    eff_nonzero_pct = float("nan")
    gap = float("nan")
    if ckpt_path is not None:
        W = _load_w_hh_from_ckpt(ckpt_path)
        if W is not None:
            eff_nonzero_pct = _eff_nonzero_pct(W, tau_scale=tau_scale)
            gap = _spectral_gap(W)

    alg = ""
    H = None
    n_tasks = None
    seed = None
    if ckpt_path is not None:
        cfg = _load_config_from_ckpt(ckpt_path)
        try:
            alg = str((cfg.get("optim", {}) or {}).get("algorithm", "")).lower()
        except Exception:
            pass
        try:
            H = int((cfg.get("model", {}) or {}).get("hidden_size", None))
        except Exception:
            H = None
        try:
            tasks = cfg.get("tasks", [])
            if isinstance(tasks, str):
                n_tasks = 1
            elif isinstance(tasks, (list, tuple)):
                n_tasks = len(tasks)
        except Exception:
            n_tasks = None
        try:
            seed = int(cfg.get("seed", None))
        except Exception:
            seed = None

    return Row(
        run=str(run_dir),
        acc_10k=acc_10k,
        steps_to_thr=steps_to_thr,
        aulc_10k=aulc_10k,
        eff_nonzero_whh_pct=eff_nonzero_pct,
        spectral_gap=gap,
        algorithm=alg,
        hidden_size=H,
        num_tasks=n_tasks,
        seed=seed,
        final_epoch=final_epoch,
    )



def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Summarize EG vs GD runs into a CSV table.")
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or globs (each should contain metrics_*.npz and checkpoints).",
    )
    p.add_argument("--out", type=str, default="summary.csv", help="Output CSV path.")
    p.add_argument("--thr", type=float, default=0.80, help="Threshold for Steps→thr (default 0.80).")
    p.add_argument("--ema_beta", type=float, default=0.98, help="EMA smoothing for step accuracy.")
    p.add_argument("--max_steps", type=int, default=10_000, help="Max steps window for AULC/threshold.")
    p.add_argument("--tau_scale", type=float, default=1e-3, help="Scale for τ in effective nonzero (%).")
    args = p.parse_args(argv)

    run_dirs = _glob_runs(args.runs)
    if not run_dirs:
        print("No runs found for the provided paths/globs.", file=sys.stderr)
        return 2

    rows: list[Row] = []
    for rd in run_dirs:
        r = summarize_run(
            rd,
            max_steps=int(args.max_steps),
            thr=float(args.thr),
            ema_beta=float(args.ema_beta),
            tau_scale=float(args.tau_scale),
        )
        if r is not None:
            rows.append(r)
        else:
            print(f"[warn] skipped {rd} (no metrics/checkpoints found)", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in rows:
            values = []
            for v in r.as_csv_row():
                if isinstance(v, str):
                    if "," in v:
                        values.append(f"\"{v}\"")
                    else:
                        values.append(v)
                elif v is None:
                    values.append("")
                elif isinstance(v, float) and not np.isfinite(v):
                    values.append("")
                else:
                    values.append(str(v))
            f.write(",".join(values) + "\n")

    print(f"[ok] wrote {out} with {len(rows)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
