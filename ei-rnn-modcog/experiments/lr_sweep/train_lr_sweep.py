import argparse, csv, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

from src.data import mod_cog_tasks as mct
from src.models.ei_rnn import EIRNN, EIConfig
from src.optim.sgd_eg import SGD_EG
from src.training.losses import ModCogLossCombined, decision_mask_from_inputs
from src.training.metrics import accuracy_with_fixation
from src.utils.seeding import set_seed_all, pick_device, device_name

EG_COLOR = "#1f77b4"
GD_COLOR = "#d62728"
TARGET_COLOR = "#6c757d"
COLORS = {"eg": EG_COLOR, "gd": GD_COLOR}

def _set_pub_style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

def ensure_tensor(x, dtype, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)

@torch.no_grad()
def evaluate_task(model, ds, device, mask_thresh, batches=50, input_noise_std=0.0):
    model.eval()
    crit = ModCogLossCombined()
    tot_l = tot_fix = tot_dec = 0.0
    acc_all = acc_dec = 0.0; n_dec = 0
    for _ in range(batches):
        X, Y = ds()
        X = ensure_tensor(X, torch.float32, device)
        Y = ensure_tensor(Y, torch.long,    device)
        dec_mask = decision_mask_from_inputs(X, thresh=mask_thresh)
        X_in = X if input_noise_std == 0 else X + torch.normal(0.0, input_noise_std, size=X.shape, device=X.device)
        logits = model(X_in)
        l, lf, ld = crit(logits, Y, dec_mask)
        tot_l += l.item(); tot_fix += lf.item(); tot_dec += ld.item()
        a_all, a_dec = accuracy_with_fixation(logits, Y, dec_mask)
        acc_all += a_all
        if not math.isnan(a_dec):
            acc_dec += a_dec
            n_dec += 1
    model.train()
    return {"loss": tot_l/batches, "loss_fix": tot_fix/batches, "loss_dec": tot_dec/batches,
            "acc": acc_all/batches, "acc_dec": acc_dec/max(1, n_dec)}

def build_task_datasets(task_names, batch_size, seq_len, data_cfg, device=None):
    source = (data_cfg.get("source", "online")).lower()
    out = {}

    if source == "online":
        from neurogym import Dataset
        for t in task_names:
            env_fn = getattr(mct, t)
            tr_env, va_env = env_fn(), env_fn()
            tr = Dataset(tr_env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
            va = Dataset(va_env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
            out[t] = (tr, va, tr_env.action_space.n, tr_env.observation_space.shape[-1])
        return out

    from src.data.dataset_cached import (
        build_cached_dataset, split_cached, save_cached_npz, load_cached_npz, CachedBatcher
    )
    cache_cfg = data_cfg.get("cache", {}) or {}
    nb      = int(cache_cfg.get("num_batches", 200))
    vfrac   = float(cache_cfg.get("val_frac", 0.1))
    seed    = cache_cfg.get("seed", None)
    build   = bool(cache_cfg.get("build_if_missing", True))
    paths   = cache_cfg.get("paths", {}) or {}

    device_str = device if device is not None else "cpu"
    for t in task_names:
        p = paths.get(t, f"runs/cache/{t}_{nb}x{batch_size}x{seq_len}.npz")
        p = str(Path(p))
        try:
            cached = load_cached_npz(p)
        except Exception:
            if not build:
                raise FileNotFoundError(f"Missing cache for {t}: {p} (set build_if_missing: true)")
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            cached, _ = build_cached_dataset(t, nb, batch_size, seq_len, seed=seed)
            save_cached_npz(p, cached, meta={"task": t, "seq_len": seq_len})
        tr_cached, va_cached = split_cached(cached, val_frac=vfrac, seed=seed)

        env = getattr(mct, t)()
        action_n  = env.action_space.n
        input_dim = env.observation_space.shape[-1]

        tr = CachedBatcher(tr_cached, batch_size, device=device_str)
        va = CachedBatcher(va_cached, batch_size, device=device_str)
        out[t] = (tr, va, int(action_n), int(input_dim))
    return out

def make_model(input_dim, output_dim, model_cfg):
    beta = float(model_cfg.get("softplus_beta", 8.0))
    th   = float(model_cfg.get("softplus_threshold", 20.0))
    return EIRNN(
        input_size=input_dim,
        output_size=output_dim,
        cfg=EIConfig(
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            exc_frac=float(model_cfg.get("exc_frac", 0.8)),
            spectral_radius=float(model_cfg.get("spectral_radius", 1.2)),
            input_scale=float(model_cfg.get("input_scale", 1.0)),
            leak=float(model_cfg.get("leak", 1.0)),
            nonlinearity=(model_cfg.get("nonlinearity", "softplus")).lower(),
            readout=(model_cfg.get("readout", "all")).lower(),
            softplus_beta=beta,
            softplus_threshold=th,
        ),
    )

def run_once(cfg, device, dsets, algo, lr_value, seed):
    set_seed_all(int(seed), deterministic=False)
    tasks = cfg.get("tasks", ["dm1"])
    tasks = [tasks] if isinstance(tasks, str) else tasks

    data     = cfg.get("data", {})
    train    = cfg.get("train", {})
    modelcfg = cfg.get("model", {})
    optimcfg = cfg.get("optim", {})
    eg_cfg   = optimcfg.get("eg", {}) or {}
    gd_cfg   = optimcfg.get("gd", {}) or {}

    E         = int(train.get("num_epochs", 5))
    S         = int(train.get("steps_per_epoch", 300))
    V         = int(train.get("val_batches", 10))
    grad_clip = float(train.get("grad_clip", 1.0))
    noise     = float(train.get("input_noise_std", 0.0))
    fixdown   = float(train.get("fixdown_weight", 0.5))
    renorm_every = int(train.get("spectral_renorm_every", 0))
    mask_thr  = float(train.get("mask_threshold", 0.5))
    label_smoothing = float(train.get("label_smoothing", 0.1))

    X0, _ = dsets[tasks[0]][0]()
    input_dim = X0.shape[-1]
    output_dim = dsets[tasks[0]][2]
    model = make_model(input_dim, output_dim, modelcfg).to(device)

    if algo == "eg":
        lr_eg = float(lr_value)
        lr_gd = float(gd_cfg.get("lr", 0.05)) 
        mom_eg = float(eg_cfg.get("momentum", 0.0))
        wd_eg  = float(eg_cfg.get("weight_decay", 1e-5))
        minmag = float(eg_cfg.get("min_magnitude", 1e-6))
        mom_gd = float(gd_cfg.get("momentum", 0.9))
        wd_gd  = float(gd_cfg.get("weight_decay", 1e-5))
        eg_params = [model.W_hh]
        gd_params = [p for _, p in model.named_parameters() if p is not model.W_hh]
        opt = SGD_EG([
            {"params": eg_params, "update_alg": "eg", "lr": lr_eg, "momentum": mom_eg,
             "weight_decay": wd_eg, "min_magnitude": minmag},
            {"params": gd_params, "update_alg": "gd", "lr": lr_gd, "momentum": mom_gd, "weight_decay": wd_gd},
        ])
    elif algo == "gd":
        lr_gd = float(lr_value)
        mom_gd = float(gd_cfg.get("momentum", 0.9))
        wd_gd  = float(gd_cfg.get("weight_decay", 1e-5))
        opt = SGD_EG([{"params": list(model.parameters()), "update_alg": "gd", "lr": lr_gd,
                       "momentum": mom_gd, "weight_decay": wd_gd}])
    else:
        raise ValueError("algo must be 'eg' or 'gd'")

    crit = ModCogLossCombined(label_smoothing=label_smoothing, fixdown_weight=fixdown)

    best_val_acc = -1.0
    val_acc_hist = []
    steps_seen = 0
    for epoch in range(1, E + 1):
        for i in range(S):
            task = tasks[i % len(tasks)]
            train_ds = dsets[task][0]
            X, Y = train_ds()
            X = ensure_tensor(X, torch.float32, device)
            Y = ensure_tensor(Y, torch.long,    device)
            dec_mask = decision_mask_from_inputs(X, thresh=mask_thr)
            X_in = X if noise == 0 else X + torch.normal(0.0, noise, size=X.shape, device=X.device)
            logits = model(X_in)
            loss, _, _ = crit(logits, Y, dec_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss became non-finite")
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if algo == "gd":
                model.project_EI_()
            if renorm_every and (steps_seen % renorm_every == 0) and steps_seen > 0:
                model.rescale_spectral_radius_(tol=0.10)
            steps_seen += 1

        val = evaluate_task(model, dsets[tasks[0]][1], device, mask_thr, batches=V, input_noise_std=noise)
        val_acc_hist.append(float(val["acc"]))
        if val["acc"] > best_val_acc:
            best_val_acc = float(val["acc"])

    final_val_acc = float(val_acc_hist[-1]) if val_acc_hist else float("nan")
    return {
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "val_acc_hist": val_acc_hist,
        "steps_seen": steps_seen,
    }

def time_to_plateau(acc_hist, window=2, tol=0.002):
    if not acc_hist: return None
    acc = np.asarray(acc_hist, dtype=float)
    if len(acc) <= window: return len(acc)
    for t in range(window, len(acc)):
        if (acc[t] - acc[t - window]) < tol:
            return t
    return len(acc)


def plot_acc_vs_lr(out_png, results_by_algo, xlog=True, target_lr=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, rows in results_by_algo.items():
        lrs    = [r["lr"] for r in rows]
        means  = [float(np.mean(r["best_val_acc_seeds"])) for r in rows]
        stds   = []
        for r in rows:
            vals = np.asarray(r["best_val_acc_seeds"], dtype=float)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        ax.errorbar(
            lrs, means, yerr=stds,
            fmt="-o", linewidth=2, markersize=6, capsize=3,
            label=algo.upper(), color=COLORS.get(algo, None)
        )

    if target_lr:
        for algo, lr in target_lr.items():
            if lr is not None:
                ax.axvline(float(lr), color=TARGET_COLOR, linestyle="--", linewidth=1)

    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel("Learning rate", fontsize=14)
    ax.set_ylabel("Best validation accuracy", fontsize=14)
    ax.legend(frameon=False)
    _set_pub_style(ax)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_plateau_vs_lr(out_png, results_by_algo, steps_per_epoch, target_lr=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, rows in results_by_algo.items():
        lrs    = [r["lr"] for r in rows]

        means_steps = []
        stds_steps  = []
        for r in rows:
            epochs_arr = np.asarray(r["plateau_epochs_seeds"], dtype=float)
            steps_arr  = epochs_arr * float(steps_per_epoch)
            means_steps.append(float(np.mean(steps_arr)))
            stds_steps.append(float(np.std(steps_arr, ddof=1)) if len(steps_arr) > 1 else 0.0)

        ax.errorbar(
            lrs, means_steps, yerr=stds_steps,
            fmt="-o", linewidth=2, markersize=6, capsize=3,
            label=algo.upper(), color=COLORS.get(algo, None)
        )

    if target_lr:
        for algo, lr in target_lr.items():
            if lr is not None:
                ax.axvline(float(lr), color=TARGET_COLOR, linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("learning rate", fontsize=14)
    ax.set_ylabel("steps to plateau (mean ± sd)", fontsize=14)
    ax.legend(frameon=False)
    _set_pub_style(ax)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def _cfg_signature(cfg):
    import hashlib
    keys = ["tasks","data","model","optim","train"]
    blob = json.dumps({k: cfg.get(k, {}) for k in keys}, sort_keys=True, default=float)
    return hashlib.md5(blob.encode()).hexdigest()[:8]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/lrsweep.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    set_seed_all(int(cfg.get("seed", 1)), deterministic=False)
    device = pick_device(cfg.get("device", "auto"))
    print(f"Using device: {device_name(device)}")

    outdir = Path(cfg.get("outdir", "outputs/lrsweep")); outdir.mkdir(parents=True, exist_ok=True)

    data = cfg.get("data", {})
    tasks = cfg.get("tasks", ["dm1"]); tasks = tasks if isinstance(tasks, list) else [tasks]
    seq_len   = int(data.get("seq_len", 350))
    batch_sz  = int(data.get("batch_size", 128))
    dsets = build_task_datasets(tasks, batch_sz, seq_len, data, device=device.type)

    sweep = cfg.get("sweep", {}) or {}
    algos = [a.lower() for a in sweep.get("algorithms", ["eg","gd"])]

    shared = sweep.get("lr_shared_grid", None)
    if shared is not None:
        lr_grid = {a: list(shared) for a in algos}
    else:
        lr_grid = sweep.get("lr_grid", {"eg":[0.05,0.1,0.2,0.5], "gd":[0.001,0.003,0.01,0.03]})

    seeds = [int(s) for s in sweep.get("seeds", [1])]
    resume = bool(sweep.get("resume", True))
    skip_existing = bool(sweep.get("skip_existing", True))
    plot_cfg = sweep.get("plot", {})
    target_lr = plot_cfg.get("target_lr", {})

    train_cfg = cfg.get("train", {})
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 300))

    sig = _cfg_signature(cfg)
    out_csv = outdir / "lrsweep_results_detailed.csv"
    existing_rows = []
    if resume and out_csv.exists():
        with open(out_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("sig", "") == sig:
                    existing_rows.append(row)
    already_done = {(row["algo"], float(row["lr"]), int(row.get("seed", 0))) for row in existing_rows}

    results_rows = []
    results_by_algo = {a: [] for a in algos}

    for algo in algos:
        grid = lr_grid.get(algo, [])
        algo_rows = []
        for lr in grid:
            best_acc_seeds = []
            plateau_epochs_seeds = []
            for seed in seeds:
                if skip_existing and (algo, float(lr), int(seed)) in already_done:
                    rows_this = [er for er in existing_rows
                                 if er["algo"]==algo and float(er["lr"])==float(lr) and int(er["seed"])==int(seed)]
                    if rows_this:
                        best_acc_seeds.append(float(rows_this[0]["best_val_acc"]))
                        plateau_epochs_seeds.append(int(float(rows_this[0]["plateau_epoch"])))
                        continue

                r = run_once(cfg, device, dsets, algo=algo, lr_value=lr, seed=seed)
                best_acc_seeds.append(r["best_val_acc"])
                p_epoch = time_to_plateau(r["val_acc_hist"],
                                          window=int(sweep.get("plateau",{}).get("window",2)),
                                          tol=float(sweep.get("plateau",{}).get("tol",0.002)))
                plateau_epochs_seeds.append(p_epoch)
                results_rows.append({
                    "sig": sig,
                    "algo": algo,
                    "lr": float(lr),
                    "seed": int(seed),
                    "final_val_acc": r["final_val_acc"],
                    "best_val_acc": r["best_val_acc"],
                    "plateau_epoch": p_epoch,
                    "steps_seen": r["steps_seen"],
                })
            algo_rows.append({
                "lr": float(lr),
                "best_val_acc_seeds": best_acc_seeds,
                "plateau_epochs_seeds": plateau_epochs_seeds,
            })
        results_by_algo[algo] = algo_rows

    all_rows = existing_rows + results_rows
    if all_rows:
        with open(out_csv, "w", newline="") as f:
            fieldnames = ["sig","algo","lr","seed","final_val_acc","best_val_acc","plateau_epoch","steps_seen"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(all_rows)
        with open(outdir / "lrsweep_summary.json", "w") as f:
            json.dump(results_by_algo, f, indent=2, default=float)
        print(f"Saved results to: {out_csv}")
    else:
        print("No new rows to write (resume/skip found all pairs already present).")

    if bool(plot_cfg.get("make_plots", True)):
        plot_acc_vs_lr(str(outdir / "lrsweep_acc_vs_lr.png"), results_by_algo, xlog=True, target_lr=target_lr)
        if bool(plot_cfg.get("plot_plateau", True)):
            plot_plateau_vs_lr(str(outdir / "lrsweep_plateau_vs_lr.png"), results_by_algo,
                               steps_per_epoch=steps_per_epoch, target_lr=target_lr)
    print("Done.")

if __name__ == "__main__":
    main()
