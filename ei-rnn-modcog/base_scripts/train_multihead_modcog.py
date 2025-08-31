import argparse, time, math, random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import mod_cog_tasks as mct
from src.analysis import viz_training as viz
from neurogym import Dataset
from src.models.ei_rnn import EIRNN, EIConfig
from src.optim.sgd_eg import SGD_EG
from src.utils.seeding import set_seed_all, pick_device, device_name

def ensure_tensor(x, dtype, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)

@torch.no_grad()
def decision_mask_from_inputs(X: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (X[..., 0] < thresh)

class ModCogLossCombined(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, fixdown_weight: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.fixdown_weight = float(fixdown_weight)
    def forward(self, outputs, labels, dec_mask):
        B, T, C = outputs.shape
        target_fix = outputs.new_zeros(B, T, C)
        target_fix[..., 0] = 1.0
        loss_fix = self.mse(outputs[~dec_mask], target_fix[~dec_mask])
        loss_dec = self.ce(outputs[dec_mask], 1 + labels[dec_mask])
        if self.fixdown_weight > 0.0 and dec_mask.any():
            fix_logits_dec = outputs[..., 0][dec_mask]
            loss_fixdown = (fix_logits_dec ** 2).mean() * self.fixdown_weight
        else:
            loss_fixdown = outputs.sum() * 0.0
        return loss_fix + loss_dec + loss_fixdown, loss_fix.detach(), (loss_dec + loss_fixdown).detach()

@torch.no_grad()
def accuracy_with_fixation(outputs, labels, dec_mask):
    pred = outputs.argmax(dim=-1) 
    labels_shifted = outputs.new_zeros(labels.shape, dtype=torch.long)
    labels_shifted.copy_(labels + 1)
    labels_full = labels_shifted.clone()
    labels_full[~dec_mask] = 0
    acc_all = (pred == labels_full).float().mean().item()
    acc_dec = (pred[dec_mask] == labels_full[dec_mask]).float().mean().item() if dec_mask.any() else float("nan")
    return acc_all, acc_dec

@torch.no_grad()
def evaluate_task(core, heads, task, ds, device, mask_thresh, batches=50, input_noise_std=0.0):
    core.eval()
    crit = ModCogLossCombined()
    tot_l = tot_fix = tot_dec = 0.0
    acc_all = acc_dec = 0.0; n_dec = 0
    for _ in range(batches):
        X, Y = ds()
        X = ensure_tensor(X, torch.float32, device)
        Y = ensure_tensor(Y, torch.long,    device)
        dec_mask = decision_mask_from_inputs(X, thresh=mask_thresh)
        X_in = X if input_noise_std == 0 else X + torch.normal(0.0, input_noise_std, size=X.shape, device=X.device)
        Hseq = run_core_hidden(core, X_in)
        logits = heads(Hseq, task) 
        l, lf, ld = crit(logits, Y, dec_mask)
        tot_l += l.item(); tot_fix += lf.item(); tot_dec += ld.item()
        a_all, a_dec = accuracy_with_fixation(logits, Y, dec_mask)
        acc_all += a_all
        if not math.isnan(a_dec): acc_dec += a_dec; n_dec += 1
    core.train()
    return {"loss": tot_l/batches, "loss_fix": tot_fix/batches, "loss_dec": tot_dec/batches,
            "acc": acc_all/batches, "acc_dec": acc_dec/max(1,n_dec)}

def build_task_datasets(task_names, batch_size, seq_len, data_cfg):
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

    from pathlib import Path
    from src.data.dataset_cached import (
        build_cached_dataset, split_cached, save_cached_npz, load_cached_npz, CachedBatcher
    )

    cache_cfg = data_cfg.get("cache", {}) or {}
    nb      = int(cache_cfg.get("num_batches", 200))
    vfrac   = float(cache_cfg.get("val_frac", 0.1))
    seed    = cache_cfg.get("seed", None)
    build   = bool(cache_cfg.get("build_if_missing", True))
    paths   = cache_cfg.get("paths", {}) or {}

    for t in task_names:
        p = paths.get(t, f"runs/cache/{t}_{nb}x{batch_size}x{seq_len}.npz")
        p = str(Path(p))
        try:
            cached = load_cached_npz(p)
        except Exception:
            if not build:
                raise FileNotFoundError(f"Missing cache for {t}: {p} (set build_if_missing: true)")
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            cached, action_n = build_cached_dataset(t, nb, batch_size, seq_len, seed=seed)
            save_cached_npz(p, cached, meta={"task": t, "seq_len": seq_len})
        tr_cached, va_cached = split_cached(cached, val_frac=vfrac, seed=seed)
        from neurogym import Dataset as NGDataset
        env = getattr(mct, t)()
        action_n = env.action_space.n
        input_dim = env.observation_space.shape[-1]
        tr = CachedBatcher(tr_cached, batch_size)
        va = CachedBatcher(va_cached, batch_size)
        out[t] = (tr, va, int(action_n), int(input_dim))
    return out


def choose_task(tasks, i_step, mode, weights):
    if mode == "round_robin": return tasks[i_step % len(tasks)]
    if not weights or len(weights) != len(tasks): return random.choice(tasks)
    return random.choices(tasks, weights=weights, k=1)[0]

class MultiHeadReadout(nn.Module):
    def __init__(self, hidden_size, head_dims: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_size, C) for t, C in head_dims.items()})
    def forward(self, h_seq, task: str):
        B, T, H = h_seq.shape
        out = self.heads[task](h_seq.reshape(B*T, H)).reshape(B, T, -1)
        return out

def run_core_hidden(model: EIRNN, x: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    H = model.cfg.hidden_size
    h = x.new_zeros(B, H)
    hs = []
    Wx, Wh, b = model.W_xh, model.W_hh, model.b_h
    for t in range(T):
        h = F.relu(x[:, t, :] @ Wx.T + h @ Wh.T + b)
        hs.append(h.unsqueeze(1))
    return torch.cat(hs, dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    data_cfg  = cfg.get("data", {})

    seed      = int(cfg.get("seed", 7))
    set_seed_all(seed, deterministic=False)
    dev_str   = cfg.get("device", "auto")
    outdir    = Path(cfg.get("outdir", "runs/exp")); outdir.mkdir(parents=True, exist_ok=True)
    tasks     = cfg.get("tasks", ["dm1"]); tasks = [tasks] if isinstance(tasks, str) else tasks

    data      = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    optim     = cfg.get("optim", {})
    train     = cfg.get("train", {})

    seq_len   = int(data.get("seq_len", 350))
    batch_sz  = int(data.get("batch_size", 128))
    hidden    = int(model_cfg.get("hidden_size", 256))
    exc_frac  = float(model_cfg.get("exc_frac", 0.8))
    spectral_radius = float(model_cfg.get("spectral_radius", 1.2))
    input_scale     = float(model_cfg.get("input_scale", 1.0))

    algo      = (optim.get("algorithm", "eg")).lower()
    eg        = optim.get("eg", {});  gd = optim.get("gd", {})
    lr_eg     = float(eg.get("lr", 1.5)); mom_eg = float(eg.get("momentum", 0.0))
    wd_eg     = float(eg.get("weight_decay", 1e-5)); minmag = float(eg.get("min_magnitude", 1e-6))
    lr_gd     = float(gd.get("lr", 0.01)); mom_gd = float(gd.get("momentum", 0.9)); wd_gd = float(gd.get("weight_decay", 1e-5))

    E         = int(train.get("num_epochs", 20))
    S         = int(train.get("steps_per_epoch", 500))
    V         = int(train.get("val_batches", 50))
    grad_clip = float(train.get("grad_clip", 1.0))
    noise     = float(train.get("input_noise_std", 0.1))
    fixdown   = float(train.get("fixdown_weight", 0.05))
    renorm_every = int(train.get("spectral_renorm_every", 1000))
    mask_thr  = float(train.get("mask_threshold", 0.5))
    sampling  = train.get("task_sampling", "round_robin")
    weights   = train.get("task_weights", []) or []
    label_smoothing = float(train.get("label_smoothing", 0.1))

    device = pick_device(dev_str)
    print(f"Using device: {device_name(device)}")

    dsets = build_task_datasets(tasks, batch_sz, seq_len, data_cfg)
    input_dims = {t: dsets[t][3] for t in tasks}
    if len(set(input_dims.values())) != 1:
        raise ValueError(f"All tasks must share the same input dim for a shared core, got {input_dims}")
    head_dims = {t: dsets[t][2] for t in tasks}

    X0, _ = dsets[tasks[0]][0]()
    input_dim = X0.shape[-1]
    print(f"Multi-head over tasks={tasks} | in={input_dim}, heads={head_dims}")

    core = EIRNN(input_dim, output_size=max(head_dims.values()),
                 cfg=EIConfig(hidden_size=hidden, exc_frac=exc_frac,
                              spectral_radius=spectral_radius, input_scale=input_scale)).to(device)
    heads = MultiHeadReadout(hidden, head_dims).to(device)

    params = []
    if algo == "eg":
        params.append({"params": [core.W_hh], "update_alg": "eg", "lr": lr_eg, "momentum": mom_eg,
                       "weight_decay": wd_eg, "min_magnitude": minmag})
        params.append({"params": [p for n,p in core.named_parameters() if p is not core.W_hh],
                       "update_alg": "gd", "lr": lr_gd, "momentum": mom_gd, "weight_decay": wd_gd})
        params.append({"params": list(heads.parameters()),
                       "update_alg": "gd", "lr": lr_gd, "momentum": mom_gd, "weight_decay": wd_gd})
    elif algo == "gd":
        params.append({"params": list(core.parameters()) + list(heads.parameters()),
                       "update_alg": "gd", "lr": lr_gd, "momentum": mom_gd, "weight_decay": wd_gd})
    else:
        raise ValueError("optim.algorithm must be 'eg' or 'gd'")
    opt = SGD_EG(params)

    crit = ModCogLossCombined(label_smoothing=label_smoothing, fixdown_weight=fixdown)

    epoch_train_losses: list[float] = []
    acc_history: dict[str, list[float]] = {t: [] for t in tasks}

    global_step = 0
    for epoch in range(1, E + 1):
        t0 = time.time()
        meter = {"loss": 0.0, "acc": 0.0, "acc_dec": 0.0}

        for i in range(S):
            task = choose_task(tasks, i, sampling, weights)
            train_ds = dsets[task][0]

            X, Y = train_ds()
            X = ensure_tensor(X, torch.float32, device)
            Y = ensure_tensor(Y, torch.long,    device)

            dec_mask = decision_mask_from_inputs(X, thresh=mask_thr)
            X_in = X if noise == 0 else X + torch.normal(0.0, noise, size=X.shape, device=X.device)

            Hseq   = run_core_hidden(core, X_in)
            logits = heads(Hseq, task)

            loss, _, _ = crit(logits, Y, dec_mask)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(list(core.parameters()) + list(heads.parameters()), grad_clip)
            opt.step()

            if renorm_every and (global_step % renorm_every == 0) and global_step > 0:
                core.rescale_spectral_radius_()

            acc_all, acc_dec = accuracy_with_fixation(logits, Y, dec_mask)
            meter["loss"] += loss.item()
            meter["acc"]  += acc_all
            meter["acc_dec"] += (0.0 if math.isnan(acc_dec) else acc_dec)

            global_step += 1

        for k in meter: meter[k] /= S
        epoch_train_losses.append(meter["loss"])
        print(f"[epoch {epoch:03d}] train: loss {meter['loss']:.3f} | acc {meter['acc']*100:5.1f}% "
              f"| dec {meter['acc_dec']*100:5.1f}% | {(time.time()-t0):.2f}s")

        per_task_val_acc = {}
        for t in tasks:
            val = evaluate_task(core, heads, t, dsets[t][1], device, mask_thr, batches=V, input_noise_std=noise)
            per_task_val_acc[t] = val["acc"]
            acc_history[t].append(val["acc"])
            print(f"  [val:{t}] loss {val['loss']:.3f} | acc {val['acc']*100:5.1f}% | dec {val['acc_dec']*100:5.1f}%")

        viz.save_loss_curve(epoch_train_losses, str(outdir / "loss.png"), smooth=1)
        viz.save_per_task_accuracy_curves(acc_history, str(outdir / "acc_tasks.png"), smooth=1)
        Xv, Yv = dsets[tasks[0]][1]()
        Xv_t = torch.from_numpy(Xv).float().to(device)
        dec_mask_v = decision_mask_from_inputs(Xv_t, thresh=mask_thr)
        Hseq_v = run_core_hidden(core, Xv_t)
        logits_v = heads(Hseq_v, tasks[0])
        viz.save_logits_time_trace(logits_v, str(outdir / f"example_trial_epoch{epoch:03d}.png"),
                                   dec_mask=dec_mask_v, topk=3, sample_idx=0, title="Example trial logits")
        viz.save_weight_hists_W_hh(core.W_hh, core.sign_vec, str(outdir / f"weights_hist_epoch{epoch:03d}.png"))
        ckpt = outdir / f"multihead_epoch{epoch:03d}.pt"
        torch.save({"core": core.state_dict(), "heads": heads.state_dict(),
                    "head_dims": head_dims, "config": cfg, "epoch": epoch}, ckpt)

    print(f"Done. Last checkpoint: {ckpt}")

if __name__ == "__main__":
    main()