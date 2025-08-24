# fixed_points/train.py
import os, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.data import mod_cog_tasks as mct
from src.analysis import viz_training as viz
from src.models.ei_rnn import EIRNN, EIConfig
from src.optim.sgd_eg import SGD_EG
from src.utils.seeding import set_seed_all, pick_device, device_name


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
    ap.add_argument("--outdir", default=cfg.get("outdir", "experiments/runs/smoke/eg/pI=0.35/seed=000"))

    # core setup
    ap.add_argument("--optim", choices=["eg", "gd"], default=cfg.get("optim", "eg"))
    ap.add_argument("--pI", type=float, default=cfg.get("pI", 0.35))
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    ap.add_argument("--task", type=str, default=cfg.get("task", "dm1"))
    ap.add_argument("--hidden", type=int, default=cfg.get("hidden", 256))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 10))
    ap.add_argument("--steps", type=int, default=cfg.get("steps", 500))
    ap.add_argument("--seq_len", type=int, default=cfg.get("seq_len", 350))
    ap.add_argument("--batch", type=int, default=cfg.get("batch", 128))

    # model hyperparams
    ap.add_argument("--spectral_radius", type=float, default=cfg.get("spectral_radius", 1.2))
    ap.add_argument("--input_scale", type=float, default=cfg.get("input_scale", 1.0))

    # opt hyperparams
    ap.add_argument("--lr_eg", type=float, default=cfg.get("lr_eg", 1.5))
    ap.add_argument("--lr_gd", type=float, default=cfg.get("lr_gd", 0.01))

    # device/perf
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=cfg.get("device", "auto"))
    ap.add_argument("--amp", dest="amp", action="store_true", default=cfg.get("amp", True),
                    help="Use mixed precision on CUDA (bfloat16 preferred).")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    ap.add_argument("--allow_tf32", action="store_true", default=cfg.get("allow_tf32", True),
                    help="Allow TF32 on CUDA matmul/cudnn.")

    # training knobs
    ap.add_argument("--label_smoothing", type=float, default=cfg.get("label_smoothing", 0.1))
    ap.add_argument("--fixdown_weight", type=float, default=cfg.get("fixdown_weight", 0.05))
    ap.add_argument("--grad_clip", type=float, default=cfg.get("grad_clip", 1.0))
    ap.add_argument("--val_batches", type=int, default=cfg.get("val_batches", 50))
    ap.add_argument("--mask_threshold", type=float, default=cfg.get("mask_threshold", 0.5))
    ap.add_argument("--input_noise_std", type=float, default=cfg.get("input_noise_std", 0.1))
    ap.add_argument("--renorm_every", type=int, default=cfg.get("renorm_every", 1000))

    # logging
    ap.add_argument("--log_every", type=int, default=cfg.get("log_every", 50),
                    help="Print a step log every N updates.")
    return ap


@torch.no_grad()
def decision_mask_from_inputs(X: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    # fixation channel 0 by convention
    return (X[..., 0] < thresh)


class ModCogLossCombined(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, fixdown_weight: float = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.fixdown_weight = float(fixdown_weight)

    def forward(self, outputs, labels, dec_mask):
        B, T, C = outputs.shape
        target_fix = outputs.new_zeros(B, T, C)
        target_fix[..., 0] = 1.0
        loss_fix = self.mse(outputs[~dec_mask], target_fix[~dec_mask]) if (~dec_mask).any() else outputs.sum() * 0.0
        if dec_mask.any():
            loss_dec = self.ce(outputs[dec_mask], 1 + labels[dec_mask])
            fix_logits_dec = outputs[..., 0][dec_mask]
            loss_fixdown = (fix_logits_dec ** 2).mean() * self.fixdown_weight
        else:
            loss_dec = outputs.sum() * 0.0
            loss_fixdown = outputs.sum() * 0.0
        return loss_fix + loss_dec + loss_fixdown, loss_fix.detach(), (loss_dec + loss_fixdown).detach()


@torch.no_grad()
def accuracy_with_fixation(outputs, labels, dec_mask):
    pred = outputs.argmax(dim=-1)
    labels_shifted = outputs.new_zeros(labels.shape, dtype=torch.long)
    labels_shifted.copy_(labels + 1)  # decision labels live in 1..C-1
    labels_full = labels_shifted.clone()
    labels_full[~dec_mask] = 0       # fixation index is 0
    acc_all = (pred == labels_full).float().mean().item()
    acc_dec = (pred[dec_mask] == labels_full[dec_mask]).float().mean().item() if dec_mask.any() else float("nan")
    return acc_all, acc_dec


def build_task(task_name: str, batch_size: int, seq_len: int):
    env_fn = getattr(mct, task_name)
    tr_env, va_env = env_fn(), env_fn()
    from neurogym import Dataset
    tr = Dataset(tr_env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
    va = Dataset(va_env, batch_size=batch_size, seq_len=seq_len, batch_first=True)
    action_n = tr_env.action_space.n
    input_dim = tr_env.observation_space.shape[-1]
    return tr, va, input_dim, action_n


def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    args = _parse_with_config(build_parser)
    set_seed_all(args.seed, deterministic=False)

    # Device pick (supports auto/cpu/cuda). Respect CUDA perf toggles.
    device = pick_device(args.device)
    if device.type == "cuda":
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"Using device: {device_name(device)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tr, va, input_dim, output_dim = build_task(args.task, args.batch, args.seq_len)

    exc_frac = 1.0 - float(args.pI)
    model = EIRNN(
        input_size=input_dim,
        output_size=output_dim,
        cfg=EIConfig(hidden_size=args.hidden, exc_frac=exc_frac,
                     spectral_radius=args.spectral_radius, input_scale=args.input_scale),
    ).to(device)

    if args.optim == "eg":
        eg_params = [model.W_hh]
        gd_params = [p for _, p in model.named_parameters() if p is not model.W_hh]
        opt = SGD_EG([
            {"params": eg_params, "update_alg": "eg", "lr": args.lr_eg, "momentum": 0.0,
             "weight_decay": 1e-5, "min_magnitude": 1e-6},
            {"params": gd_params, "update_alg": "gd", "lr": args.lr_gd, "momentum": 0.9, "weight_decay": 1e-5},
        ])
    else:
        opt = SGD_EG([{"params": list(model.parameters()), "update_alg": "gd", "lr": args.lr_gd,
                       "momentum": 0.9, "weight_decay": 1e-5}])

    crit = ModCogLossCombined(label_smoothing=args.label_smoothing, fixdown_weight=args.fixdown_weight)

    # Logging file
    logs_path = outdir / "logs.jsonl"
    with open(logs_path, "w"):
        pass

    def logj(obj):
        with open(logs_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    # AMP setup
    use_amp = (device.type == "cuda") and bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    epoch_losses = []
    acc_hist = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        meter = {"loss": 0.0, "acc": 0.0, "acc_dec": 0.0}
        t_epoch0 = time.time()
        t_block0 = t_epoch0

        for step in range(args.steps):
            X, Y = tr()
            X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
            Y = torch.from_numpy(Y).to(device=device, dtype=torch.long)

            dec_mask = decision_mask_from_inputs(X, thresh=args.mask_threshold)
            X_in = X if args.input_noise_std == 0 else X + torch.normal(0.0, args.input_noise_std, size=X.shape, device=X.device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(X_in)
                loss, _, _ = crit(logits, Y, dec_mask)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                # Unscale before clipping under AMP
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if args.renorm_every and (global_step % args.renorm_every == 0) and global_step > 0:
                model.rescale_spectral_radius_()

            # quick metrics
            a_all, a_dec = accuracy_with_fixation(logits, Y, dec_mask)
            meter["loss"] += loss.item()
            meter["acc"] += a_all
            meter["acc_dec"] += (0.0 if math.isnan(a_dec) else a_dec)

            # step logging
            do_log = ((step + 1) % max(1, args.log_every) == 0) or (step == 0) or (step + 1 == args.steps)
            if do_log:
                dt_block = time.time() - t_block0
                t_block0 = time.time()
                steps_left = (args.steps - (step + 1)) + (args.epochs - epoch) * args.steps
                # average time per step in the last block
                avg_dt = dt_block / max(1, min(args.log_every, step + 1))
                eta = _fmt_time(avg_dt * max(0, steps_left))
                dec_pct = (0.0 if math.isnan(a_dec) else a_dec) * 100.0
                print(
                    f"[ep {epoch:03d}/{args.epochs:03d} | step {step+1:05d}/{args.steps:05d} | gstep {global_step+1:08d}] "
                    f"loss {float(loss.item()):.4f} | acc {a_all*100:5.1f}% | dec {dec_pct:5.1f}% "
                    f"| dt/blk {dt_block:.2f}s | ETA {eta}",
                    flush=True,
                )

            global_step += 1

        # epoch averages
        for k in meter:
            meter[k] /= args.steps
        epoch_losses.append(meter["loss"])

        # validation (averaged over val_batches)
        with torch.no_grad():
            vals = []
            for _ in range(args.val_batches):
                Xv, Yv = va()
                Xv = torch.from_numpy(Xv).to(device=device, dtype=torch.float32)
                Yv = torch.from_numpy(Yv).to(device=device, dtype=torch.long)
                dec_mask_v = decision_mask_from_inputs(Xv, thresh=args.mask_threshold)
                logits_v = model(Xv)
                a_all_v, _ = accuracy_with_fixation(logits_v, Yv, dec_mask_v)
                vals.append(a_all_v)
            acc_hist.append(float(np.mean(vals)))

        print(
            f"[epoch {epoch:03d}] train: loss {meter['loss']:.3f} | acc {meter['acc']*100:5.1f}% "
            f"| dec {meter['acc_dec']*100:5.1f}% | val_acc {acc_hist[-1]*100:5.1f}% "
            f"| time {time.time()-t_epoch0:.2f}s",
            flush=True,
        )
        logj({"epoch": epoch, **meter, "val_acc": acc_hist[-1]})

        # simple viz
        viz.save_loss_curve(epoch_losses, str(outdir / "loss.png"), smooth=(0 if len(epoch_losses) < 3 else 1))
        viz.save_per_task_accuracy_curves({"val": acc_hist}, str(outdir / "acc_val.png"),
                                          smooth=(0 if len(acc_hist) < 3 else 1),
                                          title="Validation accuracy")
        viz.save_weight_hists_W_hh(model.W_hh, model.sign_vec, str(outdir / f"weights_epoch{epoch:03d}.png"))

        # checkpoint (rolling ckpt.pt for downstream)
        torch.save({"model": model.state_dict(), "config": vars(args), "epoch": epoch}, outdir / "ckpt.pt")

    print(f"Done. Checkpoint: {outdir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
