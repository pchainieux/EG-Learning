# python -m fixed_points.plot_ring_fixed_points --run <RUN_DIR> --contexts memory anti
# Outputs: <RUN_DIR>/eval/figs/context=<ctx>_ring_fixedpoints.png

import os, json, glob, math, numpy as np, torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from pathlib import Path

# --- repo deps (same imports you already use)
from src.data import mod_cog_tasks as mct
from neurogym import Dataset
from src.models.ei_rnn import EIRNN, EIConfig

# ---------- small CLI helper copied from your other scripts ----------
def _parse_with_config(build_parser):
    import argparse
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", type=str, default=None)
    cfg_args, remaining = cfg_parser.parse_known_args()
    cfg = {}
    if cfg_args.config:
        import yaml, os as _os
        if not _os.path.isfile(cfg_args.config):
            raise FileNotFoundError(f"Config file not found: {cfg_args.config}")
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    parser = build_parser(cfg)
    return parser.parse_args(remaining)

def build_parser(cfg):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config path")
    ap.add_argument("--run", type=str, default=cfg.get("run", None), required=not bool(cfg))
    ap.add_argument("--task", type=str, default=cfg.get("task", "dm1"))
    ap.add_argument("--contexts", nargs="+", default=cfg.get("contexts", ["memory", "anti"]))
    ap.add_argument("--seq_len", type=int, default=cfg.get("seq_len", 350))
    ap.add_argument("--batch", type=int, default=cfg.get("batch", 128))
    ap.add_argument("--device", type=str, default=cfg.get("device", "cpu"))
    ap.add_argument("--fixation_channel", type=int, default=cfg.get("fixation_channel", 0))
    ap.add_argument("--mask_threshold", type=float, default=cfg.get("mask_threshold", 0.5))
    # z-axis options
    ap.add_argument("--z_mode", choices=["ring", "logit"], default=cfg.get("z_mode", "logit"))
    ap.add_argument("--cos_index", type=int, default=cfg.get("cos_index", 0))
    ap.add_argument("--sin_index", type=int, default=cfg.get("sin_index", 1))
    ap.add_argument("--z_index",  type=int, default=cfg.get("z_index", 0))  # for logit mode (0==fixation)
    ap.add_argument("--trial_index", type=int, default=cfg.get("trial_index", 0))
    ap.add_argument("--outdir", type=str, default=cfg.get("outdir", None))  # defaults to <run>/eval/figs
    return ap

# ---------- model + fp loaders ----------
def rebuild_model_from_ckpt(ck, task="dm1"):
    """Rebuild EIRNN with shapes/config inferred from ckpt (mirrors your other scripts)."""
    cfg = ck.get("config", {})
    # fallbacks if config missing
    H = int(cfg.get("hidden", cfg.get("hidden_size", 256)))
    pI = float(cfg.get("pI", 0.35))
    spectral_radius = float(cfg.get("spectral_radius", 1.2))
    input_scale = float(cfg.get("input_scale", 1.0))
    exc_frac = 1.0 - pI

    env = getattr(mct, task)()
    input_dim = env.observation_space.shape[-1]
    output_dim = env.action_space.n
    model = EIRNN(input_size=input_dim, output_size=output_dim,
                  cfg=EIConfig(hidden_size=H, exc_frac=exc_frac,
                               spectral_radius=spectral_radius, input_scale=input_scale))
    state = ck.get("model") or ck.get("model_state") or ck.get("state_dict")
    if state is None:
        raise RuntimeError("Checkpoint does not contain a model state dict.")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def load_fixed_points(run_dir, context):
    ctx_dir = Path(run_dir) / "eval" / "fixed_points" / f"context={context}"
    pts = []
    # try via index.json for reproducible order
    idx_path = ctx_dir / "index.json"
    if idx_path.is_file():
        meta = json.load(open(idx_path))
        for row in meta:
            # tolerate either 'i' or 'index' styles
            i = row.get("i", row.get("index", None))
            if i is None:
                continue
            fn = ctx_dir / f"fp_{int(i):03d}.npz"
            if fn.is_file():
                pts.append(np.load(fn)["h_star"])
    else:
        for fn in sorted(glob.glob(str(ctx_dir / "fp_*.npz"))):
            pts.append(np.load(fn)["h_star"])
    if not pts:
        raise FileNotFoundError(f"No fixed points found under {ctx_dir}")
    return np.stack(pts, 0)  # Nfp×H

# ---------- dynamics & readout ----------
def softplus(x):  # torch
    return torch.nn.functional.softplus(x)

def step_leaky_softplus(model, h, x_t, alpha=1.0):
    pre = x_t @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    return (1 - alpha) * h + alpha * softplus(pre)

def try_readout(model, h):
    """Return logits from hidden state h. Works for fixed points (no time axis)."""
    if hasattr(model, "readout") and callable(getattr(model, "readout")):
        return model.readout(h)
    # common names
    for wname in ["W_out", "W_ho", "W_yh", "W_hy"]:
        if hasattr(model, wname):
            W = getattr(model, wname)
            b = None
            for bname in ["b_out", "b_y", "b_ho", "b_yh"]:
                if hasattr(model, bname):
                    b = getattr(model, bname); break
            y = h @ W.T
            if b is not None:
                y = y + b
            return y
    # last resort: linear probe on excitatory (not ideal, but avoids crashing)
    return None  # caller should handle None -> z = 0 plane

# ---------- masks from inputs ----------
def epoch_masks_from_inputs(X, fix_chan=0, thresh=0.5):
    """
    Heuristic masks: stim=channels that vary strongly across time; memory=~stim & fixation on;
    response=fixation off; context=early fixation.
    X: (B,T,I) numpy
    """
    B, T, I = X.shape
    fix = X[..., fix_chan] >= thresh
    # find variable channels (exclude fixation)
    var = np.var(X, axis=1).mean(0)  # I
    cand = [i for i in range(I) if i != fix_chan and var[i] > 1e-4]
    stim = (np.abs(X[..., cand]).sum(-1) > 1e-3) if cand else np.zeros((B, T), dtype=bool)

    memory = (~stim) & fix
    response = ~fix
    # context = early fixation (first 25% where not stim)
    Tq = max(1, T // 4)
    context = np.zeros((B, T), dtype=bool)
    context[:, :Tq] = fix[:, :Tq] & (~stim[:, :Tq])
    return dict(context=context, stim=stim, memory=memory, response=response)

# ---------- PCA helpers ----------
def pca_2d(X):
    """X: (N,H) -> returns Y(N,2), basis W(H,2), mean(H,)"""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T
    return Xc @ W, W, X.mean(0)

# ---------- plotting ----------
def plot_panels(axs, basis, center, traj_H, traj_Z, fp_H, fp_Z, which_masks, title_prefix):
    # project
    Y2d = (traj_H - center) @ basis
    FP2d = (fp_H - center) @ basis
    # common plane
    mnx, mxx = np.percentile(Y2d[:, 0], [1, 99])
    mny, mxy = np.percentile(Y2d[:, 1], [1, 99])
    XX, YY = np.meshgrid(np.linspace(mnx, mxx, 2), np.linspace(mny, mxy, 2))
    ZZ = np.zeros_like(XX)

    names = ["context", "stim", "memory", "response"]
    for ax, name in zip(axs, names):
        sel = which_masks[name]
        # trajectory segment
        ax.plot(Y2d[sel, 0], Y2d[sel, 1], traj_Z[sel], lw=2)
        # all fixed points (use small size)
        ax.scatter(FP2d[:, 0], FP2d[:, 1], fp_Z, c="k", s=10, alpha=0.9)
        # output-null plane
        ax.plot_surface(XX, YY, ZZ, alpha=0.12, linewidth=0)
        ax.set_xlabel("Memory period state PC1")
        ax.set_ylabel("Memory period state PC2")
        ax.set_zlabel("Output (cosθ / logit)")
        ax.set_title(f"{title_prefix}: {name.capitalize()}")

def main():
    args = _parse_with_config(build_parser)
    device = torch.device(args.device)
    run_dir = Path(args.run)
    outdir = Path(args.outdir or (run_dir / "eval" / "figs"))
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load model + a validation batch
    ck = torch.load(run_dir / "ckpt.pt", map_location=device)
    model = rebuild_model_from_ckpt(ck, task=args.task).to(device).eval()

    env = getattr(mct, args.task)()
    ds = Dataset(env, batch_size=args.batch, seq_len=args.seq_len, batch_first=True)
    X_np, Y_np = ds()  # numpy
    X = torch.from_numpy(X_np).float().to(device)

    # --- simulate one batch to get hidden states and logits (manual leaky+softplus for consistency)
    B, T, I = X.shape
    H = model.cfg.hidden_size if hasattr(model, "cfg") else model.W_hh.shape[0]
    alpha = float(getattr(getattr(model, "cfg", object()), "alpha", 1.0))
    with torch.no_grad():
        h = torch.zeros(B, H, device=device)
        H_all = []
        for t in range(T):
            h = step_leaky_softplus(model, h, X[:, t, :], alpha=alpha)
            H_all.append(h)
        H_all = torch.stack(H_all, dim=1).cpu().numpy()  # (B,T,H)

    masks = epoch_masks_from_inputs(X_np, fix_chan=args.fixation_channel, thresh=args.mask_threshold)

    # ---------- loop contexts
    for ctx in args.contexts:
        # 1) fixed points for this context
        fp_H = load_fixed_points(run_dir, ctx)  # (Nfp,H)

        # z for fixed points (cosθ or a selected logit)
        with torch.no_grad():
            fp_t = torch.from_numpy(fp_H).to(device).float()
            logits_fp = try_readout(model, fp_t)
            if args.z_mode == "ring" and logits_fp is not None and logits_fp.shape[-1] >= 2:
                z_fp = logits_fp[:, args.cos_index].detach().cpu().numpy()
            elif logits_fp is not None:
                z_fp = logits_fp[:, args.z_index].detach().cpu().numpy()
            else:
                z_fp = np.zeros(fp_H.shape[0])  # fallback: plot on plane

        # 2) choose one representative trial
        b = max(0, min(args.trial_index, B - 1))
        H_trial = H_all[b]   # (T,H)

        # memory PCA basis from **all trials during memory**
        mem_mask = masks["memory"].reshape(B*T)[..., None].repeat(H, axis=1).astype(bool)
        H_mem = H_all.reshape(B*T, H)[mem_mask[:, 0]]  # (Nm,H)
        if H_mem.shape[0] < 4:  # very degenerate; fall back to whole sequence
            H_mem = H_all.reshape(B*T, H)
        Y2d_mem, W, center = pca_2d(H_mem)

        # 3) compute trajectory z (cosθ or chosen logit) for the selected trial
        with torch.no_grad():
            h_t = torch.from_numpy(H_trial).to(device).float()  # (T,H)
            logits_tr = try_readout(model, h_t)
            if args.z_mode == "ring" and logits_tr is not None and logits_tr.shape[-1] >= 2:
                z_tr = logits_tr[:, args.cos_index].detach().cpu().numpy()
            elif logits_tr is not None:
                z_tr = logits_tr[:, args.z_index].detach().cpu().numpy()
            else:
                z_tr = np.zeros(H_trial.shape[0])

        # 4) make panel masks for that trial
        which_masks = {k: masks[k][b] for k in ["context", "stim", "memory", "response"]}

        # 5) plot
        fig = plt.figure(figsize=(14, 3.2))
        axs = [fig.add_subplot(1, 4, i+1, projection="3d") for i in range(4)]
        plot_panels(axs, W, center, H_trial, z_tr, fp_H, z_fp, which_masks,
                    title_prefix=f"{ctx.capitalize()} dynamics")

        zlabel = "Output cosθ" if args.z_mode == "ring" else f"Logit[{args.z_index}]"
        for ax in axs:
            ax.set_zlabel(zlabel)

        save_path = outdir / f"context={ctx}_ring_fixedpoints.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
