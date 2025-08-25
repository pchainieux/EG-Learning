#python -m fixed_points.fixed_point_finder --config configs/fixed-points/fixed_points.yaml

# fixed_points/fixed_point_finder.py
import os, json, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

from src.data import mod_cog_tasks as mct
from neurogym import Dataset
from src.models.ei_rnn import EIRNN, EIConfig


# ----------------- config parsing -----------------
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

    ap.add_argument("--run",  type=str, default=cfg.get("run", None),
                    help="Run dir (used if --ckpt not given). Outputs go under <run>/eval/")
    ap.add_argument("--ckpt", type=str, default=cfg.get("ckpt", None),
                    help="Path to a checkpoint file (.pt). If --run is omitted, outputs go next to this file.")

    ap.add_argument("--contexts", nargs="+", default=cfg.get("contexts", ["memory", "anti"]))
    ap.add_argument("--n_init",   type=int,  default=cfg.get("n_init", 256))
    ap.add_argument("--tol",      type=float,default=cfg.get("tol", 1e-6), help="Root residual tolerance ||F(h)-h|| < tol")
    ap.add_argument("--max_iter", type=int,  default=cfg.get("max_iter", 500))
    ap.add_argument("--seq_len",  type=int,  default=cfg.get("seq_len", 350))
    ap.add_argument("--batch",    type=int,  default=cfg.get("batch", 256))
    ap.add_argument("--task",     type=str,  default=cfg.get("task", "dm1"))
    ap.add_argument("--device",   type=str,  default=cfg.get("device", "cpu"))

    # Context input wiring (optional; sensible defaults if not provided)
    ap.add_argument("--fixation_channel", type=int, default=cfg.get("fixation_channel", 0))
    ap.add_argument("--rule_mem_channel", type=int, default=cfg.get("rule_mem_channel", -1),
                    help="Set to >=0 to assert memory rule bit ON for context=memory.")
    ap.add_argument("--rule_anti_channel", type=int, default=cfg.get("rule_anti_channel", -1),
                    help="Set to >=0 to assert anti rule bit ON for context=anti.")
    ap.add_argument("--stim_channels", type=str, default=cfg.get("stim_channels", ""),
                    help="Comma-separated indices of stimulus channels to zero for memory context, e.g. '3,4'.")

    # Leak (alpha). If not provided, read from checkpoint config; else default 1.0.
    ap.add_argument("--alpha", type=float, default=cfg.get("alpha", None),
                    help="Optional leak coefficient. If omitted, read from checkpoint config, else 1.0.")

    # De-duplication radius
    ap.add_argument("--dedup_eps", type=float, default=cfg.get("dedup_eps", 1e-2),
                    help="L2 radius in hidden space to treat two fixed points as duplicates.")
    return ap


# ----------------- model rebuild -----------------
def rebuild_model(ck, task="dm1"):
    """Rebuild EIRNN with shapes inferred from checkpoint."""
    sd = ck.get("model") or ck.get("model_state") or ck.get("state_dict") or ck
    if not isinstance(sd, dict):
        raise ValueError("Could not find a model state_dict in checkpoint.")

    if "W_hh" not in sd:
        raise KeyError("state_dict is missing 'W_hh'.")
    H = sd["W_hh"].shape[0]

    exc_frac = 0.8
    if "sign_vec" in sd:
        sv = sd["sign_vec"]
        if isinstance(sv, torch.Tensor):
            exc_frac = float((sv > 0).float().mean().item())
        else:
            sv = np.asarray(sv)
            exc_frac = float((sv > 0).mean())

    env = getattr(mct, task)()
    input_dim = env.observation_space.shape[-1]
    output_dim = env.action_space.n

    model = EIRNN(input_size=input_dim, output_size=output_dim,
                  cfg=EIConfig(hidden_size=H, exc_frac=exc_frac))
    model.load_state_dict(sd, strict=False)
    return model


# ----------------- dynamics (match training) -----------------
def step_leaky_softplus(model, h, x_t, alpha=1.0):
    """h_{t+1} = (1-α)h + α softplus(W_xh x + W_hh h + b)"""
    pre = x_t @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    return (1 - alpha) * h + alpha * F.softplus(pre)


@torch.no_grad()
def fixed_point_residual(model, h, x_bar, alpha=1.0):
    return step_leaky_softplus(model, h, x_bar, alpha=alpha) - h


def jacobian_at(model, h_star, x_bar, alpha=1.0):
    """J = (1-α)I + α diag(σ(pre)) @ W_hh, with pre = W_xh x_bar + W_hh h* + b."""
    # Work in CPU float64 for numerical stability
    with torch.no_grad():
        Whh = model.W_hh.detach().to("cpu", dtype=torch.float64)
        Wxh = model.W_xh.detach().to("cpu", dtype=torch.float64)
        bh  = model.b_h.detach().to("cpu", dtype=torch.float64)
        h64 = h_star.detach().to("cpu", dtype=torch.float64)
        x64 = x_bar.detach().to("cpu", dtype=torch.float64)
        pre = x64 @ Wxh.T + h64 @ Whh.T + bh   # [1,H]
        sig = torch.sigmoid(pre).squeeze(0)    # [H]
        I   = torch.eye(Whh.shape[0], dtype=torch.float64)
        J   = (1.0 - alpha) * I + alpha * torch.diag(sig) @ Whh
    return J.numpy()  # float64, ready for numpy eig


# ----------------- solver -----------------
def solve_fixed_point(model, h0, x_bar, tol=1e-6, max_iter=500, lr=0.2, alpha=1.0):
    """Minimize ||F(h, x_bar) - h||_2^2 by gradient steps in h."""
    h = h0.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([h], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        resid = fixed_point_residual(model, h, x_bar, alpha=alpha)
        loss = (resid * resid).sum()
        # stop on ROOT residual (better scale)
        if float(resid.norm().item()) < tol:
            break
        loss.backward()
        opt.step()
    with torch.no_grad():
        resid = fixed_point_residual(model, h, x_bar, alpha=alpha)
        return h.detach(), float(resid.norm().item())


# ----------------- classification -----------------
def classify(J, eps=1e-3):
    # eigs in complex128 from float64 J
    vals = np.linalg.eigvals(J)
    mags = np.abs(vals)
    rho = float(mags.max()) if mags.size else 0.0
    n_lt = int((mags < 1 - eps).sum())
    n_gt = int((mags > 1 + eps).sum())
    has_c_lt = bool(np.any((mags < 1 - eps) & (np.abs(np.imag(vals)) > 1e-9)))
    if rho < 1 - eps:
        label = "stable"
    elif n_gt > 0 and n_lt > 0:
        label = "saddle"
    elif has_c_lt:
        label = "rotational"
    else:
        label = "unstable"
    return vals, rho, label


# ----------------- context inputs -----------------
def parse_channels_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [int(tok) for tok in s.split(",") if tok.strip() != ""]


def make_context_input(input_dim: int,
                       context: str,
                       fixation_ch: int = 0,
                       rule_mem_ch: int = -1,
                       rule_anti_ch: int = -1,
                       stim_chs: list[int] | None = None,
                       device: torch.device | str = "cpu"):
    """Construct a single-step constant input vector for a given context."""
    x = torch.zeros(1, input_dim, device=device, dtype=torch.float64)
    # Memory/anti contexts typically keep fixation ON during the memory epoch
    if fixation_ch >= 0:
        x[0, fixation_ch] = 1.0
    # Rule bits: assert one-hot if channels provided
    if context.lower() == "memory" and rule_mem_ch >= 0:
        x[0, rule_mem_ch] = 1.0
    if context.lower() == "anti" and rule_anti_ch >= 0:
        x[0, rule_anti_ch] = 1.0
    # Zero stimulus channels if specified (esp. for memory context)
    if stim_chs:
        x[0, stim_chs] = 0.0
    return x


# ----------------- main -----------------
def main():
    args = _parse_with_config(build_parser)
    device = torch.device(args.device)

    # Locate checkpoint
    if args.run is None and args.ckpt is None:
        raise SystemExit("Provide either --run <dir> (expects <dir>/ckpt.pt) or --ckpt <file.pt>.")
    ckpt_path = args.ckpt or os.path.join(args.run, "ckpt.pt")
    ck = torch.load(ckpt_path, map_location=device)

    # Rebuild model & infer alpha (leak)
    model = rebuild_model(ck, task=args.task).to(device).eval()
    alpha = args.alpha
    if alpha is None:
        alpha = float(ck.get("config", {}).get("alpha", 1.0))  # default to no leak if not saved
    alpha = float(alpha)

    # Where to save
    run_dir = args.run or os.path.dirname(os.path.abspath(ckpt_path))
    out_dir_base = os.path.join(run_dir, "eval", "fixed_points")
    os.makedirs(out_dir_base, exist_ok=True)

    # Collect realistic initial states from short simulation (with SAME operator)
    env = getattr(mct, args.task)()
    ds = Dataset(env, batch_size=args.batch, seq_len=args.seq_len, batch_first=True)
    X_np, _ = ds()
    X = torch.from_numpy(X_np).to(device=device, dtype=torch.float64)
    B, T, I = X.shape
    H = model.cfg.hidden_size

    # promote weights to float64 for consistency
    model.W_hh = model.W_hh.to(torch.float64)
    model.W_xh = model.W_xh.to(torch.float64)
    model.b_h  = model.b_h.to(torch.float64)

    with torch.no_grad():
        h = torch.zeros(B, H, device=device, dtype=torch.float64)
        for t in range(T):
            h = step_leaky_softplus(model, h, X[:, t, :], alpha=alpha)
        h_last = h.detach().clone()          # (B,H)

    n = min(args.n_init, B)
    # Use a deterministic slice of seeds; contexts are handled via x_bar
    idxs = torch.arange(n, device=device)

    # Channels for context inputs
    stim_chs = parse_channels_list(args.stim_channels)

    # Solve per context
    for ctx in args.contexts:
        ctx_dir = os.path.join(out_dir_base, f"context={ctx}")
        os.makedirs(ctx_dir, exist_ok=True)
        meta = []

        # Build a constant input for this context
        x_bar = make_context_input(
            input_dim=I,
            context=ctx,
            fixation_ch=int(args.fixation_channel),
            rule_mem_ch=int(args.rule_mem_channel),
            rule_anti_ch=int(args.rule_anti_channel),
            stim_chs=stim_chs,
            device=device,
        )  # shape [1, I], float64

        # Greedy de-duplication buffer
        kept = []
        kept_idx = []

        for j, i in enumerate(idxs.tolist()):
            h0_i = h_last[i:i+1]  # [1,H]
            h_star, res = solve_fixed_point(model, h0_i, x_bar, tol=args.tol, max_iter=args.max_iter, lr=0.2, alpha=alpha)

            # De-dup: keep if it's not within eps of any previous one
            keep = True
            h_np = h_star.squeeze(0).detach().cpu().numpy()
            for k_np in kept:
                if np.linalg.norm(h_np - k_np) <= args.dedup_eps:
                    keep = False
                    break
            if not keep:
                meta.append({"index": int(i), "residual": res, "rho": None, "label": "duplicate", "kept": False})
                continue

            # Jacobian and spectrum (CPU float64)
            J = jacobian_at(model, h_star, x_bar, alpha=alpha)   # numpy float64
            eigvals, rho, label = classify(J)

            # Save point
            fp_path = os.path.join(ctx_dir, f"fp_{int(i):03d}.npz")
            np.savez(fp_path,
                     h_star=h_np,
                     residual=res,
                     eigvals=eigvals,
                     rho=rho,
                     label=label)
            meta.append({"index": int(i), "residual": res, "rho": rho, "label": label, "kept": True})
            kept.append(h_np); kept_idx.append(int(i))

        with open(os.path.join(ctx_dir, "index.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[fp] context={ctx} kept {len(kept)}/{len(idxs)} (dedup_eps={args.dedup_eps}) -> {ctx_dir}")

if __name__ == "__main__":
    main()
