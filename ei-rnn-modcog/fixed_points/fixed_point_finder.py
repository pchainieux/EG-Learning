#python -m fixed_points.fixed_point_finder --config configs/fixed-points/fixed_points.yaml

import os, json, numpy as np, torch
from src.data import mod_cog_tasks as mct
from neurogym import Dataset
from src.models.ei_rnn import EIRNN, EIConfig

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

    ap.add_argument("--run",  type=str, default=cfg.get("run", None),
                    help="Run dir (used if --ckpt not given). Outputs go under <run>/eval/")
    ap.add_argument("--ckpt", type=str, default=cfg.get("ckpt", None),
                    help="Path to a checkpoint file (.pt). If --run is omitted, outputs go next to this file.")

    ap.add_argument("--contexts", nargs="+", default=cfg.get("contexts", ["memory", "anti"]))
    ap.add_argument("--n_init",   type=int,  default=cfg.get("n_init", 256))
    ap.add_argument("--tol",      type=float,default=cfg.get("tol", 1e-6))
    ap.add_argument("--max_iter", type=int,  default=cfg.get("max_iter", 500))
    ap.add_argument("--seq_len",  type=int,  default=cfg.get("seq_len", 350))
    ap.add_argument("--batch",    type=int,  default=cfg.get("batch", 256))
    ap.add_argument("--task",     type=str,  default=cfg.get("task", "dm1"))
    ap.add_argument("--device",   type=str,  default=cfg.get("device", "cpu"))
    return ap

def rebuild_model(ck, task="dm1"):
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
            import numpy as np
            sv = np.asarray(sv)
            exc_frac = float((sv > 0).mean())

    env = getattr(mct, task)()
    input_dim = env.observation_space.shape[-1]
    output_dim = env.action_space.n

    model = EIRNN(input_size=input_dim, output_size=output_dim,
                  cfg=EIConfig(hidden_size=H, exc_frac=exc_frac))
    model.load_state_dict(sd, strict=False) 
    return model


def _act(pre, activation: str = "softplus"):
    if activation == "relu":
        return torch.relu(pre)
    elif activation == "tanh":
        return torch.tanh(pre)
    elif activation == "softplus":
        return torch.nn.functional.softplus(pre)
    else:
        # fallback
        return torch.nn.functional.softplus(pre)

def _dact(pre, activation: str = "softplus"):
    if activation == "relu":
        return (pre > 0).to(pre.dtype)
    elif activation == "tanh":
        return 1.0 - torch.tanh(pre)**2
    elif activation == "softplus":
        # d/dx softplus(x) = sigmoid(x)
        return torch.sigmoid(pre)
    else:
        return torch.sigmoid(pre)

def F_step(model, h, x_t, activation: str = "softplus"):
    pre = x_t @ model.W_xh.T + h @ model.W_hh.T + model.b_h
    return _act(pre, activation)

def jacobian_dFdh(model, h, x_t, activation: str = "softplus"):
    with torch.no_grad():
        pre = x_t @ model.W_xh.T + h @ model.W_hh.T + model.b_h  # [1, H]
        m = _dact(pre, activation).squeeze(0)                    # [H]
    # dF/dh = diag(phi'(pre)) @ W_hh  (consistent with h @ W_hh.T)
    J = torch.diag(m) @ model.W_hh
    return J

def solve_fixed_point(model, h0, x_t, tol=1e-6, max_iter=500, lr=0.2, activation: str = "softplus"):
    h = h0.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([h], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        resid = F_step(model, h, x_t, activation=activation) - h
        loss = (resid * resid).sum()
        if float(loss.item()) < tol: break
        loss.backward(); opt.step()
    with torch.no_grad():
        resid = F_step(model, h, x_t, activation=activation) - h
        return h.detach(), float((resid*resid).sum().item())

def classify(J, eps=1e-3):
    vals = torch.linalg.eigvals(J).detach().cpu().numpy()
    mags = np.abs(vals); rho = float(mags.max()) if mags.size else 0.0
    n_lt = (mags < 1 - eps).sum(); n_gt = (mags > 1 + eps).sum()
    has_c_lt = np.any((mags < 1 - eps) & (np.abs(np.imag(vals)) > 1e-6))
    if rho < 1 - eps: label = "stable"
    elif n_gt > 0 and n_lt > 0: label = "saddle"
    elif has_c_lt: label = "rotational"
    else: label = "unstable"
    return vals, rho, label

def main():
    args = _parse_with_config(build_parser)
    device = torch.device(args.device)

    if args.run is None and args.ckpt is None:
        raise SystemExit("Provide either --run <dir> (expects <dir>/ckpt.pt) or --ckpt <file.pt>. "
                         "In YAML, set 'run:' or 'ckpt:'.")

    ckpt_path = args.ckpt or os.path.join(args.run, "ckpt.pt")
    ck = torch.load(ckpt_path, map_location=device)
    model = rebuild_model(ck, task=args.task).to(device).eval()

    run_dir = args.run or os.path.dirname(os.path.abspath(ckpt_path))

    env = getattr(mct, args.task)()
    ds = Dataset(env, batch_size=args.batch, seq_len=args.seq_len, batch_first=True)
    X, Y = ds()
    X = torch.from_numpy(X).float().to(device)
    B, T, I = X.shape
    H = model.cfg.hidden_size
    h = torch.zeros(B, H, device=device, dtype=torch.float64)
    model.W_hh = model.W_hh.to(torch.float64)
    model.W_xh = model.W_xh.to(torch.float64)
    model.b_h   = model.b_h.to(torch.float64)
    X = X.to(torch.float64)
    hs = []
    with torch.no_grad():
        for t in range(T):
            pre = X[:, t, :] @ model.W_xh.T + h @ model.W_hh.T + model.b_h
            h   = _act(pre, activation="softplus")
            hs.append(h.clone())
    h_last = hs[-1]
    n = min(args.n_init, B)
    # Split seeds per context using fixation channel (ch=0) like training’s dec_mask
    # memory  := fixation ON at last step (X[...,0] >= 0.5)
    # anti    := fixation OFF at last step (X[...,0] <  0.5)
    fix_last = (X[:, -1, 0] >= 0.5)
    idx_mem  = torch.nonzero(fix_last, as_tuple=False).squeeze(-1)[:n]
    idx_anti = torch.nonzero(~fix_last, as_tuple=False).squeeze(-1)[:n]

    out_dir_base = os.path.join(args.run, "eval", "fixed_points")
    os.makedirs(out_dir_base, exist_ok=True)
    for ctx in args.contexts:
        ctx_dir = os.path.join(out_dir_base, f"context={ctx}")
        os.makedirs(ctx_dir, exist_ok=True)
        meta = []
        if ctx == "memory":
            idxs = idx_mem
        else:
            idxs = idx_anti
        for i_idx, i in enumerate(idxs.tolist()):
            h0_i = h_last[i:i+1]
            xT_i = X[i:i+1, -1, :]
            h_star, res = solve_fixed_point(model, h0_i, xT_i, tol=args.tol, max_iter=args.max_iter, activation="softplus")
            J = jacobian_dFdh(model, h_star, xT_i, activation="softplus")
            eigvals, rho, label = classify(J)
            np.savez(os.path.join(ctx_dir, f"fp_{i:03d}.npz"),
                     h_star=h_star.squeeze(0).cpu().numpy(),
                     residual=res, eigvals=eigvals, rho=rho, label=label)
            meta.append({"index": i, "residual": res, "rho": rho, "label": label})
        with open(os.path.join(ctx_dir, "index.json"), "w") as f:
            json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
