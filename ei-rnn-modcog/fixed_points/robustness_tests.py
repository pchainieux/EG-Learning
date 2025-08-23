# python -m fixed_points.robustness_tests --config configs/fixed-points/robustness.yaml

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
    ap.add_argument("--run", required=not bool(cfg), default=cfg.get("run", None))
    ap.add_argument("--task", type=str, default=cfg.get("task", "dm1"))
    ap.add_argument("--alphaI", nargs="+", type=float, default=cfg.get("alphaI", [0.5,0.75,1.0,1.25,1.5]))
    ap.add_argument("--noise_std", nargs="+", type=float, default=cfg.get("noise_std", [0.0,0.1,0.2]))
    ap.add_argument("--seq_len", type=int, default=cfg.get("seq_len", 350))
    ap.add_argument("--batch", type=int, default=cfg.get("batch", 128))
    ap.add_argument("--device", type=str, default=cfg.get("device", "cpu"))
    return ap

def rebuild_model(ck):
    cfg = ck["config"]
    H = int(cfg.get("hidden", cfg.get("hidden_size", 256)))
    pI = float(cfg.get("pI", 0.35))
    spectral_radius = float(cfg.get("spectral_radius", 1.2))
    input_scale = float(cfg.get("input_scale", 1.0))
    exc_frac = 1.0 - pI
    task = cfg.get("task", "dm1")
    env = getattr(mct, task)()
    input_dim = env.observation_space.shape[-1]
    output_dim = env.action_space.n
    model = EIRNN(input_size=input_dim, output_size=output_dim,
                  cfg=EIConfig(hidden_size=H, exc_frac=exc_frac,
                               spectral_radius=spectral_radius, input_scale=input_scale))
    state = ck.get("model") or ck.get("model_state")
    model.load_state_dict(state, strict=False)
    return model

def main():
    args = _parse_with_config(build_parser)
    device = torch.device(args.device)
    ck = torch.load(os.path.join(args.run, "ckpt.pt"), map_location=device)
    model = rebuild_model(ck).to(device).eval()
    env = getattr(mct, args.task)()
    ds = Dataset(env, batch_size=args.batch, seq_len=args.seq_len, batch_first=True)
    W = model.W_hh
    is_inh = (model.sign_vec < 0).bool().cpu().numpy()

    results = []
    for a in args.alphaI:
        with torch.no_grad():
            W_backup = W.clone()
            scale = torch.ones(W.shape[1], device=W.device)
            scale[torch.tensor(is_inh, device=W.device)] = a
            W.mul_(scale.unsqueeze(0))

        accs = []
        for _ in range(50):
            X, Y = ds()
            X = torch.from_numpy(X).float().to(device)
            Y = torch.from_numpy(Y).long().to(device)
            logits = model(X)
            pred = logits[:, -1, :].argmax(dim=-1)
            accs.append((pred == (Y[:, -1] + 1)).float().mean().item())
        acc_base = float(np.mean(accs))

        X, Y = ds(); X = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            B = X.shape[0]; H = model.cfg.hidden_size
            h = torch.zeros(B, H, device=device)
            for t in range(X.shape[1]):
                pre = X[:, t, :] @ model.W_xh.T + h @ model.W_hh.T + model.b_h
                h   = torch.nn.functional.softplus(pre)  # or your model’s activation            
                hT = h
            rec = []
            for s in args.noise_std:
                noise = torch.randn_like(hT) * s
                h_pert = hT + noise
                pre_n1 = X[:, -1, :] @ model.W_xh.T + h_pert @ model.W_hh.T + model.b_h
                h_next = torch.nn.functional.softplus(pre_n1)                
                dist = torch.norm(h_next - hT, dim=1).mean().item()
                rec.append({"noise_std": s, "mean_dist": dist})

        results.append({"alphaI": a, "acc": acc_base, "noise": {"recovery": rec}})
        with torch.no_grad():
            W.copy_(W_backup)

    out_path = os.path.join(args.run, "eval", "robustness.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
