#python -m fixed_points.analyze_motifs --config configs/fixed-points/analyze.yaml

import os, json, glob, numpy as np

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
    ap.add_argument("--contexts", nargs="+", default=cfg.get("contexts", ["memory", "anti"]))
    ap.add_argument("--k_slow", type=int, default=cfg.get("k_slow", 4))
    ap.add_argument("--m_overlap", type=int, default=cfg.get("m_overlap", 3))
    return ap

def load_fp_dir(fp_dir):
    import os, glob, numpy as np
    fps = []
    for f in sorted(glob.glob(os.path.join(fp_dir, "fp_*.npz"))):
        data = np.load(f, allow_pickle=True)
        fps.append({
            "path": os.path.basename(f),      # store filename only
            "residual": float(data["residual"]),
            "rho": float(data["rho"]),
            "label": str(data["label"]),
            # NOTE: no h_star / eigvals here (keeps metrics.json small & JSON-safe)
        })
    return fps

def counts_by_label(fps):
    labels = [fp["label"] for fp in fps]
    keys = ["stable", "saddle", "rotational", "unstable"]
    return {k: int(sum(1 for l in labels if l == k)) for k in keys}

def stability_margins(fps):
    margins = [1.0 - fp["rho"] for fp in fps if np.isfinite(fp["rho"])]
    return {"mean": float(np.mean(margins)) if margins else float("nan"),
            "std": float(np.std(margins)) if margins else float("nan"),
            "n": len(margins)}

def principal_angle_overlap(A, B, m=3):
    UA, _ = np.linalg.qr(A); UB, _ = np.linalg.qr(B)
    s = np.linalg.svd(UA.T @ UB, compute_uv=False)
    if s.size == 0: return float("nan")
    return float(np.mean(s[:min(m, s.size)]))

def pca_basis(H, hs, k):
    X = hs - hs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k].T

def main():
    args = _parse_with_config(build_parser)
    per_ctx = {}
    for ctx in args.contexts:
        fp_dir = os.path.join(args.run, "eval", "fixed_points", f"context={ctx}")
        fps = load_fp_dir(fp_dir) if os.path.isdir(fp_dir) else []
        per_ctx[ctx] = {"n": len(fps), "counts": counts_by_label(fps), "stability": stability_margins(fps), "fps": fps}

    reuse = {}
    ctxs = args.contexts
    H = None
    hs = {ctx: [] for ctx in ctxs}
    for ctx in ctxs:
        fp_dir = os.path.join(args.run, "eval", "fixed_points", f"context={ctx}")
        for fp in per_ctx[ctx]["fps"]:
            # Backward compat: if h_star was already attached in-memory, use it; else load from disk
            if "h_star" in fp:
                h = np.asarray(fp["h_star"]).reshape(-1)
            else:
                npz = np.load(os.path.join(fp_dir, fp["path"]), allow_pickle=True)
                h = npz["h_star"].reshape(-1)
            H = h.shape[0]
            hs[ctx].append(h)

    if H is not None and all(len(hs[c]) >= args.k_slow for c in ctxs):
        V0 = pca_basis(H, np.stack(hs[ctxs[0]], axis=0), args.k_slow)
        V1 = pca_basis(H, np.stack(hs[ctxs[1]], axis=0), args.k_slow)
        reuse["pair"] = {"contexts": ctxs[:2],
                         "overlap": principal_angle_overlap(V0, V1, m=min(args.m_overlap, args.k_slow)),
                         "k": args.k_slow}
    else:
        reuse["pair"] = {"contexts": ctxs[:2] if len(ctxs)>=2 else ctxs, "overlap": float("nan"), "k": args.k_slow}

    out_path = os.path.join(args.run, "eval", "metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_context": per_ctx, "reuse": reuse}, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
