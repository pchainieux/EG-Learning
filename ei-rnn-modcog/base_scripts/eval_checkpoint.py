from __future__ import annotations
import argparse
from pathlib import Path
import torch
import json

def main():
    p = argparse.ArgumentParser(description="Inspect a saved EI-RNN ModCog checkpoint.")
    p.add_argument("--ckpt", required=True, type=str, help="Path to .pt checkpoint")
    p.add_argument("--keys", action="store_true", help="Print raw state_dict keys")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"\n== Checkpoint: {ckpt_path}")
    epoch = ckpt.get("epoch", None)
    cfg = ckpt.get("config", None)
    if epoch is not None:
        print(f"epoch: {epoch}")
    if cfg is not None:
        tasks = cfg.get("tasks", None) if isinstance(cfg, dict) else None
        outdir = cfg.get("outdir", None) if isinstance(cfg, dict) else None
        print("config summary:", {"tasks": tasks, "outdir": outdir})

    if "model" in ckpt and "core" not in ckpt:
        sd = ckpt["model"]
        print("format: single-head")
        if args.keys:
            print(f"state_dict keys ({len(sd)}):")
            for k in sd.keys():
                print("  ", k)
        else:
            for k in list(sd.keys())[:6]:
                v = sd[k]
                print(f"  {k:30s} shape={tuple(v.shape)}")

    elif "core" in ckpt and "heads" in ckpt:
        core_sd = ckpt["core"]
        heads_sd = ckpt["heads"]
        head_dims = ckpt.get("head_dims", {})
        print("format: multi-head")
        print("head_dims:", head_dims)
        if args.keys:
            print(f"\ncore keys ({len(core_sd)}):")
            for k in core_sd.keys():
                print("  ", k)
            print(f"\nheads keys ({len(heads_sd)}):")
            for k in heads_sd.keys():
                print("  ", k)
        else:
            for k in ["W_xh", "W_hh", "b_h", "W_out.weight", "W_out.bias"]:
                if k in core_sd:
                    v = core_sd[k]
                    print(f"  core.{k:20s} shape={tuple(v.shape)}")
            if len(heads_sd) > 0:
                hk, hv = next(iter(heads_sd.items()))
                print(f"  head example: {hk} shape={tuple(hv.shape)}")

    else:
        print("Unrecognized checkpoint structure. Keys:", list(ckpt.keys()))

if __name__ == "__main__":
    main()
