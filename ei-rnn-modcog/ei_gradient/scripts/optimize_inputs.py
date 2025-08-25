# experiments/credit_assignment/scripts/optimize_inputs.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import numpy as np

from ei_gradient.src.io import load_yaml, ensure_outdir, save_tensors
from ei_gradient.src.attribution import saliency, optimize_inputs

# ============================
# TODO: wire these to your repo
# ============================
def build_model_and_load(checkpoint_path: str, device: str):
    """
    Replace with your model construction + checkpoint restore.
    Must return a model whose scoring function you can access for a chosen class/time.
    """
    raise NotImplementedError("Connect build_model_and_load() to your codebase")


def get_val_loader(cfg: Dict[str, Any]):
    """
    Replace with your dataset/dataloader for the validation split.
    """
    raise NotImplementedError("Connect get_val_loader() to your codebase")
# ============================


def make_forward_score(model, batch_template: Dict[str, torch.Tensor], cfg: Dict[str, Any]):
    """
    Returns a callable x -> scalar to maximize. The callable should:
      - Inject x (optimized inputs) into a batch built from `batch_template`
      - Run the model
      - Return a scalar score (e.g. negative loss for target class at decision time)
    """
    target = cfg["target"]  # e.g., {"type":"class","id":0,"time":"decision"}
    def forward_score(x: torch.Tensor) -> torch.Tensor:
        batch = dict(batch_template)
        batch["x_override"] = x
        # Example:
        # y_hat, cache = model.forward_with_cache(batch, return_states=False)
        # score = - model.loss_for_class(y_hat, target_id=target["id"], t=target_time, batch=batch)
        # return score
        raise NotImplementedError("Define how to compute the scalar score from model outputs")
    return forward_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    exp_id = cfg.get("exp_id", "exp_ei_credit")
    out_dir = ensure_outdir(cfg.get("out_dir", "outputs"), exp_id)
    io_dir = out_dir / "input_opt"
    io_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_and_load(cfg["checkpoint"], device)
    # Get a baseline batch (single or small batch)
    val_loader = get_val_loader(cfg)
    batch = next(iter(val_loader))
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Build an initial input tensor x0 from batch (modify as appropriate for your pipeline)
    # Example expectations: batch contains "x" of shape [T,B,D]; take B=1 slice
    if "x" not in batch:
        raise RuntimeError("Expected 'x' in batch; adjust to your data interface")
    x0 = batch["x"][:, :1, :].detach()  # [T,1,D] first item
    freeze_mask = torch.ones_like(x0)
    # Optional: freeze fixation channel if you can identify it; else leave all optimizable.

    fwd_score = make_forward_score(model, batch, cfg)

    # Saliency (optional)
    gx = saliency(fwd_score, x0, retain_graph=False)

    # Optimisation
    res = optimize_inputs(
        fwd_score,
        x0,
        steps=int(cfg.get("steps", 200)),
        step_size=float(cfg.get("step_size", 0.05)),
        epsilon=float(cfg.get("epsilon", 0.05)),
        bounds=tuple(cfg.get("bounds", (0.0, 1.0))),
        reg_l2=float(cfg.get("reg", {}).get("l2", 1e-4)),
        reg_tv=float(cfg.get("reg", {}).get("tv", 1e-4)),
        freeze_mask=freeze_mask
    )

    save_tensors(
        {
            "x0": x0.cpu(),
            "saliency": gx.cpu(),
            "x_opt": res["x_opt"].cpu(),
            "delta": res["delta"].cpu(),
            "score_before": res["score_before"].cpu() if torch.is_tensor(res["score_before"]) else torch.tensor([res["score_before"]]),
            "score_after": res["score_after"].cpu() if torch.is_tensor(res["score_after"]) else torch.tensor([res["score_after"]]),
        },
        io_dir / "input_opt.pt"
    )
    print(f"[optimize_inputs] Saved results to {io_dir / 'input_opt.pt'}")

if __name__ == "__main__":
    main()
