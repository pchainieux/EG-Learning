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
    # Reuse the implementation from collect_grads.py
    from ei_gradient.scripts.collect_grads import build_model_and_load as _build
    model, _extra = _build(checkpoint_path, device)
    return model

def get_val_loader(cfg: Dict[str, Any]):
    # Reuse the implementation from collect_grads.py
    from ei_gradient.scripts.collect_grads import get_val_loader as _get
    return _get(cfg)
# ============================


def make_forward_score(model, batch_template: Dict[str, torch.Tensor], cfg: Dict[str, Any]):
    """
    Returns a callable x -> scalar to maximize.
    By default: average target-class logit over 'decision' time steps.
    """
    import torch.nn.functional as F

    target = cfg.get("target", {}) or {}
    target_id = int(target.get("id", 0)) + 1   # +1 because index 0 is fixation
    mask_thresh = float(cfg.get("mask_threshold", 0.5))

    def _decision_mask(X: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        return (X[..., 0] < thresh)

    def _forward_logits(xBTD: torch.Tensor) -> torch.Tensor:
        # time-unroll directly using model internals (same as forward_collect, but no caches)
        B, T, _ = xBTD.shape
        H = model.cfg.hidden_size
        h = xBTD.new_zeros(B, H)
        logits = xBTD.new_zeros(B, T, model.W_out.out_features)
        alpha = model._alpha

        for t in range(T):
            pre = xBTD[:, t, :] @ model.W_xh.T + h @ model.W_hh.T + model.b_h
            phi = F.softplus(pre) if model._nl_kind == "softplus" else torch.tanh(pre)
            h = (1 - alpha) * h + alpha * phi
            h_ro = h * model.e_mask if model._readout_mode == "e_only" else h
            logits[:, t, :] = model.W_out(h_ro)
        return logits

    def forward_score(x: torch.Tensor) -> torch.Tensor:
        # x is [T,B,D]; convert to [B,T,D] and build a batch copy
        xBTD = x.transpose(0, 1).contiguous()
        logits = _forward_logits(xBTD)
        dec_mask = _decision_mask(xBTD, thresh=mask_thresh)      # [B,T]
        if not dec_mask.any():
            return logits.new_tensor(0.0, requires_grad=True)
        # mean logit of the chosen class over decision steps
        score = logits[..., target_id][dec_mask].mean()
        return score

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
