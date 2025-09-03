from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Optional

import torch

from src.models.ei_rnn import EIRNN, EIConfig


def _infer_io_dims_from_state(state: Mapping[str, torch.Tensor]) -> Tuple[Optional[int], Optional[int]]:
    in_dim = None
    out_dim = None

    if "W_xh" in state and isinstance(state["W_xh"], torch.Tensor):
        w = state["W_xh"]
        if w.ndim == 2:
            in_dim = int(w.shape[1])

    if "W_out.bias" in state and isinstance(state["W_out.bias"], torch.Tensor):
        b = state["W_out.bias"]
        if b.ndim == 1:
            out_dim = int(b.shape[0])
    elif "W_out.weight" in state and isinstance(state["W_out.weight"], torch.Tensor):
        w = state["W_out.weight"]
        if w.ndim == 2:
            out_dim = int(w.shape[0])

    return in_dim, out_dim


def rebuild_model_from_ckpt(ckpt_path: str | Path, device: torch.device) -> Tuple[EIRNN, Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg: Mapping[str, Any] = ckpt.get("config", {})
    state = ckpt.get("model") or ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise RuntimeError("Checkpoint missing model weights (expected 'model' or 'model_state' or 'state_dict').")

    in_dim, out_dim = _infer_io_dims_from_state(state)
    if in_dim is None or out_dim is None:
        raise RuntimeError(
            f"Could not infer input/output dims from checkpoint. "
            f"Got in_dim={in_dim}, out_dim={out_dim}. "
            f"Keys present: {list(state.keys())[:6]}..."
        )

    mcfg = cfg.get("model", {})
    H = int(mcfg.get("hidden_size", 256))
    exc_frac = float(mcfg.get("exc_frac", 0.8))
    sr = float(mcfg.get("spectral_radius", 1.2))
    in_scale = float(mcfg.get("input_scale", 1.0))
    leak = float(mcfg.get("leak", 1))
    nonlin = mcfg.get("nonlinearity", "softplus")
    readout = mcfg.get("readout", "all") 
    beta = float(mcfg.get("softplus_beta", 8.0))
    th = float(mcfg.get("softplus_threshold", 20.0))

    model = EIRNN(
        input_size=in_dim,
        output_size=out_dim,
        cfg=EIConfig(
            hidden_size=H,
            exc_frac=exc_frac,
            spectral_radius=sr,
            input_scale=in_scale,
            leak=leak,
            nonlinearity=nonlin,
            readout=readout,
            softplus_beta=beta,
            softplus_threshold=th,
        ),
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[model_io] load_state_dict: missing={missing}, unexpected={unexpected}")

    model.eval()
    return model, dict(cfg)
