from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import numpy as np
import random
import sys

from experiments.ei_gradient.src.io_utils import load_yaml, ensure_outdir, save_yaml, save_tensors
from experiments.ei_gradient.src.hooks import collect_hidden_and_grads, ei_indices_from_sign_matrix

def build_model_and_load(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    try:
        import numpy as _np
        if getattr(_np, "_core", None) is not None:
            sys.modules.setdefault("numpy._core", _np._core)
    except Exception:
        pass

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state = None
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict", "state", "weights", "net"):
            v = ckpt.get(k)
            if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
                state = v
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
    else:
        try:
            state = ckpt.state_dict()
        except Exception:
            raise RuntimeError("Unrecognized checkpoint format; cannot obtain state_dict.")

    if state is None:
        raise KeyError(f"Could not find a weights dict in checkpoint. "
                       f"Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    def _strip_prefix(d, prefixes=("module.", "model.", "net.")):
        out = {}
        for k, v in d.items():
            k2 = k
            for p in prefixes:
                if k2.startswith(p):
                    k2 = k2[len(p):]
            out[k2] = v
        return out

    state = _strip_prefix(state)


    def _maybe_alias_output(d):
        if "W_out.weight" in d:
            return d
        remap = dict(d)
        for k in list(d.keys()):
            if k.endswith("out.weight") or "readout.weight" in k or k.endswith("W_out_w"):
                remap["W_out.weight"] = d[k]
            if k.endswith("out.bias") or "readout.bias" in k or k.endswith("W_out_b"):
                remap["W_out.bias"] = d[k]
        return remap

    state = _maybe_alias_output(state)

    if "W_xh" not in state or "W_out.weight" not in state:
        preview = list(state.keys())[:20]
        raise KeyError(f"Missing required keys: need 'W_xh' and 'W_out.weight'. Found (preview): {preview}")

    W_xh = state["W_xh"]
    W_out_w = state["W_out.weight"]
    H, D_in = W_xh.shape
    C_out = W_out_w.shape[0]

    from src.models.ei_rnn import EIRNN, EIConfig
    cfg_full = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = (cfg_full.get("model", {}) or {}) if isinstance(cfg_full, dict) else {}

    ei_cfg = EIConfig(
        hidden_size=H,
        exc_frac=float(model_cfg.get("exc_frac", 0.8)),
        spectral_radius=float(model_cfg.get("spectral_radius", 1.2)),
        input_scale=float(model_cfg.get("input_scale", 1.0)),
        leak=float(model_cfg.get("leak", 1)),
        nonlinearity=(model_cfg.get("nonlinearity", "softplus")).lower(),
        readout=(model_cfg.get("readout", "e_only")).lower(),
        softplus_beta=float(model_cfg.get("softplus_beta", 8.0)),
        softplus_threshold=float(model_cfg.get("softplus_threshold", 20.0)),
    )

    model = EIRNN(input_size=D_in, output_size=C_out, cfg=ei_cfg).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warning] load_state_dict strict=False | missing: {missing} | unexpected: {unexpected}")
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    signs = getattr(model, "sign_vec", None)
    signs = signs.detach().to(device) if isinstance(signs, torch.Tensor) else None

    extra = {"signs": signs, "config": cfg_full}
    return model, extra


def get_val_loader(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {}) or {}
    tasks = cfg.get("tasks", ["dm1"])
    if isinstance(tasks, str):
        tasks = [tasks]
    task = tasks[0]

    batch_size = int(data_cfg.get("batch_size", 128))
    seq_len    = int(data_cfg.get("seq_len", 350))

    from src.data import mod_cog_tasks as mct
    from neurogym import Dataset
    env = getattr(mct, task)()
    ds_val = Dataset(env, batch_size=batch_size, seq_len=seq_len, batch_first=True)

    def _iterator():
        while True:
            X, Y = ds_val()
            yield {
                "x": torch.from_numpy(X).float(),
                "y": torch.from_numpy(Y).long(),
            }

    return _iterator()

def forward_collect(model: torch.nn.Module, batch: dict, return_states: bool):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    X = batch.get("x_override", None)
    if X is None:
        X = batch["x"]
    Y = batch["y"]
    B, T, D = X.shape
    H = model.cfg.hidden_size
    device = X.device

    @torch.no_grad()
    def decision_mask_from_inputs(Xbt: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        return (Xbt[..., 0] < thresh) 

    class ModCogLossCombined(nn.Module):
        def __init__(self, label_smoothing: float = 0.1, fixdown_weight: float = 0.05):
            super().__init__()
            self.mse = nn.MSELoss()
            self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.fixdown_weight = float(fixdown_weight)

        def forward(self, outputs: torch.Tensor, labels: torch.Tensor, dec_mask: torch.Tensor):
            B, T, C = outputs.shape
            target_fix = outputs.new_zeros(B, T, C)
            target_fix[..., 0] = 1.0

            if (~dec_mask).any():
                loss_fix = self.mse(outputs[~dec_mask], target_fix[~dec_mask])
            else:
                loss_fix = outputs.sum() * 0.0

            if dec_mask.any():
                labels_shift = labels + 1
                loss_dec = self.ce(outputs[dec_mask], labels_shift[dec_mask])
                fix_logits_dec = outputs[..., 0][dec_mask]
                loss_fixdown = (fix_logits_dec ** 2).mean() * self.fixdown_weight
            else:
                loss_dec = outputs.sum() * 0.0
                loss_fixdown = outputs.sum() * 0.0

            return loss_fix + loss_dec + loss_fixdown

    h = torch.zeros(B, H, device=device, requires_grad=True)

    Wx, Wh, b = model.W_xh, model.W_hh, model.b_h
    alpha = model._alpha

    h_hist = []
    u_hist = []
    logits_list = []

    for t in range(T):
        pre = X[:, t, :] @ Wx.T + h @ Wh.T + b
        phi = model._phi(pre)
        h = (1.0 - alpha) * h + alpha * phi

        h.retain_grad()

        u_hist.append(pre)
        h_hist.append(h)

        h_ro = h * model.e_mask if getattr(model, "_readout_mode", "e_only") == "e_only" else h
        logits_t = model.W_out(h_ro)
        logits_list.append(logits_t)

    logits = torch.stack(logits_list, dim=1).contiguous()
    h_seq  = torch.stack(h_hist,   dim=0).contiguous()
    u_seq  = torch.stack(u_hist,   dim=0).contiguous()

    dec_mask = decision_mask_from_inputs(X, thresh=0.5)
    loss = ModCogLossCombined(label_smoothing=0.1, fixdown_weight=0.05)(logits, Y, dec_mask)

    cache = {}
    if return_states:
        cache["h_seq"]  = h_seq
        cache["u_seq"]  = u_seq
        cache["x_seq"]  = X.transpose(0, 1).contiguous()
        cache["h_list"] = h_hist
        cache["loss"]   = loss

    return logits, cache

def _set_all_seeds(s: int):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    out_dir = ensure_outdir(cfg["out_dir"], "grads")

    seeds = cfg.get("seeds", None)
    if seeds is None:
        seeds = [int(cfg.get("seed", 12345))]
    else:
        seeds = [int(s) for s in seeds]

    device = cfg.get("device", "cpu")
    ckpt_path = cfg["checkpoint"]

    model, extra = build_model_and_load(ckpt_path, device)

    if "S" in extra:
        S = torch.tensor(extra["S"], dtype=torch.float32, device=device)
        idx_E, idx_I = ei_indices_from_sign_matrix(S)
    elif "signs" in extra:
        s = extra["signs"].detach().clone().to(torch.int8).to(device).view(-1)
        if not torch.all((s == 1) | (s == -1)):
            raise ValueError("Invalid 'signs' vector: must be ±1.")
        idx_E = (s == 1); idx_I = (s == -1)
    else:
        raise KeyError("Checkpoint must contain either 'S' (Dale columns) or 'signs' vector.")

    loader = get_val_loader(cfg)

    write_single = (len(seeds) == 1)

    for seed in seeds:
        _set_all_seeds(seed)
        data = next(loader)

        class _ModelAdapter(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module):
                super().__init__(); self.base = base_model
            def forward(self, x, y=None, return_all=False):
                logits, cache = forward_collect(self.base, {"x": x, "y": y}, return_states=True)
                if return_all:
                    return {
                        "loss": cache["loss"],
                        "h_list": cache["h_list"],
                        "u_list": cache.get("u_seq", None).unbind(dim=0) if "u_seq" in cache else None,
                    }
                return logits

        model_for_hooks = _ModelAdapter(model).to(device)

        tensors = collect_hidden_and_grads(
            model_for_hooks, data, device=device,
            window=tuple(cfg.get("window", [None, None])) if cfg.get("window") else None,
            stride=int(cfg.get("stride", 1)),
            return_grad_x=bool(cfg.get("return_grad_x", True)),
            fixation_channel=int(cfg.get("fixation_channel", 0)),
            decision_thresh=float(cfg.get("decision_thresh", 0.5)),
        )

        tensors["idx_E"] = idx_E.detach().cpu()
        tensors["idx_I"] = idx_I.detach().cpu()

        meta = {"checkpoint": str(ckpt_path), "seed": int(seed), "device": device}
        if write_single:
            pt_path = out_dir / "grads.pt"
            meta_path = out_dir / "meta.yaml"
        else:
            pt_path = out_dir / f"grads_seed{seed:04d}.pt"
            meta_path = out_dir / f"meta_seed{seed:04d}.yaml"

        save_yaml(meta, meta_path)
        save_tensors(tensors, pt_path)
        print(f"[collect_grads] Saved tensors to {pt_path}")

    print("[collect_grads] Done.")

if __name__ == "__main__":
    main()
