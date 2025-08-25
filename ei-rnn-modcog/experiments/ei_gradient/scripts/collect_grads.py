from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch

from experiments.ei_gradient.src.io_utils import load_yaml, ensure_outdir, save_yaml, save_tensors
from experiments.ei_gradient.src.hooks import collect_hidden_and_grads, ei_indices_from_sign_matrix

def build_model_and_load(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    import torch, sys
    import numpy as np

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
        leak=float(model_cfg.get("leak", 0.2)),
        nonlinearity=(model_cfg.get("nonlinearity", "softplus")).lower(),
        readout=(model_cfg.get("readout", "e_only")).lower(),
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
        phi = F.softplus(pre) if model._nl_kind == "softplus" else torch.tanh(pre)
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





def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    args = p.parse_args()

    cfg = load_yaml(args.cfg)
    exp_id = cfg.get("exp_id", "exp_ei_credit")
    out_dir = ensure_outdir(cfg.get("out_dir", "outputs"), exp_id)
    save_yaml(cfg, out_dir / "config.yaml")

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model, extra = build_model_and_load(cfg["checkpoint"], device)
    model.to(device).eval()

    # E/I indices from Dale sign matrix S or signs
    if cfg.get("ei_from_checkpoint", True):
        S = extra.get("S", None)
        if S is None:
            signs = extra.get("signs", None)
            if signs is not None:
                # fabricate S from signs if needed
                s = torch.as_tensor(signs, device=device, dtype=torch.float32)
                S = torch.sign(s)[None, :].repeat(s.numel(), 1)  # [N,N] with col-signs
            else:
                raise RuntimeError("No 'S' or 'signs' found in checkpoint extras.")
        if isinstance(S, torch.Tensor) and S.device != torch.device(device):
            S = S.to(device)
        idx_E, idx_I = ei_indices_from_sign_matrix(S)
    else:
        raise RuntimeError("Set ei_from_checkpoint=true or provide indices manually.")

    # Data loader
    loader = get_val_loader(cfg)

    # Collection options
    tw = cfg.get("time_window", {"start": None, "end": None, "stride": 1})
    time_window = (tw.get("start", None), tw.get("end", None))
    stride = int(tw.get("stride", 1))
    compute_grad_x = bool(cfg.get("save", {}).get("grad_x", False))
    max_batches = int(cfg.get("max_batches", 1))  # keep memory small; adjust as needed

    # Accumulators (concatenate over batches)
    Hs, Gs, Us = [], [], []
    Xs, GXs = [], []

    with torch.no_grad():
        # we need gradients; temporarily enable grad during the inner loop
        pass

    n_batches = 0
    for batch in loader:
        # Move tensors in batch to device if necessary
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        out = collect_hidden_and_grads(
            model, batch, forward_collect,
            compute_grad_x=compute_grad_x,
            time_window=time_window,
            stride=stride,
            retain_graph=False
        )
        Hs.append(out["h_seq"].cpu())
        Gs.append(out["grad_h"].cpu())
        if "u_seq" in out:
            Us.append(out["u_seq"].cpu())
        if "x_seq" in out:
            Xs.append(out["x_seq"].cpu())
        if "grad_x" in out:
            GXs.append(out["grad_x"].cpu())

        n_batches += 1
        if n_batches >= max_batches:
            break

    # Stack along batch dimension (dim=1).
    def _stack_time_batch(lst):
        if not lst:
            return None
        # Elements are [T',B,N] – concat B
        return torch.cat(lst, dim=1)

    h_seq = _stack_time_batch(Hs)   # [T', B_total, N]
    grad_h = _stack_time_batch(Gs)  # [T', B_total, N]
    u_seq = _stack_time_batch(Us) if Us else None
    x_seq = _stack_time_batch(Xs) if Xs else None
    grad_x = _stack_time_batch(GXs) if GXs else None

    want = cfg.get("save", {})
    tensors = {"idx_E": idx_E.cpu(), "idx_I": idx_I.cpu()}
    if want.get("ht", True):     tensors["h_seq"] = h_seq
    if want.get("grad_h", True): tensors["grad_h"] = grad_h
    if want.get("ut", False) and u_seq is not None: tensors["u_seq"] = u_seq
    if want.get("grad_x", False) and grad_x is not None:
        tensors["x_seq"] = x_seq
        tensors["grad_x"] = grad_x


    T, B, N = h_seq.shape
    assert grad_h.shape == (T, B, N), f"grad_h has shape {grad_h.shape}, expected {(T, B, N)}"
    assert idx_E.dtype == torch.bool and idx_I.dtype == torch.bool, "idx_E/idx_I must be boolean"
    # disjoint and cover all units
    assert torch.all(idx_E ^ idx_I) and torch.all(idx_E | idx_I), "E/I masks must be disjoint and exhaustive"
    if "grad_x" in locals() and grad_x is not None:
        D = grad_x.shape[-1]
        assert x_seq is not None, "grad_x present but x_seq missing"
        assert x_seq.shape == grad_x.shape == (T, B, D), \
            f"x_seq {x_seq.shape} and grad_x {grad_x.shape} must both be (T,B,D)"
        
    save_tensors(tensors, out_dir / "grads.pt")
    print(f"[collect_grads] Saved tensors to {out_dir / 'grads.pt'}")


if __name__ == "__main__":
    main()
