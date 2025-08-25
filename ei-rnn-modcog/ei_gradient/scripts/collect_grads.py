from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch

from ei_gradient.src.io import load_yaml, ensure_outdir, save_yaml, save_tensors
from ei_gradient.src.hooks import collect_hidden_and_grads, ei_indices_from_sign_matrix

def build_model_and_load(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Rebuild EIRNN from the training checkpoint and return:
      - model (on device, eval mode)
      - extra dict with 'signs' (E/I sign vector) and the original 'config'
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"]
    cfg_full = ckpt.get("config", {}) or {}
    model_cfg = (cfg_full.get("model", {}) or {})

    # Infer dims from weights to avoid depending on dataset here
    W_xh = state["W_xh"]                   # [H, D_in]
    W_out_w = state["W_out.weight"]        # [C_out, H]
    H, D_in = W_xh.shape
    C_out = W_out_w.shape[0]

    # Pull hyperparams (fallbacks match your defaults)
    from src.models.ei_rnn import EIRNN, EIConfig
    ei_cfg = EIConfig(
        hidden_size=H,  # make sure it matches checkpoint
        exc_frac=float(model_cfg.get("exc_frac", 0.8)),
        spectral_radius=float(model_cfg.get("spectral_radius", 1.2)),
        input_scale=float(model_cfg.get("input_scale", 1.0)),
        leak=float(model_cfg.get("leak", 0.2)),
        nonlinearity=(model_cfg.get("nonlinearity", "softplus")).lower(),
        readout=(model_cfg.get("readout", "e_only")).lower(),
    )

    model = EIRNN(input_size=D_in, output_size=C_out, cfg=ei_cfg).to(device)
    model.load_state_dict(state)
    model.eval()

    # Use sign_vec buffer from the loaded model for E/I masks
    extra = {"signs": model.sign_vec.detach().to(device), "config": cfg_full}
    return model, extra


def get_val_loader(cfg: Dict[str, Any]):
    """
    Build a small validation iterator using the same data pathway as training.
    Yields dicts with keys: 'x' [B,T,D] and 'y' [B,T].
    """
    data_cfg = cfg.get("data", {}) or {}
    tasks = cfg.get("tasks", ["dm1"])
    if isinstance(tasks, str):
        tasks = [tasks]
    task = tasks[0]

    # Defaults match train_singlehead_modcog.py
    batch_size = int(data_cfg.get("batch_size", 128))
    seq_len    = int(data_cfg.get("seq_len", 350))

    from src.data import mod_cog_tasks as mct
    from neurogym import Dataset
    env = getattr(mct, task)()
    ds_val = Dataset(env, batch_size=batch_size, seq_len=seq_len, batch_first=True)

    def _iterator():
        # infinite generator; caller (collect_grads.py) controls max_batches
        while True:
            X, Y = ds_val()  # numpy arrays: X [B,T,D], Y [B,T]
            yield {
                "x": torch.from_numpy(X).float(),
                "y": torch.from_numpy(Y).long(),
            }

    return _iterator()
# ============================


def forward_collect(model: torch.nn.Module, batch: Dict[str, Any], return_states: bool):
    """
    Time-unroll the EIRNN to expose h_seq and u_seq, compute the same loss as training,
    and return (logits, cache). Supports batch['x_override'] for grad_x.
    """
    X = batch.get("x_override", None)
    if X is None:
        X = batch["x"]  # [B,T,D]
    Y = batch["y"]      # [B,T] (labels in {0..D_ring-1}, to be shifted by +1 at decision)

    # Use the same helpers as training
    import torch.nn as nn
    import torch.nn.functional as F

    @torch.no_grad()
    def decision_mask_from_inputs(Xt: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        # decision time where fixation channel < thresh
        return (Xt[..., 0] < thresh)

    class ModCogLossCombined(nn.Module):
        def __init__(self, label_smoothing: float = 0.1, fixdown_weight: float = 0.05):
            super().__init__()
            self.mse = nn.MSELoss()
            self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.fixdown_weight = float(fixdown_weight)
        def forward(self, outputs, labels, dec_mask):
            # outputs: [B,T,C]; labels: [B,T] in {0..K-1} for decisions
            B, T, C = outputs.shape
            target_fix = outputs.new_zeros(B, T, C); target_fix[..., 0] = 1.0
            loss_fix = self.mse(outputs[~dec_mask], target_fix[~dec_mask]) if (~dec_mask).any() else outputs.sum()*0.0
            if dec_mask.any():
                # shift labels by +1 to account for fixation logit at index 0
                labels_shift = labels + 1
                loss_dec = self.ce(outputs[dec_mask], labels_shift[dec_mask])
                fix_logits_dec = outputs[..., 0][dec_mask]
                loss_fixdown = (fix_logits_dec ** 2).mean() * self.fixdown_weight
            else:
                loss_dec = outputs.sum()*0.0
                loss_fixdown = outputs.sum()*0.0
            return loss_fix + loss_dec + loss_fixdown

    # Unroll with access to model internals
    B, T, _ = X.shape
    H = model.cfg.hidden_size
    device = X.device

    h = X.new_zeros(B, H, requires_grad=False)
    Wx, Wh, b = model.W_xh, model.W_hh, model.b_h
    alpha = model._alpha

    h_list = []
    u_list = []
    logits = X.new_zeros(B, T, model.W_out.out_features)

    for t in range(T):
        pre = X[:, t, :] @ Wx.T + h @ Wh.T + b                 # pre-activation u_t
        if model._nl_kind == "softplus":
            phi = F.softplus(pre)
        else:
            phi = torch.tanh(pre)

        h = (1.0 - alpha) * h + alpha * phi                    # h_t
        if model._readout_mode == "e_only":
            h_ro = h * model.e_mask
        else:
            h_ro = h
        logits[:, t, :] = model.W_out(h_ro)

        if return_states:
            u_list.append(pre)
            h_list.append(h)

    # Stack to [T,B,H] and ensure they're on-graph
    if return_states:
        u_seq = torch.stack(u_list, dim=0)                     # [T,B,H]
        h_seq = torch.stack(h_list, dim=0).requires_grad_(True)
    else:
        u_seq = None
        h_seq = None

    # Loss (same as training)
    crit = ModCogLossCombined(
        label_smoothing=float(batch.get("label_smoothing", 0.1)),
        fixdown_weight=float(batch.get("fixdown_weight", 0.05)),
    )
    with torch.no_grad():
        dec_mask = decision_mask_from_inputs(X, thresh=float(batch.get("mask_threshold", 0.5)))
    loss = crit(logits, Y, dec_mask)

    cache = {
        "loss": loss,
        "h_seq": h_seq,                 # [T,B,H]
        "u_seq": u_seq,                 # [T,B,H]
        "x_seq": X.transpose(0, 1),     # [T,B,D] for optional grad_x
    }
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

    tensors = {"h_seq": h_seq, "grad_h": grad_h, "idx_E": idx_E.cpu(), "idx_I": idx_I.cpu()}
    if u_seq is not None:
        tensors["u_seq"] = u_seq
    if x_seq is not None:
        tensors["x_seq"] = x_seq
    if grad_x is not None:
        tensors["grad_x"] = grad_x

    save_tensors(tensors, out_dir / "grads.pt")
    print(f"[collect_grads] Saved tensors to {out_dir / 'grads.pt'}")


if __name__ == "__main__":
    main()
