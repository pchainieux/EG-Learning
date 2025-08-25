from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch

from ei_gradient.src.io import load_yaml, ensure_outdir, save_yaml, save_tensors
from ei_gradient.src.hooks import collect_hidden_and_grads, ei_indices_from_sign_matrix

# ============================
# TODO: wire these to your repo
# ============================
def build_model_and_load(checkpoint_path: str, device: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Replace the body with your model construction + checkpoint restore.
    Must return:
      - model: torch.nn.Module (eval mode ok)
      - extra: dict that may contain 'S' (Dale sign matrix [N,N]) or 'signs' ([N])
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Example sketch:
    # from ei_rnn_modcog.models import EIModel
    # model = EIModel.from_config(ckpt["config"]).to(device)
    # model.load_state_dict(ckpt["model_state"])
    # extra = {"S": ckpt.get("S", None), "config": ckpt.get("config", {})}
    # return model, extra
    raise NotImplementedError("Connect build_model_and_load() to your codebase")


def get_val_loader(cfg: Dict[str, Any]):
    """
    Replace with your dataset/dataloader for the validation split.
    Should yield dict-like 'batch' objects that your model understands.
    """
    # from ei_rnn_modcog.data import make_val_loader
    # return make_val_loader(cfg["task"], split=cfg["split"], batch_size=cfg["batch_size"])
    raise NotImplementedError("Connect get_val_loader() to your codebase")
# ============================


def forward_collect(model: torch.nn.Module, batch: Dict[str, Any], return_states: bool):
    """
    Adapter that runs a forward pass and returns (y_hat, cache) where cache includes:
        - h_seq [T,B,N] (requires_grad=True)
        - u_seq [T,B,N] (optional)
        - x_seq [T,B,D] (optional)
        - loss  scalar
    Modify to match your model's API.
    """
    # Example (sketch):
    # y_hat, cache = model.forward_with_cache(batch, return_states=True)
    # loss = model.loss(y_hat, batch)    # must be scalar
    # cache["loss"] = loss
    # return y_hat, cache
    raise NotImplementedError("Implement forward_collect() to return required cache")


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
