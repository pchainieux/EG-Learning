from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable, Any
import torch
from torch import Tensor

def ei_indices_from_sign_matrix(S: Tensor) -> Tuple[Tensor, Tensor]:
    if S.dim() != 2 or S.size(0) != S.size(1):
        raise ValueError("S must be square [N, N] with column-wise fixed signs.")
    col_mean = S.mean(dim=0)
    signs = torch.sign(col_mean).to(torch.int8)
    if not torch.all((signs == 1) | (signs == -1)):
        any_pos = (S > 0).any(dim=0)
        any_neg = (S < 0).any(dim=0)
        signs = torch.where(any_pos & ~any_neg, torch.tensor(1, dtype=torch.int8, device=S.device),
                 torch.where(any_neg & ~any_pos, torch.tensor(-1, dtype=torch.int8, device=S.device),
                 torch.tensor(1, dtype=torch.int8, device=S.device)))
    idx_E = signs == 1
    idx_I = signs == -1
    return idx_E, idx_I


@torch.no_grad()
def _maybe_slice_time(seq: Optional[Tensor], t0: Optional[int], t1: Optional[int], stride: int) -> Optional[Tensor]:
    if seq is None:
        return None
    T = seq.size(0)
    start = 0 if t0 is None else max(0, t0)
    end = T if t1 is None else min(T, t1)
    return seq[start:end:stride].contiguous()


def collect_hidden_and_grads(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    forward_collect: Callable[[torch.nn.Module, Dict[str, Any], bool], Tuple[Tensor, Dict[str, Tensor]]],
    *,
    compute_grad_x: bool = False,
    time_window: Optional[Tuple[Optional[int], Optional[int]]] = None,
    stride: int = 1,
    retain_graph: bool = False,
) -> Dict[str, Tensor]:
    model.eval()

    _, cache = forward_collect(model, batch, True)

    h_seq: Tensor = cache["h_seq"]
    h_list: list[Tensor] = cache.get("h_list", None)
    if h_list is None:
        raise KeyError(
            "forward_collect must return cache['h_list'] "
            "as a list of per-timestep hidden states (each [B,N])."
        )
    loss: Tensor = cache["loss"]

    grads_list = torch.autograd.grad(
        loss, h_list, retain_graph=retain_graph, allow_unused=False
    )
    grad_h_full = torch.stack([g.detach() for g in grads_list], dim=0)

    grad_x_full = None
    x_seq_TBD: Optional[Tensor] = cache.get("x_seq", None)
    if compute_grad_x and x_seq_TBD is not None:
        x_bt = x_seq_TBD.transpose(0, 1).contiguous().detach().requires_grad_(True)

        def _rerun_with_x(x_override_bt: Tensor) -> Tensor:
            b2 = dict(batch)
            b2["x_override"] = x_override_bt
            _, cache2 = forward_collect(model, b2, True)
            return cache2["loss"]

        loss2 = _rerun_with_x(x_bt)
        gx_bt = torch.autograd.grad(loss2, x_bt, retain_graph=retain_graph)[0].detach()
        grad_x_full = gx_bt.transpose(0, 1).contiguous()

    t0, t1 = (None, None) if time_window is None else time_window
    h_slice  = _maybe_slice_time(h_seq.detach(), t0, t1, stride)
    g_slice  = _maybe_slice_time(grad_h_full, t0, t1, stride)
    u_slice  = _maybe_slice_time(cache.get("u_seq", None), t0, t1, stride)
    x_slice  = _maybe_slice_time(x_seq_TBD.detach() if x_seq_TBD is not None else None, t0, t1, stride)
    gx_slice = _maybe_slice_time(grad_x_full, t0, t1, stride) if grad_x_full is not None else None

    out = {
        "h_seq": h_slice,
        "grad_h": g_slice,
        "loss": loss.detach(),
    }
    if u_slice is not None:
        out["u_seq"] = u_slice.detach()
    if x_slice is not None:
        out["x_seq"] = x_slice
    if gx_slice is not None:
        out["grad_x"] = gx_slice
    return out

