# experiments/credit_assignment/creditlib/hooks.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable, Any
import torch
from torch import Tensor

"""
Hooks & collection utilities for capturing time-resolved hidden states, pre-activations,
and gradients dℓ/dh_t (and optionally dℓ/dx_t) in Dale-constrained RNNs.

Design:
- The caller provides a `forward_collect` function that executes a forward pass and returns:
    y_hat, cache = forward_collect(model, batch, return_states=True)
  where `cache` MUST contain:
    cache["h_seq"]: Tensor [T, B, N]    — hidden states (requires_grad=True)
    cache["u_seq"]: Tensor [T, B, N] or None — pre-activations before ϕ (optional but useful)
    cache["x_seq"]: Tensor [T, B, D] or None — input sequence used for the forward (optional)
    cache["loss"]:  scalar Tensor       — the loss used for backward
  This keeps the hooks generic and model-agnostic.

- Gradients:
    After `loss.backward(retain_graph=True)`, `h_seq.grad` yields dℓ/dh_t.
    Optionally, grad w.r.t inputs is obtained via torch.autograd.grad on loss and x_seq.

- E/I indices:
    Derived once from the Dale sign matrix S (columns are presynaptic sign).
"""

def ei_indices_from_sign_matrix(S: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Given S with shape [N, N] where column j has fixed presynaptic sign s_j ∈ {+1,-1},
    derive boolean index tensors for excitatory (E) and inhibitory (I).

    Returns:
        idx_E: Bool tensor [N] True where neuron is excitatory (+1 column sign)
        idx_I: Bool tensor [N] True where neuron is inhibitory (−1 column sign)
    """
    if S.dim() != 2 or S.size(0) != S.size(1):
        raise ValueError("S must be square [N, N] with column-wise fixed signs.")
    # infer column sign by taking the sign of the column mean (robust to zeros)
    col_mean = S.mean(dim=0)
    signs = torch.sign(col_mean).to(torch.int8)  # +1 for E, -1 for I
    if not torch.all((signs == 1) | (signs == -1)):
        # Fallback: if zeros appear, infer by any non-zero entry per column
        any_pos = (S > 0).any(dim=0)
        any_neg = (S < 0).any(dim=0)
        signs = torch.where(any_pos & ~any_neg, torch.tensor(1, dtype=torch.int8, device=S.device),
                 torch.where(any_neg & ~any_pos, torch.tensor(-1, dtype=torch.int8, device=S.device),
                 torch.tensor(1, dtype=torch.int8, device=S.device)))  # default to +1 if ambiguous
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
    retain_graph: bool = False
) -> Dict[str, Tensor]:
    """
    Run a forward pass that records sequences and backprop to obtain dℓ/dh_t (and optionally dℓ/dx_t).

    Args:
        model: RNN model.
        batch: Input batch for the model.
        forward_collect: Callable returning (y_hat, cache) with:
            cache["h_seq"]: [T,B,N] requires_grad=True
            cache["u_seq"]: [T,B,N] or None
            cache["x_seq"]: [T,B,D] or None
            cache["loss"] : scalar
        compute_grad_x: If True, also compute dℓ/dx_t via autograd.
        time_window: (t0, t1) bounds (inclusive-exclusive) to slice the time dimension. None = full.
        stride: subsample time by this stride (>=1).
        retain_graph: Whether to retain graph after backward.

    Returns:
        A dict containing (sliced by time window and stride):
          - "h_seq": [T',B,N]
          - "u_seq": [T',B,N] (if available in cache; else omitted)
          - "grad_h": [T',B,N]
          - "x_seq": [T',B,D] (if available)
          - "grad_x": [T',B,D] (if requested and x_seq provided)
          - "loss": scalar tensor
    """
    model.eval()
    yhat, cache = forward_collect(model, batch, True)
    h_seq: Tensor = cache["h_seq"]
    if not h_seq.requires_grad:
        h_seq.requires_grad_(True)
    # Make sure per-step tensors retain grad
    h_seq.retain_grad()

    loss: Tensor = cache["loss"]
    # Backward to get gradients on h_seq
    loss.backward(retain_graph=retain_graph)
    grad_h_full = h_seq.grad.detach()

    # Optional grad wrt inputs
    grad_x_full = None
    x_seq: Optional[Tensor] = cache.get("x_seq", None)
    if compute_grad_x and x_seq is not None:
        if not x_seq.requires_grad:
            x_seq = x_seq.clone().detach().requires_grad_(True)
        # recompute loss with x_seq requiring grad (cheap if cached; else caller must support)
        # Re-run forward with same batch but injecting x_seq if forward supports it.
        # Fallback: use autograd.grad on the saved graph if available.
        # Here: prefer a fresh forward for correctness.
        def _rerun_with_x(x_override: Tensor) -> Tensor:
            b2 = dict(batch)
            b2["x_override"] = x_override  # caller can use this to override inputs
            y2, cache2 = forward_collect(model, b2, True)
            return cache2["loss"]

        loss2 = _rerun_with_x(x_seq)
        grad_x_full = torch.autograd.grad(loss2, x_seq, retain_graph=retain_graph)[0].detach()

    # Slice by time window + stride
    t0, t1 = (None, None) if time_window is None else time_window
    h_slice = _maybe_slice_time(h_seq.detach(), t0, t1, stride)
    g_slice = _maybe_slice_time(grad_h_full, t0, t1, stride)
    u_slice = _maybe_slice_time(cache.get("u_seq", None), t0, t1, stride)
    x_slice = _maybe_slice_time(x_seq.detach() if x_seq is not None else None, t0, t1, stride)
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
