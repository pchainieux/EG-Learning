from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable, Any
import torch
from torch import Tensor

def ei_indices_from_sign_matrix(S: Tensor) -> Tuple[Tensor, Tensor]:
    if S.dim() != 2 or S.size(0) != S.size(1):
        raise ValueError("S must be square [N, N] with column-wise fixed signs.")

    any_pos = (S > 0).any(dim=0)
    any_neg = (S < 0).any(dim=0)

    mixed = any_pos & any_neg
    if mixed.any():
        bad = mixed.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Dale violation: mixed-sign columns at indices {bad}")

    idx_E = any_pos & ~any_neg
    idx_I = any_neg & ~any_pos

    if not torch.all(idx_E ^ idx_I) or not torch.all(idx_E | idx_I):
        raise ValueError("E/I masks must be disjoint and exhaustive (no zero columns).")

    return idx_E, idx_I

@torch.no_grad()
def _maybe_slice_time(seq: Optional[Tensor], t0: Optional[int], t1: Optional[int], stride: int) -> Optional[Tensor]:
    if seq is None:
        return None
    T = seq.size(0)
    start = 0 if t0 is None else max(0, t0)
    end = T if t1 is None else min(T, t1)
    return seq[start:end:stride].contiguous()

@torch.no_grad()
def _infer_decision_mask_from_inputs(x_seq: Tensor, fixation_channel: int = 0, thresh: float = 0.5) -> Tensor:
    if x_seq.dim() != 3:
        raise ValueError(f"x_seq must be (T,B,D); got {tuple(x_seq.shape)}")
    T, B, D = x_seq.shape
    if fixation_channel >= D:
        raise ValueError(f"fixation_channel={fixation_channel} not < D={D}")
    fix = x_seq[..., fixation_channel] 
    return (fix < thresh)

def collect_hidden_and_grads(
    model: torch.nn.Module,
    batch: Dict[str, Tensor],
    device: str = "cpu",
    *,
    window: Optional[Tuple[int, int]] = None,  
    stride: int = 1,
    return_grad_x: bool = False,
    fixation_channel: int = 0,
    decision_thresh: float = 0.5,
    retain_graph: bool = False,
) -> Dict[str, Tensor]:
    model = model.to(device)
    model.eval()

    x_in: Tensor = batch["x"].to(device) 
    y = batch.get("y", None)
    dec_mask_ds: Optional[Tensor] = batch.get("mask_decision", None)

    need_x_grad = bool(return_grad_x)
    x_for_grad = x_in.detach().clone().requires_grad_(need_x_grad)

    out = model(x_for_grad, y, return_all=True)

    loss: Tensor = out["loss"]
    h_list = out["h_list"] 
    u_list = out.get("u_list", None) 

    for h in h_list:
        h.retain_grad()
    if u_list is not None:
        for u in u_list:
            u.retain_grad()

    grad_h_full = torch.stack(
        torch.autograd.grad(loss, h_list, retain_graph=True),
        dim=0
    ) 
    grad_u_full = None
    if u_list is not None:
        gu_list = torch.autograd.grad(
            loss, u_list, retain_graph=retain_graph, allow_unused=True
        )
        gu_list = [torch.zeros_like(u) if g is None else g for u, g in zip(u_list, gu_list)]
        grad_u_full = torch.stack(gu_list, dim=0)

    grad_x_full = None
    if need_x_grad:
        loss.backward(retain_graph=False)
        grad_x_full = x_for_grad.grad.detach()

    if x_in.dim() != 3:
        raise ValueError(f"x must be 3D, got shape {tuple(x_in.shape)}")
    B_or_T0, B_or_T1, D = x_in.shape
    T_h, B_h, _ = grad_h_full.shape

    if x_in.shape[0] == B_h and x_in.shape[1] == T_h:
        x_tb = x_in.transpose(0, 1).contiguous()
    elif x_in.shape[0] == T_h and x_in.shape[1] == B_h:
        x_tb = x_in.contiguous()
    else:
        x_tb = x_in.transpose(0, 1).contiguous()

    gx_tb = None
    if grad_x_full is not None:
        if grad_x_full.shape == x_in.shape:
            gx_tb = grad_x_full.transpose(0, 1).contiguous()
        elif grad_x_full.shape == x_tb.shape:
            gx_tb = grad_x_full.contiguous() 
        else:
            raise RuntimeError(
                f"grad_x shape {tuple(grad_x_full.shape)} incompatible with x shapes "
                f"{tuple(x_in.shape)} or {tuple(x_tb.shape)}"
            )

    T = grad_h_full.shape[0]
    t0, t1 = (0, T) if window is None else window
    sl = slice(t0, t1, stride)

    h_slice  = torch.stack(h_list, dim=0)[sl].detach()         
    g_slice  = grad_h_full[sl].detach()                        
    u_slice  = torch.stack(u_list, dim=0)[sl].detach() if u_list is not None else None
    gu_slice = grad_u_full[sl].detach() if grad_u_full is not None else None
    x_slice  = x_tb[sl] if need_x_grad or ("x" in batch) else None
    gx_slice = gx_tb[sl] if gx_tb is not None else None

    if dec_mask_ds is not None:
        dec_mask = dec_mask_ds.to(device)[sl]
    elif x_slice is not None:
        dec_mask = _infer_decision_mask_from_inputs(x_slice, fixation_channel, decision_thresh)
    else:
        dec_mask = torch.ones(h_slice.size(0), h_slice.size(1), dtype=torch.bool, device=device)

    out_tensors: Dict[str, Tensor] = {
        "h_seq":  h_slice,          
        "grad_h": g_slice,          
        "loss":   loss.detach(),
        "dec_mask": dec_mask.detach(),  
    }
    if u_slice is not None:
        out_tensors["u_seq"] = u_slice
    if gu_slice is not None:
        out_tensors["grad_u"] = gu_slice
    if x_slice is not None:
        out_tensors["x_seq"] = x_slice  
    if gx_slice is not None:
        out_tensors["grad_x"] = gx_slice     

    return out_tensors
