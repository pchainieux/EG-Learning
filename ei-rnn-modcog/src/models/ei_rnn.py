from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class EIConfig:
    hidden_size: int = 256
    exc_frac: float = 0.8
    spectral_radius: float = 1.2
    input_scale: float = 1.0
    leak: float = 0.2
    nonlinearity: str = "softplus"
    readout: str = "all"
    softplus_beta: float = 8.0 
    softplus_threshold: float = 20.0  

class EIRNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, cfg: EIConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size

        n_exc = int(round(cfg.exc_frac * H))
        idx = torch.randperm(H)
        exc_idx = idx[:n_exc]
        inh_idx = idx[n_exc:]
        sign = torch.ones(H)
        sign[inh_idx] = -1.0
        self.register_buffer("sign_vec", sign)  

        e_mask = (self.sign_vec > 0).float()
        self.register_buffer("e_mask", e_mask)

        self.W_xh = nn.Parameter(torch.empty(H, input_size))
        self.W_hh = nn.Parameter(torch.empty(H, H))
        self.b_h   = nn.Parameter(torch.zeros(H))

        self.W_out = nn.Linear(H, output_size)

        nl = (cfg.nonlinearity or "softplus").lower()
        if nl not in ("softplus", "tanh"):
            raise ValueError("EIConfig.nonlinearity must be 'softplus' or 'tanh'")
        self._nl_kind = nl

        ro = (cfg.readout or "e_only").lower()
        if ro not in ("e_only", "all"):
            raise ValueError("EIConfig.readout must be 'e_only' or 'all'")
        self._readout_mode = ro

        alpha = float(cfg.leak)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("EIConfig.leak must be in (0, 1]")
        self._alpha = alpha

        self.reset_parameters()

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self._nl_kind == "softplus":
            return F.softplus(x,
                            beta=float(self.cfg.softplus_beta),
                            threshold=float(self.cfg.softplus_threshold))
        else:
            return torch.tanh(x)


    @torch.jit.ignore
    def _step(self, h, xw_t):
        pre = xw_t + torch.nn.functional.linear(h, self.W_hh, self.b_h)
        h = (1.0 - self._alpha) * h + self._alpha * self._phi(pre)
        h_ro = h * self.e_mask if self._readout_mode == "e_only" else h
        y = self.W_out(h_ro)
        return h, y

    def reset_parameters(self):
        H = self.cfg.hidden_size
        nn.init.kaiming_uniform_(self.W_xh, a=0.0)
        nn.init.kaiming_uniform_(self.W_hh, a=0.0)

        self.project_EI_()

        with torch.no_grad():
            col_std = self.W_hh.abs().mean(dim=0).clamp_min(1e-6)
            self.W_hh.div_(col_std)
            U, S, Vh = torch.linalg.svd(self.W_hh, full_matrices=False)
            self.W_hh.mul_(self.cfg.spectral_radius / S.max().clamp_min(1e-6))

        with torch.no_grad():
            self.W_xh.mul_(self.cfg.input_scale)

        nn.init.zeros_(self.b_h)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    @torch.no_grad()
    def project_EI_(self):
        self.W_hh.copy_(self.W_hh.abs() * self.sign_vec)

    @torch.no_grad()
    def rescale_spectral_radius_(self, tol=0.10):
        target = float(self.cfg.spectral_radius)
        _, S, _ = torch.linalg.svd(self.W_hh, full_matrices=False)
        max_s = S.max().clamp_min(1e-6)
        if not ( (1.0 - tol) * target <= max_s <= (1.0 + tol) * target ):
            self.W_hh.mul_(target / max_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.cfg.hidden_size

        xw = torch.nn.functional.linear(x, self.W_xh, bias=None)

        h = x.new_zeros(B, H)
        out = x.new_zeros(B, T, self.W_out.out_features)

        for t in range(T):
            h, y_t = self._step(h, xw[:, t, :])
            out[:, t, :] = y_t

        return out