from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class EIConfig:
    hidden_size: int = 256
    exc_frac: float = 0.8
    spectral_radius: float = 1.2
    input_scale: float = 1.0

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

        self.W_xh = nn.Parameter(torch.empty(H, input_size))
        self.W_hh = nn.Parameter(torch.empty(H, H))
        self.b_h   = nn.Parameter(torch.zeros(H))

        self.W_out = nn.Linear(H, output_size) 

        self.reset_parameters()

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
        s = self.sign_vec
        self.W_hh.copy_(self.W_hh.abs() * s)

    @torch.no_grad()
    def rescale_spectral_radius_(self, target=None):
        target = float(self.cfg.spectral_radius if target is None else target)
        U, S, Vh = torch.linalg.svd(self.W_hh, full_matrices=False)
        max_s = S.max().clamp_min(1e-6)
        self.W_hh.mul_(target / max_s)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.cfg.hidden_size
        h = x.new_zeros(B, H)
        out = x.new_zeros(B, T, self.W_out.out_features)
        relu = torch.relu

        Wx = self.W_xh
        Wh = self.W_hh
        b  = self.b_h

        for t in range(T):
            h = relu(x[:, t, :] @ Wx.T + h @ Wh.T + b)
            out[:, t, :] = self.W_out(h)
        return out
