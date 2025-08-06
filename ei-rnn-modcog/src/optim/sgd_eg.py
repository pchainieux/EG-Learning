import torch
from torch.optim.optimizer import Optimizer

class SGD_EG(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
        weight_decay=0.0,
        update_alg="gd",
        min_magnitude=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_alg=update_alg,
            min_magnitude=min_magnitude,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd  = group["weight_decay"]
            alg = group["update_alg"]
            min_mag = float(group.get("min_magnitude", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]

                if alg == "gd":
                    if mom:
                        buf = state.get("momentum_buffer")
                        if buf is None:
                            buf = state["momentum_buffer"] = torch.zeros_like(p)
                        buf.mul_(mom).add_(g)
                        g = buf
                    if wd:
                        g = g + wd * p
                    p.add_(g, alpha=-lr)

                elif alg == "eg":
                    g_mag = p.sign() * g
                    if wd:
                        g_mag = g_mag + wd * p.abs()
                    if mom:
                        buf = state.get("momentum_buffer")
                        if buf is None:
                            buf = state["momentum_buffer"] = torch.zeros_like(p)
                        buf.mul_(mom).add_(g_mag)
                        g_mag = buf

                    p.mul_(torch.exp(-lr * g_mag))

                    if min_mag > 0.0:
                        s = p.sign()
                        p.copy_(s * p.abs().clamp_min(min_mag))
                else:
                    raise ValueError(f"Unknown update_alg: {alg}")
