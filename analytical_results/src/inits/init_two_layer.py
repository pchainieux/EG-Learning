import numpy as np

def build_init(name, Sigma_yx, N1, N2, N3, seed=0, **kwargs):
    rng = np.random.RandomState(seed)

    if name == "gauss-small":
        scale = kwargs.get("scale", 1e-1)
        W21 = rng.randn(N2, N1) * scale
        W32 = rng.randn(N3, N2) * scale
        return W21, W32

    elif name == "gauss-big":
        scale = kwargs.get("scale", 0.5)
        W21 = rng.randn(N2, N1) * scale
        W32 = rng.randn(N3, N2) * scale
        return W21, W32

    elif name == "uniform":
        scale = kwargs.get("scale", 1e-1)
        W21 = rng.uniform(-scale, scale, size=(N2, N1))
        W32 = rng.uniform(-scale, scale, size=(N3, N2))
        return W21, W32

    elif name == "diag":
        scale = kwargs.get("scale", 0.1)
        W21 = np.zeros((N2, N1))
        W32 = np.zeros((N3, N2))
        for i in range(min(N1, N2, N3)):
            W21[i, i] = np.sqrt(scale)
            W32[i, i] = np.sqrt(scale)
        return W21, W32

    elif name == "lowrank":
        scale = kwargs.get("scale", 0.1)
        u = rng.randn(N2, 1); v = rng.randn(1, N1)
        W21 = scale * (u @ v)
        u2 = rng.randn(N3, 1); v2 = rng.randn(1, N2)
        W32 = scale * (u2 @ v2)
        return W21, W32