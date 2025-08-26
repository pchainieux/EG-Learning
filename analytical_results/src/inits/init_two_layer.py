import numpy as np

'''
def init_gauss(N1, N2, N3, scale, rng):
    W21 = rng.randn(N2, N1) * scale
    W32 = rng.randn(N3, N2) * scale
    return W21, W32

def init_boost_weaker_mode(Sigma_yx, N2, weaker_mode_idx=1,
                           boost_factor=0.1, base_factor=0.01,
                           random_mixing=False, seed=None):
    U, S_vals, Vt = np.linalg.svd(Sigma_yx, full_matrices=False)
    K = len(S_vals)
    a0 = np.array([ base_factor * S_vals[i] for i in range(K) ])
    a0[weaker_mode_idx] = boost_factor * S_vals[weaker_mode_idx]
    c = np.sqrt(a0); d = np.sqrt(a0)
    if random_mixing:
        rng = np.random.RandomState(seed)
        Q, _ = np.linalg.qr(rng.randn(N2, N2))
        R = Q
    else:
        R = np.eye(N2)
    W21 = np.zeros((N2, Vt.shape[1]))
    for a in range(min(K, N2)):
        W21[a, :] = c[a] * Vt[a, :]
    W21 = R @ W21
    W32 = np.zeros((U.shape[0], N2))
    for a in range(min(K, N2)):
        W32[:, a] = d[a] * U[:, a]
    W32 = W32 @ R.T
    return W21, W32

def init_imbalanced_layers(Sigma_yx, N2, imbalanced_modes=(0,),
                           imbalance_factor=5.0, base_factor=0.01,
                           random_mixing=False, seed=None):
    U, S_vals, Vt = np.linalg.svd(Sigma_yx, full_matrices=False)
    K = len(S_vals)
    a0_base = base_factor * S_vals
    c = np.sqrt(a0_base)
    d = np.sqrt(a0_base)
    for a in imbalanced_modes:
        c[a] = np.sqrt(a0_base[a]) * imbalance_factor
        d[a] = np.sqrt(a0_base[a]) / imbalance_factor
    if random_mixing:
        rng = np.random.RandomState(seed)
        Q, _ = np.linalg.qr(rng.randn(N2, N2))
        R = Q
    else:
        R = np.eye(N2)
    W21 = np.zeros((N2, Vt.shape[1]))
    for a in range(min(K, N2)):
        W21[a, :] = c[a] * Vt[a, :]
    W21 = R @ W21
    W32 = np.zeros((U.shape[0], N2))
    for a in range(min(K, N2)):
        W32[:, a] = d[a] * U[:, a]
    W32 = W32 @ R.T
    return W21, W32

def init_offdiag_coupling(Sigma_yx, N2, coupled_pairs=((0,1),),
                          coupling_angle=np.pi/6, base_scale=1e-2, seed=0):
    rng = np.random.RandomState(seed)
    R = np.eye(N2)
    for (i, j) in coupled_pairs:
        if 0 <= i < N2 and 0 <= j < N2 and i != j:
            G = np.eye(N2)
            c = np.cos(coupling_angle); s = np.sin(coupling_angle)
            G[i, i] = c; G[j, j] = c
            G[i, j] = -s; G[j, i] = s
            R = R @ G
    W21_base = rng.randn(N2, Sigma_yx.shape[1]) * base_scale
    W32_base = rng.randn(Sigma_yx.shape[0], N2) * base_scale
    W21 = R @ W21_base
    W32 = W32_base @ R.T
    return W21, W32

def build_init(name, Sigma_yx, N1, N2, N3, seed=0, **kwargs):
    rng = np.random.RandomState(seed)
    if name == "gauss":
        scale = kwargs.get("scale", 1e-2)
        return init_gauss(N1, N2, N3, scale, rng)
    if name == "boost":
        return init_boost_weaker_mode(
            Sigma_yx, N2,
            weaker_mode_idx=kwargs.get("weaker_mode_idx", 1),
            boost_factor=kwargs.get("boost_factor", 0.1),
            base_factor=kwargs.get("base_factor", 0.01),
            random_mixing=kwargs.get("random_mixing", False),
            seed=seed
        )
    if name == "imbalanced":
        return init_imbalanced_layers(
            Sigma_yx, N2,
            imbalanced_modes=tuple(kwargs.get("imbalanced_modes", [0])),
            imbalance_factor=kwargs.get("imbalance_factor", 5.0),
            base_factor=kwargs.get("base_factor", 0.01),
            random_mixing=kwargs.get("random_mixing", False),
            seed=seed
        )
    if name == "offdiag":
        return init_offdiag_coupling(
            Sigma_yx, N2,
            coupled_pairs=tuple(tuple(p) for p in kwargs.get("coupled_pairs", [(0,1)])),
            coupling_angle=kwargs.get("coupling_angle", np.pi/6),
            base_scale=kwargs.get("base_scale", 1e-2),
            seed=seed
        )
    raise ValueError(f"Unknown init name: {name}")
'''
def build_init(name, Sigma_yx, N1, N2, N3, seed=0, **kwargs):
    """
    Two-layer initialisations.
    name ∈ {"gauss-small","gauss-big","uniform","diag","lowrank",
            "gauss","boost","imbalanced","offdiag"}
    """
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