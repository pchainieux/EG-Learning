import numpy as np

def build_init(init_id, Sigma_yx, rng, N3, N1):
    sign_S = np.sign(Sigma_yx)
    sign_S[sign_S == 0] = 1

    if init_id == 1:
        # Small Gaussian
        scale = 1e-2
        M = np.abs(rng.randn(N3, N1) * scale)
        return sign_S * M

    elif init_id == 2:
        # Big Gaussian
        scale = 0.5
        M = np.abs(rng.randn(N3, N1) * scale)
        return sign_S * M

    elif init_id == 3:
        # Uniform in [-scale, scale], small scale
        scale = 1e-2
        M = np.abs(rng.uniform(low=-scale, high=scale, size=(N3, N1)))
        return sign_S * M

    elif init_id == 4:
        # Diagonal initialisation (scaled identity, sign matched)
        scale = 0.1
        M = np.zeros((N3, N1))
        for i in range(min(N3, N1)):
            M[i, i] = scale
        return sign_S * np.abs(M)

    elif init_id == 5:
        # Sparse / Low rank: rank 1 outer product
        scale = 0.1
        u = rng.randn(N3, 1)
        v = rng.randn(1, N1)
        W_lowrank = scale * (u @ v)
        M = np.abs(W_lowrank)
        return sign_S * M

    else:
        raise ValueError("init_id must be in {1,2,3,4,5}")