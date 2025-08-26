import numpy as np

'''
def build_init(init_id, Sigma_yx, rng, N3, N1):
    sign_S = np.sign(Sigma_yx)
    sign_S[sign_S == 0] = 1

    if init_id == 1:
        # Standard random Gaussian small magnitude
        scale = 1e-2
        M = np.abs(rng.randn(N3, N1) * scale)
        return sign_S * M
    elif init_id == 2:
        # Random magnitudes 
        scale = 1e-2
        M = np.abs(rng.randn(N3, N1) * scale)
        return M
    elif init_id == 3:
        # Teacher scaled init
        scale = 0.1
        M = np.abs(Sigma_yx) * scale
        return sign_S * M
    elif init_id == 4:
        # Top mode (rank‑1 SVD) scaled init with sign match
        scale = 0.1
        U, S_vals, Vt = svd(Sigma_yx, full_matrices=False)
        W_mode = S_vals[0] * np.outer(U[:, 0], Vt[0, :])
        M = np.abs(W_mode) * scale
        return sign_S * M
    elif init_id == 5:
        # Nearly zero init
        eps = 1e-6
        M = np.ones((N3, N1)) * eps
        return sign_S * M
    elif init_id == 6:
        # Block‑out first singular mode
        scale = 1e-2
        U, S_vals, Vt = svd(Sigma_yx, full_matrices=False)
        Sigma_minus1 = Sigma_yx - S_vals[0] * np.outer(U[:,0], Vt[0,:])
        M = np.abs(Sigma_minus1) * scale
        return sign_S * M
    elif init_id == 7:
        # Bias W0 toward the *smaller* singular mode (rank-1),
        # so GD starts with a larger projection on s2 than on s1.
        U, S_vals, Vt = svd(Sigma_yx, full_matrices=False)
        # Take the *second* singular triple (assuming s1 >= s2)
        scale = 0.2
        W0 = scale * (S_vals[1] * np.outer(U[:, 1], Vt[1, :]))
        # Respect Dale sign-locking used elsewhere (positive/negative per entry)
        sign_S = np.sign(Sigma_yx); sign_S[sign_S == 0] = 1
        return sign_S * np.abs(W0)

    else:
        raise ValueError("init must be in {1,...,6}")
'''

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