from __future__ import annotations
import numpy as np
from numpy.linalg import svd, qr
import matplotlib.pyplot as plt

def gen_inputs_whitened(N1: int, P: int, *, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    M = np.random.randn(N1, P)
    cov = M @ M.T / P
    U, s, _ = svd(cov)
    X = (U @ np.diag(1.0 / np.sqrt(s)) @ U.T) @ M
    return X


def random_orthogonal_matrix(n: int, *, seed: int | None = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    Q, _ = qr(np.random.randn(n, n))
    return Q


def generate_outputs(
    X: np.ndarray, s_vals: np.ndarray, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N1, _ = X.shape
    N3 = len(s_vals)
    U = random_orthogonal_matrix(N3, seed=seed)
    V = random_orthogonal_matrix(N1, seed=None if seed is None else seed + 1)
    S = np.zeros((N3, N1))
    diag_len = min(N3, N1, len(s_vals))
    S[np.arange(diag_len), np.arange(diag_len)] = s_vals[:diag_len]
    Sigma31 = U @ S @ V.T
    Y = Sigma31 @ X
    return Y, Sigma31, U, V

def init_weights(
    N1: int,
    N2: int,
    N3: int,
    *,
    scale: float = 5e-2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    def _log_normal(shape):
        return np.exp(scale * np.random.randn(*shape))

    sign21 = np.random.choice([-1.0, 1.0], size=(N2, N1))
    sign32 = np.random.choice([-1.0, 1.0], size=(N3, N2))
    W21 = sign21 * _log_normal((N2, N1))
    W32 = sign32 * _log_normal((N3, N2))
    return W21, W32

def eg_step(W: np.ndarray, grad: np.ndarray, eta: float, eps: float = 1e-30) -> np.ndarray:
    update_factor = np.exp(-eta * grad * np.sign(W))
    W_new = W * update_factor
    pos = W_new > 0
    W_new[pos] = np.maximum(W_new[pos], eps)
    W_new[~pos] = np.minimum(W_new[~pos], -eps)
    return W_new

def simulate_eg(
    W21: np.ndarray,
    W32: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    eta: float = 0.05,
    n_steps: int = 3000,
    record_every: int = 20,
) -> tuple[np.ndarray, np.ndarray]:

    N2, N1 = W21.shape
    N3, _ = W32.shape
    Sigma11 = X @ X.T / X.shape[1]  
    Sigma31 = Y @ X.T / X.shape[1]
    size = N2 * N1 + N3 * N2

    t_list, W_hist = [], np.empty((size, (n_steps // record_every) + 1))
    write_idx = 0

    def _write(step):
        nonlocal write_idx
        W_hist[:, write_idx] = np.concatenate([W21.ravel(), W32.ravel()])
        t_list.append(step)
        write_idx += 1

    _write(0)

    for step in range(1, n_steps + 1):
        Delta = Sigma31 - W32 @ W21 @ Sigma11

        grad21 = -(W32.T @ Delta)
        grad32 = -(Delta @ W21.T)

        W21 = eg_step(W21, grad21, eta)
        W32 = eg_step(W32, grad32, eta)
        if step % record_every == 0:
            _write(step)

    return np.array(t_list), W_hist[:, :write_idx]

def extract_modes(W21: np.ndarray, W32: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    A = W21 @ V
    B = U.T @ W32
    r = min(A.shape[1], B.shape[0])
    return np.array([A[:, i].dot(B[i, :]) for i in range(r)])

if __name__ == "__main__":
    N1 = N2 = N3 = 5
    P = 200
    s_vals = np.array([2.5, 2.0, 1.5, 1.0, 0.5])

    # Hyper‑parameters for EG
    ETA = 0.001 
    STEPS = 400
    EVERY = 20 

    seed = 42
    X = gen_inputs_whitened(N1, P, seed=seed)
    Y, Sigma31, U, V = generate_outputs(X, s_vals, seed=seed)
    W21, W32 = init_weights(N1, N2, N3, seed=seed)

    # Run training under EG
    t_idx, W_hist = simulate_eg(
        W21, W32, X, Y, eta=ETA, n_steps=STEPS, record_every=EVERY
    )

    # Plot mode strengths over time 
    n_snap = t_idx.size
    u_hist = np.zeros((len(s_vals), n_snap))
    size21 = N2 * N1
    for k in range(n_snap):
        W21_k = W_hist[:size21, k].reshape(N2, N1)
        W32_k = W_hist[size21:, k].reshape(N3, N2)
        u_hist[:, k] = extract_modes(W21_k, W32_k, U, V)

    '''
    plt.figure(figsize=(8, 4))
    for i, s in enumerate(s_vals):
        plt.plot(t_idx, u_hist[i], label=f"mode {i+1} (s={s})")
    plt.xlabel("training step")
    plt.ylabel("uₐ = aₐ·bₐ")
    plt.title("Mode strengths under EG")
    plt.legend()
    plt.grid()
    '''

    plt.figure(figsize=(8, 4))
    for i, s in enumerate(s_vals):
        plt.semilogy(t_idx, np.abs(u_hist[i]),    
                    label=f"mode {i+1} (s={s})")
    plt.xlabel("training step")
    plt.ylabel("|u_a| on log scale")
    plt.title("Mode strengths (log-scale y-axis)")
    plt.legend(); plt.grid(which="both"); plt.tight_layout()


    # Singular values at convergence vs target spectrum
    W21_f = W_hist[:size21, -1].reshape(N2, N1)
    W32_f = W_hist[size21:, -1].reshape(N3, N2)
    s_learned = svd(W32_f @ W21_f, compute_uv=False)
    idx = np.arange(len(s_vals))
    plt.figure(figsize=(6, 4))
    plt.bar(idx - 0.15, s_vals, width=0.3, label="target $s_alpha$")
    plt.bar(idx + 0.15, s_learned, width=0.3, label="learned")
    plt.xlabel("mode index")
    plt.ylabel("singular value")
    plt.title("Top-N_2 singular values captured (EG)")
    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()



