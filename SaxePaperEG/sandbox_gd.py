import numpy as np
from numpy.linalg import svd, qr
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def gen_inputs_whitened(N1, P, seed=None):
    if seed is not None:
        np.random.seed(seed)
    M = np.random.randn(N1, P)
    cov = M @ M.T / P
    E, D, _ = svd(cov)
    whitening = E @ np.diag(1.0 / np.sqrt(D)) @ E.T
    return whitening @ M


def random_orthogonal_matrix(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    H = np.random.randn(n, n)
    Q, _ = qr(H)
    return Q


def generate_outputs(X, s_vals, seed=None):
    N1, P = X.shape
    N3 = len(s_vals)
    if seed is not None:
        np.random.seed(seed + 1)
    U = random_orthogonal_matrix(N3, seed=seed+2)
    V = random_orthogonal_matrix(N1, seed=seed+3)
    S_mat = np.zeros((N3, N1))
    for i, s in enumerate(s_vals):
        if i < min(N3, N1):
            S_mat[i, i] = s
    Sigma31 = U @ S_mat @ V.T
    Y = Sigma31 @ X
    return Y, Sigma31, U, V

def init_weights(N1, N2, N3, init='random', scale=1e-3, seed=None):
    if seed is not None:
        np.random.seed(seed + 4)
    if init == 'random':
        W21 = np.random.randn(N2, N1) * scale
        W32 = np.random.randn(N3, N2) * scale
    elif init == 'orthogonal':
        U21 = random_orthogonal_matrix(N2, seed=seed+5)
        V21 = random_orthogonal_matrix(N1, seed=seed+6)
        W21 = U21[:, :min(N2,N1)] @ V21[:min(N2,N1), :]
        U32 = random_orthogonal_matrix(N3, seed=seed+7)
        V32 = random_orthogonal_matrix(N2, seed=seed+8)
        W32 = U32[:, :min(N3,N2)] @ V32[:min(N3,N2), :]
    else:
        raise ValueError(f"Unknown init '{init}'")
    return W21, W32

def rhs_continuous(t, W_flat, dims, Sigma11, Sigma31, tau):
    N1, N2, N3 = dims
    size = N2 * N1
    W21 = W_flat[:size].reshape(N2, N1)
    W32 = W_flat[size:].reshape(N3, N2)
    Delta = Sigma31 - W32 @ W21 @ Sigma11
    dW21 = (W32.T @ Delta) / tau
    dW32 = (Delta @ W21.T) / tau
    return np.concatenate([dW21.ravel(), dW32.ravel()])


def simulate(W21_init, W32_init, X, Y, tau, t_span, n_eval=200):
    N2, N1 = W21_init.shape
    N3, _ = W32_init.shape
    Sigma11 = X @ X.T / X.shape[1]
    Sigma31 = Y @ X.T / X.shape[1]
    dims = (N1, N2, N3)
    W0 = np.concatenate([W21_init.ravel(), W32_init.ravel()])
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(rhs_continuous, t_span, W0,
                    args=(dims, Sigma11, Sigma31, tau),
                    t_eval=t_eval, method='RK45')
    return sol.t, sol.y

def extract_modes(W21, W32, U, V, s_vals):
    A = W21 @ V
    B = U.T @ W32
    return np.array([A[:, i].dot(B[i, :]) for i in range(len(s_vals))])


def analytical_u(u0, s, tau, t):
    exp_term = np.exp(2 * s * t / tau)
    return (s * exp_term) / (exp_term - 1 + s / u0)

def plot_phase_portrait(s, tau, lim=6.0, grid=25, with_traj=True):
    a = np.linspace(0.1, lim, grid)
    b = np.linspace(0.1, lim, grid)
    A, B = np.meshgrid(a, b)

    dA = B*(s - A*B) / tau
    dB = A*(s - A*B) / tau

    plt.figure(figsize=(4,4))

    speed = np.hypot(dA, dB)
    plt.quiver(A, B, dA, dB, speed, cmap='gray_r', width=0.002)


    a_curve = np.linspace(0.1, lim, 400)
    plt.plot(a_curve, s/a_curve, 'r--', lw=2, label=r'$ab=s$')


    if with_traj:
        def rhs(t, z): 
            a, b = z
            return [b*(s-a*b)/tau, a*(s-a*b)/tau]
        sol = solve_ivp(rhs, [0, 8*tau/s], [0.2, 0.2], max_step=0.05)
        plt.plot(sol.y[0], sol.y[1], 'b', lw=2)

    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel(r'$a$'); plt.ylabel(r'$b$')
    plt.title(f'Phase portrait, mode strength $s={s}$')
    plt.legend(); plt.grid()

def numeric_test_fixed_points(runs, params):
    errs = []
    N1, N2, N3, P, s_vals, tau, t_span = params
    for seed in range(runs):
        X = gen_inputs_whitened(N1, P, seed)
        Y, _, U, V = generate_outputs(X, s_vals, seed=seed)
        W21, W32 = init_weights(N1, N2, N3, seed=seed)
        t, W_hist = simulate(W21, W32, X, Y, tau, t_span)
        size = N2 * N1
        W21_f = W_hist[:size, -1].reshape(N2, N1)
        W32_f = W_hist[size:, -1].reshape(N3, N2)
        S_hat = W32_f @ W21_f
        s_hat = svd(S_hat, compute_uv=False)
        errs.append(np.linalg.norm(s_hat[:N2] - s_vals[:N2]))
    print(f"Fixed-point error over {runs} runs: mean={np.mean(errs):.3e}, std={np.std(errs):.3e}")

if __name__ == '__main__':
    N1 = N2 = N3 = 5
    P = 200
    s_vals = np.array([2.5, 2.0, 1.5, 1.0, 0.5])
    tau = 1.0
    t_span = (0.0, 8.0)
    seed = 42

    # simulate once and compare to theory
    X = gen_inputs_whitened(N1, P, seed)
    Y, Sigma31, U, V = generate_outputs(X, s_vals, seed=seed)
    W21, W32 = init_weights(N1, N2, N3, seed=seed)
    t, W_hist = simulate(W21, W32, X, Y, tau, t_span)

    # plot mode strengths
    plt.figure()
    for i, s in enumerate(s_vals):
        u_sim = extract_modes(W_hist[:N2*N1, :].reshape(N2, N1, -1).transpose(2,0,1),
                              W_hist[N2*N1:, :].reshape(N3, N2, -1).transpose(2,0,1), U, V, s_vals)[i]
    for i, s in enumerate(s_vals):
        u_sim = np.array([extract_modes(W_hist[:N2*N1, k].reshape(N2, N1),
                                        W_hist[N2*N1:, k].reshape(N3, N2), U, V, s_vals)[i]
                          for k in range(len(t))])
        u0 = max(u_sim[0], 1e-6)
        plt.plot(t, u_sim, label=f"mode {i+1} sim")
        plt.plot(t, analytical_u(u0, s, tau, t), '--', label=f"mode {i+1} theory")
    plt.xlabel('time'); plt.ylabel('u(t)'); plt.title('Mode strengths'); plt.legend(); plt.grid()

    # plot singular values at fixed point
    W21_f = W_hist[:N2*N1, -1].reshape(N2, N1)
    W32_f = W_hist[N2*N1:, -1].reshape(N3, N2)
    s_hat = svd(W32_f@W21_f, compute_uv=False)
    plt.figure()
    idx = np.arange(len(s_vals))
    plt.bar(idx, s_vals, alpha=0.5, label='target')
    plt.bar(idx, s_hat, alpha=0.5, label='learned')
    plt.xlabel('mode index'); plt.ylabel('singular value'); plt.title('Captured singulars'); plt.legend(); plt.grid()

    # phase portrait for strongest mode
    plot_phase_portrait(s_vals[0], tau, lim=3.0, grid=25)

    params = (N1, N2, N3, P, s_vals, tau, t_span)
    numeric_test_fixed_points(10, params)

    plt.show()
