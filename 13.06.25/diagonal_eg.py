import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd


def whiten_data(X, eps=1e-12):
    X_centered = X - X.mean(axis=0, keepdims=True)
    P = X_centered.shape[0]

    Sigma = (X_centered.T @ X_centered) / P

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T
    return X_centered @ W


def compute_teacher_cov(X, Y):
    P = X.shape[0]
    Sigma_x = (X.T @ X) / P
    Sigma_yx = (Y.T @ X) / P
    return Sigma_x, Sigma_yx


def teacher_svd(Sigma_yx):
    U, S, Vt = svd(Sigma_yx, full_matrices=False)
    V = Vt.T
    return U, S, V


def simulate_diagonal_EG(s_vals, eta=1e-2, n_steps=1000, rec_every=10, x0=None, y0=None):
    """
    Exponentiated Gradient on diagonal parameters x_i, y_i.
    """
    K = len(s_vals)
    x = x0.copy() if x0 is not None else np.full(K, 1e-3)
    y = y0.copy() if y0 is not None else np.full(K, 1e-3)
    steps = (n_steps // rec_every) + 1
    x_hist = np.zeros((K, steps))
    y_hist = np.zeros((K, steps))
    t_list = []
    idx = 0
    def record(t):
        nonlocal idx
        x_hist[:, idx] = x
        y_hist[:, idx] = y
        t_list.append(t)
        idx += 1
    record(0)
    for t in range(1, n_steps+1):
        grad_x = x * y**2 - s_vals * y
        grad_y = y * x**2 - s_vals * x
        x = x * np.exp(-eta * grad_x * np.sign(x))
        y = y * np.exp(-eta * grad_y * np.sign(y))
        if t % rec_every == 0:
            record(t)
    return np.array(t_list), x_hist[:, :idx], y_hist[:, :idx]


def compute_loss(a_vals, s_vals):
    return 0.5 * np.sum(a_vals**2 - 2 * s_vals * a_vals)


def plot_mode_strengths(t_list, a_hist, s_vals):
    K = len(s_vals)
    plt.figure(figsize=(8,5))
    for i in range(K):
        plt.plot(t_list, a_hist[i], label=f"mode {i+1}")
        plt.hlines(s_vals[i], t_list[0], t_list[-1], colors='k', linestyles='dashed')
    plt.xlabel('Step')
    plt.ylabel('Mode strength $a_i$')
    plt.title('Mode strengths vs teacher singular values')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_xy_trajectories(t_list, x_hist, y_hist):
    K = x_hist.shape[0]
    plt.figure(figsize=(8,5))
    for i in range(K):
        plt.plot(t_list, x_hist[i], '-', label=f"x_{i+1}")
        plt.plot(t_list, y_hist[i], '--', label=f"y_{i+1}")
    plt.xlabel('Step')
    plt.ylabel('Parameter value')
    plt.title('x_i and y_i trajectories')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def plot_loss(t_list, a_hist, s_vals):
    losses = [compute_loss(a_hist[:,j], s_vals) for j in range(a_hist.shape[1])]
    plt.figure(figsize=(6,4))
    plt.plot(t_list, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_basis_alignment(t_list, x_hist, y_hist, U_tchr, V_tchr):
    T = len(t_list)
    K = len(U_tchr)
    # reconstruct W_prod(t) and SVD
    cos_u = np.zeros((T, K))
    cos_v = np.zeros((T, K))
    for idx in range(T):
        a = x_hist[:,idx] * y_hist[:,idx]
        W = U_tchr @ np.diag(a) @ V_tchr.T
        U_net, _, Vt_net = svd(W, full_matrices=False)
        V_net = Vt_net.T
        cos_u[idx] = np.abs(np.diag(U_net.T @ U_tchr))
        cos_v[idx] = np.abs(np.diag(V_net.T @ V_tchr))
    plt.figure(figsize=(8,5))
    for i in range(K):
        plt.plot(t_list, cos_u[:,i], '-', label=f"EG u mode {i+1}")
        plt.plot(t_list, cos_v[:,i], '--', label=f"EG v mode {i+1}")
    plt.xlabel('Step')
    plt.ylabel(r'$|\cos|$')
    plt.title('Basis alignment under diagonal EG')
    plt.ylim(0,1.05)
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.show()


def main():
    # --- Load or generate your data here ---
    # X : shape (P, N1), Y : shape (P, N3)
    # e.g., X = np.load('X.npy'); Y = np.load('Y.npy')
    # For demonstration, generate synthetic:
    P, N1, N3 = 500, 10, 8
    X = np.random.randn(P, N1)
    M_true = np.random.randn(N3, N1)
    Y = X @ M_true.T + 0.01 * np.random.randn(P, N3)

    # Whiten data and compute teacher SVD
    X_white = whiten_data(X)
    Sigma_x, Sigma_yx = compute_teacher_cov(X_white, Y)
    U_tchr, S_tchr, V_tchr = teacher_svd(Sigma_yx)

    # Run diagonal EG on the true spectrum
    s_vals = S_tchr
    x0 = 0.1 * np.ones_like(s_vals)
    y0 = 0.1 * np.ones_like(s_vals)
    eta, n_steps, rec_every = 0.03, 2000, 50
    t_list, x_hist, y_hist = simulate_diagonal_EG(
        s_vals, eta=eta, n_steps=n_steps,
        rec_every=rec_every, x0=x0, y0=y0
    )

    # Diagnostics
    a_hist = x_hist * y_hist
    plot_mode_strengths(t_list, a_hist, s_vals)
    plot_xy_trajectories(t_list, x_hist, y_hist)
    plot_loss(t_list, a_hist, s_vals)
    plot_basis_alignment(t_list, x_hist, y_hist, U_tchr, V_tchr)

if __name__ == '__main__':
    main()


