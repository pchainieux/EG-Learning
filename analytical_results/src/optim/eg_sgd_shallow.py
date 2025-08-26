import numpy as np

def eg_step(W, grad, eta, eps=1e-30):
    update = np.exp(-eta * grad * np.sign(W))
    W_new = W * update
    pos = W_new > 0
    W_new[pos] = np.maximum(W_new[pos], eps)
    W_new[~pos] = np.minimum(W_new[~pos], -eps)
    return W_new

def simulate_EG_shallow(W_init, Sigma_x, Sigma_yx,
                        eta=1e-3, n_steps=1000, record_every=10):
    W = W_init.copy()
    N3, N1 = W.shape
    n_records = (n_steps // record_every) + 1
    W_hist = np.zeros((N3*N1, n_records))
    t_list = []
    idx = 0

    def record(t):
        nonlocal idx
        W_hist[:, idx] = W.ravel()
        t_list.append(t)
        idx += 1

    record(0)
    for t in range(1, n_steps+1):
        Delta = Sigma_yx - W @ Sigma_x
        grad = -Delta
        W = eg_step(W, grad, eta)
        if t % record_every == 0:
            record(t)

    return np.array(t_list), W_hist[:, :idx]

def simulate_GD_shallow(W_init, Sigma_x, Sigma_yx,
                        eta=1e-2, n_steps=1000, record_every=10):
    W = W_init.copy()
    N3, N1 = W.shape
    n_records = (n_steps // record_every) + 1
    W_hist = np.zeros((N3*N1, n_records))
    t_list = []
    idx = 0

    def record(t):
        nonlocal idx
        W_hist[:, idx] = W.ravel()
        t_list.append(t)
        idx += 1

    record(0)
    for t in range(1, n_steps+1):
        Delta = Sigma_yx - W @ Sigma_x
        W = W + eta * Delta
        if t % record_every == 0:
            record(t)

    return np.array(t_list), W_hist[:, :idx]

