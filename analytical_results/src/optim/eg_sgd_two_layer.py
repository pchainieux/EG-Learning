import numpy as np

def eg_step(W, grad, eta, eps=1e-12, clip=5.0):
    z = -eta * grad * np.sign(W)
    z = np.clip(z, -clip, clip)
    W_new = W * np.exp(z)
    pos = W_new > 0
    W_new[pos]  = np.maximum(W_new[pos],  eps)
    W_new[~pos] = np.minimum(W_new[~pos], -eps)
    return W_new

def simulate_EG_two_layer(W21_init, W32_init, Sigma_x, Sigma_yx,
                          eta=1e-3, n_steps=1000, record_every=10):
    W21 = W21_init.copy()
    W32 = W32_init.copy()
    N2, N1 = W21.shape
    N3, _  = W32.shape
    n_rec = (n_steps // record_every) + 1
    size  = N2*N1 + N3*N2
    W_hist = np.zeros((size, n_rec))
    t_list = []
    idx = 0
    def record(t):
        nonlocal idx
        W_hist[:N2*N1, idx] = W21.ravel()
        W_hist[N2*N1:, idx] = W32.ravel()
        t_list.append(t)
        idx += 1
    record(0)
    for t in range(1, n_steps+1):
        Delta  = Sigma_yx - W32 @ W21 @ Sigma_x
        grad21 = -(W32.T @ Delta)
        grad32 = -(Delta @ W21.T)
        W21 = eg_step(W21, grad21 + 1e-4*W21, eta)
        W32 = eg_step(W32, grad32 + 1e-4*W32, eta)
        if t % record_every == 0:
            record(t)
    return np.array(t_list), W_hist[:, :idx]

def simulate_GD_two_layer(W21_init, W32_init, Sigma_x, Sigma_yx,
                          eta=1e-3, n_steps=1000, record_every=10):
    W21 = W21_init.copy()
    W32 = W32_init.copy()
    N2, N1 = W21.shape
    N3, _  = W32.shape
    n_rec = (n_steps // record_every) + 1
    size  = N2*N1 + N3*N2
    W_hist = np.zeros((size, n_rec))
    t_list = []
    idx = 0
    def record(t):
        nonlocal idx
        W_hist[:N2*N1, idx] = W21.ravel()
        W_hist[N2*N1:, idx] = W32.ravel()
        t_list.append(t)
        idx += 1
    record(0)
    for t in range(1, n_steps+1):
        Delta  = Sigma_yx - W32 @ W21 @ Sigma_x
        grad21 =  (W32.T @ Delta)
        grad32 =  (Delta @ W21.T)
        W21 = W21 + eta * grad21
        W32 = W32 + eta * grad32
        if t % record_every == 0:
            record(t)
    return np.array(t_list), W_hist[:, :idx]