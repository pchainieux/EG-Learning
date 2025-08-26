import numpy as np

def whiten_data(X):
    P, N1 = X.shape
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Sigma_x = (X_centered.T @ X_centered) / P
    eps = 1e-12
    eigvals, eigvecs = np.linalg.eigh(Sigma_x)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T
    return X_centered @ W

def compute_teacher_cov(X_white, Y):
    P = X_white.shape[0]
    Sigma_x = (X_white.T @ X_white) / P
    Sigma_yx = (Y.T @ X_white) / P
    return Sigma_x, Sigma_yx

# Two layer only
def get_Wprod_hist(W_hist, dims):
    N1, N2, N3 = dims
    s21, s32 = N2*N1, N3*N2
    T = W_hist.shape[1]
    Wprod = np.zeros((T, N3, N1))
    for k in range(T):
        W21_k = W_hist[:s21, k].reshape(N2, N1)
        W32_k = W_hist[s21:, k].reshape(N3, N2)
        Wprod[k] = W32_k @ W21_k
    return Wprod