import numpy as np


def init_weights(N1, N2, N3, init='random', scale=0.01, random_seed=None):
    """
    Initialize weight matrices for a three-layer linear network.

    Parameters
    ----------
    N1, N2, N3 : int
        Layer dimensions.
    init : {'random', 'zeros'}
        Initialization scheme.
    scale : float
        Standard deviation for random init.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    W21 : ndarray, shape (N2, N1)
    W32 : ndarray, shape (N3, N2)
    """
    rng = np.random.RandomState(random_seed)
    if init == 'random':
        W21 = scale * rng.randn(N2, N1)
        W32 = scale * rng.randn(N3, N2)
    elif init == 'zeros':
        W21 = np.zeros((N2, N1))
        W32 = np.zeros((N3, N2))
    else:
        raise ValueError("init must be 'random' or 'zeros'")
    return W21, W32


def batch_gradient_update(W21, W32, x, y, lr):
    """
    Perform one discrete-time batch gradient descent update:

    ΔW21 = lr * W32^T (Σ31 - W32 W21 Σ11)
    ΔW32 = lr * (Σ31 - W32 W21 Σ11) W21^T

    Parameters
    ----------
    W21 : ndarray, shape (N2, N1)
    W32 : ndarray, shape (N3, N2)
    x : ndarray, shape (P, N1)
    y : ndarray, shape (P, N3)
    lr : float
        Learning rate.

    Returns
    -------
    W21_updated : ndarray, shape (N2, N1)
    W32_updated : ndarray, shape (N3, N2)
    """
    # Compute correlations
    Sigma11 = x.T.dot(x)      # (N1, N1)
    Sigma31 = y.T.dot(x)      # (N3, N1)
    # Residual term
    residual = Sigma31 - W32.dot(W21).dot(Sigma11)  # (N3, N1)
    # Updates
    dW21 = lr * W32.T.dot(residual)                # (N2, N1)
    dW32 = lr * residual.dot(W21.T)                # (N3, N2)
    # Apply updates
    W21 += dW21
    W32 += dW32
    return W21, W32


def continuous_dynamics(W21, W32, Sigma11, Sigma31, tau):
    """
    Compute time derivatives for continuous-time gradient flow:

    τ * dW21/dt = W32^T (Σ31 - W32 W21 Σ11)
    τ * dW32/dt = (Σ31 - W32 W21 Σ11) W21^T

    Parameters
    ----------
    W21 : ndarray, shape (N2, N1)
    W32 : ndarray, shape (N3, N2)
    Sigma11 : ndarray, shape (N1, N1)
    Sigma31 : ndarray, shape (N3, N1)
    tau : float
        Time constant.

    Returns
    -------
    dW21_dt : ndarray, shape (N2, N1)
    dW32_dt : ndarray, shape (N3, N2)
    """
    resid = Sigma31 - W32.dot(W21).dot(Sigma11)
    dW21_dt = (1.0 / tau) * W32.T.dot(resid)
    dW32_dt = (1.0 / tau) * resid.dot(W21.T)
    return dW21_dt, dW32_dt
