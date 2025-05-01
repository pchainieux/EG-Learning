import numpy as np
from scipy.integrate import odeint


def random_orthonormal(N, r, seed=None):
    """
    Generate an N x r random orthonormal matrix.
    """
    rng = np.random.RandomState(seed)
    M = rng.randn(N, r)
    Q, _ = np.linalg.qr(M)
    return Q


def integrate_ode(func, state0, t, args=()):
    """
    Integrate an ODE: d state / dt = func(state, t, *args).

    Parameters
    ----------
    func : callable
        ODE derivative function.
    state0 : array_like
        Initial state.
    t : array_like
        Time grid.
    args : tuple
        Additional args to func.

    Returns
    -------
    sol : ndarray
        Solution array of shape (len(t), len(state0)).
    """
    return odeint(func, state0, t, args=args)


def svd(matrix):
    """
    Compute reduced SVD: U, s, V for a given matrix.
    """
    return np.linalg.svd(matrix, full_matrices=False)
