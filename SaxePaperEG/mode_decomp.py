import numpy as np


def svd_decompose(Sigma31, full_matrices=False):
    """
    Compute the singular value decomposition of the input-output correlation matrix Sigma31.

    Parameters
    ----------
    Sigma31 : ndarray, shape (N3, N1)
        The cross-correlation matrix \Sigma_{31}.
    full_matrices : bool
        Whether to compute the full-sized U and Vh.

    Returns
    -------
    U : ndarray, shape (N3, r)
        Left singular vectors.
    s : ndarray, shape (r,)
        Singular values.
    V : ndarray, shape (N1, r)
        Right singular vectors.
    """
    U, s, Vh = np.linalg.svd(Sigma31, full_matrices=full_matrices)
    # Vh shape (r, N1) -> transpose to get V of shape (N1, r)
    V = Vh.T
    return U, s, V


def rotate_weights(W21, W32, U, V):
    """
    Rotate weight matrices into the singular-vector coordinates defined by U and V.

    Given three-layer weights W21 (N2 x N1) and W32 (N3 x N2), along with
    U (N3 x r) and V (N1 x r) from the SVD of Sigma31, compute:

      W21_tilde = W21 @ V    # shape (N2, r)
      W32_tilde = U.T @ W32  # shape (r, N2)

    Parameters
    ----------
    W21 : ndarray, shape (N2, N1)
    W32 : ndarray, shape (N3, N2)
    U : ndarray, shape (N3, r)
    V : ndarray, shape (N1, r)

    Returns
    -------
    W21_tilde : ndarray, shape (N2, r)
    W32_tilde : ndarray, shape (r, N2)
    """
    W21_tilde = W21.dot(V)
    W32_tilde = U.T.dot(W32)
    return W21_tilde, W32_tilde
