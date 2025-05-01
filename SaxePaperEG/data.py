import numpy as np


def generate_orthogonal_inputs(N1, P, random_seed=None):
    """
    Generate P orthonormal input vectors in R^{N1}, so that \Sigma_{11} = sum_mu x^mu x^{mu T} = I.

    Parameters
    ----------
    N1 : int
        Dimensionality of each input vector.
    P : int
        Number of input samples; must satisfy P <= N1.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    x : ndarray, shape (P, N1)
        Array whose rows are orthonormal vectors in R^{N1}.

    Raises
    ------
    ValueError
        If P > N1.
    """
    if P > N1:
        raise ValueError(f"Cannot generate {P} orthonormal vectors in dimension {N1}. P must be <= N1.")
    rng = np.random.RandomState(random_seed)
    # Random matrix of shape (N1, P)
    M = rng.randn(N1, P)
    # QR decomposition: Q has orthonormal columns
    Q, _ = np.linalg.qr(M)
    # Transpose so that each row is an input vector x^mu
    x = Q.T
    return x


def generate_synthetic_outputs(x, N3, rank=None, singular_values=None, noise_std=0.0, random_seed=None):
    """
    Generate synthetic outputs y in R^{N3} for each input in x, such that
    \Sigma_{31} = sum_mu y^mu x^{mu T} has specified singular spectrum.

    If singular_values is provided, its length defines the effective rank.
    Otherwise, singular_values are chosen linearly between 1 and 0.1.

    Parameters
    ----------
    x : ndarray, shape (P, N1)
        Input samples; rows are input vectors x^mu.
    N3 : int
        Dimensionality of output vectors.
    rank : int or None
        Desired rank of the correlation matrix. Used if singular_values is None.
    singular_values : array-like or None
        Sequence of singular values of length r. If provided, overrides rank.
    noise_std : float
        Standard deviation of additive Gaussian noise on outputs.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    y : ndarray, shape (P, N3)
        Generated output vectors y^mu.
    """
    P, N1 = x.shape
    rng = np.random.RandomState(random_seed)

    # Set up singular values
    if singular_values is not None:
        s = np.array(singular_values, dtype=float)
    else:
        if rank is None:
            rank = min(N1, N3)
        # default singular values: linear spacing from 1.0 down to 0.1
        s = np.linspace(1.0, 0.1, rank)
    r = len(s)

    # Random orthonormal bases U (N3 x r) and V (N1 x r)
    U_rand = rng.randn(N3, r)
    U, _ = np.linalg.qr(U_rand)
    V_rand = rng.randn(N1, r)
    V, _ = np.linalg.qr(V_rand)

    # Build outputs: Y = X @ V @ diag(s) @ U^T
    Y = x.dot(V).dot(np.diag(s)).dot(U.T)

    # Add isotropic Gaussian noise if requested
    if noise_std > 0:
        Y += noise_std * rng.randn(*Y.shape)

    return Y
