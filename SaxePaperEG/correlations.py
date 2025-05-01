import numpy as np

def compute_Sigma11(x):
    """
    Compute the input-input correlation matrix \Sigma_{11} = \sum_mu x^mu x^{mu T}.

    Parameters
    ----------
    x : ndarray, shape (P, N1)
        Input samples, each row is x^mu in R^{N1}.

    Returns
    -------
    Sigma11 : ndarray, shape (N1, N1)
        The correlation matrix \Sigma_{11}.
    """
    # x: (P, N1) -> x^T @ x sums over samples
    return x.T @ x


def compute_Sigma31(x, y):
    """
    Compute the input-output correlation matrix \Sigma_{31} = \sum_mu y^mu x^{mu T}.

    Parameters
    ----------
    x : ndarray, shape (P, N1)
        Input samples, each row is x^mu.
    y : ndarray, shape (P, N3)
        Corresponding output samples, each row is y^mu.

    Returns
    -------
    Sigma31 : ndarray, shape (N3, N1)
        The cross-correlation matrix \Sigma_{31}.
    """
    # y: (P, N3), x: (P, N1) -> y^T @ x sums over samples
    return y.T @ x
