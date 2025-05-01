import numpy as np
from scipy.integrate import odeint


def modes_deriv(state, t, s, tau, r, N2):
    """
    Compute derivatives for the coupled mode ODEs (Eq. 5) in a three-layer linear network.

    Parameters
    ----------
    state : ndarray, shape (2 * r * N2,)
        Flattened vector containing [a(0), b(0)] for all modes.
    t : float
        Current time (unused, dynamics are autonomous).
    s : ndarray, shape (r,)
        Singular values for each mode.
    tau : float
        Time constant.
    r : int
        Number of modes.
    N2 : int
        Hidden-layer dimension.

    Returns
    -------
    deriv : ndarray, shape (2 * r * N2,)
        Flattened derivatives [d a / d t, d b / d t].
    """
    # Unpack state
    a = state[:r * N2].reshape((r, N2))  # each row a_alpha
    b = state[r * N2:].reshape((r, N2))  # each row b_alpha
    da = np.zeros_like(a)
    db = np.zeros_like(b)

    # Compute mode-wise derivatives
    for alpha in range(r):
        a_alpha = a[alpha]
        b_alpha = b[alpha]
        # Interaction for mode alpha
        self_dot = np.dot(a_alpha, b_alpha)
        da_alpha = (s[alpha] - self_dot) * b_alpha.copy()
        db_alpha = (s[alpha] - self_dot) * a_alpha.copy()
        # Cross-mode interactions
        for beta in range(r):
            if beta == alpha:
                continue
            # a_alpha · b_beta
            apb = np.dot(a_alpha, b[beta])
            da_alpha -= apb * b[beta]
            # b_alpha · a_beta
            bpa = np.dot(b_alpha, a[beta])
            db_alpha -= bpa * a[beta]
        # Scale by tau
        da[alpha] = da_alpha / tau
        db[alpha] = db_alpha / tau

    return np.concatenate([da.flatten(), db.flatten()])


def ode_modes(a0, b0, s, tau, t):
    """
    Integrate the full vector ODEs for all modes over time.

    Parameters
    ----------
    a0 : ndarray, shape (r, N2)
        Initial a vectors for each mode.
    b0 : ndarray, shape (r, N2)
        Initial b vectors for each mode.
    s : ndarray, shape (r,)
        Singular values for each mode.
    tau : float
        Time constant.
    t : array_like
        1D array of time points at which to solve.

    Returns
    -------
    a_traj : ndarray, shape (len(t), r, N2)
        Time trajectories of a vectors.
    b_traj : ndarray, shape (len(t), r, N2)
        Time trajectories of b vectors.
    """
    r, N2 = a0.shape
    # Initial state
    state0 = np.concatenate([a0.flatten(), b0.flatten()])
    # Integrate ODE
    sol = odeint(modes_deriv, state0, t, args=(s, tau, r, N2))
    # Reshape solution
    a_traj = sol[:, :r * N2].reshape((len(t), r, N2))
    b_traj = sol[:, r * N2:].reshape((len(t), r, N2))
    return a_traj, b_traj


def solve_u(u0, s, tau, t):
    """
    Analytical solution of \tau \dot u = 2 u (s - u), with u(0)=u0 (Eq. 12).

    u(t) = \frac{s}{1 + (s/u0 - 1) e^{-2 s t / \tau}}

    Parameters
    ----------
    u0 : float
        Initial product a·b at t=0.
    s : float or array_like
        Singular value(s) for the mode(s).
    tau : float
        Time constant.
    t : array_like
        Time points at which to compute u(t).

    Returns
    -------
    u_t : ndarray, shape (len(t),) or matching shape of s
        Analytical u values over time.
    """
    t = np.array(t, ndmin=1)
    # Prevent division by zero
    ratio = (s / u0) - 1.0
    exp_term = np.exp(-2.0 * s * t / tau)
    u_t = s / (1.0 + ratio * exp_term)
    return u_t
