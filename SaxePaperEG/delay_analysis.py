import numpy as np
import matplotlib.pyplot as plt

def compute_half_times(u_sim, u_anal, t=None):
    """
    Compute times when each mode reaches half its asymptotic value.

    Parameters
    ----------
    u_sim : ndarray, shape (T, r)
        Simulated mode-strength trajectories.
    u_anal : ndarray, shape (T, r)
        Analytical mode-strength trajectories.
    t : array_like of length T, optional
        Time points corresponding to rows of u_sim/u_anal. If None, uses indices.

    Returns
    -------
    t_half_sim : ndarray, shape (r,)
    t_half_anal : ndarray, shape (r,)
    """
    u_sim = np.asarray(u_sim)
    u_anal = np.asarray(u_anal)
    T, r = u_sim.shape
    if t is None:
        t = np.arange(T)
    t = np.asarray(t)

    t_half_sim = np.zeros(r)
    t_half_anal = np.zeros(r)
    for alpha in range(r):
        half = u_anal[-1, alpha] / 2.0
        t_half_sim[alpha] = np.interp(half, u_sim[:, alpha], t)
        t_half_anal[alpha] = np.interp(half, u_anal[:, alpha], t)
    return t_half_sim, t_half_anal


def plot_fractional_delay(t_half_sim, t_half_anal):
    """
    Plot the fractional delay (simulated vs analytical) for each mode.
    """
    frac_delay = (t_half_sim - t_half_anal) / t_half_anal
    modes = np.arange(len(frac_delay))
    plt.figure()
    plt.bar(modes, frac_delay)
    plt.xlabel('Mode index')
    plt.ylabel('Fractional delay')
    plt.title('Fractional Learning Delay')
    plt.tight_layout()
