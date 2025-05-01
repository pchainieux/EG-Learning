import numpy as np
import matplotlib.pyplot as plt

def phase_field(s, tau, a, b):
    """
    Compute vector field derivatives for the scalar mode-reduced ODEs:
    τ da/dt = b (s - a b),  τ db/dt = a (s - a b).

    Parameters
    ----------
    s : float
        Singular value for the mode.
    tau : float
        Time constant.
    a : array_like
        Grid of a values.
    b : array_like
        Grid of b values.

    Returns
    -------
    da : ndarray
        Derivative da/dt on the grid.
    db : ndarray
        Derivative db/dt on the grid.
    """
    da = (b * (s - a * b)) / tau
    db = (a * (s - a * b)) / tau
    return da, db


def plot_phase_portrait(s, tau, grid_limits, density=20):
    """
    Plot the phase portrait for the scalar dynamics of one mode.

    Parameters
    ----------
    s : float
        Singular value for the mode.
    tau : float
        Time constant.
    grid_limits : tuple of floats (a_min, a_max, b_min, b_max)
        Plot limits for a and b axes.
    density : int
        Number of grid points per axis for the vector field.
    """
    a_min, a_max, b_min, b_max = grid_limits
    a_vals = np.linspace(a_min, a_max, density)
    b_vals = np.linspace(b_min, b_max, density)
    A, B = np.meshgrid(a_vals, b_vals)
    DA, DB = phase_field(s, tau, A, B)
    # Normalize arrows for better visualization
    M = np.hypot(DA, DB)
    DA /= M; DB /= M

    plt.figure()
    plt.quiver(A, B, DA, DB)
    # Nullclines: b=0, a=0, and ab=s
    a_nc = np.linspace(a_min, a_max, 400)
    plt.plot(a_nc, np.zeros_like(a_nc), label='b=0')
    plt.plot(np.zeros_like(a_nc), a_nc, label='a=0')
    plt.plot(a_nc, s / a_nc, label='ab=s')

    plt.xlim(a_min, a_max)
    plt.ylim(b_min, b_max)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.legend()
    plt.title('Phase Portrait (s={}, τ={})'.format(s, tau))
    plt.tight_layout()
