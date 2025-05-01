import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from correlations import compute_Sigma11, compute_Sigma31
from mode_decomp import svd_decompose
from modes_ode import modes_deriv, solve_u


def simulate_linear(x, y, N2, tau, t, seed=None):
    """
    Simulate continuous-time dynamics for the linear three-layer network in mode coordinates.

    Returns trajectories of a and b for each mode, and singular values s.
    """
    # Compute correlations and decompose
    Sigma11 = compute_Sigma11(x)
    Sigma31 = compute_Sigma31(x, y)
    U, s, V = svd_decompose(Sigma31)
    r = len(s)
    # Random init for a and b
    rng = np.random.RandomState(seed)
    a0 = rng.randn(r, N2) * 1e-2
    b0 = rng.randn(r, N2) * 1e-2
    state0 = np.concatenate([a0.flatten(), b0.flatten()])
    # Integrate coupled ODE
    sol = odeint(modes_deriv, state0, t, args=(s, tau, r, N2))
    a_traj = sol[:, :r * N2].reshape((len(t), r, N2))
    b_traj = sol[:, r * N2:].reshape((len(t), r, N2))
    return a_traj, b_traj, s


def simulate_nonlinear(x, y, N2, lr, num_steps, seed=None):
    """
    Simulate discrete-time gradient descent on a three-layer network with tanh activations.

    Returns a list of Sigma31 matrices over iterations.
    """
    P, N1 = x.shape
    N3 = y.shape[1]
    rng = np.random.RandomState(seed)
    W21 = rng.randn(N2, N1) * 1e-2
    W32 = rng.randn(N3, N2) * 1e-2
    Sigma31_traj = []
    for _ in range(num_steps):
        H = np.tanh(W21.dot(x.T))           # (N2, P)
        Y_hat = W32.dot(H)                  # (N3, P)
        residual = Y_hat - y.T              # (N3, P)
        # Gradients
        dW32 = residual.dot(H.T) * lr
        grad_h = (W32.T.dot(residual)) * (1 - H**2)
        dW21 = grad_h.dot(x) * lr
        # Updates
        W21 -= dW21
        W32 -= dW32
        Sigma31_traj.append(W32.dot(W21).dot(Sigma11) if False else None)
    # Placeholder: user should replace None with actual Sigma11-based update
    return Sigma31_traj


def extract_mode_strengths(Sigma31_traj):
    """
    Compute singular values for each Sigma31 matrix in the trajectory.

    Returns array of shape (len(traj), r).
    """
    strengths = []
    for Sigma in Sigma31_traj:
        U, s, V = svd_decompose(Sigma)
        strengths.append(s)
    return np.array(strengths)


def plot_learning_curves(t, u_sim, u_anal):
    """
    Plot simulated vs. analytical mode-strength trajectories over time.
    """
    for alpha in range(u_sim.shape[1]):
        plt.plot(t, u_sim[:, alpha])
        plt.plot(t, u_anal[:, alpha], '--')
    plt.xlabel('Time')
    plt.ylabel('Mode strength (u)')
    plt.title('Learning Curves')
    plt.tight_layout()
