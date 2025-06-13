import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def whiten_data(X):
    """
    Whiten the input data X so that its covariance Σ_x = I.

    INPUT:
    X       : Input data matrix with P samples (rows) and N1 features (columns). np.ndarray of shape (P, N1)

    OUTPUT:
    X_white : Whitened data (zero-mean, unit covariance). np.ndarray of shape (P, N1)
    """
    P, N1 = X.shape

    # Center X (subtract column‐wise mean).
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # Compute empirical covariance 
    Sigma_x = (X_centered.T @ X_centered) / P

    # Eigen‐decompose Σ_x = E Λ E^T
    eigvals, eigvecs = np.linalg.eigh(Sigma_x)

    # Build whitening matrix W = E Λ^{-1/2} E^T. Add a tiny epsilon inside sqrt to avoid division by zero.
    eps = 1e-12
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
    whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T

    # Apply W to the centered data to produce X_white
    X_white = X_centered @ whitening_matrix

    return X_white


def compute_singular_modes(X, Y):
    """
    Given input data X and output data Y, compute the singular values (s_alpha) of the teacher covariance Σ_yx after whitening X.

    Assumptions (tabula rasa regime):
      1. Σ_x = I  (whiten inputs).
      2. Compute Σ_yx = (Y^T @ X_white) / P.
      3. Perform SVD(Σ_yx) = U S V^T so that S = diag(s_1, s_2, ...).

    INPUT:
    X : Input data (rows are examples, columns are features). np.ndarray of shape (P, N1) 
    Y : Output data (rows are examples, columns are output dimensions). np.ndarray of shape (P, N3)

    OUTOUT:
    s : Sorted singular values (s_1 ≥ s_2 ≥ ... ≥ 0) of Σ_yx. np.ndarray of shape (K,)
    U : Left singular vectors of Σ_yx. np.ndarray of shape (N3, K)
    V : Right singular vectors of Σ_yx. np.ndarray of shape (N1, K)
    """
    P, N1 = X.shape
    _, N3 = Y.shape

    # Whiten X so that Σ_x = I
    X_white = whiten_data(X)

    # Form Σ_yx = (Y.T @ X_white) / P
    Sigma_yx = (Y.T @ X_white) / P

    # Perform SVD: Σ_yx = U S V^T
    U, S, Vt = np.linalg.svd(Sigma_yx, full_matrices=False)
    V = Vt.T  # shape (N1, K)

    return S, U, V


def initialize_mode_strengths(s, a0_value=1e-4):
    """
    Initialise the network's mode strengths a_{0,alpha} (the seeds) for each singular mode alpha.
    In the tabula rasa regime, each mode starts at a tiny value a0 ≪ s_alpha, and the left/right layer modes are balanced so that c_alpha(0) = d_alpha(0) = sqrt(a0).

    INPUT:
    s           : Data singular values s_alpha (constants, from SVD). np.ndarray of shape (K,)
    a0_value    : A small seed value for all modes. float, optional (default = 1e-4)
        
    OUTPUT:
    a0          : Initial seeds a_{0, alpha}.  Typically chosen so that a0 ≪ s_alpha for every alpha. np.ndarray of shape (K,)
    """
    a0 = np.full_like(s, fill_value=a0_value)
    return a0


def simulate_eg_modes(s, a0, eta, num_steps):
    """
    Simulate EG descent on each mode alpha, assuming an aligned initialization so that c_alpha(0)=d_alpha(0)=aqrt(a0_alpha).

    At each discrete step t:
      grad_c = (c_alpha d_alpha - s_alpha) * d_alpha
      grad_d = (c_alpha d_alpha - s_alpha) * c_alpha
      c_alpha ← c_alpha * exp( -η * grad_c )
      d_alpha ← d_alpha * exp( -η * grad_d )
    Then record a_alpha(t) = c_alpha(t) d_alpha(t).

    INPUT:
    s           : Data singular values s_alpha. np.ndarray of shape (K,)
    a0          : Initial seeds a_{0,alpha}. np.ndarray of shape (K,)
    eta         : EG learning rate (step size). float
    num_steps   : Number of discrete EG steps to simulate. int

    OUTPUT:
    a_history   : Mode strengths a_alpha(t) over t = 0,1,...,num_steps-1. np.ndarray of shape (K, num_steps)
    """
    K = len(s)
    # Initialise c and d so that a0_alpha = c_alpha(0) * d_alpha(0)  → c_alpha(0)=d_alpha(0)=sqrt(a0_alpha)
    c = np.sqrt(a0).copy()
    d = np.sqrt(a0).copy()

    a_history = np.zeros((K, num_steps))
    for t in range(num_steps):
        # Record current mode strengths a_alpha = c_alpha * d_alpha
        a_history[:, t] = c * d

        # Compute gradients for each mode alpha
        grad_c = (c * d - s) * d
        grad_d = (c * d - s) * c

        # Exponentiated update:
        c = c * np.exp(-eta * grad_c)
        d = d * np.exp(-eta * grad_d)

    return a_history


def plot_mode_strengths_eg(s, a0, eta, num_steps):
    """
    Plot the EG simulated mode strengths a_alpha(t) versus discrete step t.

    INPUT:
    s         : Data singular values s_alpha. np.ndarray of shape (K,)
    a0        : Initial seeds a_{0,alpha}. np.ndarray of shape (K,)
    eta       : EG learning rate. float
    num_steps : Number of steps to simulate. int
    """
    a_hist = simulate_eg_modes(s, a0, eta, num_steps)
    t_grid = np.arange(num_steps) 

    plt.figure(figsize=(8, 5))
    for alpha in range(len(s)):
        plt.plot(t_grid, a_hist[alpha, :], label=f"mode {alpha + 1}")
    plt.xlabel("EG step $t$")
    plt.ylabel(r"$a_{\alpha}(t)$ (EG)")
    plt.title("Mode Strengths Over Time (Exponentiated Gradient)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_singular_value_bar_eg(s, final_a):
    """
    Compare target singular values s_alpha against final EG-learned a_alpha.

    INPUT:
    s       : Target singular values s_alpha. np.ndarray of shape (K,)
    final_a : EG-learned mode strengths a_alpha at the final step. np.ndarray of shape (K,)
    """
    K = len(s)
    idx = np.arange(K)

    plt.figure(figsize=(6, 4))
    plt.bar(idx - 0.2, s,     width=0.4, alpha=0.7, label="target $s_\\alpha$")
    plt.bar(idx + 0.2, final_a, width=0.4, alpha=0.7, label="EG-learned $a_\\alpha$")
    plt.xlabel("mode index $\\alpha$")
    plt.ylabel("singular value / mode strength")
    plt.title("Target vs. EG-Learned Singular Values")
    plt.xticks(idx, [str(i+1) for i in idx])
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_phase_portrait_eg(s, eta, lim=3.0, grid=25, with_trajectory=True):
    """
    Phase portrait in the (c,d) plane under the continuous time EG ODE approximation:
      dc/dt = -η * c * d * (c d - s)
      dd/dt = -η * c^2 * (c d - s)
    Plot the nullcline c d = s and, optionally, a sample trajectory from (0.2, 0.2).

    INPUT:
    s               : Singular value for this mode. float
    eta             : EG learning rate (treated as continuous-step coefficient). float
    lim             : Maximum value for both c and d axes. float
    grid            : Number of grid points per axis for the vector field. int
    with_trajectory : If True, overlay a sample trajectory starting at (c(0)=0.2, d(0)=0.2). bool
    """
    a_vals = np.linspace(0.1, lim, grid)
    b_vals = np.linspace(0.1, lim, grid)
    A, B = np.meshgrid(a_vals, b_vals)

    dA = -eta * A * B * (A * B - s)
    dB = -eta * A**2 * (A * B - s)
    speed = np.hypot(dA, dB)

    plt.figure(figsize=(5, 5))
    plt.quiver(A, B, dA, dB, speed, cmap="gray_r", width=0.002)
    plt.plot(a_vals, s / a_vals, "r--", lw=2, label=r"$c\,d = s$ (nullcline)")

    if with_trajectory:
        def eg_ode(t, z):
            c, d = z
            return [
                -eta * c * d * (c * d - s),
                -eta * c**2 * (c * d - s)
            ]

        # Integrate for a bit to see the sample trajectory
        sol = solve_ivp(
            eg_ode,
            [0, 8.0 / (eta * s)],    # run long enough so trajectory approaches c*d≈s
            [0.2, 0.2],              # start at (c=0.2, d=0.2)
            max_step=0.05
        )
        plt.plot(sol.y[0], sol.y[1], "b", lw=2, label="sample trajectory")

    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel(r"$c$")
    plt.ylabel(r"$d$")
    plt.title(f"EG Phase Portrait for Mode $s = {s:.4f}$")
    plt.legend()
    plt.grid(True)
    plt.show()



# ==============
# Main
# ==============
if __name__ == "__main__":
    P = 500    # number of examples
    N1 = 5    # input dimension
    N3 = 5    # output dimension

    # Random inputs X ~ N(0, I)
    X = np.random.randn(P, N1)

    # Random linear teacher map M_true
    M_true = np.random.randn(N3, N1)

    # Outputs Y = X @ M_true^T + small Gaussian noise
    Y = X @ M_true.T + 0.01 * np.random.randn(P, N3)

    # Compute Σ_yx's singular values s_alpha (whitening X first)
    s, U, V = compute_singular_modes(X, Y)

    # Initialise tiny seeds a_{0,alpha} = 1e-4
    a0 = initialize_mode_strengths(s, a0_value=1e-4)

    # Choose an EG learning rate η (small enough to keep dynamics smooth)
    eta = 0.1

    # Plot mode strengths under EG for 500 steps
    num_steps = 1000
    plot_mode_strengths_eg(s, a0, eta, num_steps)

    # Compare target s_alpha vs. final EG-learned a_alpha
    a_hist = simulate_eg_modes(s, a0, eta, num_steps)
    final_a = a_hist[:, -1]
    plot_singular_value_bar_eg(s, final_a)

    # Phase portrait for the strongest mode (s_1) under EG ODE
    plot_phase_portrait_eg(s[0], eta, lim=3.0, grid=40, with_trajectory=True)
