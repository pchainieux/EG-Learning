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


def compute_closed_form_a_t(s, a0, t, tau):
    """
    Compute the closed-form solution for the network's mode strengths a_alpha(t):

    INPUT:
    s   : Data singular values s_alpha (constants). np.ndarray of shape (K,)
    a0  : Initial seeds a_{0,alpha} at t = 0. np.ndarray of shape (K,)
    t   : Time(s) at which to evaluate the mode strengths (t is measured in epochs in the paper). float or np.ndarray of shape (L,)
    tau : float, τ = 1 / (λ P), where λ is the gradient descent learning rate and P is the number of examples.

    OUTPUT:
    a_t : The mode strengths a_alpa(t) for each alpha, at each time t. np.ndarray of shape (K,) if t is scalar, else (K, L)
        
    """
    if np.ndim(t) > 0:
        exponent = 2.0 * np.outer(s, t) / tau   
        exp_term = np.exp(exponent) 
        numerator = s[:, np.newaxis] * exp_term        
        denominator = exp_term - 1.0 + (s[:, np.newaxis] / a0[:, np.newaxis])  
        a_t = numerator / denominator
    else:
        exponent = 2.0 * s * t / tau    
        exp_term = np.exp(exponent)    
        numerator = s * exp_term       
        denominator = exp_term - 1.0 + (s / a0) 
        a_t = numerator / denominator

    return a_t


def plot_mode_strengths(s, a0, tau, t_max=1000, num_points=200):
    """
    Plot the theoretical mode strengths a_alpha(t) for each singular mode alpha.

    INPUT:
    s           : Data singular values s_alpha. np.ndarray of shape (K,)
    a0          : Initial seeds a_{0,alpha}. np.ndarray of shape (K,)
    tau         : Time constant τ = 1 / (λ P). float
    t_max       : Maximum time to plot. float, optional (default=1000)
    num_points  : Number of time points between 0 and t_max. int, optional (default=200)
    """
    t_grid = np.linspace(0, t_max, num_points)
    a_over_time = compute_closed_form_a_t(s, a0, t_grid, tau)

    plt.figure(figsize=(8, 5))
    for alpha in range(len(s)):
        plt.plot(t_grid, a_over_time[alpha, :], label=f"mode {alpha + 1}")
    plt.xlabel("time $t$")
    plt.ylabel(r"$a_{\alpha}(t)$")
    plt.title("Theoretical Mode Strengths Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_singular_value_bar(s, final_a):
    """
    Plot a side by side bar chart comparing target singular values s_alpha and learned values final_a_alpha at final time.

    INPUT:
    s       : Target singular values s_alpha. np.ndarray of shape (K,)
    final_a : Learned singular values a_alpha(t_final). np.ndarray of shape (K,)
    """
    K = len(s)
    idx = np.arange(K)

    plt.figure(figsize=(6, 4))
    plt.bar(idx - 0.2, s, width=0.4, alpha=0.7, label="target $s_\\alpha$")
    plt.bar(idx + 0.2, final_a, width=0.4, alpha=0.7, label="learned $a_\\alpha$")
    plt.xlabel("mode index $\\alpha$")
    plt.ylabel("singular value / mode strength")
    plt.title("Target vs. Learned Singular Values")
    plt.xticks(idx, [str(i+1) for i in idx])
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_phase_portrait(s, tau, lim=3.0, grid=25, with_trajectory=True):
    """
    Plot the phase portrait for one strongest mode (s_alpha) in the (c, d) plane,
    where c and d satisfy:
        d c / d t = (d (s - c d)) / τ
        d d / d t = (c (s - c d)) / τ

    INPUT:
    s               : Singular value for the mode being plotted. float
    tau             : Time constant τ = 1 / (λ P). float
    lim             : Upper limit for both axes (0 to lim). float, optional (default=3.0)
    grid            : Number of grid points along each axis for vector field. int, optional (default=25)
    with_trajectory : Whether to overlay a sample trajectory starting from (0.2, 0.2). bool, optional (default=True)
        
    """
    a_vals = np.linspace(0.1, lim, grid)
    b_vals = np.linspace(0.1, lim, grid)
    A, B = np.meshgrid(a_vals, b_vals)

    # Compute vector field components
    dA = B * (s - A * B) / tau
    dB = A * (s - A * B) / tau
    speed = np.hypot(dA, dB)

    plt.figure(figsize=(5, 5))
    plt.quiver(A, B, dA, dB, speed, cmap="gray_r", width=0.002)
    plt.plot(a_vals, s / a_vals, "r--", lw=2, label=r"$c\,d = s$")

    if with_trajectory:
        def mode_ode(t, z):
            c, d = z
            return [d * (s - c * d) / tau, c * (s - c * d) / tau]

        sol = solve_ivp(
            mode_ode, 
            [0, 8 * tau / s], 
            [0.2, 0.2], 
            max_step=0.05
        )
        plt.plot(sol.y[0], sol.y[1], "b", lw=2, label="sample trajectory")

    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel(r"$c$")
    plt.ylabel(r"$d$")
    plt.title(f"Phase Portrait for Mode $s={s:.4f}$")
    plt.legend()
    plt.grid(True)
    plt.show()


# =================
# Main function
# =================
if __name__ == "__main__":
    P = 500    # number of training examples
    N1 = 5    # input dimension
    N3 = 5    # output dimension

    # Random inputs X ~ N(0, I)
    X = np.random.randn(P, N1)

    # Random linear teacher map M_true
    M_true = np.random.randn(N3, N1)

    # Outputs Y = X @ M_true^T + small Gaussian noise
    Y = X @ M_true.T + 0.01 * np.random.randn(P, N3)

    # Compute data singular values s_alpha (Σ_yx SVD after whitening X)
    s, U, V = compute_singular_modes(X, Y)

    # Initialise tiny seeds a_{0,alpha} = 1e-4
    a0 = initialize_mode_strengths(s, a0_value=1e-4)

    # Choose a slow learning rate 
    s1 = s[0]
    lambda_lr = 0.1 / (s1 * P)  
    tau = 1.0 / (lambda_lr * P)

    # Plot theoretical curves:
    plot_mode_strengths(s, a0, tau, t_max=1000, num_points=300)

    # Compare target s vs. a_alpha(t_final)
    t_final = 500.0
    final_a = compute_closed_form_a_t(s, a0, t_final, tau)
    plot_singular_value_bar(s, final_a)

    # Phase portrait for the strongest mode s₁
    plot_phase_portrait(s[0], tau, lim=3.0, grid=40, with_trajectory=True)
