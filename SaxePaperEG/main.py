import numpy as np
import matplotlib.pyplot as plt

from data import generate_orthogonal_inputs, generate_synthetic_outputs
from correlations import compute_Sigma11
from linear_net import init_weights, batch_gradient_update


def main():
    # Network and data parameters
    N1, N2, N3 = 50, 20, 50    # Input, hidden, output dims
    P = 200                   # Number of samples
    rank = 10                # Desired rank of Sigma31
    lr = 0.01                # Learning rate for gradient descent
    epochs = 500             # Number of training steps
    seed = 0

    # Generate orthonormal inputs and synthetic outputs
    x = generate_orthogonal_inputs(N1, P, random_seed=seed)
    y = generate_synthetic_outputs(x, N3, rank=rank, random_seed=seed)

    # Initialize linear network weights
    W21, W32 = init_weights(N1, N2, N3, init='random', scale=0.1, random_seed=seed)

    # Precompute data covariance eigenvectors
    Sigma11 = compute_Sigma11(x)          # (N1, N1)
    eigvals, eigvecs = np.linalg.eigh(Sigma11)
    # Sort eigenvectors by descending variance
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Storage for projections onto top-input PCs
    projections = np.zeros((epochs, rank))

    # Training loop with projection tracking
    for t in range(epochs):
        # One step of batch gradient descent
        W21, W32 = batch_gradient_update(W21, W32, x, y, lr)
        # Effective end-to-end mapping
        W_eff = W32.dot(W21)               # Shape (N3, N1)
        # Project onto the top 'rank' eigenvectors of Sigma11
        for i in range(rank):
            v = eigvecs[:, i]             # i-th principal direction
            projections[t, i] = np.linalg.norm(W_eff.dot(v))

    # Plot the growth of each projection over training
    plt.figure(figsize=(8, 6))
    for i in range(rank):
        plt.plot(projections[:, i], label=f"PC {i+1}")
    plt.xlabel('Epoch')
    plt.ylabel('||W_eff v_i||')
    plt.title('Projections of W on Data Eigenvectors')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
