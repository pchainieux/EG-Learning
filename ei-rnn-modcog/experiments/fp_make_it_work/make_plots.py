import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

def reproduce_pca_plots(activation_file, timing_cfg=None, show_plots=True, 
                       plot_percentage=100, save_dir=None):
    """
    Reproduce PCA trajectory plots from saved activation files
    
    Args:
        activation_file: Path to activations_epoch*.npz file
        timing_cfg: Dictionary with timing parameters (optional)
        show_plots: Whether to display plots
        plot_percentage: Percentage of the delay period dynamics (from the end) to plot (1-100)
        save_dir: Directory to save plots (if None, plots are only shown, not saved)
    """
    
    # Load the saved activations
    data = np.load(activation_file)
    
    print(f"Loaded: {activation_file}")
    print(f"Available keys: {list(data.keys())}")
    
    # Extract the arrays
    H_arr = data["H"]        # Hidden states (trials, timesteps, hidden_dim)
    labels = data["labels"]   # Target labels per trial
    
    if "epoch" in data:
        epoch = data["epoch"].item()
        print(f"Epoch: {epoch}")
    else:
        epoch = None
    
    print(f"Hidden states shape: {H_arr.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create save directory if specified
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {save_dir}")
    
    # Default timing (adjust these to match your task timing)
    if timing_cfg is None:
        timing_cfg = {
            "fixation_steps": 5,
            "stimulus_steps": 5, 
            "delay_steps": 10,
            "decision_steps": 5
        }
    
    fix_steps = int(timing_cfg.get("fixation_steps", 5))
    stim_steps = int(timing_cfg.get("stimulus_steps", 5))
    delay_steps = int(timing_cfg.get("delay_steps", 10))
    
    # Define delay period slice
    delay_start = fix_steps + stim_steps
    delay_end = delay_start + delay_steps
    
    # Check if we have enough timesteps
    if delay_end > H_arr.shape[1]:
        print(f"Warning: Not enough timesteps. Have {H_arr.shape[1]}, need {delay_end}")
        delay_end = min(delay_end, H_arr.shape[1])
        delay_steps = delay_end - delay_start
    
    delay_slice = slice(delay_start, delay_end)
    
    # Extract delay period hidden states
    H_delay = H_arr[:, delay_slice, :]  # (trials, delay_steps, hidden_dim)
    total_delay_steps = H_delay.shape[1]
    
    # Calculate steps to plot based on percentage
    if not (1 <= plot_percentage <= 100):
        raise ValueError(f"plot_percentage must be between 1 and 100, got {plot_percentage}")
    
    steps_to_plot = int(total_delay_steps * plot_percentage / 100)
    
    if steps_to_plot < 1:
        steps_to_plot = 1
        print(f"Warning: plot_percentage {plot_percentage}% resulted in 0 steps, using 1 step")
    
    # Extract only the last 'steps_to_plot' timesteps from delay period
    H_delay_subset = H_delay[:, -steps_to_plot:, :]
    H_delay_flat = H_delay_subset.reshape(-1, H_delay_subset.shape[-1])
    
    print(f"Using {steps_to_plot}/{total_delay_steps} delay steps ({plot_percentage}% from end)")
    print(f"Subset shape: {H_delay_subset.shape}")
    
    # Ensure labels match trials
    n_trials = H_arr.shape[0]
    if len(labels) != n_trials:
        print(f"Adjusting labels: {len(labels)} -> {n_trials}")
        if len(labels) > n_trials:
            labels = labels[:n_trials]
        else:
            labels = np.pad(labels, (0, n_trials - len(labels)), 
                          mode='constant', constant_values=0)
    
    # Replicate labels for each timestep in the subset
    labels_delay = np.repeat(labels, steps_to_plot)
    
    # Ensure sizes match
    if len(labels_delay) != H_delay_flat.shape[0]:
        min_size = min(len(labels_delay), H_delay_flat.shape[0])
        labels_delay = labels_delay[:min_size]
        H_delay_flat = H_delay_flat[:min_size]
    
    print(f"Final PCA input: H_delay_flat={H_delay_flat.shape}, labels_delay={labels_delay.shape}")
    
    # ==============================================
    # PLOT 1: Delay Period PCA (Subset)
    # ==============================================
    
    pca_delay = PCA(n_components=3, svd_solver="full")
    Z_delay = pca_delay.fit_transform(H_delay_flat)
    
    print(f"Delay PCA explained variance: {pca_delay.explained_variance_ratio_[:3]}")
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(Z_delay[:, 0], Z_delay[:, 1], c=labels_delay, 
                         s=6, cmap="hsv", alpha=0.7)
    plt.colorbar(scatter, label="Target Action")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # Enhanced title showing percentage
    title = f"Delay Period Dynamics (Last {plot_percentage}% of delay)"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # SAVE PLOT 1
    if save_dir is not None:
        epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
        filename1 = f"pca_delay_{epoch_str}_pct{plot_percentage}.png"
        filepath1 = save_dir / filename1
        plt.savefig(filepath1, dpi=200, bbox_inches='tight')
        print(f"Saved: {filepath1}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()  # Close figure if not showing to save memory
    
    # ==============================================
    # PLOT 2: Trial Trajectories (Full Time Course)
    # ==============================================
    
    # PCA on full trial data
    pca_full = PCA(n_components=3).fit(H_arr.reshape(-1, H_arr.shape[-1]))
    print(f"Full trial PCA explained variance: {pca_full.explained_variance_ratio_[:3]}")
    
    plt.figure(figsize=(10, 7))
    
    # Plot subset of trials for clarity
    n_plot = min(24, H_arr.shape[0])
    
    # Calculate indices for the subset region
    subset_start = delay_end - steps_to_plot
    subset_end = delay_end
    
    for i in range(n_plot):
        # Transform trial trajectory to PC space
        Zi = pca_full.transform(H_arr[i])
        
        # Plot different epochs in different colors
        # Fixation (gray)
        if fix_steps > 0:
            plt.plot(Zi[:fix_steps, 0], Zi[:fix_steps, 1], 
                    alpha=0.3, color='gray', linewidth=0.8)
        
        # Stimulus (green)
        if stim_steps > 0:
            plt.plot(Zi[fix_steps:fix_steps+stim_steps, 0], 
                    Zi[fix_steps:fix_steps+stim_steps, 1], 
                    alpha=0.6, color='green', linewidth=1.0)
        
        # Full delay period (light blue)
        plt.plot(Zi[delay_start:delay_end, 0], Zi[delay_start:delay_end, 1], 
                alpha=0.5, color='lightblue', linewidth=0.8)
        
        # Highlighted subset (dark blue, thicker)
        plt.plot(Zi[subset_start:subset_end, 0], Zi[subset_start:subset_end, 1], 
                alpha=1.0, color='darkblue', linewidth=2.0)
        
        # Decision (red)
        if delay_end < H_arr.shape[1]:
            plt.plot(Zi[delay_end:, 0], Zi[delay_end:, 1], 
                    alpha=1.0, color='red', linewidth=1.0)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    title = f"Trial Trajectories (Last {plot_percentage}% of delay highlighted)"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.6, label='Fixation'),
        Line2D([0], [0], color='green', alpha=0.8, label='Stimulus'),
        Line2D([0], [0], color='lightblue', alpha=0.7, label='Full Delay'),
        Line2D([0], [0], color='darkblue', alpha=1.0, linewidth=2, 
               label=f'Analyzed ({plot_percentage}%)'),
        Line2D([0], [0], color='red', alpha=1.0, label='Decision')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    
    # SAVE PLOT 2
    if save_dir is not None:
        epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
        filename2 = f"trajectories_{epoch_str}_pct{plot_percentage}.png"
        filepath2 = save_dir / filename2
        plt.savefig(filepath2, dpi=200, bbox_inches='tight')
        print(f"Saved: {filepath2}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()  # Close figure if not showing to save memory
    
    return Z_delay, pca_delay.explained_variance_ratio_, H_delay_subset

# ==============================================
# USAGE EXAMPLES
# ==============================================

if __name__ == "__main__":
    
    # Your activation file
    activation_file = "outputs/last_stand/dm1_gd/activations_epoch010.npz"
    
    timing_config = {
        "fixation_steps": 5,
        "stimulus_steps": 5,
        "delay_steps": 10,
        "decision_steps": 5
    }
    
    # Specify where to save plots
    output_directory = "outputs/last_stand"  # Change this to your preferred directory
    
    print("=== Generating and Saving PCA Plots ===")
    try:
        # Example 1: Save plots for different percentages
        for pct in [100, 75, 50, 25]:
            Z, explained_var, H_subset = reproduce_pca_plots(
                activation_file, 
                timing_cfg=timing_config, 
                plot_percentage=pct,
                show_plots=False,  # Don't show, just save
                save_dir=output_directory
            )
            print(f"{pct}% analysis - PC1 explains {explained_var[0]:.3f} of variance")
        
        print(f"\nAll plots saved to: {output_directory}/")
        
    except FileNotFoundError:
        print(f"File not found: {activation_file}")
        print("Please adjust the path to your actual activations file")
