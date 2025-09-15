import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Added for 3D plotting
from pathlib import Path

def reproduce_pca_plots_debug(activation_file, timing_cfg=None, show_plots=True,
                             plot_percentage=100, save_dir=None, plot_3d=False):
    """
    Reproduce PCA trajectory plots with extensive debugging and 2D/3D option
    """
    
    # Load the saved activations
    data = np.load(activation_file)
    H_arr = data["H"]        # Hidden states (trials, timesteps, hidden_dim)
    labels = data["labels"]   # Target labels per trial
    
    if "epoch" in data:
        epoch = data["epoch"].item()
    else:
        epoch = None
    
    print(f"=== DEBUGGING INFO ===")
    print(f"Original H_arr shape: {H_arr.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Plot mode: {'3D' if plot_3d else '2D'}")
    
    # Create save directory if specified
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Default timing
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
    
    print(f"Delay period: steps {delay_start} to {delay_end} (total {delay_steps} steps)")
    
    # Check if we have enough timesteps
    if delay_end > H_arr.shape[1]:
        print(f"WARNING: Not enough timesteps. Have {H_arr.shape[1]}, need {delay_end}")
        delay_end = min(delay_end, H_arr.shape[1])
        delay_steps = delay_end - delay_start
    
    # Extract delay period hidden states
    H_delay = H_arr[:, delay_start:delay_end, :]  # (trials, delay_steps, hidden_dim)
    total_delay_steps = H_delay.shape[1]
    
    print(f"H_delay shape: {H_delay.shape}")
    print(f"Total delay steps available: {total_delay_steps}")
    
    # Calculate steps to plot based on percentage
    if not (1 <= plot_percentage <= 100):
        raise ValueError(f"plot_percentage must be between 1 and 100, got {plot_percentage}")
    
    steps_to_plot = max(1, int(total_delay_steps * plot_percentage / 100))
    
    print(f"Steps to plot ({plot_percentage}%): {steps_to_plot}/{total_delay_steps}")
    
    # Extract only the last 'steps_to_plot' timesteps from delay period
    H_delay_subset = H_delay[:, -steps_to_plot:, :]
    
    print(f"H_delay_subset shape: {H_delay_subset.shape}")
    
    # DEBUGGING: Check if data actually changes with different percentages
    mean_activity_per_trial = H_delay_subset.mean(axis=(1, 2))  # Mean over time and units
    variance_activity = H_delay_subset.var(axis=(1, 2))  # Variance over time and units
    
    print(f"Mean activity per trial (first 5): {mean_activity_per_trial[:5]}")
    print(f"Activity variance per trial (first 5): {variance_activity[:5]}")
    print(f"Overall mean: {H_delay_subset.mean():.4f}, Overall std: {H_delay_subset.std():.4f}")
    
    # Check temporal evolution within subset
    if steps_to_plot > 1:
        temporal_mean = H_delay_subset.mean(axis=(0, 2))  # Mean over trials and units
        print(f"Temporal evolution within subset: {temporal_mean}")
    
    # Flatten for PCA
    H_delay_flat = H_delay_subset.reshape(-1, H_delay_subset.shape[-1])
    
    print(f"H_delay_flat shape for PCA: {H_delay_flat.shape}")
    
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
    
    print(f"Labels_delay shape: {labels_delay.shape}")
    print(f"Unique labels_delay: {np.unique(labels_delay)}")
    
    # Ensure sizes match
    if len(labels_delay) != H_delay_flat.shape[0]:
        min_size = min(len(labels_delay), H_delay_flat.shape[0])
        labels_delay = labels_delay[:min_size]
        H_delay_flat = H_delay_flat[:min_size]
        print(f"Adjusted to common size: {min_size}")
    
    # CRITICAL CHECK: Verify data is actually different for different percentages
    data_fingerprint = f"Mean: {H_delay_flat.mean():.6f}, Std: {H_delay_flat.std():.6f}, " \
                      f"Min: {H_delay_flat.min():.6f}, Max: {H_delay_flat.max():.6f}"
    print(f"Data fingerprint: {data_fingerprint}")
    
    # ==============================================
    # PCA Analysis
    # ==============================================
    
    pca_delay = PCA(n_components=3, svd_solver="full")
    Z_delay = pca_delay.fit_transform(H_delay_flat)
    
    print(f"PCA output shape: {Z_delay.shape}")
    print(f"PCA explained variance: {pca_delay.explained_variance_ratio_[:3]}")
    
    # Check PCA spread
    pca_range = [Z_delay[:, i].max() - Z_delay[:, i].min() for i in range(3)]
    print(f"PCA component ranges: PC1={pca_range[0]:.3f}, PC2={pca_range[1]:.3f}, PC3={pca_range[2]:.3f}")
    
    # ==============================================
    # PLOT: Delay Period PCA (2D or 3D)
    # ==============================================
    
    if plot_3d:
        # 3D Plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(Z_delay[:, 0], Z_delay[:, 1], Z_delay[:, 2], 
                           c=labels_delay, s=12, cmap="hsv", alpha=0.7)
        ax.set_xlabel(f"PC1 (explains {pca_delay.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 (explains {pca_delay.explained_variance_ratio_[1]:.2%})")
        ax.set_zlabel(f"PC3 (explains {pca_delay.explained_variance_ratio_[2]:.2%})")
        
        # Enhanced title
        title = f"Delay Dynamics 3D (Last {plot_percentage}% = {steps_to_plot} steps)"
        if epoch is not None:
            title += f" - Epoch {epoch}"
        title += f"\nData: μ={H_delay_flat.mean():.3f}, σ={H_delay_flat.std():.3f}"
        
        ax.set_title(title, fontsize=10)
        fig.colorbar(scatter, label="Target Action", shrink=0.8)
        
        # Save 3D plot
        if save_dir is not None:
            epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
            filename = f"pca_delay_3D_{epoch_str}_pct{plot_percentage}_steps{steps_to_plot}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            print(f"Saved 3D: {filepath}")
    
    else:
        # 2D Plot (original)
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(Z_delay[:, 0], Z_delay[:, 1], c=labels_delay,
                             s=8, cmap="hsv", alpha=0.7)
        plt.colorbar(scatter, label="Target Action")
        plt.xlabel(f"PC1 (explains {pca_delay.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (explains {pca_delay.explained_variance_ratio_[1]:.2%})")
        
        # Enhanced title
        title = f"Delay Dynamics 2D (Last {plot_percentage}% = {steps_to_plot} steps)"
        if epoch is not None:
            title += f" - Epoch {epoch}"
        title += f"\nData: μ={H_delay_flat.mean():.3f}, σ={H_delay_flat.std():.3f}"
        plt.title(title, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save 2D plot
        if save_dir is not None:
            epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
            filename = f"pca_delay_2D_{epoch_str}_pct{plot_percentage}_steps{steps_to_plot}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            print(f"Saved 2D: {filepath}")
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return Z_delay, pca_delay.explained_variance_ratio_, H_delay_subset, data_fingerprint


def analyze_by_stimulus_identity(activation_file, timing_cfg=None, show_plots=True, 
                                plot_percentage=100, save_dir=None, plot_3d=False):
    """
    NEW: Analyze and plot trajectories colored by stimulus identity (2D or 3D)
    """
    
    # Load the saved activations
    data = np.load(activation_file)
    H_arr = data["H"]        # Hidden states (trials, timesteps, hidden_dim)
    labels = data["labels"]   # Target labels per trial
    
    if "epoch" in data:
        epoch = data["epoch"].item()
    else:
        epoch = None
    
    print(f"\n=== STIMULUS IDENTITY ANALYSIS ===")
    print(f"H_arr shape: {H_arr.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique stimulus labels: {np.unique(labels)}")
    print(f"Plot mode: {'3D' if plot_3d else '2D'}")
    
    # Create save directory if specified
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Default timing
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
    
    delay_start = fix_steps + stim_steps
    delay_end = delay_start + delay_steps
    
    if delay_end > H_arr.shape[1]:
        delay_end = min(delay_end, H_arr.shape[1])
        delay_steps = delay_end - delay_start
    
    # ==============================================
    # PLOT: Delay Period PCA (colored by stimulus, 2D or 3D)
    # ==============================================
    
    # Extract delay period hidden states
    H_delay = H_arr[:, delay_start:delay_end, :]
    
    # Apply percentage filtering
    total_delay_steps = H_delay.shape[1]
    steps_to_plot = max(1, int(total_delay_steps * plot_percentage / 100))
    H_delay_subset = H_delay[:, -steps_to_plot:, :]
    H_delay_flat = H_delay_subset.reshape(-1, H_delay_subset.shape[-1])
    
    # Ensure labels match trials
    n_trials = H_arr.shape[0]
    if len(labels) != n_trials:
        if len(labels) > n_trials:
            labels = labels[:n_trials]
        else:
            labels = np.pad(labels, (0, n_trials - len(labels)), mode='constant', constant_values=0)
    
    # Replicate labels for each delay timestep
    labels_delay = np.repeat(labels, steps_to_plot)
    
    # Ensure sizes match
    if len(labels_delay) != H_delay_flat.shape[0]:
        min_size = min(len(labels_delay), H_delay_flat.shape[0])
        labels_delay = labels_delay[:min_size]
        H_delay_flat = H_delay_flat[:min_size]
    
    # PCA on delay period
    pca_delay = PCA(n_components=3, svd_solver="full")
    Z_delay = pca_delay.fit_transform(H_delay_flat)
    
    if plot_3d:
        # 3D Stimulus Identity Plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(Z_delay[:, 0], Z_delay[:, 1], Z_delay[:, 2], 
                           c=labels_delay, s=15, cmap="hsv", alpha=0.8)
        ax.set_xlabel(f"PC1 ({pca_delay.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca_delay.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_zlabel(f"PC3 ({pca_delay.explained_variance_ratio_[2]:.1%} variance)")
        
        title = f"Delay Period PCA 3D - Colored by Stimulus Identity"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        if plot_percentage < 100:
            title += f"\nLast {plot_percentage}% of delay period"
        ax.set_title(title, fontsize=12)
        
        fig.colorbar(scatter, label="Stimulus Identity", shrink=0.8)
        
        # Save 3D stimulus plot
        if save_dir is not None:
            epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
            filename = f"pca_delay_by_stimulus_3D_{epoch_str}_pct{plot_percentage}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            print(f"Saved 3D stimulus-colored delay PCA: {filepath}")
    
    else:
        # 2D Stimulus Identity Plot (original)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(Z_delay[:, 0], Z_delay[:, 1], c=labels_delay, 
                             s=12, cmap="hsv", alpha=0.8)
        plt.colorbar(scatter, label="Stimulus Identity")
        plt.xlabel(f"PC1 ({pca_delay.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca_delay.explained_variance_ratio_[1]:.1%} variance)")
        
        title = f"Delay Period PCA 2D - Colored by Stimulus Identity"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        if plot_percentage < 100:
            title += f"\nLast {plot_percentage}% of delay period"
        plt.title(title, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save 2D stimulus plot
        if save_dir is not None:
            epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
            filename = f"pca_delay_by_stimulus_2D_{epoch_str}_pct{plot_percentage}.png"
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            print(f"Saved 2D stimulus-colored delay PCA: {filepath}")
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Continue with trajectory plots (keeping 2D for now due to complexity)
    # ... (rest of the original trajectory plotting code)
    
    return Z_delay, pca_delay.explained_variance_ratio_, np.unique(labels)


# ==============================================
# USAGE WITH FLAG
# ==============================================

if __name__ == "__main__":
    
    activation_file = "outputs/last_stand/dlygo_test_2/activations_epoch060.npz"
    
    timing_config = {
        "fixation_steps": 5,
        "stimulus_steps": 5,
        "delay_steps": 10,
        "decision_steps": 5
    }
    
    save_directory = "outputs/last_stand/stimulus_analysis"
    
    # ==============================================
    # NEW: FLAG TO CHOOSE 2D OR 3D PLOTTING
    # ==============================================
    plot_in_3d = True  # Set to False for 2D plotting
    
    print(f"=== ANALYSIS MODE: {'3D' if plot_in_3d else '2D'} ===")
    
    print("=== ORIGINAL ANALYSIS (Debug) ===")
    # Original analysis with 2D/3D option
    Z, explained_var, H_subset, fingerprint = reproduce_pca_plots_debug(
        activation_file,
        timing_cfg=timing_config,
        plot_percentage=100,
        show_plots=True,
        save_dir=save_directory,
        plot_3d=plot_in_3d  # NEW FLAG
    )
    
    print("\n=== NEW STIMULUS-COLORED ANALYSIS ===")
    # NEW: Stimulus-colored analysis with 2D/3D option
    Z_stim, explained_var_stim, unique_stimuli = analyze_by_stimulus_identity(
        activation_file,
        timing_cfg=timing_config,
        plot_percentage=100,
        show_plots=True,
        save_dir=save_directory,
        plot_3d=plot_in_3d  # NEW FLAG
    )
    
    print(f"\nAnalysis complete!")
    print(f"Found {len(unique_stimuli)} unique stimuli: {unique_stimuli}")
    print(f"PCA explained variance: {explained_var_stim[:3]}")
    
    # Optional: Test both 2D and 3D for comparison
    print("\n=== TESTING BOTH 2D AND 3D ===")
    for plot_mode in [False, True]:
        mode_name = "3D" if plot_mode else "2D"
        print(f"\n--- {mode_name} Mode ---")
        analyze_by_stimulus_identity(
            activation_file,
            timing_cfg=timing_config,
            plot_percentage=100,
            show_plots=False,  # Just save, don't show
            save_dir=save_directory,
            plot_3d=plot_mode
        )
