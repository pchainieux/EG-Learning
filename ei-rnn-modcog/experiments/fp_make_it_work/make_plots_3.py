import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

def reproduce_pca_plots_debug(activation_file, timing_cfg=None, show_plots=True,
                             plot_percentage=100, save_dir=None):
    """
    Reproduce PCA trajectory plots with extensive debugging
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
    # PLOT 1: Delay Period PCA (Original - colored by timestep)
    # ==============================================
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(Z_delay[:, 0], Z_delay[:, 1], c=labels_delay,
                         s=8, cmap="hsv", alpha=0.7)
    plt.colorbar(scatter, label="Target Action")
    plt.xlabel(f"PC1 (explains {pca_delay.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 (explains {pca_delay.explained_variance_ratio_[1]:.2%})")
    
    # Enhanced title showing percentage and data fingerprint
    title = f"Delay Dynamics (Last {plot_percentage}% = {steps_to_plot} steps)"
    if epoch is not None:
        title += f" - Epoch {epoch}"
    title += f"\nData: μ={H_delay_flat.mean():.3f}, σ={H_delay_flat.std():.3f}"
    plt.title(title, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save with more descriptive filename
    if save_dir is not None:
        epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
        filename = f"pca_delay_{epoch_str}_pct{plot_percentage}_steps{steps_to_plot}.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return Z_delay, pca_delay.explained_variance_ratio_, H_delay_subset, data_fingerprint

def analyze_by_stimulus_identity(activation_file, timing_cfg=None, show_plots=True, 
                                plot_percentage=100, save_dir=None):
    """
    NEW: Analyze and plot trajectories colored by stimulus identity
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
    # PLOT 1: Delay Period PCA (colored by stimulus)
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
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(Z_delay[:, 0], Z_delay[:, 1], c=labels_delay, 
                         s=12, cmap="hsv", alpha=0.8)
    plt.colorbar(scatter, label="Stimulus Identity")
    plt.xlabel(f"PC1 ({pca_delay.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_delay.explained_variance_ratio_[1]:.1%} variance)")
    
    title = f"Delay Period PCA - Colored by Stimulus Identity"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    if plot_percentage < 100:
        title += f"\nLast {plot_percentage}% of delay period"
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save delay period plot
    if save_dir is not None:
        epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
        filename = f"pca_delay_by_stimulus_{epoch_str}_pct{plot_percentage}.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"Saved stimulus-colored delay PCA: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # ==============================================
    # PLOT 2: Full Trial Trajectories (colored by stimulus)
    # ==============================================
    
    # PCA on full trial data
    pca_full = PCA(n_components=3).fit(H_arr.reshape(-1, H_arr.shape[-1]))
    
    plt.figure(figsize=(14, 10))
    
    # Get unique stimuli for color mapping
    unique_stimuli = np.unique(labels)
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_stimuli)))
    
    print(f"Plotting trajectories for {len(unique_stimuli)} unique stimuli")
    
    # Plot each stimulus group in a different color
    for stim_id, color in zip(unique_stimuli, colors):
        # Find trials with this stimulus
        trial_indices = np.where(labels == stim_id)[0]
        
        print(f"Stimulus {stim_id}: {len(trial_indices)} trials")
        
        for trial_idx in trial_indices:
            # Transform this trial to PC space
            Zi = pca_full.transform(H_arr[trial_idx])
            
            # Plot the entire trajectory in this stimulus's color with low alpha
            plt.plot(Zi[:, 0], Zi[:, 1], color=color, alpha=0.4, linewidth=0.8)
            
            # Highlight different periods with different line styles
            # Fixation (dotted)
            plt.plot(Zi[:fix_steps, 0], Zi[:fix_steps, 1], 
                    color=color, alpha=0.3, linewidth=0.5, linestyle=':')
            
            # Stimulus (dashed)
            plt.plot(Zi[fix_steps:fix_steps+stim_steps, 0], 
                    Zi[fix_steps:fix_steps+stim_steps, 1], 
                    color=color, alpha=0.6, linewidth=1.0, linestyle='--')
            
            # Delay period (thick solid) - this is the key!
            plt.plot(Zi[delay_start:delay_end, 0], Zi[delay_start:delay_end, 1], 
                    color=color, alpha=0.9, linewidth=2.5)
            
            # Decision (thin solid)
            if delay_end < H_arr.shape[1]:
                plt.plot(Zi[delay_end:, 0], Zi[delay_end:, 1], 
                        color=color, alpha=0.7, linewidth=1.0)
    
    plt.xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]:.1%} variance)")
    
    title = f"Trial Trajectories - Colored by Stimulus Identity"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    title += "\nThick lines = delay period"
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create custom legend for stimuli
    legend_elements = [plt.Line2D([0], [0], color=color, linewidth=2, 
                                 label=f'Stimulus {int(stim_id)}')
                      for stim_id, color in zip(unique_stimuli[:10], colors[:10])]  # Limit to first 10
    if len(unique_stimuli) <= 10:
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    else:
        plt.text(0.02, 0.98, f"{len(unique_stimuli)} unique stimuli", 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save trajectory plot
    if save_dir is not None:
        epoch_str = f"epoch{epoch:03d}" if epoch is not None else "unknown_epoch"
        filename = f"trajectories_by_stimulus_{epoch_str}_pct{plot_percentage}.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"Saved stimulus-colored trajectories: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return Z_delay, pca_delay.explained_variance_ratio_, unique_stimuli

def validate_percentage_differences(activation_file, percentages=[100, 75, 50, 25], timing_cfg=None):
    """
    Validate that different percentages actually produce different data
    """
    print(f"\n=== VALIDATING PERCENTAGE DIFFERENCES ===")
    print(f"File: {activation_file}")
    
    fingerprints = {}
    
    for pct in percentages:
        print(f"\n--- Testing {pct}% ---")
        try:
            Z, explained_var, H_subset, fingerprint = reproduce_pca_plots_debug(
                activation_file,
                timing_cfg=timing_cfg,
                plot_percentage=pct,
                show_plots=False,  # Don't show during validation
                save_dir=None
            )
            fingerprints[pct] = {
                'fingerprint': fingerprint,
                'shape': H_subset.shape,
                'explained_var': explained_var[:2],
                'pca_range': [Z[:, 0].max() - Z[:, 0].min(), Z[:, 1].max() - Z[:, 1].min()]
            }
            
        except Exception as e:
            print(f"Error with {pct}%: {e}")
    
    # Compare fingerprints
    print(f"\n=== COMPARISON SUMMARY ===")
    for pct, data in fingerprints.items():
        print(f"{pct:3d}%: Shape={data['shape']}, PC1 range={data['pca_range'][0]:.3f}, "
              f"PC1 var={data['explained_var'][0]:.3f}")
    
    # Check if all fingerprints are identical (indicating a problem)
    unique_fingerprints = set(data['fingerprint'] for data in fingerprints.values())
    if len(unique_fingerprints) == 1:
        print("\n⚠️  WARNING: All percentages have identical data fingerprints!")
        print("This suggests the delay dynamics are constant or there's a slicing bug.")
    else:
        print(f"\n✅ SUCCESS: Found {len(unique_fingerprints)} different data patterns.")
    
    return fingerprints

# ==============================================
# USAGE
# ==============================================

if __name__ == "__main__":
    
    activation_file = "outputs/last_stand/dlygo_test_2/activations_epoch060.npz"  # Use your latest epoch
    
    timing_config = {
        "fixation_steps": 5,
        "stimulus_steps": 5,
        "delay_steps": 10,
        "decision_steps": 5
    }
    
    save_directory = "outputs/last_stand/stimulus_analysis"
    
    print("=== ORIGINAL ANALYSIS (Debug) ===")
    # Original analysis
    Z, explained_var, H_subset, fingerprint = reproduce_pca_plots_debug(
        activation_file,
        timing_cfg=timing_config,
        plot_percentage=100,
        show_plots=True,
        save_dir=save_directory
    )
    
    print("\n=== NEW STIMULUS-COLORED ANALYSIS ===")
    # NEW: Stimulus-colored analysis
    Z_stim, explained_var_stim, unique_stimuli = analyze_by_stimulus_identity(
        activation_file,
        timing_cfg=timing_config,
        plot_percentage=100,  # Use full delay period
        show_plots=True,
        save_dir=save_directory
    )
    
    print(f"\nAnalysis complete!")
    print(f"Found {len(unique_stimuli)} unique stimuli: {unique_stimuli}")
    print(f"PCA explained variance: {explained_var_stim[:3]}")
    
    # Optional: Compare different percentages with stimulus coloring
    print("\n=== TESTING DIFFERENT DELAY PERCENTAGES ===")
    for pct in [100, 75, 50]:
        print(f"\n--- {pct}% of delay period ---")
        analyze_by_stimulus_identity(
            activation_file,
            timing_cfg=timing_config,
            plot_percentage=pct,
            show_plots=False,  # Just save, don't show
            save_dir=save_directory
        )
