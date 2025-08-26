EG_COLOR = "#1f77b4"
GD_COLOR = "#d62728" 
TARGET_COLOR = "#6c757d" 

def _set_pub_style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)