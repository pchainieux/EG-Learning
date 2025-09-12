import pandas as pd
import matplotlib.pyplot as plt

EG_COLOR = "#1f77b4"
GD_COLOR = "#d62728" 

def _set_pub_style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

eg_csv_files = ['outputs/ablation_eg_1.csv', 'outputs/ablation_eg_2.csv']
eg_dfs = [pd.read_csv(f) for f in eg_csv_files]
eg_df = pd.concat(eg_dfs, ignore_index=True)

gd_csv_files = ['outputs/ablation_gd_1.csv', 'outputs/ablation_gd_2.csv']
gd_dfs = [pd.read_csv(f) for f in gd_csv_files]
gd_df = pd.concat(gd_dfs, ignore_index=True)

metrics = {
    'Validation Accuracy': 'acc_10k',
    'AUC': 'aulc_10k',
    'Spectral Gap': 'spectral_gap',
}

def get_label(run):
    if 'control' in run:
        return 'Control'
    for key in ['row_sum', 'leak', 'nonlin', 'readout']:
        if key in run:
            condition = run.split(f"{key}/")[-1]
            return f"{key.replace('_', ' ').title()}: {condition.replace('_', '.')}"
    return run.split('/')[-1]

eg_df['Condition'] = eg_df['run'].apply(get_label)
gd_df['Condition'] = gd_df['run'].apply(get_label)

eg_summary = eg_df.groupby('Condition')[list(metrics.values())].agg(['mean', 'std'])
eg_summary.columns = ['_'.join(col) for col in eg_summary.columns]
eg_summary = eg_summary.reset_index()

gd_summary = gd_df.groupby('Condition')[list(metrics.values())].agg(['mean', 'std'])
gd_summary.columns = ['_'.join(col) for col in gd_summary.columns]
gd_summary = gd_summary.reset_index()

eg_summary.to_csv('eg_ablation_summary.csv', index=False)
gd_summary.to_csv('gd_ablation_summary.csv', index=False)

conditions = sorted(eg_summary['Condition'].unique())

for metric_name, col in metrics.items():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.35
    x_positions = range(len(conditions))
    
    eg_means = [eg_summary.loc[eg_summary['Condition'] == cond, f'{col}_mean'].values[0] 
                for cond in conditions]
    eg_stds = [eg_summary.loc[eg_summary['Condition'] == cond, f'{col}_std'].values[0] 
               for cond in conditions]
    
    gd_means = [gd_summary.loc[gd_summary['Condition'] == cond, f'{col}_mean'].values[0] 
                for cond in conditions]
    gd_stds = [gd_summary.loc[gd_summary['Condition'] == cond, f'{col}_std'].values[0] 
               for cond in conditions]
    
    bars_eg = ax.bar([x - bar_width/2 for x in x_positions], eg_means, 
                     yerr=eg_stds, width=bar_width, color=EG_COLOR, 
                     capsize=6, label='EG', alpha=0.8)
    
    bars_gd = ax.bar([x + bar_width/2 for x in x_positions], gd_means, 
                     yerr=gd_stds, width=bar_width, color=GD_COLOR, 
                     capsize=6, label='GD', alpha=0.8)
    
    _set_pub_style(ax)
    
#    ax.set_xlabel('Condition', fontsize=20)
    ax.set_ylabel(metric_name, fontsize=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=35, ha='right', fontsize=16)
    
    ax.legend(frameon=False, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'ablation_{col}_comparison_pubstyle.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Publication-style plots created and saved!")
