import ray.tune
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess
from pandas.plotting import parallel_coordinates
import matplotlib.cm as cm




# A list of experiment paths to load and combine
experiment_paths = [
    "/home/tim/query_optimization/hpo_results/optuna_20251220_170039/gbjo_hpo",
]

all_dfs = []
for path in experiment_paths:
    print(f"Loading results from: {path}")
    try:
        analysis = ExperimentAnalysis(path)
        all_dfs.append(analysis.results_df)
    except Exception as e:
        print(f"Warning: Could not load results from {path}. Error: {e}")

if not all_dfs:
    raise ValueError("No data could be loaded. Please check the paths in 'experiment_paths'.")

# Combine results from all experiments into a single DataFrame
df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined {len(all_dfs)} experiments, resulting in {len(df)} total trials.")

df.to_csv('results.csv', index=False)


# Create directory for plots
os.makedirs('analysis_plots', exist_ok=True)

# 1. Basic parameter analysis plot
plt.figure(figsize=(18, 8))

# Parameter importance plot
plt.subplot(2, 3, 1)
param_cols = [col for col in df.columns if col.startswith('config/')]
numeric_params = [col for col in param_cols if df[col].dtype in ['float64', 'int64']]
numeric_params = [col for col in numeric_params if col not in ['config/lambda_entropy', 'config/lr_warmup_steps',
                                                                'config/gradient_clip_norm', 'config/failure_rate', 'config/lambda_triple_in']]
correlations = df[numeric_params + ['mean_cost']].corr()['mean_cost'].sort_values()
sns.barplot(x=correlations.values[:-1], y=correlations.index[:-1])
plt.title("Parameter Importance")
plt.xticks(rotation=45)

# PLOT: Learning rate vs cost
plt.subplot(2, 3, 2)
sns.scatterplot(data=df, x='config/learning_rate', y='mean_cost')
plt.xscale('log')
plt.yscale('log')
plt.title("Learning Rate vs Cost")

# Optimization steps vs cost
plt.subplot(2, 3, 3)
sns.scatterplot(data=df, x='config/optimization_steps', y='mean_cost')
plt.yscale('log')
plt.title("Optimization Steps vs Cost")

# Lambda ramping effect
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='config/use_lambda_ramping', y='mean_cost')
plt.title("Effect of Lambda Ramping")

# Logit sampling effect
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='config/logit_sampling', y='mean_cost')
plt.title("Effect of Logit Sampling")

plt.tight_layout()
plt.savefig('analysis_plots/basic_analysis.png')
plt.close()


# 3. Parameter distributions for top vs bottom performers
plt.figure(figsize=(15, 10))
n_params = len(numeric_params)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

for i, param in enumerate(numeric_params, 1):
    plt.subplot(n_rows, n_cols, i)
    param_name = param.replace('config/', '')
    # Split into top 25% and bottom 75%
    threshold = df['mean_cost'].quantile(0.25)
    sns.kdeplot(data=df[df['mean_cost'] <= threshold], x=param, label='Top 25%')
    sns.kdeplot(data=df[df['mean_cost'] > threshold], x=param, label='Bottom 75%')
    plt.title(f'{param_name} Distribution')
    plt.legend()

plt.tight_layout()
plt.savefig('analysis_plots/param_distributions.png')
plt.close()

# 4. Top K trials visualization
k = 10
top_k = df.nsmallest(k, 'mean_cost')
numeric_top_k = top_k[numeric_params]

# Normalize the numeric parameters for better visualization
normalized_top_k = (numeric_top_k - numeric_top_k.min()) / (numeric_top_k.max() - numeric_top_k.min())

plt.figure(figsize=(12, 6))
sns.heatmap(normalized_top_k.T, 
            cmap='viridis',
            xticklabels=[f"Trial {i+1}" for i in range(k)],
            yticklabels=[col.replace('config/', '') for col in numeric_params])
plt.title(f'Normalized Parameter Values for Top {k} Trials')
plt.tight_layout()
plt.savefig('analysis_plots/top_k_heatmap.png')
plt.close()

# Print summary statistics
failure_threshold = 0.25  # Set maximum acceptable failure rate
filtered_df = df[df['failure_rate'] <= failure_threshold]
print(f"\nTop 5 configurations with failure rate <= {failure_threshold}:")
best_trials = filtered_df.nsmallest(10, 'mean_cost')

# Calculate average parameters for the top 5 trials
print(f"\nAverage parameters of top 5 configurations:")
print(f"Mean Cost: {best_trials['mean_cost'].mean():.2f}")
print(f"Mean Failure Rate: {best_trials['failure_rate'].mean():.3f}")

for param in param_cols:
    param_name = param.replace('config/', '')
    param_values = best_trials[param]
    
    if param in numeric_params:
        # Continuous parameter - calculate mean
        avg_value = param_values.mean()
        print(f"{param_name}: {avg_value:.3f} (mean)")
    else:
        # Discrete parameter - find mode (most frequent value)
        mode_value = param_values.mode()
        if len(mode_value) > 0:
            most_frequent = mode_value.iloc[0]
            frequency = (param_values == most_frequent).sum()
            print(f"{param_name}: {most_frequent} (mode, {frequency}/{len(param_values)} trials)")
        else:
            print(f"{param_name}: N/A (no clear mode)")

print(f"\nIndividual top 5 configurations:")
for i, (_, trial) in enumerate(best_trials.iterrows(), 1):
    print(f"\nTrial {i}:")
    print(f"  Mean Cost: {trial['mean_cost']:.2f}")
    print(f"  Failure Rate: {trial['failure_rate']:.3f}")
    for param in param_cols:
        param_name = param.replace('config/', '')
        if isinstance(trial[param], (int, float)):
            print(f"  {param_name}: {trial[param]:.3f}")
        else:
            print(f"  {param_name}: {trial[param]}")

# 5. Correlation matrix of parameters
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_params + ['mean_cost']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Parameter Correlation Matrix')
plt.tight_layout()
plt.savefig('analysis_plots/correlation_matrix.png')
plt.close()

# Cost Scatter  + Mean
plt.figure(figsize=(15, 10))
n_params = len(numeric_params)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

for i, param in enumerate(numeric_params, 1):
    # Create subplot with two y-axes
    ax1 = plt.subplot(n_rows, n_cols, i)
    #ax2 = ax1.twinx()
    param_name = param.replace('config/', '')
    
    # Sort the data by parameter value
    sorted_data = df.sort_values(by=param)
    
    # Apply LOWESS smoothing for mean_cost
    smoothed_cost = lowess(sorted_data['mean_cost'], 
                          sorted_data[param],
                          frac=0.3,  # span for smoothing
                          it=1)      # number of iterations
                          
    # Apply LOWESS smoothing for failure_rate
    smoothed_failure = lowess(sorted_data['failure_rate'],
                            sorted_data[param], 
                            frac=0.3,
                            it=1)
    
    # Plot raw data points and cost trend on left axis
    ax1.scatter(df[param], df['mean_cost'], alpha=0.2, color='gray', s=10)
    ax1.plot(smoothed_cost[:, 0], smoothed_cost[:, 1], color='red', linewidth=2, label='Mean Cost')
    
    # Plot failure rate trend on right axis
    #ax2.plot(smoothed_failure[:, 0], smoothed_failure[:, 1], color='blue', linewidth=2, label='Failure Rate')
    
    # Set axis labels and scales
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Mean Cost', color='red')
    #ax2.set_ylabel('Failure Rate', color='blue')
    
    if 'learning_rate' in param.lower():
        ax1.set_xscale('log')
    ax1.set_yscale('log')
    #ax1.set_ylim(10, 10**12)
    #ax2.set_ylim(0, 1)  # Set failure rate axis from 0 to 1
    
    plt.title(f'Cost vs {param_name}')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

plt.tight_layout()
plt.savefig('analysis_plots/parameter_cost_trends.pdf')
plt.close()

# Failure Rate Scatter + Mean Failure Rate Trend
plt.figure(figsize=(15, 10))
n_params = len(numeric_params)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

for i, param in enumerate(numeric_params, 1):
    plt.subplot(n_rows, n_cols, i)
    param_name = param.replace('config/', '')
    
    # Sort the data by parameter value
    sorted_data = df.sort_values(by=param)
    
    # Apply LOWESS smoothing for mean_cost
    smoothed_cost = lowess(sorted_data['failure_rate'], 
                          sorted_data[param],
                          frac=0.3,
                          it=1)
    
    # Plot failure rate scatter and mean cost trend
    plt.scatter(df[param], df['failure_rate'], alpha=0.2, color='gray', s=10, label='Failure Rate')
    plt.plot(smoothed_cost[:, 0], smoothed_cost[:, 1], color='red', linewidth=2, label='Mean Failure Rate')
    
    plt.xlabel(param_name)
    plt.ylabel('Value')
    
    if 'learning_rate' in param.lower():
        plt.xscale('log')
    
    plt.title(f'Failure Rate & Mean Cost vs {param_name}')
    plt.legend()

plt.tight_layout()
plt.savefig('analysis_plots/parameter_failure_trends.png')
plt.close()

# 6. Pareto front: Mean Cost vs Failure Rate

def compute_pareto_front(dataframe: pd.DataFrame, cost_col: str, fail_col: str) -> pd.DataFrame:
    """Return the subset of dataframe that lies on the Pareto front for two minimisation objectives."""
    # Sort by the first objective (cost) so we can do a single pass
    sorted_df = dataframe.sort_values(cost_col)
    pareto_rows = []
    min_failure = np.inf  # keep track of best (lowest) failure rate seen so far
    for _, row in sorted_df.iterrows():
        failure_val = row[fail_col]
        if failure_val < min_failure:
            pareto_rows.append(row)
            min_failure = failure_val
    return pd.DataFrame(pareto_rows)

# Compute Pareto-efficient trials
pareto_df = compute_pareto_front(df, cost_col="mean_cost", fail_col="failure_rate")

# Plot all trials and highlight the Pareto front
plt.figure(figsize=(8, 6))
plt.scatter(df['mean_cost'], df['failure_rate'], alpha=0.2, label='All Trials', color='gray', s=20)
plt.scatter(pareto_df['mean_cost'], pareto_df['failure_rate'], color='red', label='Pareto Front', s=40)
plt.plot(pareto_df['mean_cost'], pareto_df['failure_rate'], color='red', linewidth=2)
plt.xscale('log')
plt.xlabel('Mean Cost (log scale)')
plt.ylabel('Failure Rate')
plt.title('Pareto Front: Mean Cost vs Failure Rate')
plt.legend()
plt.tight_layout()
plt.savefig('analysis_plots/pareto_front.png')
plt.close()

















plt.figure(figsize=(24, 6))

# Prepare data for parallel coordinates
plot_df = df.copy()

# Include both numeric and boolean parameters
boolean_params = [col for col in param_cols if df[col].dtype == 'bool' or df[col].nunique() == 2]
all_plot_params = numeric_params + boolean_params

# Clean the data first - remove rows with any NaN values
plot_df = plot_df.dropna(subset=all_plot_params + ['mean_cost', 'failure_rate'])

# Create a subset for cleaner visualization (focus on more diverse sample)
if len(plot_df) > 30000000000000000000:
    # Stratified sampling across performance quantiles for better representation
    n_quantiles = 5
    plot_df['cost_quantile'] = pd.qcut(plot_df['mean_cost'], n_quantiles, labels=False)
    sampled_dfs = []
    for q in range(n_quantiles):
        q_data = plot_df[plot_df['cost_quantile'] == q]
        n_sample = min(60, len(q_data))  # 60 per quantile max
        sampled_dfs.append(q_data.sample(n=n_sample, random_state=42))
    plot_df = pd.concat(sampled_dfs)

print(f"Plotting {len(plot_df)} strategically sampled trials")

# Normalize all parameters to [0, 1]
plot_data = plot_df[all_plot_params].copy()
for col in all_plot_params:
    if col in boolean_params:
        plot_data[col] = plot_data[col].fillna(False).astype(int)
    else:
        col_min, col_max = plot_data[col].min(), plot_data[col].max()
        if col_max > col_min:
            plot_data[col] = (plot_data[col] - col_min) / (col_max - col_min)
        else:
            plot_data[col] = 0

# Use log-normalized mean_cost for better color gradation
log_cost = np.log10(plot_df['mean_cost'])
normalized_log_cost = (log_cost - log_cost.min()) / (log_cost.max() - log_cost.min())

# Create the plot with better layout
fig, ax = plt.subplots(figsize=(24, 6))

# Reorder parameters by importance for better visual flow
param_importance = df[all_plot_params + ['mean_cost']].corr()['mean_cost'].abs().sort_values(ascending=False)
ordered_params = [p for p in param_importance.index if p in all_plot_params]

# Set up x-axis positions
param_names = []
for param in ordered_params:
    name = param.replace('config/', '').replace('_', ' ').title()
    if param in boolean_params:
        name #+= '\n(Binary)'
    param_names.append(name)

x_pos = np.arange(len(param_names))

# Create enhanced colormap with more contrast
from matplotlib.colors import LinearSegmentedColormap
colors_list = ['#8b0000', '#ff4500', '#ffa500', '#ffff00', '#90ee90', '#32cd32', '#006400']  # Dark red to dark green
cmap = LinearSegmentedColormap.from_list('performance', colors_list, N=256)

# Plot lines with varying thickness based on performance
for idx, (_, row) in enumerate(plot_data.iterrows()):
    y_values = [row[param] for param in ordered_params]
    color_val = 1 - normalized_log_cost.iloc[idx]  # Invert so green = good
    
    # Vary line thickness - thicker for better performers
    line_width = 0.3 + (color_val * 1.5)  # 0.3 to 1.8
    alpha = 0.4 + (color_val * 0.4)  # 0.4 to 0.8
    
    ax.plot(x_pos, y_values, color=cmap(color_val), alpha=alpha, linewidth=line_width)

# Add trend lines for top 10% performers
top_10_pct = plot_df.nsmallest(int(len(plot_df) * 0.01), 'mean_cost')
if len(top_10_pct) > 3:
    top_means = []
    for param in ordered_params:
        if param in boolean_params:
            top_means.append(top_10_pct[param].fillna(False).astype(int).mean())
        else:
            param_data = top_10_pct[param]
            col_min, col_max = plot_data[param].min(), plot_data[param].max()
            if col_max > col_min:
                normalized_val = (param_data.mean() - df[param].min()) / (df[param].max() - df[param].min())
            else:
                normalized_val = 0
            top_means.append(normalized_val)
    

    from scipy.interpolate import make_interp_spline
    
    # Create a smooth spline through the top means
    x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
    spline = make_interp_spline(x_pos, top_means, k=3)  # k=3 for cubic spline
    y_smooth = spline(x_smooth)
    
    ax.plot(x_smooth, y_smooth, color='black', linewidth=4, alpha=0.8, 
            label='Top 1% Average')
    ax.scatter(x_pos, top_means, color='black', marker='x', s=80, zorder=5)
    #ax.plot(x_pos, top_means, color='black', linewidth=4, alpha=0.8, 
    #        label='Top 1% Average', linestyle='-', marker='x', markersize=8)

# Customize the plot with better styling
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=15)
ax.set_ylabel('Normalized Parameter Value', fontsize=18)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_ylim(-0.1, 1.1)

# Add horizontal reference lines
for y in [0, 0.25, 0.5, 0.75, 1.0]:
    ax.axhline(y=y, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

# Enhanced colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
cbar.set_label('Relative Cost', 
               fontsize=20, rotation=270, labelpad=25)
cbar.ax.tick_params(labelsize=24)

# Add legend for the trend line
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

# Add parameter importance annotations
for i, param in enumerate(ordered_params):
    importance = param_importance[param]
    ax.text(i, 1.0, f'r={importance:.2f}', ha='center', va='bottom', 
            fontsize=15, style='italic', color='black')

plt.tight_layout()
plt.savefig('analysis_plots/parallel_coordinates_enhanced.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis_plots/parallel_coordinates_enhanced.png', dpi=300, bbox_inches='tight')
plt.close()


exit()

