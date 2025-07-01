import ray.tune
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess

# Use absolute path with file:// scheme
experiment_path = f"file://{os.path.abspath('/home/tim/query_optimization/ray_results/join_optim_hpo')}"
analysis = ExperimentAnalysis(experiment_path)

# Get results as DataFrame  
df = analysis.results_df

df.to_csv('ray_results/results.csv', index=False)


# Create directory for plots
os.makedirs('analysis_plots', exist_ok=True)

# 1. Basic parameter analysis plot
plt.figure(figsize=(12, 8))

# Parameter importance plot
plt.subplot(2, 2, 1)
param_cols = [col for col in df.columns if col.startswith('config/')]
numeric_params = [col for col in param_cols if df[col].dtype in ['float64', 'int64']]
correlations = df[numeric_params + ['mean_cost']].corr()['mean_cost'].sort_values()
sns.barplot(x=correlations.values[:-1], y=correlations.index[:-1])
plt.title("Parameter Importance")
plt.xticks(rotation=45)

# PLOT: Learning rate vs cost
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='config/learning_rate', y='mean_cost')
plt.xscale('log')
plt.yscale('log')
plt.title("Learning Rate vs Cost")

# Optimization steps vs cost
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='config/optimization_steps', y='mean_cost')
plt.yscale('log')
plt.title("Optimization Steps vs Cost")

# Lambda ramping effect
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='config/use_lambda_ramping', y='mean_cost')
plt.title("Effect of Lambda Ramping")

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
failure_threshold = 0.1  # Set maximum acceptable failure rate
filtered_df = df[df['failure_rate'] <= failure_threshold]
print(f"\nTop 5 configurations with failure rate <= {failure_threshold}:")
best_trials = filtered_df.nsmallest(5, 'mean_cost')
for _, trial in best_trials.iterrows():
    print(f"\nMean Cost: {trial['mean_cost']:.2f}")
    print(f"Failure Rate: {trial['failure_rate']:.3f}")
    for param in param_cols:
        param_name = param.replace('config/', '')
        if isinstance(trial[param], (int, float)):
            print(f"{param_name}: {trial[param]:.3f}")
        else:
            print(f"{param_name}: {trial[param]}")

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
    ax2 = ax1.twinx()
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
    ax2.plot(smoothed_failure[:, 0], smoothed_failure[:, 1], color='blue', linewidth=2, label='Failure Rate')
    
    # Set axis labels and scales
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Mean Cost', color='red')
    ax2.set_ylabel('Failure Rate', color='blue')
    
    if 'learning_rate' in param.lower():
        ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_ylim(0, 1)  # Set failure rate axis from 0 to 1
    
    plt.title(f'Cost & Failure Rate vs {param_name}')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('analysis_plots/parameter_cost_trends.png')
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