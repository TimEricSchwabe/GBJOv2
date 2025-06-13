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
experiment_path = f"file://{os.path.abspath('ray_results/join_optim_hpofull')}"
analysis = ExperimentAnalysis(experiment_path)

# Get results as DataFrame
df = analysis.results_df

df.to_csv('ray_results/results.csv', index=False)

exit()

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

# Learning rate vs cost
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
print("\nTop 5 configurations:")
best_trials = df.nsmallest(5, 'mean_cost')
for _, trial in best_trials.iterrows():
    print(f"\nMean Cost: {trial['mean_cost']:.2f}")
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

# Add new plot for average cost across parameter values
plt.figure(figsize=(15, 10))
n_params = len(numeric_params)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

for i, param in enumerate(numeric_params, 1):
    plt.subplot(n_rows, n_cols, i)
    param_name = param.replace('config/', '')
    
    # Sort the data by parameter value
    sorted_data = df.sort_values(by=param)
    
    # Apply LOWESS smoothing
    smoothed = lowess(sorted_data['mean_cost'], 
                     sorted_data[param],
                     frac=0.3,  # span for smoothing
                     it=1)      # number of iterations
    
    # Plot raw data points with transparency
    plt.scatter(df[param], df['mean_cost'], alpha=0.2, color='gray', s=10)
    
    # Plot smoothed line
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
    
    plt.title(f'Average Cost vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Mean Cost')
    if 'learning_rate' in param.lower():
        plt.xscale('log')
    plt.yscale('log')

plt.tight_layout()
plt.savefig('analysis_plots/parameter_cost_trends.png')
plt.close()