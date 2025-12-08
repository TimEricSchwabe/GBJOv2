#!/usr/bin/env python3
"""
Plot optimization results from saved JSON data.

This script loads the detailed_results.json file from optimization runs
and creates visualizations.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional

import scienceplots
plt.style.use('science')


import argparse

RESULTS_DIR = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot optimization results from saved JSON data."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs='?',
        default="optimization_results/lubm_path_good_1",
        help="Directory containing detailed_results.json from an optimization run.",
    )
    return parser.parse_args()

OUTPUT_DIR = None 

# Plot type flags
SKIP_BOXPLOT = False
SKIP_BARPLOT = False
SKIP_SCATTER = True
SKIP_RATIOS = False
SKIP_SIZE_ANALYSIS = False
SKIP_SUMMARY = False
EXCLUDE_TRUE_COSTS = False  # New flag to exclude true costs from plots

# Data inclusion flags
INCLUDE_PREDICTED = True  # Include predicted costs in boxplot
EXCLUDE_EXHAUSTIVE = True  # Exclude exhaustive search from plots
EXCLUDE_GREEDY = False  # Exclude greedy method from plots
EXCLUDE_GRADIENT = False  # Exclude gradient method from plots
EXCLUDE_DP = False  # Exclude DP method from plots
USE_RANDOM = False  # Include random plans in plots

def load_optimization_results(results_dir: str) -> Dict[str, Any]:
    """
    Load optimization results from the specified directory.
    
    Args:
        results_dir: Directory containing detailed_results.json
        
    Returns:
        Dictionary containing the loaded results
    """
    results_file = os.path.join(results_dir, "detailed_results.json")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} query results from {results_file}")
    return data

def check_data_availability(data: List[Dict]) -> Dict[str, bool]:
    """
    Check what types of data are available in the loaded results.
    
    Args:
        data: List of query result dictionaries
        
    Returns:
        Dictionary indicating what data types are available
    """
    availability = {
        'exhaustive': False,
        'greedy': False,
        'gradient': False,
        'dp': False,
        'random': False,
        'exhaustive_real': False,
        'greedy_real': False,
        'gradient_real': False,
        'dp_real': False,
        'random_real': False,
        'exhaustive_pred': False,
        'greedy_pred': False,
        'gradient_pred': False,
        'dp_pred': False,
        'random_pred': False
    }
    
    if not data:
        return availability
    
    # Check ALL queries to see what methods are available
    # (some methods might be missing for some queries, e.g. DP for large queries)
    for query_result in data:
        if 'plans' in query_result:
            for method in ['exhaustive', 'greedy', 'gradient', 'dp', 'random']:
                if method in query_result['plans']:
                    availability[method] = True
                    # Check if real and predicted costs are available
                    plan_data = query_result['plans'][method]
                    if 'real_cost' in plan_data and plan_data['real_cost'] is not None:
                        availability[f'{method}_real'] = True
                    if 'predicted_cost' in plan_data and plan_data['predicted_cost'] is not None:
                        availability[f'{method}_pred'] = True

    print("Data availability:")
    for method in ['exhaustive', 'greedy', 'gradient', 'dp', 'random']:
        if availability[method]:
            real_status = '✓' if availability[f'{method}_real'] else '✗'
            pred_status = '✓' if availability[f'{method}_pred'] else '✗'
            print(f"  {method.capitalize()}: Method ✓, Real costs {real_status}, Predicted costs {pred_status}")
        else:
            print(f"  {method.capitalize()}: ✗")
    
    return availability

def extract_costs_and_metrics(data: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract costs and metrics from the loaded data and convert to format expected by plot_statistics.
    
    Args:
        data: List of query result dictionaries
        
    Returns:
        Dictionary containing extracted metrics in the format expected by plot_statistics
    """
    # Check what data is available
    availability = check_data_availability(data)
    
    stats = {
        'gradient_costs': [],
        'greedy_costs': [],
        'random_costs': [], 
        'predicted_best_costs': [], 
        'predicted_exhaustive_costs': [],  
        'predicted_gradient_costs': [],
        'predicted_greedy_costs': [],
        'predicted_random_costs': [],  
        'true_best_predicted_costs': [],  
        'exhaustive_real': [],
        'exhaustive_pred': [],
        'greedy_real': [],
        'greedy_pred': [],
        'gradient_real': [],
        'gradient_pred': [],
        'dp_real': [],
        'dp_pred': [],
        'random_real': [],
        'random_pred': [],
        'greedy_equal_exhaustive': [],
        'gradient_equal_exhaustive': [],
        'query_sizes': [],
        'predicted_best_costs_filtered': [],  
        'predicted_gradient_costs_filtered': [],  
        'predicted_greedy_costs_filtered': [],  
        'predicted_exhaustive_costs_filtered': []  
    }
    
    for query_result in data:
        
        if 'plans' not in query_result:
            continue
            
        # Check for infinite costs in available methods and skip if any are infinite
        infinite_costs = []
        
        # Only check for infinite costs if real costs are available
        if availability['exhaustive_real'] and 'exhaustive' in query_result['plans']:
            real_cost = query_result['plans']['exhaustive'].get('real_cost')
            if real_cost == float('inf'):
                infinite_costs.append('exhaustive')
        
        if availability['greedy_real'] and 'greedy' in query_result['plans']:
            real_cost = query_result['plans']['greedy'].get('real_cost')
            if real_cost == float('inf'):
                infinite_costs.append('greedy')
                
        if availability['gradient_real'] and 'gradient' in query_result['plans']:
            real_cost = query_result['plans']['gradient'].get('real_cost')
            if real_cost == float('inf'):
                infinite_costs.append('gradient')
        
        # Only check random costs if we're actually using random in the plots
        if availability['random_real'] and 'random' in query_result['plans'] and USE_RANDOM:
            real_cost = query_result['plans']['random'].get('real_cost')
            if real_cost == float('inf'):
                infinite_costs.append('random')
        
        # Skip this query if any available method has infinite real cost
        if infinite_costs:
            continue
        
        # Extract costs for available methods
        # Gradient
        if availability['gradient']:
            if 'gradient' in query_result['plans']:
                plan_data = query_result['plans']['gradient']
                
                if availability['gradient_real']:
                    real_cost = plan_data.get('real_cost') if 'real_cost' in plan_data else None
                    if real_cost is not None and real_cost != float('inf'):
                        stats['gradient_costs'].append(real_cost)
                        stats['gradient_real'].append(real_cost)
                    else:
                        stats['gradient_costs'].append(np.nan)
                        stats['gradient_real'].append(np.nan)
                
                if availability['gradient_pred']:
                    pred_cost = plan_data.get('predicted_cost') if 'predicted_cost' in plan_data else None
                    if pred_cost is not None and pred_cost != float('inf'):
                        stats['gradient_pred'].append(pred_cost)
                        stats['predicted_gradient_costs'].append(pred_cost)
                    else:
                        stats['gradient_pred'].append(np.nan)
                        stats['predicted_gradient_costs'].append(np.nan)
            else:
                if availability['gradient_real']:
                    stats['gradient_costs'].append(np.nan)
                    stats['gradient_real'].append(np.nan)
                if availability['gradient_pred']:
                    stats['gradient_pred'].append(np.nan)
                    stats['predicted_gradient_costs'].append(np.nan)
        
        # Greedy
        if availability['greedy']:
            if 'greedy' in query_result['plans']:
                plan_data = query_result['plans']['greedy']
                
                if availability['greedy_real']:
                    real_cost = plan_data.get('real_cost') if 'real_cost' in plan_data else None
                    if real_cost is not None and real_cost != float('inf'):
                        stats['greedy_costs'].append(real_cost)
                        stats['greedy_real'].append(real_cost)
                    else:
                        stats['greedy_costs'].append(np.nan)
                        stats['greedy_real'].append(np.nan)
                
                if availability['greedy_pred']:
                    pred_cost = plan_data.get('predicted_cost') if 'predicted_cost' in plan_data else None
                    if pred_cost is not None and pred_cost != float('inf'):
                        stats['greedy_pred'].append(pred_cost)
                        stats['predicted_greedy_costs'].append(pred_cost)
                    else:
                        stats['greedy_pred'].append(np.nan)
                        stats['predicted_greedy_costs'].append(np.nan)
            else:
                if availability['greedy_real']:
                    stats['greedy_costs'].append(np.nan)
                    stats['greedy_real'].append(np.nan)
                if availability['greedy_pred']:
                    stats['greedy_pred'].append(np.nan)
                    stats['predicted_greedy_costs'].append(np.nan)
        
        # DP
        if availability['dp']:
            if 'dp' in query_result['plans']:
                plan_data = query_result['plans']['dp']
                
                if availability['dp_real']:
                    real_cost = plan_data.get('real_cost') if 'real_cost' in plan_data else None
                    if real_cost is not None and real_cost != float('inf'):
                        stats['random_costs'].append(real_cost)  # Use DP as "random" for legacy reasons
                        stats['true_best_predicted_costs'].append(real_cost)
                        stats['dp_real'].append(real_cost)
                    else:
                        stats['random_costs'].append(np.nan)
                        stats['true_best_predicted_costs'].append(np.nan)
                        stats['dp_real'].append(np.nan)
                
                if availability['dp_pred']:
                    pred_cost = plan_data.get('predicted_cost') if 'predicted_cost' in plan_data else None
                    if pred_cost is not None and pred_cost != float('inf'):
                        stats['predicted_best_costs'].append(pred_cost)
                        stats['dp_pred'].append(pred_cost)
                    else:
                        stats['predicted_best_costs'].append(np.nan)
                        stats['dp_pred'].append(np.nan)
            else:
                if availability['dp_real']:
                    stats['random_costs'].append(np.nan)
                    stats['true_best_predicted_costs'].append(np.nan)
                    stats['dp_real'].append(np.nan)
                if availability['dp_pred']:
                    stats['predicted_best_costs'].append(np.nan)
                    stats['dp_pred'].append(np.nan)
        
        # Random
        if availability['random']:
            if 'random' in query_result['plans']:
                plan_data = query_result['plans']['random']
                
                if availability['random_real']:
                    real_cost = plan_data.get('real_cost') if 'real_cost' in plan_data else None
                    if real_cost is not None and real_cost != float('inf'):
                        stats['random_real'].append(real_cost)
                    else:
                        stats['random_real'].append(np.nan)
                
                if availability['random_pred']:
                    pred_cost = plan_data.get('predicted_cost') if 'predicted_cost' in plan_data else None
                    if pred_cost is not None and pred_cost != float('inf'):
                        stats['random_pred'].append(pred_cost)
                        stats['predicted_random_costs'].append(pred_cost)
                    else:
                        stats['random_pred'].append(np.nan)
                        stats['predicted_random_costs'].append(np.nan)
            else:
                if availability['random_real']:
                    stats['random_real'].append(np.nan)
                if availability['random_pred']:
                    stats['random_pred'].append(np.nan)
                    stats['predicted_random_costs'].append(np.nan)
        
        # Exhaustive
        if availability['exhaustive']:
            if 'exhaustive' in query_result['plans']:
                plan_data = query_result['plans']['exhaustive']
                
                if availability['exhaustive_real']:
                    real_cost = plan_data.get('real_cost') if 'real_cost' in plan_data else None
                    if real_cost is not None and real_cost != float('inf'):
                        stats['exhaustive_real'].append(real_cost)
                    else:
                        stats['exhaustive_real'].append(np.nan)
                
                if availability['exhaustive_pred']:
                    pred_cost = plan_data.get('predicted_cost') if 'predicted_cost' in plan_data else None
                    if pred_cost is not None and pred_cost != float('inf'):
                        stats['predicted_exhaustive_costs'].append(pred_cost)
                        stats['exhaustive_pred'].append(pred_cost)
                    else:
                        stats['predicted_exhaustive_costs'].append(np.nan)
                        stats['exhaustive_pred'].append(np.nan)
            else:
                if availability['exhaustive_real']:
                    stats['exhaustive_real'].append(np.nan)
                if availability['exhaustive_pred']:
                    stats['predicted_exhaustive_costs'].append(np.nan)
                    stats['exhaustive_pred'].append(np.nan)
        
        if availability['exhaustive']:
            if 'greedy_equal_exhaustive' in query_result:
                stats['greedy_equal_exhaustive'].append(query_result['greedy_equal_exhaustive'])
            else:
                stats['greedy_equal_exhaustive'].append(False)
                
            if 'gradient_equal_exhaustive' in query_result:
                stats['gradient_equal_exhaustive'].append(query_result['gradient_equal_exhaustive'])
            else:
                stats['gradient_equal_exhaustive'].append(False)
        
        # Extract query size
        if 'ntriplepattern' in query_result:
            stats['query_sizes'].append(query_result['ntriplepattern'])
    
    # Get the maximum length across all populated lists to report
    non_empty_lists = [v for v in stats.values() if isinstance(v, list) and len(v) > 0]
    max_length = max(len(v) for v in non_empty_lists) if non_empty_lists else 0
    
    print(f"Extracted metrics for {max_length} valid queries")
    print(f"Available methods: {[method for method in ['exhaustive', 'greedy', 'gradient', 'dp', 'random'] if availability[method]]}")
    print(f"Real costs available for: {[method for method in ['exhaustive', 'greedy', 'gradient', 'dp', 'random'] if availability[f'{method}_real']]}")
    print(f"Predicted costs available for: {[method for method in ['exhaustive', 'greedy', 'gradient', 'dp', 'random'] if availability[f'{method}_pred']]}")
    
    # Create filtered arrays for scatter plots (only include data where DP is available)
    if len(stats['query_sizes']) > 0:
        for i, query_size in enumerate(stats['query_sizes']):
            # Check if DP predicted cost is available (not NaN)
            if i < len(stats['predicted_best_costs']) and stats['predicted_best_costs'][i] is not None and not np.isnan(stats['predicted_best_costs'][i]):
                stats['predicted_best_costs_filtered'].append(stats['predicted_best_costs'][i])
                
                # Only add other method costs if DP cost is available at this index
                if i < len(stats['predicted_gradient_costs']):
                    stats['predicted_gradient_costs_filtered'].append(stats['predicted_gradient_costs'][i])
                if i < len(stats['predicted_greedy_costs']):
                    stats['predicted_greedy_costs_filtered'].append(stats['predicted_greedy_costs'][i])
                if i < len(stats['predicted_exhaustive_costs']):
                    stats['predicted_exhaustive_costs_filtered'].append(stats['predicted_exhaustive_costs'][i])
    
    print(f"Filtered arrays for scatter plots (DP available): {len(stats['predicted_best_costs_filtered'])} queries")
    
    # Store availability info in stats for use in plotting
    stats['_availability'] = availability
    
    return stats

def plot_statistics(stats, show_plots=True, suffix="", save_directory="."):
    """
    
    Args:
        stats: Dictionary with statistics from evaluate_optimization
        show_plots: Whether to display the plots (if False, only save them)
        suffix: Optional suffix to add to saved filenames (e.g., "_iteration_10")
        save_directory: Directory to save the plots to
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Get availability info
    availability = stats.get('_availability', {})
    has_exhaustive = availability.get('exhaustive', False)
    has_greedy = availability.get('greedy', False)
    has_gradient = availability.get('gradient', False)
    has_dp = availability.get('dp', False)
    
    # Check for real cost availability
    has_exhaustive_real = availability.get('exhaustive_real', False)
    has_greedy_real = availability.get('greedy_real', False)
    has_gradient_real = availability.get('gradient_real', False)
    has_dp_real = availability.get('dp_real', False)
    
    # Check for predicted cost availability
    has_exhaustive_pred = availability.get('exhaustive_pred', False)
    has_greedy_pred = availability.get('greedy_pred', False)
    has_gradient_pred = availability.get('gradient_pred', False)
    has_dp_pred = availability.get('dp_pred', False)
    has_random_pred = availability.get('random_pred', False)
    
    # Check if we have any real costs at all
    has_any_real_costs = has_exhaustive_real or has_greedy_real or has_gradient_real or has_dp_real
    
    # Calculate mean costs for different strategies (only if real costs are available)
    if not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        if has_gradient_real and len(stats['gradient_costs']) > 0:
            mean_gradient = np.nanmean(stats['gradient_costs'])
            if has_greedy_real and len(stats['greedy_costs']) > 0:
                mean_greedy = np.nanmean(stats['greedy_costs'])
            if has_dp_real and len(stats['random_costs']) > 0:
                mean_random = np.nanmean(stats['random_costs'])
    
    has_predicted = 'predicted_best_costs' in stats and len(stats['predicted_best_costs']) > 0
    has_pred_grad = 'predicted_gradient_costs' in stats and len(stats['predicted_gradient_costs']) > 0
    has_pred_greedy = 'predicted_greedy_costs' in stats and len(stats['predicted_greedy_costs']) > 0
    has_pred_random = 'predicted_random_costs' in stats and len(stats['predicted_random_costs']) > 0
    has_true_best = 'true_best_predicted_costs' in stats and len(stats['true_best_predicted_costs']) > 0
    has_exhaustive_pred_data = has_exhaustive_pred and 'predicted_exhaustive_costs' in stats and len(stats['predicted_exhaustive_costs']) > 0
    
    if has_predicted:
        mean_predicted = np.nanmean(stats['predicted_best_costs'])
    if has_pred_grad:
        mean_pred_grad = np.nanmean(stats['predicted_gradient_costs'])
    if has_pred_greedy:
        mean_pred_greedy = np.nanmean(stats['predicted_greedy_costs'])
    if has_pred_random and USE_RANDOM:
        mean_pred_random = np.nanmean(stats['predicted_random_costs'])
    if has_true_best and not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        mean_true_best = np.nanmean(stats['true_best_predicted_costs'])
    if has_exhaustive_pred_data:
        mean_exhaustive = np.nanmean(stats['predicted_exhaustive_costs'])
    
    # Plot mean costs comparison
    plt.figure(figsize=(12, 6))
    
    labels = []
    means = []
    
    if not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        if has_gradient_real and len(stats['gradient_costs']) > 0:
            labels.append('Gradient')
            means.append(mean_gradient)
        if has_greedy_real and len(stats['greedy_costs']) > 0:
            labels.append('Greedy')
            means.append(mean_greedy)
        if has_dp_real and len(stats['random_costs']) > 0:
            labels.append('DP')
            means.append(mean_random)
    
    if has_predicted:
        labels.append('DP-Best')
        means.append(mean_predicted)
    if has_exhaustive_pred_data:
        labels.append('Exhaustive')
        means.append(mean_exhaustive)
    if has_pred_grad:
        labels.append('GradPred')
        means.append(mean_pred_grad)
    if has_pred_greedy:
        labels.append('GreedyPred')
        means.append(mean_pred_greedy)
    if has_pred_random and USE_RANDOM:
        labels.append('RandomPred')
        means.append(mean_pred_random)
    if has_true_best and not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        labels.append('TrueBestPred')
        means.append(mean_true_best)
    
    if not labels:
        print("Warning: No data available for mean costs comparison plot")
        return
    
    bar_colors_master = ['blue', 'green', 'orange', 'purple', 'cyan', 'red', 'brown', 'pink']
    plt.bar(labels, means, color=bar_colors_master[:len(labels)])
    plt.ylabel('Mean Cost')
    plt.title('Comparison of Mean Costs')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(means):
        plt.text(i, v * 1.05, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'mean_costs_comparison{suffix}.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot cost comparison as boxplot (log scale)
    plt.figure(figsize=(12, 6))
    
    data = []
    labels_box = []
    
    # Only include real costs if they're available and not excluded
    if not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        if has_gradient_real and len(stats['gradient_costs']) > 0:
            costs = [c for c in stats['gradient_costs'] if c is not None and not np.isnan(c)]
            if costs:
                data.append(costs)
                labels_box.append('Gradient')
        if has_greedy_real and len(stats['greedy_costs']) > 0:
            costs = [c for c in stats['greedy_costs'] if c is not None and not np.isnan(c)]
            if costs:
                data.append(costs)
                labels_box.append('Greedy')
        if has_dp_real and len(stats['random_costs']) > 0:
            costs = [c for c in stats['random_costs'] if c is not None and not np.isnan(c)]
            if costs:
                data.append(costs)
                labels_box.append('DP')
    
    # Always include predicted costs if available
    if has_predicted:
        costs = [c for c in stats['predicted_best_costs'] if c is not None and not np.isnan(c)]
        if costs:
            data.append(costs)
            labels_box.append('DP-Best')
    if has_exhaustive_pred_data:
        costs = [c for c in stats['predicted_exhaustive_costs'] if c is not None and not np.isnan(c)]
        if costs:
            data.append(costs)
            labels_box.append('Exhaustive')
    if has_pred_grad:
        costs = [c for c in stats['predicted_gradient_costs'] if c is not None and not np.isnan(c)]
        if costs:
            data.append(costs)
            labels_box.append('GradPred')
    if has_pred_greedy:
        costs = [c for c in stats['predicted_greedy_costs'] if c is not None and not np.isnan(c)]
        if costs:
            data.append(costs)
            labels_box.append('GreedyPred')
    if has_pred_random and USE_RANDOM:
        costs = [c for c in stats['predicted_random_costs'] if c is not None and not np.isnan(c)]
        if costs:
            data.append(costs)
            labels_box.append('RandomPred')
    if has_true_best and not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        costs = [c for c in stats['true_best_predicted_costs'] if c is not None and not np.isnan(c)]
        if costs:
            data.append(costs)
            labels_box.append('TrueBestPred')
    
    if data:
        plt.boxplot(data, labels=labels_box)
        plt.yscale('log')
        plt.ylabel('Cost (log scale)')
        plt.title('Cost Distribution Comparison')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'cost_distribution_comparison{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("Warning: No data available for cost distribution comparison plot")
    
    # Calculate and print ratio comparisons if not excluding true costs and real costs are available
    if not EXCLUDE_TRUE_COSTS and has_dp_real:
        if has_gradient_real and len(stats['gradient_costs']) > 0 and len(stats['random_costs']) > 0:
            # Create masked arrays to handle NaNs
            grad_costs = np.array(stats['gradient_costs'])
            dp_costs = np.array(stats['random_costs'])
            
            # Filter where both are valid
            valid_mask = ~(np.isnan(grad_costs) | np.isnan(dp_costs))
            
            if np.sum(valid_mask) > 0:
                valid_grad = grad_costs[valid_mask]
                valid_dp = dp_costs[valid_mask]
                
                gradient_to_random_ratio = np.mean(valid_grad / valid_dp)
                print(f"Mean ratio of gradient optimizer cost to DP cost: {gradient_to_random_ratio:.2f}x")
                
                # Calculate how often gradient beats DP
                gradient_wins = np.sum(valid_grad < valid_dp)
                gradient_win_pct = gradient_wins / len(valid_grad) * 100
                print(f"Gradient optimizer beats DP in {gradient_win_pct:.1f}% of queries ({gradient_wins}/{len(valid_grad)})")
            else:
                gradient_win_pct = 0
        
        if has_greedy_real and len(stats['greedy_costs']) > 0 and len(stats['random_costs']) > 0:
            # Create masked arrays to handle NaNs
            greedy_costs_arr = np.array(stats['greedy_costs'])
            dp_costs = np.array(stats['random_costs'])
            
            # Filter where both are valid
            valid_mask = ~(np.isnan(greedy_costs_arr) | np.isnan(dp_costs))
            
            if np.sum(valid_mask) > 0:
                valid_greedy = greedy_costs_arr[valid_mask]
                valid_dp = dp_costs[valid_mask]
                
                greedy_to_random_ratio = np.mean(valid_greedy / valid_dp)
                print(f"Mean ratio of greedy heuristic cost to DP cost: {greedy_to_random_ratio:.2f}x")
                
                # Calculate how often greedy beats DP
                greedy_wins = np.sum(valid_greedy < valid_dp)
                greedy_win_pct = greedy_wins / len(valid_greedy) * 100
                print(f"Greedy heuristic beats DP in {greedy_win_pct:.1f}% of queries ({greedy_wins}/{len(valid_greedy)})")
            else:
                greedy_win_pct = 0
        
        # Plot win percentage (only if we have both methods with real costs)
        if (has_gradient_real and len(stats['gradient_costs']) > 0 and 
            has_greedy_real and len(stats['greedy_costs']) > 0):
            
            plt.figure(figsize=(8, 6))
            win_pcts = []
            win_labels = []
            
            if has_gradient_real:
                win_pcts.append(gradient_win_pct)
                win_labels.append('Gradient vs. DP')
            if has_greedy_real:
                win_pcts.append(greedy_win_pct)
                win_labels.append('Greedy vs. DP')
            
            plt.bar(win_labels, win_pcts, color=['blue', 'green'][:len(win_pcts)])
            plt.ylabel('Win Percentage (%)')
            plt.title('Percentage of Queries Where Optimizer Beats DP')
            plt.ylim(0, 100)
            
            # Add percentage labels on bars
            for i, v in enumerate(win_pcts):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'win_percentage{suffix}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()
    elif not EXCLUDE_TRUE_COSTS and not has_any_real_costs:
        print("Note: Real costs not available - ratio comparisons and win rates cannot be calculated")

    # Only show scatter plots of true costs if not excluded and real costs are available
    if not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        # Plot scatter of gradient vs greedy costs
        if (has_gradient_real and has_greedy_real and 
            len(stats['gradient_costs']) > 0 and len(stats['greedy_costs']) > 0):
            
            gradient_costs = np.array(stats['gradient_costs'])
            greedy_costs = np.array(stats['greedy_costs'])
            
            plt.figure(figsize=(10, 8))
            plt.scatter(gradient_costs, greedy_costs, alpha=0.7, s=70, c='blue', edgecolors='black')
            
            max_val = max(np.nanmax(gradient_costs), np.nanmax(greedy_costs))
            min_val = min(np.nanmin(gradient_costs), np.nanmin(greedy_costs))
            line_min = min_val * 0.9
            line_max = max_val * 1.1
            plt.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.7)
            
            plt.xlabel('Gradient-Based Optimization Cost')
            plt.ylabel('Greedy Optimization Cost')
            plt.title('Gradient vs Greedy Optimization Cost Comparison')
            plt.grid(alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'gradient_vs_greedy{suffix}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # Plot scatter of gradient vs DP costs
        if (has_gradient_real and has_dp_real and 
            len(stats['gradient_costs']) > 0 and len(stats['random_costs']) > 0):
            
            gradient_costs = np.array(stats['gradient_costs'])
            random_costs = np.array(stats['random_costs'])
            
            plt.figure(figsize=(10, 8))
            plt.scatter(gradient_costs, random_costs, alpha=0.7, s=70, c='orange', edgecolors='black')
            
            max_val = max(np.nanmax(gradient_costs), np.nanmax(random_costs))
            min_val = min(np.nanmin(gradient_costs), np.nanmin(random_costs))
            line_min = min_val * 0.9
            line_max = max_val * 1.1
            plt.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.7)
            
            plt.xlabel('Gradient-Based Optimization Cost')
            plt.ylabel('DP Cost')
            plt.title('Gradient vs DP Cost Comparison')
            plt.grid(alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'gradient_vs_random{suffix}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # Plot scatter of greedy vs DP costs
        if (has_greedy_real and has_dp_real and 
            len(stats['greedy_costs']) > 0 and len(stats['random_costs']) > 0):
            
            greedy_costs = np.array(stats['greedy_costs'])
            random_costs = np.array(stats['random_costs'])
            
            plt.figure(figsize=(10, 8))
            plt.scatter(greedy_costs, random_costs, alpha=0.7, s=70, c='orange', edgecolors='black')
            
            max_val = max(np.nanmax(greedy_costs), np.nanmax(random_costs))
            min_val = min(np.nanmin(greedy_costs), np.nanmin(random_costs))
            line_min = min_val * 0.9
            line_max = max_val * 1.1
            plt.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.7)
            
            plt.xlabel('Greedy Optimization Cost')
            plt.ylabel('DP Cost')
            plt.title('Greedy vs DP Cost Comparison')
            plt.grid(alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'greedy_vs_random{suffix}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()
    elif not EXCLUDE_TRUE_COSTS and not has_any_real_costs:
        print("Note: Real costs not available - real cost scatter plots will be skipped")

    # NEW: Combined scatter plot of predicted costs vs exhaustive (only if exhaustive predicted data exists)
    if has_pred_grad and has_pred_greedy and has_exhaustive_pred_data:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_gradient_costs'], stats['predicted_exhaustive_costs'], 
                   alpha=0.7, s=70, c='blue', edgecolors='black', label='Gradient')
        plt.scatter(stats['predicted_greedy_costs'], stats['predicted_exhaustive_costs'], 
                   alpha=0.7, s=70, c='green', edgecolors='black', label='Greedy')
        
        # Get min and max for both methods
        all_pred_costs = np.concatenate([stats['predicted_gradient_costs'], 
                                       stats['predicted_greedy_costs'], 
                                       stats['predicted_exhaustive_costs']])
        min_val = np.nanmin(all_pred_costs) * 0.9
        max_val = np.nanmax(all_pred_costs) * 1.1
        
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Predicted Cost (Gradient/Greedy)')
        plt.ylabel('Predicted Cost (Exhaustive)')
        plt.title('Predicted Costs vs Exhaustive Search Predictions')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'predicted_costs_vs_exhaustive{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_predicted:
        # Gradient vs best predicted (predicted cost)
        if has_pred_grad:
            plt.figure(figsize=(10, 8))
            plt.scatter(stats['predicted_gradient_costs_filtered'], stats['predicted_best_costs_filtered'], alpha=0.7, s=70, c='purple', edgecolors='black')
            if len(stats['predicted_gradient_costs_filtered']) > 0 and len(stats['predicted_best_costs_filtered']) > 0:
                min_val = min(np.nanmin(stats['predicted_gradient_costs_filtered']), np.nanmin(stats['predicted_best_costs_filtered'])) * 0.9
                max_val = max(np.nanmax(stats['predicted_gradient_costs_filtered']), np.nanmax(stats['predicted_best_costs_filtered'])) * 1.1
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            plt.xscale('log'); plt.yscale('log')
            plt.xlabel('Predicted Gradient Cost')
            plt.ylabel('Best Predicted Cost')
            plt.title('Predicted Gradient vs Best-Predicted Cost')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'pred_gradient_vs_best_predicted{suffix}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # Greedy vs best predicted (predicted cost)
        if has_pred_greedy:
            plt.figure(figsize=(10, 8))
            plt.scatter(stats['predicted_greedy_costs_filtered'], stats['predicted_best_costs_filtered'], alpha=0.7, s=70, c='purple', edgecolors='black')
            if len(stats['predicted_greedy_costs_filtered']) > 0 and len(stats['predicted_best_costs_filtered']) > 0:
                min_val = min(np.nanmin(stats['predicted_greedy_costs_filtered']), np.nanmin(stats['predicted_best_costs_filtered'])) * 0.9
                max_val = max(np.nanmax(stats['predicted_greedy_costs_filtered']), np.nanmax(stats['predicted_best_costs_filtered'])) * 1.1
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            plt.xscale('log'); plt.yscale('log')
            plt.xlabel('Predicted Greedy Cost')
            plt.ylabel('Best Predicted Cost')
            plt.title('Predicted Greedy vs Best-Predicted Cost')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_directory, f'pred_greedy_vs_best_predicted{suffix}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

    if has_pred_grad and has_pred_greedy:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_gradient_costs'], stats['predicted_greedy_costs'], alpha=0.7, s=70, c='brown', edgecolors='black')
        mn = min(np.nanmin(stats['predicted_gradient_costs']), np.nanmin(stats['predicted_greedy_costs'])) * 0.9
        mx = max(np.nanmax(stats['predicted_gradient_costs']), np.nanmax(stats['predicted_greedy_costs'])) * 1.1
        plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Predicted Gradient Cost')
        plt.ylabel('Predicted Greedy Cost')
        plt.title('Predicted Gradient vs Predicted Greedy Cost')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'pred_gradient_vs_pred_greedy{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_pred_grad and has_predicted:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_gradient_costs_filtered'], stats['predicted_best_costs_filtered'], alpha=0.7, s=70, c='darkgreen', edgecolors='black')
        if len(stats['predicted_gradient_costs_filtered']) > 0 and len(stats['predicted_best_costs_filtered']) > 0:
            mn = min(np.nanmin(stats['predicted_gradient_costs_filtered']), np.nanmin(stats['predicted_best_costs_filtered'])) * 0.9
            mx = max(np.nanmax(stats['predicted_gradient_costs_filtered']), np.nanmax(stats['predicted_best_costs_filtered'])) * 1.1
            plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Predicted Gradient Cost')
        plt.ylabel('Exhaustive Best Predicted Cost')
        plt.title('Predicted Gradient vs Exhaustive Best Predicted')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'pred_gradient_vs_exhaustive_pred{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_pred_greedy and has_predicted:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_greedy_costs_filtered'], stats['predicted_best_costs_filtered'], alpha=0.7, s=70, c='darkorange', edgecolors='black')
        if len(stats['predicted_greedy_costs_filtered']) > 0 and len(stats['predicted_best_costs_filtered']) > 0:
            mn = min(np.nanmin(stats['predicted_greedy_costs_filtered']), np.nanmin(stats['predicted_best_costs_filtered'])) * 0.9
            mx = max(np.nanmax(stats['predicted_greedy_costs_filtered']), np.nanmax(stats['predicted_best_costs_filtered'])) * 1.1
            plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Predicted Greedy Cost')
        plt.ylabel('Exhaustive Best Predicted Cost')
        plt.title('Predicted Greedy vs Exhaustive Best Predicted')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'pred_greedy_vs_exhaustive_pred{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_predicted and has_exhaustive_pred_data:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_best_costs_filtered'], stats['predicted_exhaustive_costs_filtered'], alpha=0.7, s=70, c='purple', edgecolors='black')
        if len(stats['predicted_best_costs_filtered']) > 0 and len(stats['predicted_exhaustive_costs_filtered']) > 0:
            mn = min(np.nanmin(stats['predicted_best_costs_filtered']), np.nanmin(stats['predicted_exhaustive_costs_filtered'])) * 0.9
            mx = max(np.nanmax(stats['predicted_best_costs_filtered']), np.nanmax(stats['predicted_exhaustive_costs_filtered'])) * 1.1
            plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('DP Best Predicted Cost')
        plt.ylabel('Exhaustive Best Predicted Cost')
        plt.title('Dynamic Programming vs Exhaustive Search Comparison')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'dp_vs_exhaustive{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if (len(stats['query_sizes']) > 0 and 
        ((has_gradient_pred and len(stats['predicted_gradient_costs']) > 0) or 
         (has_greedy_pred and len(stats['predicted_greedy_costs']) > 0) or
         (has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0))):
        
        if ((has_gradient_pred and len(stats['predicted_gradient_costs']) > 0) or 
            (has_greedy_pred and len(stats['predicted_greedy_costs']) > 0) or
            (has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0)):
            
            # Get unique query sizes and sort them
            unique_sizes = sorted(list(set(stats['query_sizes'])))

            #unique_sizes.remove(3)
            
            # Calculate mean predicted costs for each query size
            gradient_mean_by_size = []
            greedy_mean_by_size = []
            random_mean_by_size = []
            dp_mean_by_size = []
            valid_sizes_mean = []
            
            for size in unique_sizes:
                # Get indices for this query size
                size_indices = [i for i, s in enumerate(stats['query_sizes']) if s == size]
                
                if len(size_indices) < 1:  # Need at least 1 query for mean
                    continue
                
                valid_sizes_mean.append(size)
                
                # Calculate mean for gradient predicted if available
                if has_gradient_pred and len(stats['predicted_gradient_costs']) > 0:
                    gradient_pred_costs_size = [stats['predicted_gradient_costs'][i] for i in size_indices 
                                              if i < len(stats['predicted_gradient_costs'])]
                    if gradient_pred_costs_size:
                        gradient_mean_by_size.append(np.median(gradient_pred_costs_size)) # TODO: change to mean
                    else:
                        gradient_mean_by_size.append(np.nan)
                else:
                    gradient_mean_by_size.append(np.nan)
                
                # Calculate mean for greedy predicted if available
                if has_greedy_pred and len(stats['predicted_greedy_costs']) > 0:
                    greedy_pred_costs_size = [stats['predicted_greedy_costs'][i] for i in size_indices 
                                            if i < len(stats['predicted_greedy_costs'])]
                    if greedy_pred_costs_size:
                        greedy_mean_by_size.append(np.median(greedy_pred_costs_size)) # TODO: change to mean
                    else:
                        greedy_mean_by_size.append(np.nan)
                else:
                    greedy_mean_by_size.append(np.nan)
                
                # Calculate mean for random predicted if available and USE_RANDOM is enabled
                if has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0:
                    random_pred_costs_size = [stats['predicted_random_costs'][i] for i in size_indices 
                                            if i < len(stats['predicted_random_costs'])]
                    if random_pred_costs_size:
                        random_mean_by_size.append(np.median(random_pred_costs_size))
                    else:
                        random_mean_by_size.append(np.nan)
                else:
                    random_mean_by_size.append(np.nan)
                
                # Calculate mean for DP predicted costs for reference (only if DP data is available)
                if has_dp_pred and len(stats['predicted_best_costs']) > 0:
                    dp_pred_costs_size = [stats['predicted_best_costs'][i] for i in size_indices 
                                        if i < len(stats['predicted_best_costs'])]
                    # Filter out NaNs
                    dp_pred_costs_size = [c for c in dp_pred_costs_size if c is not None and not np.isnan(c)]
                    
                    if dp_pred_costs_size:
                        dp_mean_by_size.append(np.median(dp_pred_costs_size))
                    else:
                        dp_mean_by_size.append(np.nan)
                else:
                    dp_mean_by_size.append(np.nan)
            
            # Create the mean predicted costs bar plot
            if valid_sizes_mean and len(valid_sizes_mean) >= 1:
                plt.figure(figsize=(12, 8))
                
                x_positions = np.arange(len(valid_sizes_mean))
                
                # Determine how many methods we have to adjust bar width
                num_methods = 0
                if has_gradient_pred and not all(np.isnan(gradient_mean_by_size)):
                    num_methods += 1
                if has_greedy_pred and not all(np.isnan(greedy_mean_by_size)):
                    num_methods += 1
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mean_by_size)):
                    num_methods += 1
                if has_dp_pred and not all(np.isnan(dp_mean_by_size)):
                    num_methods += 1
                
                width = 0.8 / max(num_methods, 1)  # Adjust width based on number of methods
                
                bar_pos = 0
                
                # Plot gradient mean if available
                if has_gradient_pred and not all(np.isnan(gradient_mean_by_size)):
                    gradient_mean_clean = [mean if not np.isnan(mean) else 0 for mean in gradient_mean_by_size]
                    plt.bar(x_positions + (bar_pos - num_methods/2 + 0.5) * width, gradient_mean_clean, width, 
                           label='Mean Pred Gradient Cost', color='blue', alpha=0.7)
                    bar_pos += 1
                
                # Plot greedy mean if available
                if has_greedy_pred and not all(np.isnan(greedy_mean_by_size)):
                    greedy_mean_clean = [mean if not np.isnan(mean) else 0 for mean in greedy_mean_by_size]
                    plt.bar(x_positions + (bar_pos - num_methods/2 + 0.5) * width, greedy_mean_clean, width, 
                           label='Mean Pred Greedy Cost', color='green', alpha=0.7)
                    bar_pos += 1
                
                # Plot random mean if available and USE_RANDOM is enabled
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mean_by_size)):
                    random_mean_clean = [mean if not np.isnan(mean) else 0 for mean in random_mean_by_size]
                    plt.bar(x_positions + (bar_pos - num_methods/2 + 0.5) * width, random_mean_clean, width, 
                           label='Mean Pred Random Cost', color='red', alpha=0.7)
                    bar_pos += 1
                
                # Plot DP mean for reference (only if DP data is available)
                if has_dp_pred and not all(np.isnan(dp_mean_by_size)):
                    dp_mean_clean = [mean if not np.isnan(mean) else 0 for mean in dp_mean_by_size]
                    plt.bar(x_positions + (bar_pos - num_methods/2 + 0.5) * width, dp_mean_clean, width, 
                           label='Mean Pred DP Cost', color='purple', alpha=0.7)
                    bar_pos += 1
                
                plt.xlabel('Query Size (Number of Triple Patterns)')
                plt.ylabel('Mean Predicted Cost')
                plt.title('Mean Predicted Costs by Query Size')
                plt.xticks(x_positions, valid_sizes_mean)
                plt.yscale('log')
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                bar_pos = 0
                if has_gradient_pred and not all(np.isnan(gradient_mean_by_size)):
                    for i, mean_cost in enumerate(gradient_mean_clean):
                        if mean_cost > 0:
                            plt.text(i + (bar_pos - num_methods/2 + 0.5) * width, mean_cost * 1.1, f"{mean_cost:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    bar_pos += 1
                
                if has_greedy_pred and not all(np.isnan(greedy_mean_by_size)):
                    for i, mean_cost in enumerate(greedy_mean_clean):
                        if mean_cost > 0:
                            plt.text(i + (bar_pos - num_methods/2 + 0.5) * width, mean_cost * 1.1, f"{mean_cost:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    bar_pos += 1
                
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mean_by_size)):
                    for i, mean_cost in enumerate(random_mean_clean):
                        if mean_cost > 0:
                            plt.text(i + (bar_pos - num_methods/2 + 0.5) * width, mean_cost * 1.1, f"{mean_cost:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    bar_pos += 1
                
                if has_dp_pred and not all(np.isnan(dp_mean_by_size)):
                    for i, mean_cost in enumerate(dp_mean_clean):
                        if mean_cost > 0:
                            plt.text(i + (bar_pos - num_methods/2 + 0.5) * width, mean_cost * 1.1, f"{mean_cost:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    bar_pos += 1
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_directory, f'mean_predicted_costs_by_query_size{suffix}.png'))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

                plt.figure()
                #####################################################
                ### LINE PLOT PREDICTED COSTS BY QUERY SIZE
                #####################################################
                # Plot lines for available methods
                if has_gradient_pred and not all(np.isnan(gradient_mean_by_size)):
                    gradient_mean_clean = [mean if not np.isnan(mean) else None for mean in gradient_mean_by_size]
                    # Filter out None values for plotting
                    gradient_sizes = [valid_sizes_mean[i] for i, mean in enumerate(gradient_mean_clean) if mean is not None]
                    gradient_costs = [mean for mean in gradient_mean_clean if mean is not None]
                        
                    plt.plot(gradient_sizes, gradient_costs, 'o-', 
                            label='Gradient', markeredgecolor='white', markersize=5)
                
                if has_greedy_pred and not all(np.isnan(greedy_mean_by_size)):
                    greedy_mean_clean = [mean if not np.isnan(mean) else None for mean in greedy_mean_by_size]
                    # Filter out None values for plotting
                    greedy_sizes = [valid_sizes_mean[i] for i, mean in enumerate(greedy_mean_clean) if mean is not None]
                    greedy_costs = [mean for mean in greedy_mean_clean if mean is not None]
                    
                    plt.plot(greedy_sizes, greedy_costs, 's-', 
                            label='Greedy', markeredgecolor='white', markersize=5)
                
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mean_by_size)):
                    random_mean_clean = [mean if not np.isnan(mean) else None for mean in random_mean_by_size]
                    # Filter out None values for plotting
                    random_sizes = [valid_sizes_mean[i] for i, mean in enumerate(random_mean_clean) if mean is not None]
                    random_costs = [mean for mean in random_mean_clean if mean is not None]
                    
                    plt.plot(random_sizes, random_costs, '^-', 
                            label='Random', markeredgecolor='white')
                
                if has_dp_pred and not EXCLUDE_DP and not all(np.isnan(dp_mean_by_size)):
                    dp_mean_clean = [mean if not np.isnan(mean) else None for mean in dp_mean_by_size]
                    # Filter out None values for plotting
                    dp_sizes = [valid_sizes_mean[i] for i, mean in enumerate(dp_mean_clean) if mean is not None]
                    dp_costs = [mean for mean in dp_mean_clean if mean is not None]
                    
                    plt.plot(dp_sizes, dp_costs, 'd-', 
                            label='Dynamic Programming', markeredgecolor='white')
                
                plt.xlabel('Query Size')
                plt.ylabel('Median Predicted Cost')
                plt.yscale('log')
                
                # Customize legend
                plt.legend(frameon=True,
                          loc='best', framealpha=0.)
                
                # Customize grid
                #plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                
                # Customize tick labels
                #plt.xticks(fontsize=12)
                #plt.yticks(fontsize=12)
                
                # Set margins and layout
                plt.margins(x=0.05, y=0.05)
                plt.tight_layout()
                
                # Save with high DPI for publication
                plt.savefig(os.path.join(save_directory, f'mean_predicted_costs_lineplot{suffix}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(save_directory, f'mean_predicted_costs_lineplot{suffix}.pdf'), 
                           bbox_inches='tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                # Print summary statistics
                print(f"\nMean Predicted Costs Analysis by Query Size:")
                print(f"Query sizes analyzed: {valid_sizes_mean}")
                if has_gradient_pred and not all(np.isnan(gradient_mean_by_size)):
                    print(f"Mean Predicted Gradient Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in gradient_mean_by_size]}")
                if has_greedy_pred and not all(np.isnan(greedy_mean_by_size)):
                    print(f"Mean Predicted Greedy Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in greedy_mean_by_size]}")
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mean_by_size)):
                    print(f"Mean Predicted Random Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in random_mean_by_size]}")
                if has_dp_pred and not all(np.isnan(dp_mean_by_size)):
                    print(f"Mean Predicted DP Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in dp_mean_by_size]}")
            else:
                print("Note: No query sizes have sufficient data for mean predicted costs analysis")
        
        # New: Mean Real Costs Line Plot
        if not EXCLUDE_TRUE_COSTS and has_any_real_costs and len(valid_sizes_mean) >= 1:
            # Calculate mean real costs for each query size
            gradient_real_mean_by_size = []
            greedy_real_mean_by_size = []
            dp_real_mean_by_size = []
            
            for size in valid_sizes_mean:
                # Get indices for this query size
                size_indices = [i for i, s in enumerate(stats['query_sizes']) if s == size]
                
                # Calculate mean for gradient real if available
                if has_gradient_real and len(stats['gradient_real']) > 0:
                    # Note: gradient_real list might be shorter than query_sizes if some queries failed
                    # We need to map back correctly. stats['gradient_real'] is populated in order of successful queries
                    # But here we need to match by query index.
                    # The current extract_costs_and_metrics logic appends to lists sequentially.
                    # If we assume stats arrays are aligned (which they should be if we filtered consistent failures),
                    # we can use the same indices.
                    
                    # Safer approach: Re-extract based on availability
                    # But simpler here: if len(stats['gradient_real']) == len(stats['query_sizes']), we are good.
                    # Let's assume arrays are aligned as per extract_costs_and_metrics logic
                    
                    if len(stats['gradient_real']) > max(size_indices):
                        costs = [stats['gradient_real'][i] for i in size_indices if i < len(stats['gradient_real'])]
                        if costs:
                            gradient_real_mean_by_size.append(np.median(costs))
                        else:
                            gradient_real_mean_by_size.append(np.nan)
                    else:
                        gradient_real_mean_by_size.append(np.nan)
                else:
                    gradient_real_mean_by_size.append(np.nan)

                # Calculate mean for greedy real if available
                if has_greedy_real and len(stats['greedy_real']) > 0:
                    if len(stats['greedy_real']) > max(size_indices):
                        costs = [stats['greedy_real'][i] for i in size_indices if i < len(stats['greedy_real'])]
                        if costs:
                            greedy_real_mean_by_size.append(np.median(costs))
                        else:
                            greedy_real_mean_by_size.append(np.nan)
                    else:
                        greedy_real_mean_by_size.append(np.nan)
                else:
                    greedy_real_mean_by_size.append(np.nan)

                # Calculate mean for DP real if available
                if has_dp_real and len(stats['dp_real']) > 0:
                    if len(stats['dp_real']) > max(size_indices):
                        costs = [stats['dp_real'][i] for i in size_indices if i < len(stats['dp_real'])]
                        # Filter out NaNs
                        costs = [c for c in costs if c is not None and not np.isnan(c)]
                        
                        if costs:
                            dp_real_mean_by_size.append(np.median(costs))
                        else:
                            dp_real_mean_by_size.append(np.nan)
                    else:
                        dp_real_mean_by_size.append(np.nan)
                else:
                    dp_real_mean_by_size.append(np.nan)

            plt.figure()
            
            # Plot lines for available methods
            if has_gradient_real and not all(np.isnan(gradient_real_mean_by_size)):
                gradient_sizes = [valid_sizes_mean[i] for i, mean in enumerate(gradient_real_mean_by_size) if not np.isnan(mean)]
                gradient_costs = [mean for mean in gradient_real_mean_by_size if not np.isnan(mean)]
                plt.plot(gradient_sizes, gradient_costs, 'o-', 
                        label='Gradient', markeredgecolor='white', markersize=5)
            
            if has_greedy_real and not all(np.isnan(greedy_real_mean_by_size)):
                greedy_sizes = [valid_sizes_mean[i] for i, mean in enumerate(greedy_real_mean_by_size) if not np.isnan(mean)]
                greedy_costs = [mean for mean in greedy_real_mean_by_size if not np.isnan(mean)]
                plt.plot(greedy_sizes, greedy_costs, 's-', 
                        label='Greedy', markeredgecolor='white', markersize=5)
            
            if has_dp_real and not EXCLUDE_DP and not all(np.isnan(dp_real_mean_by_size)):
                dp_sizes = [valid_sizes_mean[i] for i, mean in enumerate(dp_real_mean_by_size) if not np.isnan(mean)]
                dp_costs = [mean for mean in dp_real_mean_by_size if not np.isnan(mean)]
                plt.plot(dp_sizes, dp_costs, 'd-', 
                        label='Dynamic Programming', markeredgecolor='white')
            
            plt.xlabel('Query Size')
            plt.ylabel('Median Real Cost')
            plt.yscale('log')
            plt.legend(frameon=True, loc='best', framealpha=0.)
            plt.margins(x=0.05, y=0.05)
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_directory, f'mean_real_costs_lineplot{suffix}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_directory, f'mean_real_costs_lineplot{suffix}.pdf'), 
                       bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            print(f"\nMean Real Costs Analysis by Query Size:")
            if has_gradient_real:
                print(f"Mean Real Gradient Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in gradient_real_mean_by_size]}")
            if has_greedy_real:
                print(f"Mean Real Greedy Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in greedy_real_mean_by_size]}")
            if has_dp_real:
                print(f"Mean Real DP Costs by size: {[f'{mean:.2e}' if not np.isnan(mean) else 'N/A' for mean in dp_real_mean_by_size]}")

            # New: Grouped Boxplot for Real Costs by Query Size
            plt.figure(figsize=(14, 8))
            
            # Collect data for boxplot
            bp_data = []
            bp_positions = []
            bp_colors = []
            bp_labels = []
            
            width = 0.2
            offsets = []
            
            # Determine offsets based on available methods
            methods_to_plot = []
            if has_gradient_real: methods_to_plot.append(('Gradient', 'blue'))
            if has_greedy_real: methods_to_plot.append(('Greedy', 'green'))
            if has_dp_real and not EXCLUDE_DP: methods_to_plot.append(('Dynamic Programming', 'purple'))
            
            num_methods = len(methods_to_plot)
            if num_methods > 0:
                # Calculate offsets to center the group around the integer tick
                # e.g. for 2 methods: -0.1, +0.1
                # e.g. for 3 methods: -0.2, 0, +0.2
                start_offset = - (num_methods - 1) * width / 2
                for i in range(num_methods):
                    offsets.append(start_offset + i * width)
                
                valid_sizes_sorted = sorted(list(set(valid_sizes_mean)))
                
                for size_idx, size in enumerate(valid_sizes_sorted):
                    # Get indices for this query size
                    size_indices = [i for i, s in enumerate(stats['query_sizes']) if s == size]
                    
                    for method_idx, (method_name, color) in enumerate(methods_to_plot):
                        method_costs = []
                        
                        if method_name == 'Gradient':
                            if len(stats['gradient_real']) > max(size_indices):
                                method_costs = [stats['gradient_real'][i] for i in size_indices if i < len(stats['gradient_real'])]
                        elif method_name == 'Greedy':
                            if len(stats['greedy_real']) > max(size_indices):
                                method_costs = [stats['greedy_real'][i] for i in size_indices if i < len(stats['greedy_real'])]
                        elif method_name == 'Dynamic Programming':
                            if len(stats['dp_real']) > max(size_indices):
                                method_costs = [stats['dp_real'][i] for i in size_indices if i < len(stats['dp_real'])]
                        
                        # Filter NaNs
                        method_costs = [c for c in method_costs if c is not None and not np.isnan(c)]
                        
                        if method_costs:
                            bp_data.append(method_costs)
                            bp_positions.append(size + offsets[method_idx])
                            bp_colors.append(color)
                            # We only add labels once for the legend
                            if size_idx == 0:
                                bp_labels.append(method_name)
                            else:
                                bp_labels.append(None)
                        else:
                            # Add empty data to keep alignment or skip? 
                            # Better to skip position, but then coloring logic needs to be robust.
                            # For simplicity, let's just skip adding to data/positions if empty.
                            pass

                if bp_data:
                    bplot = plt.boxplot(bp_data, positions=bp_positions, widths=width * 0.8, 
                                      patch_artist=True, showfliers=False) # outliers can clutter
                    
                    # Color the boxes
                    # The boxes are created in the order of bp_data.
                    # We need to match colors. Since we iterated size then method, 
                    # the colors list should match exactly.
                    for patch, color in zip(bplot['boxes'], bp_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    
                    # Set median line color to black or distinct
                    for median in bplot['medians']:
                        median.set_color('black')
                        median.set_linewidth(1.5)

                    plt.xlabel('Query Size')
                    plt.ylabel('Real Cost')
                    plt.yscale('log')
                    plt.title('Real Cost Distribution by Query Size')
                    
                    # Custom legend
                    from matplotlib.lines import Line2D
                    legend_elements = [Line2D([0], [0], color=m[1], lw=4, label=m[0], alpha=0.6) for m in methods_to_plot]
                    plt.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.9)
                    
                    # Ensure ticks are at integers
                    plt.xticks(valid_sizes_sorted, valid_sizes_sorted)
                    plt.grid(axis='y', alpha=0.3, which='both')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_directory, f'real_costs_boxplot_by_size{suffix}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.savefig(os.path.join(save_directory, f'real_costs_boxplot_by_size{suffix}.pdf'), 
                               bbox_inches='tight')
                    
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()


        # Check if we have gradient or greedy predicted data to compare with DP predicted (only do MSE analysis if DP is available)
        if (has_dp_pred and len(stats['predicted_best_costs']) > 0 and 
            ((has_gradient_pred and len(stats['predicted_gradient_costs']) > 0) or 
             (has_greedy_pred and len(stats['predicted_greedy_costs']) > 0) or
             (has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0))):
            
            # Get unique query sizes and sort them
            unique_sizes = sorted(list(set(stats['query_sizes'])))
            
            # Calculate MSE for each query size
            gradient_mse_by_size = []
            greedy_mse_by_size = []
            random_mse_by_size = []
            valid_sizes = []
            
            for size in unique_sizes:
                # Get indices for this query size
                size_indices = [i for i, s in enumerate(stats['query_sizes']) if s == size]
                
                if len(size_indices) < 2:  # Need at least 2 queries for meaningful MSE
                    print(f"Skipping query size {size}: insufficient data ({len(size_indices)} queries, need at least 2)")
                    continue
                
                # DP cost estimates check
                # We check if we have enough valid DP costs
                dp_pred_costs_size = [stats['predicted_best_costs'][i] for i in size_indices 
                                    if i < len(stats['predicted_best_costs'])]
                
                # Filter out NaNs/Nones and check count
                valid_dp_costs = [c for c in dp_pred_costs_size if c is not None and not np.isnan(c)]
                
                if len(valid_dp_costs) < 2:
                    print(f"Skipping query size {size}: insufficient valid DP predicted costs ({len(valid_dp_costs)} costs, need at least 2)")
                    continue
                
                valid_sizes.append(size)
                dp_pred_costs_array = np.array(dp_pred_costs_size) # Keep NaNs for alignment with other arrays
                
                # Calculate MSE for gradient predicted if available
                if has_gradient_pred and len(stats['predicted_gradient_costs']) > 0:
                    gradient_pred_costs_size = [stats['predicted_gradient_costs'][i] for i in size_indices 
                                              if i < len(stats['predicted_gradient_costs'])]
                    if len(gradient_pred_costs_size) == len(dp_pred_costs_size):
                        gradient_pred_costs_array = np.array(gradient_pred_costs_size)
                        mse_gradient = np.nanmean((gradient_pred_costs_array - dp_pred_costs_array) ** 2)
                        gradient_mse_by_size.append(mse_gradient)
                    else:
                        gradient_mse_by_size.append(np.nan)
                else:
                    gradient_mse_by_size.append(np.nan)
                
                # Calculate MSE for greedy predicted if available
                if has_greedy_pred and len(stats['predicted_greedy_costs']) > 0:
                    greedy_pred_costs_size = [stats['predicted_greedy_costs'][i] for i in size_indices 
                                            if i < len(stats['predicted_greedy_costs'])]
                    if len(greedy_pred_costs_size) == len(dp_pred_costs_size):
                        greedy_pred_costs_array = np.array(greedy_pred_costs_size)
                        mse_greedy = np.nanmean((greedy_pred_costs_array - dp_pred_costs_array) ** 2)
                        greedy_mse_by_size.append(mse_greedy)
                    else:
                        greedy_mse_by_size.append(np.nan)
                else:
                    greedy_mse_by_size.append(np.nan)
                
                # Calculate MSE for random predicted if available and USE_RANDOM is enabled
                if has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0:
                    random_pred_costs_size = [stats['predicted_random_costs'][i] for i in size_indices 
                                            if i < len(stats['predicted_random_costs'])]
                    if len(random_pred_costs_size) == len(dp_pred_costs_size):
                        random_pred_costs_array = np.array(random_pred_costs_size)
                        mse_random = np.nanmean((random_pred_costs_array - dp_pred_costs_array) ** 2)
                        random_mse_by_size.append(mse_random)
                    else:
                        random_mse_by_size.append(np.nan)
                else:
                    random_mse_by_size.append(np.nan)
            
            # Always create the plot if we have at least one valid query size
            if valid_sizes and len(valid_sizes) >= 1:
                plt.figure(figsize=(12, 8))
                
                x_positions = np.arange(len(valid_sizes))
                
                # Determine how many methods we have for MSE
                num_mse_methods = 0
                if has_gradient_pred and not all(np.isnan(gradient_mse_by_size)):
                    num_mse_methods += 1
                if has_greedy_pred and not all(np.isnan(greedy_mse_by_size)):
                    num_mse_methods += 1
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mse_by_size)):
                    num_mse_methods += 1
                
                width = 0.8 / max(num_mse_methods, 1)
                mse_bar_pos = 0
                
                # Plot gradient MSE if available
                if has_gradient_pred and not all(np.isnan(gradient_mse_by_size)):
                    gradient_mse_clean = [mse if not np.isnan(mse) else 0 for mse in gradient_mse_by_size]
                    plt.bar(x_positions + (mse_bar_pos - num_mse_methods/2 + 0.5) * width, gradient_mse_clean, width, 
                           label='MSE(Pred Gradient, Pred DP)', color='blue', alpha=0.7)
                    mse_bar_pos += 1
                
                # Plot greedy MSE if available
                if has_greedy_pred and not all(np.isnan(greedy_mse_by_size)):
                    greedy_mse_clean = [mse if not np.isnan(mse) else 0 for mse in greedy_mse_by_size]
                    plt.bar(x_positions + (mse_bar_pos - num_mse_methods/2 + 0.5) * width, greedy_mse_clean, width, 
                           label='MSE(Pred Greedy, Pred DP)', color='green', alpha=0.7)
                    mse_bar_pos += 1
                
                # Plot random MSE if available and USE_RANDOM is enabled
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mse_by_size)):
                    random_mse_clean = [mse if not np.isnan(mse) else 0 for mse in random_mse_by_size]
                    plt.bar(x_positions + (mse_bar_pos - num_mse_methods/2 + 0.5) * width, random_mse_clean, width, 
                           label='MSE(Pred Random, Pred DP)', color='red', alpha=0.7)
                    mse_bar_pos += 1
                
                plt.xlabel('Query Size (Number of Triple Patterns)')
                plt.ylabel('Mean Squared Error')
                plt.title('MSE between Predicted Optimization Methods and Predicted DP by Query Size')
                plt.xticks(x_positions, valid_sizes)
                plt.yscale('log')
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                mse_bar_pos = 0
                if has_gradient_pred and not all(np.isnan(gradient_mse_by_size)):
                    for i, mse in enumerate(gradient_mse_clean):
                        if mse > 0:
                            plt.text(i + (mse_bar_pos - num_mse_methods/2 + 0.5) * width, mse * 1.1, f"{mse:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    mse_bar_pos += 1
                
                if has_greedy_pred and not all(np.isnan(greedy_mse_by_size)):
                    for i, mse in enumerate(greedy_mse_clean):
                        if mse > 0:
                            plt.text(i + (mse_bar_pos - num_mse_methods/2 + 0.5) * width, mse * 1.1, f"{mse:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    mse_bar_pos += 1
                
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mse_by_size)):
                    for i, mse in enumerate(random_mse_clean):
                        if mse > 0:
                            plt.text(i + (mse_bar_pos - num_mse_methods/2 + 0.5) * width, mse * 1.1, f"{mse:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                    mse_bar_pos += 1
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_directory, f'predicted_mse_by_query_size{suffix}.png'))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                # Print summary statistics
                print(f"\nPredicted MSE Analysis by Query Size:")
                print(f"Query sizes analyzed: {valid_sizes}")
                print(f"Query sizes excluded due to insufficient data: {[size for size in unique_sizes if size not in valid_sizes]}")
                if has_gradient_pred and not all(np.isnan(gradient_mse_by_size)):
                    print(f"Predicted Gradient MSE by size: {[f'{mse:.2e}' if not np.isnan(mse) else 'N/A' for mse in gradient_mse_by_size]}")
                if has_greedy_pred and not all(np.isnan(greedy_mse_by_size)):
                    print(f"Predicted Greedy MSE by size: {[f'{mse:.2e}' if not np.isnan(mse) else 'N/A' for mse in greedy_mse_by_size]}")
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mse_by_size)):
                    print(f"Predicted Random MSE by size: {[f'{mse:.2e}' if not np.isnan(mse) else 'N/A' for mse in random_mse_by_size]}")
            else:
                print("Note: No query sizes have sufficient data for predicted MSE analysis (need at least 2 queries per size)")
        
        # NEW: Calculate MAE (Mean Absolute Error) for each query size against DP costs
        if (has_dp_pred and len(stats['predicted_best_costs']) > 0 and 
            ((has_gradient_pred and len(stats['predicted_gradient_costs']) > 0) or 
             (has_greedy_pred and len(stats['predicted_greedy_costs']) > 0) or
             (has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0))):
            
            # Reuse unique_sizes calculated above
            gradient_mae_by_size = []
            greedy_mae_by_size = []
            random_mae_by_size = []
            mae_valid_sizes = []
            
            for size in unique_sizes:
                size_indices = [i for i, s in enumerate(stats['query_sizes']) if s == size]
                
                # Need at least 1 query for MAE if we just want mean difference
                if len(size_indices) < 1:
                    continue
                
                dp_pred_costs_size = [stats['predicted_best_costs'][i] for i in size_indices if i < len(stats['predicted_best_costs'])]
                
                # Check for valid DP costs
                valid_dp_costs = [c for c in dp_pred_costs_size if c is not None and not np.isnan(c)]
                
                if len(valid_dp_costs) < 1:
                    continue
                
                mae_valid_sizes.append(size)
                dp_pred_costs_array = np.array(dp_pred_costs_size)
                
                # Calculate MAE for gradient
                if has_gradient_pred and len(stats['predicted_gradient_costs']) > 0:
                    gradient_pred_costs_size = [stats['predicted_gradient_costs'][i] for i in size_indices if i < len(stats['predicted_gradient_costs'])]
                    if len(gradient_pred_costs_size) == len(dp_pred_costs_size):
                        mae_gradient = np.nanmean(np.abs(np.array(gradient_pred_costs_size) - dp_pred_costs_array))
                        gradient_mae_by_size.append(mae_gradient)
                    else:
                        gradient_mae_by_size.append(np.nan)
                else:
                    gradient_mae_by_size.append(np.nan)
                
                # Calculate MAE for greedy
                if has_greedy_pred and len(stats['predicted_greedy_costs']) > 0:
                    greedy_pred_costs_size = [stats['predicted_greedy_costs'][i] for i in size_indices if i < len(stats['predicted_greedy_costs'])]
                    if len(greedy_pred_costs_size) == len(dp_pred_costs_size):
                        mae_greedy = np.nanmean(np.abs(np.array(greedy_pred_costs_size) - dp_pred_costs_array))
                        greedy_mae_by_size.append(mae_greedy)
                    else:
                        greedy_mae_by_size.append(np.nan)
                else:
                    greedy_mae_by_size.append(np.nan)
                
                # Calculate MAE for random
                if has_pred_random and USE_RANDOM and len(stats['predicted_random_costs']) > 0:
                    random_pred_costs_size = [stats['predicted_random_costs'][i] for i in size_indices if i < len(stats['predicted_random_costs'])]
                    if len(random_pred_costs_size) == len(dp_pred_costs_size):
                        mae_random = np.nanmean(np.abs(np.array(random_pred_costs_size) - dp_pred_costs_array))
                        random_mae_by_size.append(mae_random)
                    else:
                        random_mae_by_size.append(np.nan)
                else:
                    random_mae_by_size.append(np.nan)
            
            # Create the MAE line plot
            if mae_valid_sizes and len(mae_valid_sizes) >= 1:
                plt.figure()
                
                if has_gradient_pred and not all(np.isnan(gradient_mae_by_size)):
                    gradient_mae_clean = [mae if not np.isnan(mae) else None for mae in gradient_mae_by_size]
                    grad_sizes = [mae_valid_sizes[i] for i, mae in enumerate(gradient_mae_clean) if mae is not None]
                    grad_maes = [mae for mae in gradient_mae_clean if mae is not None]
                    plt.plot(grad_sizes, grad_maes, 'o-', label='Gradient MAE', markeredgecolor='white', markersize=5)
                
                if has_greedy_pred and not all(np.isnan(greedy_mae_by_size)):
                    greedy_mae_clean = [mae if not np.isnan(mae) else None for mae in greedy_mae_by_size]
                    greedy_sizes = [mae_valid_sizes[i] for i, mae in enumerate(greedy_mae_clean) if mae is not None]
                    greedy_maes = [mae for mae in greedy_mae_clean if mae is not None]
                    plt.plot(greedy_sizes, greedy_maes, 's-', label='Greedy MAE', markeredgecolor='white', markersize=5)
                
                if has_pred_random and USE_RANDOM and not all(np.isnan(random_mae_by_size)):
                    random_mae_clean = [mae if not np.isnan(mae) else None for mae in random_mae_by_size]
                    rand_sizes = [mae_valid_sizes[i] for i, mae in enumerate(random_mae_clean) if mae is not None]
                    rand_maes = [mae for mae in random_mae_clean if mae is not None]
                    plt.plot(rand_sizes, rand_maes, '^-', label='Random MAE', markeredgecolor='white')
                
                plt.xlabel('Query Size')
                plt.ylabel('Mean Absolute Error (vs DP)')
                plt.yscale('log')
                plt.legend(frameon=True, loc='best', framealpha=0.)
                plt.margins(x=0.05, y=0.05)
                plt.tight_layout()
                
                plt.savefig(os.path.join(save_directory, f'predicted_mae_lineplot{suffix}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(save_directory, f'predicted_mae_lineplot{suffix}.pdf'), 
                           bbox_inches='tight')
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                print(f"\nPredicted MAE Analysis by Query Size:")
                if has_gradient_pred:
                    print(f"Gradient MAE: {[f'{mae:.2e}' if not np.isnan(mae) else 'N/A' for mae in gradient_mae_by_size]}")
                if has_greedy_pred:
                    print(f"Greedy MAE: {[f'{mae:.2e}' if not np.isnan(mae) else 'N/A' for mae in greedy_mae_by_size]}")

    
    elif not has_dp_pred:
        print("Note: Predicted DP costs not available - predicted MSE analysis by query size cannot be performed")
    elif not (has_gradient_pred or has_greedy_pred or (has_pred_random and USE_RANDOM)):
        print("Note: No predicted gradient, greedy, or random costs available - predicted MSE analysis by query size cannot be performed")

def main(results_dir=None):
    # Use provided directory or fall back to global
    target_dir = results_dir if results_dir else RESULTS_DIR
    
    # Validate input directory
    if not target_dir or not os.path.exists(target_dir):
        print(f"Error: Results directory does not exist: {target_dir}")
        sys.exit(1)
    
    # Set output directory
    if OUTPUT_DIR is None:
        output_dir = os.path.join(target_dir, 'plots')
    else:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    data = load_optimization_results(target_dir)
    stats = extract_costs_and_metrics(data)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_statistics(stats, show_plots=False, save_directory=output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR = args.results_dir
    main(RESULTS_DIR)
