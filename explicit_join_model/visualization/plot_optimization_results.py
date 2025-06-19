#!/usr/bin/env python3
"""
Plot optimization results from saved JSON data.

This script loads the detailed_results.json file from optimization runs
and creates comprehensive visualizations exactly as done in optimization_evaluation.py.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration flags
RESULTS_DIR = "/home/tim/query_optimization/optimization_results/run_20250619_095821"  # Directory containing detailed_results.json
OUTPUT_DIR = None  # If None, will use RESULTS_DIR/plots

# Plot type flags
SKIP_BOXPLOT = False
SKIP_BARPLOT = False
SKIP_SCATTER = False
SKIP_RATIOS = False
SKIP_SIZE_ANALYSIS = False
SKIP_SUMMARY = False
EXCLUDE_TRUE_COSTS = True  # New flag to exclude true costs from plots

# Data inclusion flags
INCLUDE_PREDICTED = True  # Include predicted costs in boxplot
EXCLUDE_EXHAUSTIVE = True  # Exclude exhaustive search from plots
EXCLUDE_GREEDY = False  # Exclude greedy method from plots
EXCLUDE_GRADIENT = False  # Exclude gradient method from plots
EXCLUDE_DP = False  # Exclude DP method from plots

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
        'exhaustive_real': False,
        'greedy_real': False,
        'gradient_real': False,
        'dp_real': False,
        'exhaustive_pred': False,
        'greedy_pred': False,
        'gradient_pred': False,
        'dp_pred': False
    }
    
    if not data:
        return availability
    
    # Check the first query result to see what methods are available
    first_query = data[0]
    if 'plans' in first_query:
        for method in ['exhaustive', 'greedy', 'gradient', 'dp']:
            if method in first_query['plans']:
                availability[method] = True
                # Check if real and predicted costs are available
                plan_data = first_query['plans'][method]
                if 'real_cost' in plan_data and plan_data['real_cost'] is not None:
                    availability[f'{method}_real'] = True
                if 'predicted_cost' in plan_data and plan_data['predicted_cost'] is not None:
                    availability[f'{method}_pred'] = True

    # Double-check by looking at a few more results to be more thorough
    for i, query_result in enumerate(data[:5]):  # Check first 5 queries
        if 'plans' not in query_result:
            continue
        for method in ['exhaustive', 'greedy', 'gradient', 'dp']:
            if method in query_result['plans']:
                if availability[method] is False:
                    availability[method] = True
                
                plan_data = query_result['plans'][method]
                if 'real_cost' in plan_data and plan_data['real_cost'] is not None:
                    if availability[f'{method}_real'] is False:
                        availability[f'{method}_real'] = True
                if 'predicted_cost' in plan_data and plan_data['predicted_cost'] is not None:
                    if availability[f'{method}_pred'] is False:
                        availability[f'{method}_pred'] = True
    
    print("Data availability:")
    for method in ['exhaustive', 'greedy', 'gradient', 'dp']:
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
        'random_costs': [],  # We'll use DP costs as "random" for compatibility
        'predicted_best_costs': [],  # DP predicted costs
        'predicted_exhaustive_costs': [],  # Exhaustive predicted costs
        'predicted_gradient_costs': [],
        'predicted_greedy_costs': [],
        'true_best_predicted_costs': [],  # DP real costs
        'exhaustive_real': [],
        'exhaustive_pred': [],
        'greedy_real': [],
        'greedy_pred': [],
        'gradient_real': [],
        'gradient_pred': [],
        'dp_real': [],
        'dp_pred': [],
        'greedy_equal_exhaustive': [],
        'gradient_equal_exhaustive': [],
        'query_sizes': []
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
        
        # Skip this query if any available method has infinite real cost
        if infinite_costs:
            continue
        
        # Extract costs for available methods - only extract what's actually available
        if availability['gradient'] and 'gradient' in query_result['plans']:
            plan_data = query_result['plans']['gradient']
            
            if availability['gradient_real'] and 'real_cost' in plan_data:
                real_cost = plan_data['real_cost']
                if real_cost is not None and real_cost != float('inf'):
                    stats['gradient_costs'].append(real_cost)
                    stats['gradient_real'].append(real_cost)
            
            if availability['gradient_pred'] and 'predicted_cost' in plan_data:
                pred_cost = plan_data['predicted_cost']
                if pred_cost is not None and pred_cost != float('inf'):
                    stats['gradient_pred'].append(pred_cost)
                    stats['predicted_gradient_costs'].append(pred_cost)
        
        if availability['greedy'] and 'greedy' in query_result['plans']:
            plan_data = query_result['plans']['greedy']
            
            if availability['greedy_real'] and 'real_cost' in plan_data:
                real_cost = plan_data['real_cost']
                if real_cost is not None and real_cost != float('inf'):
                    stats['greedy_costs'].append(real_cost)
                    stats['greedy_real'].append(real_cost)
            
            if availability['greedy_pred'] and 'predicted_cost' in plan_data:
                pred_cost = plan_data['predicted_cost']
                if pred_cost is not None and pred_cost != float('inf'):
                    stats['greedy_pred'].append(pred_cost)
                    stats['predicted_greedy_costs'].append(pred_cost)
        
        if availability['dp'] and 'dp' in query_result['plans']:
            plan_data = query_result['plans']['dp']
            
            if availability['dp_real'] and 'real_cost' in plan_data:
                real_cost = plan_data['real_cost']
                if real_cost is not None and real_cost != float('inf'):
                    stats['random_costs'].append(real_cost)  # Use DP as "random"
                    stats['true_best_predicted_costs'].append(real_cost)
                    stats['dp_real'].append(real_cost)
            
            if availability['dp_pred'] and 'predicted_cost' in plan_data:
                pred_cost = plan_data['predicted_cost']
                if pred_cost is not None and pred_cost != float('inf'):
                    stats['predicted_best_costs'].append(pred_cost)
                    stats['dp_pred'].append(pred_cost)
        
        if availability['exhaustive'] and 'exhaustive' in query_result['plans']:
            plan_data = query_result['plans']['exhaustive']
            
            if availability['exhaustive_real'] and 'real_cost' in plan_data:
                real_cost = plan_data['real_cost']
                if real_cost is not None and real_cost != float('inf'):
                    stats['exhaustive_real'].append(real_cost)
            
            if availability['exhaustive_pred'] and 'predicted_cost' in plan_data:
                pred_cost = plan_data['predicted_cost']
                if pred_cost is not None and pred_cost != float('inf'):
                    stats['predicted_exhaustive_costs'].append(pred_cost)
                    stats['exhaustive_pred'].append(pred_cost)
        
        # Extract equivalence flags only if exhaustive data is available
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
    print(f"Available methods: {[method for method in ['exhaustive', 'greedy', 'gradient', 'dp'] if availability[method]]}")
    print(f"Real costs available for: {[method for method in ['exhaustive', 'greedy', 'gradient', 'dp'] if availability[f'{method}_real']]}")
    print(f"Predicted costs available for: {[method for method in ['exhaustive', 'greedy', 'gradient', 'dp'] if availability[f'{method}_pred']]}")
    
    # Store availability info in stats for use in plotting
    stats['_availability'] = availability
    
    return stats

def plot_statistics(stats, show_plots=True, suffix="", save_directory="."):
    """
    Plot statistics about the optimization performance.
    EXACT COPY from optimization_evaluation.py with minor adaptations for our data structure.
    
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
    
    # Check if we have any real costs at all
    has_any_real_costs = has_exhaustive_real or has_greedy_real or has_gradient_real or has_dp_real
    
    # Calculate mean costs for different strategies (only if real costs are available)
    if not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        if has_gradient_real and len(stats['gradient_costs']) > 0:
            mean_gradient = np.mean(stats['gradient_costs'])
            if has_greedy_real and len(stats['greedy_costs']) > 0:
                mean_greedy = np.mean(stats['greedy_costs'])
            if has_dp_real and len(stats['random_costs']) > 0:
                mean_random = np.mean(stats['random_costs'])
    
    # NEW – optional categories ------------------------------------------------
    has_predicted = 'predicted_best_costs' in stats and len(stats['predicted_best_costs']) > 0
    has_pred_grad = 'predicted_gradient_costs' in stats and len(stats['predicted_gradient_costs']) > 0
    has_pred_greedy = 'predicted_greedy_costs' in stats and len(stats['predicted_greedy_costs']) > 0
    has_true_best = 'true_best_predicted_costs' in stats and len(stats['true_best_predicted_costs']) > 0
    has_exhaustive_pred_data = has_exhaustive_pred and 'predicted_exhaustive_costs' in stats and len(stats['predicted_exhaustive_costs']) > 0
    
    if has_predicted:
        mean_predicted = np.mean(stats['predicted_best_costs'])
    if has_pred_grad:
        mean_pred_grad = np.mean(stats['predicted_gradient_costs'])
    if has_pred_greedy:
        mean_pred_greedy = np.mean(stats['predicted_greedy_costs'])
    if has_true_best and not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        mean_true_best = np.mean(stats['true_best_predicted_costs'])
    if has_exhaustive_pred_data:
        mean_exhaustive = np.mean(stats['predicted_exhaustive_costs'])
    
    # Plot mean costs comparison
    plt.figure(figsize=(12, 6))
    
    labels = []
    means = []
    
    # Only include real costs if they're available and not excluded
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
    
    # Always include predicted costs if available
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
            data.append(stats['gradient_costs'])
            labels_box.append('Gradient')
        if has_greedy_real and len(stats['greedy_costs']) > 0:
            data.append(stats['greedy_costs'])
            labels_box.append('Greedy')
        if has_dp_real and len(stats['random_costs']) > 0:
            data.append(stats['random_costs'])
            labels_box.append('DP')
    
    # Always include predicted costs if available
    if has_predicted:
        data.append(stats['predicted_best_costs'])
        labels_box.append('DP-Best')
    if has_exhaustive_pred_data:
        data.append(stats['predicted_exhaustive_costs'])
        labels_box.append('Exhaustive')
    if has_pred_grad:
        data.append(stats['predicted_gradient_costs'])
        labels_box.append('GradPred')
    if has_pred_greedy:
        data.append(stats['predicted_greedy_costs'])
        labels_box.append('GreedyPred')
    if has_true_best and not EXCLUDE_TRUE_COSTS and has_any_real_costs:
        data.append(stats['true_best_predicted_costs'])
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
            gradient_to_random_ratio = np.mean(np.array(stats['gradient_costs']) / np.array(stats['random_costs']))
            print(f"Mean ratio of gradient optimizer cost to DP cost: {gradient_to_random_ratio:.2f}x")
            
            # Calculate how often gradient beats DP
            gradient_costs = np.array(stats['gradient_costs'])
            random_costs = np.array(stats['random_costs'])
            gradient_wins = np.sum(gradient_costs < random_costs)
            gradient_win_pct = gradient_wins / len(gradient_costs) * 100
            print(f"Gradient optimizer beats DP in {gradient_win_pct:.1f}% of queries")
        
        if has_greedy_real and len(stats['greedy_costs']) > 0 and len(stats['random_costs']) > 0:
            greedy_to_random_ratio = np.mean(np.array(stats['greedy_costs']) / np.array(stats['random_costs']))
            print(f"Mean ratio of greedy heuristic cost to DP cost: {greedy_to_random_ratio:.2f}x")
            
            # Calculate how often greedy beats DP
            greedy_costs = np.array(stats['greedy_costs'])
            random_costs = np.array(stats['random_costs'])
            greedy_wins = np.sum(greedy_costs < random_costs)
            greedy_win_pct = greedy_wins / len(greedy_costs) * 100
            print(f"Greedy heuristic beats DP in {greedy_win_pct:.1f}% of queries")
        
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
            
            max_val = max(np.max(gradient_costs), np.max(greedy_costs))
            min_val = min(np.min(gradient_costs), np.min(greedy_costs))
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
            
            max_val = max(np.max(gradient_costs), np.max(random_costs))
            min_val = min(np.min(gradient_costs), np.min(random_costs))
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
            
            max_val = max(np.max(greedy_costs), np.max(random_costs))
            min_val = min(np.min(greedy_costs), np.min(random_costs))
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
        min_val = np.min(all_pred_costs) * 0.9
        max_val = np.max(all_pred_costs) * 1.1
        
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

    # Rest of the existing scatter plots for predicted costs...
    if has_predicted:
        # Gradient vs best predicted (predicted cost)
        if has_pred_grad:
            plt.figure(figsize=(10, 8))
            plt.scatter(stats['predicted_gradient_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='purple', edgecolors='black')
            min_val = min(min(stats['predicted_gradient_costs']), min(stats['predicted_best_costs'])) * 0.9
            max_val = max(max(stats['predicted_gradient_costs']), max(stats['predicted_best_costs'])) * 1.1
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
            plt.scatter(stats['predicted_greedy_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='purple', edgecolors='black')
            min_val = min(min(stats['predicted_greedy_costs']), min(stats['predicted_best_costs'])) * 0.9
            max_val = max(max(stats['predicted_greedy_costs']), max(stats['predicted_best_costs'])) * 1.1
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
        mn = min(min(stats['predicted_gradient_costs']), min(stats['predicted_greedy_costs'])) * 0.9
        mx = max(max(stats['predicted_gradient_costs']), max(stats['predicted_greedy_costs'])) * 1.1
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
        plt.scatter(stats['predicted_gradient_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='darkgreen', edgecolors='black')
        mn = min(min(stats['predicted_gradient_costs']), min(stats['predicted_best_costs'])) * 0.9
        mx = max(max(stats['predicted_gradient_costs']), max(stats['predicted_best_costs'])) * 1.1
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
        plt.scatter(stats['predicted_greedy_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='darkorange', edgecolors='black')
        mn = min(min(stats['predicted_greedy_costs']), min(stats['predicted_best_costs'])) * 0.9
        mx = max(max(stats['predicted_greedy_costs']), max(stats['predicted_best_costs'])) * 1.1
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

    # NEW: Scatter plot comparing DP vs Exhaustive search results (only if both are available)
    if has_predicted and has_exhaustive_pred_data:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_best_costs'], stats['predicted_exhaustive_costs'], alpha=0.7, s=70, c='purple', edgecolors='black')
        mn = min(min(stats['predicted_best_costs']), min(stats['predicted_exhaustive_costs'])) * 0.9
        mx = max(max(stats['predicted_best_costs']), max(stats['predicted_exhaustive_costs'])) * 1.1
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

    # NEW: MSE between predicted methods and predicted DP costs by query size
    if (has_dp_pred and len(stats['query_sizes']) > 0 and len(stats['predicted_best_costs']) > 0):
        
        # Check if we have gradient or greedy predicted data to compare with DP predicted
        if ((has_gradient_pred and len(stats['predicted_gradient_costs']) > 0) or 
            (has_greedy_pred and len(stats['predicted_greedy_costs']) > 0)):
            
            # Get unique query sizes and sort them
            unique_sizes = sorted(list(set(stats['query_sizes'])))
            
            # Calculate MSE for each query size
            gradient_mse_by_size = []
            greedy_mse_by_size = []
            valid_sizes = []
            
            for size in unique_sizes:
                # Get indices for this query size
                size_indices = [i for i, s in enumerate(stats['query_sizes']) if s == size]
                
                if len(size_indices) < 2:  # Need at least 2 queries for meaningful MSE
                    print(f"Skipping query size {size}: insufficient data ({len(size_indices)} queries, need at least 2)")
                    continue
                
                # Extract DP predicted costs for this size
                dp_pred_costs_size = [stats['predicted_best_costs'][i] for i in size_indices 
                                    if i < len(stats['predicted_best_costs'])]
                
                if len(dp_pred_costs_size) < 2:
                    print(f"Skipping query size {size}: insufficient DP predicted costs ({len(dp_pred_costs_size)} costs, need at least 2)")
                    continue
                
                valid_sizes.append(size)
                dp_pred_costs_array = np.array(dp_pred_costs_size)
                
                # Calculate MSE for gradient predicted if available
                if has_gradient_pred and len(stats['predicted_gradient_costs']) > 0:
                    gradient_pred_costs_size = [stats['predicted_gradient_costs'][i] for i in size_indices 
                                              if i < len(stats['predicted_gradient_costs'])]
                    if len(gradient_pred_costs_size) == len(dp_pred_costs_size):
                        gradient_pred_costs_array = np.array(gradient_pred_costs_size)
                        mse_gradient = np.mean((gradient_pred_costs_array - dp_pred_costs_array) ** 2)
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
                        mse_greedy = np.mean((greedy_pred_costs_array - dp_pred_costs_array) ** 2)
                        greedy_mse_by_size.append(mse_greedy)
                    else:
                        greedy_mse_by_size.append(np.nan)
                else:
                    greedy_mse_by_size.append(np.nan)
            
            # Always create the plot if we have at least one valid query size
            if valid_sizes and len(valid_sizes) >= 1:
                plt.figure(figsize=(12, 8))
                
                x_positions = np.arange(len(valid_sizes))
                width = 0.35
                
                # Plot gradient MSE if available
                if has_gradient_pred and not all(np.isnan(gradient_mse_by_size)):
                    gradient_mse_clean = [mse if not np.isnan(mse) else 0 for mse in gradient_mse_by_size]
                    plt.bar(x_positions - width/2, gradient_mse_clean, width, 
                           label='MSE(Pred Gradient, Pred DP)', color='blue', alpha=0.7)
                
                # Plot greedy MSE if available
                if has_greedy_pred and not all(np.isnan(greedy_mse_by_size)):
                    greedy_mse_clean = [mse if not np.isnan(mse) else 0 for mse in greedy_mse_by_size]
                    plt.bar(x_positions + width/2, greedy_mse_clean, width, 
                           label='MSE(Pred Greedy, Pred DP)', color='green', alpha=0.7)
                
                plt.xlabel('Query Size (Number of Triple Patterns)')
                plt.ylabel('Mean Squared Error')
                plt.title('MSE between Predicted Optimization Methods and Predicted DP by Query Size')
                plt.xticks(x_positions, valid_sizes)
                plt.yscale('log')
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                if has_gradient_pred and not all(np.isnan(gradient_mse_by_size)):
                    for i, mse in enumerate(gradient_mse_clean):
                        if mse > 0:
                            plt.text(i - width/2, mse * 1.1, f"{mse:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                
                if has_greedy_pred and not all(np.isnan(greedy_mse_by_size)):
                    for i, mse in enumerate(greedy_mse_clean):
                        if mse > 0:
                            plt.text(i + width/2, mse * 1.1, f"{mse:.1e}", 
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                
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
            else:
                print("Note: No query sizes have sufficient data for predicted MSE analysis (need at least 2 queries per size)")
    
    elif not has_dp_pred:
        print("Note: Predicted DP costs not available - predicted MSE analysis by query size cannot be performed")
    elif not (has_gradient_pred or has_greedy_pred):
        print("Note: No predicted gradient or greedy costs available - predicted MSE analysis by query size cannot be performed")

def main():
    # Validate input directory
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory does not exist: {RESULTS_DIR}")
        sys.exit(1)
    
    # Set output directory
    if OUTPUT_DIR is None:
        output_dir = os.path.join(RESULTS_DIR, 'plots')
    else:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    data = load_optimization_results(RESULTS_DIR)
    stats = extract_costs_and_metrics(data)

    
    # Generate plots using the exact same function from optimization_evaluation.py
    print("\nGenerating plots...")
    
    # Use the exact plot_statistics function from optimization_evaluation.py
    plot_statistics(stats, show_plots=False, save_directory=output_dir)
    

    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
