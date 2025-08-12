"""
Main evaluation script for query optimization.

This script evaluates different optimization strategies (gradient-based, greedy, random)
on SPARQL queries and compares their performance using a trained cost model.
"""

import sys
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from datetime import datetime
import itertools
import time

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

# Import the classes
from src.create_data.create_optimization_data import SPARQLQuery
from data import Triple, Join, Query, Entity
from model import CostGNNv2
from src.create_data.create_cost_model_training_data import SPARQLQuery

from optimization import (
    optimize_query_gumbel,
    optimize_query_gumbel_efficient_reduced,
    greedy_optimize_query,
    random_join_plan,
    dp_leftdeep_best_plan,
    exhaustive_leftdeep_best_plan
)

from utils.data_utils import (
    adjacency_to_query_with_real_triples,
    count_triples_in_plan,
    collect_triples_in_plan,
    validate_plan,
    plan_to_string,
    plans_are_equivalent,
    load_sparql_queries,
    #query_to_adjacency_matrix
)


def evaluate_optimization(sparql_queries, model_path, num_queries=None, optimization_steps=500, 
                         verbose=False, optimization_params=None, optimization_function=None, save_directory=".", 
                         use_exhaustive=True, use_true_costs=True, use_dp=True):
    """
    Evaluate the optimization algorithm on the given SPARQL queries.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model_path: Path to the trained cost model
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        verbose: Whether to print and plot detailed progress information
        optimization_params: Dictionary of optimization hyperparameters
        optimization_function: Function to use for optimization (optimize_query_gumbel or optimize_query)
        save_directory: Directory to save all outputs to
        use_exhaustive: Whether to perform exhaustive search (default: True)
        use_true_costs: Whether to calculate true costs for plans (default: True)
        use_dp: Whether to perform dynamic programming search (default: True)
        
    Returns:
        Statistics about the optimization performance
    """
    # Set default optimization function if not provided
    if optimization_function is None:
        optimization_function = optimize_query_gumbel
    
    # Create visualization directory
    visualization_dir = os.path.join(save_directory, "plan_visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Create animation data directory
    animation_data_dir = os.path.join(save_directory, "animation_data")
    os.makedirs(animation_data_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    node_feature_dim = 307
    hidden_dim = 512
    model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Limit number of queries if specified
    if num_queries is not None:
        sparql_queries = sparql_queries[:num_queries]
    
    # Initialize statistics
    gradient_costs = []
    greedy_costs = []
    random_costs = []
    predicted_best_costs = []     
    true_best_predicted_costs = [] 
    predicted_gradient_costs = []
    predicted_greedy_costs = []
    predicted_random_costs = []
    if use_exhaustive:
        predicted_exhaustive_costs = []
    
    gradient_failures = 0
    
    detailed_results = []
    
    # Process each query
    for i, query in enumerate(tqdm(sparql_queries, desc="Evaluating queries")):

        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]
        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]

        
        if torch_data is None:
            print(f"Warning: Query {i} has null torch_data for plan {plan_idx}. Skipping.")
            continue
        
        # Prepare query triples for JSON
        query_triples = [[str(triple.s), str(triple.p), str(triple.o)] for triple in triple_objs]
        
        # Run DP-based best plan search (only if enabled)
        if use_dp:
            if len(query_triples) <= 9:
                # start timer
                start_time = time.time()
                best_adj, best_pred_cost = dp_leftdeep_best_plan(torch_data, model, device)
                end_time = time.time()
                print(f"Time taken for DP-based best plan search: {end_time - start_time:.2f} seconds")
            else:
                best_adj = None
                best_pred_cost = float('inf')
        else:
            best_adj = None
            best_pred_cost = float('inf')
        
        # Run exhaustive search for comparison (only if enabled)
        if use_exhaustive:
            start_time = time.time()
            exhaustive_adj, exhaustive_pred_cost = exhaustive_leftdeep_best_plan(torch_data, model, device)
            end_time = time.time()
            print(f"Time taken for exhaustive search: {end_time - start_time:.2f} seconds")
        else:
            exhaustive_adj = None
            exhaustive_pred_cost = float('inf')

        best_pred_plan = None
        true_cost_best_pred = float('inf')
        try:
            triples_num = len(triple_objs)
            if use_dp and best_adj is not None:
                best_pred_plan = adjacency_to_query_with_real_triples(
                    best_adj, triples_num, triple_objs)
                if use_true_costs:
                    true_cost_best_pred = best_pred_plan.root.get_cost()
        except Exception as e:
            print(f"Warning: Failed to compute best predicted plan for query {i}: {e}")
            true_cost_best_pred = float('inf')
        else:
            triples_num = len(triple_objs)

        # Convert exhaustive plan (only if exhaustive search was performed)
        exhaustive_plan = None
        if use_exhaustive:
            try:
                exhaustive_plan = adjacency_to_query_with_real_triples(
                    exhaustive_adj, triples_num, triple_objs)
            except Exception as e:
                print(f"Warning: Failed to convert exhaustive plan for query {i}: {e}")

        # starting timer
        start_time = time.time()
        
        # Track success of each method
        gradient_success = False
        greedy_success = False
        random_success = False
        gradient_cost = float('inf')
        greedy_cost = float('inf')
        random_cost = float('inf')
        grad_pred_cost = float('inf')  
        greedy_pred_cost = float('inf')
        random_pred_cost = float('inf')
        # Initialize plan variables
        gradient_plan = None
        greedy_plan = None
        random_plan = None
        
        # Step 2: Run gradient-based optimization
        try:
            if verbose:
                print(f"\nRunning gradient-based optimization for query {i}")
            
            # Run gradient optimization k times and pick the best result
            k = optimization_params.get('k', 1)  # Number of runs, default to 1
            best_final_adjacency = None
            best_triples_num = None
            best_grad_pred_cost = float('inf')
            best_animation_data = None
            
            for run_idx in range(k):
                # Handle different return values based on optimization function
                optimization_result = optimization_function(
                    torch_data, model, device, 
                    optimization_steps=optimization_steps, 
                    verbose=verbose if run_idx == 0 else False,  # Only verbose for first run
                    **optimization_params
                )
                
                if len(optimization_result) == 4:
                    final_adjacency, triples_num, grad_pred_cost, animation_data = optimization_result
                elif len(optimization_result) == 3:
                    final_adjacency, triples_num, grad_pred_cost = optimization_result
                    animation_data = None
                else:
                    raise ValueError("Unexpected return tuple from optimization_function")

                # Check if this run produced a better result
                if grad_pred_cost < best_grad_pred_cost:
                    best_final_adjacency = final_adjacency
                    best_triples_num = triples_num
                    best_grad_pred_cost = grad_pred_cost
                    best_animation_data = animation_data
            
            # Use the best result from all runs
            final_adjacency = best_final_adjacency
            triples_num = best_triples_num
            grad_pred_cost = best_grad_pred_cost
            animation_data = best_animation_data

            # Save animation data to disk if available
            if animation_data is not None:
                animation_file = os.path.join(animation_data_dir, f"query_{i}_animation_data.pkl")
                try:
                    import pickle
                    with open(animation_file, 'wb') as f:
                        pickle.dump(animation_data, f)
                    print(f"Saved animation data to {animation_file}")
                except Exception as e:
                    print(f"Warning: Failed to save animation data: {e}")

            
            # Create animation if data is available (but don't do it during evaluation to save time)
            # Animation can be generated later using the saved data
            if animation_data is not None and verbose:
                try:
                    print("Creating optimization animation...")
                    # create_optimization_animation(
                    #     animation_data, 
                    #     visualization_dir, 
                    #     i, 
                    #     fps=10,
                    #     use_tree_layout=True,
                    #     max_edge_weight=2.0  # For dual-softmax which can go up to 2
                    # )
                    print(f"Saved optimization animation to {visualization_dir}/")
                except Exception as e:
                    print(f"Warning: Failed to create optimization animation: {e}")
            
            # Convert adjacency to query plan (always create for saving plan structure)
            gradient_plan = None
            try:
                gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
                
                # Validate that the plan contains all expected triple patterns
                is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
                if not is_valid:
                    print(f"Warning: Invalid gradient plan for query {i}: {validation_msg}")
                    print("Skipping this query")
                    gradient_failures += 1
                    continue
            except Exception as e:
                print(f"Warning: Failed to convert gradient plan for query {i}: {e}")
                print("Skipping this query")
                gradient_failures += 1
                continue

            # Visualize the plan
            gradient_plan.visualize(output_file=f"{visualization_dir}/gradient_plan_query_{i}")

            end_time = time.time()
            print(f"Time taken for gradient optimization: {end_time - start_time:.2f} seconds")

            # Calculate the actual cost using the get_cost method (only if enabled)
            if use_true_costs and gradient_plan is not None:
                gradient_cost = gradient_plan.root.get_cost()
            else:
                gradient_cost = float('inf')  # Skip true cost calculation
            gradient_success = True

            # Attempt to visualize the plan – if Graphviz fails, continue without stopping
            if use_true_costs and gradient_plan is not None:
                try:
                    gradient_plan.visualize(output_file=f"{visualization_dir}/gradient_plan_query_{i}")
                except Exception as viz_err:
                    print(f"Warning: Failed to visualize gradient plan for query {i}: {viz_err}")
            
            if verbose:
                if use_true_costs and gradient_plan is not None:
                    print(f"Gradient optimization complete. Final cost: {gradient_cost}")
                    print(f"Saved gradient plan visualization to {visualization_dir}/gradient_plan_query_{i}.png")
                else:
                    print(f"Gradient optimization complete. Predicted cost: {grad_pred_cost}")

                
        except Exception as e:
            raise e
            print(f"Error in gradient optimization for query {i}: {e}")
            # Skip this query
            gradient_failures += 1
            continue
        
        # Run greedy optimization
        try:
            if verbose:
                print(f"\nRunning greedy optimization for query {i}")
                
            greedy_plan, greedy_pred_cost = greedy_optimize_query(
                torch_data, model, triple_objs, device, verbose=verbose
            )
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(greedy_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid greedy plan for query {i}: {validation_msg}")
                greedy_cost = float('inf')
            else:
                # Calculate the actual cost (only if enabled)
                if use_true_costs:
                    greedy_cost = greedy_plan.root.get_cost()
                else:
                    greedy_cost = float('inf')  # Skip true cost calculation
                greedy_success = True

                # Visualize the plan
                greedy_plan.visualize(output_file=f"{visualization_dir}/greedy_plan_query_{i}")
            
            if verbose:
                if use_true_costs:
                    print(f"Greedy optimization complete. Final cost: {greedy_cost}")
                    if greedy_success:
                        # Visualize the plan if verbose
                        greedy_plan.visualize(output_file=f"{visualization_dir}/greedy_plan_query_{i}")
                        print(f"Saved greedy plan visualization to {visualization_dir}/greedy_plan_query_{i}.png")
                else:
                    print(f"Greedy optimization complete. Predicted cost: {greedy_pred_cost}")
            
        except Exception as e:
            print(f"Error in greedy optimization for query {i}: {e}")
            # Use infinity as a placeholder for failed optimizations
            greedy_cost = float('inf')
        
        # random plan
        try:
            if verbose:
                print(f"\nCreating random plan for query {i}")

            log_pred_cost = model(query.torch_data[0].x, edge_index=query.torch_data[0]['edge_index']).item()
            random_pred_cost = float(np.exp(log_pred_cost))

        except Exception as e:
            print(f"Error creating random plan for query {i}: {e}")
            # Use infinity as a placeholder for failed random plans
            random_cost = float('inf')
        
        gradient_costs.append(gradient_cost)
        greedy_costs.append(greedy_cost)
        random_costs.append(random_cost)
        predicted_best_costs.append(best_pred_cost)
        true_best_predicted_costs.append(true_cost_best_pred)
        predicted_gradient_costs.append(grad_pred_cost)
        predicted_greedy_costs.append(greedy_pred_cost)
        predicted_random_costs.append(random_pred_cost)
        if use_exhaustive:
            predicted_exhaustive_costs.append(exhaustive_pred_cost)
        
        query_result = {
            "query_id": i,
            "query_triples": query_triples,
            "ntriplepattern": len(triple_objs),
            "plans": {
                "greedy": {
                    "predicted_cost": float(greedy_pred_cost),
                    "plan_string": plan_to_string(greedy_plan) if greedy_plan else None
                },
                "gradient": {
                    "predicted_cost": float(grad_pred_cost),
                    "plan_string": plan_to_string(gradient_plan) if gradient_plan else None
                },
                "random": {
                    "predicted_cost": float(random_pred_cost),
                    "plan_string": plan_to_string(random_plan) if random_plan else None
                }
            }
        }
        
        # Add DP results only if DP search was performed
        if use_dp:
            query_result["plans"]["dp"] = {
                "predicted_cost": float(best_pred_cost),
                "plan_string": plan_to_string(best_pred_plan) if best_pred_plan else None
            }
        
        # Add true costs only if enabled
        if use_true_costs:
            query_result["plans"]["greedy"]["real_cost"] = float(greedy_cost)
            query_result["plans"]["gradient"]["real_cost"] = float(gradient_cost)
            if use_dp:
                query_result["plans"]["dp"]["real_cost"] = float(true_cost_best_pred)
        
        # Add exhaustive results only if exhaustive search was performed
        if use_exhaustive:
            query_result["plans"]["exhaustive"] = {
                "predicted_cost": float(exhaustive_pred_cost),
                "plan_string": plan_to_string(exhaustive_plan) if exhaustive_plan else None
            }
            if use_true_costs:
                query_result["plans"]["exhaustive"]["real_cost"] = exhaustive_plan.root.get_cost() if exhaustive_plan else float('inf')
                query_result["greedy_equal_exhaustive"] = plans_are_equivalent(greedy_plan, exhaustive_plan)
                query_result["gradient_equal_exhaustive"] = plans_are_equivalent(gradient_plan, exhaustive_plan)
        
        detailed_results.append(query_result)
        # Print progress every query
        if (i + 1) % 1 == 0:
            print(f"\nProcessed {i+1}/{len(sparql_queries)} queries")
            if gradient_costs:
                print(f"Median gradient cost: {np.median(gradient_costs):.2f}")
            if greedy_costs:
                print(f"Median greedy cost: {np.median(greedy_costs):.2f}")
            if random_costs:
                print(f"Median random cost: {np.median(random_costs):.2f}")
    
        # Save detailed results to JSON
        detailed_results_file = os.path.join(save_directory, "detailed_results.json")
        with open(detailed_results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    # Calculate statistics
    stats = {
        'gradient_costs': gradient_costs,
        'greedy_costs': greedy_costs,
        'random_costs': random_costs,
        'predicted_best_costs': predicted_best_costs,
        'true_best_predicted_costs': true_best_predicted_costs,
        'predicted_gradient_costs': predicted_gradient_costs,
        'predicted_greedy_costs': predicted_greedy_costs,
        'predicted_random_costs': predicted_random_costs,
        'gradient_failures': gradient_failures
    }
    
    # Add exhaustive costs only if exhaustive search was performed
    if use_exhaustive:
        stats['predicted_exhaustive_costs'] = predicted_exhaustive_costs
    
    
    # Save metadata for animation generation
    animation_metadata = {
        'num_queries': len(sparql_queries),
        'animation_data_dir': animation_data_dir,
        'visualization_dir': visualization_dir,
        'animation_params': {
            'fps': 10,
            'use_tree_layout': True,
            'max_edge_weight': 2.0
        }
    }
    
    metadata_file = os.path.join(save_directory, "animation_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(animation_metadata, f, indent=2)
    
    print(f"\nAnimation data saved to: {animation_data_dir}")
    print(f"Animation metadata saved to: {metadata_file}")
    print(f"To generate animations later, run: python optim_animation.py {save_directory}")
    
    return stats


if __name__ == "__main__":
    # Configuration for optimization


    config_wikidata_star = {
  "queries_file": "datasets/plans/wikidata_star_plan_datasets_optimization/queries.pkl",
  "model_path": "datasets/models/wikidata/star_model.pt",
  "num_queries": 8200,
  "optimization_steps": 2500,
  "verbose": False,
  "use_exhaustive": False,
  "use_dp": False,
  "use_true_costs": False,
  "save_path": "optimization_results",
  "optimization_params": {
    "optimization_procedure": "gumbel",
    "k": 1,  # Number of gradient optimization runs
    "learning_rate": 0.35,
    "lambda_acyclic": 3391.0,
    "lambda_triple_in": 3334.0,
    "lambda_triple_out": 2026.0,
    "lambda_join_in": 2150.0,
    "lambda_join_out": 1295.0,
    "lambda_entropy": 0.0,
    "lambda_total_penalty": 0.7,
    "lambda_left_linear": 2157.0,
    "init_tau": 15,
    "min_tau": 1.0,
    "tau_decay": 0.973,
    "use_temperature_annealing": True,
    "return_best": True,
    "min_penalty_threshold": 3,
    "use_lambda_ramping": True,
    "logit_sampling": "dual-softmax",
    "save_animation_data": False,
    "animation_save_interval": 10,
    "lambda_ramp_exponent": 5.3,
    "lr_warmup_steps": 46,
    "gradient_clip_norm": 3.3,
    "use_lr_scheduling": True,
    "decoding_method": "greedy"
  },
  "save_directory": "optimization_results/run_20250721_200442",
  "timestamp": "20250721_200442"
}
    

    config_wikidata_path = {
  "queries_file": "datasets/plans/wikidata_path_plan_datasets_optimization/queries.pkl",
  "model_path": "datasets/models/wikidata/path_model.pt",
  "num_queries": 20,
  "optimization_steps": 1000,
  "verbose": False,
  "use_exhaustive": False,
  "use_dp": True,
  "use_true_costs": False,
  "save_path": "optimization_results",
  "optimization_params": {
    "optimization_procedure": "gumbel",
    "k": 5,  # Number of gradient optimization runs
    "learning_rate": 0.5,
    "lambda_acyclic": 467.0,
    "lambda_triple_in": 3194.0,
    "lambda_triple_out": 3661.0,
    "lambda_join_in": 1919.0,
    "lambda_join_out": 1900.0,
    "lambda_entropy": 0.0,
    "lambda_total_penalty": 1.8,
    "lambda_left_linear": 759.0,
    "init_tau": 5,
    "min_tau": 1.0,
    "tau_decay": 0.973,
    "use_temperature_annealing": True,
    "return_best": True,
    "min_penalty_threshold": 0.5,
    "use_lambda_ramping": True,
    "logit_sampling": "softmax",
    "save_animation_data": False,
    "animation_save_interval": 10,
    "lambda_ramp_exponent": 7,
    "lr_warmup_steps": 150,
    "gradient_clip_norm": 4.1,
    "use_lr_scheduling": True,
    "decoding_method": "greedy"
  },
}
    


    config = config_wikidata_path

    
    # Create unique save directory based on datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(config['save_path'], f"run_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Saving all results to: {save_directory}")
    
    # Save configuration to JSON file
    config_copy = config.copy()
    config_copy['save_directory'] = save_directory
    config_copy['timestamp'] = timestamp
    with open(os.path.join(save_directory, "config.json"), 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    # Print configuration
    print("Running optimization with the following configuration:")
    print(f"Number of queries: {config['num_queries']}")
    print(f"Optimization steps: {config['optimization_steps']}")
    print("Optimization hyperparameters:")
    for param, value in config['optimization_params'].items():
        print(f"  {param}: {value}")
    
    # Load queries
    sparql_queries = load_sparql_queries(config['queries_file'], config['num_queries'])
    
    # Select optimization function based on config
    optimization_procedure = config['optimization_params'].pop('optimization_procedure')
    if optimization_procedure == 'gumbel':
        optimization_function = optimize_query_gumbel
    else: 
        raise ValueError(f"Invalid optimization procedure: {optimization_procedure}")
    # Evaluate optimization
    stats = evaluate_optimization(
        sparql_queries, 
        config['model_path'],
        num_queries=config['num_queries'],
        optimization_steps=config['optimization_steps'],
        verbose=config['verbose'],
        optimization_params=config['optimization_params'],
        optimization_function=optimization_function,
        save_directory=save_directory,
        use_exhaustive=config['use_exhaustive'],
        use_true_costs=config.get('use_true_costs', True),  # Default to True for backward compatibility
        use_dp=config.get('use_dp', True) # Default to True for backward compatibility
    )
    
    # Calculate final statistics
    final_stats = {
        'gradient': {
            'mean': float(np.mean(stats['gradient_costs'])),
            'median': float(np.median(stats['gradient_costs'])),
            'std': float(np.std(stats['gradient_costs'])),
            'min': float(np.min(stats['gradient_costs'])),
            'max': float(np.max(stats['gradient_costs'])),
            'failure_rate': float(stats['gradient_failures'] / len(sparql_queries))
        },
        'greedy': {
            'mean': float(np.mean(stats['greedy_costs'])),
            'median': float(np.median(stats['greedy_costs'])),
            'std': float(np.std(stats['greedy_costs'])),
            'min': float(np.min(stats['greedy_costs'])),
            'max': float(np.max(stats['greedy_costs']))
        },
        'random': {
            'mean': float(np.mean(stats['random_costs'])),
            'median': float(np.median(stats['random_costs'])),
            'std': float(np.std(stats['random_costs'])),
            'min': float(np.min(stats['random_costs'])),
            'max': float(np.max(stats['random_costs']))
        },
        'ratios': {
            'gradient_to_random_mean': float(np.mean(np.array(stats['gradient_costs']) / np.array(stats['random_costs']))),
            'greedy_to_random_mean': float(np.mean(np.array(stats['greedy_costs']) / np.array(stats['random_costs']))),
            'gradient_to_greedy_mean': float(np.mean(np.array(stats['gradient_costs']) / np.array(stats['greedy_costs'])))  
        },
        'win_rates': {
            'gradient_vs_random': float(np.sum(np.array(stats['gradient_costs']) < np.array(stats['random_costs'])) / len(stats['gradient_costs']) * 100),
            'greedy_vs_random': float(np.sum(np.array(stats['greedy_costs']) < np.array(stats['random_costs'])) / len(stats['greedy_costs']) * 100)
        }
    }
    
    # Save final statistics to JSON file
    with open(os.path.join(save_directory, "final_statistics.json"), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    print(f"Gradient - Mean: {final_stats['gradient']['mean']:.2f}, Median: {final_stats['gradient']['median']:.2f}")
    print(f"Gradient failure rate: {final_stats['gradient']['failure_rate']:.1%}")
    print(f"Greedy - Mean: {final_stats['greedy']['mean']:.2f}, Median: {final_stats['greedy']['median']:.2f}")
    print(f"Random - Mean: {final_stats['random']['mean']:.2f}, Median: {final_stats['random']['median']:.2f}")
    print(f"Gradient win rate vs Random: {final_stats['win_rates']['gradient_vs_random']:.1f}%")
    print(f"Greedy win rate vs Random: {final_stats['win_rates']['greedy_vs_random']:.1f}%")
    
    
    print(f"\nAll results saved to: {save_directory}")
    print(f"- Configuration: config.json")
    print(f"- Final statistics: final_statistics.json") 
    print(f"- Plots: *.png files")
    print(f"- Plan visualizations: plan_visualizations/ subdirectory")


