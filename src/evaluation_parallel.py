"""
Parallel evaluation script for query optimization.

This script evaluates different optimization strategies (gradient-based, greedy, random)
on SPARQL queries in parallel and compares their performance using a trained cost model.
Removes all visualization and plotting, focusing only on detailed results.
"""

import sys
import os
import pickle
import numpy as np
import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from datetime import datetime
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

# Import the classes
from src.create_data.create_optimization_data import SPARQLQuery
from data import Triple, Join, Query, Entity
from model import CostGNNv2, CostGNNv3

from optimization import (
    optimize_query_gumbel,
    optimize_query_neuralsort,
    greedy_optimize_query,
    random_join_plan,
    dp_leftdeep_best_plan,
    exhaustive_leftdeep_best_plan,
    optimize_query_neuralsort_v2
)

from utils.data_utils import (
    adjacency_to_query_with_real_triples,
    count_triples_in_plan,
    collect_triples_in_plan,
    validate_plan,
    plan_to_string,
    plans_are_equivalent,
    load_sparql_queries,
)

# Import plotting functions
from visualization.plot_optimization_results import extract_costs_and_metrics, plot_statistics

# Add module compatibility for old pickle files
import sys
import src.data as data_module
sys.modules['explicit_join_model.data'] = data_module
sys.modules['explicit_join_model'] = sys.modules['src']


def add_fingerprints_to_query_data(query_data, fingerprint_dim=14, max_joins=14):
    """
    Add orthonormal fingerprints to join nodes in query data.
    Call this ONCE before starting gradient optimization.
    """
    x = query_data.x.clone()
    
    is_join = (x[:, -1] == 1.0)
    join_indices = torch.where(is_join)[0]
    n_joins = len(join_indices)
    
    if n_joins == 0:
        return query_data
    
    # Orthonormal basis
    fingerprint_basis = torch.eye(max_joins, fingerprint_dim, device=x.device)
    perm = torch.randperm(max_joins)[:n_joins]
    fingerprints = fingerprint_basis[perm]
    
    for i, join_idx in enumerate(join_indices):
        x[join_idx, :fingerprint_dim] = fingerprints[i]
    
    query_data.x = x
    return query_data



def process_single_query(args):
    """
    Process a single query with all optimization methods.
    
    Args:
        args: Tuple containing (query_index, query, model_path, device_str, optimization_params, 
              optimization_function_name, use_exhaustive, use_true_costs, use_dp, optimization_steps, dp_limit)
    
    Returns:
        Dictionary with detailed results for this query
    """
    (query_index, query, model_path, device_str, optimization_params, 
     optimization_function_name, use_exhaustive, use_true_costs, use_dp, optimization_steps, dp_limit) = args
    
    # Set device
    device = torch.device(device_str)
    
    # Load model
    node_feature_dim = 307
    hidden_dim = 128
    model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Get optimization function
    if optimization_function_name == 'optimize_query_gumbel':
        optimization_function = optimize_query_gumbel
    elif optimization_function_name == 'optimize_query_gumbel_lbfgs':
        optimization_function = optimize_query_gumbel_lbfgs
    elif optimization_function_name == 'optimize_query_neuralsort':
        optimization_function = optimize_query_neuralsort
    elif optimization_function_name == 'optimize_query_neuralsort_v2':
        optimization_function = optimize_query_neuralsort_v2
    else:
        raise ValueError(f"Unknown optimization function: {optimization_function_name}")
    
    try:
        # Get the torch data from one of the plans
        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]


        # Todo decide whether to add this or not !
        raise ValueError("Decide whether to add this or not !")
        torch_data = add_fingerprints_to_query_data(torch_data, fingerprint_dim=14)


        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]
        
        if torch_data is None:
            print(f"Warning: Query {query_index} has null torch_data for plan {plan_idx}. Skipping.")
            return None
        
        # Prepare query triples for JSON
        query_triples = [[str(triple.s), str(triple.p), str(triple.o)] for triple in triple_objs]
        
        # Initialize results
        result = {
            "query_id": query_index,
            "query_triples": query_triples,
            "ntriplepattern": len(triple_objs),
            "plans": {}
        }
        
        # Run DP-based best plan search (only if enabled)
        best_adj = None
        best_pred_cost = float('inf')
        best_pred_plan = None
        true_cost_best_pred = float('inf')
        
        # Only run DP if enabled AND query size is within the limit
        if use_dp and len(query_triples) <= dp_limit:
            try:
                best_adj, best_pred_cost = dp_leftdeep_best_plan(torch_data, model, device)
                triples_num = len(triple_objs)
                best_pred_plan = adjacency_to_query_with_real_triples(
                    best_adj, triples_num, triple_objs)
                if use_true_costs:
                    true_cost_best_pred = best_pred_plan.root.get_cost()
                
                result["plans"]["dp"] = {
                    "predicted_cost": float(best_pred_cost),
                    "plan_string": plan_to_string(best_pred_plan) if best_pred_plan else None
                }
                if use_true_costs:
                    result["plans"]["dp"]["real_cost"] = float(true_cost_best_pred)
                    
            except Exception as e:
                print(f"Warning: DP search failed for query {query_index}: {e}")
        
        # Run exhaustive search for comparison (only if enabled)
        exhaustive_adj = None
        exhaustive_pred_cost = float('inf')
        exhaustive_plan = None
        
        if use_exhaustive:
            try:
                exhaustive_adj, exhaustive_pred_cost = exhaustive_leftdeep_best_plan(torch_data, model, device)
                triples_num = len(triple_objs)
                exhaustive_plan = adjacency_to_query_with_real_triples(
                    exhaustive_adj, triples_num, triple_objs)
                
                result["plans"]["exhaustive"] = {
                    "predicted_cost": float(exhaustive_pred_cost),
                    "plan_string": plan_to_string(exhaustive_plan) if exhaustive_plan else None
                }
                if use_true_costs:
                    result["plans"]["exhaustive"]["real_cost"] = float(exhaustive_plan.root.get_cost()) if exhaustive_plan else float('inf')
                    
            except Exception as e:
                print(f"Warning: Exhaustive search failed for query {query_index}: {e}")
        
        # Initialize plan variables
        gradient_plan = None
        greedy_plan = None
        random_plan = None
        gradient_cost = float('inf')
        greedy_cost = float('inf')
        random_cost = float('inf')
        grad_pred_cost = float('inf')
        greedy_pred_cost = float('inf')
        random_pred_cost = float('inf')
        
        # Run gradient-based optimization
        try:
            # Run gradient optimization k times and pick the best result
            k = optimization_params.get('k', 1)  # Number of runs, default to 1
            best_adjacency = None
            best_triples_num = None
            best_grad_pred_cost = float('inf')
            best_animation_data = None
            
            for run_idx in range(k):
                optimization_result = optimization_function(
                    torch_data, model, device, 
                    optimization_steps=optimization_steps, 
                    verbose=False,  # Always false for parallel execution
                    **optimization_params
                )
                
                # Handle different return types
                if len(optimization_result) == 4:
                    final_adjacency, triples_num, grad_pred_cost, animation_data = optimization_result
                elif len(optimization_result) == 3:
                    final_adjacency, triples_num, grad_pred_cost = optimization_result
                    animation_data = None
                else:
                    raise ValueError("Unexpected return tuple from optimization_function")
                
                # Check if this run produced a better result
                if grad_pred_cost < best_grad_pred_cost:
                    best_adjacency = final_adjacency
                    best_triples_num = triples_num
                    best_grad_pred_cost = grad_pred_cost
                    best_animation_data = animation_data
            
            # Use the best result from all runs
            final_adjacency = best_adjacency
            triples_num = best_triples_num
            grad_pred_cost = best_grad_pred_cost
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid gradient plan for query {query_index}: {validation_msg}")
                gradient_plan = None
                grad_pred_cost = float('inf')
            else:
                # Calculate the actual cost (only if enabled)
                if use_true_costs:
                    gradient_cost = gradient_plan.root.get_cost()
                    
        except Exception as e:
            print(f"Error in gradient optimization for query {query_index}: {e}")
            gradient_plan = None
            grad_pred_cost = float('inf')
        
        # Run greedy optimization
        try:
            greedy_plan, greedy_pred_cost = greedy_optimize_query(
                torch_data, model, triple_objs, device, verbose=False
            )
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(greedy_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid greedy plan for query {query_index}: {validation_msg}")
                greedy_cost = float('inf')
            else:
                # Calculate the actual cost (only if enabled)
                if use_true_costs:
                    greedy_cost = greedy_plan.root.get_cost()
                    
        except Exception as e:
            print(f"Error in greedy optimization for query {query_index}: {e}")
            greedy_cost = float('inf')
        
        # Create a random plan
        try:
            log_pred_cost = model(query.torch_data[0].x, edge_index=query.torch_data[0]['edge_index']).item()
            random_pred_cost = float(np.exp(log_pred_cost))
        except Exception as e:
            print(f"Error creating random plan for query {query_index}: {e}")
            random_cost = float('inf')
        
        # Add results to the result dictionary
        result["plans"]["gradient"] = {
            "predicted_cost": float(grad_pred_cost),
            "plan_string": plan_to_string(gradient_plan) if gradient_plan else None
        }
        result["plans"]["greedy"] = {
            "predicted_cost": float(greedy_pred_cost),
            "plan_string": plan_to_string(greedy_plan) if greedy_plan else None
        }
        result["plans"]["random"] = {
            "predicted_cost": float(random_pred_cost),
            "plan_string": plan_to_string(random_plan) if random_plan else None
        }
        
        # Add true costs only if enabled
        if use_true_costs:
            result["plans"]["gradient"]["real_cost"] = float(gradient_cost)
            result["plans"]["greedy"]["real_cost"] = float(greedy_cost)
            result["plans"]["random"]["real_cost"] = float(random_cost)
        
        # Add exhaustive comparison only if exhaustive search was performed
        if use_exhaustive and exhaustive_plan is not None:
            result["greedy_equal_exhaustive"] = plans_are_equivalent(greedy_plan, exhaustive_plan)
            result["gradient_equal_exhaustive"] = plans_are_equivalent(gradient_plan, exhaustive_plan)
        
        return result
        
    except Exception as e:
        print(f"Error processing query {query_index}: {e}")
        return None


def evaluate_optimization_parallel(sparql_queries, model_path, num_queries=None, optimization_steps=500, 
                                 optimization_params=None, optimization_function=None, save_directory=".", 
                                 use_exhaustive=True, use_true_costs=True, use_dp=True, num_workers=None, dp_limit=9):
    """
    Evaluate the optimization algorithm on the given SPARQL queries in parallel.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model_path: Path to the trained cost model
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        optimization_params: Dictionary of optimization hyperparameters
        optimization_function: Function to use for optimization (optimize_query_gumbel or optimize_query)
        save_directory: Directory to save all outputs to
        use_exhaustive: Whether to perform exhaustive search (default: True)
        use_true_costs: Whether to calculate true costs for plans (default: True)
        use_dp: Whether to perform dynamic programming search (default: True)
        num_workers: Number of parallel workers (default: number of CPU cores)
        dp_limit: Maximum number of triples for DP execution (default: 9)
        
    Returns:
        List of detailed results for each query
    """
    # Set default optimization function if not provided
    if optimization_function is None:
        optimization_function = optimize_query_gumbel
    
    # Get optimization function name for serialization
    optimization_function_name = optimization_function.__name__
    
    # Set device string for serialization
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Limit number of queries if specified
    if num_queries is not None:
        sparql_queries = sparql_queries[:num_queries]
    
    # Set number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(sparql_queries))
    
    print(f"Processing {len(sparql_queries)} queries using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    args_list = []
    for i, query in enumerate(sparql_queries):
        args = (i, query, model_path, device_str, optimization_params, 
                optimization_function_name, use_exhaustive, use_true_costs, use_dp, optimization_steps, dp_limit)
        args_list.append(args)
    
    # Process queries in parallel
    detailed_results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_args = {executor.submit(process_single_query, args): args for args in args_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                if result is not None:
                    detailed_results.append(result)
                completed += 1
                
                # Print progress
                if completed % max(1, len(sparql_queries) // 10) == 0:
                    print(f"Completed {completed}/{len(sparql_queries)} queries ({completed/len(sparql_queries)*100:.1f}%)")
                    
            except Exception as e:
                args = future_to_args[future]
                query_index = args[0]
                print(f"Query {query_index} generated an exception: {e}")
    
    # Sort results by query_id to maintain order
    detailed_results.sort(key=lambda x: x['query_id'])
    
    # Save detailed results to JSON
    detailed_results_file = os.path.join(save_directory, "detailed_results.json")
    with open(detailed_results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Saved detailed results to: {detailed_results_file}")
    
    return detailed_results


if __name__ == "__main__":
    # Configuration for optimization
    config_wikidata_star = {
        "queries_file": "/home/tim/query_optimization/datasets/wikidata_star_plan_datasets_optimization/queries.pkl",
        "model_path": "/home/tim/query_optimization/explicit_join_model/models/wikidata/star_model.pt",
        "num_queries": 20,
        "optimization_steps": 1000, # 2500
        "use_exhaustive": False,
        "use_dp": True,
        "use_true_costs": False,
        "save_path": "optimization_results",
        "num_workers": 6,  # Use all available cores
        "optimization_params": {
            "optimization_procedure": "gumbel",
            "k": 1,  # Number of gradient optimization runs
            "learning_rate": 1, # 0.35
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
        }
    }

    config_wikidata_path = {
        "queries_file": "/home/tim/query_optimization/datasets/wikidata_path_plan_datasets_optimization/queries.pkl",
        "model_path": "/home/tim/query_optimization/explicit_join_model/models/wikidata/path_model.pt",
        "num_queries": 100000,
        "optimization_steps": 1000, #2500
        "use_exhaustive": False,
        "use_dp": True,
        "use_true_costs": False,
        "save_path": "optimization_results",
        "num_workers": None,  # Use all available cores
        "optimization_params": {
            "optimization_procedure": "gumbel",
            "k": 1,  # Number of gradient optimization runs
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
        }
    }

    config_lubm_star = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm_star_plan_datasets_optimization/optimization_stars_3_to_14/queries.pkl",
        "model_path": "/home/tim/query_optimization/training_results/lubm-star-new-v2/model.pt",
        "num_queries": 20,
        "max_query_size": None,  # Filter queries larger than this (None for no filter)
        "optimization_steps": 1000,
        "use_exhaustive": False,
        "use_dp": True,
        "dp_limit": 9,  # Set the limit here (e.g., 15 for star queries)
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": 6,  # Use all available cores
        "optimization_params": {
            "optimization_procedure": "gumbel",
            "k": 5,  # Number of gradient optimization runs
            "learning_rate": 1.7,
            "lambda_acyclic": 3081.0,
            "lambda_triple_in": 3714.0,
            "lambda_triple_out": 135.0,
            "lambda_join_in": 1742.0,
            "lambda_join_out": 1558.0,
            "lambda_entropy": 0.0,
            "lambda_total_penalty": 2.6,
            "lambda_left_linear": 2300.0,
            "init_tau": 4.5,
            "min_tau": 1.0,
            "tau_decay": 0.963,
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 5,
            "use_lambda_ramping": True,
            "logit_sampling": "dual-softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 6.5,
            "lr_warmup_steps": 50,
            "gradient_clip_norm": 2,
            "use_lr_scheduling": True,
            "decoding_method": "greedy"
        }
    }

    config_lubm_path = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm_path_plan_datasets_optimization/optimization_paths_3_to_5/queries.pkl",
        "model_path": "/home/tim/query_optimization/datasets/models/lubm/path_model.pt",
        "num_queries": 20,
        "optimization_steps": 1000,
        "use_exhaustive": False,
        "use_dp": True,
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": None,  # Use all available cores
        "optimization_params": {
            "optimization_procedure": "gumbel",
            "k": 5,  # Number of gradient optimization runs - 5
            "learning_rate": 1.8, # 1.8
            "lambda_acyclic": 4415.0,
            "lambda_triple_in": 3027.0,
            "lambda_triple_out": 790.0,
            "lambda_join_in": 2197.0,
            "lambda_join_out": 2204.0,
            "lambda_entropy": 0, # 0
            "lambda_total_penalty": 4.2 ,#4.2
            "lambda_left_linear": 1910, # 1910
            "init_tau": 3.7, #3.7
            "min_tau": 1.0,
            "tau_decay": 0.963, # 0.963
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 8.6,
            "use_lambda_ramping": True,
            "logit_sampling": "dual-softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 6.8, # 6.8
            "lr_warmup_steps": 200,
            "gradient_clip_norm": 1.9,
            "use_lr_scheduling": True,
            "decoding_method": "greedy"
        }
    }

    config_lubm_path_gumbel_sinkhorn = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm_star_plan_datasets_optimization/optimization_stars_3_to_14/queries.pkl",
        "model_path": "/home/tim/query_optimization/datasets/models/lubm/star_model.pt",
        "num_queries": 20,
        "optimization_steps": 1000,
        "use_exhaustive": False,
        "use_dp": True,
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": None,  # Use all available cores
        "optimization_params": {
            "optimization_procedure": "neuralsort_v2",
            "k":3,  # Number of gradient optimization runs - 5
            "learning_rate": 1, # 1.8
            "lambda_acyclic": 4415.0,
            "lambda_triple_in": 3027.0,
            "lambda_triple_out": 790.0,
            "lambda_join_in": 2197.0,
            "lambda_join_out": 2204.0,
            "lambda_entropy": 0.0,
            "lambda_total_penalty": 4.2 ,#4.2
            "lambda_left_linear": 1910, # 1910
            "init_tau": 3.0, #3.7
            "min_tau": 0.1,
            "tau_decay": 0.985, # 0.963
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 8.6,
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 6.8, # 6.8
            "lr_warmup_steps": 200,
            "gradient_clip_norm": 1.9,
            "use_lr_scheduling": True,
            "decoding_method": "greedy"
        }
    }

    config = config_lubm_star
    
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
    print("Running parallel optimization with the following configuration:")
    print(f"Number of queries: {config['num_queries']}")
    print(f"Optimization steps: {config['optimization_steps']}")
    print(f"Number of workers: {config.get('num_workers', 'auto')}")
    print("Optimization hyperparameters:")
    for param, value in config['optimization_params'].items():
        print(f"  {param}: {value}")
    
    # Load queries
    sparql_queries = load_sparql_queries(config['queries_file'], config['num_queries'])
    
    # Filter queries by size if max_query_size is set
    if config.get('max_query_size') is not None:
        max_size = config['max_query_size']
        print(f"Filtering queries with size > {max_size}")
        original_len = len(sparql_queries)
        sparql_queries = [q for q in sparql_queries if len(q.triples) <= max_size]
        print(f"Retained {len(sparql_queries)}/{original_len} queries")
        
        # Update num_queries in config for accurate logging
        config['num_queries'] = len(sparql_queries)
    
    # Select optimization function based on config
    optimization_procedure = config['optimization_params'].pop('optimization_procedure')
    if optimization_procedure == 'gumbel':
        optimization_function = optimize_query_gumbel 
    elif optimization_procedure == 'neuralsort':
        optimization_function = optimize_query_neuralsort
    elif optimization_procedure == 'neuralsort_v2':
        optimization_function = optimize_query_neuralsort_v2
    else:  # 'normal'
        raise ValueError(f"Invalid optimization procedure: {optimization_procedure}")
    
    # Start timing
    start_time = time.time()
    
    # Evaluate optimization in parallel
    detailed_results = evaluate_optimization_parallel(
        sparql_queries, 
        config['model_path'],
        num_queries=config['num_queries'],
        optimization_steps=config['optimization_steps'],
        optimization_params=config['optimization_params'],
        optimization_function=optimization_function,
        save_directory=save_directory,
        use_exhaustive=config['use_exhaustive'],
        use_true_costs=config.get('use_true_costs', True),
        use_dp=config.get('use_dp', True),
        num_workers=config.get('num_workers', None),
        dp_limit=config.get('dp_limit', 9)  # Pass dp_limit from config or default to 9
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate summary statistics
    successful_results = [r for r in detailed_results if r is not None]
    
    summary_stats = {
        'total_queries_processed': len(successful_results),
        'total_queries_attempted': len(sparql_queries),
        'success_rate': len(successful_results) / len(sparql_queries) * 100,
        'total_time_seconds': total_time,
        'average_time_per_query': total_time / len(sparql_queries),
        'timestamp': timestamp
    }
    
    # Save summary statistics
    with open(os.path.join(save_directory, "summary_stats.json"), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Generate plots automatically
    try:
        print("\nGenerating plots...")
        stats = extract_costs_and_metrics(detailed_results)
        plots_dir = os.path.join(save_directory, 'plots')
        plot_statistics(stats, show_plots=False, save_directory=plots_dir)
        print(f"Plots saved to: {plots_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    print(f"\n" + "="*50)
    print("PARALLEL EVALUATION COMPLETE")
    print("="*50)
    print(f"Total queries processed: {len(successful_results)}/{len(sparql_queries)}")
    print(f"Success rate: {summary_stats['success_rate']:.1f}%")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per query: {summary_stats['average_time_per_query']:.2f} seconds")
    print(f"\nResults saved to: {save_directory}")
    print(f"- Configuration: config.json")
    print(f"- Detailed results: detailed_results.json")
    print(f"- Summary statistics: summary_stats.json")
