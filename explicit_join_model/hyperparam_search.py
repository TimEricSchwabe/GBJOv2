# hyperparameter_search_ray_tune.py
"""
Run a large‑scale hyper‑parameter sweep for the gradient‑based join‑order optimiser
using Ray Tune + Optuna (TPE) search and ASHA scheduler.

* Requires:  ray[tune]>=2.9, optuna>=3.6, numpy, torch, and **your own** code base
  that exposes the helpers below (adjust the import line to your project layout):
      - load_sparql_queries
      - evaluate_optimization
      - optimize_query
      - optimize_query_gumbel

Usage (single node):
    python hyperparameter_search_ray_tune.py \
        --queries-file datasets/sparql_queries_4_tp/queries.pkl \
        --model-path   models/join_plus_tp_prediction_all_sizes.pt \
        --num-queries  30 \
        --max-trials   200 \
        --gpus-per-trial 1

If you have a Ray cluster already running, add `--address \"auto\"`.
The script will dump results to ./ray_results/ and print the best config at the end.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import time

from ray import tune, air
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

# Add missing imports
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Replace this import with the path to **your** helpers (same ones in main code)
# -----------------------------------------------------------------------------
from optimization_evaluation_leftlinear import (
    load_sparql_queries,
    evaluate_optimization,
    optimize_query,
    optimize_query_gumbel,
    adjacency_to_query_with_real_triples,
    validate_plan,
    visualize_adjacency_matrix,
)

from data import Triple, Entity
from model import CostGNNv2
from process_dataset_single_file import SPARQLQuery







def evaluate_optimization(sparql_queries, model_path, num_queries=None, optimization_steps=500, 
                         verbose=False, optimization_params=None, optimization_function=None, save_directory="."):
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
        
    Returns:
        Statistics about the optimization performance
    """
    # Set default optimization function if not provided
    if optimization_function is None:
        optimization_function = optimize_query_gumbel
    #optimization_function = optimize_query_gumbel_rnn
    
    # Create visualization directory
    visualization_dir = os.path.join(save_directory, "plan_visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    node_feature_dim = 307
    hidden_dim = 512
    model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Limit number of queries if specified
    if num_queries is not None:
        sparql_queries = sparql_queries[:num_queries]
    
    # Initialize statistics
    gradient_costs = []
    greedy_costs = []
    random_costs = []
    
    # Process each query
    for i, query in enumerate(tqdm(sparql_queries, desc="Evaluating queries")):
        # Get the torch data from one of the plans
        # For 8TP, we select one of the random plans as the base for optimization
        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]
        
        if torch_data is None:
            print(f"Warning: Query {i} has null torch_data for plan {plan_idx}. Skipping.")
            continue
        
        # Prepare the triple objects
        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]
        

        # starting timer
        start_time = time.time()
        # Run gradient-based optimization
        try:
            if verbose:
                print(f"\nRunning gradient-based optimization for query {i}")
            
            final_adjacency, triples_num = optimization_function(
                torch_data, model, device, 
                optimization_steps=optimization_steps, 
                verbose=verbose,
                **optimization_params
            )

            try:
                # Visualize the adjacency matrix
                print("\nVisualizing the optimized adjacency matrix:")
                # Try both layouts
                visualize_adjacency_matrix(final_adjacency, triples_num, visualization_dir, i, use_tree_layout=False)
                visualize_adjacency_matrix(final_adjacency, triples_num, visualization_dir, i, use_tree_layout=True)
                print(f"Saved adjacency matrix visualizations to {visualization_dir}/")
            except Exception as e:
                print(f"Warning: Failed to visualize adjacency matrix: {e}")
            
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid gradient plan for query {i}: {validation_msg}")
                print("Skipping this query")
                continue
            
            end_time = time.time()
            print(f"Time taken for gradient optimization: {end_time - start_time:.2f} seconds")

            # Calculate the actual cost using the get_cost method
            gradient_cost = gradient_plan.root.get_cost()
            gradient_costs.append(gradient_cost)

                
        except Exception as e:
            #raise e
            print(f"Error in gradient optimization for query {i}: {e}")
            gradient_costs.append(1000000) # High cost to indicate failure
            # Skip this query
            continue
    
        # Calculate statistics
        stats = {
            'gradient_costs': gradient_costs,
        }
    return stats





















# -----------------------------------------------------------------------------
# Objective function – executed once per trial/configuration
# -----------------------------------------------------------------------------

def make_objective(queries, model, procedure: str):
    """Factory that closes over constant data so we don't reload it every step."""

    # pick the correct optimisation function once
    opt_fn = optimize_query_gumbel if procedure == "gumbel" else optimize_query

    def _objective(config):
        # ------------- call your existing evaluation pipeline -------------
        stats = evaluate_optimization_efficient(
            sparql_queries=queries,
            model=model,  # Pass pre-loaded model instead of path
            num_queries=len(queries),           # full set – they are already sliced
            verbose=False,                      # turn off per‑step prints
            optimization_params=config,         # config contains *only* tunables
            optimization_function=opt_fn,
            save_directory="/tmp/hpo_artifacts",  # lightweight I/O during search
        )

        # The scalar we want to minimise – median gradient cost across queries (still named mean_cost)
        metric = float(np.median(stats["gradient_costs"]))

        tune.report({"mean_cost": metric})

    return _objective


def evaluate_optimization_efficient(sparql_queries, model, num_queries=None, 
                         verbose=False, optimization_params=None, optimization_function=None, save_directory=".", device=None):
    """
    Efficient version of evaluate_optimization that takes a pre-loaded model and device.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model: Pre-loaded and pre-positioned model (already on correct device)
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        verbose: Whether to print and plot detailed progress information
        optimization_params: Dictionary of optimization hyperparameters
        optimization_function: Function to use for optimization (optimize_query_gumbel or optimize_query)
        save_directory: Directory to save all outputs to
        device: Pre-determined device (if None, will be detected)
        
    Returns:
        Statistics about the optimization performance
    """
    # Set default optimization function if not provided
    if optimization_function is None:
        optimization_function = optimize_query_gumbel
    
    # Skip expensive I/O operations during hyperparameter search
    if not verbose:
        visualization_dir = None
    else:
        # Create visualization directory only when needed
        visualization_dir = os.path.join(save_directory, "plan_visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Use provided device or detect
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Using device: {device}")
    
    # Model is already loaded and positioned
    model.eval()
    
    # Limit number of queries if specified
    if num_queries is not None:
        sparql_queries = sparql_queries[:num_queries]
    
    # Initialize statistics
    gradient_costs = []
    
    # Process each query
    progress_iter = tqdm(sparql_queries, desc="Evaluating queries") if verbose else sparql_queries
    for i, query in enumerate(progress_iter, 1):
        # Get the torch data from one of the plans
        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]
        
        if torch_data is None:
            if verbose:
                print(f"Warning: Query {i} has null torch_data for plan {plan_idx}. Skipping.")
            continue
        
        # Prepare the triple objects
        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]
        
        # Run gradient-based optimization
        try:
            if verbose:
                print(f"\nRunning gradient-based optimization for query {i}")
                start_time = time.time()
            
            final_adjacency, triples_num = optimization_function(
                torch_data, model, device, 
                verbose=False,  # Never verbose during HPO
                **optimization_params
            )

            
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                if verbose:
                    print(f"Warning: Invalid gradient plan for query {i}: {validation_msg}")
                    print("Skipping this query")
                continue
            
            if verbose:
                end_time = time.time()
                print(f"Time taken for gradient optimization: {end_time - start_time:.2f} seconds")

            # Calculate the actual cost using the get_cost method
            gradient_cost = gradient_plan.root.get_cost()
            gradient_costs.append(gradient_cost)

            # Running median so far
            tune.report({
                "mean_cost": float(np.median(gradient_costs)),
                "training_iteration": i
            })

        except Exception as e:
            print(f"Error in gradient optimization for query {i}: {e}")
            gradient_costs.append(1000000) # High cost to indicate failure
            # Skip this query
            continue
    
    # Calculate statistics
    stats = {
        'gradient_costs': gradient_costs,
    }
    return stats


# -----------------------------------------------------------------------------
# Search space definition (feel free to expand)
# -----------------------------------------------------------------------------

SEARCH_SPACE = {
    # number of optimization steps per query
    "optimization_steps": tune.randint(100, 3000),

    # optimiser learning rate
    "learning_rate": tune.loguniform(1e-1, 100.0),

    # penalty weights (keep log‑uniform – they span orders of magnitude)
    "lambda_acyclic": tune.uniform(1e2, 5e3),
    "lambda_triple_in": tune.uniform(1e2, 5e3),
    "lambda_triple_out": tune.uniform(1e2, 5e3),
    "lambda_join_in": tune.uniform(1e2, 2e3),
    "lambda_join_out": tune.uniform(1e2, 5e3),
    "lambda_entropy": 0,
    "lambda_left_linear": tune.uniform(1e2, 5e3),

    # binary knobs
    "use_lambda_ramping": tune.choice([True, False]),
    "logit_sampling": tune.choice(['sigmoid', 'softmax', 'dual-softmax']),

    # Gumbel‑Sigmoid (ignored for normal procedure but harmless)
    "init_tau": tune.uniform(1.0, 20.0),
    "tau_decay": tune.loguniform(0.95, 0.9999),

    # Always keep this constant – easier for search
    "lambda_total_penalty": 1.0,
}


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Direct configuration instead of command line args
    queries_file = "/home/tim/query_optimization/datasets/sparql_queries_4_tp/queries.pkl"
    model_path = "/home/tim/query_optimization/explicit_join_model/models/join_plus_tp_prediction_all_sizes.pt"
    num_queries = 50  # Number of queries to evaluate per trial (subset for speed)
    procedure = "gumbel"  # Which optimiser variant to tune ("normal" or "gumbel")
    max_trials = 1000  # HPO budget – number of configs to explore
    cpus_per_trial = 2
    gpus_per_trial = 0
    ray_address = None  # Ray cluster address ("auto" to connect to existing)

    import ray


    # --------------------------------------------------------------
    # 1) Initialise / connect to Ray
    # --------------------------------------------------------------
    if ray_address:
        ray.init(address=ray_address)
    else:
        context = ray.init()
        print("Dashboard URL:")
        print(context.dashboard_url)


    # --------------------------------------------------------------
    # 2) Load queries once (shared by all trials on this worker)
    # --------------------------------------------------------------
    all_queries = load_sparql_queries(queries_file, num_queries)

    # --------------------------------------------------------------
    # 3) Load model once (shared by all trials)
    # --------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on device: {device}")
    
    # Load model architecture and weights
    node_feature_dim = 307
    hidden_dim = 512
    model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --------------------------------------------------------------
    # 4) Build Tune components
    # --------------------------------------------------------------
    max_t = len(all_queries)          # 50 in your example
    scheduler = ASHAScheduler(
        max_t=max_t,
        grace_period=int(max_t * 0.5),   # e.g. 5 queries
        reduction_factor=3,
    )

    search_algo = OptunaSearch(metric="mean_cost", mode="min")

    reporter = CLIReporter(metric_columns=["mean_cost", "training_iteration"])

    objective_fn = make_objective(
        queries=all_queries,
        model=model,
        procedure=procedure,
    )

    # --------------------------------------------------------------
    # 5) Launch the sweep
    # --------------------------------------------------------------
    trainable = tune.with_resources(
        objective_fn,
        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    )

    tune_result = tune.Tuner(
        objective_fn,  # Use the resource-wrapped trainable
        tune_config=tune.TuneConfig(
            metric="mean_cost",
            mode="min",
            num_samples=max_trials,
            search_alg=search_algo,
            scheduler=scheduler,
            max_concurrent_trials=12
        ),
        param_space=SEARCH_SPACE,
        run_config=air.RunConfig(
            name="join_optim_hpo",
            # Use an explicit "file://" URI so that pyarrow recognises the scheme.
            # A plain relative path (e.g. "ray_results") causes pyarrow.fs.FileSystem.from_uri
            # to raise `ArrowInvalid: URI has empty scheme`. We therefore convert the desired
            # output directory to an absolute path and prepend the local-filesystem scheme.
            storage_path=f"file://{os.path.abspath('ray_results')}",
            progress_reporter=reporter,
            log_to_file=True,
        ),
    ).fit()

    best_result = tune_result.get_best_result(metric="mean_cost", mode="min")
    print("\nBest config found:")
    print(best_result.config)
    print("Best mean_cost:", best_result.metrics["mean_cost"])
