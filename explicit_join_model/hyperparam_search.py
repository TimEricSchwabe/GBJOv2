# hyperparameter_search_ray_tune.py

from __future__ import annotations

import argparse
import os
import numpy as np
import time

from ray import tune, air
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

import torch
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', '..'))


from optimization import (
    optimize_query_gumbel,
)


from utils.data_utils import (
    adjacency_to_query_with_real_triples,
    validate_plan,
    load_sparql_queries
)

from data import Triple, Entity
from model import CostGNNv2
from create_data.process_dataset_single_file import SPARQLQuery



# -----------------------------------------------------------------------------
# Objective function – executed once per trial/configuration
# Now returns *mean_cost*, *failure_rate* and *composite_cost* so that Tune
# can be run either on a single composite scalar or in true multi-objective
# mode.  A large `penalty_per_failure` connects the two objectives for the
# scalar case.
# -----------------------------------------------------------------------------

def make_objective(queries, model, procedure: str, penalty_per_failure: float = 1e4):
    """Factory that closes over constant data so we don't reload it every step."""

    opt_fn = optimize_query_gumbel

    def _objective(config):
        """Inner Ray Tune objective that gets executed once per trial."""

        stats = evaluate_optimization_efficient(
            sparql_queries=queries,
            model=model,                 # pre-loaded model
            num_queries=len(queries),
            verbose=False,               # keep noisy output out of Ray logs
            optimization_params=config,
            optimization_function=opt_fn,
            save_directory="/tmp/hpo_artifacts",
        )

        # ------------------------------------------------------------------
        # Aggregate metrics
        # ------------------------------------------------------------------
        grad_costs   = stats["gradient_costs"]
        num_failures = stats["num_failures"]
        num_total    = stats["num_total"]

        mean_cost = float(np.mean(grad_costs)) if grad_costs else penalty_per_failure
        failure_rate = num_failures / num_total if num_total else 1.0

        # A single scalar that can be optimised by standard algorithms.
        composite_cost = mean_cost + penalty_per_failure * failure_rate

        # Report **all** metrics – which one Ray actually optimises depends on
        # the TuneConfig that is supplied in the main section.
        tune.report({
            "mean_cost": mean_cost,
            "failure_rate": failure_rate,
            #"composite_cost": composite_cost,
        })

    return _objective


def evaluate_optimization_efficient(sparql_queries, model, num_queries=None, 
                         verbose=True, optimization_params=None, optimization_function=None, save_directory=".", device=None):
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
    
    # Model is already loaded and position
    model.eval()
    
    # Limit number of queries if specified
    if num_queries is not None:
        sparql_queries = sparql_queries[:num_queries]
    
    # Initialize statistics
    gradient_costs: list[float] = []
    num_failures = 0
    total_queries = len(sparql_queries)
    
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
            
            final_adjacency, triples_num, cost_pred = optimization_function(
                torch_data, model, device, 
                verbose=False,  # Never verbose during HPO
                **optimization_params
            )

            
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                num_failures += 1  # count but do not poison the cost list
                if verbose:
                    print(f"Warning: Invalid gradient plan for query {i}: {validation_msg}")
                continue
            
            if verbose:
                end_time = time.time()
                print(f"Time taken for gradient optimization: {end_time - start_time:.2f} seconds")

            # Calculate the actual cost using the get_cost method or the predicted cost
            gradient_costs.append(cost_pred)
            if verbose:
                print(f"Gradient cost: {cost_pred}")

        except Exception as e:
            print(f"Error in gradient optimization for query {i}: {e}")
            if verbose:
                print(f"Error in gradient optimization for query {i}: {e}")
            num_failures += 1
            continue
    
    # Calculate statistics
    stats = {
        'gradient_costs': gradient_costs,
        'num_failures': num_failures,
        'num_total': total_queries,
    }
    return stats



SEARCH_SPACE = {
    # number of optimization steps per query
    "optimization_steps": tune.randint(100, 3000),

    # optimiser learning rate
    "learning_rate": tune.loguniform(1e-1, 100.0),

    # penalty weights (keep log‑uniform – they span orders of magnitude)
    "lambda_acyclic": tune.uniform(1e2, 5e3),
    "lambda_triple_in": tune.uniform(1e2, 5e3),
    "lambda_triple_out": tune.uniform(1e2, 5e3),
    "lambda_join_in": tune.uniform(1e2, 5e3),
    "lambda_join_out": tune.uniform(1e2, 5e3),
    "lambda_entropy": 0,
    "lambda_left_linear": tune.uniform(1e2, 5e3),

    "use_lambda_ramping": tune.choice([True, False]),
    "logit_sampling": tune.choice(['sigmoid', 'softmax', 'dual-softmax']),
    "lambda_ramp_exponent": tune.uniform(0.1, 10.0),
    "min_penalty_threshold": tune.uniform(0.1, 10.0),

    "init_tau": tune.uniform(1.0, 20.0),
    "tau_decay": tune.loguniform(0.95, 0.9999),
    "lambda_total_penalty": tune.uniform(0.1, 5.0),
    "lr_warmup_steps": tune.uniform(0, 200),
    "gradient_clip_norm": tune.uniform(1, 5.0),
}


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    queries_file = "/home/tim/query_optimization/datasets/optimization_stars_3_to_14/queries.pkl"
    model_path = "/home/tim/query_optimization/explicit_join_model/models/star_model.pt"
    num_queries = 80  # Number of queries to evaluate per trial
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
    max_t = len(all_queries)

    objective_mode = "pareto"  # Options: "composite" or "pareto"

    if objective_mode == "composite":
        # Optimise the scalar composite_cost metric
        search_algo = OptunaSearch(metric="composite_cost", mode="min")
        tune_metrics = "composite_cost"
        tune_modes   = "min"
        reporter = CLIReporter(metric_columns=["composite_cost", "mean_cost", "failure_rate", "training_iteration"])
    else:
        # True multi-objective optimisation using Optuna's NSGA-II algorithm
        search_algo = OptunaSearch(metric=["failure_rate", "mean_cost"], mode=["min", "min"])
        tune_metrics = ["failure_rate", "mean_cost"]
        tune_modes   = ["min", "min"]
        reporter = CLIReporter(metric_columns=["failure_rate", "mean_cost", "training_iteration"])

    objective_fn = make_objective(queries=all_queries, model=model, procedure=procedure)

    # --------------------------------------------------------------
    # 5) Launch the sweep
    # --------------------------------------------------------------
    trainable = tune.with_resources(
        objective_fn,
        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    )

    tune_result = tune.Tuner(
        objective_fn,
        tune_config=tune.TuneConfig(
            #metric=tune_metrics,
            #mode=tune_modes,
            num_samples=max_trials,
            search_alg=search_algo,
            max_concurrent_trials=12,
        ),
        param_space=SEARCH_SPACE,
        run_config=air.RunConfig(
            name="join_optim_hpo",
            storage_path=f"file://{os.path.abspath('ray_results')}",
            progress_reporter=reporter,
            log_to_file=True,
        ),
    ).fit()

    if objective_mode == "composite":
        best_result = tune_result.get_best_result(metric="composite_cost", mode="min")
        print("\nBest config found (composite):")
        print(best_result.config)
        print("Best composite_cost:", best_result.metrics["composite_cost"])
    else:
        best_result_cost = tune_result.get_best_result(metric="mean_cost", mode="min")
        best_result_failure_rate = tune_result.get_best_result(metric="failure_rate", mode="min")
        print("\nBest config found (Pareto):")
        print(best_result_cost.config)
        print("Best mean_cost:", best_result_cost.metrics["mean_cost"])
        print("Best failure_rate:", best_result_failure_rate.metrics["failure_rate"])

