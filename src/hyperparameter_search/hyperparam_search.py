# hyperparameter_search.py
"""
Hyperparameter search for GBJO (Gradient-Based Join Optimization).

Search backend:
- Nevergrad CMA-ES (continuous hyperparameters; discrete params are fixed)

Usage:
    Modify the CONFIG dictionary at the bottom of this file, then run:
    python hyperparam_search.py
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))


from optimization import GBJO

from utils.data_utils import (
    adjacency_to_query_with_real_triples,
    validate_plan,
)

from data import Triple, Entity
from model import CostGNNv2, CostGNNv3
from data_loader import SingleFileQueryDataset, AddRandomGaussianFingerprints


# Data Loading
def load_plans_from_pt(dataset_path: str, num_plans: int = None, fingerprint_dim: int = 64):
    """
    Load plans from a dataset.pt file (same format as cost_model_training.py).
    
    Args:
        dataset_path: Path to directory containing dataset.pt or path to .pt file directly
        num_plans: Maximum number of plans to load (None for all)
        fingerprint_dim: Dimension of random fingerprints to add
        
    Returns:
        List of torch geometric Data objects (plans)
    """
    if os.path.isdir(dataset_path):
        pt_file = os.path.join(dataset_path, 'dataset.pt')
    else:
        pt_file = dataset_path
    
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Dataset file not found at {pt_file}")
    
    # Load the dataset
    data_dict = torch.load(pt_file, weights_only=False)
    
    if isinstance(data_dict, dict) and 'data' in data_dict:
        data_list = data_dict['data']
        triples_list = data_dict.get('triples', None)
    elif isinstance(data_dict, list):
        data_list = data_dict
        triples_list = None
    else:
        raise ValueError(f"Unexpected dataset format: {type(data_dict)}")
    
    # Filter out samples with zero or invalid costs
    valid_indices = []
    for i, d in enumerate(data_list):
        if hasattr(d, 'y') and d.y is not None and len(d.x) > 5: # TODO: remove this condition
            if torch.isfinite(d.y).all() and (d.y > 0).all():
                valid_indices.append(i)
    
    data_list = [data_list[i] for i in valid_indices]
    if triples_list is not None:
        triples_list = [triples_list[i] for i in valid_indices]
    
    n_filtered = len(valid_indices)
    print(f"Loaded {len(data_list)} valid plans (filtered {len(data_dict.get('data', data_dict)) - n_filtered} invalid)")
    
    if num_plans is not None and num_plans < len(data_list):
        rng = np.random.default_rng(seed=42)  
        indices = rng.permutation(len(data_list))[:num_plans]

        data_list = [data_list[i] for i in indices]
        if triples_list is not None:
            triples_list = [triples_list[i] for i in indices]
            
    # Add fingerprints to each plan
    fingerprint_transform = AddRandomGaussianFingerprints(fingerprint_dim=fingerprint_dim)
    data_list = [fingerprint_transform(d) for d in data_list]
    
    return data_list, triples_list


# Single Plan Processing (for parallel execution)

def process_single_plan(args):
    """
    Process a single plan with GBJO optimization.
    
    This function is designed to be called in a separate process, so it loads
    the model independently and doesn't share state with the main process.
    
    Args:
        args: Tuple containing (plan_index, plan_data, triples, model_path, model_params, 
              device_str, optimization_params)
    
    Returns:
        Dictionary with result for this plan, or None on failure
    """
    (plan_index, plan_data, triples, model_path, model_params, 
     device_str, optimization_params) = args
    
    # Import here to ensure each worker has its own imports
    from optimization import GBJO
    from utils.data_utils import adjacency_to_query_with_real_triples, validate_plan
    from data import Triple, Entity
    from model import CostGNNv2, CostGNNv3
    
    device = torch.device(device_str)
    
    # Load model in this worker
    if model_params.get('version') == 'v3':
        model = CostGNNv3(
            node_feature_dim=model_params['node_feature_dim'],
            hidden_dim=model_params['hidden_dim'],
            n_layers=model_params.get('n_layers', 6),
            use_jk=model_params.get('use_jk', False),
            jk_mode=model_params.get('jk_mode', 'cat'),
            use_residual=model_params.get('use_residual', False),
            use_layer_norm=model_params.get('use_layer_norm', False),
            dropout=model_params.get('dropout', 0.0),
        ).to(device)
    else:
        model = CostGNNv2(
            node_feature_dim=model_params['node_feature_dim'],
            hidden_dim=model_params['hidden_dim'],
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    if plan_data is None:
        return {'index': plan_index, 'success': False, 'cost': None}
    
    triple_objs = None
    if triples is not None:
        try:
            triple_objs = [Triple(*(Entity(name=str(name)) for name in triple[:3])) for triple in triples]
        except Exception:
            triple_objs = None
    
    try:
        torch.manual_seed(42 + plan_index)
        
        result = GBJO(
            plan_data, model, device,
            verbose=False,
            **optimization_params
        )
        
        if len(result) == 4:
            final_adjacency, triples_num, cost_pred, _ = result
        elif len(result) == 3:
            final_adjacency, triples_num, cost_pred = result
        else:
            return {'index': plan_index, 'success': False, 'cost': None}
        
        # Validate if we have triple objects
        if triple_objs is not None:
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                return {'index': plan_index, 'success': False, 'cost': None}
        
        return {'index': plan_index, 'success': True, 'cost': cost_pred}
        
    except Exception as e:
        return {'index': plan_index, 'success': False, 'cost': None, 'error': str(e)}


# Evaluation function
def evaluate_optimization(
    plans,
    triples_list,
    model_path,
    model_params,
    device,
    optimization_params,
    num_workers=1,
    verbose=False,
):
    """
    Evaluate GBJO on a set of plans, optionally in parallel.
    
    Args:
        plans: List of torch geometric Data objects (already preprocessed plans)
        triples_list: List of triples for each plan (for validation), can be None
        model_path: Path to the trained model file
        model_params: Dictionary of model parameters
        device: torch device 
        optimization_params: Dictionary of GBJO hyperparameters
        num_workers: Number of parallel workers (1 for sequential)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with statistics about optimization performance
    """
    total_queries = len(plans)
    device_str = str(device)
    
    # Parallel execution using ProcessPoolExecutor
    args_list = []
    for i, plan_data in enumerate(plans):
        triples = triples_list[i] if triples_list is not None and i < len(triples_list) else None
        args = (i, plan_data, triples, model_path, model_params, device_str, optimization_params)
        args_list.append(args)
    
    gradient_costs = []
    num_failures = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures_dict = {executor.submit(process_single_plan, args): args[0] for args in args_list}
        
        if verbose:
            pbar = tqdm(total=len(args_list), desc="Evaluating (parallel)")
        
        for future in as_completed(futures_dict):
            try:
                result = future.result()
                if result['success']:
                    gradient_costs.append(result['cost'])
                else:
                    num_failures += 1
            except Exception as e:
                num_failures += 1
                if verbose:
                    print(f"Worker exception: {e}")
            
            if verbose:
                pbar.update(1)
        
        if verbose:
            pbar.close()
    
    return {
        'gradient_costs': gradient_costs,
        'num_failures': num_failures,
        'num_total': total_queries,
    }
    
    


# Nevergrad CMA-ES Search
def get_cma_search_space():
    """Define the continuous search space for CMA-ES."""
    import nevergrad as ng
    
    return ng.p.Dict(
        learning_rate=ng.p.Log(lower=0.1, upper=10.0, init=1),

        lambda_acyclic=ng.p.Log(lower=1.0, upper=100.0, init=1.0),
        lambda_triple_in=ng.p.Log(lower=1.0, upper=100.0, init=1.0),
        lambda_triple_out=ng.p.Log(lower=1.0, upper=100.0, init=1.0),
        lambda_join_in=ng.p.Log(lower=1.0, upper=100.0, init=1.0),
        lambda_join_out=ng.p.Log(lower=1.0, upper=100.0, init=1.0),
        lambda_left_linear=ng.p.Log(lower=1.0, upper=100.0, init=1.0),
        lambda_entropy=ng.p.Constant(0),

        lambda_total_penalty=ng.p.Scalar(lower=0., upper=1.0, init=0.5),
        lambda_total_penalty_start=ng.p.Constant(0),

        init_tau=ng.p.Log(lower=1.0, upper=20.0, init=3.0),
        min_tau=ng.p.Log(lower=0.1, upper=2.0, init=1),

        lambda_ramp_exponent=ng.p.Scalar(lower=1.0, upper=10.0, init=2.0),
        min_penalty_threshold=ng.p.Constant(0),

        gradient_clip_norm=ng.p.Scalar(lower=1.0, upper=100.0, init=2.0),
    )


def run_cma_search(config):
    """Run hyperparameter search using Nevergrad.
    
    Query evaluation is parallelized across num_workers processes.
    trials run sequentially, but each trial evaluates all queries in parallel.
    """
    import nevergrad as ng
    
    # Load plans
    plans, triples_list = load_plans_from_pt(
        config['dataset_path'],
        num_plans=config['num_plans'],
        fingerprint_dim=config.get('fingerprint_dim', 64)
    )
    print(f"Loaded {len(plans)} plans")
    
    # Device and model configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_params = config['model_params']
    model_path = config['model_path']
    num_workers = config.get('num_workers', 1)
    
    print(f"Parallel query evaluation: {num_workers} workers")
    
    # Fixed discrete params
    fixed_params = config.get('fixed_discrete_params', {})
    optimization_steps = config.get('optimization_steps', 500)
    penalty_per_failure = config.get('penalty_per_failure', 1e4)
    
    def objective(params):
        """Objective function for CMA-ES optimization."""
        # Merge continuous params with fixed discrete params
        full_params = {
            **fixed_params,
            **params,
            "optimization_steps": optimization_steps,
            "save_animation_data": False,
        }

        
        stats = evaluate_optimization(
            plans=plans,
            triples_list=triples_list,
            model_path=model_path,
            model_params=model_params,
            device=device,
            optimization_params=full_params,
            num_workers=num_workers,
            verbose=False,
        )
        
        grad_costs = stats["gradient_costs"]
        num_failures = stats["num_failures"]
        num_total = stats["num_total"]
        
        mean_cost = float(np.median(grad_costs)) if grad_costs else penalty_per_failure
        failure_rate = num_failures / num_total if num_total else 1.0
        
        # Composite objective (weighted sum) (with new decoding there shouldnt be any failures...)
        composite = mean_cost + penalty_per_failure * failure_rate
        return composite
    
    # Create optimizer (sequential trial execution, parallelism is within query evaluation)
    parametrization = get_cma_search_space()
    
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization, 
        budget=config.get('max_trials', 500),
        num_workers=1 
    )
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hpo_results/cma_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nStarting CMA-ES optimization with budget={optimizer.budget}")
    print(f"Query parallelism: {num_workers} workers")
    print(f"Fixed discrete params: {json.dumps(fixed_params, indent=2)}")
    
    best_loss = float('inf')
    worst_loss = float('-inf')
    best_params = None
    history = []
    
    pbar = tqdm(range(optimizer.budget), desc="CMA-ES", unit="iter")
    
    for i in pbar:
        candidate = optimizer.ask()
        params = candidate.value
        
        loss = objective(params)
        optimizer.tell(candidate, loss)
        
        history.append({
            'iteration': i,
            'loss': loss,
            'params': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in params.items()},
        })
        
        # Track best and worst
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
        if loss > worst_loss:
            worst_loss = loss
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'best': f'{best_loss:.1f}',
            'worst': f'{worst_loss:.1f}',
            'current': f'{loss:.1f}'
        })
        
        # Save checkpoint every 5 iterations (in case of crash)
        if (i + 1) % 5 == 0:
            checkpoint = {
                'iteration': i + 1,
                'best_loss': best_loss,
                'worst_loss': worst_loss,
                'best_params': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in best_params.items()} if best_params else None,
                'history': history,
            }
            with open(os.path.join(results_dir, "checkpoint.json"), 'w') as f:
                json.dump(checkpoint, f, indent=2)
    
    # Get final recommendation
    recommendation = optimizer.recommend()
    final_params = recommendation.value
    
    # Merge with fixed params for final output
    full_best_params = {
        **{k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in final_params.items()},
        **fixed_params,
        "optimization_steps": optimization_steps,
    }
    
    print("\n" + "=" * 60)
    print("CMA-ES SEARCH COMPLETE")
    print("=" * 60)
    print("\nBest parameters found:")
    print(json.dumps(full_best_params, indent=2))
    print(f"\nBest composite loss: {best_loss:.2f}")
    
    # Save results
    with open(os.path.join(results_dir, "best_config.json"), 'w') as f:
        json.dump({
            "best_params": full_best_params,
            "best_loss": best_loss,
            "fixed_discrete_params": fixed_params,
            "model_params": model_params,
        }, f, indent=2)
    
    with open(os.path.join(results_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    return final_params, best_loss


if __name__ == "__main__":
    
    CONFIG = {
        # Data paths
        "dataset_path": "datasets/plans/wikidata_path_plan_datasets_training/new/dataset.pt",
        "model_path": "training_results/wikidata-path-log1p/model.pt",
        
        # Evaluation settings
        "num_plans": 400,  # Number of plans to evaluate per trial
        "optimization_steps": 10,  # GBJO optimization steps per plan
        "penalty_per_failure": 1e-4,  # Penalty added for failed optimizations
        "fingerprint_dim": 64,  # Fingerprint dimension for join nodes
        
        # Search settings
        "max_trials": 500,  # Number of trials/iterations for the search
        
        "num_workers": 8,  # Number of parallel workers for query evaluation (1 = sequential)
        
        # Model configuration (CostGNNv3)
        "model_params": {
            "version": "v3",
            "node_feature_dim": 307,
            "hidden_dim": 128,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": True,
            "use_layer_norm": False,
            "dropout": 0.0,
        },
        
        # Fixed discrete hyperparameters
        # These are not optimized
        "fixed_discrete_params": {
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "use_temperature_annealing": True,
            "return_best": True,
            "use_lr_scheduling": True,
            "decoding_method": "beam",
            "use_gumbel_noise": False,
            "use_swa": False,
            "save_animation_data": False,
            "lr_warmup_steps": 0,
            "tau_decay": 0.999,
        },
    }
    
    # Run the search    
    print("=" * 60)
    print("GBJO Hyperparameter Search")
    print("=" * 60)
    print(f"Dataset path: {CONFIG['dataset_path']}")
    print(f"Model path: {CONFIG['model_path']}")
    print(f"Num plans: {CONFIG['num_plans']}")
    print(f"Max trials: {CONFIG['max_trials']}")
    print(f"Optimization steps: {CONFIG['optimization_steps']}")
    print(f"Model version: {CONFIG['model_params'].get('version', 'v2')}")
    print(f"Query parallelism: {CONFIG.get('num_workers', 1)} workers")
    print("=" * 60)
    
    start_time = time.time()
    
    run_cma_search(CONFIG)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Query parallelism: {CONFIG.get('num_workers', 1)} workers")
    print(f"{'='*60}")
