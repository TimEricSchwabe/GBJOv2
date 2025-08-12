import sys
import os
import numpy as np
import torch
import random
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from src.create_data.create_optimization_data import SPARQLQuery
from data import Triple, Join, Query, Entity
from model import CostGNNv2
from optimization import optimize_query_gumbel
from utils.data_utils import (
    adjacency_to_query_with_real_triples,
    validate_plan,
    load_sparql_queries,
)


def run_gradient_optimization_max_times(query, model, device, optimization_steps, optimization_params, 
                                       optimization_function, max_runs, query_index):
    """Run gradient descent max_runs times and return all valid costs."""
    torch_data = query.torch_data[0]
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]
    
    if torch_data is None:
        return []
    
    costs = []
    for run_idx in range(max_runs):
        try:
            result = optimization_function(
                torch_data, model, device, 
                optimization_steps=optimization_steps, 
                verbose=False,
                **optimization_params
            )
            
            # Handle different return types
            if len(result) == 4:
                final_adjacency, triples_num, cost, _ = result
            elif len(result) == 3:
                final_adjacency, triples_num, cost = result
            else:
                continue
            
            # Validate plan
            plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            is_valid, _ = validate_plan(plan, triple_objs)
            
            if is_valid:
                costs.append(cost)
                
        except Exception as e:
            continue
    
    return costs


def simulate_n_runs(all_costs, n, num_simulations=1):
    """Simulate running gradient descent n times by sampling from all_costs."""
    if len(all_costs) < n:
        return []
    
    best_costs = []
    for _ in range(num_simulations):
        sampled_costs = random.sample(all_costs, n)
        best_costs.append(min(sampled_costs))
    
    return best_costs


def analyze_k_vs_cost_efficient(sparql_queries, model_path, optimization_steps=500, 
                               optimization_params=None, max_n=10, num_queries=50):
    """Efficient K vs Cost analysis."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    sparql_queries = sparql_queries[:num_queries]
    print(f"Running gradient descent {max_n} times for {len(sparql_queries)} queries...")
    
    # Run gradient descent max_n times for each query
    all_query_costs = []
    start_time = time.time()
    
    for i, query in enumerate(sparql_queries):
        costs = run_gradient_optimization_max_times(
            query, model, device, optimization_steps, optimization_params,
            optimize_query_gumbel, max_n, i
        )
        all_query_costs.append(costs)
        
        if (i + 1) % max(1, len(sparql_queries) // 10) == 0:
            print(f"  Processed {i + 1}/{len(sparql_queries)} queries")
    
    print(f"Gradient descent completed in {time.time() - start_time:.2f} seconds")
    
    results = {}
    
    for n in range(1, max_n + 1):
        all_simulated_costs = []
        
        for query_costs in all_query_costs:
            if len(query_costs) >= n:
                simulated_costs = simulate_n_runs(query_costs, n)
                if simulated_costs:
                    all_simulated_costs.extend(simulated_costs)
        
        if all_simulated_costs:
            results[n] = {
                'median_cost': np.median(all_simulated_costs),
                'mean_cost': np.mean(all_simulated_costs),
                'std_cost': np.std(all_simulated_costs),
                'individual_costs': all_simulated_costs
            }
        
        print(f"  N = {n}: {len(all_simulated_costs)} simulated results, median = {results[n]['median_cost']:.2f}")
    
    return results


def create_plot(results, save_directory):
    """Create line plot of median cost vs N."""
    if isinstance(list(results.keys())[0], str):
        results = {int(k): v for k, v in results.items()}
    
    n_values = sorted(results.keys())
    median_costs = [results[n]['median_cost'] for n in n_values]
    
    plt.figure(figsize=(10, 3))
    plt.plot(n_values, median_costs, 'o-', linestyle='-', color='black')
    plt.xlabel('$k$', fontsize=20)
    plt.ylabel('Median Cost', fontsize=20)
    plt.xticks(n_values, fontsize=18)
    
    plot_file = os.path.join(save_directory, 'search_fronts.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_file}")


if __name__ == "__main__":
    config = {
        "queries_file": "datasets/wikidata_star_plan_datasets_optimization/queries.pkl",
        "model_path": "datasets/models/wikidata/star_model.pt",
        "num_queries": 20,
        "optimization_steps": 100,
        "max_k": 10,
        "optimization_params": {
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


    LOAD_EXISTING_RESULTS = True

    if not LOAD_EXISTING_RESULTS:
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = os.path.join("k_vs_cost_results", f"efficient_analysis_{timestamp}")
        os.makedirs(save_directory, exist_ok=True)
        
        print("K vs Cost Analysis (Efficient Implementation)")
        print(f"Saving results to: {save_directory}")
        
        # Save config
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Load queries and run analysis
        sparql_queries = load_sparql_queries(config['queries_file'], config['num_queries'])
        
        start_time = time.time()
        results = analyze_k_vs_cost_efficient(
            sparql_queries,
            config['model_path'],
            optimization_steps=config['optimization_steps'],
            optimization_params=config['optimization_params'],
            max_k=config['max_k'],
            num_queries=config['num_queries']
        )
        total_time = time.time() - start_time
        
        # Save results
        results_for_json = {str(n): {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                    for k, v in data.items()} 
                        for n, data in results.items()}
        
        with open(os.path.join(save_directory, "results.json"), 'w') as f:
            json.dump(results_for_json, f, indent=2)

    else:
        save_directory = "k_vs_cost_results/efficient_analysis_20250729_154048"
        # Load results
        with open(os.path.join(save_directory, "results.json"), 'r') as f:
            results = json.load(f)
        
        print("Loaded existing results")

    # Create plot
    create_plot(results, save_directory)
    
    # Print summary
    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print("\nResults:")
    # Convert string keys to integers for consistent printing
    if isinstance(list(results.keys())[0], str):
        results_for_print = {int(k): v for k, v in results.items()}
    else:
        results_for_print = results
        
    for n in sorted(results_for_print.keys()):
        data = results_for_print[n]
        print(f"N = {n:2d}: median = {data['median_cost']:8.2f}, mean = {data['mean_cost']:8.2f}")
