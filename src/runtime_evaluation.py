"""
Runtime evaluation script for query optimization approaches.

This script generates random queries of increasing sizes and measures the runtime
of different optimization approaches: DP, Greedy, GBJO, II, GEQO, CMA-ES, NeuralSort.

All functions are redefined to make it easily portable to other machines (not elegant)
"""

import sys
import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import warnings
from dataclasses import dataclass
import torch.optim as optim
import graphviz
import itertools

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

import src.data as data_module
sys.modules['explicit_join_model.data'] = data_module
sys.modules['explicit_join_model'] = sys.modules['src']



import requests
import re
from typing import Union

# Add the parent directory to Python path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, spmm
from typing import Callable, Union

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

#import scienceplots
#plt.style.use('science')


from optimization import (
    GBJO,
    GreedySearch,
    DPLinear,
    IterativeImprovement,
    GEQO,
    CMA,
    NeuralSort,
)
from model import CostGNNv2
from data import Triple, Join, Query, Entity





def left_deep_adj_from_perm(pi):
    """
    Create adjacency matrix for a left-deep join tree from a permutation.
    
    Args:
        pi: Tensor of length n with the (0-based) permutation of triple nodes.
        
    Returns:
        A: (2n-1, 2n-1) adjacency matrix for a left-deep tree
    """
    n = len(pi)
    N = 2 * n - 1
    A = torch.zeros(N, N, dtype=torch.float32)
    # indices: triple 0..n-1, join nodes n..2n-2 (root = 2n-2)
    # first join joins pi0 and pi1 -> node idx = n
    A[pi[0], n] = 1.0
    A[pi[1], n] = 1.0
    last_join = n
    for k in range(2, n):
        new_join = n + k - 1
        A[last_join, new_join] = 1.0
        A[pi[k], new_join] = 1.0
        last_join = new_join
    return A



def create_dummy_model(device: str = 'cpu', use_compile: bool = False) -> CostGNNv2:
    """
    Create a dummy CostGNNv2 model for benchmarking.
    
    Args:
        device: Target device
        use_compile: Whether to compile the model with torch.compile
        
    Returns:
        Initialized CostGNNv2 model
    """
    node_feature_dim = 307  # Standard feature dimension
    hidden_dim = 512        # Standard hidden dimension
    
    model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    model.eval()
    
    # Freeze parameters to speed up backward pass
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Initialize with random weights (for benchmarking purposes)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    # Compile the model if requested
    if use_compile:
        print(f"Compiling model with torch.compile...")
        model = torch.compile(model)
        print(f"Model compilation completed.")
    
    return model

def warmup_compiled_model(model, device: str = 'cpu', warmup_cycles: int = 5, verbose: bool = True):
    """
    Warmup a compiled model to trigger JIT compilation and kernel optimization.
    
    Args:
        model: The compiled model to warmup
        device: Target device
        warmup_cycles: Number of warmup cycles to run
        verbose: Whether to print warmup progress
    """
    if verbose:
        print(f"Warming up compiled model with {warmup_cycles} cycles...")
    
    model.eval()
    
    # Create dummy data of various sizes for comprehensive warmup
    warmup_sizes = [3, 5, 8, 10]  # Different query sizes for warmup
    
    with torch.no_grad():
        for cycle in range(warmup_cycles):
            if verbose:
                print(f"  Warmup cycle {cycle + 1}/{warmup_cycles}")
            
            for size in warmup_sizes:
                # Generate dummy query data
                dummy_data = generate_random_query_data(size, device, seed=42)
                
                # Create dummy edge weights
                n_nodes = dummy_data.num_nodes
                src, dst = torch.where(~torch.eye(n_nodes, dtype=torch.bool, device=device))
                edge_index = torch.stack([src, dst], dim=0)
                edge_weights = torch.rand(edge_index.size(1), device=device)
                
                # Forward pass to trigger compilation
                _ = model(dummy_data.x, edge_index, edge_weight=edge_weights)
    
    if verbose:
        print("Model warmup completed!")

def generate_random_query_data(n_triples: int, device: str = 'cpu', seed: int = None) -> Data:
    """
    Generate random query data for benchmarking.
    
    Args:
        n_triples: Number of triple patterns in the query
        device: Target device
        seed: Random seed for reproducibility
        
    Returns:
        torch_geometric.data.Data object representing a random query
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Calculate total number of nodes: n triple nodes + (n-1) join nodes
    n_nodes = 2 * n_triples - 1
    feature_dim = 307
    
    # Create random node features
    node_features = torch.randn(n_nodes, feature_dim, device=device)
    
    # Set join node features: last dimension = 1.0 for join nodes, 0.0 for triple nodes
    # Triple nodes: indices 0 to n_triples-1
    # Join nodes: indices n_triples to n_nodes-1
    node_features[:n_triples, -1] = 0.0  # Triple nodes
    node_features[n_triples:, :] = 0.0   # Zero out join node features
    node_features[n_triples:, -1] = 1.0  # Mark join nodes
    
    # Create a random valid left-deep adjacency matrix using permutation
    perm = torch.randperm(n_triples, device=device)
    adjacency = left_deep_adj_from_perm(perm)
    
    # Convert adjacency matrix to edge_index format
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
    
    # Create torch_geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=n_nodes
    )
    
    return data

def create_dummy_triples(n_triples: int) -> List[Triple]:
    """
    Create dummy triple objects for greedy optimization.
    
    Args:
        n_triples: Number of triples to create
        
    Returns:
        List of dummy Triple objects
    """
    triples = []
    for i in range(n_triples):
        s = Entity(name=f"?s{i}")
        p = Entity(name=f"<p{i}>")
        o = Entity(name=f"?o{i}")
        triples.append(Triple(s=s, p=p, o=o))
    
    return triples

def benchmark_method(method_func, query_data, model, device, method_name: str, **kwargs) -> Tuple[float, bool]:
    """
    Benchmark a single optimization method.
    
    Args:
        method_func: The optimization function to benchmark
        query_data: Query data
        model: Cost model
        device: Compute device
        method_name: Name of the method for logging
        **kwargs: Additional arguments for the method
        
    Returns:
        Tuple of (runtime_seconds, success_flag)
    """
    try:
        query_data = query_data.to(device)
        start_time = time.time()

        # Keep all methods silent for benchmarking
        if method_name == "Greedy":
            n_triples = (query_data.num_nodes + 1) // 2
            original_triples = create_dummy_triples(n_triples)
            _ = method_func(query_data, model, original_triples, device, verbose=False)

        elif method_name == "GBJO":
            _ = method_func(
                query_data,
                model,
                device,
                optimization_steps=kwargs.get("optimization_steps", 250),
                verbose=False,
                **{k: v for k, v in kwargs.items() if k != "optimization_steps"},
            )

        elif method_name == "NeuralSort":
            _ = method_func(
                query_data,
                model,
                device,
                optimization_steps=kwargs.get("optimization_steps", 250),
                **{k: v for k, v in kwargs.items() if k != "optimization_steps"},
            )

        elif method_name == "CMA":
            _ = method_func(
                query_data,
                model,
                device,
                optimization_steps=kwargs.get("optimization_steps", 250),
                verbose=False,
                **{k: v for k, v in kwargs.items() if k != "optimization_steps"},
            )

        elif method_name in ("II", "GEQO"):
            # Both accept (query_data, model, optimization_steps, device)
            _ = method_func(
                query_data,
                model,
                kwargs.get("optimization_steps", 250),
                device,
            )

        elif method_name == "DP":
            _ = method_func(query_data, model, device)

        else:
            raise ValueError(f"Unknown method_name for benchmarking: {method_name}")

        end_time = time.time()
        runtime = end_time - start_time
        return runtime, True

    except Exception as e:
        print(f"Error in {method_name}: {e}")
        return float("inf"), False

def run_runtime_evaluation(
    optimization_steps: int = 10,
    query_sizes: List[int] = list(range(3, 11)),
    num_trials_per_size: int = 5,
    device: str = 'cpu',
    save_plot: bool = True,
    plot_filename: str = 'runtime_comparison.png',
    include_dp: bool = True,
    use_compile: bool = False
) -> Dict[str, Dict[int, List[float]]]:
    """
    Run runtime evaluation across different query sizes.
    
    Args:
        query_sizes: List of query sizes to test
        num_trials_per_size: Number of trials per query size
        device: Compute device
        save_plot: Whether to save the plot
        plot_filename: Filename for the saved plot
        include_dp: Whether to include Dynamic Programming in the evaluation
        use_compile: Whether to compile the model with torch.compile
        
    Returns:
        Dictionary containing runtime results for each method and query size
    """
    print(f"Running runtime evaluation on device: {device}")
    print(f"Query sizes: {query_sizes}")
    print(f"Trials per size: {num_trials_per_size}")
    print(f"Include DP: {include_dp}")
    print(f"Use compile: {use_compile}")
    
    # Create dummy model
    model = create_dummy_model(device, use_compile=use_compile)
    
    # Warmup the optimization pipeline by running actual benchmarks
    if True:
        print("Warming up optimization pipeline...")
        warmup_sizes = list(range(3, 8))  # Sizes 3, 4, 5
        warmup_trials = 3
        
        for warmup_size in tqdm(warmup_sizes, desc="Warmup sizes"):
            for warmup_trial in range(warmup_trials):
                # Generate random query for warmup
                warmup_query_data = generate_random_query_data(warmup_size, device, seed=warmup_trial + 1000)
                
                # Run all methods without recording results
                try:
                    
                    # Warmup Greedy
                    _, _ = benchmark_method(
                        GreedySearch, warmup_query_data, model, device, "Greedy"
                    )
                    
                    warmup_gbjo_config = {
                        "optimization_steps": 50,  # Shorter for warmup
                        "learning_rate": 1.0,
                        "lambda_acyclic": 1000.0,
                        "lambda_triple_in": 1000.0,
                        "lambda_triple_out": 1000.0,
                        "lambda_join_in": 500.0,
                        "lambda_join_out": 1000.0,
                        "lambda_left_linear": 1000.0,
                        "lambda_entropy": 0.0,
                        "lambda_total_penalty": 1.0,
                        "init_tau": 10.0,
                        "min_tau": 1.0,
                        "tau_decay": 0.999,
                        "use_temperature_annealing": True,
                        "return_best": True,
                        "min_penalty_threshold": 0.1,
                        "use_lambda_ramping": False,
                        "logit_sampling": "softmax",
                        "save_animation_data": False,
                        "animation_save_interval": 10,
                    }

                    _, _ = benchmark_method(
                        GBJO,
                        warmup_query_data,
                        model,
                        device,
                        "GBJO",
                        **warmup_gbjo_config,
                    )

                    # Optional warmups for other methods (best-effort)
                    _, _ = benchmark_method(DPLinear, warmup_query_data, model, device, "DP")
                    _, _ = benchmark_method(IterativeImprovement, warmup_query_data, model, device, "II", optimization_steps=50)
                    _, _ = benchmark_method(GEQO, warmup_query_data, model, device, "GEQO", optimization_steps=50)
                    _, _ = benchmark_method(NeuralSort, warmup_query_data, model, device, "NeuralSort", optimization_steps=50, learning_rate=0.1)
                    # CMA depends on nevergrad; keep best-effort
                    _, _ = benchmark_method(CMA, warmup_query_data, model, device, "CMA", optimization_steps=50)
                    
                except Exception as e:
                    print(f"Warning: Warmup failed for size {warmup_size}, trial {warmup_trial}: {e}")
                    continue
        
        print("Warmup completed!")
    
    results = {
        "Greedy": {size: [] for size in query_sizes},
        "GBJO": {size: [] for size in query_sizes},
        "II": {size: [] for size in query_sizes},
        "GEQO": {size: [] for size in query_sizes},
        "CMA": {size: [] for size in query_sizes},
        "NeuralSort": {size: [] for size in query_sizes},
    }
    
    if include_dp:
        results["DP"] = {size: [] for size in query_sizes}
    
    # Method configurations
    gbjo_config = {
        "optimization_steps": optimization_steps,
        "learning_rate": 1.0,
        "lambda_acyclic": 1000.0,
        "lambda_triple_in": 1000.0,
        "lambda_triple_out": 1000.0,
        "lambda_join_in": 500.0,
        "lambda_join_out": 1000.0,
        "lambda_left_linear": 1000.0,
        "lambda_entropy": 0.0,
        "lambda_total_penalty": 1.0,
        "init_tau": 10.0,
        "min_tau": 1.0,
        "tau_decay": 0.999,
        "use_temperature_annealing": True,
        "return_best": True,
        "min_penalty_threshold": 0.1,
        "use_lambda_ramping": False,
        "logit_sampling": "softmax",
        "save_animation_data": False,
        "animation_save_interval": 10,
    }

    # Method-specific knobs (kept small-ish by default)
    ii_config = {"optimization_steps": 50} # TODO
    geqo_config = {"optimization_steps": 50}
    cma_config = {"optimization_steps": 1000}
    neuralsort_config = {
        "optimization_steps": optimization_steps,
        "learning_rate": 0.1,
        "init_tau": 1.0,
        "tau_decay": 0.99,
        "min_tau": 0.1,
        "return_best": True,
    }
    
    # Run evaluation for each query size
    for query_size in tqdm(query_sizes, desc="Query sizes"):
        print(f"\nEvaluating query size: {query_size}")
        
        for trial in tqdm(range(num_trials_per_size), desc="Trials", leave=False):
            # Generate random query
            query_data = generate_random_query_data(query_size, device, seed=trial)
            
            # Benchmark DP (only if enabled)
            if include_dp:
                dp_time, dp_success = benchmark_method(
                    DPLinear, query_data, model, device, "DP"
                )
                if dp_success:
                    results["DP"][query_size].append(dp_time)
                print(f"  DP trial {trial+1}: {dp_time:.4f}s")
            
            # Benchmark Greedy
            greedy_time, greedy_success = benchmark_method(
                GreedySearch, query_data, model, device, "Greedy"
            )
            if greedy_success:
                results["Greedy"][query_size].append(greedy_time)
            print(f"  Greedy trial {trial+1}: {greedy_time:.4f}s")
            
            # Benchmark GBJO
            gbjo_time, gbjo_success = benchmark_method(
                GBJO, query_data, model, device, "GBJO", **gbjo_config
            )
            if gbjo_success:
                results["GBJO"][query_size].append(gbjo_time)
            print(f"  GBJO trial {trial+1}: {gbjo_time:.4f}s")

            # Benchmark Iterative Improvement
            ii_time, ii_success = benchmark_method(
                IterativeImprovement, query_data, model, device, "II", **ii_config
            )
            if ii_success:
                results["II"][query_size].append(ii_time)
            print(f"  II trial {trial+1}: {ii_time:.4f}s")

            # Benchmark GEQO
            geqo_time, geqo_success = benchmark_method(
                GEQO, query_data, model, device, "GEQO", **geqo_config
            )
            if geqo_success:
                results["GEQO"][query_size].append(geqo_time)
            print(f"  GEQO trial {trial+1}: {geqo_time:.4f}s")

            # Benchmark CMA-ES (Nevergrad)
            cma_time, cma_success = benchmark_method(
                CMA, query_data, model, device, "CMA", **cma_config
            )
            if cma_success:
                results["CMA"][query_size].append(cma_time)
            print(f"  CMA trial {trial+1}: {cma_time:.4f}s")

            # Benchmark NeuralSort
            ns_time, ns_success = benchmark_method(
                NeuralSort, query_data, model, device, "NeuralSort", **neuralsort_config
            )
            if ns_success:
                results["NeuralSort"][query_size].append(ns_time)
            print(f"  NeuralSort trial {trial+1}: {ns_time:.4f}s")
    
    # Calculate and print summary statistics
    print("\n" + "="*50)
    print("RUNTIME EVALUATION SUMMARY")
    print("="*50)
    
    for method in results:
        print(f"\n{method} Method:")
        for size in query_sizes:
            if results[method][size]:
                times = results[method][size]
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"  Size {size}: {mean_time:.4f}±{std_time:.4f}s (n={len(times)})")
            else:
                print(f"  Size {size}: No successful runs")
    
    # Create runtime plot
    create_runtime_plot(results, query_sizes, save_plot, plot_filename)
    
    return results

def create_runtime_plot(
    results: Dict[str, Dict[int, List[float]]],
    query_sizes: List[int],
    save_plot: bool = True,
    plot_filename: str = 'runtime_comparison.pdf'
):
    """
    Create and display runtime comparison plot.
    
    Args:
        results: Runtime results dictionary
        query_sizes: List of query sizes
        save_plot: Whether to save the plot
        plot_filename: Filename for the saved plot
    """
    plt.figure(figsize=(6, 3.5))

    fontsize = 12
    
    colors = {
        "DP": "#999999",
        "Greedy": "#666666",
        "GBJO": "black",
        "II": "#9b59b6",
        "GEQO": "#f39c12",
        "CMA": "#e91e63",
        "NeuralSort": "#1abc9c",
    }
    markers = {
        "DP": "o",
        "Greedy": "s",
        "GBJO": "^",
        "II": "D",
        "GEQO": "P",
        "CMA": "X",
        "NeuralSort": "v",
    }
    linestyles = {
        "DP": ":",
        "Greedy": "--",
        "GBJO": "-",
        "II": "-.",
        "GEQO": (0, (3, 1, 1, 1)),
        "CMA": (0, (1, 1)),
        "NeuralSort": (0, (5, 2)),
    }
    
    for method in results:
        sizes = []
        mean_times = []
        std_times = []
        
        for size in query_sizes:
            if results[method][size]:
                times = results[method][size]
                sizes.append(size)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))
        
        if sizes:
            plt.plot(
                sizes,
                mean_times,
                label=method,
                color=colors.get(method, "black"),
                marker=markers.get(method, "o"),
                linestyle=linestyles.get(method, "-"),
                markersize=6,
                linewidth=1.5,
            )
    
    plt.xlabel('Query Size', fontsize=fontsize)
    plt.ylabel('Time (s)', fontsize=fontsize)

    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    
    plt.legend(fontsize=fontsize-1)
    plt.yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved as: {plot_filename}")
    
    plt.show()

def save_results_to_file(results: Dict, filename: str = 'runtime_results.txt'):
    with open(filename, 'w') as f:
        f.write("Runtime Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        for method in results:
            f.write(f"{method} Method:\n")
            for size in sorted(results[method].keys()):
                if results[method][size]:
                    times = results[method][size]
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    f.write(f"  Size {size}: {mean_time:.6f}±{std_time:.6f}s (trials: {len(times)})\n")
                    f.write(f"    Individual times: {[f'{t:.6f}' for t in times]}\n")
                else:
                    f.write(f"  Size {size}: No successful runs\n")
            f.write("\n")
    
    print(f"Detailed results saved to: {filename}")

if __name__ == "__main__":
    # Configuration
    config = {
        'query_sizes': list(range(3, 12)), 
        'num_trials_per_size': 1,           # Number of trials per size
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimization_steps': 10,
        'save_plot': True,
        'plot_filename': 'runtime_comparison.pdf',
        'include_dp': True,  # Set to False to exclude DP from the comparison
        'use_compile': False  # Set to True to enable torch.compile optimization TODO we need to implement this again
    }
    
    print("Starting Runtime Evaluation")
    print(f"Configuration: {config}")
    
    # Run the evaluation
    results = run_runtime_evaluation(**config)
    
    # Save detailed results
    save_results_to_file(results, filename='runtime_results.txt')
    
    print("\nRuntime evaluation completed!")
