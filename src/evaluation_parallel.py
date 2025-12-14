"""
Parallel evaluation script for query optimization.

This script evaluates different optimization strategies (gradient-based, greedy, random)
on SPARQL queries in parallel and compares their performance using a trained cost model.
Removes all visualization and plotting, focusing only on detailed results.
"""

import os
# critical: must be set before importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
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
import graphviz

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

# Import the classes
from src.create_data.create_optimization_data import SPARQLQuery
from data import Triple, Join, Query, Entity
from model import CostGNNv2, CostGNNv3

from optimization import (
    GBJO,
    GEQO,
    NeuralSort,
    GreedySearch,
    random_join_plan,
    DPLinear,
    exhaustive_leftdeep_best_plan,
    optimize_query_nevergrad,
    IterativeImprovement
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
from visualization.plot_optimization_results import (
    load_data, 
    plot_overall_boxplot, 
    plot_mean_costs_bar, 
    plot_lineplots_by_size, 
    plot_boxplot_per_size, 
    plot_scatter_correlations
)

# Add module compatibility for old pickle files
import sys
import src.data as data_module
sys.modules['explicit_join_model.data'] = data_module
sys.modules['explicit_join_model'] = sys.modules['src']


def add_fingerprints_to_query_data(query_data, fingerprint_dim=64):
    """
    Add random Gaussian fingerprints to join nodes in query data.
    Matches AddRandomGaussianFingerprints from data_loader.py
    """
    x = query_data.x.clone()
    
    is_join = (x[:, -1] == 1.0)
    join_indices = torch.where(is_join)[0]
    n_joins = len(join_indices)
    
    if n_joins == 0:
        return query_data
    
    # random fingerprints, normalized (same as training)
    fingerprints = torch.randn(n_joins, fingerprint_dim, device=x.device)
    fingerprints = fingerprints / fingerprints.norm(dim=1, keepdim=True)
    
    for i, join_idx in enumerate(join_indices):
        x[join_idx, :fingerprint_dim] = fingerprints[i]
    
    query_data.x = x
    return query_data


def extract_join_order(plan, triple_objs):
    """
    Extract join order from a linear plan by finding the leaf at each level.
    
    For linear plans, at each Join node one child is a Triple (leaf).
    We collect leaves from root to bottom, then reverse to get the join order.
    
    Args:
        plan: Query object representing a join plan
        triple_objs: List of Triple objects (the original triples)
        
    Returns:
        List of indices representing the join order (from first joined to last joined)
    """
    if plan is None:
        return None
    
    def find_triple_index(triple):
        """Find the index of a triple in triple_objs by comparing string representation."""
        triple_str = str(triple)
        for i, t in enumerate(triple_objs):
            if str(t) == triple_str:
                return i
        return -1
    
    leaves_from_root = []
    
    # Handle case where root is just a Triple
    if isinstance(plan.root, Triple):
        return [find_triple_index(plan.root)]
        
    current = plan.root
    
    while isinstance(current, Join):
        left_is_leaf = isinstance(current.left, Triple)
        right_is_leaf = isinstance(current.right, Triple)
        
        if left_is_leaf and right_is_leaf:
            # Both are leaves - bottom of tree
            # Convention: left first, then right (this order handles the base pair)
            # When we reverse, right will be first (base), left will be second
            # e.g., Join(t0, t1) -> leaves=[0, 1] -> reversed=[1, 0] -> t1 joined with t0
            leaves_from_root.append(find_triple_index(current.left))
            leaves_from_root.append(find_triple_index(current.right))
            break
        elif left_is_leaf:
            # Left is leaf, right is Join - right-linear (most common for right-deep)
            # e.g. Join(t2, Join(t0, t1))
            leaves_from_root.append(find_triple_index(current.left))
            current = current.right
        elif right_is_leaf:
            # Right is leaf, left is Join - left-linear
            # e.g. Join(Join(t0, t1), t2)
            leaves_from_root.append(find_triple_index(current.right))
            current = current.left
        else:
            # Neither is leaf - bushy plan
            # For bushy plans, we can't easily map to a linear sequence
            # Just do a best-effort traversal to find leaves
            print("Warning: Bushy plan detected, join order extraction may be approximate")
            leaves_from_root.append(find_triple_index(current.right) if isinstance(current.right, Triple) else -1)
            current = current.left
            
    # Reverse to get join order (first joined first)
    return list(reversed(leaves_from_root))


def compute_all_join_costs(triple_objs, timeout=60):
    """
    Compute the cumulative costs for all possible join orderings.
    
    Args:
        triple_objs: List of Triple objects
        timeout: Maximum time in seconds for the entire computation (default: 60)
        
    Returns:
        Dictionary mapping tuple of triple indices (representing partial join order) 
        to cumulative cost at that point.
        Example: {(0,): 0, (1,): 0, (0, 1): 150, (0, 1, 2): 280, ...}
    """
    n = len(triple_objs)
    costs = {}
    start_time = time.time()
    
    # Level 0: Each single triple uses its cardinality
    for i in range(n):
        if time.time() - start_time > timeout:
            print(f"Warning: Cost computation timed out after {timeout}s at level 0")
            return costs
        try:
            cardinality = triple_objs[i].get_cardinality()
            costs[(i,)] = cardinality
        except Exception:
            costs[(i,)] = 0
    
    # Generate all permutations and compute costs at each level
    for perm in itertools.permutations(range(n)):
        if time.time() - start_time > timeout:
            print(f"Warning: Cost computation timed out after {timeout}s")
            return costs
            
        # Build the join tree incrementally and compute cost at each level
        current_node = triple_objs[perm[0]]
        
        for level in range(1, n):
            if time.time() - start_time > timeout:
                print(f"Warning: Cost computation timed out after {timeout}s")
                return costs
                
            # Join current_node with the next triple
            next_triple = triple_objs[perm[level]]
            join_node = Join(left=current_node, right=next_triple)
            
            # Compute cost for this partial join
            partial_order = tuple(perm[:level + 1])
            
            if partial_order not in costs:
                try:
                    # get_cost() computes c_out cost (cumulative)
                    cost = join_node.get_cost()
                    costs[partial_order] = cost
                except Exception as e:
                    # If SPARQL query fails, use infinity
                    costs[partial_order] = float('inf')
            
            # Update current_node for next iteration
            current_node = join_node
    
    return costs


def visualize_join_order_tree(triple_objs, gradient_plan, greedy_plan, dp_plan, save_path):
    """
    Visualize all possible join orderings as a tree with costs, highlighting
    the paths taken by gradient, greedy, and DP optimizers.
    
    Args:
        triple_objs: List of Triple objects
        gradient_plan: Query object from gradient optimization (or None)
        greedy_plan: Query object from greedy optimization (or None)
        dp_plan: Query object from DP optimization (or None)
        save_path: Path to save the visualization (without extension)
    """
    n = len(triple_objs)
    
    if n > 5:
        print(f"Skipping visualization: {n} triples exceeds limit of 5")
        return
    
    # Extract join orders from each plan
    gradient_order = extract_join_order(gradient_plan, triple_objs)
    greedy_order = extract_join_order(greedy_plan, triple_objs)
    dp_order = extract_join_order(dp_plan, triple_objs)
    
    # Compute costs for all permutations
    costs = compute_all_join_costs(triple_objs)
    
    # Create graphviz digraph
    graph = graphviz.Digraph(
        'Join Order Tree',
        comment='All possible join orderings with costs',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'line',
            'nodesep': '0.3',
            'ranksep': '0.8'
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': 'white',
            'fontname': 'Helvetica'
        },
        edge_attr={
            'dir': 'none'
        }
    )
    
    # Helper to create node ID from partial order
    def node_id(partial_order):
        return '_'.join(map(str, partial_order))
    
    # Helper to format cost for display
    def format_cost(cost):
        if cost == float('inf'):
            return '∞'
        elif cost >= 1e6:
            return f'{cost:.1e}'
        elif cost >= 1000:
            return f'{cost:.0f}'
        else:
            return f'{cost:.1f}'
    
    # Helper to check if an edge is on a highlighted path
    def edge_on_path(parent_order, child_order, plan_order):
        if plan_order is None:
            return False
        parent_len = len(parent_order)
        child_len = len(child_order)
        # Check if child_order is an extension of parent_order matching plan_order
        return (list(parent_order) == list(plan_order[:parent_len]) and 
                list(child_order) == list(plan_order[:child_len]))
    
    # Build tree level by level
    for level in range(n):
        # Create subgraph for this level to ensure same rank
        with graph.subgraph() as s:
            s.attr(rank='same')
            
            if level == 0:
                # Level 0: individual triples
                for i in range(n):
                    partial = (i,)
                    nid = node_id(partial)
                    cost = costs.get(partial, 0)
                    label = f't{i}|{format_cost(cost)}'
                    s.node(nid, label=label)
            else:
                # Subsequent levels: all permutation prefixes of length level+1
                seen = set()
                for perm in itertools.permutations(range(n)):
                    partial = tuple(perm[:level + 1])
                    if partial in seen:
                        continue
                    seen.add(partial)
                    
                    nid = node_id(partial)
                    cost = costs.get(partial, float('inf'))
                    # Label shows only the last triple added and the cumulative cost
                    last_triple = partial[-1]
                    label = f't{last_triple}|{format_cost(cost)}'
                    s.node(nid, label=label)
    
    # Add edges between levels
    for level in range(n - 1):
        seen_edges = set()
        if level == 0:
            # Edges from level 0 to level 1
            for perm in itertools.permutations(range(n)):
                if len(perm) < 2:
                    continue
                parent = (perm[0],)
                child = (perm[0], perm[1])
                
                edge_key = (parent, child)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                
                parent_nid = node_id(parent)
                child_nid = node_id(child)
                
                # Determine edge color based on highlighted paths
                colors = []
                if edge_on_path(parent, child, gradient_order):
                    colors.append('#3498db')  # Blue for gradient
                if edge_on_path(parent, child, greedy_order):
                    colors.append('#2ecc71')  # Green for greedy
                if edge_on_path(parent, child, dp_order):
                    colors.append('#e74c3c')  # Red for DP
                
                if colors:
                    # Use colon-separated colors for multiple paths
                    edge_color = ':'.join(colors)
                    graph.edge(parent_nid, child_nid, color=edge_color, penwidth='3')
                else:
                    graph.edge(parent_nid, child_nid, color='#cccccc')
        else:
            # Edges from level to level+1
            for perm in itertools.permutations(range(n)):
                if len(perm) < level + 2:
                    continue
                parent = tuple(perm[:level + 1])
                child = tuple(perm[:level + 2])
                edge_key = (parent, child)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                
                parent_nid = node_id(parent)
                child_nid = node_id(child)
                
                # Determine edge color
                colors = []
                if edge_on_path(parent, child, gradient_order):
                    colors.append('#3498db')  # Blue
                if edge_on_path(parent, child, greedy_order):
                    colors.append('#2ecc71')  # Green
                if edge_on_path(parent, child, dp_order):
                    colors.append('#e74c3c')  # Red
                
                if colors:
                    edge_color = ':'.join(colors)
                    graph.edge(parent_nid, child_nid, color=edge_color, penwidth='3')
                else:
                    graph.edge(parent_nid, child_nid, color='#cccccc')
    
    # Add legend
    with graph.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Method', style='rounded', color='gray')
        legend.node('legend_gradient', 'Gradient', fillcolor='#3498db', fontcolor='white')
        legend.node('legend_greedy', 'Greedy', fillcolor='#2ecc71', fontcolor='white')
        legend.node('legend_dp', 'DP', fillcolor='#e74c3c', fontcolor='white')
        legend.edge('legend_gradient', 'legend_greedy', style='invis')
        legend.edge('legend_greedy', 'legend_dp', style='invis')
    
    # Render the graph
    try:
        graph.render(save_path, format='png', cleanup=True)
        print(f"Saved join order tree visualization to: {save_path}.png")
    except Exception as e:
        print(f"Error rendering join order tree: {e}")
    
    return graph


def process_single_query(args):
    """
    Process a single query with all optimization methods.
    
    Args:
        args: Tuple containing (query_index, query, model_path, device_str, optimization_params, 
              optimization_function_name, use_exhaustive, use_true_costs, use_dp, optimization_steps, dp_limit, save_directory)
    
    Returns:
        Dictionary with detailed results for this query
    """
    (query_index, query, model_path, device_str, optimization_params, 
     optimization_algorithms, use_exhaustive, use_true_costs, optimization_steps, dp_limit, save_directory, model_params) = args
    
    # Set device
    device = torch.device(device_str)
    
    # Load model
    if model_params is None:
        node_feature_dim = 307
        hidden_dim = 128
        model = CostGNNv3(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    else:
        params = model_params.copy()
        if params.get('version') == 'v3':
            model = CostGNNv3(**params).to(device)
        elif params.get('version') == 'v2':
            model = CostGNNv2(**params).to(device)
        else:
            raise ValueError(f"Unknown model version: {params.get('version')}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    
    try:
        # Get the torch data from one of the plans
        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]



        torch_data = add_fingerprints_to_query_data(torch_data, fingerprint_dim=64)


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


        if "GEQO" in optimization_algorithms:
            GEQO_adj, GEQO_pred_cost = GEQO(torch_data, model, optimization_steps, device)
            GEQO_plan = adjacency_to_query_with_real_triples(GEQO_adj, len(triple_objs), triple_objs)
            GEQO_real_cost = GEQO_plan.root.get_cost()

            result["plans"]["GEQO"] = {
                "predicted_cost": float(GEQO_pred_cost),
                "plan_string": plan_to_string(GEQO_plan) if GEQO_plan else None
            }
            if use_true_costs:
                result["plans"]["GEQO"]["real_cost"] = float(GEQO_real_cost)


        if "IterativeImprovement" in optimization_algorithms:
            # Run Iterative Improvement
            II_adj, II_pred_cost = IterativeImprovement(torch_data, model, optimization_steps, device)
            II_plan = adjacency_to_query_with_real_triples(II_adj, len(triple_objs), triple_objs)
            II_real_cost = II_plan.root.get_cost()

            result["plans"]["II"] = {
                "predicted_cost": float(II_pred_cost),
                "plan_string": plan_to_string(II_plan) if II_plan else None
            }
            if use_true_costs:
                result["plans"]["II"]["real_cost"] = float(II_real_cost)


        
        # Run DP-based best plan search (only if enabled)
        best_adj = None
        best_pred_cost = float('inf')
        best_pred_plan = None
        true_cost_best_pred = float('inf')
        
        # Only run DP if enabled AND query size is within the limit
        if 'DP' in optimization_algorithms and len(query_triples) <= dp_limit:
            try:
                best_adj, best_pred_cost = DPLinear(torch_data, model, device)
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
        if 'GBJO' in optimization_algorithms:
            # Run gradient optimization k times and pick the best result
            k = optimization_params.get('k', 1)  # Number of runs, default to 1
            best_adjacency = None
            best_triples_num = None
            best_grad_pred_cost = float('inf')
            best_animation_data = None
            
            for run_idx in range(k):
                optimization_result = GBJO(
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

        
        # Run greedy optimization
        if 'GreedySearch' in optimization_algorithms:
            greedy_plan, greedy_pred_cost = GreedySearch(
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
        
        # Create a random plan
        try:
            log_pred_cost = model(query.torch_data[-1].x, edge_index=query.torch_data[-1]['edge_index']).item()
            random_pred_cost = float(np.exp(log_pred_cost))

            if use_true_costs:
                random_true_cost = query.join_plans[-1].root.get_cost()
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
            result["plans"]["random"]["real_cost"] = float(random_true_cost)
        
        # Add exhaustive comparison only if exhaustive search was performed
        if use_exhaustive and exhaustive_plan is not None:
            result["greedy_equal_exhaustive"] = plans_are_equivalent(greedy_plan, exhaustive_plan)
            result["gradient_equal_exhaustive"] = plans_are_equivalent(gradient_plan, exhaustive_plan)
        
        # Generate join order tree visualization for small queries with true costs
        if use_true_costs and len(triple_objs) <= -1:
            try:
                # Create visualizations directory if needed
                viz_dir = os.path.join(save_directory, "join_order_trees")
                os.makedirs(viz_dir, exist_ok=True)
                
                save_path = os.path.join(viz_dir, f"query_{query_index}_join_tree")
                visualize_join_order_tree(
                    triple_objs, 
                    gradient_plan, 
                    greedy_plan, 
                    best_pred_plan,  # DP plan
                    save_path
                )
            except Exception as e:
                print(f"Warning: Could not generate join order tree for query {query_index}: {e}")
        
        return result
        
    except Exception as e:
        #raise e
        print(f"Error processing query {query_index}: {e}")
        return None


def evaluate_optimization_parallel(sparql_queries, model_path, num_queries=None, optimization_steps=500, 
                                 optimization_params=None, optimization_algorithms=None, save_directory=".", 
                                 use_exhaustive=True, use_true_costs=True, num_workers=None, dp_limit=9, model_params=None):
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
        model_params: Dictionary of model parameters for CostGNNv3
        
    Returns:
        List of detailed results for each query
    """

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
                optimization_algorithms, use_exhaustive, use_true_costs, optimization_steps, dp_limit, save_directory, model_params)
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
                #raise e
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
        "queries_file": "/home/tim/query_optimization/datasets/plans/wikidata_star_plan_datasets_optimization/queries.pkl",
        "model_path": "/home/tim/query_optimization/training_results/wikidata-star-new/model.pt",
        "num_queries": 200,
        "optimization_steps": 500, # 2500
        "use_exhaustive": False,
        "use_dp": True,
        "dp_limit": 9,  # Set the limit here (e.g., 15 for star queries)
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": 3,  # Use all available cores
        "optimization_algorithms": ["GBJO", "DP", "GreedySearch", "IterativeImprovement", "GEQO"],
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": False,
            "use_layer_norm": True,
            "dropout": 0.0,
        },
        "optimization_params": {
            "k": 3,  # Number of gradient optimization runs
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
            "decoding_method": "greedy",
            "use_gumbel_noise": False
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
        "optimization_algorithms": ["GBJO", "DP", "GreedySearch", "IterativeImprovement", "GEQO"],
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": False,
            "use_layer_norm": True,
            "dropout": 0.0,
        },
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
            "decoding_method": "greedy",
            "use_gumbel_noise": True
        }
    }

    config_lubm_star = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm_star_plan_datasets_optimization/optimization_stars_3_to_14/queries.pkl",
        "model_path": "/home/tim/query_optimization/datasets/models/lubm/6-layers-v3-with-layer-norm/model.pt",
        "num_queries": 200,
        "max_query_size": None,  # Filter queries larger than this (None for no filter)
        "optimization_steps": 500,
        "use_exhaustive": False,
        "use_dp": True,
        "dp_limit": 9,  # Set the limit here (e.g., 15 for star queries)
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": 6,  # Use all available cores
        "optimization_algorithms": ["GBJO", "DP", "GreedySearch", "IterativeImprovement", "GEQO"],
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": False,
            "use_layer_norm": True,
            "dropout": 0.0,
        },
        "optimization_params": { # params for GBJO
            "k": 1,  # Number of gradient optimization runs
            "learning_rate": 1.7, # 1.7
            "lambda_acyclic": 3081.0,
            "lambda_triple_in": 3714.0,
            "lambda_triple_out": 135.0,
            "lambda_join_in": 1742.0,
            "lambda_join_out": 1558.0,
            "lambda_entropy": 0.0,
            "lambda_total_penalty": 2.6, # 2.6
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
            "decoding_method": "greedy",
            "use_gumbel_noise": False
        }
    }

    config_lubm_path = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm_path_plan_datasets_optimization/optimization_paths_3_to_5/queries.pkl",
        "model_path": "/home/tim/query_optimization/training_results/lubm-path-nice-v3-6-layer/model.pt",
        "num_queries": 200,
        "optimization_steps": 500,
        "use_exhaustive": False,
        "max_query_size": 5,  # Filter queries larger than this (None for no filter)
        "use_dp": True,
        "use_true_costs": True,
        "save_path": "optimization_results",
        "optimization_algorithms": ["GBJO", "DP", "GreedySearch", "IterativeImprovement", "GEQO"],
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": False,
            "use_layer_norm": False,
            "dropout": 0.0,
        },
        "num_workers": 10,  # Use all available cores
        "optimization_params": {
            "k": 1,  # Number of gradient optimization runs - 5
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
            "return_best": False,
            "min_penalty_threshold": 8.6,
            "use_lambda_ramping": True,
            "logit_sampling": "dual-softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 6.8, # 6.8
            "lr_warmup_steps": 200,
            "gradient_clip_norm": 1.9,
            "use_lr_scheduling": False,
            "decoding_method": "greedy",
            "use_gumbel_noise": False
        }
    }

    config_lubm_path_gumbel_sinkhorn = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm_star_plan_datasets_optimization/optimization_stars_3_to_14/queries.pkl",
        "model_path": "/home/tim/query_optimization/training_results/gnn_20251210_185635/model.pt",
        "num_queries": 20,
        "optimization_steps": 1000,
        "use_exhaustive": False,
        "use_dp": True,
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": None,  # Use all available cores
        "optimization_params": {
            "optimization_procedure": "neuralsort_v2",
            "k":1,  # Number of gradient optimization runs - 5
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
            "decoding_method": "greedy",
            "use_gumbel_noise": True
        }
    }

    config = config_wikidata_star
    
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
        #sparql_queries = [q for q in sparql_queries if len(q.triples) >= 8] # TODO just for now to visulaize

        print(f"Retained {len(sparql_queries)}/{original_len} queries")
        
        # Update num_queries in config for accurate logging
        config['num_queries'] = len(sparql_queries)
    

    start_time = time.time()
    
    # Evaluate optimization in parallel
    detailed_results = evaluate_optimization_parallel(
        sparql_queries, 
        config['model_path'],
        num_queries=config['num_queries'],
        optimization_steps=config['optimization_steps'],
        optimization_params=config['optimization_params'],
        optimization_algorithms=config['optimization_algorithms'],
        save_directory=save_directory,
        use_exhaustive=config['use_exhaustive'],
        use_true_costs=config.get('use_true_costs', True),
        num_workers=config.get('num_workers', None),
        dp_limit=config.get('dp_limit', 9),  # Pass dp_limit from config or default to 9
        model_params=config.get('model_params', None)
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
        # Load data from the saved file using the new Pandas-based loader
        stats_df = load_data(save_directory)
        plots_dir = os.path.join(save_directory, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate all plots
        plot_overall_boxplot(stats_df, plots_dir)
        plot_mean_costs_bar(stats_df, plots_dir)
        plot_lineplots_by_size(stats_df, plots_dir)
        plot_boxplot_per_size(stats_df, plots_dir)
        plot_scatter_correlations(stats_df, plots_dir)
        
        print(f"Plots saved to: {plots_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
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
