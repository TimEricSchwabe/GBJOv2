import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

from data import Triple, Join, Query, Entity
from model import CostGNNv2
from process_8_tp_dataset_single_file import SPARQLQuery


def load_sparql_queries(queries_file: str, num_queries: None):
    """
    Load all the SPARQL query objects from the given file.
    
    Args:
        queries_file: Path to the file containing saved SPARQLQuery objects
        
    Returns:
        List of SPARQLQuery objects
    """
    with open(queries_file, 'rb') as f:
        sparql_queries = pickle.load(f)
    
    if num_queries is not None:
        print(f"Loaded {num_queries} SPARQL queries from {queries_file}")
        return sparql_queries[:num_queries]
    print(f"Loaded {len(sparql_queries)} SPARQL queries from {queries_file}")
    return sparql_queries


def optimize_query(query_data, model, device='cpu', optimization_steps=500, verbose=True):
    """
    Run the optimization algorithm for a query.
    
    Args:
        query_data: The torch geometric data for the query
        model: The trained cost model
        device: Device to run the optimization on
        optimization_steps: Number of optimization steps
        verbose: Whether to print progress information
        
    Returns:
        The optimized adjacency matrix
    """
    import torch.optim as optim
    
    # Process the query data
    test_datapoint = query_data.to(device)
    N_NODES = len(test_datapoint.x)
    triples_num = (N_NODES + 1) // 2  # n triples -> 2n-1 total nodes
    
    # Create all possible edges (fully connected graph)
    possible_edges = []
    for src in range(N_NODES):
        for dst in range(N_NODES):
            if src != dst:
                possible_edges.append([src, dst])
    
    edge_index = torch.tensor(possible_edges, dtype=torch.long).t().contiguous().to(device)
    num_edges = edge_index.size(1)
    
    # Initialize edge weights with small random variations
    edge_weights = torch.tensor(0.5 + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)
    
    # Set up optimizer - use LBFGS instead of Adam
    optimizer_opt = optim.LBFGS([edge_weights], lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
    
    # Define penalty coefficients
    lambda_acyclic = 1000.0
    lambda_triple_in = 1000.0
    lambda_triple_out = 1000.0
    lambda_join_in = 500.0
    lambda_join_out = 1000.0
    lambda_entropy = 100.0
    
    # Tracking metrics for plotting
    cost_history = []
    total_penalty_history = []
    acyclic_penalty_history = []
    triple_in_penalty_history = []
    triple_out_penalty_history = []
    join_in_penalty_history = []
    join_out_penalty_history = []
    entropy_penalty_history = []
    
    # Entropy penalty function
    def entropy_penalty(weights, temperature=1.0):
        epsilon = 1e-10
        return -torch.sum(weights * torch.log(weights + epsilon) + 
                        (1 - weights) * torch.log(1 - weights + epsilon)) * temperature
    
    # Optimization loop for LBFGS
    for step in range(optimization_steps):
        # Define closure function for LBFGS
        def closure():
            optimizer_opt.zero_grad()
            
            # Get cost prediction from model
            cost_pred = model(test_datapoint.x, edge_index, edge_weight=edge_weights)
            
            # Convert edge weights to adjacency matrix
            A = torch.zeros((N_NODES, N_NODES), device=device)
            A[edge_index[0], edge_index[1]] = edge_weights
            
            # Calculate in-degree and out-degree for all nodes
            in_degree = torch.sum(A, dim=0)
            out_degree = torch.sum(A, dim=1)
            
            # Split nodes into triple and join nodes
            triple_nodes_indices = torch.arange(triples_num, device=device)
            join_nodes_indices = torch.arange(triples_num, N_NODES, device=device)
            
            # Structural penalties
            P_triple_in = torch.sum(torch.square(in_degree[triple_nodes_indices]))
            P_triple_out = torch.sum(torch.square(out_degree[triple_nodes_indices] - 1.0))
            P_join_in = torch.sum(torch.square(in_degree[join_nodes_indices] - 2.0))
            
            # Root node (last join node) has no outgoing edges, others have one
            root_index = N_NODES - 1
            non_root_join_indices = torch.arange(triples_num, root_index, device=device)
            P_join_out = torch.sum(torch.square(out_degree[non_root_join_indices] - 1.0)) + \
                        torch.square(out_degree[root_index])
            
            # Acyclicity penalty
            trace_exp = torch.trace(torch.matrix_exp(A)) - N_NODES
            P_acyclic = trace_exp
            
            # Use fixed temperature
            temperature = 1.0
            
            # Compute entropy penalties
            P_entropy = torch.tensor(0.0, device=device)
            for i in range(N_NODES):
                weights = A[i, :]
                mask = weights > 0.01
                if torch.any(mask):
                    P_entropy += entropy_penalty(weights[mask], temperature)
            
            # Total penalty
            total_penalty = lambda_acyclic * P_acyclic + \
                            lambda_triple_in * P_triple_in + \
                            lambda_triple_out * P_triple_out + \
                            lambda_join_in * P_join_in + \
                            lambda_join_out * P_join_out + \
                            lambda_entropy * P_entropy
            
            # Total loss
            loss = cost_pred + 0.1 * total_penalty
            
            # Backward pass
            loss.backward()
            
            # Store metrics for this step (outside of history to avoid duplicating)
            closure.cost_pred = cost_pred.item()
            closure.total_penalty = total_penalty.item()
            closure.P_acyclic = P_acyclic.item()
            closure.P_triple_in = P_triple_in.item()
            closure.P_triple_out = P_triple_out.item()
            closure.P_join_in = P_join_in.item()
            closure.P_join_out = P_join_out.item()
            closure.P_entropy = P_entropy.item()
            
            return loss
        
        # Perform optimization step
        optimizer_opt.step(closure)
        
        # Clamp edge weights to [0,1]
        with torch.no_grad():
            edge_weights.clamp_(0, 1)
        
        # Record metrics
        cost_history.append(closure.cost_pred)
        total_penalty_history.append(closure.total_penalty)
        acyclic_penalty_history.append(closure.P_acyclic)
        triple_in_penalty_history.append(closure.P_triple_in)
        triple_out_penalty_history.append(closure.P_triple_out)
        join_in_penalty_history.append(closure.P_join_in)
        join_out_penalty_history.append(closure.P_join_out)
        entropy_penalty_history.append(closure.P_entropy)
        
        if verbose and (step + 1) % 100 == 0:
            print(f'Step {step+1}/{optimization_steps}, Cost: {closure.cost_pred:.2f}, Penalty: {closure.total_penalty:.2f}')
    
    # Convert edge weights to final adjacency matrix
    with torch.no_grad():
        final_adjacency = torch.zeros((N_NODES, N_NODES), device=device)
        final_adjacency[edge_index[0], edge_index[1]] = edge_weights
    
    # Threshold to get binary decisions
    final_adjacency[final_adjacency < 0.5] = 0.0
    final_adjacency[final_adjacency >= 0.5] = 1.0
    
    # Plot metrics if verbose
    if verbose:
        plot_optimization_metrics(
            cost_history, 
            total_penalty_history,
            acyclic_penalty_history,
            triple_in_penalty_history,
            triple_out_penalty_history,
            join_in_penalty_history,
            join_out_penalty_history,
            entropy_penalty_history
        )
    
    return final_adjacency, triples_num


def plot_optimization_metrics(cost_history, total_penalty_history, acyclic_penalty_history, 
                             triple_in_penalty_history, triple_out_penalty_history,
                             join_in_penalty_history, join_out_penalty_history, entropy_penalty_history):
    """
    Plot optimization metrics over iterations.
    
    Args:
        cost_history: List of cost values
        total_penalty_history: List of total penalty values
        acyclic_penalty_history: List of acyclicity penalty values
        triple_in_penalty_history: List of triple in-degree penalty values
        triple_out_penalty_history: List of triple out-degree penalty values
        join_in_penalty_history: List of join in-degree penalty values
        join_out_penalty_history: List of join out-degree penalty values
        entropy_penalty_history: List of entropy penalty values
    """
    iterations = range(1, len(cost_history) + 1)
    
    # Plot cost and total penalty
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, cost_history, 'b-', label='Predicted Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost During Optimization')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(iterations, total_penalty_history, 'r-', label='Total Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Total Penalty During Optimization')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_cost_penalty.png')
    plt.show()
    
    # Plot individual penalties
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(iterations, acyclic_penalty_history, 'g-', label='Acyclicity Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Acyclicity Penalty')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(iterations, entropy_penalty_history, 'm-', label='Entropy Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Entropy Penalty')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 2, 3)
    plt.plot(iterations, triple_in_penalty_history, 'c-', label='Triple In-Degree Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Triple In-Degree Penalty')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot(iterations, triple_out_penalty_history, 'y-', label='Triple Out-Degree Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Triple Out-Degree Penalty')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 2, 5)
    plt.plot(iterations, join_in_penalty_history, 'k-', label='Join In-Degree Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Join In-Degree Penalty')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 2, 6)
    plt.plot(iterations, join_out_penalty_history, color='orange', label='Join Out-Degree Penalty')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.title('Join Out-Degree Penalty')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_individual_penalties.png')
    plt.show()


def adjacency_to_query_with_real_triples(A, triples_num, original_triples):
    """
    Convert an adjacency matrix to a Query object using the original triples.
    
    Args:
        A: The adjacency matrix (torch tensor or numpy array)
        triples_num: Number of triple nodes
        original_triples: List of original Triple objects
        
    Returns:
        A Query object representing the plan
    """
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    
    N_NODES = A.shape[0]
    
    # Ensure we have the right number of triples
    if len(original_triples) != triples_num:
        raise ValueError(f"Number of original triples ({len(original_triples)}) doesn't match triples_num ({triples_num})")
    
    def build_tree(node_idx):
        """Recursively build the query tree from the adjacency matrix"""
        # For triple nodes, return the corresponding original triple
        if node_idx < triples_num:
            return original_triples[node_idx]
        
        # For join nodes, find children and build recursively
        children = np.where(A[:, node_idx] > 0.5)[0]
        
        if len(children) != 2:
            raise ValueError(f"Join node {node_idx} has {len(children)} children, expected 2")
        
        left = build_tree(children[0])
        right = build_tree(children[1])
        
        return Join(left=left, right=right)
    
    # Find the root node (join node with no outgoing edges)
    root_idx = N_NODES - 1  # Default to the last node
    for i in range(triples_num, N_NODES):
        if np.sum(A[i, :]) < 0.1:  # No outgoing edges
            root_idx = i
            break
    
    root = build_tree(root_idx)
    return Query(root=root, triples_num=triples_num)


def greedy_optimize_query(query_data, model, original_triples, device='cpu', verbose=True):
    """
    Use a greedy heuristic to build a query plan using the cost model.
    
    Args:
        query_data: The torch geometric data for the query
        model: The trained cost model
        original_triples: List of original Triple objects
        device: Device to run the optimization on
        verbose: Whether to print progress information
        
    Returns:
        A Query object representing the optimized plan
    """
    model.eval()  # Ensure model is in evaluation mode
    triples_num = len(original_triples)
    
    if verbose:
        print("Starting greedy query optimization")
        print(f"Number of triple patterns: {triples_num}")
    
    # Create a mapping from triple pattern to its features in the original data
    # For 8tp, we need to adapt this approach since the data format might be different
    # We'll use the first 8 nodes of the original data assuming they're the triple patterns
    original_features = query_data.x[:triples_num].clone()
    
    # Initialize remaining triples and current subplan
    remaining_triples = list(range(triples_num))
    current_subplan = None
    
    # Keep track of which triple indices we've used
    used_triple_indices = []
    
    # Repeat until all triples are used
    while remaining_triples:
        best_cost = float('inf')
        best_idx = -1
        best_new_subplan = None
        
        # If we don't have a current subplan, evaluate each triple individually
        if current_subplan is None:
            # Evaluate each triple pattern individually
            for i in remaining_triples:
                # Create a graph with just this triple pattern
                single_node_x = original_features[i:i+1]
                # No edges for a single node
                empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                
                # Predict cost
                with torch.no_grad():
                    cost = model(single_node_x, empty_edge_index).item()
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
                    best_new_subplan = original_triples[i]
            
            # Update current subplan with best triple
            current_subplan = best_new_subplan
        else:
            # Evaluate each remaining triple joined with the current subplan
            for i in remaining_triples:
                # Create a join between current subplan and this triple
                new_subplan = Join(left=current_subplan, right=original_triples[i])
                
                # For cost estimation, we can create a simple query with this join
                temp_query = Query(root=new_subplan, triples_num=len(used_triple_indices) + 1)
                
                # Use model to estimate cost (this is approximate)
                # In a more accurate implementation, we'd create proper graph representations for each join
                
                # Use a simplified approach with existing query data
                # For a proper implementation, we would need to create the correct graph representation
                # with nodes and edges that match this specific join
                
                # For now, we'll estimate based on the number of variables shared between the subplan and the new triple
                # This is a heuristic and not as accurate as using the model directly
                
                # Count shared variables (simplified for this example)
                current_vars = set()
                if isinstance(current_subplan, Triple):
                    for entity in [current_subplan.s, current_subplan.p, current_subplan.o]:
                        if entity.is_variable:
                            current_vars.add(entity.name)
                else:
                    current_vars = current_subplan.variables
                
                triple_vars = set()
                for entity in [original_triples[i].s, original_triples[i].p, original_triples[i].o]:
                    if entity.is_variable:
                        triple_vars.add(entity.name)
                
                shared_vars = len(current_vars.intersection(triple_vars))
                
                # Use the model to predict cost if possible
                # This is a simplified approach; in practice, we would create the proper graph representation
                try:
                    # Create a small graph with the current subplan and this candidate
                    # This is simplified and might not work correctly in all cases
                    estimated_cost = (triples_num - shared_vars) * 100  # Simple heuristic
                    
                    if estimated_cost < best_cost:
                        best_cost = estimated_cost
                        best_idx = i
                        best_new_subplan = new_subplan
                except Exception as e:
                    if verbose:
                        print(f"Error estimating cost for join with triple {i}: {e}")
                    continue
            
            # Update current subplan with best join
            current_subplan = best_new_subplan
        
        # Remove the chosen triple from remaining triples
        remaining_triples.remove(best_idx)
        used_triple_indices.append(best_idx)
        
        if verbose:
            print(f"Selected triple {best_idx} with estimated cost {best_cost:.4f}")
    
    # Current subplan should now be the full plan
    return Query(root=current_subplan, triples_num=triples_num)


def random_join_plan(original_triples, seed=None):
    """
    Create a random join plan using the original triples.
    
    Args:
        original_triples: List of original Triple objects
        seed: Random seed
        
    Returns:
        A Query object representing a random plan
    """
    from data import random_join_order
    
    # Convert triples to format expected by random_join_order
    triple_strs = []
    for triple in original_triples:
        triple_strs.append([str(triple.s), str(triple.p), str(triple.o)])
    
    # Use the existing random_join_order function
    random_plan = random_join_order(triple_strs, seed=seed)
    
    return random_plan


def plot_statistics(stats, show_plots=True, suffix=""):
    """
    Plot statistics about the optimization performance.
    
    Args:
        stats: Dictionary with statistics from evaluate_optimization
        show_plots: Whether to display the plots (if False, only save them)
        suffix: Optional suffix to add to saved filenames (e.g., "_iteration_10")
    """
    # Calculate mean costs for different strategies
    mean_gradient = np.mean(stats['gradient_costs'])
    mean_greedy = np.mean(stats['greedy_costs'])
    mean_random = np.mean(stats['random_costs'])
    
    # Plot mean costs comparison
    plt.figure(figsize=(10, 6))
    
    labels = ['Gradient', 'Greedy', 'Random']
    means = [mean_gradient, mean_greedy, mean_random]
    
    plt.bar(labels, means, color=['blue', 'green', 'orange'])
    plt.ylabel('Mean Cost')
    plt.title('Comparison of Mean Costs')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(means):
        plt.text(i, v * 1.05, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'mean_costs_comparison.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot cost comparison as boxplot (log scale)
    plt.figure(figsize=(10, 6))
    
    data = [
        stats['gradient_costs'],
        stats['greedy_costs'],
        stats['random_costs']
    ]
    
    plt.boxplot(data, labels=labels)
    plt.yscale('log')
    plt.ylabel('Cost (log scale)')
    plt.title('Cost Distribution Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'cost_distribution_comparison.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Calculate and print ratio comparisons
    gradient_to_random_ratio = np.mean(np.array(stats['gradient_costs']) / np.array(stats['random_costs']))
    greedy_to_random_ratio = np.mean(np.array(stats['greedy_costs']) / np.array(stats['random_costs']))
    
    print(f"Mean ratio of gradient optimizer cost to random plan cost: {gradient_to_random_ratio:.2f}x")
    print(f"Mean ratio of greedy heuristic cost to random plan cost: {greedy_to_random_ratio:.2f}x")
    
    # Calculate and print how often each method beats random selection
    gradient_costs = np.array(stats['gradient_costs'])
    greedy_costs = np.array(stats['greedy_costs'])
    random_costs = np.array(stats['random_costs'])
    
    gradient_wins = np.sum(gradient_costs < random_costs)
    greedy_wins = np.sum(greedy_costs < random_costs)
    
    gradient_win_pct = gradient_wins / len(gradient_costs) * 100
    greedy_win_pct = greedy_wins / len(greedy_costs) * 100
    
    print(f"Gradient optimizer beats random selection in {gradient_win_pct:.1f}% of queries")
    print(f"Greedy heuristic beats random selection in {greedy_win_pct:.1f}% of queries")
    
    # Plot win percentage
    plt.figure(figsize=(8, 6))
    win_pcts = [gradient_win_pct, greedy_win_pct]
    plt.bar(['Gradient vs. Random', 'Greedy vs. Random'], win_pcts, color=['blue', 'green'])
    plt.ylabel('Win Percentage (%)')
    plt.title('Percentage of Queries Where Optimizer Beats Random Selection')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for i, v in enumerate(win_pcts):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'win_percentage.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot scatter of gradient vs greedy costs
    plt.figure(figsize=(10, 8))
    plt.scatter(gradient_costs, greedy_costs, alpha=0.7, s=70, c='blue', edgecolors='black')
    
    # Add 45-degree line (y=x)
    max_val = max(np.max(gradient_costs), np.max(greedy_costs))
    min_val = min(np.min(gradient_costs), np.min(greedy_costs))
    # Add some padding to the line
    line_min = min_val * 0.9
    line_max = max_val * 1.1
    plt.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.7)
    
    plt.xlabel('Gradient-Based Optimization Cost')
    plt.ylabel('Greedy Optimization Cost')
    plt.title('Gradient vs Greedy Optimization Cost Comparison')
    plt.grid(alpha=0.3)
    
    # Add annotations for points far from the line
    for i in range(len(gradient_costs)):
        ratio = greedy_costs[i] / gradient_costs[i] if gradient_costs[i] > 0 else 0
        # Annotate points where one method is significantly better
        if ratio > 2 or ratio < 0.5:
            plt.annotate(f"Q{i}", (gradient_costs[i], greedy_costs[i]), 
                         xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(f'gradient_vs_greedy.png')
    if show_plots:
        plt.show()
    else:
        plt.close()


def evaluate_optimization(sparql_queries, model_path, num_queries=None, optimization_steps=500, verbose=False):
    """
    Evaluate the optimization algorithm on the given SPARQL queries.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model_path: Path to the trained cost model
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        verbose: Whether to print and plot detailed progress information
        
    Returns:
        Statistics about the optimization performance
    """
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
        
        # Run gradient-based optimization
        try:
            if verbose:
                print(f"\nRunning gradient-based optimization for query {i}")
            
            final_adjacency, triples_num = optimize_query(
                torch_data, model, device, optimization_steps=optimization_steps, verbose=verbose
            )
            
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Calculate the actual cost using the get_cost method
            gradient_cost = gradient_plan.root.get_cost()
            gradient_costs.append(gradient_cost)
            
            if verbose:
                print(f"Gradient optimization complete. Final cost: {gradient_cost}")
                # Visualize the plan if verbose
                visualization_dir = "plan_visualizations"
                os.makedirs(visualization_dir, exist_ok=True)
                gradient_plan.visualize(output_file=f"{visualization_dir}/gradient_plan_query_{i}")
                print(f"Saved gradient plan visualization to {visualization_dir}/gradient_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error in gradient optimization for query {i}: {e}")
            # Skip this query
            continue
        
        # Run greedy optimization
        try:
            if verbose:
                print(f"\nRunning greedy optimization for query {i}")
                
            greedy_plan = greedy_optimize_query(
                torch_data, model, triple_objs, device, verbose=verbose
            )
            
            # Calculate the actual cost
            greedy_cost = greedy_plan.root.get_cost()
            greedy_costs.append(greedy_cost)
            
            if verbose:
                print(f"Greedy optimization complete. Final cost: {greedy_cost}")
                # Visualize the plan if verbose
                visualization_dir = "plan_visualizations"
                os.makedirs(visualization_dir, exist_ok=True)
                greedy_plan.visualize(output_file=f"{visualization_dir}/greedy_plan_query_{i}")
                print(f"Saved greedy plan visualization to {visualization_dir}/greedy_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error in greedy optimization for query {i}: {e}")
            # Use infinity as a placeholder for failed optimizations
            greedy_costs.append(float('inf'))
        
        # Create a random plan
        try:
            if verbose:
                print(f"\nCreating random plan for query {i}")
                
            random_plan = random_join_plan(triple_objs, seed=i)
            
            # Calculate the actual cost
            random_cost = random_plan.root.get_cost()
            random_costs.append(random_cost)
            
            if verbose:
                print(f"Random plan created. Cost: {random_cost}")
                # Visualize the plan if verbose
                visualization_dir = "plan_visualizations"
                os.makedirs(visualization_dir, exist_ok=True)
                random_plan.visualize(output_file=f"{visualization_dir}/random_plan_query_{i}")
                print(f"Saved random plan visualization to {visualization_dir}/random_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error creating random plan for query {i}: {e}")
            # Use infinity as a placeholder for failed random plans
            random_costs.append(float('inf'))
        
        # Print progress every 5 queries and generate plots
        if (i + 1) % 1 == 0:
            print(f"\nProcessed {i+1}/{len(sparql_queries)} queries")
            if gradient_costs:
                print(f"Average gradient cost: {np.mean(gradient_costs):.2f}")
            if greedy_costs:
                print(f"Average greedy cost: {np.mean(greedy_costs):.2f}")
            if random_costs:
                print(f"Average random cost: {np.mean(random_costs):.2f}")
                
            # Create intermediate statistics and save plots (without displaying)
            intermediate_stats = {
                'gradient_costs': gradient_costs,
                'greedy_costs': greedy_costs,
                'random_costs': random_costs
            }
            # Save plots without showing them, with a suffix indicating the iteration
            plot_statistics(intermediate_stats, show_plots=False, suffix=f"_iter_{i+1}")
            print(f"Saved intermediate plots at iteration {i+1}")
    
    # Calculate statistics
    stats = {
        'gradient_costs': gradient_costs,
        'greedy_costs': greedy_costs,
        'random_costs': random_costs
    }
    
    return stats


if __name__ == "__main__":
    # Set paths
    queries_file = "sparql_queries_8_single/queries.pkl"
    model_path = "/home/tim/query_optimization/8tp_v3.pt"
    num_queries = 50  # Adjust as needed
    
    # Load queries
    sparql_queries = load_sparql_queries(queries_file, num_queries)
    
    # Run with verbose to see plots for the first query
    verbose = False  # Set to True to see detailed progress and plots
    
    # Evaluate optimization
    stats = evaluate_optimization(
        sparql_queries, 
        model_path,
        num_queries=num_queries,  # Set to None to evaluate all queries
        optimization_steps=4000,  # Adjust number of steps as needed
        verbose=verbose  # Pass the verbose flag
    )
    
    # Plot final statistics with display
    plot_statistics(stats, show_plots=True) 