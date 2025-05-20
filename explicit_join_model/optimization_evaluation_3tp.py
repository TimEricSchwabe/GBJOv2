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
from model import CostGNN
from process_3tp_dataset import SPARQLQuery


def load_sparql_queries(queries_dir: str, num_queries: int):
    """
    Load all the SPARQL query objects from the given directory.
    
    Args:
        queries_dir: Directory containing the saved SPARQLQuery objects
        
    Returns:
        List of SPARQLQuery objects
    """
    sparql_queries = []
    # Only load 2 queries for now
    for filename in list(os.listdir(queries_dir))[:num_queries]:
        if filename.startswith('query_') and filename.endswith('.pkl'):
            filepath = os.path.join(queries_dir, filename)
            with open(filepath, 'rb') as f:
                query = pickle.load(f)
                sparql_queries.append(query)
    
    print(f"Loaded {len(sparql_queries)} SPARQL queries from {queries_dir}")
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
    #edge_weights = torch.full((num_edges,), 0.5, requires_grad=True, device='cpu')

    
    # Set up optimizer
    optimizer_opt = optim.Adam([edge_weights], lr=0.01)
    
    # Define penalty coefficients
    lambda_acyclic = 1000.0
    lambda_triple_in = 1000.0
    lambda_triple_out = 1000.0
    lambda_join_in = 500.0
    lambda_join_out = 1000.0
    lambda_entropy = 100.0
    
    # Optimization loop
    for step in range(optimization_steps):
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
        
        # Entropy penalty for binary decisions
        def entropy_penalty(weights, temperature=1.0):
            epsilon = 1e-10
            return -torch.sum(weights * torch.log(weights + epsilon) + 
                            (1 - weights) * torch.log(1 - weights + epsilon)) * temperature
        
        # Use annealing temperature
        #temperature = max(0.5, 5.0 * (1.0 - step / optimization_steps))
        temperature = 1 #todo
        
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

        # New approach: Gradually increase the weight of structural penalties
        # This gives the optimizer time to find a good solution before being constrained
                #For the first 500 steps, only optimize for cost prediction
        #After 500 steps, add penalties to guide the structure
        #if step < 1000:
        #    loss = cost_pred
        #else:
        #    loss = cost_pred + 0.01 * total_penalty
        loss = cost_pred + 0.01 * total_penalty
        #penalty_weight = min(1.0, step / (0.7 * optimization_steps)) * 0.01
        #loss = cost_pred + penalty_weight * total_penalty
        
        # Backward pass and optimization step
        loss.backward()
        optimizer_opt.step()
        
        # Clamp edge weights to [0,1]
        with torch.no_grad():
            edge_weights.clamp_(0, 1)
            
        if verbose and (step + 1) % 100 == 0:
            print(f'Step {step+1}/{optimization_steps}, Cost: {cost_pred.item():.2f}, Penalty: {total_penalty.item():.2f}')
            #print(f'Penalties - Triple In: {P_triple_in.item():.4f}, Triple Out: {P_triple_out.item():.4f}')
            #print(f'Penalties - Join In: {P_join_in.item():.4f}, Join Out: {P_join_out.item():.4f}')
            #print(f'Acyclicity Penalty: {P_acyclic.item():.4f}')
            #print(f'Entropy Penalty: {P_entropy.item():.4f}')

        # Early stopping if the entropy penalty is small enough
        # This indicates we have converged to mostly binary decisions
        #if total_penalty.item() < 0.1:
        #    if verbose:
        #        print(f"Early stopping at step {step+1}: Entropy penalty {P_entropy.item():.6f} < 0.0001")
        #    break

    
    # Final adjacency matrix
    final_adjacency = A.detach().clone()
    
    # Threshold to get binary decisions
    final_adjacency[final_adjacency < 0.5] = 0.0
    final_adjacency[final_adjacency >= 0.5] = 1.0
    
    return final_adjacency, triples_num


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


def compare_query_plans(plan1, plan2):
    """
    Compare two query plans to see if they are equivalent.
    The comparison is based on the structure of the join tree.
    
    Args:
        plan1: First Query object
        plan2: Second Query object
        
    Returns:
        True if the plans are equivalent, False otherwise
    """
    def get_plan_structure(root):
        """Extract the structure of a plan as a nested tuple encoding the tree structure"""
        if isinstance(root, Triple):
            # For triples, return a tuple with type marker and string representation
            return ('T', str(root))
        else:
            # For joins, recursively get the structure of left and right subtrees
            left_structure = get_plan_structure(root.left)
            right_structure = get_plan_structure(root.right)
            # Sort the subtrees for commutativity of joins
            # Since both are now tuples with the same structure, they can be compared
            return ('J', tuple(sorted([left_structure, right_structure])))
    
    structure1 = get_plan_structure(plan1.root)
    structure2 = get_plan_structure(plan2.root)
    
    return structure1 == structure2


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
    from data import Triple, Join, Query, Entity
    import torch
    import numpy as np
    
    model.eval()  # Ensure model is in evaluation mode
    triples_num = len(original_triples)
    
    if verbose:
        print("Starting greedy query optimization")
        print(f"Number of triple patterns: {triples_num}")
    
    # Create a mapping from triple pattern to its features in the original data
    original_features = query_data.x[:triples_num].clone()
    
    # STEP 1: Evaluate each triple pattern individually to find the one with lowest cost
    triple_costs = []
    for i in range(triples_num):
        # Create a graph with just this triple pattern
        single_node_x = original_features[i:i+1]
        # No edges for a single node
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Predict cost
        with torch.no_grad():
            cost = model(single_node_x, empty_edge_index).item()
        
        triple_costs.append(cost)
        if verbose:
            print(f"Triple {i} cost: {cost:.4f}")
    
    # Find the triple with the lowest cost
    best_triple_idx = np.argmin(triple_costs)
    remaining_triples = list(range(triples_num))
    remaining_triples.remove(best_triple_idx)
    
    if verbose:
        print(f"Best triple: {best_triple_idx} with cost {triple_costs[best_triple_idx]:.4f}")
    
    # Current plan is just the best triple
    current_plan = original_triples[best_triple_idx]
    
    # STEP 2: Evaluate joining with each remaining triple
    join_costs = []
    for i in remaining_triples:
        # Create a small graph with the current best triple and this candidate
        join_x = torch.cat([
            original_features[best_triple_idx:best_triple_idx+1],
            original_features[i:i+1],
            torch.zeros(1, original_features.size(1), device=device)  # Join node features (zeros with 1 at the end)
        ], dim=0)
        join_x[2, -1] = 1.0  # Set join node marker
        
        # Create edges: both triples point to the join node
        join_edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long, device=device)
        
        # Predict cost
        with torch.no_grad():
            cost = model(join_x, join_edge_index).item()
        
        join_costs.append((i, cost))
        if verbose:
            print(f"Join with triple {i} cost: {cost:.4f}")
    
    # Find the best join
    best_join_idx, best_join_cost = min(join_costs, key=lambda x: x[1])
    remaining_triples.remove(best_join_idx)
    
    if verbose:
        print(f"Best join: with triple {best_join_idx} with cost {best_join_cost:.4f}")
    
    # Update current plan
    first_join = Join(
        left=current_plan,
        right=original_triples[best_join_idx]
    )
    
    # STEP 3: Join with the final triple
    last_triple_idx = remaining_triples[0]
    final_plan = Join(
        left=first_join,
        right=original_triples[last_triple_idx]
    )
    
    # Wrap in Query object
    result_query = Query(root=final_plan, triples_num=triples_num)
    
    if verbose:
        print(f"Final plan: ({current_plan} JOIN {original_triples[best_join_idx]}) JOIN {original_triples[last_triple_idx]}")
    
    return result_query


def evaluate_optimization(sparql_queries, model_path, num_queries=None, optimization_steps=500):
    """
    Evaluate the optimization algorithm on the given SPARQL queries.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model_path: Path to the trained cost model
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        
    Returns:
        Statistics about the optimization performance
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    node_feature_dim = 307
    hidden_dim = 256
    model = CostGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Limit number of queries if specified
    if num_queries is not None:
        sparql_queries = sparql_queries[:num_queries]
    
    # Initialize statistics
    found_best_plan = []
    found_middle_plan = []
    found_worst_plan = []
    optimizer_costs = []
    greedy_costs = []
    best_costs = []
    random_costs = []
    middle_costs = []
    worst_costs = []
    
    # Process each query
    for i, query in enumerate(tqdm(sparql_queries, desc="Evaluating queries")):
        # Get costs of the three plans
        costs = query.costs
        if len(costs) != 3:
            print(f"Warning: Query {i} has {len(costs)} plans, expected 3. Skipping.")
            continue
            
        # Sort plans by cost
        sorted_indices = np.argsort(costs)
        best_idx = sorted_indices[0]
        middle_idx = sorted_indices[1]
        worst_idx = sorted_indices[2]
        

        
        # Get the best torch data
        torch_data = query.torch_data[best_idx]
        
        # Prepare the triple objects
        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]
        
        # Run gradient-based optimization
        final_adjacency, triples_num = optimize_query(
            torch_data, model, device, optimization_steps=optimization_steps
        )
        
        # Convert adjacency to query plan
        try:
            optimized_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
        except Exception as e:
            print(f"Error converting adjacency to query plan: {e}")
            continue
        
        # Run greedy optimization
        try:
            greedy_plan = greedy_optimize_query(
                torch_data, model, triple_objs, device, verbose=False
            )
        except Exception as e:
            print(f"Error in greedy optimization: {e}")
            continue
        
        # Check which plan was found by gradient-based optimization
        found_best = compare_query_plans(optimized_plan, query.join_plans[best_idx])
        found_middle = compare_query_plans(optimized_plan, query.join_plans[middle_idx])
        found_worst = compare_query_plans(optimized_plan, query.join_plans[worst_idx])
        
        # Check which plan was found by greedy approach
        greedy_best = compare_query_plans(greedy_plan, query.join_plans[best_idx])
        greedy_middle = compare_query_plans(greedy_plan, query.join_plans[middle_idx])
        greedy_worst = compare_query_plans(greedy_plan, query.join_plans[worst_idx])
        
        # Record results for gradient-based optimization
        found_best_plan.append(found_best)
        found_middle_plan.append(found_middle)
        found_worst_plan.append(found_worst)

        # Record costs
        best_costs.append(costs[best_idx])
        middle_costs.append(costs[middle_idx])
        worst_costs.append(costs[worst_idx])
        random_costs.append(costs[random.randint(0, 2)])
        
        # Get the optimizer cost from the saved costs, not by recalculating
        if found_best:
            optimizer_costs.append(costs[best_idx])
        elif found_middle:
            optimizer_costs.append(costs[middle_idx])
        elif found_worst:
            optimizer_costs.append(costs[worst_idx])
        else:
            # This should rarely happen - if the optimizer found a plan that doesn't match any of the three
            print(f"Warning: Query {i} - optimizer found a plan that doesn't match any of the three predefined plans")
            optimizer_costs.append(float('inf'))
        
        # Get the greedy optimizer cost
        if greedy_best:
            greedy_costs.append(costs[best_idx])
        elif greedy_middle:
            greedy_costs.append(costs[middle_idx])
        elif greedy_worst:
            greedy_costs.append(costs[worst_idx])
        else:
            # This should rarely happen - if the greedy algorithm found a plan that doesn't match any of the three
            print(f"Warning: Query {i} - greedy algorithm found a plan that doesn't match any of the three predefined plans")
            greedy_costs.append(float('inf'))
        
        # Print progress every 10 queries
        if (i + 1) % 10 == 0:
            print(f"\nProcessed {i+1}/{len(sparql_queries)} queries")
            print(f"Found best plan: {sum(found_best_plan)}/{len(found_best_plan)} ({sum(found_best_plan)/len(found_best_plan)*100:.1f}%)")
            print(f"Found middle plan: {sum(found_middle_plan)}/{len(found_middle_plan)} ({sum(found_middle_plan)/len(found_middle_plan)*100:.1f}%)")
            print(f"Found worst plan: {sum(found_worst_plan)}/{len(found_worst_plan)} ({sum(found_worst_plan)/len(found_worst_plan)*100:.1f}%)")
            

    
    # Calculate statistics
    stats = {
        'found_best_plan': found_best_plan,
        'found_middle_plan': found_middle_plan,
        'found_worst_plan': found_worst_plan,
        'optimizer_costs': optimizer_costs,
        'greedy_costs': greedy_costs,
        'best_costs': best_costs,
        'random_costs': random_costs,
        'middle_costs': middle_costs,
        'worst_costs': worst_costs
    }
    
    return stats


def plot_statistics(stats):
    """
    Plot statistics about the optimization performance.
    
    Args:
        stats: Dictionary with statistics from evaluate_optimization
    """
    # Calculate percentages
    total_queries = len(stats['found_best_plan'])
    pct_best = sum(stats['found_best_plan']) / total_queries * 100
    pct_middle = sum(stats['found_middle_plan']) / total_queries * 100
    pct_worst = sum(stats['found_worst_plan']) / total_queries * 100
    
    # Plot plan selection distribution
    labels = ['Best Plan', 'Middle Plan', 'Worst Plan']
    values = [pct_best, pct_middle, pct_worst]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['green', 'orange', 'red'])
    plt.ylabel('Percentage of Queries (%)')
    plt.title('Distribution of Plans Found by Gradient Optimizer')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate percentages for greedy heuristic
    greedy_best = 0
    greedy_middle = 0
    greedy_worst = 0
    
    for i in range(total_queries):
        greedy_cost = stats['greedy_costs'][i]
        best_cost = stats['best_costs'][i]
        middle_cost = stats['middle_costs'][i]
        worst_cost = stats['worst_costs'][i]
        
        if abs(greedy_cost - best_cost) < 0.001:
            greedy_best += 1
        elif abs(greedy_cost - middle_cost) < 0.001:
            greedy_middle += 1
        elif abs(greedy_cost - worst_cost) < 0.001:
            greedy_worst += 1
    
    pct_greedy_best = greedy_best / total_queries * 100
    pct_greedy_middle = greedy_middle / total_queries * 100
    pct_greedy_worst = greedy_worst / total_queries * 100
    
    # Plot greedy plan selection distribution
    plt.figure(figsize=(10, 6))
    values = [pct_greedy_best, pct_greedy_middle, pct_greedy_worst]
    plt.bar(labels, values, color=['green', 'orange', 'red'])
    plt.ylabel('Percentage of Queries (%)')
    plt.title('Distribution of Plans Found by Greedy Heuristic')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Plot cost comparison (log scale)
    plt.figure(figsize=(12, 8))
    
    # Compute mean costs for different strategies
    mean_optimizer = np.mean(stats['optimizer_costs'])
    mean_greedy = np.mean(stats['greedy_costs'])
    mean_best = np.mean(stats['best_costs'])
    mean_random = np.mean(stats['random_costs'])
    mean_middle = np.mean(stats['middle_costs'])
    mean_worst = np.mean(stats['worst_costs'])
    
    labels = ['Gradient', 'Greedy', 'Best', 'Random', 'Middle', 'Worst']
    means = [mean_optimizer, mean_greedy, mean_best, mean_random, mean_middle, mean_worst]
    
    plt.bar(labels, means, color=['blue', 'purple', 'green', 'yellow', 'orange', 'red'])
    plt.ylabel('Mean Cost')
    plt.title('Comparison of Mean Costs')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(means):
        plt.text(i, v * 1.05, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Plot cost comparison as boxplot (log scale)
    plt.figure(figsize=(12, 8))
    
    data = [
        stats['optimizer_costs'],
        stats['greedy_costs'],
        stats['best_costs'],
        stats['random_costs'],
        stats['middle_costs'],
        stats['worst_costs']
    ]
    
    plt.boxplot(data, labels=labels)
    plt.yscale('log')
    plt.ylabel('Cost (log scale)')
    plt.title('Cost Distribution Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print ratio of optimizer cost to best cost
    optimizer_to_best_ratio = np.mean(np.array(stats['optimizer_costs']) / np.array(stats['best_costs']))
    greedy_to_best_ratio = np.mean(np.array(stats['greedy_costs']) / np.array(stats['best_costs']))
    print(f"Mean ratio of gradient optimizer cost to best cost: {optimizer_to_best_ratio:.2f}x")
    print(f"Mean ratio of greedy heuristic cost to best cost: {greedy_to_best_ratio:.2f}x")
    
    # Calculate and print how often each method beats random selection
    optimizer_costs = np.array(stats['optimizer_costs'])
    greedy_costs = np.array(stats['greedy_costs'])
    random_costs = np.array(stats['random_costs'])
    
    optimizer_wins = np.sum(optimizer_costs < random_costs)
    greedy_wins = np.sum(greedy_costs < random_costs)
    
    optimizer_win_pct = optimizer_wins / len(optimizer_costs) * 100
    greedy_win_pct = greedy_wins / len(greedy_costs) * 100
    
    print(f"Gradient optimizer beats random selection in {optimizer_win_pct:.1f}% of queries")
    print(f"Greedy heuristic beats random selection in {greedy_win_pct:.1f}% of queries")


if __name__ == "__main__":
    # Set paths
    queries_dir = "sparql_queries_3"
    model_path = "/home/tim/query_optimization/best_model.pt"
    num_queries = 50
    
    # Load queries
    sparql_queries = load_sparql_queries(queries_dir, num_queries)
    
    # Evaluate optimization
    stats = evaluate_optimization(
        sparql_queries, 
        model_path,
        num_queries=num_queries,  # Set to None to evaluate all queries
        optimization_steps=2000  # Fewer steps for faster evaluation
    )
    
    # Plot statistics
    plot_statistics(stats) 