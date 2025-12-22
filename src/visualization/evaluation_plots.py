"""
Plotting functions for optimization evaluation.

Contains functions for visualizing optimization metrics, statistics comparisons,
adjacency matrices, and creating various plots for analysis.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_optimization_metrics(cost_history, total_penalty_history, acyclic_penalty_history, 
                             triple_in_penalty_history, triple_out_penalty_history,
                             join_in_penalty_history, join_out_penalty_history, entropy_penalty_history,
                             save_directory=None, show_plots=True):
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
        save_directory: Directory to save plots to (if None, saves to current directory)
        show_plots: Whether to display plots interactively (default True)
    """
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
    
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
    save_path = os.path.join(save_directory, 'optimization_cost_penalty.png') if save_directory else 'optimization_cost_penalty.png'
    plt.savefig(save_path)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
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
    save_path = os.path.join(save_directory, 'optimization_individual_penalties.png') if save_directory else 'optimization_individual_penalties.png'
    plt.savefig(save_path)
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_statistics(stats, show_plots=True, suffix="", save_directory="."):
    """
    Plot statistics about the optimization performance.
    
    Args:
        stats: Dictionary with statistics from evaluate_optimization
        show_plots: Whether to display the plots (if False, only save them)
        suffix: Optional suffix to add to saved filenames (e.g., "_iteration_10")
        save_directory: Directory to save the plots to
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    # Calculate mean costs for different strategies
    mean_gradient = np.mean(stats['gradient_costs'])
    mean_greedy = np.mean(stats['greedy_costs'])
    mean_random = np.mean(stats['random_costs'])
    
    # NEW – optional categories ------------------------------------------------
    has_predicted = 'predicted_best_costs' in stats and len(stats['predicted_best_costs']) > 0
    has_pred_grad = 'predicted_gradient_costs' in stats and len(stats['predicted_gradient_costs']) > 0
    has_pred_greedy = 'predicted_greedy_costs' in stats and len(stats['predicted_greedy_costs']) > 0
    has_true_best = 'true_best_predicted_costs' in stats and len(stats['true_best_predicted_costs']) > 0
    has_exhaustive = 'predicted_exhaustive_costs' in stats and len(stats['predicted_exhaustive_costs']) > 0
    
    if has_predicted:
        mean_predicted = np.mean(stats['predicted_best_costs'])
    if has_pred_grad:
        mean_pred_grad = np.mean(stats['predicted_gradient_costs'])
    if has_pred_greedy:
        mean_pred_greedy = np.mean(stats['predicted_greedy_costs'])
    if has_true_best:
        mean_true_best = np.mean(stats['true_best_predicted_costs'])
    if has_exhaustive:
        mean_exhaustive = np.mean(stats['predicted_exhaustive_costs'])
    
    # Plot mean costs comparison
    plt.figure(figsize=(12, 6))
    
    labels = ['Gradient', 'Greedy', 'Random']
    means = [mean_gradient, mean_greedy, mean_random]
    
    if has_predicted:
        labels.append('DP-Best')
        means.append(mean_predicted)
    if has_exhaustive:
        labels.append('Exhaustive')
        means.append(mean_exhaustive)
    if has_pred_grad:
        labels.append('GradPred')
        means.append(mean_pred_grad)
    if has_pred_greedy:
        labels.append('GreedyPred')
        means.append(mean_pred_greedy)
    if has_true_best:
        labels.append('TrueBestPred')
        means.append(mean_true_best)
    
    bar_colors_master = ['blue', 'green', 'orange', 'purple', 'cyan', 'red', 'brown', 'pink']
    plt.bar(labels, means, color=bar_colors_master[:len(labels)])
    plt.ylabel('Mean Cost')
    plt.title('Comparison of Mean Costs')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(means):
        plt.text(i, v * 1.05, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'mean_costs_comparison{suffix}.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot cost comparison as boxplot (log scale)
    plt.figure(figsize=(12, 6))
    
    data = [
        stats['gradient_costs'],
        stats['greedy_costs'],
        stats['random_costs']
    ]
    
    if has_predicted:
        data.append(stats['predicted_best_costs'])
    if has_exhaustive:
        data.append(stats['predicted_exhaustive_costs'])
    if has_pred_grad:
        data.append(stats['predicted_gradient_costs'])
    if has_pred_greedy:
        data.append(stats['predicted_greedy_costs'])
    if has_true_best:
        data.append(stats['true_best_predicted_costs'])
    
    labels_box = ['Gradient', 'Greedy', 'Random']
    if has_predicted:
        labels_box.append('DP-Best')
    if has_exhaustive:
        labels_box.append('Exhaustive')
    if has_pred_grad:
        labels_box.append('GradPred')
    if has_pred_greedy:
        labels_box.append('GreedyPred')
    if has_true_best:
        labels_box.append('TrueBestPred')
    
    plt.boxplot(data, labels=labels_box)
    plt.yscale('log')
    plt.ylabel('Cost (log scale)')
    plt.title('Cost Distribution Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'cost_distribution_comparison{suffix}.png'))
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
    plt.savefig(os.path.join(save_directory, f'win_percentage{suffix}.png'))
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
    
    # Set both axes to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    

    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'gradient_vs_greedy{suffix}.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot scatter of gradient vs random costs (new)
    plt.figure(figsize=(10, 8))
    plt.scatter(gradient_costs, random_costs, alpha=0.7, s=70, c='orange', edgecolors='black')
    
    # Add 45-degree line (y=x)
    max_val = max(np.max(gradient_costs), np.max(random_costs))
    min_val = min(np.min(gradient_costs), np.min(random_costs))
    # Add some padding to the line
    line_min = min_val * 0.9
    line_max = max_val * 1.1
    plt.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.7)
    
    plt.xlabel('Gradient-Based Optimization Cost')
    plt.ylabel('Random Plan Cost')
    plt.title('Gradient vs Random Plan Cost Comparison')
    plt.grid(alpha=0.3)
    
    # Set both axes to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'gradient_vs_random{suffix}.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()


    # Plot scatter of greedy vs random costs (new)
    plt.figure(figsize=(10, 8))
    plt.scatter(greedy_costs, random_costs, alpha=0.7, s=70, c='orange', edgecolors='black')
    
    # Add 45-degree line (y=x)
    max_val = max(np.max(greedy_costs), np.max(random_costs))
    min_val = min(np.min(greedy_costs), np.min(random_costs))
    # Add some padding to the line
    line_min = min_val * 0.9
    line_max = max_val * 1.1
    plt.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.7)
    
    plt.xlabel('Greedy Optimization Cost')
    plt.ylabel('Random Plan Cost')
    plt.title('Greedy vs Random Plan Cost Comparison')
    plt.grid(alpha=0.3)
    
    # Set both axes to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f'greedy_vs_random{suffix}.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()


    # ------------------------------------------------------------------
    # NEW scatter plots: Gradient & Greedy vs best predicted cost(s)
    # ------------------------------------------------------------------
    if has_predicted:
        # Gradient vs best predicted (predicted cost)
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['gradient_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='purple', edgecolors='black')
        min_val = min(min(stats['gradient_costs']), min(stats['predicted_best_costs'])) * 0.9
        max_val = max(max(stats['gradient_costs']), max(stats['predicted_best_costs'])) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Gradient-Based Cost')
        plt.ylabel('Best Predicted Cost')
        plt.title('Gradient vs Best-Predicted Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'gradient_vs_best_predicted{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Greedy vs best predicted (predicted cost)
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['greedy_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='purple', edgecolors='black')
        min_val = min(min(stats['greedy_costs']), min(stats['predicted_best_costs'])) * 0.9
        max_val = max(max(stats['greedy_costs']), max(stats['predicted_best_costs'])) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Greedy Cost')
        plt.ylabel('Best Predicted Cost')
        plt.title('Greedy vs Best-Predicted Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'greedy_vs_best_predicted{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_true_best:
        # Gradient vs true cost of best-predicted plan
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['gradient_costs'], stats['true_best_predicted_costs'], alpha=0.7, s=70, c='red', edgecolors='black')
        min_val = min(min(stats['gradient_costs']), min(stats['true_best_predicted_costs'])) * 0.9
        max_val = max(max(stats['gradient_costs']), max(stats['true_best_predicted_costs'])) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Gradient-Based Cost')
        plt.ylabel('True Cost of Best-Predicted Plan')
        plt.title('Gradient vs True Best-Predicted Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'gradient_vs_true_best_predicted{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Greedy vs true best predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['greedy_costs'], stats['true_best_predicted_costs'], alpha=0.7, s=70, c='red', edgecolors='black')
        min_val = min(min(stats['greedy_costs']), min(stats['true_best_predicted_costs'])) * 0.9
        max_val = max(max(stats['greedy_costs']), max(stats['true_best_predicted_costs'])) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Greedy Cost')
        plt.ylabel('True Cost of Best-Predicted Plan')
        plt.title('Greedy vs True Best-Predicted Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'greedy_vs_true_best_predicted{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()


    if has_pred_grad and has_pred_greedy:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_gradient_costs'], stats['predicted_greedy_costs'], alpha=0.7, s=70, c='brown', edgecolors='black')
        mn = min(min(stats['predicted_gradient_costs']), min(stats['predicted_greedy_costs'])) * 0.9
        mx = max(max(stats['predicted_gradient_costs']), max(stats['predicted_greedy_costs'])) * 1.1
        plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Predicted Gradient Cost')
        plt.ylabel('Predicted Greedy Cost')
        plt.title('Predicted Gradient vs Predicted Greedy Cost')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'pred_gradient_vs_pred_greedy{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_pred_grad and has_predicted:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_gradient_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='darkgreen', edgecolors='black')
        mn = min(min(stats['predicted_gradient_costs']), min(stats['predicted_best_costs'])) * 0.9
        mx = max(max(stats['predicted_gradient_costs']), max(stats['predicted_best_costs'])) * 1.1
        plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Predicted Gradient Cost')
        plt.ylabel('Exhaustive Best Predicted Cost')
        plt.title('Predicted Gradient vs Exhaustive Best Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'pred_gradient_vs_exhaustive_pred{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    if has_pred_greedy and has_predicted:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_greedy_costs'], stats['predicted_best_costs'], alpha=0.7, s=70, c='darkorange', edgecolors='black')
        mn = min(min(stats['predicted_greedy_costs']), min(stats['predicted_best_costs'])) * 0.9
        mx = max(max(stats['predicted_greedy_costs']), max(stats['predicted_best_costs'])) * 1.1
        plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Predicted Greedy Cost')
        plt.ylabel('Exhaustive Best Predicted Cost')
        plt.title('Predicted Greedy vs Exhaustive Best Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'pred_greedy_vs_exhaustive_pred{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()

    # NEW: Scatter plot comparing DP vs Exhaustive search results
    if has_predicted and has_exhaustive:
        plt.figure(figsize=(10, 8))
        plt.scatter(stats['predicted_best_costs'], stats['predicted_exhaustive_costs'], alpha=0.7, s=70, c='purple', edgecolors='black')
        mn = min(min(stats['predicted_best_costs']), min(stats['predicted_exhaustive_costs'])) * 0.9
        mx = max(max(stats['predicted_best_costs']), max(stats['predicted_exhaustive_costs'])) * 1.1
        plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('DP Best Predicted Cost')
        plt.ylabel('Exhaustive Best Predicted Cost')
        plt.title('Dynamic Programming vs Exhaustive Search Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'dp_vs_exhaustive{suffix}.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()


def visualize_adjacency_matrix(adjacency_matrix, triples_num, visualization_dir, query_idx, use_tree_layout=False):
    """
    Visualize the adjacency matrix as a directed graph using NetworkX.
    
    Args:
        adjacency_matrix: PyTorch tensor or numpy array of shape (N_NODES, N_NODES)
        triples_num: The number of triple nodes
        visualization_dir: Directory to save the visualization
        query_idx: Index of the current query (for filename)
        use_tree_layout: If True, use a tree layout; otherwise use force-directed
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Convert to numpy if it's a PyTorch tensor
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.cpu().detach().numpy()
    
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph()
    
    # Add nodes
    n_nodes = adjacency_matrix.shape[0]
    
    # Add all nodes with colors
    for i in range(n_nodes):
        # Add node with appropriate color - blue for triple nodes, red for join nodes
        if i < triples_num:
            G.add_node(i, color='blue', node_type='triple')
        else:
            G.add_node(i, color='red', node_type='join')
    
    # Add all edges with their weights
    for i in range(n_nodes):
        for j in range(n_nodes):
            weight = adjacency_matrix[i, j]
            G.add_edge(i, j, weight=weight)
    
    # Get node colors
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Choose layout based on the parameter
    if use_tree_layout:
        # Find the root node - the join node with no outgoing edges
        root = n_nodes - 1  # Default fallback
        
        # Only look for root when use_tree_layout is true
        for node_idx in range(triples_num, n_nodes):
            # Check if this node has no outgoing edges
            if np.sum(adjacency_matrix[node_idx, :]) < 0.01:
                root = node_idx
                print(f"Found root node at index {root} (Join {root-triples_num})")
                break
        
        # Use a tree layout
        try:
            # Remove edges with weights <= 0.3 to make tree structure clearer
            G_tree = nx.DiGraph()
            for node in G.nodes():
                G_tree.add_node(node, **G.nodes[node])
            
            for u, v, d in G.edges(data=True):
                if d['weight'] > 0.3:  # Only keep strong connections
                    G_tree.add_edge(u, v, **d)
            
            # Use a hierarchical layout with the identified root at the top
            pos = nx.drawing.nx_agraph.graphviz_layout(G_tree, prog='dot', root=root)
            print(f"Using hierarchical layout with root = {root}")
        except Exception as e:
            print(f"Graphviz layout failed: {e}, falling back to basic tree layout")
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot', root=root)
                print("Using PyDot 'dot' layout")
            except Exception as e:
                print(f"PyDot layout failed: {e}, falling back to spring layout")
                pos = nx.spring_layout(G, seed=42)
    else:
        # Use the original force-directed layout
        pos = nx.spring_layout(G, seed=42)
        print("Using spring layout")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    
    # Draw edges with width proportional to weight
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    edge_colors = [plt.cm.coolwarm(weight/5) for weight in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_weights, arrowsize=20, 
                          edge_color=edge_colors, alpha=0.8)
    
    # Add labels
    labels = {}
    for i in range(n_nodes):
        if i < triples_num:
            labels[i] = f"T{i}"
        else:
            labels[i] = f"J{i-triples_num}"
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white')
    
    plt.title("Query Plan Visualization", fontsize=16)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01, aspect=40)
    cbar.set_label('Edge Weight', fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save with layout type in filename
    layout_type = "tree" if use_tree_layout else "force"
    plt.savefig(f"{visualization_dir}/adjacency_matrix_query_{query_idx}_{layout_type}.png")
    plt.close()
    
    return G