from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from torch.nn.utils import clip_grad_norm_
from data import random_join_order, join_order_to_adjacency_matrix
from torch_geometric.data import DataLoader
from data_loader import QueryDataset, load_dataset_metadata
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from model import CostGNN
import torch.optim as optim
import pickle
import networkx as nx


def visualize_adjacency_matrix(adjacency_matrix, triples_num, use_tree_layout=False):
    """
    Visualize the adjacency matrix as a directed graph using NetworkX.
    
    Args:
        adjacency_matrix: PyTorch tensor or numpy array of shape (N_NODES, N_NODES)
        triples_num: The number of triple nodes
        use_tree_layout: If True, use a tree layout for visualization; otherwise use the default layout
    
    Returns:
        None (displays the plot)
    """
    # Convert to numpy if it's a PyTorch tensor
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.cpu().detach().numpy()
    
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph()
    
    # Add nodes
    n_nodes = adjacency_matrix.shape[0]
    
    # Add all nodes
    for i in range(n_nodes):
        # Add node with appropriate color - blue for triple nodes, red for join nodes
        if i < triples_num:
            G.add_node(i, color='blue', node_type='triple')
        else:
            G.add_node(i, color='red', node_type='join')
    
    # Add all edges with their weights
    for i in range(n_nodes):
        for j in range(n_nodes):
            #if i != j:  # Avoid self-loops
            if True:
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
        pos = nx.spring_layout(G, seed=42)  # or nx.forceatlas2_layout(G)
        print("Using spring layout")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    
    # Draw edges with width proportional to weight
    # Get edge weights for width calculation
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]  # Scale weights for better visibility
    
    # Create a color map based on edge weights (blue->red gradient)
    edge_colors = [plt.cm.coolwarm(weight/5) for weight in edge_weights]
    
    # Draw edges with arrows and width based on weights
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
    
    # Add title
    plt.title("Query Plan Visualization", fontsize=16)
    
    # Add a colorbar legend for edge weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01, aspect=40)
    cbar.set_label('Edge Weight', fontsize=12)
    
    # Remove axes
    plt.axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return G


dataset_dir = "dataset"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
dataset = QueryDataset(root=dataset_dir)


test_datapoint = dataset[0].to("cpu")

N_NODES = len(test_datapoint.x)

triples_num = (N_NODES + 1) // 2  # n triples -> 2n-1 total nodes



# Calculate the number of triple nodes

possible_edges = []
for src in range(N_NODES):
    for dst in range(N_NODES):
        if src != dst:
            possible_edges.append([src, dst])

        
edge_index = torch.tensor(possible_edges, dtype=torch.long).t().contiguous()
num_edges = edge_index.size(1)

# Initialize edge weights as learnable parameters
#edge_weights = torch.full((num_edges,), 0.5, requires_grad=True, device='cpu')
edge_weights = torch.tensor(0.5 + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device='cpu')
# ╰─▶ value for each of edge in possible_edges

# We optimize the adjacency weights using adam optimizer
optimizer_opt = optim.Adam([edge_weights], lr=0.1)
OPTIMIZATION_STEPS = 4000

# Define penalty coefficients
lambda_acyclic = 1000.0  # For acyclicity penalty
lambda_triple_in = 1000.0  # Triple nodes should have no incoming edges
lambda_triple_out = 1000.0  # Triple nodes should have exactly one outgoing edge
lambda_join_in = 500.0  # Join nodes should have exactly two incoming edges
lambda_join_out = 1000.0  # Join nodes should have exactly one outgoing edge (except root)
lambda_entropy = 100.0  # Weight for entropy penalties
lambda_l1 = 100.0        # Weight for L1 penalties


# When True, adds entropy penalty that encourages weights to be either 0 or 1, not intermediate values
# This helps get discrete decisions instead of distributed weights across many edges
USE_ENTROPY_PENALTY = True

# When True, adds L1 sparsity penalty that encourages keeping only the strongest connections
# For triple nodes: keeps the strongest outgoing edge, pushes others to zero
# For join nodes: keeps the two strongest incoming edges, pushes others to zero
USE_L1_PENALTY = False

# We are storing the adjacencies over time so that we can later plot them in a video
weight_history = []
weight_history.append(edge_weights.detach().clone().numpy())

# Create lists to store individual penalties during optimization
triple_in_penalties = []
triple_out_penalties = []
join_in_penalties = []
join_out_penalties = []
acyclic_penalties = []
entropy_penalties = []  # Track entropy penalties
l1_penalties = []       # Track L1 penalties

# Lets now define the optimal cost and optimal plan of this query (this comes from our example dataset)
OPTIMAL_COST = test_datapoint.y
OPTIMAL_PLAN = test_datapoint.x

costs_during_optimization = []
total_costs_during_optimization = []
plan_distances = []

costs_during_optimization = []
total_costs_during_optimization = []
plan_distances = []


node_feature_dim = 307  # Based on the data format
hidden_dim = 64
model = CostGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load("best_model.pt"))



#### Optimization loop ###
for step in tqdm(range(OPTIMIZATION_STEPS)):
    
    optimizer_opt.zero_grad()
    cost_pred = model(test_datapoint.x, edge_index, edge_weight=edge_weights)
    loss = cost_pred  # We aim to minimize the predicted cost

    # turning edge_weights into Adjacency matrix format
    A = torch.zeros((N_NODES, N_NODES))
    A[edge_index[0], edge_index[1]] = edge_weights

    # Calculate in-degree and out-degree for all nodes
    in_degree = torch.sum(A, dim=0)
    out_degree = torch.sum(A, dim=1)
    
    # Separate nodes into triple nodes and join nodes
    triple_nodes_indices = torch.arange(triples_num)
    join_nodes_indices = torch.arange(triples_num, N_NODES)
    
    # 1. Triple nodes have no incoming edges
    P_triple_in = torch.sum(torch.square(in_degree[triple_nodes_indices]))
    
    # 2. Triple nodes have exactly one outgoing edge
    P_triple_out = torch.sum(torch.square(out_degree[triple_nodes_indices] - 1.0))
    
    # Add entropy penalty to encourage weights to be either 0 or 1, not in between
    # This effectively penalizes intermediate values in [0,1] and pushes weights toward binary decisions
    def entropy_penalty(weights, temperature=1.0):
        """
        Entropy penalty with temperature parameter:
        - Higher temperature = softer penalty
        - Lower temperature = sharper penalty that pushes more aggressively to 0/1
        """
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-10
        # Calculate -w*log(w) - (1-w)*log(1-w) which is minimized at w=0 or w=1
        return -torch.sum(weights * torch.log(weights + epsilon) + 
                         (1 - weights) * torch.log(1 - weights + epsilon)) * temperature
    
    # Use annealing temperature that decreases over time to encourage more binary decisions
    temperature = max(0.5, 10.0 * (1.0 - step / OPTIMIZATION_STEPS))
    #temperature = 0.5

    # Initialize penalties
    P_triple_entropy = torch.tensor(0.0, device=A.device)
    P_join_entropy = torch.tensor(0.0, device=A.device)
    P_l1_triple_out = torch.tensor(0.0, device=A.device)
    P_l1_join_in = torch.tensor(0.0, device=A.device)

    # Compute entropy and L1 penalties only if they are enabled
    if USE_ENTROPY_PENALTY or USE_L1_PENALTY:
        # Compute L1 and entropy penalties for triple outgoing edges
        for i in range(triples_num):
            outgoing = A[i, :]
            
            if USE_L1_PENALTY:
                # For each triple node, we want one strong outgoing edge, others close to 0
                strongest_idx = torch.argmax(outgoing)
                # Apply L1 penalty to all edges except the strongest one
                mask = torch.ones_like(outgoing, dtype=torch.bool)
                mask[strongest_idx] = False
                P_l1_triple_out += torch.sum(torch.abs(outgoing[mask]))
            
            if USE_ENTROPY_PENALTY:
                # Only compute entropy for non-zero weights to encourage sparsity
                mask = outgoing > 0.01
                if torch.any(mask):
                    P_triple_entropy += entropy_penalty(outgoing[mask], temperature)
        
        # Compute L1 and entropy penalties for join nodes
        for i in range(triples_num, N_NODES):
            incoming = A[:, i]
            
            if USE_L1_PENALTY:
                # For each join node, we want exactly two strong incoming edges
                if len(incoming) > 2:  # Only if we have more than 2 edges
                    values, _ = torch.topk(incoming, 2)
                    threshold = values[1]  # Value of the second strongest edge
                    # Apply L1 penalty to all edges below this threshold
                    mask = incoming < threshold
                    P_l1_join_in += torch.sum(torch.abs(incoming[mask]))
            
            if USE_ENTROPY_PENALTY:
                mask = incoming > 0.01
                if torch.any(mask):
                    P_join_entropy += entropy_penalty(incoming[mask], temperature)
    
    # 3. Join nodes have exactly 2 incoming edges
    P_join_in = torch.sum(torch.square(in_degree[join_nodes_indices] - 2.0))
    
    # 4. Join nodes have exactly one outgoing edge, except for one (root) which has zero
    # First, we need a way to identify the root - assuming it's the one with highest index
    root_index = N_NODES - 1
    non_root_join_indices = torch.arange(triples_num, root_index)
    
    P_join_out = torch.sum(torch.square(out_degree[non_root_join_indices] - 1.0)) + \
                torch.square(out_degree[root_index])  # Root should have 0 outgoing edges
    
    # Acyclicity penalty using trace of matrix exponential
    trace_exp = torch.trace(torch.matrix_exp(A)) - N_NODES
    P_acyclic = trace_exp

    # Total Penalty with entropy terms
    total_penalty = lambda_acyclic * P_acyclic + \
                    lambda_triple_in * P_triple_in + \
                    lambda_triple_out * P_triple_out + \
                    lambda_join_in * P_join_in + \
                    lambda_join_out * P_join_out
    
    # Add entropy penalty if enabled
    if USE_ENTROPY_PENALTY:
        total_penalty += lambda_entropy * (P_triple_entropy + P_join_entropy)
    
    # Add L1 penalty if enabled
    if USE_L1_PENALTY:
        total_penalty += lambda_l1 * (P_l1_triple_out + P_l1_join_in)
                    

    # Total loss is predicted cost + penalties
    loss = loss + 0.01 * total_penalty
    #loss = total_penalty

    # Here we calculcate the gradient of loss with respect to A and then perform a gradient descent step
    loss.backward()
    optimizer_opt.step()

    # Clamp edge weights to [0,1]
    with torch.no_grad():
        edge_weights.clamp_(0, 1)

    costs_during_optimization.append(cost_pred.item())
    total_costs_during_optimization.append(total_penalty.item())

    # Record weights every 10 epochs
    if (step + 1) % 10 == 0 or step == 0:
        weight_history.append(edge_weights.detach().clone().numpy())
    
    # Record raw penalties (before lambda multiplication)
    triple_in_penalties.append(P_triple_in.item())
    triple_out_penalties.append(P_triple_out.item())
    join_in_penalties.append(P_join_in.item())
    join_out_penalties.append(P_join_out.item())
    acyclic_penalties.append(P_acyclic.item())
    
    # Track entropy penalties (sum of triple and join entropy)
    total_entropy = (P_triple_entropy + P_join_entropy).item()
    entropy_penalties.append(total_entropy)

    # Track L1 penalties
    total_l1 = (P_l1_triple_out + P_l1_join_in).item()
    l1_penalties.append(total_l1)

    if (step + 1) % 50 == 0 or step == 0:
        pass
        print(f'Optimization Step {step + 1}, Predicted Cost: {cost_pred.item():.4f}')
        print(f'Total Penalty: {total_penalty.item():.4f}')
        print(f'Penalties - Triple In: {P_triple_in.item():.4f}, Triple Out: {P_triple_out.item():.4f}')
        print(f'Penalties - Join In: {P_join_in.item():.4f}, Join Out: {P_join_out.item():.4f}')
        print(f'Acyclicity Penalty: {P_acyclic.item():.4f}')
        print(f'Entropy Penalty: {total_entropy:.4f} ({"enabled" if USE_ENTROPY_PENALTY else "disabled"})')
        print(f'L1 Penalty: {total_l1:.4f} ({"enabled" if USE_L1_PENALTY else "disabled"})')
        print(f'Temperature: {temperature:.4f}')
        # Print the adjacency matrix
        adj_matrix = A.detach().clone()
        #print("\nAdjacency Matrix:")
        #print(adj_matrix.numpy())
        #print("-" * 30)

plt.plot(costs_during_optimization)
plt.xlabel('t')
plt.ylabel('Predicted Cost')
plt.title('Predicted Cost During Optimization')
plt.show()

#plt.plot(plan_distances)
#plt.xlabel('t')
#plt.ylabel('$L_2(true plan, predicted plan)$')
#plt.show()

# Visualize the final adjacency matrix
print("Visualizing the final optimized query plan:")
final_adjacency = A.detach().clone()
visualize_adjacency_matrix(final_adjacency, triples_num, use_tree_layout=False)

# Thresholding
final_adjacency[final_adjacency < 0.5] = 0.0
final_adjacency[final_adjacency >= 0.5] = 1.0
visualize_adjacency_matrix(final_adjacency, triples_num, use_tree_layout=True)

print("Final Adjacency Matrix:")
print(final_adjacency.numpy())

# Convert the original test datapoint edge_index to adjacency matrix and visualize it
print("\nVisualizing the original (ground truth) query plan:")
original_edge_index = test_datapoint.edge_index
original_adjacency = torch.zeros((N_NODES, N_NODES))
original_adjacency[original_edge_index[0], original_edge_index[1]] = 1.0
visualize_adjacency_matrix(original_adjacency, triples_num, use_tree_layout=True)

# Plot the individual penalties over time
plt.figure(figsize=(12, 8))
plt.plot(triple_in_penalties, label='Triple In Penalty')
plt.plot(triple_out_penalties, label='Triple Out Penalty')
plt.plot(join_in_penalties, label='Join In Penalty')
plt.plot(join_out_penalties, label='Join Out Penalty')
plt.plot(acyclic_penalties, label='Acyclicity Penalty')
plt.plot(entropy_penalties, label='Entropy Penalty')
plt.plot(l1_penalties, label='L1 Penalty')
plt.xlabel('Optimization Steps')
plt.ylabel('Raw Penalty Value')
plt.title('Individual Penalties During Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot each penalty on separate subplots for better visibility
fig, axes = plt.subplots(8, 1, figsize=(12, 20), sharex=True)

axes[0].plot(triple_in_penalties, color='blue')
axes[0].set_title('Triple In Penalty')
axes[0].grid(True, alpha=0.3)

axes[1].plot(triple_out_penalties, color='green')
axes[1].set_title('Triple Out Penalty')
axes[1].grid(True, alpha=0.3)

axes[2].plot(join_in_penalties, color='red')
axes[2].set_title('Join In Penalty')
axes[2].grid(True, alpha=0.3)

axes[3].plot(join_out_penalties, color='purple')
axes[3].set_title('Join Out Penalty')
axes[3].grid(True, alpha=0.3)

axes[4].plot(acyclic_penalties, color='orange')
axes[4].set_title('Acyclicity Penalty')
axes[4].grid(True, alpha=0.3)

axes[5].plot(entropy_penalties, color='brown')
axes[5].set_title('Entropy Penalty')
axes[5].grid(True, alpha=0.3)

axes[6].plot(l1_penalties, color='pink')
axes[6].set_title('L1 Penalty')
axes[6].grid(True, alpha=0.3)

# Add a subplot for the total penalty
axes[7].plot(total_costs_during_optimization, color='black', linewidth=2)
axes[7].set_title('Total Penalty')
axes[7].grid(True, alpha=0.3)
axes[7].set_xlabel('Optimization Steps')

plt.tight_layout()
plt.show()