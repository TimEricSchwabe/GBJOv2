import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from model import CostGNNv2, CostGNNv3
from src.create_data.create_cost_model_training_data import SPARQLQuery
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim

import torch_optimizer as optim_extra



from utils.data_utils import (
    adjacency_to_query_with_real_triples,
    count_triples_in_plan,
    collect_triples_in_plan,
    validate_plan,
    plan_to_string,
    plans_are_equivalent,
    load_sparql_queries,
    left_deep_adj_from_perm
)
from optimization.gumbel_utils import sample_binary_concrete, sample_grouped_gumbel_softmax, _temperature_anneal


def add_fingerprints_to_query_data(query_data, fingerprint_dim: int = 64):
    """
    Add random Gaussian fingerprints to join nodes in query data.
    Matches add_fingerprints_to_query_data() from src/evaluation_parallel.py.
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


def visualize_cost_transition(query_file, model_path, device='cpu', num_steps=100):
    """
    Visualize how predicted cost changes when transitioning from 
    plan '1 JOIN 2 JOIN 3' to plan '1 JOIN 3 JOIN 2' for a 3-triple query.
    """
    # Load model
    #model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6, use_jk=False, jk_mode='cat', use_residual=True, use_layer_norm=False, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query (assuming single query with 3 triples)
    queries = load_sparql_queries(query_file, 1)
    query_data = queries[0].torch_data[0]
    
    # Create adjacency matrices for both plans
    # Plan 1: "1 JOIN 2 JOIN 3" -> permutation [0, 1, 2]
    A1 = left_deep_adj_from_perm(torch.tensor([0, 1, 2]))
    
    # Plan 2: "1 JOIN 3 JOIN 2" -> permutation [0, 2, 1] 
    A2 = left_deep_adj_from_perm(torch.tensor([0, 2, 1]))
    
    # Create edge_index for all possible edges
    N_NODES = len(query_data.x)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    
    costs = []
    alphas = np.linspace(0, 1, num_steps)
    
    with torch.no_grad():
        for alpha in alphas:
            # Interpolate between adjacency matrices
            A_interp = (1 - alpha) * A1 + alpha * A2
            
            # Extract edge weights from interpolated adjacency matrix
            edge_weights = A_interp[edge_index[0], edge_index[1]].to(device)
            
            # Predict cost
            cost = model(query_data.x, edge_index, edge_weight=edge_weights)
            costs.append(cost.item())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, costs, 'b-', linewidth=2)
    plt.xlabel('Interpolation α', fontsize=12)
    plt.ylabel('Predicted Cost', fontsize=12)
    plt.title('Cost Transition: "1 JOIN 2 JOIN 3" → "1 JOIN 3 JOIN 2"', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for the two plans
    plt.annotate('1 JOIN 2 JOIN 3', xy=(0, costs[0]), xytext=(0.1, costs[0]),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
    plt.annotate('1 JOIN 3 JOIN 2', xy=(1, costs[-1]), xytext=(0.9, costs[-1]),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cost_transition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plan 1 cost: {costs[0]:.4f}")
    print(f"Plan 2 cost: {costs[-1]:.4f}")
    print(f"Cost difference: {costs[-1] - costs[0]:.4f}")


def canonicalize_perm(perm):
    """Ensure perm[0] < perm[1] to get canonical representative of equivalence class."""
    perm = perm.clone()
    if perm[0] > perm[1]:
        perm[0], perm[1] = perm[1].item(), perm[0].item()
    return perm


def visualize_cost_landscape_3d(query_file, model_path, device='cpu', num_steps=80, include_penalty=False, penalty_config=None):
    """
    Visualize the 3D cost landscape when interpolating between three join plans:
    - Base plan: random left-deep plan
    - α direction: towards a second random (distinct) left-deep plan
    - β direction: towards a third random (distinct) left-deep plan
    
    Args:
        include_penalty: If True, visualize cost + penalty landscape instead of just cost
        penalty_config: Dict with penalty weights (lambda values)
    """
    # Load model
    #model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6, use_jk=False, jk_mode='cat', use_residual=True, use_layer_norm=False, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query
    torch.manual_seed(42)
    np.random.seed(42)
    queries = load_sparql_queries(query_file, 10000)
    queries = [q for q in queries if len(q.triples) == 5]
    query_data = queries[4].torch_data[0]
    query_data = add_fingerprints_to_query_data(query_data, fingerprint_dim=64)
    
    # Calculate query size
    query_size = (query_data.x.shape[0] + 1) // 2
    
    # Generate 3 distinct canonical permutations
    seen = set()
    perms = []

    while len(perms) < 4:
        p = torch.randperm(query_size)
        p = canonicalize_perm(p)
        key = tuple(p.tolist())
        if key not in seen:
            seen.add(key)
            perms.append(p)

    perm1, perm2, perm3, perm4 = perms
    
    A_base = left_deep_adj_from_perm(perm1)
    A_alpha = left_deep_adj_from_perm(perm2)
    A_beta = left_deep_adj_from_perm(perm3)
    A_gamma = left_deep_adj_from_perm(perm4)  # Corner (1,1) - NEW!
    # or use a 3rd random matrix with basic constraints
    # Random matrix with GBJO structural constraints
    #N_NODES = 2 * query_size - 1
    #A_beta = torch.rand(N_NODES, N_NODES)
    
    # Zero out invalid connections:
    # 1. Triple-to-triple: triples can't connect to other triples
    A_beta[:query_size, :query_size] = 0.0
    
    # 2. Join-to-triple: joins can't connect to triples  
    #A_beta[query_size:, :query_size] = 0.0
    
    # 3. Root (last join) has no outgoing edges
    #A_beta[N_NODES - 1, :] = 0.0
    
    # 4. No self-loops
    #A_beta.fill_diagonal_(0.0)
    
    # Create edge_index for all possible edges
    N_NODES = len(query_data.x)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    
    # Set up penalty calculation if needed
    if include_penalty and penalty_config is None:
        # Use default penalty config
        penalty_config = {
            "lambda_acyclic": 29, # 3391
            "lambda_triple_in": 1.5,# 3334.0
            "lambda_triple_out": 1.4,# 2026.0
            "lambda_join_in": 3.6, # 2150.0
            "lambda_join_out": 4.1,# 1295.0
            "lambda_entropy": 0.0,# 0.0
            "lambda_left_linear": 60,# 2157.0

        }
    
    def compute_penalty(A, edge_weights):
        """Compute structural penalty for given adjacency matrix and edge weights"""
        triples_num = (N_NODES + 1) // 2
        
        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)
        
        # Structural penalties
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES
        
        # Left-linear penalty
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)
        child_join_counts = A[join_nodes, :][:, join_nodes].sum(0)
        
        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)
        
        # Entropy penalty - computed on adjacency matrix A
        eps = 1e-10
        probs = A.clamp(min=eps, max=1-eps)
        P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()
        
        # Total penalty
        total_penalty = (
            penalty_config['lambda_triple_in'] * P_triple_in +
            penalty_config['lambda_triple_out'] * P_triple_out +
            penalty_config['lambda_join_in'] * P_join_in +
            penalty_config['lambda_join_out'] * P_join_out +
            penalty_config['lambda_acyclic'] * P_acyclic +
            penalty_config['lambda_left_linear'] * P_left_linear +
            penalty_config['lambda_entropy'] * P_entropy
        )

        # Print all the penalties for debugging
        if include_penalty:
            pass
            #print(f"P_acyclic: {P_acyclic.item():.4f}")
            #print(f"P_triple_in: {P_triple_in.item():.4f}")
            #print(f"P_triple_out: {P_triple_out.item():.4f}")
            #print(f"P_join_in: {P_join_in.item():.4f}")
            #print(f"P_join_out: {P_join_out.item():.4f}")
            #print(f"P_left_linear: {P_left_linear.item():.4f}")
            #print(f"P_entropy: {P_entropy.item():.4f}")
        
        return total_penalty
    
    # Create grid of interpolation parameters
    alphas = np.linspace(0, 1, num_steps)
    betas = np.linspace(0, 1, num_steps)
    Alpha, Beta = np.meshgrid(alphas, betas)
    
    # Initialize cost surface
    Cost = np.zeros_like(Alpha)
    
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Bilinear interpolation between three plans
                A_interp = (
                    (1 - alpha) * (1 - beta) * A_base +
                    alpha * (1 - beta) * A_alpha +
                    (1 - alpha) * beta * A_beta +
                    alpha * beta * A_gamma
                    #alpha * beta * (0.5 * A_alpha + 0.5 * A_beta)  # Average at corner
                )
                
                # Extract edge weights from interpolated adjacency matrix
                edge_weights = A_interp[edge_index[0], edge_index[1]].to(device)
                
                # Predict cost
                cost = model(query_data.x, edge_index, edge_weight=edge_weights)
                
                if include_penalty:
                    penalty = compute_penalty(A_interp.to(device), edge_weights)
                    Cost[j, i] = cost.item() + 0.99 * penalty.item()
                else:
                    Cost[j, i] = cost.item()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(Alpha, Beta, Cost, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Mark the three corner plans
    ax.scatter([0, 1, 0], [0, 0, 1], [Cost[0,0], Cost[0,-1], Cost[-1,0]], 
              c='red', s=100, alpha=1.0)
    
    # Labels and title
    ax.set_xlabel('α → "1 JOIN 3 JOIN 2"', fontsize=11)
    ax.set_ylabel('β → "2 JOIN 3 JOIN 1"', fontsize=11)
    
    if include_penalty:
        ax.set_zlabel('$C + P$', fontsize=11)
        ax.set_title('3D Cost + Penalty Landscape Between Join Plans', fontsize=14)
        filename = 'cost_penalty_landscape_3d.png'
    else:
        ax.set_zlabel('Predicted Cost', fontsize=11)
        ax.set_title('3D Cost Landscape Between Join Plans', fontsize=14)
        filename = 'cost_landscape_3d.png'
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    # Add text annotations for the three plans
    ax.text(0, 0, Cost[0,0], '  1→2→3', fontsize=9)
    ax.text(1, 0, Cost[0,-1], '  1→3→2', fontsize=9) 
    ax.text(0, 1, Cost[-1,0], '  2→3→1', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print corner costs
    landscape_type = "Cost + Penalty" if include_penalty else "Cost"
    print(f"Base plan (1→2→3) {landscape_type.lower()}: {Cost[0,0]:.4f}")
    print(f"Alpha plan (1→3→2) {landscape_type.lower()}: {Cost[0,-1]:.4f}")
    print(f"Beta plan (2→3→1) {landscape_type.lower()}: {Cost[-1,0]:.4f}")
    print(f"Mixed corner {landscape_type.lower()}: {Cost[-1,-1]:.4f}")
    print(f"Min {landscape_type.lower()}: {Cost.min():.4f} at α={alphas[np.unravel_index(Cost.argmin(), Cost.shape)[1]]:.2f}, β={betas[np.unravel_index(Cost.argmin(), Cost.shape)[0]]:.2f}")
    print(f"Max {landscape_type.lower()}: {Cost.max():.4f} at α={alphas[np.unravel_index(Cost.argmax(), Cost.shape)[1]]:.2f}, β={betas[np.unravel_index(Cost.argmax(), Cost.shape)[0]]:.2f}")
    
    # Create contour plot
    fig_contour, ax_contour = plt.subplots(figsize=(10, 8))

        # Light version → Base color → Dark version
    dark = '#a9d6e5' # white
    base = '#468faf'
    light = '#012a4a'

    cmap = LinearSegmentedColormap.from_list('custom_intensity', [light, base, dark])
    
    # Filled contour plot
    contour_filled = ax_contour.contourf(Alpha, Beta, Cost, levels=40, cmap="viridis")
    
    # Add contour lines
    contour_lines = ax_contour.contour(Alpha, Beta, Cost, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax_contour.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Mark the three corner plans
    ax_contour.scatter([0, 1, 0], [0, 0, 1], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
    ax_contour.annotate('1→2→3', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=10, color='white')
    ax_contour.annotate('1→3→2', (1, 0), textcoords="offset points", xytext=(-50, 10), fontsize=10, color='white')
    ax_contour.annotate('2→3→1', (0, 1), textcoords="offset points", xytext=(10, -15), fontsize=10, color='white')
    
    # Mark minimum point
    min_idx = np.unravel_index(Cost.argmin(), Cost.shape)
    min_alpha = alphas[min_idx[1]]
    min_beta = betas[min_idx[0]]
    ax_contour.scatter([min_alpha], [min_beta], c='yellow', s=150, marker='*', zorder=6, edgecolors='black', linewidths=1)
    ax_contour.annotate(f'Min: {Cost.min():.2f}', (min_alpha, min_beta), 
                       textcoords="offset points", xytext=(10, 10), fontsize=9, color='yellow')
    
    # Labels and title
    ax_contour.set_xlabel('α → "1 JOIN 3 JOIN 2"', fontsize=12)
    ax_contour.set_ylabel('β → "2 JOIN 3 JOIN 1"', fontsize=12)
    
    if include_penalty:
        ax_contour.set_title('Cost + Penalty Landscape Contour Plot', fontsize=14)
        contour_filename = 'cost_penalty_landscape_contour.pdf'
    else:
        ax_contour.set_title('Cost Landscape Contour Plot', fontsize=14)
        contour_filename = 'cost_landscape_contour.pdf'
    
    # Add colorbar
    cbar = fig_contour.colorbar(contour_filled, ax=ax_contour)
    cbar.set_label(landscape_type, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(contour_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Contour plot saved as '{contour_filename}'")
    
    # Create clean contour plot (surface only, no annotations)
    fig_clean, ax_clean = plt.subplots(figsize=(10, 8))
    
    # Filled contour plot only
    ax_clean.contourf(Alpha, Beta, Cost, levels=30, cmap="viridis")
    
    # Add subtle gray contour lines
    ax_clean.contour(Alpha, Beta, Cost, levels=30, colors='#000000', alpha=1, linewidths=0.5)
    
    # Remove all axes, ticks, and borders
    ax_clean.set_axis_off()
    
    # Set filename
    if include_penalty:
        clean_contour_filename = 'cost_penalty_landscape_contour_clean.pdf'
    else:
        clean_contour_filename = 'cost_landscape_contour_clean.svg'
    
    plt.tight_layout()
    plt.savefig(clean_contour_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print(f"Clean contour plot saved as '{clean_contour_filename}'")
    
    # Create contour plot with quiver (gradient field)
    fig_quiver, ax_quiver = plt.subplots(figsize=(10, 8))
    
    # Filled contour plot
    contour_filled_q = ax_quiver.contourf(Alpha, Beta, Cost, levels=40, cmap='viridis')
    
    # Add contour lines
    #contour_lines_q = ax_quiver.contour(Alpha, Beta, Cost, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    #ax_quiver.clabel(contour_lines_q, inline=True, fontsize=8, fmt='%.1f')
    
    # Calculate gradients using numpy
    # np.gradient returns (dCost/dbeta, dCost/dalpha) for a 2D array
    # where the first axis is beta (rows) and second axis is alpha (columns)
    dCost_dbeta, dCost_dalpha = np.gradient(Cost, betas, alphas)
    
    # Downsample for cleaner quiver plot (every nth point)
    step = max(1, num_steps // 15)  # Aim for ~15x15 arrows
    Alpha_ds = Alpha[::step, ::step]
    Beta_ds = Beta[::step, ::step]
    dCost_dalpha_ds = dCost_dalpha[::step, ::step]
    dCost_dbeta_ds = dCost_dbeta[::step, ::step]
    
    # Normalize arrows to unit length for cleaner visualization
    magnitude = np.sqrt(dCost_dalpha_ds**2 + dCost_dbeta_ds**2)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    u_norm = -dCost_dalpha_ds / magnitude
    v_norm = -dCost_dbeta_ds / magnitude
    
    # Plot quiver with paper-style thin arrows (negative gradient = descent direction)
    quiver = ax_quiver.quiver(Alpha_ds, Beta_ds, u_norm, v_norm,
                               color='#333333', alpha=0.85,
                               scale=25, width=0.003, headwidth=3, headlength=4, headaxislength=3.5)
    
    # Mark the three corner plans
    ax_quiver.scatter([0, 1, 0], [0, 0, 1], c='#d62728', s=60, zorder=5, edgecolors='black', linewidths=0.8)
    ax_quiver.annotate('$P_1$', (0, 0), textcoords="offset points", xytext=(8, 8), fontsize=10, color='black')
    ax_quiver.annotate('$P_2$', (1, 0), textcoords="offset points", xytext=(-50, 8), fontsize=10, color='black')
    ax_quiver.annotate('$P_3$', (0, 1), textcoords="offset points", xytext=(8, -15), fontsize=10, color='black')
    
    # Mark minimum point
    #ax_quiver.scatter([min_alpha], [min_beta], c='#ff7f0e', s=100, marker='*', zorder=6, edgecolors='black', linewidths=0.5)
    #ax_quiver.annotate(f'Min: {Cost.min():.2f}', (min_alpha, min_beta), 
    #                   textcoords="offset points", xytext=(8, 8), fontsize=9, color='black')
    
    # Labels and title
    ax_quiver.set_xlabel('$(1-\\alpha) P_1 + \\alpha P_2$', fontsize=12)
    ax_quiver.set_ylabel('$(1-\\beta) P_1 + \\beta P_3$', fontsize=12)
    
    if include_penalty:
        #ax_quiver.set_title('Cost + Penalty Landscape with Gradient Field (Descent Direction)', fontsize=14)
        quiver_filename = 'cost_penalty_landscape_quiver.pdf'
    else:
        #ax_quiver.set_title('Cost Landscape with Gradient Field (Descent Direction)', fontsize=14)
        quiver_filename = 'cost_landscape_quiver.pdf'
    
    # Add colorbar for contour
    #cbar_q = fig_quiver.colorbar(contour_filled_q, ax=ax_quiver)
    #cbar_q.set_label(landscape_type, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(quiver_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Quiver plot saved as '{quiver_filename}'")
    
    # Create streamplot (gradient flow visualization)
    fig_stream, ax_stream = plt.subplots(figsize=(10, 10))
    
    # Optional: add contour background for context
    contour_bg = ax_stream.contourf(Alpha, Beta, Cost, levels=40, cmap='viridis', alpha=0.7)
    #ax_stream.contour(Alpha, Beta, Cost, levels=20, colors='white', alpha=0.2, linewidths=0.3)
    
    # Streamplot uses the full resolution data (not downsampled)
    # Note: streamplot requires 1D arrays for x, y coordinates
    stream = ax_stream.streamplot(alphas, betas, -dCost_dalpha, -dCost_dbeta,
                                   color='#222222', linewidth=0.8, density=1.5,
                                   arrowstyle='->', arrowsize=1.0)
    
    # Mark the three corner plans
    ax_stream.scatter([0, 1, 0, 1], [0, 0, 1, 1], c='#d62728', s=60, zorder=5, edgecolors='white', linewidths=0.8)
    ax_stream.annotate('$P_1$', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=35, color='black')
    ax_stream.annotate('$P_2$', (1, 0), textcoords="offset points", xytext=(-25, 10), fontsize=35, color='black')
    ax_stream.annotate('$P_3$', (0, 1), textcoords="offset points", xytext=(10, -20), fontsize=35, color='black')
    ax_stream.annotate('$P_4$', (1, 1), textcoords="offset points", xytext=(10, 10), fontsize=35, color='black')
    
    # Mark minimum point
    #ax_stream.scatter([min_alpha], [min_beta], c='#ffcc00', s=120, marker='*', zorder=6, edgecolors='black', linewidths=0.5)
    #ax_stream.annotate(f'Min: {Cost.min():.2f}', (min_alpha, min_beta), 
    #                   textcoords="offset points", xytext=(8, 8), fontsize=9, color='white')
    
    # Labels and title
    #ax_stream.set_xlabel('$(1-\\alpha) P_1 + \\alpha P_2$', fontsize=30)
    #ax_stream.set_ylabel('$(1-\\beta) P_1 + \\beta P_3$', fontsize=30)
    ax_stream.tick_params(axis='both', which='major', labelsize=24)
    
    if include_penalty:
        #ax_stream.set_title('Cost + Penalty Landscape with Gradient Streamlines', fontsize=14)
        stream_filename = 'cost_penalty_landscape_streamplot.pdf'
    else:
        #ax_stream.set_title('Cost Landscape with Gradient Streamlines', fontsize=14)
        stream_filename = 'cost_landscape_streamplot.pdf'
    
    # Add colorbar
    #cbar_stream = fig_stream.colorbar(contour_bg, ax=ax_stream)
    #cbar_stream.set_label(landscape_type, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(stream_filename, dpi=300)
    plt.show()
    
    print(f"Streamplot saved as '{stream_filename}'")

    exit()

    # ============================================================
    # BARYCENTRIC INTERPOLATION PLOT (Triangular domain, 3 plans)
    # ============================================================
    # Uses barycentric coordinates: A = λ₁A₁ + λ₂A₂ + λ₃A₃ where Σλᵢ = 1, λᵢ ≥ 0
    
    def bary_to_cartesian(l1, l2, l3):
        """Convert barycentric coordinates to 2D Cartesian for equilateral triangle.
        Vertices: A_base at (0, 0), A_alpha at (1, 0), A_beta at (0.5, sqrt(3)/2)
        """
        x = l2 + 0.5 * l3
        y = (np.sqrt(3) / 2) * l3
        return x, y
    
    # Generate barycentric grid points
    bary_steps = num_steps
    bary_costs = []
    bary_coords = []
    bary_l1 = []
    bary_l2 = []
    bary_l3 = []
    
    with torch.no_grad():
        for i in range(bary_steps + 1):
            for j in range(bary_steps + 1 - i):
                # Barycentric coordinates (sum to 1)
                lambda1 = i / bary_steps  # weight for A_base
                lambda2 = j / bary_steps  # weight for A_alpha
                lambda3 = 1 - lambda1 - lambda2  # weight for A_beta
                
                # Barycentric interpolation (no artificial 4th corner!)
                A_interp_bary = lambda1 * A_base + lambda2 * A_alpha + lambda3 * A_beta
                
                # Extract edge weights and predict cost
                edge_weights_bary = A_interp_bary[edge_index[0], edge_index[1]].to(device)
                cost_bary = model(query_data.x, edge_index, edge_weight=edge_weights_bary)
                
                if include_penalty:
                    penalty_bary = compute_penalty(A_interp_bary.to(device), edge_weights_bary)
                    bary_costs.append(cost_bary.item() + 0.99 * penalty_bary.item())
                else:
                    bary_costs.append(cost_bary.item())
                
                bary_coords.append((lambda1, lambda2, lambda3))
                bary_l1.append(lambda1)
                bary_l2.append(lambda2)
                bary_l3.append(lambda3)
    
    # Convert to numpy arrays
    bary_costs = np.array(bary_costs)
    bary_l1 = np.array(bary_l1)
    bary_l2 = np.array(bary_l2)
    bary_l3 = np.array(bary_l3)
    
    # Convert barycentric to Cartesian coordinates
    X_bary = bary_l2 + 0.5 * bary_l3
    Y_bary = (np.sqrt(3) / 2) * bary_l3
    
    # Create 3D triangular surface plot
    fig_bary_3d = plt.figure(figsize=(12, 9))
    ax_bary_3d = fig_bary_3d.add_subplot(111, projection='3d')
    
    # Use trisurf for triangular data
    trisurf = ax_bary_3d.plot_trisurf(X_bary, Y_bary, bary_costs, cmap='viridis', alpha=0.8,
                                       linewidth=0, antialiased=True)
    
    # Mark the three corner plans
    corner_coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # (λ₁, λ₂, λ₃) for each corner
    corner_labels = ['$P_1$ (base)', '$P_2$ (α)', '$P_3$ (β)']
    corner_x = []
    corner_y = []
    corner_z = []
    
    for (l1, l2, l3), label in zip(corner_coords, corner_labels):
        x, y = bary_to_cartesian(l1, l2, l3)
        # Find the cost at this corner
        idx = next(i for i, c in enumerate(bary_coords) if abs(c[0]-l1) < 1e-6 and abs(c[1]-l2) < 1e-6)
        corner_x.append(x)
        corner_y.append(y)
        corner_z.append(bary_costs[idx])
    
    ax_bary_3d.scatter(corner_x, corner_y, corner_z, c='red', s=100, zorder=5)
    for (x, y, z), label in zip(zip(corner_x, corner_y, corner_z), corner_labels):
        ax_bary_3d.text(x, y, z, f'  {label}', fontsize=9)
    
    ax_bary_3d.set_xlabel('x', fontsize=11)
    ax_bary_3d.set_ylabel('y', fontsize=11)
    
    if include_penalty:
        ax_bary_3d.set_zlabel('$C + P$', fontsize=11)
        ax_bary_3d.set_title('Barycentric Cost + Penalty Landscape (3 Plans)', fontsize=14)
        bary_3d_filename = 'cost_penalty_landscape_barycentric_3d.png'
    else:
        ax_bary_3d.set_zlabel('Predicted Cost', fontsize=11)
        ax_bary_3d.set_title('Barycentric Cost Landscape (3 Plans)', fontsize=14)
        bary_3d_filename = 'cost_landscape_barycentric_3d.png'
    
    fig_bary_3d.colorbar(trisurf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(bary_3d_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Barycentric 3D plot saved as '{bary_3d_filename}'")
    
    # Create 2D triangular contour plot
    fig_bary_2d, ax_bary_2d = plt.subplots(figsize=(10, 9))
    
    # Use tricontourf for filled contours on triangular mesh
    from matplotlib.tri import Triangulation
    triang = Triangulation(X_bary, Y_bary)
    
    # Filled contour
    bary_contour = ax_bary_2d.tricontourf(triang, bary_costs, levels=40, cmap='viridis')
    
    # Add contour lines
    ax_bary_2d.tricontour(triang, bary_costs, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Draw triangle boundary
    triangle_x = [0, 1, 0.5, 0]  # x coords of vertices + back to start
    triangle_y = [0, 0, np.sqrt(3)/2, 0]  # y coords of vertices + back to start
    ax_bary_2d.plot(triangle_x, triangle_y, 'k-', linewidth=2)
    
    # Mark the three corner plans
    ax_bary_2d.scatter(corner_x, corner_y, c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
    ax_bary_2d.annotate('$P_1$', (0, 0), textcoords="offset points", xytext=(10, -15), fontsize=14, color='black')
    ax_bary_2d.annotate('$P_2$', (1, 0), textcoords="offset points", xytext=(-5, -15), fontsize=14, color='black')
    ax_bary_2d.annotate('$P_3$', (0.5, np.sqrt(3)/2), textcoords="offset points", xytext=(-5, 10), fontsize=14, color='black')
    
    # Mark minimum point
    min_idx_bary = np.argmin(bary_costs)
    min_x_bary = X_bary[min_idx_bary]
    min_y_bary = Y_bary[min_idx_bary]
    ax_bary_2d.scatter([min_x_bary], [min_y_bary], c='yellow', s=150, marker='*', zorder=6, edgecolors='black', linewidths=1)
    ax_bary_2d.annotate(f'Min: {bary_costs[min_idx_bary]:.2f}', (min_x_bary, min_y_bary), 
                        textcoords="offset points", xytext=(10, 10), fontsize=9, color='black')
    
    # Add axis labels showing barycentric meaning
    ax_bary_2d.set_xlabel('Barycentric x (→ $P_2$)', fontsize=12)
    ax_bary_2d.set_ylabel('Barycentric y (→ $P_3$)', fontsize=12)
    ax_bary_2d.set_aspect('equal')
    
    if include_penalty:
        ax_bary_2d.set_title('Barycentric Cost + Penalty Landscape', fontsize=14)
        bary_2d_filename = 'cost_penalty_landscape_barycentric_2d.pdf'
    else:
        ax_bary_2d.set_title('Barycentric Cost Landscape', fontsize=14)
        bary_2d_filename = 'cost_landscape_barycentric_2d.pdf'
    
    cbar_bary = fig_bary_2d.colorbar(bary_contour, ax=ax_bary_2d)
    cbar_bary.set_label(landscape_type, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(bary_2d_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Barycentric 2D contour plot saved as '{bary_2d_filename}'")
    
    # Print barycentric corner costs
    print(f"\n--- Barycentric Interpolation Results ---")
    print(f"$A(\\lambda) = \\lambda_1 A_1 + \\lambda_2 A_2 + \\lambda_3 A_3$ where $\\sum \\lambda_i = 1$")
    for (l1, l2, l3), label, z in zip(corner_coords, ['P_1 (base)', 'P_2 (alpha)', 'P_3 (beta)'], corner_z):
        print(f"{label} at (λ₁={l1}, λ₂={l2}, λ₃={l3}): {landscape_type.lower()} = {z:.4f}")
    min_bary_coord = bary_coords[min_idx_bary]
    print(f"Min {landscape_type.lower()}: {bary_costs.min():.4f} at λ₁={min_bary_coord[0]:.2f}, λ₂={min_bary_coord[1]:.2f}, λ₃={min_bary_coord[2]:.2f}")


def visualize_plan_space_projection(
    query_file: str,
    model_path: str,
    device: str = "cpu",
    num_samples: int = 1000,
    include_penalty: bool = False,
    penalty_config: dict = None,
    projection_method: str = "tsne",  # "tsne", "pca", "umap"
    perplexity: int = 30,  # for t-SNE
    random_seed: int = 42,
    representation: str = "gnn_embedding",  # "gnn_embedding" or "spectral"
    spectral_k: int = None,  # Number of eigenvalues to use (default: all)
    query_size: int = 5,
):
    """
    Sample random valid plans, compute costs (optionally with penalties), 
    and project to 2D using t-SNE or PCA.
    
    Args:
        query_file: Path to SPARQL query file
        model_path: Path to trained cost model
        device: Device to run on ("cpu" or "cuda")
        num_samples: Number of random plans to sample
        include_penalty: If True, add structural penalties to the cost
        penalty_config: Dict with penalty weights (lambda values)
        projection_method: "tsne", "pca", or "umap"
        perplexity: Perplexity parameter for t-SNE
        random_seed: Random seed for reproducibility
        representation: How to represent each plan for projection
            - "gnn_embedding": Hidden state from cost model (before fc2)
            - "spectral": Eigenvalues of the adjacency matrix
        spectral_k: Number of eigenvalues to use for spectral representation (default: all N_NODES)
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Set seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Load model
    model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6, use_jk=False, 
                      jk_mode='cat', use_residual=True, use_layer_norm=False, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query
    queries = load_sparql_queries(query_file, 10000)
    queries = [q for q in queries if len(q.triples) == query_size]
    query_data = queries[0].torch_data[0]
    query_data = add_fingerprints_to_query_data(query_data, fingerprint_dim=64)
    
    # Calculate dimensions
    query_size = (query_data.x.shape[0] + 1) // 2  # number of triples
    N_NODES = len(query_data.x)
    
    # Create edge_index for all possible edges
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)
    
    # Identify valid edge positions (non-structural-zero)
    # Valid edges: NOT (into triples) AND NOT (out of root) AND NOT (triple-to-triple) AND NOT (self-loop)
    valid_edge_mask = torch.ones(N_NODES, N_NODES, dtype=torch.bool)
    valid_edge_mask[:, :query_size] = False  # No edges INTO triples
    valid_edge_mask[N_NODES - 1, :] = False  # No edges OUT OF root
    valid_edge_mask[:query_size, :query_size] = False  # No triple-to-triple
    valid_edge_mask.fill_diagonal_(False)  # No self-loops
    
    valid_positions = torch.where(valid_edge_mask)
    num_valid_edges = len(valid_positions[0])
    print(f"Number of valid edge positions: {num_valid_edges}")
    
    # Set up penalty calculation if needed
    if include_penalty and penalty_config is None:
        penalty_config = {
            "lambda_acyclic": 29,
            "lambda_triple_in": 1.5,
            "lambda_triple_out": 1.4,
            "lambda_join_in": 3.6,
            "lambda_join_out": 4.1,
            "lambda_entropy": 100,
            "lambda_left_linear": 60,
        }
    
    def compute_penalty(A, edge_weights):
        """Compute structural penalty for given adjacency matrix."""
        triples_num = (N_NODES + 1) // 2
        
        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)
        
        # Structural penalties
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES
        
        # Left-linear penalty
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)
        child_join_counts = A[join_nodes, :][:, join_nodes].sum(0)
        
        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)
        
        # Entropy penalty
        A_valid = A[A > 1e-6]
        if len(A_valid) > 0:
            P_entropy = -(A_valid * torch.log(A_valid + 1e-8)).sum()
        else:
            P_entropy = torch.tensor(0.0, device=device)
        
        total_penalty = (
            penalty_config['lambda_triple_in'] * P_triple_in +
            penalty_config['lambda_triple_out'] * P_triple_out +
            penalty_config['lambda_join_in'] * P_join_in +
            penalty_config['lambda_join_out'] * P_join_out +
            penalty_config['lambda_acyclic'] * P_acyclic +
            penalty_config['lambda_left_linear'] * P_left_linear +
            penalty_config['lambda_entropy'] * P_entropy
        )
        
        return total_penalty
    
    def generate_random_valid_adjacency():
        """Generate a random adjacency matrix matching GBJO's softmax sampling.
        
        Mimics GBJO: sample random logits, mask invalid edges, apply grouped softmax.
        This produces soft adjacencies similar to what the model sees during optimization.
        """
        from optimization.gumbel_utils import sample_grouped_gumbel_softmax
        
        # Sample random logits (like GBJO initialization + noise)
        edge_logits = torch.randn(num_edges) * 0.5  # Random logits
        
        # Mask invalid edges (triple-to-triple, join-to-triple)
        triple_to_triple_mask = (edge_index[0] < query_size) & (edge_index[1] < query_size)
        join_to_triple_mask = (edge_index[0] >= query_size) & (edge_index[1] < query_size)
        
        masked_logits = edge_logits.clone()
        masked_logits[triple_to_triple_mask.cpu()] = float('-inf')
        masked_logits[join_to_triple_mask.cpu()] = float('-inf')
        
        # Apply grouped softmax (one outgoing edge per source node, like GBJO)
        # Use temperature tau=1.0 for "soft" but peaked distributions
        #tau = tau = 10 ** (torch.rand(1).item() * 3 - 2)
        #tau = np.random.uniform(0.01, 5.0)
        tau = 0.01
        edge_weights = sample_grouped_gumbel_softmax(
            masked_logits.to(device), 
            edge_index[0], 
            temperature=tau, 
            use_gumbel_noise=False  # Deterministic softmax
        )
        
        # Root (final join) has no outgoing edges
        edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        
        # Build adjacency matrix from edge weights
        A = torch.zeros((N_NODES, N_NODES))
        A[edge_index[0].cpu(), edge_index[1].cpu()] = edge_weights.cpu()
        
        return A
    
    def extract_spectral_features(A, k=None):
        """Extract spectral features (Laplacian eigenvalues) from adjacency matrix.
        
        Uses the symmetric normalized Laplacian for better numerical properties.
        For directed graphs, we first symmetrize the adjacency matrix.
        
        Args:
            A: Adjacency matrix (N x N)
            k: Number of eigenvalues to return (default: all)
        
        Returns:
            Sorted Laplacian eigenvalues (ascending) as a numpy array
        """
        A_np = A.numpy()
        
        # Symmetrize for directed graphs: A_sym = (A + A^T) / 2
        A_sym = (A_np + A_np.T) / 2
        
        # Compute degree matrix (sum of rows for undirected/symmetrized)
        degrees = A_sym.sum(axis=1)
        D = np.diag(degrees)
        
        # Compute unnormalized Laplacian: L = D - A
        L = D - A_sym
        
        # For normalized Laplacian (better numerical properties):
        # L_sym = I - D^(-1/2) A D^(-1/2)
        # Handle zero degrees to avoid division by zero
        degrees_safe = np.where(degrees > 1e-10, degrees, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
        # Set to 0 for zero-degree nodes
        D_inv_sqrt[degrees < 1e-10] = 0
        
        L_normalized = np.eye(len(A_np)) - D_inv_sqrt @ A_sym @ D_inv_sqrt
        
        # Compute eigenvalues of the normalized Laplacian
        # eigenvalues are real for symmetric matrices
        eigenvalues = np.linalg.eigvalsh(L_normalized)  # eigvalsh for symmetric
        
        # Sort ascending (smallest first - includes 0 for connected components)
        eigenvalues_sorted = np.sort(eigenvalues)
        
        if k is not None and k < len(eigenvalues_sorted):
            return eigenvalues_sorted[:k]
        return eigenvalues_sorted
    
    # Set default spectral_k if not provided
    if spectral_k is None:
        spectral_k = N_NODES  # Use all eigenvalues by default
    
    # Set up hook for capturing GNN hidden state if needed
    captured_embeddings = []
    hook_handle = None
    
    if representation == "gnn_embedding":
        def _hook(_module, inputs, _output):
            # inputs is a tuple; for Linear, inputs[0] is the pooled embedding
            x_in = inputs[0].detach().cpu()
            # Ensure 2D shape (1, D) for single samples
            if x_in.dim() == 1:
                x_in = x_in.unsqueeze(0)
            captured_embeddings.append(x_in)
        
        # Hook into fc2 to capture the input (pooled graph embedding)
        if not hasattr(model, "fc2"):
            raise AttributeError("Model has no attribute `fc2`; cannot hook pooled embedding.")
        hook_handle = model.fc1.register_forward_hook(_hook)
        print(f"Using GNN embedding representation (hooking into fc2)")
    elif representation == "spectral":
        print(f"Using spectral representation (Laplacian eigenvalues, k={spectral_k})")
    else:
        raise ValueError(f"Unknown representation: {representation}. Use 'gnn_embedding' or 'spectral'.")
    
    # ============================================================
    # First, sample DISCRETE LEFT-DEEP plans
    # ============================================================
    import math
    max_discrete_plans = min(100, math.factorial(query_size))  # Can't have more than query_size! plans
    max_discrete_plans = 0
    print(f"Sampling up to {max_discrete_plans} distinct left-deep discrete plans...")
    
    discrete_plans = []
    seen_perms = set()
    attempts = 0
    max_attempts = max_discrete_plans * 10  # Limit attempts to avoid infinite loop
    
    while len(discrete_plans) < max_discrete_plans and attempts < max_attempts:
        attempts += 1
        p = torch.randperm(query_size)
        p = canonicalize_perm(p)
        key = tuple(p.tolist())
        if key not in seen_perms:
            seen_perms.add(key)
            A_discrete = left_deep_adj_from_perm(p)
            discrete_plans.append(A_discrete)
    
    num_discrete = len(discrete_plans)
    print(f"  Generated {num_discrete} distinct left-deep discrete plans")
    discrete_plans = []
    
    # ============================================================
    # Sample DISCRETE BUSHY plans (non-left-deep)
    # ============================================================
    def generate_random_bushy_plan():
        """Generate a random bushy (non-left-deep) discrete plan.
        
        Builds a random binary tree over the triples by repeatedly
        picking two random unjoined nodes and joining them.
        """
        # Nodes: 0..query_size-1 are triples, query_size..N_NODES-1 are joins
        # We'll build the tree bottom-up
        
        A = torch.zeros((N_NODES, N_NODES))
        
        # Start with all triples as "available" roots
        available_roots = list(range(query_size))  # Triple indices
        next_join_idx = query_size  # First join node
        
        # Shuffle the triples for randomness
        import random
        random.shuffle(available_roots)
        
        # Keep joining until we have one root
        while len(available_roots) > 1 and next_join_idx < N_NODES:
            # Pick two random available roots to join
            idx1 = random.randint(0, len(available_roots) - 1)
            root1 = available_roots.pop(idx1)
            
            idx2 = random.randint(0, len(available_roots) - 1)
            root2 = available_roots.pop(idx2)
            
            # Create edges: both children point to the new join
            A[root1, next_join_idx] = 1.0
            A[root2, next_join_idx] = 1.0
            
            # The new join becomes an available root
            available_roots.append(next_join_idx)
            next_join_idx += 1
        
        return A
    
    def adjacency_to_key(A):
        """Convert adjacency to hashable key for deduplication."""
        return tuple(A.flatten().tolist())
    
    max_bushy_plans = 0
    print(f"Sampling up to {max_bushy_plans} distinct bushy discrete plans...")
    
    bushy_plans = []
    seen_bushy = set()
    attempts = 0
    max_attempts = max_bushy_plans * 20  # More attempts since bushy space is larger
    
    while len(bushy_plans) < max_bushy_plans and attempts < max_attempts:
        attempts += 1
        A_bushy = generate_random_bushy_plan()
        key = adjacency_to_key(A_bushy)
        if key not in seen_bushy:
            seen_bushy.add(key)
            bushy_plans.append(A_bushy)
    
    num_bushy = len(bushy_plans)
    bushy_plans = []
    print(f"  Generated {num_bushy} distinct bushy discrete plans")
    
    # ============================================================
    # Sample random SOFT plans and compute costs for all
    # ============================================================
    print(f"Sampling {num_samples} random soft plans...")
    representations = []
    costs = []
    adjacencies = []
    discrete_indices = []  # Track which indices are left-deep discrete plans
    bushy_indices = []  # Track which indices are bushy discrete plans
    
    with torch.no_grad():
        # First, process left-deep discrete plans
        for i, A in enumerate(discrete_plans):
            adjacencies.append(A)
            discrete_indices.append(len(adjacencies) - 1)
            
            # Compute cost (this also triggers the hook if using gnn_embedding)
            edge_weights = A[edge_index[0], edge_index[1]].to(device)
            cost = model(query_data.x.to(device), edge_index, edge_weight=edge_weights)
            # clamp to larger 0
            cost = torch.clamp(cost, min=0.0)
            
            if representation == "spectral":
                rep = extract_spectral_features(A, k=spectral_k)
                representations.append(rep)
            
            if include_penalty:
                penalty = compute_penalty(A.to(device), edge_weights)
                total_cost = cost.item() + 0.99 * penalty.item()
            else:
                total_cost = cost.item()
            
            costs.append(total_cost)
        
        print(f"  Processed {num_discrete} left-deep discrete plans")
        
        # Process bushy discrete plans
        for i, A in enumerate(bushy_plans):
            adjacencies.append(A)
            bushy_indices.append(len(adjacencies) - 1)
            
            # Compute cost (this also triggers the hook if using gnn_embedding)
            edge_weights = A[edge_index[0], edge_index[1]].to(device)
            cost = model(query_data.x.to(device), edge_index, edge_weight=edge_weights)
            cost = torch.clamp(cost, min=0.0)
            
            if representation == "spectral":
                rep = extract_spectral_features(A, k=spectral_k)
                representations.append(rep)
            
            if include_penalty:
                penalty = compute_penalty(A.to(device), edge_weights)
                total_cost = cost.item() + 0.99 * penalty.item()
            else:
                total_cost = cost.item()
            
            costs.append(total_cost)
        bushy_indices = []
        
        print(f"  Processed {num_bushy} bushy discrete plans")
        
        # Then, process random soft plans
        for i in range(num_samples):
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{num_samples} soft plans")
            
            # Generate random valid adjacency
            A = generate_random_valid_adjacency()
            adjacencies.append(A)
            
            # Compute cost (this also triggers the hook if using gnn_embedding)
            edge_weights = A[edge_index[0], edge_index[1]].to(device)
            cost = model(query_data.x.to(device), edge_index, edge_weight=edge_weights)
            # clamp to larger 0
            cost = torch.clamp(cost, min=0.0)
            if representation == "spectral":
                # Extract spectral features as representation
                rep = extract_spectral_features(A, k=spectral_k)
                representations.append(rep)
            # For gnn_embedding, the hook captures the representation
            
            if include_penalty:
                penalty = compute_penalty(A.to(device), edge_weights)
                total_cost = cost.item() + 0.99 * penalty.item()
            else:
                total_cost = cost.item()
            
            costs.append(total_cost)
    
    # Clean up hook and collect embeddings
    if hook_handle is not None:
        hook_handle.remove()
    
    if representation == "gnn_embedding":
        # Convert captured embeddings to representations
        representations = torch.cat(captured_embeddings, dim=0).numpy()
        embedding_dim = representations.shape[1]
        print(f"GNN embedding dimension: {embedding_dim}")
    
    # Convert to numpy arrays
    if representation == "spectral":
        representations = np.array(representations)
    # For gnn_embedding, already converted above
    costs = np.array(costs)
    
    print(f"Representation shape: {representations.shape}")
    print(f"Cost range: [{costs.min():.4f}, {costs.max():.4f}]")
    
    # Set representation description for titles
    if representation == "gnn_embedding":
        rep_desc = f"GNN Embedding, {representations.shape[1]} dims"
    else:
        rep_desc = f"Laplacian Eigenvalues (k={spectral_k})"
    
    # Project to 2D
    print(f"Projecting to 2D using {projection_method.upper()}...")
    
    if projection_method.lower() == "tsne":
        projector = TSNE(n_components=2, random_state=random_seed, init='pca')
        embedding = projector.fit_transform(representations)
    elif projection_method.lower() == "pca":
        projector = PCA(n_components=2, random_state=random_seed)
        embedding = projector.fit_transform(representations)
        explained_var = projector.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%} (total: {sum(explained_var):.2%})")
    elif projection_method.lower() == "umap":
        try:
            import umap
            projector = umap.UMAP(n_components=2, random_state=random_seed, n_neighbors=15, min_dist=0.1)
            embedding = projector.fit_transform(representations)
        except ImportError:
            print("UMAP not installed. Falling back to t-SNE.")
            projector = TSNE(n_components=2, perplexity=perplexity, random_state=random_seed)
            embedding = projector.fit_transform(representations)
    else:
        raise ValueError(f"Unknown projection method: {projection_method}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot colored by cost
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=costs, cmap='viridis', 
                         s=20, alpha=0.7, edgecolors='none')
    
    # Add colorbar
    #cbar = fig.colorbar(scatter, ax=ax)
    landscape_type = "Cost + Penalty" if include_penalty else "Cost"
    #cbar.set_label(landscape_type, fontsize=12)
    
    # Mark min and max cost points
    min_idx = np.argmin(costs)
    max_idx = np.argmax(costs)
    
    #ax.scatter(embedding[min_idx, 0], embedding[min_idx, 1], c='lime', s=200, marker='*', 
    #           edgecolors='black', linewidths=1.5, zorder=5, label=f'Min {landscape_type}: {costs[min_idx]:.2f}')
    #ax.scatter(embedding[max_idx, 0], embedding[max_idx, 1], c='red', s=200, marker='*', 
   #            edgecolors='black', linewidths=1.5, zorder=5, label=f'Max {landscape_type}: {costs[max_idx]:.2f}')
    
    # Mark left-deep discrete plans with small black stars
    if len(discrete_indices) > 0:
        discrete_embedding = embedding[discrete_indices]
        ax.scatter(discrete_embedding[:, 0], discrete_embedding[:, 1], 
                   c='black', s=50, marker='*', alpha=0.9, zorder=4,
                   label=f'Left-Deep ({len(discrete_indices)})')
    
    # Mark bushy discrete plans with orange diamonds
    if len(bushy_indices) > 0:
        bushy_embedding = embedding[bushy_indices]
        ax.scatter(bushy_embedding[:, 0], bushy_embedding[:, 1], 
                   c='orange', s=50, marker='D', alpha=0.9, zorder=4,
                   edgecolors='black', linewidths=0.5,
                   label=f'Bushy ({len(bushy_indices)})')
    
    # Labels and title
    ax.set_xlabel(f'{projection_method.upper()} Dimension 1', fontsize=30)
    ax.set_ylabel(f'{projection_method.upper()} Dimension 2', fontsize=30)
    total_plans = num_samples + num_discrete + num_bushy
    #ax.set_title(f'{projection_method.upper()} Projection of {total_plans} Plans\n'
    #             f'({num_discrete} left-deep + {num_bushy} bushy + {num_samples} soft, {rep_desc})', fontsize=14)
    #ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    penalty_suffix = "_with_penalty" if include_penalty else ""
    rep_suffix = "_gnn" if representation == "gnn_embedding" else "_spectral"
    filename = f'plan_space_{projection_method.lower()}{rep_suffix}{penalty_suffix}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Projection plot saved as '{filename}'")
    
    # Print statistics
    print(f"\n--- Plan Space Projection Results ---")
    print(f"Total plans: {total_plans} ({num_discrete} left-deep + {num_bushy} bushy + {num_samples} soft)")
    print(f"Representation: {rep_desc}")
    print(f"Projection: {projection_method.upper()}")
    print(f"Min {landscape_type.lower()}: {costs.min():.4f}")
    print(f"Max {landscape_type.lower()}: {costs.max():.4f}")
    print(f"Mean {landscape_type.lower()}: {costs.mean():.4f}")
    print(f"Std {landscape_type.lower()}: {costs.std():.4f}")
    
    # Statistics for discrete vs soft plans
    if len(discrete_indices) > 0:
        discrete_costs = costs[discrete_indices]
        print(f"\nLeft-deep plans - Min: {discrete_costs.min():.4f}, Max: {discrete_costs.max():.4f}, Mean: {discrete_costs.mean():.4f}")
    
    if len(bushy_indices) > 0:
        bushy_costs = costs[bushy_indices]
        print(f"Bushy plans - Min: {bushy_costs.min():.4f}, Max: {bushy_costs.max():.4f}, Mean: {bushy_costs.mean():.4f}")
    
    # Also create a hexbin plot for density visualization
    fig_hex, ax_hex = plt.subplots(figsize=(12, 10))
    
    hb = ax_hex.hexbin(embedding[:, 0], embedding[:, 1], C=costs, gridsize=30, 
                       cmap='viridis', reduce_C_function=np.mean)
    
    cbar_hex = fig_hex.colorbar(hb, ax=ax_hex)
    cbar_hex.set_label(f'Mean {landscape_type}', fontsize=12)
    
    # Mark left-deep discrete plans on hexbin plot
    if len(discrete_indices) > 0:
        ax_hex.scatter(discrete_embedding[:, 0], discrete_embedding[:, 1], 
                       c='black', s=50, marker='*', alpha=0.9, zorder=4,
                       label=f'Left-Deep ({len(discrete_indices)})')
    
    # Mark bushy discrete plans on hexbin plot
    if len(bushy_indices) > 0:
        ax_hex.scatter(bushy_embedding[:, 0], bushy_embedding[:, 1], 
                       c='orange', s=50, marker='D', alpha=0.9, zorder=4,
                       edgecolors='black', linewidths=0.5,
                       label=f'Bushy ({len(bushy_indices)})')
    
    ax_hex.legend(loc='upper right', fontsize=10)
    
    ax_hex.set_xlabel(f'{projection_method.upper()} Dimension 1', fontsize=12)
    ax_hex.set_ylabel(f'{projection_method.upper()} Dimension 2', fontsize=12)
    ax_hex.set_title(f'{projection_method.upper()} Projection with Hexbin Density\n'
                     f'({num_discrete} left-deep + {num_bushy} bushy + {num_samples} soft, {rep_desc})', fontsize=14)
    
    plt.tight_layout()
    
    hexbin_filename = f'plan_space_{projection_method.lower()}_hexbin{rep_suffix}{penalty_suffix}.pdf'
    plt.savefig(hexbin_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Hexbin plot saved as '{hexbin_filename}'")
    
    # ============================================================
    # 3D PROJECTION
    # ============================================================
    print(f"\nProjecting to 3D using {projection_method.upper()}...")
    
    if projection_method.lower() == "tsne":
        projector_3d = TSNE(n_components=3, random_state=random_seed, init='pca')
        embedding_3d = projector_3d.fit_transform(representations)
    elif projection_method.lower() == "pca":
        projector_3d = PCA(n_components=3, random_state=random_seed)
        embedding_3d = projector_3d.fit_transform(representations)
        explained_var_3d = projector_3d.explained_variance_ratio_
        print(f"PCA 3D explained variance: {explained_var_3d[0]:.2%}, {explained_var_3d[1]:.2%}, {explained_var_3d[2]:.2%} (total: {sum(explained_var_3d):.2%})")
    elif projection_method.lower() == "umap":
        try:
            import umap
            projector_3d = umap.UMAP(n_components=3, random_state=random_seed, n_neighbors=15, min_dist=0.1)
            embedding_3d = projector_3d.fit_transform(representations)
        except ImportError:
            print("UMAP not installed. Falling back to t-SNE for 3D.")
            projector_3d = TSNE(n_components=3, random_state=random_seed, init='pca')
            embedding_3d = projector_3d.fit_transform(representations)
    else:
        raise ValueError(f"Unknown projection method: {projection_method}")
    
    # Create 3D visualization
    fig_3d = plt.figure(figsize=(14, 11))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Scatter plot colored by cost
    scatter_3d = ax_3d.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], 
                                c=costs, cmap='viridis', s=15, alpha=0.6, edgecolors='none')
    
    # Add colorbar
    cbar_3d = fig_3d.colorbar(scatter_3d, ax=ax_3d, shrink=0.6, aspect=15)
    cbar_3d.set_label(landscape_type, fontsize=12)
    
    # Mark min and max cost points
    ax_3d.scatter(embedding_3d[min_idx, 0], embedding_3d[min_idx, 1], embedding_3d[min_idx, 2], 
                  c='lime', s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=5,
                  label=f'Min {landscape_type}: {costs[min_idx]:.2f}')
    ax_3d.scatter(embedding_3d[max_idx, 0], embedding_3d[max_idx, 1], embedding_3d[max_idx, 2], 
                  c='red', s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=5,
                  label=f'Max {landscape_type}: {costs[max_idx]:.2f}')
    
    # Mark left-deep discrete plans with small black stars
    if len(discrete_indices) > 0:
        discrete_embedding_3d = embedding_3d[discrete_indices]
        ax_3d.scatter(discrete_embedding_3d[:, 0], discrete_embedding_3d[:, 1], discrete_embedding_3d[:, 2],
                      c='black', s=60, marker='*', alpha=0.9, zorder=4,
                      label=f'Left-Deep ({len(discrete_indices)})')
    
    # Mark bushy discrete plans with orange diamonds
    if len(bushy_indices) > 0:
        bushy_embedding_3d = embedding_3d[bushy_indices]
        ax_3d.scatter(bushy_embedding_3d[:, 0], bushy_embedding_3d[:, 1], bushy_embedding_3d[:, 2],
                      c='orange', s=60, marker='D', alpha=0.9, zorder=4,
                      label=f'Bushy ({len(bushy_indices)})')
    
    # Labels and title
    ax_3d.set_xlabel(f'{projection_method.upper()} Dim 1', fontsize=11)
    ax_3d.set_ylabel(f'{projection_method.upper()} Dim 2', fontsize=11)
    ax_3d.set_zlabel(f'{projection_method.upper()} Dim 3', fontsize=11)
    ax_3d.set_title(f'3D {projection_method.upper()} Projection of {total_plans} Plans\n'
                    f'({num_discrete} left-deep + {num_bushy} bushy + {num_samples} soft, {rep_desc})', fontsize=14)
    ax_3d.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    filename_3d = f'plan_space_{projection_method.lower()}_3d{rep_suffix}{penalty_suffix}.pdf'
    plt.savefig(filename_3d, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"3D projection plot saved as '{filename_3d}'")
    
    return embedding, costs, representations, adjacencies, discrete_indices, bushy_indices


def create_interactive_plan_explorer(
    embedding,
    costs,
    adjacencies,
    discrete_indices,
    bushy_indices,
    query_size: int,
    projection_method: str = "tsne",
    include_penalty: bool = False,
):
    """
    Create an interactive plot where clicking on points shows the adjacency matrix.
    
    Args:
        embedding: 2D projected coordinates (N, 2)
        costs: Cost values for each plan
        adjacencies: List of adjacency matrices
        discrete_indices: Indices of left-deep discrete plans
        bushy_indices: Indices of bushy discrete plans
        query_size: Number of triples in the query
        projection_method: Name of projection method (for title)
        include_penalty: Whether costs include penalty
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    
    N_NODES = 2 * query_size - 1
    costs = np.array(costs)
    
    # Create figure with two subplots: scatter plot and adjacency heatmap
    fig = plt.figure(figsize=(16, 8))
    
    # Main scatter plot
    ax_scatter = fig.add_subplot(1, 2, 1)
    
    # Scatter plot colored by cost
    scatter = ax_scatter.scatter(embedding[:, 0], embedding[:, 1], c=costs, cmap='viridis', 
                                  s=20, alpha=0.7, edgecolors='none', picker=True, pickradius=5)
    
    # Add colorbar
    landscape_type = "Cost + Penalty" if include_penalty else "Cost"
    cbar = fig.colorbar(scatter, ax=ax_scatter)
    cbar.set_label(landscape_type, fontsize=12)
    
    # Mark left-deep discrete plans
    if len(discrete_indices) > 0:
        discrete_embedding = embedding[discrete_indices]
        ax_scatter.scatter(discrete_embedding[:, 0], discrete_embedding[:, 1], 
                           c='black', s=50, marker='*', alpha=0.9, zorder=4, picker=True,
                           label=f'Left-Deep ({len(discrete_indices)})')
    
    # Mark bushy discrete plans
    if len(bushy_indices) > 0:
        bushy_embedding = embedding[bushy_indices]
        ax_scatter.scatter(bushy_embedding[:, 0], bushy_embedding[:, 1], 
                           c='orange', s=50, marker='D', alpha=0.9, zorder=4, picker=True,
                           edgecolors='black', linewidths=0.5,
                           label=f'Bushy ({len(bushy_indices)})')
    
    ax_scatter.set_xlabel(f'{projection_method.upper()} Dimension 1', fontsize=12)
    ax_scatter.set_ylabel(f'{projection_method.upper()} Dimension 2', fontsize=12)
    ax_scatter.set_title(f'Click on a point to see its adjacency matrix', fontsize=14)
    ax_scatter.legend(loc='upper right', fontsize=10)
    
    # Adjacency matrix heatmap (initially empty)
    ax_adj = fig.add_subplot(1, 2, 2)
    
    # Initial placeholder
    placeholder = np.zeros((N_NODES, N_NODES))
    heatmap = ax_adj.imshow(placeholder, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax_adj.set_title('Click on a point to view adjacency', fontsize=14)
    ax_adj.set_xlabel('Target Node')
    ax_adj.set_ylabel('Source Node')
    
    # Add grid lines
    ax_adj.set_xticks(np.arange(-0.5, N_NODES, 1), minor=True)
    ax_adj.set_yticks(np.arange(-0.5, N_NODES, 1), minor=True)
    ax_adj.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Add node labels
    triple_labels = [f'T{i}' for i in range(query_size)]
    join_labels = [f'J{i}' for i in range(query_size - 1)]
    all_labels = triple_labels + join_labels
    ax_adj.set_xticks(range(N_NODES))
    ax_adj.set_yticks(range(N_NODES))
    ax_adj.set_xticklabels(all_labels, fontsize=8)
    ax_adj.set_yticklabels(all_labels, fontsize=8)
    
    # Add colorbar for heatmap
    cbar_adj = fig.colorbar(heatmap, ax=ax_adj)
    cbar_adj.set_label('Edge Weight', fontsize=10)
    
    # Text annotation for plan info
    info_text = ax_adj.text(0.5, -0.15, '', transform=ax_adj.transAxes, 
                            fontsize=10, ha='center', va='top')
    
    # Store selected point marker
    selected_marker = [None]
    
    def on_pick(event):
        """Handle click events on scatter points."""
        if event.artist != scatter:
            return
        
        # Get index of clicked point
        ind = event.ind[0]
        
        # Get adjacency matrix
        A = adjacencies[ind]
        if hasattr(A, 'numpy'):
            A_np = A.numpy()
        else:
            A_np = np.array(A)
        
        # Update heatmap
        heatmap.set_data(A_np)
        heatmap.set_clim(vmin=0, vmax=max(1, A_np.max()))
        
        # Determine plan type
        if ind in discrete_indices:
            plan_type = "LEFT-DEEP DISCRETE"
        elif ind in bushy_indices:
            plan_type = "BUSHY DISCRETE"
        else:
            plan_type = "SOFT"
        
        # Compute some statistics about the adjacency
        out_degrees = A_np.sum(axis=1)
        in_degrees = A_np.sum(axis=0)
        max_edge = A_np.max()
        num_nonzero = (A_np > 0.01).sum()
        
        # Check if near-discrete (max edge weight close to 1)
        discreteness = A_np[A_np > 0.01].max() if (A_np > 0.01).any() else 0
        
        # Update title and info
        ax_adj.set_title(f'Plan #{ind} - {plan_type}\n{landscape_type}: {costs[ind]:.4f}', fontsize=12)
        info_text.set_text(f'Max edge: {max_edge:.3f} | Non-zero edges: {num_nonzero} | Discreteness: {discreteness:.3f}')
        
        # Update selected point marker
        if selected_marker[0] is not None:
            selected_marker[0].remove()
        selected_marker[0] = ax_scatter.scatter([embedding[ind, 0]], [embedding[ind, 1]], 
                                                 c='red', s=200, marker='o', facecolors='none',
                                                 edgecolors='red', linewidths=3, zorder=10)
        
        # Print adjacency to console for detailed inspection
        print(f"\n{'='*50}")
        print(f"Plan #{ind} - {plan_type}")
        print(f"{landscape_type}: {costs[ind]:.4f}")
        print(f"Max edge weight: {max_edge:.4f}")
        print(f"Non-zero edges (>0.01): {num_nonzero}")
        print(f"\nOut-degrees (sum of rows): {out_degrees.round(2)}")
        print(f"In-degrees (sum of cols): {in_degrees.round(2)}")
        print(f"\nAdjacency Matrix:")
        print(A_np.round(2))
        print(f"{'='*50}")
        
        fig.canvas.draw_idle()
    
    # Connect the pick event
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    plt.tight_layout()
    plt.show()
    
    print("\nInteractive plot ready! Click on any point to see its adjacency matrix.")
    print("Plan details will be printed to the console.")
    
    return fig


def optimize_query_with_trajectory_tracking(
    query_data,
    model,
    A_base, A_alpha, A_beta,  # Reference adjacency matrices for projection
    edge_index,
    device: str = "cpu",
    *,
    optimization_steps: int = 500,
    verbose: bool = True,
    learning_rate: float = 0.01,
    lambda_acyclic: float = 1000.0,
    lambda_triple_in: float = 1000.0,
    lambda_triple_out: float = 1000.0,
    lambda_join_in: float = 500.0,
    lambda_join_out: float = 1000.0,
    lambda_entropy: float = 10.0,
    lambda_total_penalty: float = 1.0,
    lambda_left_linear: float = 1000.0,
    init_tau: float = 10.0,
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    min_penalty_threshold: float = 30.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = 'dual-softmax',
    trajectory_save_interval: int = 1,
    gradient_clip_norm: float = 5.0,
    use_lr_scheduling: bool = True,
    lr_warmup_steps: int = 200,
    use_gumbel_noise: bool = True,
):
    """
    Modified optimization function that tracks trajectory in the (α, β) space.
    Updated to match the latest optimization logic from methods.py.
    """
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2
    
    num_edges = edge_index.size(1)
    
    # Initialize edge logits (matching methods.py initialization)
    #edge_logits = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)
    #edge_logits_slot2 = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device) # TODO

    edge_logits = torch.zeros(num_edges, requires_grad=True, device=device)
    edge_logits_slot2 = torch.zeros(num_edges, requires_grad=True, device=device)
    
    # Optimizer
    if logit_sampling == 'dual-softmax':
        #optimiser = optim.AdamW([edge_logits, edge_logits_slot2], lr=learning_rate)
        optimiser = optim.RAdam([edge_logits, edge_logits_slot2], lr=learning_rate)
        #optimiser = optim_extra.Lookahead(optimiser, k=10, alpha=0.5)
        #optimiser = optim.SGD([edge_logits, edge_logits_slot2], lr=learning_rate, momentum=0.9)
    else:
        optimiser = optim.AdamW([edge_logits], lr=learning_rate)
    
    # Optional Learning rate scheduler for warmup and decay
    if use_lr_scheduling:
        def lr_schedule(step):
            if step < lr_warmup_steps:
                if lr_warmup_steps == 0:
                    return 1
                else:
                    return (step + 1) / lr_warmup_steps
            else:
                return 1
        
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_schedule)
    
    # Trajectory tracking
    trajectory_alphas = []
    trajectory_betas = []
    trajectory_costs = []
    trajectory_steps = []
    
    for step in range(optimization_steps):
        optimiser.zero_grad()
        
        # Temperature annealing
        if use_temperature_annealing:
            tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps)
        else:
            tau = init_tau
        
        # Sample edge weights based on method
        if logit_sampling == 'dual-softmax':
            masked_logits_1 = edge_logits.clone()
            masked_logits_2 = edge_logits_slot2.clone()
            
            # Invalid edge types are masked out
            triple_to_triple = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits_1[triple_to_triple] = float('-inf')
            masked_logits_2[triple_to_triple] = float('-inf')
            join_to_triple = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits_1[join_to_triple] = float('-inf')
            masked_logits_2[join_to_triple] = float('-inf')
            
            join_target_mask = (edge_index[1] >= triples_num)
            slot1 = torch.zeros_like(edge_logits)
            slot2 = torch.zeros_like(edge_logits)
            
            # Sample only on join targets to avoid NaNs for empty groups
            slot1[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits_1[join_target_mask], edge_index[1][join_target_mask], tau, use_gumbel_noise)
            slot2[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits_2[join_target_mask], edge_index[1][join_target_mask], tau, use_gumbel_noise)
            
            edge_weights = slot1 + slot2  # relaxed 2-hot (values in (0,2))
            # Ensure root join has no outgoing edges (w.l.o.g.)
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        elif logit_sampling == 'softmax':
            masked_logits = edge_logits.clone()
            triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits[triple_to_triple_mask] = float('-inf')
            join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits[join_to_triple_mask] = float('-inf')
            edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], tau, use_gumbel_noise)
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        else:
            # Use Binary Concrete (Gumbel-Sigmoid) sampling
            edge_weights = sample_binary_concrete(edge_logits, tau)
        
        # Build current adjacency matrix
        A_current = torch.zeros((N_NODES, N_NODES), device=device)
        A_current[edge_index[0], edge_index[1]] = edge_weights
        
        # Project current adjacency onto the (α, β) coordinate system
        if step % trajectory_save_interval == 0:
            with torch.no_grad():
                A_current_flat = A_current.flatten()
                A_base_flat = A_base.flatten().to(device)
                A_alpha_flat = A_alpha.flatten().to(device)
                A_beta_flat = A_beta.flatten().to(device)
                
                def compute_interpolated_A(alpha, beta):
                    return ((1 - alpha) * (1 - beta) * A_base_flat +
                            alpha * (1 - beta) * A_alpha_flat +
                            (1 - alpha) * beta * A_beta_flat +
                            alpha * beta * (0.5 * A_alpha_flat + 0.5 * A_beta_flat))
                
                # Grid search for best α, β
                best_alpha, best_beta = 0.0, 0.0
                min_error = float('inf')
                
                for alpha_test in np.linspace(0, 1, 11):
                    for beta_test in np.linspace(0, 1, 11):
                        interpolated = compute_interpolated_A(alpha_test, beta_test)
                        error = torch.sum((A_current_flat - interpolated) ** 2).item()
                        if error < min_error:
                            min_error = error
                            best_alpha, best_beta = alpha_test, beta_test
                
                trajectory_alphas.append(best_alpha)
                trajectory_betas.append(best_beta)
                trajectory_steps.append(step)
        
        # Cost prediction
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)
        
        if step % trajectory_save_interval == 0:
            trajectory_costs.append(cost_pred.item())
        
        # Compute penalties (matching methods.py)
        A = A_current
        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)
        
        # Structural penalties
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES
        
        # Left-linear penalty
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)
        child_join_counts = A[join_nodes, :][:, join_nodes].sum(0)
        
        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)
        
        # Entropy penalty
        if logit_sampling == 'dual-softmax':
            eps = 1e-10
            probs1 = slot1.clamp(min=eps)
            probs2 = slot2.clamp(min=eps)
            P_entropy = -(probs1 * torch.log(probs1) + probs2 * torch.log(probs2)).sum()
        elif logit_sampling == 'softmax':
            eps = 1e-10
            probs = edge_weights.clamp(min=eps)
            P_entropy = -(probs * torch.log(probs)).sum()
        else:
            eps = 1e-10
            probs = torch.sigmoid(edge_logits)
            P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()
        
        # Total penalty
        total_penalty = (
            lambda_triple_in * P_triple_in +
            lambda_triple_out * P_triple_out +
            lambda_join_in * P_join_in +
            lambda_join_out * P_join_out +
            lambda_acyclic * P_acyclic +
            lambda_entropy * P_entropy +
            lambda_left_linear * P_left_linear
        )
        
        # Lambda ramping (matching methods.py)
        if use_lambda_ramping:
            frac = min(1.0, step / optimization_steps)
            lambda_total = lambda_total_penalty * (frac ** lambda_ramp_exponent)
        else:
            lambda_total = lambda_total_penalty
        
        loss = cost_pred + lambda_total * total_penalty
        
        # Backprop
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm > 0:
            if logit_sampling == 'dual-softmax':
                params_to_clip = [edge_logits, edge_logits_slot2]
            else:
                params_to_clip = [edge_logits]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=gradient_clip_norm)
        
        optimiser.step()
        
        # Update learning rate schedule
        if use_lr_scheduling:
            scheduler.step()
        
        if verbose and (step + 1) % 100 == 0:
            current_lr = optimiser.param_groups[0]['lr']
            print(f"Step {step+1}/{optimization_steps} Cost: {cost_pred.item():.2f} Penalty: {total_penalty.item():.2f} LR: {current_lr:.6f}")
    
    return {
        'trajectory_alphas': trajectory_alphas,
        'trajectory_betas': trajectory_betas,
        'trajectory_costs': trajectory_costs,
        'trajectory_steps': trajectory_steps
    }


def visualize_optimization_trajectory_3d(query_file, model_path, config, device='cpu', landscape_resolution=25, include_penalty=False, clean_plot=False):
    """
    Visualize the 3D cost landscape with optimization trajectory overlay.
    Shows how the optimization moves through the space between different join plans.
    
    Args:
        include_penalty: If True, visualize cost + penalty landscape instead of just cost
        clean_plot: If True, remove all axes, grids, and labels for minimal visualization
    """
    # Load model
    #model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    #model.eval()

    # elsenif using v3
    model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6, use_jk=False, jk_mode='cat', use_residual=False, use_layer_norm=True, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    
    # Load query
    queries = load_sparql_queries(query_file, 100)
    queries = [q for q in queries if len(q.triples) == 3]
    query_data = queries[1].torch_data[0]
    
    # Reference adjacency matrices
    A_base = left_deep_adj_from_perm(torch.tensor([0, 1, 2]))   # "1 JOIN 2 JOIN 3"
    A_alpha = left_deep_adj_from_perm(torch.tensor([0, 2, 1]))  # "1 JOIN 3 JOIN 2"
    A_beta = left_deep_adj_from_perm(torch.tensor([1, 2, 0]))   # "2 JOIN 3 JOIN 1"
    
    # Create edge_index
    N_NODES = len(query_data.x)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    
    # Extract penalty config from main config
    penalty_config = {
        'lambda_acyclic': config['optimization_params']['lambda_acyclic'],
        'lambda_triple_in': config['optimization_params']['lambda_triple_in'],
        'lambda_triple_out': config['optimization_params']['lambda_triple_out'],
        'lambda_join_in': config['optimization_params']['lambda_join_in'],
        'lambda_join_out': config['optimization_params']['lambda_join_out'],
        'lambda_left_linear': config['optimization_params']['lambda_left_linear'],
        'lambda_entropy': config['optimization_params']['lambda_entropy'],
    }
    
    def compute_penalty(A, edge_weights):
        """Compute structural penalty for given adjacency matrix and edge weights"""
        triples_num = (N_NODES + 1) // 2
        
        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)
        
        # Structural penalties
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES
        
        # Left-linear penalty
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)
        child_join_counts = A[join_nodes, :][:, join_nodes].sum(0)
        
        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)
        
        # Entropy penalty - computed on adjacency matrix A
        eps = 1e-10
        probs = A.clamp(min=eps, max=1-eps)
        P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()
        
        # Total penalty
        total_penalty = (
            penalty_config['lambda_triple_in'] * P_triple_in +
            penalty_config['lambda_triple_out'] * P_triple_out +
            penalty_config['lambda_join_in'] * P_join_in +
            penalty_config['lambda_join_out'] * P_join_out +
            penalty_config['lambda_acyclic'] * P_acyclic +
            penalty_config['lambda_left_linear'] * P_left_linear +
            penalty_config['lambda_entropy'] * P_entropy
        )

        return total_penalty
    
    # First, create the cost landscape
    landscape_type = "cost + penalty" if include_penalty else "cost"
    print(f"Computing {landscape_type} landscape...")
    alphas = np.linspace(0, 1, landscape_resolution)
    betas = np.linspace(0, 1, landscape_resolution)
    Alpha, Beta = np.meshgrid(alphas, betas)
    Cost = np.zeros_like(Alpha)
    
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                A_interp = (
                    (1 - alpha) * (1 - beta) * A_base +
                    alpha * (1 - beta) * A_alpha +
                    (1 - alpha) * beta * A_beta +
                    alpha * beta * (0.5 * A_alpha + 0.5 * A_beta)
                )
                edge_weights = A_interp[edge_index[0], edge_index[1]].to(device)
                cost = model(query_data.x, edge_index, edge_weight=edge_weights)
                
                if include_penalty:
                    penalty = compute_penalty(A_interp.to(device), edge_weights)
                    Cost[j, i] = cost.item() + 1 * penalty.item()
                else:
                    Cost[j, i] = cost.item()
    
    # Now run optimization with trajectory tracking
    print("Running optimization with trajectory tracking...")
    trajectory_data = optimize_query_with_trajectory_tracking(
        query_data, model, A_base, A_alpha, A_beta, edge_index, device,
        **config['optimization_params'],
        optimization_steps=config['optimization_steps'],
        verbose=config['verbose']
    )
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=40, azim=0, roll=15)

    
    # Plot cost surface
    surf = ax.plot_surface(Alpha, Beta, Cost, cmap='viridis', alpha=0.6, 
                          linewidth=0, antialiased=True)
    
    # Plot optimization trajectory
    traj_alphas = np.array(trajectory_data['trajectory_alphas'])
    traj_betas = np.array(trajectory_data['trajectory_betas'])
    traj_costs = np.array(trajectory_data['trajectory_costs'])
    
    # If including penalty, we need to recompute trajectory costs with penalty
    if include_penalty:
        traj_costs_with_penalty = []
        with torch.no_grad():
            for i, (alpha, beta) in enumerate(zip(traj_alphas, traj_betas)):
                A_interp = (
                    (1 - alpha) * (1 - beta) * A_base +
                    alpha * (1 - beta) * A_alpha +
                    (1 - alpha) * beta * A_beta +
                    alpha * beta * (0.5 * A_alpha + 0.5 * A_beta)
                )
                edge_weights = A_interp[edge_index[0], edge_index[1]].to(device)
                cost = model(query_data.x, edge_index, edge_weight=edge_weights)
                penalty = compute_penalty(A_interp.to(device), edge_weights)
                traj_costs_with_penalty.append(cost.item() + 1* penalty.item())
        traj_costs = np.array(traj_costs_with_penalty)
    
    # Plot trajectory line
    ax.plot(traj_alphas, traj_betas, traj_costs, 'r-', linewidth=3, alpha=0.8, label='Optimization path')
    
    # Mark start and end points
    ax.scatter([traj_alphas[0]], [traj_betas[0]], [traj_costs[0]], 
              c='green', s=150, marker='o', label='Start', alpha=1.0)
    ax.scatter([traj_alphas[-1]], [traj_betas[-1]], [traj_costs[-1]], 
              c='red', s=150, marker='*', label='End', alpha=1.0)
    
    # Mark corner plans
    #ax.scatter([0, 1, 0], [0, 0, 1], [Cost[0,0], Cost[0,-1], Cost[-1,0]], 
    #          c='blue', s=100, alpha=1.0, marker='s', label='Reference plans')
    
    if clean_plot:
        # Remove all axes, grids, and labels for minimal visualization
        ax.set_axis_off()
        filename_suffix = '_clean'
    else:
        # Normal plot with labels and annotations
        ax.set_xlabel('α → "1 JOIN 3 JOIN 2"', fontsize=11)
        ax.set_ylabel('β → "2 JOIN 3 JOIN 1"', fontsize=11)
        
        if include_penalty:
            ax.set_zlabel('Predicted Cost + Penalty', fontsize=11)
            ax.set_title('Optimization Trajectory in 3D Cost + Penalty Landscape', fontsize=14)
        else:
            ax.set_zlabel('Predicted Cost', fontsize=11)
            ax.set_title('Optimization Trajectory in 3D Cost Landscape', fontsize=14)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add text annotations
        ax.text(0, 0, Cost[0,0], '  1→2→3', fontsize=9)
        ax.text(1, 0, Cost[0,-1], '  1→3→2', fontsize=9)
        ax.text(0, 1, Cost[-1,0], '  2→3→1', fontsize=9)
        filename_suffix = ''
    
    # Add colorbar only if not clean plot
    if not clean_plot:
        fig.colorbar(surf, shrink=0.5, aspect=10)
    
    # Set filename
    if include_penalty:
        filename = f'optimization_trajectory_cost_penalty_3d{filename_suffix}.png'
    else:
        filename = f'optimization_trajectory_3d{filename_suffix}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print trajectory statistics
    landscape_type = "cost + penalty" if include_penalty else "cost"
    print(f"\nTrajectory Statistics ({landscape_type}):")
    print(f"Start: α={traj_alphas[0]:.3f}, β={traj_betas[0]:.3f}, {landscape_type}={traj_costs[0]:.4f}")
    print(f"End: α={traj_alphas[-1]:.3f}, β={traj_betas[-1]:.3f}, {landscape_type}={traj_costs[-1]:.4f}")
    print(f"{landscape_type.capitalize()} improvement: {traj_costs[0] - traj_costs[-1]:.4f}")
    print(f"Trajectory length: {len(traj_alphas)} points")
    
    # Create contour plot with trajectory
    fig_contour, ax_contour = plt.subplots(figsize=(10, 8))
    
    # Filled contour plot
    contour_filled = ax_contour.contourf(Alpha, Beta, Cost, levels=40, cmap='viridis')
    
    # Add contour lines
    contour_lines = ax_contour.contour(Alpha, Beta, Cost, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax_contour.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot trajectory on contour
    ax_contour.plot(traj_alphas, traj_betas, 'r-', linewidth=2, alpha=0.8, label='Optimization path')
    ax_contour.scatter([traj_alphas[0]], [traj_betas[0]], c='green', s=100, marker='o', 
                       zorder=6, edgecolors='white', linewidths=2, label='Start')
    ax_contour.scatter([traj_alphas[-1]], [traj_betas[-1]], c='red', s=150, marker='*', 
                       zorder=6, edgecolors='white', linewidths=1, label='End')
    
    # Mark the three corner plans
    ax_contour.scatter([0, 1, 0], [0, 0, 1], c='blue', s=80, zorder=5, edgecolors='white', linewidths=2, marker='s')
    ax_contour.annotate('1→2→3', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=10, color='white')
    ax_contour.annotate('1→3→2', (1, 0), textcoords="offset points", xytext=(-50, 10), fontsize=10, color='white')
    ax_contour.annotate('2→3→1', (0, 1), textcoords="offset points", xytext=(10, -15), fontsize=10, color='white')
    
    # Mark minimum point
    min_idx = np.unravel_index(Cost.argmin(), Cost.shape)
    min_alpha = alphas[min_idx[1]]
    min_beta = betas[min_idx[0]]
    ax_contour.scatter([min_alpha], [min_beta], c='yellow', s=150, marker='*', zorder=7, edgecolors='black', linewidths=1)
    ax_contour.annotate(f'Min: {Cost.min():.2f}', (min_alpha, min_beta), 
                       textcoords="offset points", xytext=(10, 10), fontsize=9, color='yellow')
    
    # Labels and title
    ax_contour.set_xlabel('α → "1 JOIN 3 JOIN 2"', fontsize=12)
    ax_contour.set_ylabel('β → "2 JOIN 3 JOIN 1"', fontsize=12)
    ax_contour.legend(loc='upper right')
    
    if include_penalty:
        ax_contour.set_title('Optimization Trajectory on Cost + Penalty Contour', fontsize=14)
        contour_filename = 'optimization_trajectory_cost_penalty_contour.pdf'
    else:
        ax_contour.set_title('Optimization Trajectory on Cost Contour', fontsize=14)
        contour_filename = 'optimization_trajectory_contour.pdf'
    
    # Add colorbar
    cbar = fig_contour.colorbar(contour_filled, ax=ax_contour)
    cbar.set_label(landscape_type.capitalize(), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(contour_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Contour plot with trajectory saved as '{contour_filename}'")
    
    # Create clean contour plot with trajectory (surface only, no annotations)
    fig_clean, ax_clean = plt.subplots(figsize=(10, 8))
    
    # Filled contour plot only
    ax_clean.contourf(Alpha, Beta, Cost, levels=30, cmap='viridis')
    
    # Add subtle gray contour lines
    ax_clean.contour(Alpha, Beta, Cost, levels=30, colors='#404040', alpha=0.4, linewidths=0.5)
    
    # Plot trajectory on clean contour
    ax_clean.plot(traj_alphas, traj_betas, 'r-', linewidth=2.5, alpha=0.9)
    ax_clean.scatter([traj_alphas[0]], [traj_betas[0]], c='green', s=120, marker='o', 
                    zorder=6, edgecolors='white', linewidths=2)
    ax_clean.scatter([traj_alphas[-1]], [traj_betas[-1]], c='red', s=180, marker='*', 
                    zorder=6, edgecolors='white', linewidths=1)
    
    # Remove all axes, ticks, and borders
    ax_clean.set_axis_off()
    
    # Set filename
    if include_penalty:
        clean_contour_filename = 'optimization_trajectory_cost_penalty_contour_clean.pdf'
    else:
        clean_contour_filename = 'optimization_trajectory_contour_clean.pdf'
    
    plt.tight_layout()
    plt.savefig(clean_contour_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print(f"Clean contour plot with trajectory saved as '{clean_contour_filename}'")


if __name__ == "__main__":
    # Configuration (matching the latest methods.py config)
    config = {
        'optimization_steps': 500,
        'verbose': False,
        'optimization_params': {
            'learning_rate': 1.8,
            'lambda_acyclic': 3081.0,
            'lambda_triple_in': 3714.0,
            'lambda_triple_out': 135.0,
            'lambda_join_in': 1742.0,
            'lambda_join_out': 1558.0,
            'lambda_entropy': 1.0,
            'lambda_total_penalty': 2.6,
            'lambda_left_linear': 2300.0,
            'init_tau': 4.5,
            'min_tau': 1.0,
            'tau_decay': 0.963,
            'use_temperature_annealing': True,
            'min_penalty_threshold': 30.0,
            'use_lambda_ramping': True,
            'lambda_ramp_exponent': 6.5,
            'logit_sampling': 'dual-softmax',
            'trajectory_save_interval': 1,
            'gradient_clip_norm': 2,
            'use_lr_scheduling': True,
            'use_gumbel_noise': False,
            'lr_warmup_steps': 50,
        }
    }
    

    query_file = ".../datasets/plans/wikidata_star_plan_datasets_optimization/queries.pkl"
    model_path = ".../training_results/wikidata-star-log1p-add-aggr/model.pt"
    
    # Visualization options
    show_penalty_landscape = False  # Toggle to include penalty in landscape

    query_size = 5

    # Run the projection first
    embedding, costs, reps, adjs, discrete_indices, bushy_indices = visualize_plan_space_projection(
        query_file=query_file,
        model_path=model_path,
        device="cpu",
        num_samples=10000,
        include_penalty=False,
        projection_method="umap",
        query_size=query_size,
    )

    exit()

    # Then create the interactive plot
    create_interactive_plan_explorer(
        embedding=embedding,
        costs=costs,
        adjacencies=adjs,
        discrete_indices=discrete_indices,
        bushy_indices=bushy_indices,
        query_size=query_size,  # Set to your query's number of triples
        projection_method="umap",
        include_penalty=True,
    )
        
    # Choose which visualization to run:
    #visualize_cost_transition(query_file, model_path)  # 2D version
    #visualize_cost_landscape_3d(query_file, model_path, include_penalty=show_penalty_landscape)  # 3D version
    #visualize_optimization_trajectory_3d(query_file, model_path, config, include_penalty=show_penalty_landscape, clean_plot=True)  # 3D with trajectory
