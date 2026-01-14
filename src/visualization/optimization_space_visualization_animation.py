import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
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


def visualize_cost_transition(query_file, model_path, device='cpu', num_steps=100):
    """
    Visualize how predicted cost changes when transitioning from 
    plan '1 JOIN 2 JOIN 3' to plan '1 JOIN 3 JOIN 2' for a 3-triple query.
    """
    # Load model
    model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
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


def visualize_cost_landscape_3d(query_file, model_path, device='cpu', num_steps=80, include_penalty=False, penalty_config=None):
    """
    Visualize the 3D cost landscape when interpolating between three join plans:
    - Base plan: "1 JOIN 2 JOIN 3" [0, 1, 2]
    - α direction: towards "1 JOIN 3 JOIN 2" [0, 2, 1]
    - β direction: towards "2 JOIN 3 JOIN 1" [1, 2, 0]
    
    Args:
        include_penalty: If True, visualize cost + penalty landscape instead of just cost
        penalty_config: Dict with penalty weights (lambda values)
    """
    # Load model
    #model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6, use_jk=False, jk_mode='cat', use_residual=False, use_layer_norm=False, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query (assuming single query with 3 triples)
    queries = load_sparql_queries(query_file, 100)
    queries = [q for q in queries if len(q.triples) == 3]
    query_data = queries[8].torch_data[0]
    
    # Create adjacency matrices for three plans
    A_base = left_deep_adj_from_perm(torch.tensor([0, 1, 2]))  # "1 JOIN 2 JOIN 3"
    A_alpha = left_deep_adj_from_perm(torch.tensor([0, 2, 1]))  # "1 JOIN 3 JOIN 2"
    A_beta = left_deep_adj_from_perm(torch.tensor([1, 2, 0]))   # "2 JOIN 3 JOIN 1"
    
    # Create edge_index for all possible edges
    N_NODES = len(query_data.x)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    
    # Set up penalty calculation if needed
    if include_penalty and penalty_config is None:
        # Use default penalty config
        penalty_config = {
            'lambda_acyclic': 2065.0,
            'lambda_triple_in': 2390.0,
            'lambda_triple_out': 105.0,
            'lambda_join_in': 387.0,
            'lambda_join_out': 2610.0,
            'lambda_left_linear': 3290.0,
            'lambda_entropy': 1000.0,
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
            print(f"P_acyclic: {P_acyclic.item():.4f}")
            print(f"P_triple_in: {P_triple_in.item():.4f}")
            print(f"P_triple_out: {P_triple_out.item():.4f}")
            print(f"P_join_in: {P_join_in.item():.4f}")
            print(f"P_join_out: {P_join_out.item():.4f}")
            print(f"P_left_linear: {P_left_linear.item():.4f}")
            print(f"P_entropy: {P_entropy.item():.4f}")
        
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
                    alpha * beta * (0.5 * A_alpha + 0.5 * A_beta)  # Average at corner
                )
                
                # Extract edge weights from interpolated adjacency matrix
                edge_weights = A_interp[edge_index[0], edge_index[1]].to(device)
                
                # Predict cost
                cost = model(query_data.x, edge_index, edge_weight=edge_weights)
                
                if include_penalty:
                    penalty = compute_penalty(A_interp.to(device), edge_weights)
                    Cost[j, i] = cost.item() + 0.0001 * penalty.item()
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
    contour_filled = ax_contour.contourf(Alpha, Beta, Cost, levels=40, cmap=cmap)
    
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
    ax_clean.contourf(Alpha, Beta, Cost, levels=30, cmap=cmap)
    
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


# =============================================================================
# ANIMATION FUNCTIONS FOR EPOCH-WISE COST LANDSCAPE VISUALIZATION
# =============================================================================

def compute_cost_landscape_for_model(model, query_data, device='cpu', num_steps=50):
    """
    Compute cost landscape grid for a given model.
    Returns the Alpha, Beta meshgrids, Cost grid, and alpha/beta arrays.
    """
    # Create adjacency matrices for three plans
    A_base = left_deep_adj_from_perm(torch.tensor([0, 1, 2]))   # "1 JOIN 2 JOIN 3"
    A_alpha = left_deep_adj_from_perm(torch.tensor([0, 2, 1]))  # "1 JOIN 3 JOIN 2"
    A_beta = left_deep_adj_from_perm(torch.tensor([1, 2, 0]))   # "2 JOIN 3 JOIN 1"
    
    # Create edge_index for all possible edges
    N_NODES = len(query_data.x)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    
    # Create grid of interpolation parameters
    alphas = np.linspace(0, 1, num_steps)
    betas = np.linspace(0, 1, num_steps)
    Alpha, Beta = np.meshgrid(alphas, betas)
    
    # Initialize cost surface
    Cost = np.zeros_like(Alpha)
    
    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Bilinear interpolation between three plans
                A_interp = (
                    (1 - alpha) * (1 - beta) * A_base +
                    alpha * (1 - beta) * A_alpha +
                    (1 - alpha) * beta * A_beta +
                    alpha * beta * (0.5 * A_alpha + 0.5 * A_beta)
                )
                
                edge_weights = A_interp[edge_index[0], edge_index[1]].to(device)
                cost = model(query_data.x.to(device), edge_index, edge_weight=edge_weights)
                Cost[j, i] = cost.item()
    
    return Alpha, Beta, Cost, alphas, betas


def interpolate_landscapes(Cost1, Cost2, num_interp_frames=10):
    """
    Smoothly interpolate between two cost landscapes.
    Returns a list of interpolated cost grids.
    
    Args:
        Cost1: First cost landscape (numpy array)
        Cost2: Second cost landscape (numpy array)
        num_interp_frames: Number of intermediate frames to generate
    """
    interp_costs = []
    for t in np.linspace(0, 1, num_interp_frames):
        # Linear interpolation of cost surfaces
        interp_cost = (1 - t) * Cost1 + t * Cost2
        interp_costs.append(interp_cost)
    return interp_costs


def create_epoch_animation(
    model_paths,
    query_file,
    model_class,
    model_kwargs,
    device='cpu',
    num_steps=50,
    interp_frames_per_epoch=5,
    output_path='cost_landscape_evolution.mp4',
    fps=10,
    plot_type='contour',
    query_idx=8
):
    """
    Create an animation showing cost landscape evolution across training epochs.
    
    Args:
        model_paths: List of paths to model checkpoints for each epoch
                    e.g., ["model_epoch1.pt", "model_epoch5.pt", ...]
        query_file: Path to query file (pickle)
        model_class: The model class (e.g., CostGNNv3)
        model_kwargs: Dict of kwargs for model initialization
        device: 'cpu' or 'cuda'
        num_steps: Grid resolution for landscape (num_steps x num_steps)
        interp_frames_per_epoch: Number of interpolation frames between epochs
                                 Set to 0 for no interpolation (1 frame per epoch)
        output_path: Output file path (.mp4 or .gif)
        fps: Frames per second in the output video/gif
        plot_type: 'contour' or '3d'
        query_idx: Which query to use from the query file (default 8)
    
    Returns:
        The animation object
    """
    # Load query
    queries = load_sparql_queries(query_file, 1000)
    queries = [q for q in queries if len(q.triples) == 3]
    if query_idx >= len(queries):
        query_idx = 0
        print(f"Warning: query_idx out of range, using query 0")
    query_data = queries[query_idx].torch_data[0]
    
    print(f"Computing cost landscapes for {len(model_paths)} epochs...")
    
    # Compute cost landscape for each epoch
    all_costs = []
    Alpha, Beta, alphas, betas = None, None, None, None
    
    for i, model_path in enumerate(model_paths):
        print(f"  Processing epoch {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        # Load model for this epoch
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        Alpha, Beta, Cost, alphas, betas = compute_cost_landscape_for_model(
            model, query_data, device, num_steps
        )
        all_costs.append(Cost)
    
    # Build frame sequence with interpolation
    frames = []
    epoch_labels = []
    
    # Extract epoch numbers from filenames if possible
    def get_epoch_num(path):
        basename = os.path.basename(path)
        # Try to extract epoch number from filename like "model_epoch5.pt"
        import re
        match = re.search(r'epoch[_]?(\d+)', basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    epoch_nums = [get_epoch_num(p) for p in model_paths]
    
    for i in range(len(all_costs)):
        # Add actual epoch frame
        frames.append(all_costs[i])
        if epoch_nums[i] is not None:
            epoch_labels.append(f"Epoch {epoch_nums[i]}")
        else:
            epoch_labels.append(f"Model {i+1}")
        
        # Add interpolation frames (except after last epoch)
        if interp_frames_per_epoch > 0 and i < len(all_costs) - 1:
            interp = interpolate_landscapes(all_costs[i], all_costs[i+1], interp_frames_per_epoch + 2)
            # Skip first and last (they're the actual epoch frames)
            for j, interp_cost in enumerate(interp[1:-1]):
                frames.append(interp_cost)
                progress = (j + 1) / (interp_frames_per_epoch + 1)
                if epoch_nums[i] is not None and epoch_nums[i+1] is not None:
                    epoch_labels.append(f"Epoch {epoch_nums[i]} → {epoch_nums[i+1]} ({progress:.0%})")
                else:
                    epoch_labels.append(f"Model {i+1} → {i+2} ({progress:.0%})")
    
    print(f"Total frames: {len(frames)} ({len(all_costs)} epochs + {len(frames) - len(all_costs)} interpolation frames)")
    
    # Get global min/max for consistent colormap across all frames
    all_costs_array = np.array(all_costs)
    vmin, vmax = all_costs_array.min(), all_costs_array.max()
    print(f"Cost range: [{vmin:.4f}, {vmax:.4f}]")
    
    # Custom colormap (same as in visualize_cost_landscape_3d)
    dark = '#a9d6e5'
    base = '#468faf'
    light = '#012a4a'
    cmap = LinearSegmentedColormap.from_list('custom_intensity', [light, base, dark])
    
    # Track minimum trajectory across epochs
    min_trajectory = []
    for cost in all_costs:
        min_idx = np.unravel_index(cost.argmin(), cost.shape)
        min_alpha, min_beta = alphas[min_idx[1]], betas[min_idx[0]]
        min_trajectory.append((min_alpha, min_beta, cost.min()))
    
    # Create animation
    if plot_type == 'contour':
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Store colorbar reference to update it
        contour_filled = ax.contourf(Alpha, Beta, frames[0], levels=40, cmap='viridis', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(contour_filled, ax=ax)
        cbar.set_label('Predicted Cost', fontsize=11)
        
        def update(frame_idx):
            ax.clear()
            Cost = frames[frame_idx]
            
            # Filled contour plot
            ax.contourf(Alpha, Beta, Cost, levels=40, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.contour(Alpha, Beta, Cost, levels=20, colors='white', alpha=0.3, linewidths=0.5)
            
            # Mark corner plans
            ax.scatter([0, 1, 0], [0, 0, 1], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
            ax.annotate('1→2→3', (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=10, color='white')
            ax.annotate('1→3→2', (1, 0), textcoords="offset points", xytext=(-50, 10), fontsize=10, color='white')
            ax.annotate('2→3→1', (0, 1), textcoords="offset points", xytext=(10, -15), fontsize=10, color='white')
            
            # Mark current minimum
            min_idx = np.unravel_index(Cost.argmin(), Cost.shape)
            min_alpha, min_beta = alphas[min_idx[1]], betas[min_idx[0]]
            ax.scatter([min_alpha], [min_beta], c='yellow', s=150, marker='*', zorder=6, edgecolors='black', linewidths=1)
            ax.annotate(f'Min: {Cost.min():.2f}', (min_alpha, min_beta), 
                       textcoords="offset points", xytext=(10, 10), fontsize=9, color='yellow')
            
            ax.set_xlabel('α → "1 JOIN 3 JOIN 2"', fontsize=12)
            ax.set_ylabel('β → "2 JOIN 3 JOIN 1"', fontsize=12)
            ax.set_title(f'Cost Landscape Evolution - {epoch_labels[frame_idx]}', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            return ax,
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
        
    else:  # 3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame_idx):
            ax.clear()
            Cost = frames[frame_idx]
            
            surf = ax.plot_surface(Alpha, Beta, Cost, cmap='viridis', alpha=0.8,
                                   vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
            
            # Mark corner plans
            ax.scatter([0, 1, 0], [0, 0, 1], [Cost[0,0], Cost[0,-1], Cost[-1,0]], 
                      c='red', s=100, alpha=1.0)
            
            ax.set_xlabel('α', fontsize=11)
            ax.set_ylabel('β', fontsize=11)
            ax.set_zlabel('Predicted Cost', fontsize=11)
            #ax.set_zlim(vmin * 0.95, vmax * 1.05)
            ax.set_zlim(vmin * 0.95, 2.0)

            ax.set_title(f'Cost Landscape - {epoch_labels[frame_idx]}', fontsize=14)
            
            # Add text annotations
            ax.text(0, 0, Cost[0,0], '  1→2→3', fontsize=9)
            ax.text(1, 0, Cost[0,-1], '  1→3→2', fontsize=9) 
            ax.text(0, 1, Cost[-1,0], '  2→3→1', fontsize=9)
            
            return ax,
        
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    if output_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Cost Landscape Evolution'}, bitrate=2000)
    
    anim.save(output_path, writer=writer, dpi=150)
    print(f"Animation saved to {output_path}!")
    
    plt.close()
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Epochs processed: {len(model_paths)}")
    print(f"Total frames: {len(frames)}")
    print(f"Min trajectory across epochs:")
    for i, (ma, mb, mc) in enumerate(min_trajectory):
        epoch_label = f"Epoch {epoch_nums[i]}" if epoch_nums[i] else f"Model {i+1}"
        print(f"  {epoch_label}: min={mc:.4f} at (α={ma:.3f}, β={mb:.3f})")
    
    return anim


def create_epoch_animation_clean(
    model_paths,
    query_file,
    model_class,
    model_kwargs,
    device='cpu',
    num_steps=50,
    interp_frames_per_epoch=5,
    output_path='cost_landscape_evolution_clean.mp4',
    fps=10,
    query_idx=8
):
    """
    Create a clean animation (no axes/labels) showing cost landscape evolution.
    Good for presentations or overlaying on other content.
    
    Args: Same as create_epoch_animation
    """
    # Load query
    queries = load_sparql_queries(query_file, 100)
    queries = [q for q in queries if len(q.triples) == 3]
    if query_idx >= len(queries):
        query_idx = 0
    query_data = queries[query_idx].torch_data[0]
    
    print(f"Computing cost landscapes for {len(model_paths)} epochs (clean version)...")
    
    # Compute cost landscape for each epoch
    all_costs = []
    Alpha, Beta, alphas, betas = None, None, None, None
    
    for i, model_path in enumerate(model_paths):
        print(f"  Processing epoch {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        Alpha, Beta, Cost, alphas, betas = compute_cost_landscape_for_model(
            model, query_data, device, num_steps
        )
        all_costs.append(Cost)
    
    # Build frame sequence with interpolation
    frames = []
    for i in range(len(all_costs)):
        frames.append(all_costs[i])
        if interp_frames_per_epoch > 0 and i < len(all_costs) - 1:
            interp = interpolate_landscapes(all_costs[i], all_costs[i+1], interp_frames_per_epoch + 2)
            for interp_cost in interp[1:-1]:
                frames.append(interp_cost)
    
    print(f"Total frames: {len(frames)}")
    
    # Global min/max
    all_costs_array = np.array(all_costs)
    vmin, vmax = all_costs_array.min(), all_costs_array.max()
    
    # Colormap
    dark = '#a9d6e5'
    base = '#468faf'
    light = '#012a4a'
    cmap = LinearSegmentedColormap.from_list('custom_intensity', [light, base, dark])
    
    # Create clean animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame_idx):
        ax.clear()
        Cost = frames[frame_idx]
        
        ax.contourf(Alpha, Beta, Cost, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(Alpha, Beta, Cost, levels=30, colors='#000000', alpha=1, linewidths=0.5)
        
        ax.set_axis_off()
        return ax,
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    
    print(f"Saving clean animation to {output_path}...")
    if output_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
    
    anim.save(output_path, writer=writer, dpi=150)
    print(f"Clean animation saved!")
    
    plt.close()
    return anim




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
    
    query_file = ".../datasets/plans/wn18rr/stars/queries.pt"
    model_path = ".../training_results/lubm-path-nice-v3-6-layer/model.pt"
    
    # Visualization options
    show_penalty_landscape = False  # Toggle to include penalty in landscape
    
    # ==========================================================================
    # STATIC VISUALIZATION (single model)
    # ==========================================================================
    # Uncomment to run static visualization:
    # visualize_cost_transition(query_file, model_path)  # 2D version
    # visualize_cost_landscape_3d(query_file, model_path, include_penalty=show_penalty_landscape)  # 3D version
    
    # ==========================================================================
    # ANIMATION: Cost Landscape Evolution Across Epochs
    # ==========================================================================
    # 
    # Option 1: Manually specify model paths for each epoch
    # model_paths = [
    #     "/path/to/model_epoch1.pt",
    #     "/path/to/model_epoch5.pt",
    #     "/path/to/model_epoch10.pt",
    #     "/path/to/model_epoch20.pt",
    #     "/path/to/model_epoch50.pt",
    # ]
    
    # Option 2: Auto-discover epoch checkpoints using glob
    # Adjust the pattern to match your checkpoint naming convention
    checkpoint_dir = ".../training_results/lubm-path-nice-v3-6-layer"
    model_paths = sorted(glob.glob(f"{checkpoint_dir}/model_epoch*.pt"))

    model_paths = [
        ".../training_results/wn18rr-v3-ordering/models/model_epoch_1.pt",
        ".../training_results/wn18rr-v3-ordering/models/model_epoch_10.pt",
        ...
    ]
    
    # If no epoch checkpoints found, fall back to just the final model for testing
    if not model_paths:
        print("No epoch checkpoints found. Using single model for demo.")
        model_paths = [model_path]
    
    print(f"Found {len(model_paths)} model checkpoints")
    for p in model_paths:
        print(f"  - {os.path.basename(p)}")
    
    # Model configuration (must match how the model was trained)
    model_kwargs = {
        'node_feature_dim': 307,
        'hidden_dim': 128,
        'n_layers': 6,
        'use_jk': False,
        'jk_mode': 'cat',
        'use_residual': False,
        'use_layer_norm': True,
        'dropout': 0.0
    }
    
    # Create the animation
    if len(model_paths) >= 1:
        create_epoch_animation(
            model_paths=model_paths,
            query_file=query_file,
            model_class=CostGNNv3,
            model_kwargs=model_kwargs,
            device='cpu',
            num_steps=50,              # Grid resolution (50x50)
            interp_frames_per_epoch=10, # Set to 0 for no interpolation (1 frame per epoch)
            output_path='cost_landscape_evolution.mp4',  # or .gif
            fps=30,
            plot_type='contour',       # 'contour' or '3d'
            query_idx=0                # Which query to visualize
        )
        
        # Optionally create a clean version (no axes/labels)
        # create_epoch_animation_clean(
        #     model_paths=model_paths,
        #     query_file=query_file,
        #     model_class=CostGNNv3,
        #     model_kwargs=model_kwargs,
        #     device='cpu',
        #     num_steps=50,
        #     interp_frames_per_epoch=5,
        #     output_path='cost_landscape_evolution_clean.mp4',
        #     fps=10,
        #     query_idx=8
        # )
