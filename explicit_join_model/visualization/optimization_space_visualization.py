import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from optimization_evaluation import left_deep_adj_from_perm, load_sparql_queries, sample_binary_concrete, sample_grouped_gumbel_softmax, sample_gumbel, _temperature_anneal
from model import CostGNNv2
from create_data.process_dataset_single_file import SPARQLQuery
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim


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


def visualize_cost_landscape_3d(query_file, model_path, device='cpu', num_steps=40, include_penalty=False, penalty_config=None):
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
    model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query (assuming single query with 3 triples)
    queries = load_sparql_queries(query_file, 1)
    query_data = queries[0].torch_data[0]
    
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
                    Cost[j, i] = cost.item() + penalty.item()
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
    logit_sampling: str = 'dual-softmax',
    trajectory_save_interval: int = 1,
):
    """
    Modified optimization function that tracks trajectory in the (α, β) space.
    """
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2
    
    num_edges = edge_index.size(1)
    
    # Initialize edge logits
    edge_logits = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)
    edge_logits_slot2 = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)
    
    # Optimizer
    if logit_sampling == 'dual-softmax':
        optimiser = optim.AdamW([edge_logits, edge_logits_slot2], lr=learning_rate)
    else:
        optimiser = optim.AdamW([edge_logits], lr=learning_rate)
    
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
            
            triple_to_triple = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits_1[triple_to_triple] = float('-inf')
            masked_logits_2[triple_to_triple] = float('-inf')
            join_to_triple = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits_1[join_to_triple] = float('-inf')
            masked_logits_2[join_to_triple] = float('-inf')
            
            join_target_mask = (edge_index[1] >= triples_num)
            slot1 = torch.zeros_like(edge_logits)
            slot2 = torch.zeros_like(edge_logits)
            
            slot1[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits_1[join_target_mask], edge_index[1][join_target_mask], tau)
            slot2[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits_2[join_target_mask], edge_index[1][join_target_mask], tau)
            
            edge_weights = slot1 + slot2
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        else:
            edge_weights = sample_binary_concrete(edge_logits, tau)
        
        # Build current adjacency matrix
        A_current = torch.zeros((N_NODES, N_NODES), device=device)
        A_current[edge_index[0], edge_index[1]] = edge_weights
        
        # Project current adjacency onto the (α, β) coordinate system
        if step % trajectory_save_interval == 0:
            with torch.no_grad():
                # Solve for α, β that minimize ||A_current - (interpolated A)||
                # This is a least squares problem
                A_current_flat = A_current.flatten()
                A_base_flat = A_base.flatten()
                A_alpha_flat = A_alpha.flatten()
                A_beta_flat = A_beta.flatten()
                
                # Set up linear system: A_current ≈ (1-α)(1-β)A_base + α(1-β)A_alpha + (1-α)βA_beta + αβ(0.5A_alpha + 0.5A_beta)
                # This is a 2D optimization problem we solve with least squares
                def compute_interpolated_A(alpha, beta):
                    return ((1 - alpha) * (1 - beta) * A_base_flat +
                            alpha * (1 - beta) * A_alpha_flat +
                            (1 - alpha) * beta * A_beta_flat +
                            alpha * beta * (0.5 * A_alpha_flat + 0.5 * A_beta_flat))
                
                # Grid search for best α, β (simple but effective)
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
        
        # Compute penalties (same as original)
        A = A_current
        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)
        
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
        
        # Lambda ramping
        if use_lambda_ramping:
            def annealed_lam(lam_max, step, ramp_steps=150):
                frac = min(1.0, step / ramp_steps)
                return lam_max * (frac ** 2)
            lambda_total = annealed_lam(lambda_total_penalty, step, ramp_steps=optimization_steps)
        else:
            lambda_total = lambda_total_penalty
        
        loss = cost_pred + lambda_total * total_penalty
        
        # Backprop
        loss.backward()
        optimiser.step()
        
        if verbose and (step + 1) % 100 == 0:
            print(f"Step {step+1}/{optimization_steps} Cost: {cost_pred.item():.2f} Penalty: {total_penalty.item():.2f}")
    
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
    model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query
    queries = load_sparql_queries(query_file, 100)
    query_data = queries[10].torch_data[0]
    
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
                    Cost[j, i] = cost.item() + 0.003 * penalty.item()
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
                traj_costs_with_penalty.append(cost.item() + 0.003 * penalty.item())
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


if __name__ == "__main__":
    # Configuration (matching your provided config)
    config = {
        'optimization_steps': 100,
        'verbose': False,
        'optimization_params': {
            'learning_rate': 0.1,
            'lambda_acyclic': 2065.0,
            'lambda_triple_in': 2390.0,
            'lambda_triple_out': 105.0,
            'lambda_join_in': 387.0,
            'lambda_join_out': 2610.0,
            'lambda_entropy': 1000,
            'lambda_total_penalty': 1.0,
            'lambda_left_linear': 3290.0,
            'init_tau': 8.2,
            'min_tau': 1.0,
            'tau_decay': 0.976,
            'use_temperature_annealing': True,
            'min_penalty_threshold': 30.0,
            'use_lambda_ramping': True,
            'logit_sampling': 'dual-softmax',
            'trajectory_save_interval': 1,  # Save trajectory points every 20 steps
        }
    }
    
    query_file = "/home/tim/query_optimization/sparql_path_queries/queries.pkl"
    model_path = "/home/tim/query_optimization/explicit_join_model/models/path_model.pt"
    
    # Visualization options
    show_penalty_landscape = True  # Toggle to include penalty in landscape
    
    # Choose which visualization to run:
    # visualize_cost_transition(query_file, model_path)  # 2D version
    # visualize_cost_landscape_3d(query_file, model_path, include_penalty=show_penalty_landscape)  # 3D version
    visualize_optimization_trajectory_3d(query_file, model_path, config, include_penalty=show_penalty_landscape, clean_plot=True)  # 3D with trajectory
