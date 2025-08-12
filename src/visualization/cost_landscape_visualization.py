import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from model import CostGNNv2
from src.create_data.create_cost_model_training_data import SPARQLQuery
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim

from matplotlib.collections import LineCollection
from matplotlib import cm

#plt.rc('font', family='serif', size=9)

import scienceplots
plt.style.use('science')

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


def visualize_cost_transition(query_file, model_path, device='cpu', num_steps=100, include_penalty=False, penalty_config=None):
    """
    Visualize how predicted cost changes when transitioning from 
    one random plan to another random plan for a query.
    
    Args:
        include_penalty: If True, visualize cost + penalty landscape instead of just cost
        penalty_config: Dict with penalty weights (lambda values)
    """
    # Load model
    model = CostGNNv2(node_feature_dim=307, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load query
    queries = load_sparql_queries(query_file, 10)
    query_data = queries[0].torch_data[0]
    
    # Calculate query size
    query_size = (query_data.x.shape[0] + 1) // 2
    
    # Create two random permutations
    perm1 = torch.randperm(query_size)
    perm2 = torch.randperm(query_size)
    
    # Create adjacency matrices for both plans
    A1 = left_deep_adj_from_perm(perm1)
    A2 = left_deep_adj_from_perm(perm2)
    
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
            penalty_config['lambda_left_linear'] * P_left_linear
            #penalty_config['lambda_entropy'] * P_entropy
        )
        
        return total_penalty
    
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
            
            if include_penalty:
                penalty = compute_penalty(A_interp.to(device), edge_weights)
                costs.append(cost.item() + 0.00005 * penalty.item())
            else:
                costs.append(cost.item())
    

    fig, ax = plt.subplots(figsize=(12, 3.5))

    fontsize = 20

    # convert costs to numpy array and normalise to [0,1] for the colormap
    costs = np.array(costs)
    norm   = plt.Normalize(costs.min(), costs.max())
    cmap   = cm.get_cmap('coolwarm')            

    pts       = np.column_stack([alphas, costs])
    segments  = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0)
    lc.set_array((costs[:-1] + costs[1:]) / 2)
    ax.add_collection(lc)

    for i in range(len(alphas) - 1):
        ax.fill_between(
            [alphas[i], alphas[i+1]],
            [costs[i],  costs[i+1]],
            0,
            color=cmap(norm((costs[i] + costs[i+1]) / 2)),
            alpha=0.25,
            linewidth=0
        )

    ax.set_xlabel(r'$(1-\alpha)\,P_1 + \alpha\,P_2$', fontsize=fontsize)
    if include_penalty:
        ax.set_ylabel(r'$\hat{C} + \lambda\,P_{\text{struct}}$', fontsize=fontsize)
        filename = 'cost_penalty_transition.pdf'
    else:
        ax.set_ylabel('Predicted Cost', fontsize=fontsize)
        filename = 'cost_transition.pdf'

    ax.plot(0, costs[0], 'ko', markersize=8)
    ax.plot(1, costs[-1], 'ko', markersize=8)
    ax.annotate('P1', xy=(0,  costs[0]), xytext=(0.05,  costs[0]),  fontsize=fontsize, fontweight='bold')
    ax.annotate('P2', xy=(1,  costs[-1]), xytext=(0.95, costs[-1]), fontsize=fontsize, fontweight='bold')

    ax.set_xlim(alphas.min(), alphas.max())
    ax.set_ylim(costs.min() * 0.95, costs.max()*1.05)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'], fontsize=fontsize)
    ax.set_yticks([])  # Remove y-axis ticks
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plan 1 permutation: {perm1.tolist()}")
    print(f"Plan 2 permutation: {perm2.tolist()}")
    
    landscape_type = "cost + penalty" if include_penalty else "cost"
    print(f"Plan 1 {landscape_type}: {costs[0]:.4f}")
    print(f"Plan 2 {landscape_type}: {costs[-1]:.4f}")
    print(f"{landscape_type.capitalize()} difference: {costs[-1] - costs[0]:.4f}")




if __name__ == "__main__":
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
            'trajectory_save_interval': 1,
        }
    }
    
    query_file = "datasets/plans_for_landscape_visualization/wikidata_star_14/queries.pkl"
    model_path = "datasets/models/wikidata/star_model.pt"
    
    penalty_config = {
        'lambda_acyclic': config['optimization_params']['lambda_acyclic'],
        'lambda_triple_in': config['optimization_params']['lambda_triple_in'],
        'lambda_triple_out': config['optimization_params']['lambda_triple_out'],
        'lambda_join_in': config['optimization_params']['lambda_join_in'],
        'lambda_join_out': config['optimization_params']['lambda_join_out'],
        'lambda_left_linear': config['optimization_params']['lambda_left_linear'],
        'lambda_entropy': config['optimization_params']['lambda_entropy'],
    }
    


    print("\nGenerating cost + penalty visualization...")
    visualize_cost_transition(query_file, model_path, include_penalty=True, penalty_config=penalty_config)  # 2D cost + penalty
    