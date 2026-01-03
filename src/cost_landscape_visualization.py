import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
from model import CostGNNv2, CostGNNv3
from src.create_data.create_cost_model_training_data import SPARQLQuery
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim

from matplotlib.collections import LineCollection
from matplotlib import cm

#plt.rc('font', family='serif', size=9)

#import scienceplots
#plt.style.use('science')

from optimization import GBJO
from optimization.gumbel_utils import sample_grouped_gumbel_softmax

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

def add_fingerprints_to_query_data(query_data, fingerprint_dim: int = 64):
    """
    Add random Gaussian fingerprints to join nodes in query data.
    Matches `AddRandomGaussianFingerprints` used during training.
    """
    x = query_data.x.clone()

    is_join = (x[:, -1] == 1.0)
    join_indices = torch.where(is_join)[0]
    n_joins = len(join_indices)

    if n_joins == 0:
        return query_data

    fingerprints = torch.randn(n_joins, fingerprint_dim, device=x.device)
    fingerprints = fingerprints / fingerprints.norm(dim=1, keepdim=True)

    for i, join_idx in enumerate(join_indices):
        x[join_idx, :fingerprint_dim] = fingerprints[i]

    query_data.x = x
    return query_data


def _gbjo_zero_logits_start_adjacency(query_data: torch.Tensor, device: str, init_tau: float = 1.0):
    """
    Recreate the GBJO 'midpoint' initialization: all-zero logits, then apply the
    same structural masking + grouped softmax to get initial edge weights and
    corresponding adjacency matrix A0.
    """
    data = query_data.to(device)
    n_nodes = int(data.x.shape[0])
    triples_num = (n_nodes + 1) // 2

    # Enumerate all candidate edges (excluding self-loops), same as GBJO.
    src, dst = torch.where(~torch.eye(n_nodes, dtype=torch.bool, device=device))
    edge_index = torch.stack([src, dst], dim=0)
    num_edges = int(edge_index.size(1))

    edge_logits = torch.zeros(num_edges, device=device, dtype=torch.float32)
    masked_logits = edge_logits.clone()

    triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
    join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
    masked_logits[triple_to_triple_mask] = float("-inf")
    masked_logits[join_to_triple_mask] = float("-inf")

    tau = float(init_tau) if init_tau is not None else 1.0
    if tau <= 0:
        tau = 1.0

    edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], temperature=tau, use_gumbel_noise=False)
    # Root (final join) has no outgoing edge (same as GBJO).
    edge_weights[edge_index[0] == (n_nodes - 1)] = 0.0

    A0 = torch.zeros((n_nodes, n_nodes), device=device, dtype=torch.float32)
    A0[edge_index[0], edge_index[1]] = edge_weights

    return A0, edge_index, triples_num


def visualize_gbjo_transition(
    query_file: str,
    model_path: str,
    *,
    query_index: int = 0,
    plan_index: int = 0,
    device: str = "cpu",
    num_steps: int = 100,
    optimization_steps: int = 500,
    optimization_params: dict | None = None,
    model_params: dict | None = None,
    add_fingerprints: bool = True,
    fingerprint_dim: int = 64,
    include_penalty: bool = False,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Pick a single query (by index), run GBJO to obtain a discrete final plan,
    and visualize the cost transition when interpolating from GBJO's starting
    point (all-zero logits -> masked grouped-softmax edge weights) to that final plan.
    """
    optimization_params = (optimization_params or {}).copy()

    # Device
    device = str(device)

    def _gbjo_total_penalty_from_A(A: torch.Tensor, edge_weights: torch.Tensor, *, triples_num: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GBJO's penalty terms exactly as in `src/optimization/methods.py` (GBJO).

        Returns:
            (total_penalty_weighted, total_penalty_raw)
        """
        n_nodes = int(A.shape[0])
        root = n_nodes - 1

        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=A.device)
        join_nodes = torch.arange(triples_num, n_nodes, device=A.device)
        non_root_joins = torch.arange(triples_num, root, device=A.device)

        # Structural penalties
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - n_nodes

        # Left-deep penalty
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
            P_left_linear = torch.tensor(0.0, device=A.device)

        # Entropy penalty (same as GBJO: entropy of relaxed edge weights)
        eps = 1e-10
        probs = edge_weights.clamp(min=eps)
        P_entropy = -(probs * torch.log(probs)).sum()

        # Weighted total penalty (GBJO)
        lambda_triple_in = float(optimization_params.get("lambda_triple_in", 1000.0))
        lambda_triple_out = float(optimization_params.get("lambda_triple_out", 1000.0))
        lambda_join_in = float(optimization_params.get("lambda_join_in", 500.0))
        lambda_join_out = float(optimization_params.get("lambda_join_out", 1000.0))
        lambda_acyclic = float(optimization_params.get("lambda_acyclic", 1000.0))
        lambda_entropy = float(optimization_params.get("lambda_entropy", 10.0))
        lambda_left_linear = float(optimization_params.get("lambda_left_linear", 1000.0))

        total_penalty = (
            lambda_triple_in * P_triple_in
            + lambda_triple_out * P_triple_out
            + lambda_join_in * P_join_in
            + lambda_join_out * P_join_out
            + lambda_acyclic * P_acyclic
            + lambda_entropy * P_entropy
            #+ lambda_left_linear * P_left_linear
        )

        total_penalty_raw = (
            P_triple_in
            + P_triple_out
            + P_join_in
            + P_join_out
            + P_acyclic
            + P_entropy
            #+ P_left_linear
        )

        return total_penalty, total_penalty_raw

    # Load model
    if model_params is None:
        model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6, use_jk=False, jk_mode='cat', use_residual=True, use_layer_norm=False, dropout=0.0).to(device)
    else:
        params = model_params.copy()
        version = params.pop("version", "v3")
        if version == "v3":
            model = CostGNNv3(**params).to(device)
        elif version == "v2":
            model = CostGNNv2(**params).to(device)
        else:
            raise ValueError(f"Unknown model version: {version}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load queries and select one
    queries = load_sparql_queries(query_file)
    if query_index < 0 or query_index >= len(queries):
        raise IndexError(f"query_index={query_index} out of range (0..{len(queries)-1})")
    query = queries[query_index]

    # Get torch_data similarly to evaluation_parallel.py
    try:
        torch_data = query.torch_data[plan_index]
    except Exception:
        torch_data = query

    if torch_data is None:
        raise ValueError(f"Query {query_index} has null torch_data at plan_index={plan_index}")

    if add_fingerprints:
        torch_data = add_fingerprints_to_query_data(torch_data, fingerprint_dim=fingerprint_dim)

    # Compute GBJO start adjacency from all-zero logits (the requested "midpoint")
    init_tau = optimization_params.get("init_tau", 1.0)
    A0, edge_index, _triples_num = _gbjo_zero_logits_start_adjacency(torch_data, device=device, init_tau=init_tau)

    # Run GBJO to get final discrete adjacency
    final_A, triples_num, final_pred_cost = GBJO(
        torch_data,
        model,
        device,
        optimization_steps=int(optimization_steps),
        verbose=bool(optimization_params.get("gbjo_verbose", False)),
        save_directory=optimization_params.get("save_directory", None),
        **{k: v for k, v in optimization_params.items() if k not in ["gbjo_verbose", "save_directory"]},
    )

    # Interpolate A0 -> final_A and evaluate predicted cost along the path
    alphas = np.linspace(0.0, 1.0, int(num_steps))
    costs = []
    penalties = []
    objectives = []
    lambda_totals = []

    lambda_total_penalty = float(optimization_params.get("lambda_total_penalty", 1.0))
    use_lambda_ramping = bool(optimization_params.get("use_lambda_ramping", False))
    lambda_ramp_exponent = float(optimization_params.get("lambda_ramp_exponent", 2.0))
    # Ramp across the visualization steps themselves (idx-based),
    # using the same GBJO formula for frac.
    vis_steps = max(1, len(alphas) - 1)

    with torch.no_grad():
        x = torch_data.x.to(device)
        for idx, alpha in enumerate(alphas):
            A_interp = (1.0 - float(alpha)) * A0 + float(alpha) * final_A
            edge_weights = A_interp[edge_index[0], edge_index[1]]
            cost = model(x, edge_index, edge_weight=edge_weights)
            c = float(cost.item())
            costs.append(c)

            if include_penalty:
                # GBJO ramping: frac = min(1.0, step / optimization_steps)
                # Here, step is the visualization step index (0..vis_steps).
                frac = min(1.0, idx / vis_steps)
                if use_lambda_ramping:
                    lambda_total = lambda_total_penalty * (frac ** lambda_ramp_exponent)
                else:
                    lambda_total = lambda_total_penalty
                lambda_totals.append(float(lambda_total))

                total_penalty, _total_penalty_raw = _gbjo_total_penalty_from_A(A_interp, edge_weights, triples_num=int(triples_num))
                p = float(total_penalty.item())
                penalties.append(p)
                objectives.append(c + float(lambda_total) * p)

    # Plot (same style as visualize_cost_transition)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fontsize = 20

    y = np.array(objectives if include_penalty else costs)
    norm = plt.Normalize(y.min(), y.max())
    cmap = cm.get_cmap("coolwarm")

    pts = np.column_stack([alphas, y])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0)
    lc.set_array((y[:-1] + y[1:]) / 2)
    ax.add_collection(lc)

    for i in range(len(alphas) - 1):
        ax.fill_between(
            [alphas[i], alphas[i + 1]],
            [y[i], y[i + 1]],
            0,
            color=cmap(norm((y[i] + y[i + 1]) / 2)),
            alpha=0.25,
            linewidth=0,
        )

    ax.set_xlabel(r"$(1-\alpha)\,A_0 + \alpha\,A_{\mathrm{GBJO}}$", fontsize=fontsize)
    if include_penalty:
        ax.set_ylabel(r"$\hat{C} + \lambda\,P_{\mathrm{GBJO}}$", fontsize=fontsize)
    else:
        ax.set_ylabel("Predicted Cost", fontsize=fontsize)

    ax.plot(0, y[0], "ko", markersize=8)
    ax.plot(1, y[-1], "ko", markersize=8)
    ax.annotate("A0", xy=(0, y[0]), xytext=(0.05, y[0]), fontsize=fontsize, fontweight="bold")
    ax.annotate("GBJO", xy=(1, y[-1]), xytext=(0.92, y[-1]), fontsize=fontsize, fontweight="bold")

    ax.set_xlim(alphas.min(), alphas.max())
    ax.set_ylim(y.min() * 0.95, y.max() * 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=fontsize)
    ax.set_yticks([])

    plt.tight_layout()

    if save_path is None:
        if include_penalty:
            save_path = f"gbjo_cost_penalty_transition_query_{query_index}.pdf"
        else:
            save_path = f"gbjo_cost_transition_query_{query_index}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[GBJO transition] Query index: {query_index}")
    print(f"[GBJO transition] Saved plot to: {save_path}")
    print(f"[GBJO transition] A0 predicted cost: {float(costs[0]):.6f}")
    print(f"[GBJO transition] GBJO predicted cost (interp end): {float(costs[-1]):.6f}")
    print(f"[GBJO transition] GBJO predicted cost (returned): {float(final_pred_cost):.6f}")
    print(f"[GBJO transition] Δ predicted cost: {float(costs[-1] - costs[0]):.6f}")
    if include_penalty:
        print(f"[GBJO transition] Using objective: cost + lambda_total_penalty * total_penalty (lambda_total_penalty={lambda_total_penalty})")
        print(f"[GBJO transition] A0 total_penalty: {float(penalties[0]):.6f}")
        print(f"[GBJO transition] GBJO total_penalty (interp end): {float(penalties[-1]):.6f}")
        print(f"[GBJO transition] A0 objective: {float(objectives[0]):.6f}")
        print(f"[GBJO transition] GBJO objective (interp end): {float(objectives[-1]):.6f}")

    return {
        "query_index": query_index,
        "save_path": save_path,
        "costs": costs,
        "alphas": alphas,
        "penalties": penalties if include_penalty else None,
        "objectives": objectives if include_penalty else None,
        "lambda_totals": lambda_totals if include_penalty else None,
        "lambda_total_penalty": lambda_total_penalty if include_penalty else None,
        "A0": A0.detach().cpu(),
        "final_A": final_A.detach().cpu(),
        "final_pred_cost": float(final_pred_cost),
    }


def visualize_cost_transition(query_file, model_path, device='cpu', num_steps=100, include_penalty=False, penalty_config=None):
    """
    Visualize how predicted cost changes when transitioning from 
    one random plan to another random plan for a query.
    
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
    queries = load_sparql_queries(query_file, 10)
    query_data = queries[5].torch_data[0]
    
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
        'optimization_steps': 10,
        'verbose': False,
        "optimization_params": {
            "k": 1,  # 1 Number of gradient optimization runs
            "learning_rate": 4.9, # 0.35 or 1; best 0.85; 3 or 50 timesteps
            "lambda_acyclic": 29, # 3391
            "lambda_triple_in": 1.5,# 3334.0
            "lambda_triple_out": 1.4,# 2026.0
            "lambda_join_in": 3.6, # 2150.0
            "lambda_join_out": 4.1,# 1295.0
            "lambda_entropy": 0.0,# 0.0
            "lambda_total_penalty": 0.99,# 0.7
            "lambda_left_linear": 60,# 60
            "init_tau": 4, # 15
            "min_tau": 0.49, # 1.0
            "tau_decay": 0.973,
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 9.96,
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 1.01, # 5.3 best: 1.09
            "lr_warmup_steps": 46,
            "gradient_clip_norm": 4.7,
            "use_lr_scheduling": True,
            "decoding_method": "beam",
            "use_gumbel_noise": False,
            "gbjo_verbose": True
        }
    }
    
    query_file = "/home/tim/query_optimization/datasets/plans/wikidata_star_plan_datasets_optimization/queries.pkl"
    model_path = "/home/tim/query_optimization/training_results/wikidata-star-log1p/model.pt"
    
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
    visualize_cost_transition(query_file, model_path, include_penalty=False, penalty_config=penalty_config)  # 2D cost + penalty

    # Example: GBJO transition from zero-logits start (A0) to final discrete GBJO plan
    # (set query_index to pick a specific query from the dataset)
    print("\nGenerating GBJO transition visualization...")
    visualize_gbjo_transition(
        query_file=query_file,
        model_path=model_path,
        query_index=11,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_steps=100,
        optimization_steps=10,
        include_penalty=False,
        optimization_params={
            **config["optimization_params"],
            "gbjo_verbose": False,
        },
        save_path="gbjo_cost_transition_query_0.pdf",
        show=True,
    )
    