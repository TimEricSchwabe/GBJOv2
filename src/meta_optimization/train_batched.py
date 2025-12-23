import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from torch_geometric.data import Batch

from model import CostGNNv3
from optimization.gumbel_utils import sample_grouped_gumbel_softmax
from utils.data_utils import load_sparql_queries


def _temperature_anneal(init_tau: torch.Tensor, min_tau: float, decay: float, step: int, max_step: int, device="cpu") -> torch.Tensor:
    """Exponential temperature annealing every step (differentiable version)."""
    annealed = init_tau - (init_tau - min_tau) * (step / max_step)
    return torch.maximum(annealed, torch.tensor(min_tau, device=device))


def compute_structure_penalties_batched(edge_index, edge_weights, N_NODES, triples_num, batch_size, device):
    """
    Compute structural penalties for a batch of same-size query plans.
    Uses a loop over batch items (fast for moderate batch sizes).
    
    Args:
        edge_index: [2, batch_size * edges_per_graph] batched edge indices
        edge_weights: [batch_size * edges_per_graph] batched edge weights
        N_NODES: number of nodes per graph
        triples_num: number of triple nodes per graph
        batch_size: number of graphs in batch
        device: torch device
    
    Returns:
        Tuple of summed penalties across batch
    """
    edges_per_graph = edge_index.size(1) // batch_size
    
    # Initialize accumulators
    P_triple_total = torch.tensor(0.0, device=device)
    P_join_in_total = torch.tensor(0.0, device=device)
    P_join_out_total = torch.tensor(0.0, device=device)
    P_acyclic_total = torch.tensor(0.0, device=device)
    P_left_linear_total = torch.tensor(0.0, device=device)
    P_entropy_total = torch.tensor(0.0, device=device)
    
    n_joins = N_NODES - triples_num
    root = N_NODES - 1
    
    for b in range(batch_size):
        # Extract this graph's edges
        start_idx = b * edges_per_graph
        end_idx = (b + 1) * edges_per_graph
        
        local_weights = edge_weights[start_idx:end_idx]
        local_edge_index = edge_index[:, start_idx:end_idx]
        
        # Convert to local node indices (remove batch offset)
        local_src = local_edge_index[0] - b * N_NODES
        local_dst = local_edge_index[1] - b * N_NODES
        
        # Build adjacency matrix for this graph
        A = torch.zeros((N_NODES, N_NODES), device=device)
        A[local_src, local_dst] = local_weights
        
        in_deg = A.sum(0)
        out_deg = A.sum(1)
        
        # Triple constraints: out_deg=1
        P_triple = ((out_deg[:triples_num] - 1) ** 2).sum()
        
        # Join constraints: in_deg=2, out_deg=1 (root: out_deg=0)
        P_join_in = ((in_deg[triples_num:] - 2) ** 2).sum()
        P_join_out = ((out_deg[triples_num:root] - 1) ** 2).sum() + out_deg[root] ** 2
        
        # Acyclicity via matrix exponential
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES
        
        # Left-deep: first join gets 2 triples, rest get 1 triple + 1 join
        child_triples = A[:triples_num, triples_num:].sum(0)
        child_joins = A[triples_num:, triples_num:].sum(0)
        
        target_t = torch.ones(n_joins, device=device)
        target_t[0] = 2
        target_j = torch.ones(n_joins, device=device)
        target_j[0] = 0
        
        P_left_linear = ((child_triples - target_t) ** 2).sum() + ((child_joins - target_j) ** 2).sum()
        
        # Entropy
        safe_weights = local_weights.nan_to_num(0.0)
        P_entropy = -(safe_weights * torch.log(safe_weights.clamp(min=1e-9))).sum()
        
        # Accumulate
        P_triple_total = P_triple_total + P_triple
        P_join_in_total = P_join_in_total + P_join_in
        P_join_out_total = P_join_out_total + P_join_out
        P_acyclic_total = P_acyclic_total + P_acyclic
        P_left_linear_total = P_left_linear_total + P_left_linear
        P_entropy_total = P_entropy_total + P_entropy
    
    return P_triple_total, P_join_in_total, P_join_out_total, P_acyclic_total, P_left_linear_total, P_entropy_total


def add_fingerprints_to_query_data(query_data, fingerprint_dim=64):
    """Add random Gaussian fingerprints to join nodes in query data."""
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


def freeze_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def bucket_queries_by_size(queries):
    """Group queries by number of nodes (determines graph structure)."""
    buckets = defaultdict(list)
    for q in queries:
        n_nodes = len(q.x)
        buckets[n_nodes].append(q)
    return buckets


class Hyperparams(nn.Module):
    """Hyperparams of C_theta to be meta-optimized."""
    def __init__(self, init_lambda_triple_out=1.0, init_lambda_join_in=1.0, 
                 init_lambda_join_out=1.0, init_lambda_acyclic=1.0,
                 init_lambda_left_linear=1.0, init_lambda_entropy=1.0, 
                 init_eta=0.8, init_tau=5.0) -> None:
        super().__init__()
        self._lambda_triple_out = nn.Parameter(torch.tensor(float(init_lambda_triple_out)))
        self._lambda_join_in = nn.Parameter(torch.tensor(float(init_lambda_join_in)))
        self._lambda_join_out = nn.Parameter(torch.tensor(float(init_lambda_join_out)))
        self._lambda_acyclic = nn.Parameter(torch.tensor(float(init_lambda_acyclic)))
        self._lambda_left_linear = nn.Parameter(torch.tensor(float(init_lambda_left_linear)))
        self._lambda_entropy = nn.Parameter(torch.tensor(float(init_lambda_entropy)))
        self._eta = nn.Parameter(torch.tensor(float(init_eta)))
        self._init_tau = nn.Parameter(torch.tensor(float(init_tau)))

    def lambda_triple_out(self) -> torch.Tensor:
        return F.softplus(self._lambda_triple_out).clamp(min=0, max=500)

    def lambda_join_in(self) -> torch.Tensor:
        return F.softplus(self._lambda_join_in).clamp(min=0, max=500)

    def lambda_join_out(self) -> torch.Tensor:
        return F.softplus(self._lambda_join_out).clamp(min=0, max=500)

    def lambda_acyclic(self) -> torch.Tensor:
        return F.softplus(self._lambda_acyclic).clamp(min=0, max=500)

    def lambda_left_linear(self) -> torch.Tensor:
        return F.softplus(self._lambda_left_linear).clamp(min=0, max=500)

    def lambda_entropy(self) -> torch.Tensor:
        return F.softplus(self._lambda_entropy).clamp(min=0, max=500)

    def eta(self) -> torch.Tensor:
        return F.softplus(self._eta).clamp(min=1e-4, max=2)

    def init_tau(self) -> torch.Tensor:
        return F.softplus(self._init_tau).clamp(min=1, max=10)


def gbjo_batched(queries, C_theta, hyperparams, device="cpu"):
    """
    Batched GBJO algorithm for queries of the SAME size.
    
    Args:
        queries: List of query Data objects (all must have same N_NODES)
        C_theta: Cost model
        hyperparams: Hyperparameters module
        device: torch device
    
    Returns:
        final_logits: [batch_size, edges_per_graph] logits for each query
    """
    N_STEPS = 100
    batch_size = len(queries)
    
    # All queries have same structure
    N_NODES = len(queries[0].x)
    triples_num = (N_NODES + 1) // 2
    
    # Create single-graph edge_index (all-to-all excluding self-loops)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
    single_edge_index = torch.stack([src, dst], dim=0)
    edges_per_graph = single_edge_index.size(1)
    
    # Batch the edge_index: offset each graph's nodes
    edge_indices = []
    for i in range(batch_size):
        offset = i * N_NODES
        edge_indices.append(single_edge_index + offset)
    edge_index = torch.cat(edge_indices, dim=1)  # [2, batch_size * edges_per_graph]
    
    # Batch query features using PyG
    batch = Batch.from_data_list(queries)
    batch_x = batch.x.to(device)
    batch_batch = batch.batch.to(device)
    
    # Initialize batched logits
    total_edges = batch_size * edges_per_graph
    logits = torch.zeros(total_edges, requires_grad=True, device=device)
    v = torch.zeros_like(logits)
    
    # Source nodes for grouped softmax (absolute indices)
    src_nodes = edge_index[0]
    
    # Precompute local indices for masking (same pattern repeated)
    local_src = edge_index[0] % N_NODES
    local_dst = edge_index[1] % N_NODES
    
    # Precompute masks (same for all graphs in batch)
    triple_to_triple_mask = (local_src < triples_num) & (local_dst < triples_num)
    join_to_triple_mask = (local_src >= triples_num) & (local_dst < triples_num)
    root_mask = local_src == (N_NODES - 1)
    
    for step in range(N_STEPS):
        tau = _temperature_anneal(torch.tensor(5.0, device=device), 1.0, 0.999, step, N_STEPS, device=device)
        learning_rate = hyperparams.eta()
        lambda_triple_out = hyperparams.lambda_triple_out()
        lambda_join_in = hyperparams.lambda_join_in()
        lambda_join_out = hyperparams.lambda_join_out()
        lambda_left_linear = hyperparams.lambda_left_linear()
        lambda_acyclic = hyperparams.lambda_acyclic()
        
        # Apply masks to logits
        masked_logits = logits.clone()
        masked_logits[triple_to_triple_mask] = float('-1e9')
        masked_logits[join_to_triple_mask] = float('-1e9')
        
        # Grouped Gumbel-Softmax (groups by absolute src node, so works for batch)
        edge_weights = sample_grouped_gumbel_softmax(masked_logits, src_nodes, tau)
        
        # Root nodes have no outgoing edges
        edge_weights = edge_weights.clone()
        edge_weights[root_mask] = 0.0
        
        # Batched forward pass through cost model
        cost_pred = C_theta(batch_x, edge_index, edge_weight=edge_weights, batch=batch_batch)
        
        # Compute penalties (loop over batch - still fast)
        P_triple_out, P_join_in, P_join_out, P_acyclic, P_left_linear, P_entropy = \
            compute_structure_penalties_batched(edge_index, edge_weights, N_NODES, triples_num, batch_size, device)
        
        total_penalty = (
            lambda_triple_out * P_triple_out
            + lambda_join_in * P_join_in
            + lambda_join_out * P_join_out
            + lambda_acyclic * P_acyclic
            + lambda_left_linear * P_left_linear
        )
        
        # Ramping coefficient
        frac = min(1.0, step / N_STEPS)
        coefficient = frac ** 3
        
        # Total cost (sum over batch)
        cost = (cost_pred.sum() + coefficient * total_penalty)
        
        # Gradient descent update
        momentum = 0.9
        (g,) = torch.autograd.grad(cost, logits, create_graph=True)
        v = momentum * v + g
        logits = logits - learning_rate * (momentum * v + g)
    
    # Return logits reshaped to [batch_size, edges_per_graph]
    return logits.view(batch_size, edges_per_graph), edge_index, edges_per_graph


def plot_hyperparameter_history(hyperparam_history, save_directory: str) -> None:
    for name, values in hyperparam_history.items():
        plt.figure()
        plt.plot(values)
        plt.title(f"{name} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f"hyperparam_{name}.png"))
        plt.close()


if __name__ == "__main__":
    # Model parameters
    MODEL_PATH = "/home/tim/query_optimization/training_results/wikidata-star-log1p-add-aggr/model.pt"
    QUERY_PATH = "/home/tim/query_optimization/datasets/plans/wikidata_star_plan_datasets_training/new/dataset.pt"
    DROPOUT = 0.0
    HIDDEN_DIM = 128
    NODE_FEATURE_DIM = 307
    N_LAYERS = 6
    USE_JK = False
    JK_MODE = 'cat'
    USE_RESIDUAL = True
    USE_LAYER_NORM = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 16  # Number of same-size queries to process together
    ACCUMULATION_STEPS = 1  # Gradient accumulation (effective batch = BATCH_SIZE * ACCUMULATION_STEPS)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join("meta_optimization_results", f"batched_run_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)
    print(f"Saving all training outputs to: {save_directory}")

    # Load queries
    sparql_queries = torch.load(QUERY_PATH, weights_only=False)
    sparql_queries = sparql_queries['data']
    random.shuffle(sparql_queries)
    sparql_queries = sparql_queries[:10000]
    print(f"INFO: Loaded {len(sparql_queries)} queries")

    # Bucket queries by size
    buckets = bucket_queries_by_size(sparql_queries)
    print(f"INFO: Bucketed into {len(buckets)} size groups:")
    for n_nodes, bucket in sorted(buckets.items()):
        print(f"  N_NODES={n_nodes}: {len(bucket)} queries")

    # Define models
    C_theta = CostGNNv3(
        node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, 
        use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT
    ).to(device)
    C_theta.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    C_psi = CostGNNv3(
        node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL,
        use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT
    ).to(device)
    C_psi.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    freeze_(C_psi)

    # Save config
    config = {
        "timestamp": timestamp,
        "save_directory": save_directory,
        "MODEL_PATH": MODEL_PATH,
        "QUERY_PATH": QUERY_PATH,
        "device": device,
        "model_params": {
            "DROPOUT": DROPOUT,
            "HIDDEN_DIM": HIDDEN_DIM,
            "NODE_FEATURE_DIM": NODE_FEATURE_DIM,
            "N_LAYERS": N_LAYERS,
            "USE_JK": USE_JK,
            "JK_MODE": JK_MODE,
            "USE_RESIDUAL": USE_RESIDUAL,
            "USE_LAYER_NORM": USE_LAYER_NORM,
        },
        "training_params": {
            "BATCH_SIZE": BATCH_SIZE,
            "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
            "EPOCHS": 100,
        },
        "hyperparams_init": {
            "init_lambda_triple_out": 100.0,
            "init_lambda_join_in": 100.0,
            "init_lambda_join_out": 100.0,
            "init_lambda_acyclic": 100.0,
            "init_lambda_left_linear": 100.0,
            "init_lambda_entropy": 100.0,
            "init_eta": 0.9,
            "init_tau": 5.0,
        },
    }
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Initialize hyperparams
    hyperparams = Hyperparams(
        init_lambda_triple_out=config["hyperparams_init"]["init_lambda_triple_out"],
        init_lambda_join_in=config["hyperparams_init"]["init_lambda_join_in"],
        init_lambda_join_out=config["hyperparams_init"]["init_lambda_join_out"],
        init_lambda_acyclic=config["hyperparams_init"]["init_lambda_acyclic"],
        init_lambda_left_linear=config["hyperparams_init"]["init_lambda_left_linear"],
        init_lambda_entropy=config["hyperparams_init"]["init_lambda_entropy"],
        init_eta=config["hyperparams_init"]["init_eta"],
        init_tau=config["hyperparams_init"]["init_tau"],
    ).to(device)

    hyperparam_history = {
        "lambda_triple_out": [], "lambda_join_in": [], "lambda_join_out": [],
        "lambda_acyclic": [], "lambda_left_linear": [], "lambda_entropy": [],
        "eta": [], "init_tau": [],
    }

    # Optimizer
    opt_theta = torch.optim.AdamW(C_theta.parameters(), lr=1e-4)
    anchor_loss_fn = nn.HuberLoss()

    # Fixed penalty weights for outer loss
    lambda_triple_out = 1.0
    lambda_join_in = 1.0
    lambda_join_out = 1.0
    lambda_acyclic = 1.0
    lambda_left_linear = 1.0
    lambda_entropy = 1.0

    EPOCHS = 100
    average_loss_per_epoch = []
    average_penalty_per_epoch = []
    average_anchor_loss_per_epoch = []
    best_loss = float('inf')

    # Step-wise tracking
    avg_loss_per_100_steps = []
    avg_penalty_per_100_steps = []
    avg_anchor_loss_per_100_steps = []
    window_loss = 0.0
    window_penalty = 0.0
    window_anchor_loss = 0.0
    global_step = 0

    for epoch in range(EPOCHS):
        print("INFO: Starting epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0
        epoch_penalty = 0.0
        epoch_anchor_loss = 0.0
        epoch_samples = 0
        
        opt_theta.zero_grad(set_to_none=True)
        accum_count = 0

        # Iterate over buckets (each bucket has same-size queries)
        bucket_items = list(buckets.items())
        random.shuffle(bucket_items)  # Shuffle bucket order each epoch
        
        pbar = tqdm(bucket_items, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for n_nodes, bucket_queries in pbar:
            # Shuffle within bucket
            random.shuffle(bucket_queries)
            
            # Process in mini-batches
            for batch_start in range(0, len(bucket_queries), BATCH_SIZE):
                print(f"INFO: Processing batch {batch_start}/{len(bucket_queries)}")
                batch_end = min(batch_start + BATCH_SIZE, len(bucket_queries))
                batch_queries = bucket_queries[batch_start:batch_end]
                actual_batch_size = len(batch_queries)
                
                if actual_batch_size < 2:
                    continue  # Skip very small batches
                
                # Add fingerprints and move to device
                for q in batch_queries:
                    q = add_fingerprints_to_query_data(q, fingerprint_dim=64)
                    q.x = q.x.to(device)
                    q.edge_index = q.edge_index.to(device)
                    q.y = q.y.to(device)
                
                # Run batched GBJO
                final_logits, edge_index, edges_per_graph = gbjo_batched(
                    batch_queries, C_theta, hyperparams, device=device
                )
                
                # Compute outer loss for each query in batch
                N_NODES = n_nodes
                triples_num = (N_NODES + 1) // 2
                
                # Precompute masks
                local_src = edge_index[0] % N_NODES
                local_dst = edge_index[1] % N_NODES
                triple_to_triple_mask = (local_src < triples_num) & (local_dst < triples_num)
                join_to_triple_mask = (local_src >= triples_num) & (local_dst < triples_num)
                root_mask = local_src == (N_NODES - 1)
                
                # Apply final softmax to get edge weights
                masked_logits = final_logits.view(-1).clone()
                masked_logits[triple_to_triple_mask] = float('-inf')
                masked_logits[join_to_triple_mask] = float('-inf')
                
                edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], temperature=1.0)
                edge_weights = edge_weights.clone()
                edge_weights[root_mask] = 0.0
                
                # Batch forward pass with C_psi
                batch = Batch.from_data_list(batch_queries)
                batch_x = batch.x.to(device)
                batch_batch = batch.batch.to(device)
                
                cost_pred_psi = C_psi(batch_x, edge_index, edge_weight=edge_weights, batch=batch_batch)
                L_outer = cost_pred_psi.mean()
                
                # Compute penalties
                P_triple_out, P_join_in, P_join_out, P_acyclic, P_left_linear, P_entropy = \
                    compute_structure_penalties_batched(edge_index, edge_weights, N_NODES, triples_num, actual_batch_size, device)
                
                L_struct = (
                    lambda_triple_out * P_triple_out
                    + lambda_join_in * P_join_in
                    + lambda_join_out * P_join_out
                    + lambda_acyclic * P_acyclic
                    + lambda_left_linear * P_left_linear
                    + lambda_entropy * P_entropy
                ) / actual_batch_size  # Average over batch
                
                # Anchor loss: supervised on original query plans
                batch_original = Batch.from_data_list(batch_queries)
                anchor_pred = C_theta(batch_original.x.to(device), batch_original.edge_index.to(device), batch=batch_original.batch.to(device))
                anchor_target = torch.log(batch_original.y.to(device))
                L_anchor = anchor_loss_fn(anchor_pred, anchor_target)
                
                # Total loss
                L_total = L_outer + L_anchor + 0.1 * L_struct
                L_total.backward()
                
                accum_count += 1
                if accum_count >= ACCUMULATION_STEPS:
                    grad_norm = torch.nn.utils.clip_grad_norm_(C_theta.parameters(), max_norm=1.0)
                    opt_theta.step()
                    opt_theta.zero_grad(set_to_none=True)
                    accum_count = 0
                
                # Tracking
                epoch_loss += cost_pred_psi.sum().item()
                epoch_penalty += L_struct.item() * actual_batch_size
                epoch_anchor_loss += L_anchor.item() * actual_batch_size
                epoch_samples += actual_batch_size
                
                global_step += 1
                window_loss += cost_pred_psi.mean().item()
                window_penalty += L_struct.item()
                window_anchor_loss += L_anchor.item()
                
                pbar.set_postfix({
                    'loss': f'{cost_pred_psi.mean().item():.4f}',
                    'penalty': f'{L_struct.item():.4f}',
                    'anchor': f'{L_anchor.item():.4f}'
                })
                
                # Log every 100 steps
                if global_step % 100 == 0:
                    avg_loss_per_100_steps.append(window_loss / 100)
                    avg_penalty_per_100_steps.append(window_penalty / 100)
                    avg_anchor_loss_per_100_steps.append(window_anchor_loss / 100)
                    window_loss = 0.0
                    window_penalty = 0.0
                    window_anchor_loss = 0.0
                    
                    # Save plots
                    plt.figure()
                    plt.plot(avg_loss_per_100_steps)
                    plt.xlabel("100-step Window")
                    plt.ylabel("Loss")
                    plt.title(f"Average Loss per 100 Steps (step {global_step})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_directory, 'step_loss_plot.png'))
                    plt.close()
                    
                    plt.figure()
                    plt.plot(avg_penalty_per_100_steps)
                    plt.xlabel("100-step Window")
                    plt.ylabel("Penalty")
                    plt.title(f"Average Penalty per 100 Steps (step {global_step})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_directory, 'step_penalty_plot.png'))
                    plt.close()
                    
                    plt.figure()
                    plt.plot(avg_anchor_loss_per_100_steps)
                    plt.xlabel("100-step Window")
                    plt.ylabel("Anchor Loss")
                    plt.title(f"Average Anchor Loss per 100 Steps (step {global_step})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_directory, 'step_anchor_loss_plot.png'))
                    plt.close()
                
                if global_step % 1000 == 0:
                    torch.save(C_theta.state_dict(), os.path.join(save_directory, f'model_step_{global_step}.pt'))
        
        # End of epoch
        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            avg_penalty = epoch_penalty / epoch_samples
            avg_anchor = epoch_anchor_loss / epoch_samples
        else:
            avg_loss = avg_penalty = avg_anchor = 0.0
        
        average_loss_per_epoch.append(avg_loss)
        average_penalty_per_epoch.append(avg_penalty)
        average_anchor_loss_per_epoch.append(avg_anchor)
        
        for name in hyperparam_history.keys():
            hyperparam_history[name].append(getattr(hyperparams, name)().item())
        
        # Save epoch plots
        plt.figure()
        plt.plot(average_loss_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Loss per Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'loss_plot.png'))
        plt.close()
        
        plt.figure()
        plt.plot(average_penalty_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Average Penalty")
        plt.title("Penalty per Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'penalty_plot.png'))
        plt.close()
        
        plt.figure()
        plt.plot(average_anchor_loss_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Average Anchor Loss")
        plt.title("Anchor Loss per Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'anchor_loss_plot.png'))
        plt.close()
        
        plot_hyperparameter_history(hyperparam_history, save_directory)
        
        torch.save(C_theta.state_dict(), os.path.join(save_directory, f'model_epoch_{epoch}.pt'))
        
        total_epoch_loss = avg_loss + avg_penalty + avg_anchor
        if total_epoch_loss < best_loss:
            best_loss = total_epoch_loss
            torch.save(C_theta.state_dict(), os.path.join(save_directory, 'best_model.pt'))
            hyperparams_dict = {
                name: getattr(hyperparams, name)().item()
                for name in dir(hyperparams)
                if not name.startswith('_')
                and callable(getattr(hyperparams, name))
                and name not in dir(nn.Module)
            }
            with open(os.path.join(save_directory, 'best_hyperparams.json'), 'w') as f:
                json.dump(hyperparams_dict, f, indent=4)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Penalty={avg_penalty:.4f}, Anchor={avg_anchor:.4f}")

