import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from datetime import datetime

import torch.optim as optim

from data import Triple, Join, Query, Entity
from model import CostGNNv2
from process_dataset_single_file import SPARQLQuery

import time










# ---------------- helpers ----------------------------------------------------
def sample_gumbel(shape, eps=1e-10, device="cpu"):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def sinkhorn(log_alpha, iters=20):          # log_alpha: (n,n)
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()                  # doubly-stochastic

def gumbel_sinkhorn(L, tau, iters=20):
    g = sample_gumbel(L.shape, device=L.device)
    return sinkhorn((L + g) / tau, iters)

def left_deep_adj_from_perm(pi):
    """
    pi: Tensor of length n with the (0-based) permutation of triple nodes.
    Returns A (2n-1, 2n-1) adjacency for a left-deep tree:
       (((T_pi0 ▷◁ T_pi1) ▷◁ T_pi2) … )
    """
    n = len(pi)
    N = 2 * n - 1
    A = torch.zeros(N, N, dtype=torch.float32)
    # indices: triple 0..n-1, join nodes n..2n-2 (root = 2n-2)
    # first join joins pi0 and pi1 -> node idx = n
    A[pi[0], n] = 1.0
    A[pi[1], n] = 1.0
    last_join = n
    for k in range(2, n):
        new_join = n + k - 1
        A[last_join, new_join] = 1.0
        A[pi[k],  new_join] = 1.0
        last_join = new_join
    return A

@torch.no_grad()
def _anneal_tau(init_tau, min_tau, step, max_step):
    return max(min_tau, init_tau * (0.95 ** step))





def load_sparql_queries(queries_file: str, num_queries):
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
        return sparql_queries[-num_queries:]
    print(f"Loaded {len(sparql_queries)} SPARQL queries from {queries_file}")
    return sparql_queries

def sample_binary_concrete(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample from the Binary Concrete (Gumbel‑Sigmoid) distribution using the
    re‑parameterisation trick and return a straight‑through sample.
    Args:
        logits: Raw, unconstrained edge logits (shape: [num_edges]).
        temperature: Positive temperature τ controlling smoothness.
    Returns:
        edge_weights: Straight‑through hard sample in [0,1] with gradients.
    """
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    y_soft = torch.sigmoid((logits + gumbel) / temperature)
    hard=False
    if hard:
        y_hard = (y_soft >= 0.5).float()
        # Straight‑through estimator: replace forward value with hard, keep soft gradient
        return y_hard.detach() - y_soft.detach() + y_soft
    else:
        return y_soft


@torch.no_grad()
def _temperature_anneal(init_tau: float, min_tau: float, decay: float, step: int, max_step: int) -> float:
    """Exponential temperature annealing every step."""
    return max(min_tau, init_tau - (init_tau - min_tau) * (step / max_step))

    #return max(min_tau, init_tau * (decay ** step)) #exponential deacy

    #return 1.



def optimize_queries_gumbel_batch(
    queries_data,
    model,
    device: str = "cpu",
    *,
    optimization_steps: int = 500,
    verbose: bool = False,
    learning_rate: float = 0.01,
    lambda_acyclic: float = 1000.0,
    lambda_triple_in: float = 1000.0,
    lambda_triple_out: float = 1000.0,
    lambda_join_in: float = 500.0,
    lambda_join_out: float = 1000.0,
    lambda_entropy: float = 10.0,
    lambda_total_penalty: float = 1.0,
    lambda_left_linear: float = 1000.0,
    # Gumbel-Sigmoid hyper-params
    init_tau: float = 10.0,
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = False,
    min_penalty_threshold: float = 30.0,
    use_lambda_ramping: bool = True
):
    """Optimize a *list* of query graphs **in a single mini-batch**.

    The graph structure constraints (acyclicity, degree, left-linear) are
    evaluated per query; the total loss is the mean across the batch which
    allows a single backwards() pass to update all *edge_logits* at once.
    """

    from torch_geometric.data import Batch

    B = len(queries_data)
    assert B > 0, "queries_data must contain at least one Data object"

    # ------------------------------------------------------------------
    # Pre-compute per-graph candidate edges and bookkeeping tensors
    # ------------------------------------------------------------------
    edge_indices_local = []        # per graph (2,E_i)
    n_nodes_list = []             # N_i
    n_triples_list = []           # T_i
    edge_slices = []              # (start,end) into *edge_logits*

    total_edges = 0
    total_nodes = 0
    for g_idx, g in enumerate(queries_data):
        N = len(g.x)
        T = (N + 1) // 2

        # all directed edges sans self-loops
        src, dst = torch.where(~torch.eye(N, dtype=torch.bool))
        e_idx = torch.stack([src, dst], dim=0)

        edge_indices_local.append(e_idx)
        n_nodes_list.append(N)
        n_triples_list.append(T)

        edge_slices.append((total_edges, total_edges + e_idx.size(1)))
        total_edges += e_idx.size(1)
        total_nodes += N

    # ------------------------------------------------------------------
    # Build *global* structures for the batched forward on CostGNNv2
    # ------------------------------------------------------------------
    node_offsets = torch.tensor([0] + list(np.cumsum(n_nodes_list)[:-1]), dtype=torch.long)

    global_edge_index_parts = []
    for off, e_idx in zip(node_offsets, edge_indices_local):
        global_edge_index_parts.append(e_idx + off.unsqueeze(0))
    global_edge_index = torch.cat(global_edge_index_parts, dim=1).to(device)

    # Mini-batch of node features (concatenate) & batch vector ----------
    batch_data = Batch.from_data_list(queries_data).to(device)

    # ------------------------------------------------------------------
    # Optimised parameters: *one* big tensor for all edge logits
    # ------------------------------------------------------------------
    edge_logits = 0.1 * (torch.rand(total_edges, device=device) - 0.5)
    edge_logits.requires_grad_(True)

    optimiser = optim.AdamW([edge_logits], lr=learning_rate)

    for step in range(optimization_steps):
        optimiser.zero_grad()

        # Temperature / Gumbel-Sigmoid sampling -------------------------
        tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps) if use_temperature_annealing else init_tau
        edge_weights = sample_binary_concrete(edge_logits, tau)

        # ------------------------------------------------------------------
        # Cost prediction – single forward pass on the whole batch
        # ------------------------------------------------------------------
        cost_preds = model(batch_data.x, global_edge_index, edge_weight=edge_weights, batch=batch_data.batch)

        # Ensure we have B scalar costs
        if cost_preds.dim() == 0:  # corner-case B==1, squeeze()
            cost_preds = cost_preds.unsqueeze(0)

        # ------------------------------------------------------------------
        # PER-GRAPH structural penalties & aggregation
        # ------------------------------------------------------------------
        total_penalty = 0.0
        ptr_nodes = 0
        for g_idx in range(B):
            N = n_nodes_list[g_idx]
            T = n_triples_list[g_idx]
            E_start, E_end = edge_slices[g_idx]

            # local adjacency
            A = torch.zeros((N, N), device=device)
            local_edges = edge_indices_local[g_idx]
            local_weights = edge_weights[E_start:E_end]
            A[local_edges[0], local_edges[1]] = local_weights

            in_deg, out_deg = A.sum(0), A.sum(1)
            triple_nodes = torch.arange(T, device=device)
            join_nodes = torch.arange(T, N, device=device)
            root = N - 1
            non_root_joins = torch.arange(T, root, device=device)

            P_triple_in  = (in_deg[triple_nodes] ** 2).sum()
            P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
            P_join_in    = ((in_deg[join_nodes] - 2) ** 2).sum()
            P_join_out   = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
            P_acyclic    = torch.trace(torch.matrix_exp(A)) - N

            # left-linear penalty --------------------------------------
            child_triples = A[:T, :][:, join_nodes].sum(0)
            child_joins   = A[join_nodes, :][:, join_nodes].sum(0)
            if len(join_nodes) > 0:
                P_first = (child_triples[0] - 2) ** 2 + child_joins[0] ** 2
                if len(join_nodes) > 1:
                    P_rest_t = ((child_triples[1:] - 1) ** 2).sum()
                    P_rest_j = ((child_joins[1:] - 1) ** 2).sum()
                    P_left_lin = P_first + P_rest_t + P_rest_j
                else:
                    P_left_lin = P_first
            else:
                P_left_lin = torch.tensor(0.0, device=device)

            eps = 1e-10
            probs = torch.sigmoid(edge_logits[E_start:E_end])
            P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()

            total_penalty += (
                lambda_triple_in * P_triple_in
                + lambda_triple_out * P_triple_out
                + lambda_join_in * P_join_in
                + lambda_join_out * P_join_out
                + lambda_acyclic * P_acyclic
                + lambda_entropy * P_entropy
                + lambda_left_linear * P_left_lin
            )

        # Normalise by batch size -------------------------------------
        total_penalty = total_penalty / B
        loss = cost_preds.mean() + lambda_total_penalty * total_penalty

        loss.backward()
        optimiser.step()

        if verbose and (step + 1) % 100 == 0:
            print(f"[BatchOpt] step {step+1}/{optimization_steps}  loss={loss.item():.2f}")

    # ------------------------------------------------------------------
    # Build final hard adjacency matrices per query
    # ------------------------------------------------------------------
    final_adjs = []
    with torch.no_grad():
        hard_weights = (torch.sigmoid(edge_logits) >= 0.5).float()
        for g_idx in range(B):
            N = n_nodes_list[g_idx]
            A = torch.zeros((N, N), device=device)
            E_start, E_end = edge_slices[g_idx]
            local_edges = edge_indices_local[g_idx]
            local_hard = hard_weights[E_start:E_end]
            A[local_edges[0], local_edges[1]] = local_hard
            final_adjs.append(A)

    return final_adjs, n_triples_list










def optimize_query_gumbel(
    query_data,
    model,
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
    # Enforce left-deep / linear join tree structure
    lambda_left_linear: float = 1000.0,
    # Gumbel‑Sigmoid specific hyper‑parameters
    init_tau: float = 10.0,
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = True,
    min_penalty_threshold: float = 30.0,
    use_lambda_ramping: bool = True,
):
    """Gradient-based join-order search with **Straight-Through Gumbel-Sigmoid**.

    The signature and return values mirror `optimize_query()` so the rest of
    your code remains unchanged.
    """
    # Move data ----------------------------------------------------------------
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n‑1 nodes

    # Enumerate all candidate edges (excluding self‑loops) ----------------------
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)

    # Optimised parameters: edge logits (initially 0 ⇒ p≈0.5) ------------------
    edge_logits = torch.zeros(num_edges, device=device, requires_grad=True)
    edge_logits = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)

    # Optimiser ----------------------------------------------------------------
    optimiser = optim.AdamW([edge_logits], lr=learning_rate)
    
    # Track best solution if return_best is True
    best_cost = float('inf')
    best_edge_logits = None

    # Tracking metrics for plotting -------------------------------------------
    cost_history = []
    total_penalty_history = []
    acyclic_penalty_history = []
    triple_in_penalty_history = []
    triple_out_penalty_history = []
    join_in_penalty_history = []
    join_out_penalty_history = []
    entropy_penalty_history = []

    for step in range(optimization_steps):
        optimiser.zero_grad()

        # Gumbel‑Sigmoid sampling ---------------------------------------------
        if use_temperature_annealing:
            tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps)
        else:
            tau = init_tau
            
        edge_weights = sample_binary_concrete(edge_logits, tau)

        # Cost prediction ------------------------------------------------------
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # Build adjacency ------------------------------------------------------
        A = torch.zeros((N_NODES, N_NODES), device=device)
        A[edge_index[0], edge_index[1]] = edge_weights

        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)

        # Structural penalties -------------------------------------------------
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES

        # -------------------------------------------------------------
        # Additional constraint: enforce left-deep / linear join order
        # -------------------------------------------------------------
        # For a valid left-deep tree with n triples and join nodes
        #   J_n, J_{n+1}, …, J_{2n−2} (root = J_{2n−2}) we expect:
        #     • J_n  : exactly 2 triple children and 0 join children
        #     • J_{n+k>n}: exactly 1 triple child and 1 join child
        # The existing degree-based penalties already ensure every join
        # has in-degree 2, out-degree ≤1 etc.  Here we explicitly check
        # the *composition* of its children so that no bushy shapes can
        # occur.

        # Count, for every join node, how many of its incoming edges stem
        # from triple nodes vs. join nodes ("children").
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)   # (#joins,)
        child_join_counts   = A[join_nodes, :][:, join_nodes].sum(0)      # (#joins,)

        if len(join_nodes) > 0:  # Guard against trivial 0-TP queries
            # (1) first join (index 0 in join_nodes): [2 triple, 0 join]
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2

            # (2) remaining joins:           [1 triple, 1 join]
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join   = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)

        # Entropy penalty (optional) ------------------------------------------
        eps = 1e-10
        probs = torch.sigmoid(edge_logits)
        P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()

        # Aggregate -----------------------------------------------------------
        total_penalty = (
            lambda_triple_in * P_triple_in
            + lambda_triple_out * P_triple_out
            + lambda_join_in * P_join_in
            + lambda_join_out * P_join_out
            + lambda_acyclic * P_acyclic
            + lambda_entropy * P_entropy
            + lambda_left_linear * P_left_linear
        )

        # Lambda ramping logic ------------------------------------------------
        if use_lambda_ramping:
            def annealed_lam(lam_max, step, ramp_steps=150):
                frac = min(1.0, step / ramp_steps)
                return lam_max * (frac ** 2)  
            
            lambda_total = annealed_lam(lambda_total_penalty, step, ramp_steps=optimization_steps)
        else:
            lambda_total = lambda_total_penalty

        loss = cost_pred + lambda_total * total_penalty

        # Track best solution if return_best is True
        if return_best and total_penalty < min_penalty_threshold and cost_pred < best_cost:
            best_cost = cost_pred
            best_edge_logits = edge_logits.clone().detach()

        # Track metrics for plotting
        cost_history.append(cost_pred.item())
        total_penalty_history.append(total_penalty.item())
        acyclic_penalty_history.append(P_acyclic.item())
        triple_in_penalty_history.append(P_triple_in.item())
        triple_out_penalty_history.append(P_triple_out.item())
        join_in_penalty_history.append(P_join_in.item())
        join_out_penalty_history.append(P_join_out.item())
        entropy_penalty_history.append(P_entropy.item())

        # Back‑prop & step -----------------------------------------------------
        loss.backward()
        
        # Clip gradients
        #torch.nn.utils.clip_grad_norm_([edge_logits], max_norm=5.0)
        
        optimiser.step()

        # Log ------------------------------------------------------------------
        if verbose and (step + 1) % 100 == 0:
            print(
                f"Step {step+1}/{optimization_steps}  "
                f"Cost: {cost_pred.item():.2f}  Penalty: {total_penalty.item():.2f}  "
            )

    # Final hard adjacency -----------------------------------------------------
    final_A = torch.zeros((N_NODES, N_NODES), device=device)
    with torch.no_grad():
        if return_best and best_edge_logits is not None:
            final_edge_weights = (torch.sigmoid(best_edge_logits) >= 0.5).float()
        else:
            final_edge_weights = (torch.sigmoid(edge_logits) >= 0.5).float()
        final_A[edge_index[0], edge_index[1]] = final_edge_weights

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

    return final_A, triples_num



def optimize_query(query_data, model, device='cpu', optimization_steps=500, verbose=True,
                  learning_rate=0.01, lambda_acyclic=1000.0, lambda_triple_in=1000.0, 
                  lambda_triple_out=1000.0, lambda_join_in=500.0, lambda_join_out=1000.0, 
                  lambda_entropy=100.0, lambda_total_penalty=1.0):
    """
    Run the optimization algorithm for a query.
    
    Args:
        query_data: The torch geometric data for the query
        model: The trained cost model
        device: Device to run the optimization on
        optimization_steps: Number of optimization steps
        verbose: Whether to print progress information
        learning_rate: Learning rate for the optimizer
        lambda_acyclic: Weight for acyclicity penalty
        lambda_triple_in: Weight for triple in-degree penalty
        lambda_triple_out: Weight for triple out-degree penalty
        lambda_join_in: Weight for join in-degree penalty
        lambda_join_out: Weight for join out-degree penalty
        lambda_entropy: Weight for entropy penalty
        lambda_total_penalty: Weight for the total penalty
        
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
    optimizer_opt = optim.AdamW([edge_weights], lr=learning_rate)
    
    # Tracking metrics for plotting
    cost_history = []
    total_penalty_history = []
    acyclic_penalty_history = []
    triple_in_penalty_history = []
    triple_out_penalty_history = []
    join_in_penalty_history = []
    join_out_penalty_history = []
    entropy_penalty_history = []
    
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
        total_penalty = lambda_triple_in * P_triple_in + \
                        lambda_triple_out * P_triple_out + \
                        lambda_join_in * P_join_in + \
                        lambda_join_out * P_join_out + \
                        lambda_entropy * P_entropy + \
                        lambda_acyclic * P_acyclic
        
        # Total loss
        loss = cost_pred + lambda_total_penalty * total_penalty
        
        # Track metrics for plotting
        cost_history.append(cost_pred.item())
        total_penalty_history.append(total_penalty.item())
        acyclic_penalty_history.append(P_acyclic.item())
        triple_in_penalty_history.append(P_triple_in.item())
        triple_out_penalty_history.append(P_triple_out.item())
        join_in_penalty_history.append(P_join_in.item())
        join_out_penalty_history.append(P_join_out.item())
        entropy_penalty_history.append(P_entropy.item())
        
        # Backward pass and optimization step
        loss.backward()
        optimizer_opt.step()
        
        # Clamp edge weights to [0,1]
        with torch.no_grad():
            edge_weights.clamp_(0, 1)
            
        if verbose and (step + 1) % 100 == 0:
            print(f'Step {step+1}/{optimization_steps}, Cost: {cost_pred.item():.2f}, Penalty: {total_penalty.item():.2f}')

    # Final adjacency matrix
    final_adjacency = A.detach().clone()
    
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
    After picking the first triple pattern, every further candidate is
    evaluated by creating a new join node that the current (sub-)plan
    root and the candidate triple both point to.
    """
    import torch                                               # (local import keeps global namespace clean)

    model.eval()
    triples_num = len(original_triples)
    
    if verbose:
        print("Starting greedy query optimization")
        print(f"Number of triple patterns: {triples_num}")
    
    # ------------------------------------------------------------------
    # Helper: build a graph consisting of the current plan + new triple
    # ------------------------------------------------------------------
    def build_join_graph(curr_x, curr_edge_index, curr_root_idx, candidate_feat):
        """
        curr_x            : node feature matrix of current plan
        curr_edge_index   : edge index of current plan
        curr_root_idx     : index of the root node of the current plan
        candidate_feat    : (1, F) feature tensor of the triple to be added

        returns:
            new_x, new_edge_index, new_root_idx
        """
        # (1) new join node feature  (all-zeros + last dim = 1 to mark join)
        join_feat = torch.zeros_like(candidate_feat)
        join_feat[..., -1] = 1.0

        # (2) concatenate features   [ current | candidate | join ]
        new_x = torch.cat([curr_x, candidate_feat, join_feat], dim=0)

        cand_node_idx = curr_x.size(0)          # position of the new triple node
        join_node_idx = cand_node_idx + 1       # position of the new join node

        # (3) copy existing edges and add two new ones (child → parent)
        additional_edges = torch.tensor(
            [[curr_root_idx, cand_node_idx],    # sources  (children)
             [join_node_idx, join_node_idx]],   # targets  (parent - join)
            dtype=torch.long,
            device=device
        )

        if curr_edge_index.numel() == 0:
            new_edge_index = additional_edges
        else:
            new_edge_index = torch.cat([curr_edge_index, additional_edges], dim=1)

        return new_x, new_edge_index, join_node_idx

    # ------------------------------------------------------------------
    # Step 1 : choose the cheapest single triple
    # ------------------------------------------------------------------
    original_features = query_data.x[:triples_num].clone().to(device)

    choose_random = False # todo !!
    if choose_random:
        best_first_idx = random.randrange(triples_num)
        with torch.no_grad():
            best_first_cost = model(original_features[best_first_idx:best_first_idx + 1],
                                  torch.zeros((2, 0), dtype=torch.long, device=device)).item()
    else:
        best_first_cost, best_first_idx = float('inf'), -1
        for i in range(triples_num):
            with torch.no_grad():
                cost = model(original_features[i:i + 1],
                             torch.zeros((2, 0), dtype=torch.long, device=device)).item()
            if cost < best_first_cost:
                best_first_cost, best_first_idx = cost, i

    if verbose:
        print(f"Initial best triple: {best_first_idx} (cost={best_first_cost:.4f})")

    # initialise current plan ------------------------------------------------
    current_x = original_features[best_first_idx:best_first_idx + 1]           # one node
    current_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)  # no edges yet
    current_root_idx = 0                                                       # only node is root
    current_plan = original_triples[best_first_idx]

    remaining_triples = list(range(triples_num))
    remaining_triples.remove(best_first_idx)

    # ------------------------------------------------------------------
    # Greedily add triples one by one
    # ------------------------------------------------------------------
    while remaining_triples:
        best_cost, best_idx = float('inf'), -1
        best_x = best_edge_index = None
        best_root_idx = None

        for cand_idx in remaining_triples:
            cand_feat = original_features[cand_idx:cand_idx + 1]

            # build graph with extra join
            new_x, new_edge_index, new_root_idx = build_join_graph(
                current_x, current_edge_index, current_root_idx, cand_feat
            )

            # predict cost
            with torch.no_grad():
                cost = model(new_x, new_edge_index).item()

            if cost < best_cost:
                best_cost = cost
                best_idx = cand_idx
                best_x = new_x
                best_edge_index = new_edge_index
                best_root_idx = new_root_idx

        # update current state with the best candidate -----------------
        current_x = best_x
        current_edge_index = best_edge_index
        current_root_idx = best_root_idx
        current_plan = Join(left=current_plan, right=original_triples[best_idx])

        remaining_triples.remove(best_idx)
        
        if verbose:
            print(f"Joined triple {best_idx}  ->  new cost {best_cost:.4f}  |  {len(remaining_triples)} remaining")
    
    # wrap everything into a Query object
    return Query(root=current_plan, triples_num=triples_num)


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
    plt.savefig(os.path.join(save_directory, f'mean_costs_comparison{suffix}.png'))
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


def count_triples_in_plan(plan):
    """
    Count the number of triple patterns in a query plan.
    
    Args:
        plan: Query object representing a join plan
        
    Returns:
        int: The number of triple patterns in the plan
    """
    def traverse_count(node):
        if isinstance(node, Triple):
            return 1
        elif isinstance(node, Join):
            return traverse_count(node.left) + traverse_count(node.right)
        else:
            return 0
    
    return traverse_count(plan.root)


def collect_triples_in_plan(plan):
    """
    Collect all triple patterns in a query plan.
    
    Args:
        plan: Query object representing a join plan
        
    Returns:
        list: All triple patterns in the plan
    """
    triples = []
    
    def traverse_collect(node):
        if isinstance(node, Triple):
            triples.append(node)
        elif isinstance(node, Join):
            traverse_collect(node.left)
            traverse_collect(node.right)
    
    traverse_collect(plan.root)
    return triples


def validate_plan(plan, expected_triples):
    """
    Validate that a query plan contains all expected triple patterns.
    
    Args:
        plan: Query object representing a join plan
        expected_triples: List of Triple objects that should be in the plan
        
    Returns:
        tuple: (is_valid, message) 
               where is_valid is a boolean and message is a description of any issues
    """
    # Check if the plan has the right number of triples
    triples_in_plan = collect_triples_in_plan(plan)
    
    if len(triples_in_plan) != len(expected_triples):
        return False, f"Plan has {len(triples_in_plan)} triples but expected {len(expected_triples)}"
    
    # Check if all expected triples are in the plan
    # Create a simple string representation for comparison
    plan_triple_strs = set(str(t) for t in triples_in_plan)
    expected_triple_strs = set(str(t) for t in expected_triples)
    
    if plan_triple_strs != expected_triple_strs:
        missing = expected_triple_strs - plan_triple_strs
        extra = plan_triple_strs - expected_triple_strs
        message = ""
        if missing:
            message += f"Missing triples: {missing}"
        if extra:
            message += f"Unexpected triples: {extra}"
        return False, message
    
    return True, "Plan is valid"


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


def optimize_query_gumbel_lbfgs(
    query_data,
    model,
    device: str = "cpu",
    *,
    optimization_steps: int = 500,
    verbose: bool = True,
    learning_rate: float = 0.1,
    lbfgs_max_iter: int = 20,
    lambda_acyclic: float = 1000.0,
    lambda_triple_in: float = 1000.0,
    lambda_triple_out: float = 1000.0,
    lambda_join_in: float = 500.0,
    lambda_join_out: float = 1000.0,
    lambda_entropy: float = 10.0,
    lambda_total_penalty: float = 1.0,
    # Enforce left-deep / linear join tree structure
    lambda_left_linear: float = 1000.0,
    # Gumbel-Sigmoid specific hyper-parameters
    init_tau: float = 10.0,
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    return_best: bool = True,
):
    """Gradient-based join-order search using **LBFGS** instead of AdamW.

    This function is largely identical to :pyfunc:`optimize_query_gumbel` but
    replaces the AdamW optimiser with PyTorch's LBFGS. The signature is kept
    compatible so it can be used as a drop-in replacement.
    """
    import torch.optim as optim  # local import keeps global namespace clean

    # ------------------------------------------------------------------
    # Move data & bookkeeping
    # ------------------------------------------------------------------
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2

    # All directed candidate edges (no self-loops) ---------------------
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)

    # Optimised parameters: edge logits (≈0 ⇒ p≈0.5) -------------------
    edge_logits = 0.1 * (torch.rand(num_edges, device=device) - 0.5)
    edge_logits.requires_grad_(True)

    # ------------------------------------------------------------------
    # LBFGS optimiser with more conservative line search
    # ------------------------------------------------------------------
    optimiser = optim.LBFGS([edge_logits], lr=learning_rate,
                           max_iter=lbfgs_max_iter,
                           max_eval=None,  # Allow more func evals
                           tolerance_grad=1e-5,
                           tolerance_change=1e-7,
                           history_size=100,
                           line_search_fn='strong_wolfe')

    best_cost = float('inf')
    best_edge_logits = None
    best_penalty = float('inf')
    no_improve_count = 0
    max_no_improve = 5  # Early stopping if no improvement for this many steps

    # Metric histories for optional plotting ---------------------------
    cost_hist = []
    tot_pen_hist = []
    acyc_hist = []
    triple_in_hist = []
    triple_out_hist = []
    join_in_hist = []
    join_out_hist = []
    entropy_hist = []
    had_nan = False

    def _compute_loss(tau, step_idx):
        """Helper to compute loss & penalties given current *edge_logits*."""
        nonlocal best_cost, best_edge_logits, best_penalty, had_nan
        
        edge_weights = sample_binary_concrete(edge_logits, tau)
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # Check for NaN values early
        if torch.isnan(cost_pred):
            had_nan = True
            raise ValueError("Cost prediction returned NaN")

        # Build adjacency
        A = torch.zeros((N_NODES, N_NODES), device=device)
        A[edge_index[0], edge_index[1]] = edge_weights

        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)

        # Penalties
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES

        # -------------------------------------------------------------
        # Additional constraint: enforce left-deep / linear join order
        # -------------------------------------------------------------
        # For a valid left-deep tree with n triples and join nodes
        #   J_n, J_{n+1}, …, J_{2n−2} (root = J_{2n−2}) we expect:
        #     • J_n  : exactly 2 triple children and 0 join children
        #     • J_{n+k>n}: exactly 1 triple child and 1 join child
        # The existing degree-based penalties already ensure every join
        # has in-degree 2, out-degree ≤1 etc.  Here we explicitly check
        # the *composition* of its children so that no bushy shapes can
        # occur.

        # Count, for every join node, how many of its incoming edges stem
        # from triple nodes vs. join nodes ("children").
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)   # (#joins,)
        child_join_counts   = A[join_nodes, :][:, join_nodes].sum(0)      # (#joins,)

        if len(join_nodes) > 0:  # Guard against trivial 0-TP queries
            # (1) first join (index 0 in join_nodes): [2 triple, 0 join]
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2

            # (2) remaining joins:           [1 triple, 1 join]
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join   = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)

        # Entropy penalty for binary decisions
        def entropy_penalty(weights, temperature=1.0):
            epsilon = 1e-10
            return -torch.sum(weights * torch.log(weights + epsilon) + 
                            (1 - weights) * torch.log(1 - weights + epsilon)) * temperature
        
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
        total_penalty = (
            lambda_triple_in * P_triple_in
            + lambda_triple_out * P_triple_out
            + lambda_join_in * P_join_in
            + lambda_join_out * P_join_out
            + lambda_acyclic * P_acyclic
            + lambda_entropy * P_entropy
            + lambda_left_linear * P_left_linear
        )

        # Check for NaN in penalties
        if torch.isnan(total_penalty):
            had_nan = True
            raise ValueError("Penalty computation returned NaN")

        # Gentle ramp-up for penalty weight ----------------------------
        frac = min(1.0, step_idx / optimization_steps)
        lambda_total = lambda_total_penalty * (frac ** 2)

        loss_val = cost_pred + lambda_total * total_penalty

        # Track best feasible solution ---------------------------------
        curr_cost = float(cost_pred.item())
        curr_penalty = float(total_penalty.item())
        
        if return_best and curr_penalty < best_penalty and curr_cost < best_cost:
            best_cost = curr_cost
            best_penalty = curr_penalty
            best_edge_logits = edge_logits.clone().detach()

        # Store metrics for plotting -----------------------------------
        cost_hist.append(curr_cost)
        tot_pen_hist.append(curr_penalty)
        acyc_hist.append(float(P_acyclic.item()))
        triple_in_hist.append(float(P_triple_in.item()))
        triple_out_hist.append(float(P_triple_out.item()))
        join_in_hist.append(float(P_join_in.item()))
        join_out_hist.append(float(P_join_out.item()))
        entropy_hist.append(float(P_entropy.item()))

        return loss_val

    # ------------------------------------------------------------------
    # Outer loop – we call LBFGS.step() *optimization_steps* times
    # ------------------------------------------------------------------
    try:
        for step in range(optimization_steps):
            tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps)

            def closure():
                optimiser.zero_grad()
                try:
                    loss_val = _compute_loss(tau, step)
                    if torch.isnan(loss_val):
                        raise ValueError("Loss computation returned NaN")
                    loss_val.backward()
                    return loss_val
                except Exception as e:
                    print(f"Error in closure at step {step}: {e}")
                    return None

            try:
                closure_result = closure()
                if closure_result is None or had_nan:
                    print("Optimization failed due to NaN values")
                    break
                    
                optimiser.step(closure)
            except RuntimeError as e:
                print(f"LBFGS step failed at step {step}: {e}")
                break

            # Early stopping check
            if len(cost_hist) > 1 and abs(cost_hist[-1] - cost_hist[-2]) < 1e-4:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= max_no_improve:
                if verbose:
                    print(f"Early stopping at step {step+1} - no improvement")
                break

            if verbose and (step + 1) % 100 == 0:
                print(f"Step {step+1}/{optimization_steps}  Cost: {cost_hist[-1]:.2f}  Penalty: {tot_pen_hist[-1]:.2f}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted - using best solution found so far")
    except Exception as e:
        print(f"Optimization failed with error: {e}")

    # ------------------------------------------------------------------
    # Build final *hard* adjacency
    # ------------------------------------------------------------------
    with torch.no_grad():
        use_logits = best_edge_logits if (return_best and best_edge_logits is not None) else edge_logits
        hard_weights = (torch.sigmoid(use_logits) >= 0.5).float()
        final_A = torch.zeros((N_NODES, N_NODES), device=device)
        final_A[edge_index[0], edge_index[1]] = hard_weights

    # Plot optimisation metrics (if requested) -------------------------
    if verbose and len(cost_hist) > 1 and not had_nan:  # Only plot if we have valid history
        try:
            plot_optimization_metrics(cost_hist, tot_pen_hist, acyc_hist,
                                      triple_in_hist, triple_out_hist,
                                      join_in_hist, join_out_hist, entropy_hist)
        except Exception as e:
            print(f"Failed to plot optimization metrics: {e}")

    if had_nan:
        raise ValueError("Optimization failed due to NaN values")

    return final_A, triples_num


def evaluate_optimization(sparql_queries, model_path, num_queries=None, optimization_steps=500, 
                         verbose=False, optimization_params=None, optimization_function=None, save_directory="."):
    """
    Evaluate the optimization algorithm on the given SPARQL queries.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model_path: Path to the trained cost model
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        verbose: Whether to print and plot detailed progress information
        optimization_params: Dictionary of optimization hyperparameters
        optimization_function: Function to use for optimization (optimize_query_gumbel or optimize_query)
        save_directory: Directory to save all outputs to
        
    Returns:
        Statistics about the optimization performance
    """
    # Set default optimization function if not provided
    if optimization_function is None:
        optimization_function = optimize_query_gumbel
    #optimization_function = optimize_query_gumbel_rnn
    
    # Create visualization directory
    visualization_dir = os.path.join(save_directory, "plan_visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
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
    
    # ------------------------------------------------------------------
    # Batched processing -------------------------------------------------
    # ------------------------------------------------------------------
    batch_size = optimization_params.get("batch_size", 1)
    opt_params_clean = {k: v for k, v in optimization_params.items() if k != "batch_size"}

    gradient_costs, greedy_costs, random_costs = [], [], []

    total_queries = len(sparql_queries)
    batch_start_idx = 0

    while batch_start_idx < total_queries:
        batch_queries = sparql_queries[batch_start_idx: batch_start_idx + batch_size]

        # starting timer
        start_time = time.time()
        # ---------------------------
        # Prepare batch inputs
        # ---------------------------
        batch_torch_data = []
        batch_triple_objs = []
        valid_indices_in_batch = []  # keep mapping for skipped ones

        for local_idx, query in enumerate(batch_queries):
            plan_idx = 0
            td = query.torch_data[plan_idx]
            if td is None:
                print(f"Warning: Query {batch_start_idx+local_idx} has null torch_data. Skipping.")
                continue
            batch_torch_data.append(td)
            batch_triple_objs.append([
                Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples
            ])
            valid_indices_in_batch.append(local_idx)

        # If nothing valid in this batch simply move on ----------------
        if not batch_torch_data:
            batch_start_idx += batch_size
            continue

        # ---------------------------
        # Run BATCHED gradient search
        # ---------------------------
        if optimization_function == optimize_query_gumbel and len(batch_torch_data) > 1:
            final_adjs, triples_nums = optimize_queries_gumbel_batch(
                batch_torch_data, model, device,
                optimization_steps=optimization_steps,
                verbose=verbose,
                **opt_params_clean
            )
        else:
            # Fallback (batch_size==1 or non-gumbel optimisation)
            final_adjs, triples_nums = [], []
            for td in batch_torch_data:
                adj, tnum = optimization_function(td, model, device,
                                                  optimization_steps=optimization_steps,
                                                  verbose=verbose,
                                                  **opt_params_clean)
                final_adjs.append(adj)
                triples_nums.append(tnum)

        # ---------------------------
        # Post-processing each query sequentially
        # ---------------------------
        end_time = time.time()
        print(f"Time taken for batch processing: {end_time - start_time:.2f} seconds")
        for inner_idx, (adj, tnum, triple_objs) in enumerate(zip(final_adjs, triples_nums, batch_triple_objs)):
            global_idx = batch_start_idx + valid_indices_in_batch[inner_idx]

            # ----- adjacency ➜ plan / cost -----
            try:
                grad_plan = adjacency_to_query_with_real_triples(adj, tnum, triple_objs)
                is_valid, msg = validate_plan(grad_plan, triple_objs)
                if not is_valid:
                    print(f"Invalid gradient plan for query {global_idx}: {msg}")
                    continue
                gradient_costs.append(grad_plan.root.get_cost())
            except Exception as e:
                print(f"Gradient optimisation failed for query {global_idx}: {e}")
                continue

            # ----- greedy heuristic -----
            try:
                greedy_plan = greedy_optimize_query(batch_torch_data[inner_idx], model, triple_objs, device, verbose=False)
                is_valid, msg = validate_plan(greedy_plan, triple_objs)
                if not is_valid:
                    greedy_costs.append(float('inf'))
                else:
                    greedy_costs.append(greedy_plan.root.get_cost())
            except Exception as e:
                print(f"Greedy optimisation failed for query {global_idx}: {e}")
                greedy_costs.append(float('inf'))

            # ----- random plan -----
            try:
                random_plan = random_join_plan(triple_objs, seed=global_idx)
                is_valid, msg = validate_plan(random_plan, triple_objs)
                if not is_valid:
                    random_costs.append(float('inf'))
                else:
                    random_costs.append(random_plan.root.get_cost())
            except Exception as e:
                print(f"Random plan failed for query {global_idx}: {e}")
                random_costs.append(float('inf'))

        # progress & plotting every batch
        print(f"Processed {min(batch_start_idx + batch_size, total_queries)}/{total_queries} queries")
        batch_start_idx += batch_size

        stats_current = {
            'gradient_costs': gradient_costs,
            'greedy_costs': greedy_costs,
            'random_costs': random_costs
        }
        plot_statistics(stats_current, show_plots=False, save_directory=save_directory)

    # finished all batches --------------------------------------------------
    stats = {
        'gradient_costs': gradient_costs,
        'greedy_costs': greedy_costs,
        'random_costs': random_costs
    }
    return stats


if __name__ == "__main__":
    # Configuration for optimization
    config = {
        # General parameters
        'queries_file': "/home/tim/query_optimization/datasets/sparql_queries_path_4/queries.pkl",
        'model_path': "/home/tim/query_optimization/explicit_join_model/models/path_model.pt",
        'num_queries': 50,
        'optimization_steps': 2000,
        'verbose': False,
        'save_path': "optimization_results",  # Base directory for saving results
        
        # Query optimization hyperparameters
        'optimization_params': {
            # Optimization procedure selection
            'optimization_procedure': 'gumbel',  # 'gumbel' or 'normal'
            
            # Optimizer parameters
            'learning_rate': 10,
            'batch_size': 2,
            
            # Penalty weights
            'lambda_acyclic': 1000.0,    # Weight for acyclicity penalty
            'lambda_triple_in': 1000.0,  # Weight for triple in-degree penalty
            'lambda_triple_out': 1000.0, # Weight for triple out-degree penalty
            'lambda_join_in': 500.0,     # Weight for join in-degree penalty
            'lambda_join_out': 1000.0,   # Weight for join out-degree penalty
            'lambda_entropy': 10.0,      # Weight for entropy penalty
            'lambda_total_penalty': 1.0, # Overall weight for the total penalty
            'lambda_left_linear': 1000.0, # Weight for left-linear penalty
            
            # Gumbel-Sigmoid specific parameters
            'init_tau': 10.0,            # Initial temperature for Gumbel-Sigmoid
            'min_tau': 1.0,              # Minimum temperature for Gumbel-Sigmoid
            'tau_decay': 0.999,          # Temperature decay rate
            'use_temperature_annealing': True,  # Whether to use temperature annealing
            
            # Solution selection and penalty ramping
            'return_best': True,         # Whether to return best feasible solution
            'min_penalty_threshold': 30.0,  # Minimum penalty for accepting a solution
            'use_lambda_ramping': True,  # Whether to ramp up lambda_total_penalty
        }
    }
    
    # Create unique save directory based on datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(config['save_path'], f"run_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Saving all results to: {save_directory}")
    
    # Save configuration to JSON file
    config_copy = config.copy()
    config_copy['save_directory'] = save_directory
    config_copy['timestamp'] = timestamp
    with open(os.path.join(save_directory, "config.json"), 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    # Print configuration
    print("Running optimization with the following configuration:")
    print(f"Number of queries: {config['num_queries']}")
    print(f"Optimization steps: {config['optimization_steps']}")
    print("Optimization hyperparameters:")
    for param, value in config['optimization_params'].items():
        print(f"  {param}: {value}")
    
    # Load queries
    sparql_queries = load_sparql_queries(config['queries_file'], config['num_queries'])
    
    # Select optimization function based on config
    optimization_procedure = config['optimization_params'].pop('optimization_procedure')
    if optimization_procedure == 'gumbel':
        optimization_function = optimize_query_gumbel
    else:  # 'normal'
        optimization_function = optimize_query
    
    # Evaluate optimization
    stats = evaluate_optimization(
        sparql_queries, 
        config['model_path'],
        num_queries=config['num_queries'],
        optimization_steps=config['optimization_steps'],
        verbose=config['verbose'],
        optimization_params=config['optimization_params'],
        optimization_function=optimization_function,
        save_directory=save_directory
    )
    
    # Calculate final statistics
    final_stats = {
        'gradient': {
            'mean': float(np.mean(stats['gradient_costs'])),
            'median': float(np.median(stats['gradient_costs'])),
            'std': float(np.std(stats['gradient_costs'])),
            'min': float(np.min(stats['gradient_costs'])),
            'max': float(np.max(stats['gradient_costs']))
        },
        'greedy': {
            'mean': float(np.mean(stats['greedy_costs'])),
            'median': float(np.median(stats['greedy_costs'])),
            'std': float(np.std(stats['greedy_costs'])),
            'min': float(np.min(stats['greedy_costs'])),
            'max': float(np.max(stats['greedy_costs']))
        },
        'random': {
            'mean': float(np.mean(stats['random_costs'])),
            'median': float(np.median(stats['random_costs'])),
            'std': float(np.std(stats['random_costs'])),
            'min': float(np.min(stats['random_costs'])),
            'max': float(np.max(stats['random_costs']))
        },
        'ratios': {
            'gradient_to_random_mean': float(np.mean(np.array(stats['gradient_costs']) / np.array(stats['random_costs']))),
            'greedy_to_random_mean': float(np.mean(np.array(stats['greedy_costs']) / np.array(stats['random_costs']))),
            'gradient_to_greedy_mean': float(np.mean(np.array(stats['gradient_costs']) / np.array(stats['greedy_costs'])))  
        },
        'win_rates': {
            'gradient_vs_random': float(np.sum(np.array(stats['gradient_costs']) < np.array(stats['random_costs'])) / len(stats['gradient_costs']) * 100),
            'greedy_vs_random': float(np.sum(np.array(stats['greedy_costs']) < np.array(stats['random_costs'])) / len(stats['greedy_costs']) * 100)
        }
    }
    
    # Save final statistics to JSON file
    with open(os.path.join(save_directory, "final_statistics.json"), 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    print(f"Gradient - Mean: {final_stats['gradient']['mean']:.2f}, Median: {final_stats['gradient']['median']:.2f}")
    print(f"Greedy - Mean: {final_stats['greedy']['mean']:.2f}, Median: {final_stats['greedy']['median']:.2f}")
    print(f"Random - Mean: {final_stats['random']['mean']:.2f}, Median: {final_stats['random']['median']:.2f}")
    print(f"Gradient win rate vs Random: {final_stats['win_rates']['gradient_vs_random']:.1f}%")
    print(f"Greedy win rate vs Random: {final_stats['win_rates']['greedy_vs_random']:.1f}%")
    
    # Plot final statistics with display
    plot_statistics(stats, show_plots=True, save_directory=save_directory)
    
    print(f"\nAll results saved to: {save_directory}")
    print(f"- Configuration: config.json")
    print(f"- Final statistics: final_statistics.json") 
    print(f"- Plots: *.png files")
    print(f"- Plan visualizations: plan_visualizations/ subdirectory")


