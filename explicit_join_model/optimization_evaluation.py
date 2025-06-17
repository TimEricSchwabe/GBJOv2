import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

# Import the classes
from create_data.process_dataset_with_subplans_individual import SPARQLQuery

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
import itertools  # <-- NEW: needed for exhaustive permutation search

import torch.optim as optim

from data import Triple, Join, Query, Entity
from model import CostGNNv2
from create_data.process_dataset_single_file import SPARQLQuery

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


def optimize_query_gumbel_BACKUP(
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
    
def sample_grouped_gumbel_softmax(edge_logits: torch.Tensor,
                                  src_nodes: torch.Tensor,
                                  temperature: float) -> torch.Tensor:
    """Return relaxed one-hot edge weights such that every *source* node
    emits exactly one outgoing edge (in expectation) using the Gumbel-Softmax
    trick.

    Args
    ----
    edge_logits : Tensor of shape (E,)
        Unconstrained logits of every candidate edge.
    src_nodes   : Tensor of shape (E,)
        Source node index for each edge (aligned with edge_logits).
    temperature : float
        Positive softmax temperature τ.

    Returns
    -------
    Tensor of shape (E,) – edge weights in (0,1) summing to 1 for every
    set of edges that share the same source node.
    """
    device = edge_logits.device
    edge_weights = torch.empty_like(edge_logits)

    for v in torch.unique(src_nodes):
        mask = (src_nodes == v)
        logits_group = edge_logits[mask]
        g = sample_gumbel(logits_group.shape, device=device)
        edge_weights[mask] = torch.softmax((logits_group + g) / temperature, dim=0)

    return edge_weights


@torch.no_grad()
def _temperature_anneal(init_tau: float, min_tau: float, decay: float, step: int, max_step: int) -> float:
    """Exponential temperature annealing every step."""
    return max(min_tau, init_tau - (init_tau - min_tau) * (step / max_step))

    #return max(min_tau, init_tau * (decay ** step)) #exponential deacy

    #return 1.





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
    # Gumbel-Sigmoid specific hyper-parameters
    init_tau: float = 10.0,
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = True,
    min_penalty_threshold: float = 30.0,
    use_lambda_ramping: bool = True,
    logit_sampling: str = 'sigmoid',  # 'sigmoid', 'softmax' or 'dual-softmax'
    # Animation parameters
    save_animation_data: bool = False,
    animation_save_interval: int = 10,
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

    # Optimised parameters: edge logits ------------------------------------------------
    edge_logits = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)
    # Second slot only needed for dual-slot variant
    edge_logits_slot2 = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)

    # Optimiser ------------------------------------------------------------------------
    if logit_sampling == 'dual-softmax':
        optimiser = optim.AdamW([edge_logits, edge_logits_slot2], lr=learning_rate)
    else:
        optimiser = optim.AdamW([edge_logits], lr=learning_rate)
    
    # Track best solution if return_best is True
    best_cost = float('inf')
    best_edge_logits = None
    best_edge_logits_slot2 = None

    # Tracking metrics for plotting -------------------------------------------
    cost_history = []
    total_penalty_history = []
    acyclic_penalty_history = []
    triple_in_penalty_history = []
    triple_out_penalty_history = []
    join_in_penalty_history = []
    join_out_penalty_history = []
    entropy_penalty_history = []

    # Animation data storage ---------------------------------------------------
    animation_data = {
        'edge_weights_history': [],
        'step_numbers': [],
        'edge_index': edge_index.cpu(),
        'n_nodes': N_NODES,
        'triples_num': triples_num,
        'cost_history': [],
        'penalty_history': []
    } if save_animation_data else None

    for step in range(optimization_steps):
        optimiser.zero_grad()

        # Gumbel-based edge sampling ----------------------------------------------------
        if use_temperature_annealing:
            tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps)
        else:
            tau = init_tau

        if logit_sampling == 'dual-softmax':
            # -------------------------------------------------------------
            # Dual-slot: every join node picks *two* incoming edges
            # -------------------------------------------------------------
            masked_logits_1 = edge_logits.clone()
            masked_logits_2 = edge_logits_slot2.clone()
            # Invalid edge types ------------------------------------------------
            triple_to_triple = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits_1[triple_to_triple] = float('-inf')
            masked_logits_2[triple_to_triple] = float('-inf')
            join_to_triple = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits_1[join_to_triple] = float('-inf')
            masked_logits_2[join_to_triple] = float('-inf')
            # slot-wise grouped softmax BY TARGET (only for join targets)
            join_target_mask = (edge_index[1] >= triples_num)
            slot1 = torch.zeros_like(edge_logits)
            slot2 = torch.zeros_like(edge_logits)

            # Sample only on join targets to avoid NaNs for empty groups
            slot1[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits_1[join_target_mask], edge_index[1][join_target_mask], tau)
            slot2[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits_2[join_target_mask], edge_index[1][join_target_mask], tau)
            
            edge_weights = slot1 + slot2  # relaxed 2-hot (values in (0,2))
            # Ensure root join has no outgoing edges
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        elif logit_sampling == 'softmax':            # Mask out invalid edges before softmax sampling
            masked_logits = edge_logits.clone()
            
            # Triple nodes cannot connect to other triple nodes
            triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits[triple_to_triple_mask] = float('-inf')
            
            # Join nodes cannot connect to triple nodes
            join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits[join_to_triple_mask] = float('-inf')
            
            # Use grouped Gumbel-Softmax for exactly one outgoing edge per source node
            edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], tau)
            # Root (final join) should have *no* outgoing edge
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        else:
            # Use original Binary Concrete (Gumbel-Sigmoid) sampling
            edge_weights = sample_binary_concrete(edge_logits, tau)
        
        # Save animation data if enabled ----------------------------------------
        if save_animation_data and step % animation_save_interval == 0:
            # Clamp edge weights to [0,1] for consistent visualization
            clamped_weights = torch.clamp(edge_weights, 0.0, 1.0)
            animation_data['edge_weights_history'].append(clamped_weights.detach().cpu().numpy())
            animation_data['step_numbers'].append(step)
        
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

        # Entropy penalty -------------------------------------------------------------
        if logit_sampling == 'dual-softmax':
            eps = 1e-10
            probs1 = slot1.clamp(min=eps)
            probs2 = slot2.clamp(min=eps)
            P_entropy = -(probs1 * torch.log(probs1) + probs2 * torch.log(probs2)).sum()
        elif logit_sampling == 'softmax':
            # For softmax sampling, use entropy of the relaxed edge weights
            eps = 1e-10
            probs = edge_weights.clamp(min=eps)
            P_entropy = -(probs * torch.log(probs)).sum()
        else:
            # For sigmoid sampling, use binary entropy of the edge probabilities
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

        # Save cost and penalty for animation if enabled ----------------------
        if save_animation_data and step % animation_save_interval == 0:
            animation_data['cost_history'].append(cost_pred.item())
            animation_data['penalty_history'].append(total_penalty.item())

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
        if logit_sampling == 'dual-softmax':
            if return_best and total_penalty < min_penalty_threshold and cost_pred < best_cost:
                best_cost = cost_pred
                best_edge_logits = edge_logits.clone().detach()
                best_edge_logits_slot2 = edge_logits_slot2.clone().detach()
        else:
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
    with torch.no_grad():
        if logit_sampling == 'dual-softmax':
            chosen_logits1 = best_edge_logits if (return_best and best_cost < float('inf')) else edge_logits
            chosen_logits2 = best_edge_logits_slot2 if (return_best and best_cost < float('inf')) else edge_logits_slot2
            # Apply same masks
            mask_tt = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            mask_jt = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            chosen_logits1[mask_tt | mask_jt] = float('-inf')
            chosen_logits2[mask_tt | mask_jt] = float('-inf')
            final_edge_weights = torch.zeros(num_edges, device=device)
            for j in torch.unique(edge_index[1]):  # iterate over join-targets
                # skip triple targets
                if j < triples_num:
                    continue
                cand = (edge_index[1] == j)
                # slot 1
                idx1 = torch.argmax(chosen_logits1[cand])
                global_idx1 = torch.where(cand)[0][idx1]
                final_edge_weights[global_idx1] = 1.0
                # slot 2 (allow duplicate -> still 1)
                idx2 = torch.argmax(chosen_logits2[cand])
                global_idx2 = torch.where(cand)[0][idx2]
                final_edge_weights[global_idx2] = 1.0
        elif logit_sampling == 'softmax':
            # For softmax sampling, build final adjacency using hard one-hot selection
            # Apply the same masking as during training
            masked_chosen_logits = edge_logits.clone()
            
            # Triple nodes cannot connect to other triple nodes
            triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_chosen_logits[triple_to_triple_mask] = float('-inf')
            
            # Join nodes cannot connect to triple nodes
            join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_chosen_logits[join_to_triple_mask] = float('-inf')
            
            final_edge_weights = torch.zeros_like(edge_logits)
            for v in torch.unique(edge_index[0]):
                # Skip the root join (must not have outgoing edges)
                if v == (N_NODES - 1):
                    continue
                m = (edge_index[0] == v)
                idx = torch.argmax(masked_chosen_logits[m])
                selected_global_idx = torch.where(m)[0][idx]
                final_edge_weights[selected_global_idx] = 1.0
        else:
            # For sigmoid sampling, use threshold-based hard assignment
            final_edge_weights = (torch.sigmoid(edge_logits) >= 0.5).float()

    # Write hard one-hot selection into adjacency matrix
    final_A = torch.zeros((N_NODES, N_NODES), device=device)
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

    with torch.no_grad():
        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
    predicted_cost_exp = float(np.exp(final_log_cost))

    if save_animation_data:
        return final_A, triples_num, predicted_cost_exp, animation_data
    else:
        return final_A, triples_num, predicted_cost_exp


def optimize_query(query_data, model, device='cpu', *,
                  optimization_steps: int = 500, verbose: bool = True,
                  learning_rate: float = 0.01,
                  lambda_acyclic: float = 1000.0, lambda_triple_in: float = 1000.0,
                  lambda_triple_out: float = 1000.0, lambda_join_in: float = 500.0,
                  lambda_join_out: float = 1000.0, lambda_entropy: float = 100.0,
                  lambda_total_penalty: float = 1.0,
                  # NEW – additional constraints / behaviour
                  lambda_left_linear: float = 1000.0,
                  return_best: bool = True,
                  min_penalty_threshold: float = 30.0,
                  use_lambda_ramping: bool = True,
                  **kwargs):
    """
    Gradient-based join-order optimisation *without* Gumbel-Sigmoid sampling.

    This variant mirrors the behaviour of :func:`optimize_query_gumbel` –
    left-linear tree penalty, entropy penalty, λ-ramping, and best-solution
    tracking – but keeps deterministic continuous edge weights instead of
    sampling from a Binary-Concrete distribution.
    """
    import torch.optim as optim

    # ------------------------------------------------------------------
    # Move data & set-up
    # ------------------------------------------------------------------
    device = torch.device(device)
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2  # n triples ⇒ 2n−1 nodes

    # All candidate directed edges (no self-loops) ---------------------
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)

    # Optimised parameters: edge weights ∈ [0,1] -----------------------
    edge_weights = 0.5 + 0.1 * (torch.rand(num_edges, device=device) - 0.5)
    edge_weights.requires_grad_(True)

    optimiser = optim.AdamW([edge_weights], lr=learning_rate)

    # ------------------------------------------------------------------
    # Book-keeping for best solution & metrics
    # ------------------------------------------------------------------
    best_cost: float = float('inf')
    best_edge_weights = None

    cost_hist, tot_pen_hist = [], []
    acyc_hist = []
    triple_in_hist, triple_out_hist = [], []
    join_in_hist, join_out_hist = [], []
    entropy_hist = []

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------
    for step in range(optimization_steps):
        optimiser.zero_grad()

        # Cost prediction --------------------------------------------------
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # Build adjacency matrix ------------------------------------------
        A = torch.zeros((N_NODES, N_NODES), device=device)
        A[edge_index[0], edge_index[1]] = edge_weights

        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)

        # -------------------------------------------------------------
        # Penalties (same as optimise_query_gumbel)
        # -------------------------------------------------------------
        P_triple_in  = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in    = ((in_deg[join_nodes]   - 2) ** 2).sum()
        P_join_out   = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic    = torch.trace(torch.matrix_exp(A)) - N_NODES

        # Left-linear tree constraint -----------------------------------
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)
        child_join_counts   = A[join_nodes, :][:, join_nodes].sum(0)
        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join   = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)

        # Entropy penalty --------------------------------------------------
        eps = 1e-10
        P_entropy = -(edge_weights * torch.log(edge_weights + eps) +
                      (1 - edge_weights) * torch.log(1 - edge_weights + eps)).sum()

        # Aggregate penalty ----------------------------------------------
        total_penalty = (
            lambda_triple_in   * P_triple_in +
            lambda_triple_out  * P_triple_out +
            lambda_join_in     * P_join_in +
            lambda_join_out    * P_join_out +
            lambda_acyclic     * P_acyclic +
            lambda_entropy     * P_entropy +
            lambda_left_linear * P_left_linear
        )

        # λ-ramping --------------------------------------------------------
        if use_lambda_ramping:
            frac = min(1.0, step / optimization_steps)
            lambda_total = lambda_total_penalty * (frac ** 2)
        else:
            lambda_total = lambda_total_penalty

        loss = cost_pred + lambda_total * total_penalty

        # Track best feasible solution ----------------------------------
        if return_best and total_penalty < min_penalty_threshold and cost_pred < best_cost:
            best_cost = cost_pred.item()
            best_edge_weights = edge_weights.detach().clone()

        # History --------------------------------------------------------
        cost_hist.append(cost_pred.item())
        tot_pen_hist.append(total_penalty.item())
        acyc_hist.append(P_acyclic.item())
        triple_in_hist.append(P_triple_in.item())
        triple_out_hist.append(P_triple_out.item())
        join_in_hist.append(P_join_in.item())
        join_out_hist.append(P_join_out.item())
        entropy_hist.append(P_entropy.item())

        # Optimiser step -------------------------------------------------
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            edge_weights.clamp_(0.0, 1.0)

        if verbose and (step + 1) % 100 == 0:
            print(f"Step {step+1}/{optimization_steps}  Cost: {cost_pred.item():.2f}  Penalty: {total_penalty.item():.2f}")

    # ------------------------------------------------------------------
    # Build final hard adjacency using best solution (if any)
    # ------------------------------------------------------------------
    with torch.no_grad():
        use_weights = best_edge_weights if (return_best and best_edge_weights is not None) else edge_weights
        hard_weights = (use_weights >= 0.5).float()
        final_A = torch.zeros((N_NODES, N_NODES), device=device)
        final_A[edge_index[0], edge_index[1]] = hard_weights

    # Plot metrics ---------------------------------------------------------
    if verbose:
        plot_optimization_metrics(cost_hist, tot_pen_hist, acyc_hist,
                                  triple_in_hist, triple_out_hist,
                                  join_in_hist, join_out_hist, entropy_hist)

    return final_A, triples_num


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
    greedy_query = Query(root=current_plan, triples_num=triples_num)

    with torch.no_grad():
        log_pred_cost = model(current_x, current_edge_index).item()
    predicted_cost_exp = float(np.exp(log_pred_cost))

    return greedy_query, predicted_cost_exp


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





# ----------------------------------------------------------------------
# Dynamic-Programming join-order enumeration 
# ----------------------------------------------------------------------
def dp_leftdeep_best_plan(query_data, model, device="cpu"):
    """
    Return the *predicted-cost–optimal* left-deep join plan for the given
    query under the learnt CostGNN model, using dynamic programming instead
    of factorial exhaustive search.

    Parameters
    ----------
    query_data : torch_geometric.data.Data
        Node-feature matrix x (nTP + nJoin × F) of *one* random plan plus
        triple-count.  We ignore the supplied edges and create our own.
    model      : CostGNNv2
        Trained cost model in eval mode.
    device     : "cpu" | "cuda"
        Device on which to run the CostGNN.

    Returns
    -------
    best_A     : torch.Tensor  (2n-1, 2n-1)  hard 0/1 adjacency matrix
    best_cost  : float         exp(predicted log-cost)
    """
    model.eval()
    data = query_data.to(device)
    n_triples = (data.x.size(0) + 1) // 2
    F = data.x.size(1)

    # ------------------------------------------------------------------
    # Pre-build template node-feature matrix: first n triple features,
    # followed by (n-1) identical join-node features.
    # ------------------------------------------------------------------
    triple_feats = data.x[:n_triples].clone()
    join_feat    = torch.zeros(F, device=device);  join_feat[-1] = 1.0
    join_feats   = join_feat.unsqueeze(0).repeat(n_triples - 1, 1)
    node_feats   = torch.cat([triple_feats, join_feats], dim=0)

    # DP table: key = frozenset({indices of triples}); value = (cost, A)
    dp = {}

    # Level k = 1 : singleton plans (cost = 0, no joins)
    for i in range(n_triples):
        key = frozenset({i})
        dp[key] = (0.0,
                   torch.zeros((2 * n_triples - 1,
                                2 * n_triples - 1),
                               device=device))

    # Levels k = 2 … n_triples
    for k in range(2, n_triples + 1):
        for subset in itertools.combinations(range(n_triples), k):
            S = frozenset(subset)
            best_cost, best_A = float("inf"), None

            # Try every triple as the *last* right child
            for last in subset:
                left_set = S - {last}
                left_cost, left_A = dp[left_set]

                # Build adjacency for (left ⨝ last)
                A = left_A.clone()
                idx_join = n_triples + k - 2            # next free join idx
                # connect children → parent
                #   a) root of left plan
                if len(left_set) == 1:
                    child_left = list(left_set)[0]      # single triple
                else:
                    child_left = n_triples + len(left_set) - 2  # left sub-plan root
                A[child_left, idx_join] = 1.
                #   b) last triple
                A[last, idx_join] = 1.

                # Build edge_index and weights for CostGNN
                src, dst = torch.where(A > 0.5)
                edge_idx = torch.stack([src, dst], dim=0)

                with torch.no_grad():
                    log_pred = model(node_feats, edge_idx).item()
                    pred_cost = float(np.exp(log_pred))

                total_cost = pred_cost

                if total_cost < best_cost:
                    best_cost, best_A = total_cost, A

            dp[S] = (best_cost, best_A)

    full_key = frozenset(range(n_triples))
    return dp[full_key][1], dp[full_key][0]


def exhaustive_leftdeep_best_plan(query_data, model, device="cpu"):
    """
    Return the *predicted-cost–optimal* left-deep join plan for the given
    query under the learnt CostGNN model, using exhaustive search over all
    n! permutations of triple patterns.

    Parameters
    ----------
    query_data : torch_geometric.data.Data
        Node-feature matrix x (nTP + nJoin × F) of *one* random plan plus
        triple-count.  We ignore the supplied edges and create our own.
    model      : CostGNNv2
        Trained cost model in eval mode.
    device     : "cpu" | "cuda"
        Device on which to run the CostGNN.

    Returns
    -------
    best_A     : torch.Tensor  (2n-1, 2n-1)  hard 0/1 adjacency matrix
    best_cost  : float         exp(predicted log-cost)
    """
    model.eval()
    data = query_data.to(device)
    n_triples = (data.x.size(0) + 1) // 2
    F = data.x.size(1)

    # Pre-build template node-feature matrix: first n triple features,
    # followed by (n-1) identical join-node features.
    triple_features = data.x[:n_triples].clone()
    join_feature = torch.zeros(F, device=device)
    join_feature[-1] = 1.0  # mark join node
    join_features = join_feature.unsqueeze(0).repeat(n_triples - 1, 1)
    node_features_template = torch.cat([triple_features, join_features], dim=0)

    best_pred_cost = float('inf')
    best_adj = None

    for perm in itertools.permutations(range(n_triples)):
        perm_tensor = torch.tensor(perm, device=device)
        A_candidate = left_deep_adj_from_perm(perm_tensor).to(device)

        src_e, dst_e = torch.where(A_candidate > 0.5)
        if src_e.numel() == 0:
            continue  # should never happen

        edge_idx = torch.stack([src_e, dst_e], dim=0)

        with torch.no_grad():
            pred_cost_val = model(node_features_template, edge_idx).item()
            pred_cost_val = float(np.exp(pred_cost_val))

        if pred_cost_val < best_pred_cost:
            best_pred_cost = pred_cost_val
            best_adj = A_candidate

    return best_adj, best_pred_cost


def plan_to_string(plan):
    """
    Convert a query plan (Query object) to a string representation.
    
    Args:
        plan: Query object representing a join plan
        
    Returns:
        str: String representation of the plan structure
    """
    def node_to_string(node):
        if isinstance(node, Triple):
            return f"({node.s} {node.p} {node.o})"
        elif isinstance(node, Join):
            left_str = node_to_string(node.left)
            right_str = node_to_string(node.right)
            return f"Join({left_str}, {right_str})"
        else:
            return str(node)
    
    if plan is None:
        return "None"
    
    return node_to_string(plan.root)


def plans_are_equivalent(plan1, plan2):
    """
    Check if two query plans are equivalent, considering that joins are symmetric.
    
    Args:
        plan1: First Query object to compare
        plan2: Second Query object to compare
        
    Returns:
        bool: True if the plans are equivalent, False otherwise
    """
    if plan1 is None or plan2 is None:
        return plan1 == plan2
    
    def normalize_node(node):
        """
        Normalize a node to a canonical form for comparison.
        For joins, we sort the children to handle symmetry.
        """
        if isinstance(node, Triple):
            # For triples, create a normalized representation
            return ('Triple', str(node.s), str(node.p), str(node.o))
        elif isinstance(node, Join):
            # For joins, normalize both children and sort them
            left_norm = normalize_node(node.left)
            right_norm = normalize_node(node.right)
            # Sort to handle join symmetry - smaller one first
            children = sorted([left_norm, right_norm])
            return ('Join', children[0], children[1])
        else:
            return str(node)
    
    # Compare the normalized forms
    try:
        norm1 = normalize_node(plan1.root)
        norm2 = normalize_node(plan2.root)
        return norm1 == norm2
    except Exception:
        # If there's any error in comparison, fall back to False
        return False


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
    
    # Create animation data directory
    animation_data_dir = os.path.join(save_directory, "animation_data")
    os.makedirs(animation_data_dir, exist_ok=True)
    
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
    
    # Initialize statistics
    gradient_costs = []
    greedy_costs = []
    random_costs = []
    predicted_best_costs = []     # NEW: predicted cost of best left-linear plan per query
    true_best_predicted_costs = []  # NEW: true cost of the best-predicted plan
    # NEW arrays for predicted costs of gradient and greedy methods
    predicted_gradient_costs = []
    predicted_greedy_costs = []
    # NEW: exhaustive search results
    predicted_exhaustive_costs = []
    
    # NEW: Detailed results for JSON export
    detailed_results = []
    
    # Process each query
    for i, query in enumerate(tqdm(sparql_queries, desc="Evaluating queries")):
        # Get the torch data from one of the plans
        # For 8TP, we select one of the random plans as the base for optimization
        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]
        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]

        
        if torch_data is None:
            print(f"Warning: Query {i} has null torch_data for plan {plan_idx}. Skipping.")
            continue
        
        # Prepare query triples for JSON
        query_triples = [[str(triple.s), str(triple.p), str(triple.o)] for triple in triple_objs]
        
        # Run DP-based best plan search
        best_adj, best_pred_cost = dp_leftdeep_best_plan(torch_data, model, device)
        
        # Run exhaustive search for comparison
        exhaustive_adj, exhaustive_pred_cost = exhaustive_leftdeep_best_plan(torch_data, model, device)

        best_pred_plan = None
        true_cost_best_pred = float('inf')
        try:
            triples_num = len(triple_objs)
            best_pred_plan = adjacency_to_query_with_real_triples(
                best_adj, triples_num, triple_objs)
            true_cost_best_pred = best_pred_plan.root.get_cost()
        except Exception as e:
            print(f"Warning: Failed to compute true cost for best predicted plan for query {i}: {e}")
            true_cost_best_pred = float('inf')

        # Convert exhaustive plan
        exhaustive_plan = None
        try:
            exhaustive_plan = adjacency_to_query_with_real_triples(
                exhaustive_adj, triples_num, triple_objs)
        except Exception as e:
            print(f"Warning: Failed to convert exhaustive plan for query {i}: {e}")

        # starting timer
        start_time = time.time()
        
        # Track success of each method
        gradient_success = False
        greedy_success = False
        random_success = False
        gradient_cost = float('inf')
        greedy_cost = float('inf')
        random_cost = float('inf')
        grad_pred_cost = float('inf')  
        greedy_pred_cost = float('inf')
        
        # Initialize plan variables
        gradient_plan = None
        greedy_plan = None
        random_plan = None
        
        # ---------------------------------------------------------------   ---
        # Step 2: Run gradient-based optimization
        # ------------------------------------------------------------------
        try:
            if verbose:
                print(f"\nRunning gradient-based optimization for query {i}")
            
            # Handle different return values based on optimization function
            optimization_result = optimization_function(
                torch_data, model, device, 
                optimization_steps=optimization_steps, 
                verbose=verbose,
                **optimization_params
            )
            
            # Handle different return types that now include predicted cost
            if len(optimization_result) == 4:
                final_adjacency, triples_num, grad_pred_cost, animation_data = optimization_result
            elif len(optimization_result) == 3:
                final_adjacency, triples_num, grad_pred_cost = optimization_result
                animation_data = None
            else:
                raise ValueError("Unexpected return tuple from optimization_function")

            # Save animation data to disk if available
            if animation_data is not None:
                animation_file = os.path.join(animation_data_dir, f"query_{i}_animation_data.pkl")
                try:
                    import pickle
                    with open(animation_file, 'wb') as f:
                        pickle.dump(animation_data, f)
                    print(f"Saved animation data to {animation_file}")
                except Exception as e:
                    print(f"Warning: Failed to save animation data: {e}")

            try:
                # Visualize the adjacency matrix
                print("\nVisualizing the optimized adjacency matrix:")
                # Try both layouts
                visualize_adjacency_matrix(final_adjacency, triples_num, visualization_dir, i, use_tree_layout=True)
                print(f"Saved adjacency matrix visualizations to {visualization_dir}/")
            except Exception as e:
                print(f"Warning: Failed to visualize adjacency matrix: {e}")
            
            # Create animation if data is available (but don't do it during evaluation to save time)
            # Animation can be generated later using the saved data
            if animation_data is not None and verbose:
                try:
                    print("Creating optimization animation...")
                    # create_optimization_animation(
                    #     animation_data, 
                    #     visualization_dir, 
                    #     i, 
                    #     fps=10,
                    #     use_tree_layout=True,
                    #     max_edge_weight=2.0  # For dual-softmax which can go up to 2
                    # )
                    print(f"Saved optimization animation to {visualization_dir}/")
                except Exception as e:
                    print(f"Warning: Failed to create optimization animation: {e}")
            
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid gradient plan for query {i}: {validation_msg}")
                print("Skipping this query")
                continue
            
            end_time = time.time()
            print(f"Time taken for gradient optimization: {end_time - start_time:.2f} seconds")

            # Calculate the actual cost using the get_cost method
            gradient_cost = gradient_plan.root.get_cost()
            gradient_success = True

            # Attempt to visualize the plan – if Graphviz fails, continue without stopping
            try:
                gradient_plan.visualize(output_file=f"{visualization_dir}/gradient_plan_query_{i}")
            except Exception as viz_err:
                print(f"Warning: Failed to visualize gradient plan for query {i}: {viz_err}")
            
            if verbose:
                print(f"Gradient optimization complete. Final cost: {gradient_cost}")
                print(f"Saved gradient plan visualization to {visualization_dir}/gradient_plan_query_{i}.png")

                
        except Exception as e:
            #raise e
            print(f"Error in gradient optimization for query {i}: {e}")
            # Skip this query
            continue
        
        # Run greedy optimization
        try:
            if verbose:
                print(f"\nRunning greedy optimization for query {i}")
                
            greedy_plan, greedy_pred_cost = greedy_optimize_query(
                torch_data, model, triple_objs, device, verbose=verbose
            )
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(greedy_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid greedy plan for query {i}: {validation_msg}")
                # Don't append here - we'll handle all appends at the end
                greedy_cost = float('inf')
            else:
                # Calculate the actual cost
                greedy_cost = greedy_plan.root.get_cost()
                greedy_success = True
            
            if verbose:
                print(f"Greedy optimization complete. Final cost: {greedy_cost}")
                if greedy_success:
                    # Visualize the plan if verbose
                    greedy_plan.visualize(output_file=f"{visualization_dir}/greedy_plan_query_{i}")
                    print(f"Saved greedy plan visualization to {visualization_dir}/greedy_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error in greedy optimization for query {i}: {e}")
            # Use infinity as a placeholder for failed optimizations
            greedy_cost = float('inf')
        
        # Create a random plan
        try:
            if verbose:
                print(f"\nCreating random plan for query {i}")
                
            random_plan = random_join_plan(triple_objs, seed=i)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(random_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid random plan for query {i}: {validation_msg}")
                # Don't append here - we'll handle all appends at the end
                random_cost = float('inf')
            else:
                # Calculate the actual cost
                random_cost = random_plan.root.get_cost()
                random_success = True
            
            if verbose:
                print(f"Random plan created. Cost: {random_cost}")
                if random_success:
                    # Visualize the plan if verbose
                    random_plan.visualize(output_file=f"{visualization_dir}/random_plan_query_{i}")
                    print(f"Saved random plan visualization to {visualization_dir}/random_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error creating random plan for query {i}: {e}")
            # Use infinity as a placeholder for failed random plans
            random_cost = float('inf')
        
        # Now append all costs to arrays (synchronized)
        gradient_costs.append(gradient_cost)
        greedy_costs.append(greedy_cost)
        random_costs.append(random_cost)
        predicted_best_costs.append(best_pred_cost)
        true_best_predicted_costs.append(true_cost_best_pred)
        predicted_gradient_costs.append(grad_pred_cost)
        predicted_greedy_costs.append(greedy_pred_cost)
        predicted_exhaustive_costs.append(exhaustive_pred_cost)
        
        # NEW: Create detailed result for this query
        query_result = {
            "query_id": i,
            "query_triples": query_triples,
            "ntriplepattern": len(triple_objs),
            "plans": {
                "exhaustive": {
                    "plan_string": plan_to_string(exhaustive_plan),
                    "real_cost": exhaustive_plan.root.get_cost() if exhaustive_plan else float('inf'),
                    "predicted_cost": float(exhaustive_pred_cost)
                },
                "greedy": {
                    "plan_string": plan_to_string(greedy_plan),
                    "real_cost": float(greedy_cost),
                    "predicted_cost": float(greedy_pred_cost)
                },
                "gradient": {
                    "plan_string": plan_to_string(gradient_plan),
                    "real_cost": float(gradient_cost),
                    "predicted_cost": float(grad_pred_cost)
                },
                "dp": {
                    "plan_string": plan_to_string(best_pred_plan),
                    "real_cost": float(true_cost_best_pred),
                    "predicted_cost": float(best_pred_cost)
                }
            },
            "greedy_equal_exhaustive": plans_are_equivalent(greedy_plan, exhaustive_plan),
            "gradient_equal_exhaustive": plans_are_equivalent(gradient_plan, exhaustive_plan)
        }
        detailed_results.append(query_result)
        
        # Print progress every query
        if (i + 1) % 1 == 0:
            print(f"\nProcessed {i+1}/{len(sparql_queries)} queries")
            if gradient_costs:
                print(f"Median gradient cost: {np.median(gradient_costs):.2f}")
            if greedy_costs:
                print(f"Median greedy cost: {np.median(greedy_costs):.2f}")
            if random_costs:
                print(f"Median random cost: {np.median(random_costs):.2f}")
    
    # Save detailed results to JSON
    detailed_results_file = os.path.join(save_directory, "detailed_results.json")
    with open(detailed_results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Saved detailed results to: {detailed_results_file}")
    
    # Calculate statistics
    stats = {
        'gradient_costs': gradient_costs,
        'greedy_costs': greedy_costs,
        'random_costs': random_costs,
        'predicted_best_costs': predicted_best_costs,
        'true_best_predicted_costs': true_best_predicted_costs,
        'predicted_gradient_costs': predicted_gradient_costs,
        'predicted_greedy_costs': predicted_greedy_costs,
        'predicted_exhaustive_costs': predicted_exhaustive_costs
    }
    # Save plots without showing them, with a suffix indicating the iteration
    print(f"Saving plots to {save_directory}")
    plot_statistics(stats, show_plots=False, save_directory=save_directory)
    
    # Save metadata for animation generation
    animation_metadata = {
        'num_queries': len(sparql_queries),
        'animation_data_dir': animation_data_dir,
        'visualization_dir': visualization_dir,
        'animation_params': {
            'fps': 10,
            'use_tree_layout': True,
            'max_edge_weight': 2.0
        }
    }
    
    metadata_file = os.path.join(save_directory, "animation_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(animation_metadata, f, indent=2)
    
    print(f"\nAnimation data saved to: {animation_data_dir}")
    print(f"Animation metadata saved to: {metadata_file}")
    print(f"To generate animations later, run: python optim_animation.py {save_directory}")
    
    return stats


if __name__ == "__main__":
    # Configuration for optimization

    configold = {
        # General parameters
        'queries_file': "/home/tim/query_optimization/datasets/sparql_path_queries/queries.pkl",
        'model_path': "/home/tim/query_optimization/explicit_join_model/models/path_model.pt",
        'num_queries': 50,
        'optimization_steps': 1000,
        'verbose': False,
        'save_path': "optimization_results",  # Base directory for saving results
        
        # Query optimization hyperparameters
        'optimization_params': {
            # Optimization procedure selection
            'optimization_procedure': 'gumbel',  # 'gumbel' or 'normal'
            
            # Optimizer parameters
            'learning_rate': 1,
            
            # Penalty weights
            'lambda_acyclic': 1000.0,    # Weight for acyclicity penalty
            'lambda_triple_in': 1000.0,  # Weight for triple in-degree penalty
            'lambda_triple_out': 1000.0, # Weight for triple out-degree penalty
            'lambda_join_in': 500.0,     # Weight for join in-degree penalty
            'lambda_join_out': 1000.0,   # Weight for join out-degree penalty
            'lambda_entropy': 0.0,      # Weight for entropy penalty
            'lambda_total_penalty': 1.0, # Overall weight for the total penalty
            'lambda_left_linear': 1000.0, # Weight for left-linear penalty - DISABLE to allow for bushy plans
            
            # Gumbel-Sigmoid specific parameters
            'init_tau': 10.0,            # Initial temperature for Gumbel-Sigmoid
            'min_tau': 1.0,              # Minimum temperature for Gumbel-Sigmoid
            'tau_decay': 0.999,          # Temperature decay rate
            'use_temperature_annealing': True,  # Whether to use temperature annealing
            
            # Solution selection and penalty ramping
            'return_best': False,         # Whether to return best feasible solution
            'min_penalty_threshold': 30.0,  # Minimum penalty for accepting a solution
            'use_lambda_ramping': True,  # Whether to ramp up lambda_total_penalty
            
            # Sampling method selection
            'logit_sampling': 'dual-softmax',  # 'sigmoid', 'softmax' or 'dual-softmax',
        }
    }



    config = {
        # General parameters
        'queries_file': "/home/tim/query_optimization/datasets/sparql_queries_path_4_tp/queries.pkl",
        'model_path': "/home/tim/query_optimization/explicit_join_model/models/path_model.pt",
        'num_queries': 100,
        'optimization_steps': 1746,
        'verbose': False,
        'save_path': "optimization_results",  # Base directory for saving results
        
        # Query optimization hyperparameters
        'optimization_params': {
            # Optimization procedure selection
            'optimization_procedure': 'gumbel',  # 'gumbel' or 'normal'
            
            # Optimizer parameters
            'learning_rate': 0.133,
            
            # Penalty weights
            'lambda_acyclic': 2065.0,    # Weight for acyclicity penalty
            'lambda_triple_in': 2390.0,  # Weight for triple in-degree penalty
            'lambda_triple_out': 105.0, # Weight for triple out-degree penalty
            'lambda_join_in': 387.0,     # Weight for join in-degree penalty
            'lambda_join_out': 2610.0,   # Weight for join out-degree penalty
            'lambda_entropy': 0.0,      # Weight for entropy penalty
            'lambda_total_penalty': 1.0, # Overall weight for the total penalty
            'lambda_left_linear': 3290.0, # Weight for left-linear penalty
            
            # Gumbel-Sigmoid specific parameters
            'init_tau': 8.2,            # Initial temperature for Gumbel-Sigmoid
            'min_tau': 1.0,              # Minimum temperature for Gumbel-Sigmoid
            'tau_decay': 0.976,          # Temperature decay rate
            'use_temperature_annealing': True,  # Whether to use temperature annealing
            
            # Solution selection and penalty ramping
            'return_best': True,         # Whether to return best feasible solution
            'min_penalty_threshold': 30.0,  # Minimum penalty for accepting a solution
            'use_lambda_ramping': True,  # Whether to ramp up lambda_total_penalty
            
            # Sampling method selection
            'logit_sampling': 'dual-softmax',  # 'sigmoid', 'softmax' or 'dual-softmax',

            # Animation parameters
            'save_animation_data': False,    # Whether to save data for creating animations
            'animation_save_interval': 10,   # Save animation data every N steps
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




