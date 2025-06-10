import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch.optim as optim

from data import Triple, Join, Query, Entity
from model import CostGNNv2
from process_dataset_single_file import SPARQLQuery


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
    # Gumbel-Sigmoid specific hyper-parameters
    init_tau: float = 10.0,
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    return_best: bool = True,
    # NEW: number of independent optimisation restarts before aggregation
    n_restarts: int = 50,
    # NEW: how to aggregate the restarts – 'average' or 'best'
    restart_aggregation: str = 'average',
):
    """Gradient-based join-order search with **Straight-Through Gumbel-Sigmoid**

    The optimisation is now executed in three stages:
    1. *Independent restarts*  –  ``n_restarts`` fully independent runs are
       performed with random initial edge logits.  For each run we keep the
       **raw** (non-thresholded) adjacency obtained by applying the sigmoid
       to the final edge logits.
    2. *Averaging*  –  the collected raw adjacencies are averaged element-wise
       to obtain a prior on promising edges.
    3. *Final run*  –  a last optimisation run is started whose edge logits
       are initialised such that their corresponding sigmoid equals the
       averaged adjacency from step 2.

    The function signature is kept compatible with the previous version so
    that existing code does not need to change.  If ``n_restarts == 0`` the
    behaviour is identical to the original implementation.
    """

    import math  # local to avoid polluting global namespace

    # ---------------------------------------------------------------------
    # Move data & basic bookkeeping                                        
    # ---------------------------------------------------------------------
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n-1 nodes

    # Enumerate all candidate edges (excluding self-loops) -----------------
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)

    # ---------------------------------------------------------------------
    # Helper : *one* optimisation run starting from given edge logits
    # ---------------------------------------------------------------------
    def _single_run(initial_edge_logits: torch.Tensor,
                    collect_history: bool = False):
        """Run one optimisation pass.

        Args
        ----
        initial_edge_logits: 1-D tensor of shape ``[num_edges]`` on *device*.
                             Gradient will *not* flow into this argument.
        collect_history    : Whether to store per-iteration statistics and
                             create plots (used only for the final run).

        Returns
        -------
        final_logits  : Detached tensor containing the (optionally *best*)
                        edge logits after optimisation.
        raw_adj       : ``(N_NODES, N_NODES)`` tensor with *sigmoid(final_logits)*
                        written into the appropriate off-diagonal entries.
        histories     : Tuple of history lists if *collect_history* else None.
        cost_val      : Predicted cost of the plan represented by final_logits
        """
        edge_logits = initial_edge_logits.clone().detach().to(device)
        edge_logits.requires_grad_(True)

        optimiser = optim.AdamW([edge_logits], lr=learning_rate)

        best_cost: float = float('inf')
        best_edge_logits = None

        # histories ------------------------------------------------------
        if collect_history:
            cost_hist, tot_pen_hist = [], []
            acyc_hist = []
            tri_in_hist, tri_out_hist = [], []
            join_in_hist, join_out_hist = [], []
            ent_hist = []
        else:
            cost_hist = tot_pen_hist = acyc_hist = None
            tri_in_hist = tri_out_hist = join_in_hist = join_out_hist = ent_hist = None

        for step in range(optimization_steps):
            optimiser.zero_grad()

            # ---------------------------------------------------------
            # Gumbel-Sigmoid sampling
            # ---------------------------------------------------------
            tau = _temperature_anneal(init_tau, min_tau, tau_decay,
                                      step, optimization_steps)
            edge_weights = sample_binary_concrete(edge_logits, tau)

            # ---------------------------------------------------------
            # Cost prediction
            # ---------------------------------------------------------
            cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

            # ---------------------------------------------------------
            # Build adjacency matrix and compute structural penalties
            # ---------------------------------------------------------
            A = torch.zeros((N_NODES, N_NODES), device=device)
            A[edge_index[0], edge_index[1]] = edge_weights

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

            # entropy penalty
            eps = 1e-10
            probs = torch.sigmoid(edge_logits)
            P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()

            total_penalty = (
                lambda_triple_in * P_triple_in
                + lambda_triple_out * P_triple_out
                + lambda_join_in * P_join_in
                + lambda_join_out * P_join_out
                + lambda_acyclic * P_acyclic
                + lambda_entropy * P_entropy
            )

            # gentle ramp-up for the total penalty
            def _annealed_lam(lam_max, step, ramp_steps=150):
                frac = min(1.0, step / ramp_steps)
                return lam_max * (frac ** 2)

            lambda_total = _annealed_lam(lambda_total_penalty, step, optimization_steps)
            loss = cost_pred + lambda_total * total_penalty

            # keep best (feasible) solution
            if return_best and total_penalty < 30 and cost_pred < best_cost:
                best_cost = cost_pred.item()
                best_edge_logits = edge_logits.clone().detach()

            # histories ------------------------------------------------
            if collect_history:
                cost_hist.append(cost_pred.item())
                tot_pen_hist.append(total_penalty.item())
                acyc_hist.append(P_acyclic.item())
                tri_in_hist.append(P_triple_in.item())
                tri_out_hist.append(P_triple_out.item())
                join_in_hist.append(P_join_in.item())
                join_out_hist.append(P_join_out.item())
                ent_hist.append(P_entropy.item())

            # backward & step ----------------------------------------
            loss.backward()
            optimiser.step()

            if verbose and collect_history and (step + 1) % 100 == 0:
                print(
                    f"Step {step+1}/{optimization_steps}  Cost: {cost_pred.item():.2f}  "
                    f"Penalty: {total_penalty.item():.2f}")

        # -------------------------------------------------------------
        # build *raw* adjacency (sigmoid) from best / last logits
        # -------------------------------------------------------------
        with torch.no_grad():
            final_logits = best_edge_logits if (return_best and best_edge_logits is not None) else edge_logits.detach()
            raw_A = torch.zeros((N_NODES, N_NODES), device=device)
            raw_A[edge_index[0], edge_index[1]] = torch.sigmoid(final_logits)

        # compute predicted cost of the plan represented by final_logits
        with torch.no_grad():
            cost_val = model(data.x, edge_index, edge_weight=torch.sigmoid(final_logits)).item()

        histories = None
        if collect_history:
            histories = (
                cost_hist, tot_pen_hist, acyc_hist,
                tri_in_hist, tri_out_hist, join_in_hist, join_out_hist, ent_hist
            )

        return final_logits, raw_A, histories, cost_val

    # =====================================================================
    # 1) Independent optimisation restarts
    # =====================================================================
    if n_restarts < 0:
        raise ValueError("n_restarts must be non-negative")

    if restart_aggregation not in {'average', 'best'}:
        raise ValueError("restart_aggregation must be either 'average' or 'best'")

    raw_adjs = []
    restart_costs = []
    for run_idx in range(max(0, n_restarts)):
        if verbose:
            print(f"\n▶ Independent restart {run_idx+1}/{n_restarts}")
        init_logits = 0.1 * (torch.rand(num_edges, device=device) - 0.5)  # centred at 0 → p≈0.5 ± noise
        _, raw_A, _, cost_val = _single_run(init_logits, collect_history=False)
        raw_adjs.append(raw_A)
        restart_costs.append(cost_val)

    # =====================================================================
    # 2) Aggregation of restarts
    # =====================================================================
    if not raw_adjs:
        raise ValueError("No raw adjacencies collected – ensure n_restarts > 0")

    if restart_aggregation == 'best':
        # choose plan with minimum predicted cost
        best_idx = int(torch.tensor(restart_costs).argmin().item())
        final_raw_A = raw_adjs[best_idx]
        histories = None  # no final run, hence no detailed history
    else:  # 'average'
        avg_A = torch.stack(raw_adjs, dim=0).mean(dim=0)

        if verbose:
            print("\n▶ Final optimisation run (initialised from averaged adjacency)")

        # convert averaged *edge weights* into logits: log(p/(1-p))
        avg_edge_w = avg_A[edge_index[0], edge_index[1]]
        eps = 1e-6
        avg_edge_w = avg_edge_w.clamp(eps, 1 - eps)
        final_init_logits = torch.log(avg_edge_w) - torch.log(1 - avg_edge_w)

        _, final_raw_A, histories, _ = _single_run(final_init_logits, collect_history=True)

    # ---------------------------------------------------------------------
    # Convert *raw* adjacency to hard (binary) adjacency for downstream code
    # ---------------------------------------------------------------------
    final_A = torch.zeros((N_NODES, N_NODES), device=device)
    final_A[edge_index[0], edge_index[1]] = (final_raw_A[edge_index[0], edge_index[1]] >= 0.5).float()

    # ---------------------------------------------------------------------
    # Plot metrics of the *final* run (only) if requested and available
    # ---------------------------------------------------------------------
    if verbose and histories is not None:
        (cost_hist, tot_pen_hist, acyc_hist,
         tri_in_hist, tri_out_hist, join_in_hist, join_out_hist, ent_hist) = histories
        plot_optimization_metrics(cost_hist, tot_pen_hist, acyc_hist,
                                  tri_in_hist, tri_out_hist, join_in_hist,
                                  join_out_hist, ent_hist)

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


def plot_statistics(stats, show_plots=True, suffix=""):
    """
    Plot statistics about the optimization performance.
    
    Args:
        stats: Dictionary with statistics from evaluate_optimization
        show_plots: Whether to display the plots (if False, only save them)
        suffix: Optional suffix to add to saved filenames (e.g., "_iteration_10")
    """
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
    plt.savefig(f'mean_costs_comparison.png')
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
    plt.savefig(f'cost_distribution_comparison.png')
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
    plt.savefig(f'win_percentage.png')
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
    plt.savefig(f'gradient_vs_greedy.png')
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
    plt.savefig(f'gradient_vs_random.png')
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
    plt.savefig(f'greedy_vs_random.png')
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


def evaluate_optimization(sparql_queries, model_path, num_queries=None, optimization_steps=500, 
                         verbose=False, optimization_params=None):
    """
    Evaluate the optimization algorithm on the given SPARQL queries.
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        model_path: Path to the trained cost model
        num_queries: Number of queries to evaluate (None for all)
        optimization_steps: Number of optimization steps per query
        verbose: Whether to print and plot detailed progress information
        optimization_params: Dictionary of optimization hyperparameters
        
    Returns:
        Statistics about the optimization performance
    """
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
    
    # Process each query
    for i, query in enumerate(tqdm(sparql_queries, desc="Evaluating queries")):
        # Get the torch data from one of the plans
        # For 8TP, we select one of the random plans as the base for optimization
        plan_idx = 0  # Just use the first plan
        torch_data = query.torch_data[plan_idx]
        
        if torch_data is None:
            print(f"Warning: Query {i} has null torch_data for plan {plan_idx}. Skipping.")
            continue
        
        # Prepare the triple objects
        triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in query.triples]
        
        # Run gradient-based optimization
        try:
            if verbose:
                print(f"\nRunning gradient-based optimization for query {i}")
            
            final_adjacency, triples_num = optimize_query_gumbel(
                torch_data, model, device, 
                optimization_steps=optimization_steps, 
                verbose=verbose,
                **optimization_params
            )

            try:
                visualization_dir = "plan_visualizations"
                os.makedirs(visualization_dir, exist_ok=True)
                # Visualize the adjacency matrix
                print("\nVisualizing the optimized adjacency matrix:")
                # Try both layouts
                visualize_adjacency_matrix(final_adjacency, triples_num, visualization_dir, i, use_tree_layout=False)
                visualize_adjacency_matrix(final_adjacency, triples_num, visualization_dir, i, use_tree_layout=True)
                print(f"Saved adjacency matrix visualizations to {visualization_dir}/")
            except Exception as e:
                print(f"Warning: Failed to visualize adjacency matrix: {e}")
            
            # Convert adjacency to query plan
            gradient_plan = adjacency_to_query_with_real_triples(final_adjacency, triples_num, triple_objs)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(gradient_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid gradient plan for query {i}: {validation_msg}")
                print("Skipping this query")
                continue
            
            # Calculate the actual cost using the get_cost method
            gradient_cost = gradient_plan.root.get_cost()
            gradient_costs.append(gradient_cost)

            # Create visualization directory if it doesn't exist

            gradient_plan.visualize(output_file=f"{visualization_dir}/gradient_plan_query_{i}")
            
            if verbose:
                print(f"Gradient optimization complete. Final cost: {gradient_cost}")
                print(f"Saved gradient plan visualization to {visualization_dir}/gradient_plan_query_{i}.png")

                
        except Exception as e:
            print(f"Error in gradient optimization for query {i}: {e}")
            # Skip this query
            #pass
            continue
        #gradient_costs.append(float('30000'))
        
        # Run greedy optimization
        try:
            if verbose:
                print(f"\nRunning greedy optimization for query {i}")
                
            greedy_plan = greedy_optimize_query(
                torch_data, model, triple_objs, device, verbose=verbose
            )
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(greedy_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid greedy plan for query {i}: {validation_msg}")
                greedy_costs.append(float('inf'))
                continue
            
            # Calculate the actual cost
            greedy_cost = greedy_plan.root.get_cost()
            greedy_costs.append(greedy_cost)
            
            if verbose:
                print(f"Greedy optimization complete. Final cost: {greedy_cost}")
                # Visualize the plan if verbose
                visualization_dir = "plan_visualizations"
                os.makedirs(visualization_dir, exist_ok=True)
                greedy_plan.visualize(output_file=f"{visualization_dir}/greedy_plan_query_{i}")
                print(f"Saved greedy plan visualization to {visualization_dir}/greedy_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error in greedy optimization for query {i}: {e}")
            # Use infinity as a placeholder for failed optimizations
            greedy_costs.append(float('inf'))
        
        # Create a random plan
        try:
            if verbose:
                print(f"\nCreating random plan for query {i}")
                
            random_plan = random_join_plan(triple_objs, seed=i)
            
            # Validate that the plan contains all expected triple patterns
            is_valid, validation_msg = validate_plan(random_plan, triple_objs)
            if not is_valid:
                print(f"Warning: Invalid random plan for query {i}: {validation_msg}")
                random_costs.append(float('inf'))
                continue
            
            # Calculate the actual cost
            random_cost = random_plan.root.get_cost()
            random_costs.append(random_cost)
            
            if verbose:
                print(f"Random plan created. Cost: {random_cost}")
                # Visualize the plan if verbose
                visualization_dir = "plan_visualizations"
                os.makedirs(visualization_dir, exist_ok=True)
                random_plan.visualize(output_file=f"{visualization_dir}/random_plan_query_{i}")
                print(f"Saved random plan visualization to {visualization_dir}/random_plan_query_{i}.png")
                
        except Exception as e:
            print(f"Error creating random plan for query {i}: {e}")
            # Use infinity as a placeholder for failed random plans
            random_costs.append(float('inf'))
        
        # Print progress every 5 queries and generate plots
        if (i + 1) % 1 == 0:
            print(f"\nProcessed {i+1}/{len(sparql_queries)} queries")
            if gradient_costs:
                print(f"Average gradient cost: {np.mean(gradient_costs):.2f}")
            if greedy_costs:
                print(f"Average greedy cost: {np.mean(greedy_costs):.2f}")
            if random_costs:
                print(f"Average random cost: {np.mean(random_costs):.2f}")
                
            # Create intermediate statistics and save plots (without displaying)
            intermediate_stats = {
                'gradient_costs': gradient_costs,
                'greedy_costs': greedy_costs,
                'random_costs': random_costs
            }
            # Save plots without showing them, with a suffix indicating the iteration
            plot_statistics(intermediate_stats, show_plots=False, suffix=f"_iter_{i+1}")
            print(f"Saved intermediate plots at iteration {i+1}")
    
    # Calculate statistics
    stats = {
        'gradient_costs': gradient_costs,
        'greedy_costs': greedy_costs,
        'random_costs': random_costs
    }
    
    return stats

def prefilter_queries(queries, limit=1000):
    """
    Prefilter queries to remove those that have a cardinality smaller than limit.
    Takes the first plan from each query and checks its cardinality.
    
    Args:
        queries: List of SPARQLQuery objects
        limit: Minimum cardinality threshold
        
    Returns:
        List of filtered SPARQLQuery objects where first plan has cardinality >= limit
    """
    filtered_queries = []
    total = len(queries)
    
    print(f"\nPrefiltering {total} queries (cardinality threshold: {limit})...")
    
    for i, query in enumerate(queries):
        try:
            # Get first plan's cardinality - each SPARQLQuery has join_plans
            first_plan = query.join_plans[0]  # Take first plan
            cardinality = first_plan.root.get_cardinality()
            print(f"Query {i} has cardinality {cardinality}")
            
            if cardinality >= limit:
                filtered_queries.append(query)
                
        except Exception as e:
            print(f"Warning: Error checking cardinality for query {i}: {e}")
            continue
            
    kept = len(filtered_queries)
    removed = total - kept
    print(f"Kept {kept}/{total} queries ({removed} removed)")
    print(f"Filtering rate: {removed/total*100:.1f}%")
    
    return filtered_queries


if __name__ == "__main__":
    # Configuration for optimization
    config = {
        # General parameters
        'queries_file': "sparql_queries_4_single/queries.pkl",
        'model_path': "/home/tim/query_optimization/explicit_join_model/models/join_plus_tp_prediction_all_sizes.pt",
        'num_queries': 300,
        'optimization_steps': 100,
        'verbose': False,
        
        # Query optimization hyperparameters
        'optimization_params': {
            # Optimizer parameters
            'learning_rate': 10,
            
            # Penalty weights
            'lambda_acyclic': 1000.0,    # Weight for acyclicity penalty
            'lambda_triple_in': 1000.0,  # Weight for triple in-degree penalty
            'lambda_triple_out': 1000.0, # Weight for triple out-degree penalty
            'lambda_join_in': 500.0,     # Weight for join in-degree penalty300
            'lambda_join_out': 1000.0,   # Weight for join out-degree penalty
            'lambda_entropy': 0.0,     # Weight for entropy penalty
            'lambda_total_penalty': 1.  # Overall weight for the total penalty
        }
    }
    
    # Print configuration
    print("Running optimization with the following configuration:")
    print(f"Number of queries: {config['num_queries']}")
    print(f"Optimization steps: {config['optimization_steps']}")
    print("Optimization hyperparameters:")
    for param, value in config['optimization_params'].items():
        print(f"  {param}: {value}")
    
    # Load queries
    sparql_queries = load_sparql_queries(config['queries_file'], config['num_queries'])
    
    # Evaluate optimization
    stats = evaluate_optimization(
        sparql_queries, 
        config['model_path'],
        num_queries=config['num_queries'],
        optimization_steps=config['optimization_steps'],
        verbose=config['verbose'],
        optimization_params=config['optimization_params']
    )

    
    # Plot final statistics with display
    plot_statistics(stats, show_plots=True)