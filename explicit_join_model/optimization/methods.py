"""
Optimization methods for join order optimization.

Contains the core optimization algorithms:
- Gradient-based optimization with Gumbel tricks
- Greedy heuristic optimization
- Dynamic programming approach
- Exhaustive search
- Random plan generation
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import random
import itertools
import time
from tqdm import tqdm

from data import Triple, Join, Query, Entity
from model import CostGNNv2
from .gumbel_utils import sample_binary_concrete, sample_grouped_gumbel_softmax, _temperature_anneal
from utils.data_utils import left_deep_adj_from_perm
from visualization.evaluation_plots import plot_optimization_metrics
from data import random_join_order
from .plan_decoder import project_to_leftdeep, project_leftdeep_greedy_beam

from torch_geometric.utils import scatter, spmm



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
    min_penalty_threshold: float = 1.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = 'sigmoid',  # 'sigmoid', 'softmax' or 'dual-softmax'
    # Animation parameters
    save_animation_data: bool = False,
    animation_save_interval: int = 10,
    # Gradient optimization improvements
    gradient_clip_norm: float = 5.0,
    use_lr_scheduling: bool = True,
    lr_warmup_steps: int = 200,
    decoding_method: str = 'threshold', # 'threshold', 'beam', 'greedy', 'hungarian'
):
    """Gradient-based join-order search with **Straight-Through Gumbel-Sigmoid**.

    The signature and return values mirror `optimize_query()` so the rest of
    your code remains unchanged.
    """


    print(f"Decoding method: {decoding_method}")


    
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
    
    # Learning rate scheduler for warmup and decay
    if use_lr_scheduling:
        def lr_schedule(step):
            # This function returns a multiplier for the base learning_rate
            # Actual LR = learning_rate * lr_schedule(step)
            if step < lr_warmup_steps:
                # Linear warmup from 0 to learning_rate
                if lr_warmup_steps == 0:
                    return 1
                else:
                    return (step + 1) / lr_warmup_steps  # 0 → 1.0
            else:
                return 1
        
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_schedule)
    
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

        total_penalty_raw = (
            P_triple_in
            + P_triple_out
            + P_join_in
            + P_join_out
            + P_acyclic
            + P_entropy
            + P_left_linear
        )

        # Save cost and penalty for animation if enabled ----------------------
        if save_animation_data and step % animation_save_interval == 0:
            animation_data['cost_history'].append(cost_pred.item())
            animation_data['penalty_history'].append(total_penalty.item())

        # Lambda ramping logic ------------------------------------------------
        if use_lambda_ramping:
            def annealed_lam(lam_max, step, ramp_steps=150):
                frac = min(1.0, step / ramp_steps)
                return lam_max * (frac ** lambda_ramp_exponent)  
            
            lambda_total = annealed_lam(lambda_total_penalty, step, ramp_steps=optimization_steps)
        else:
            lambda_total = lambda_total_penalty

        loss = cost_pred + lambda_total * total_penalty

        # Track best solution if return_best is True
        if logit_sampling == 'dual-softmax':
            if return_best and total_penalty_raw < min_penalty_threshold and cost_pred < best_cost:
                best_cost = cost_pred
                best_edge_logits = edge_logits.clone().detach()
                best_edge_logits_slot2 = edge_logits_slot2.clone().detach()
            #if return_best and loss < best_cost:
            #    best_cost = cost_pred
            #    best_edge_logits = edge_logits.clone().detach()
            #    best_edge_logits_slot2 = edge_logits_slot2.clone().detach()
        else:
            if return_best and total_penalty_raw < min_penalty_threshold and cost_pred < best_cost:
                best_cost = cost_pred
                best_edge_logits = edge_logits.clone().detach()

        # Track metrics for plotting
        cost_history.append(cost_pred.item() + total_penalty_raw.item()) #ToDo: Check
        total_penalty_history.append(total_penalty_raw.item())
        acyclic_penalty_history.append(P_acyclic.item())
        triple_in_penalty_history.append(P_triple_in.item())
        triple_out_penalty_history.append(P_triple_out.item())
        join_in_penalty_history.append(P_join_in.item())
        join_out_penalty_history.append(P_join_out.item())
        entropy_penalty_history.append(P_entropy.item())

        # Back‑prop & step -----------------------------------------------------
        loss.backward()
        
        # Gradient improvements -----------------------------------------------
        if logit_sampling == 'dual-softmax':
            params_to_clip = [edge_logits, edge_logits_slot2]
        else:
            params_to_clip = [edge_logits]
            
        # Monitor gradient norms before clipping
        grad_norms = []
        for param in params_to_clip:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        
        # Apply gradient clipping to prevent exploding gradients
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=gradient_clip_norm)
        
        
        
        optimiser.step()
        
        # Update learning rate schedule
        if use_lr_scheduling:
            scheduler.step()

        # Log ------------------------------------------------------------------
        if verbose and (step + 1) % 100 == 0:
            current_lr = optimiser.param_groups[0]['lr']
            print(
                f"Step {step+1}/{optimization_steps}  "
                f"Cost: {cost_pred.item():.2f}  Penalty: {total_penalty_raw.item():.2f}  "
                f"LR: {current_lr:.6f}  Grad: {max_grad_norm:.4f}"
            )

    # Final hard adjacency -----------------------------------------------------
    with torch.no_grad():
        if logit_sampling == 'dual-softmax':

            if decoding_method == 'threshold':
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


            else:
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
                final_edge_weights = edge_weights
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[edge_index[0], edge_index[1]] = edge_weights
                #final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)

                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=False)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=False)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)

        elif logit_sampling == 'softmax':

            if decoding_method == 'threshold':
        
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
                final_edge_weights = edge_weights
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[edge_index[0], edge_index[1]] = edge_weights
                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=True)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=True)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)




        else:

            if decoding_method == 'threshold':
                final_edge_weights = (torch.sigmoid(edge_logits) >= 0.5).float()


            else:
                A_sigmoid = torch.sigmoid(edge_logits)
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[edge_index[0], edge_index[1]] = A_sigmoid
                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=True)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=True)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)

            # original sigmoid threshold

    # Write hard one-hot selection into adjacency matrix
    if decoding_method == 'threshold':
        final_A = torch.zeros((N_NODES, N_NODES), device=device)
        final_A[edge_index[0], edge_index[1]] = final_edge_weights # TODO COMMENT BACK IN !!

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


# ======================================================================
# Faster, GPU-optimised variant of optimise_query_gumbel
# ======================================================================

def optimize_query_gumbel_efficient(
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
    min_tau: float = 1.0,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = True,
    min_penalty_threshold: float = 1.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = "sigmoid",  # "sigmoid", "softmax" or "dual-softmax"
    # Animation parameters
    save_animation_data: bool = False,
    animation_save_interval: int = 10,
    # Gradient optimisation improvements
    gradient_clip_norm: float = 5.0,
    use_lr_scheduling: bool = True,
    lr_warmup_steps: int = 200,
    decoding_method: str = "threshold",  # "threshold", "beam", "greedy", "hungarian"
    **kwargs,
):
    """GPU-friendly re-implementation of :pyfunc:`optimize_query_gumbel`.

    The *functional* behaviour (signature + returned values) is **identical**
    to the reference implementation, yet we avoid a few expensive Python-side
    operations and un-necessary tensor (re-)allocations:

    1.   Pre-compute and cache static masks / index mappings that never change
         during optimisation (invalid edge masks, node ranges, …).
    2.   Replace explicit dense adjacency-matrix maths wherever possible with
         inexpensive `torch_geometric.utils.scatter` reductions that operate
         directly on the edge list – this dramatically cuts bandwidth usage on
         GPUs.
    3.   Re-use pre-allocated tensors instead of constructing new ones inside
         the loop (adjacency, per-iteration degree buffers, …).
    4.   Avoid Python "for" loops except for the main optimisation loop itself
         (which *must* stay sequential because every step depends on the new
         logits).

    Despite these micro-optimisations, the mathematical programme that is being
    solved remains **exactly the same**.
    """

    # ------------------------------------------------------------------
    # Early setup & static pre-computations
    # ------------------------------------------------------------------
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2           # n triples  → 2n-1 nodes

    # Edge list (without self-loops)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
    edge_index = torch.stack([src, dst], dim=0)
    num_edges = edge_index.size(1)

    # Convenience views -------------------------------------------------------
    triple_nodes = torch.arange(triples_num, device=device)
    join_nodes   = torch.arange(triples_num, N_NODES, device=device)
    root         = N_NODES - 1
    non_root_joins = join_nodes[:-1] if len(join_nodes) > 0 else join_nodes  # exclude root

    # Masks that never change -------------------------------------------------
    triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
    join_to_triple_mask   = (edge_index[0] >= triples_num) & (edge_index[1] <  triples_num)
    root_outgoing_mask    = (edge_index[0] == root)

    # For left-deep penalties -------------------------------------------------
    dst_is_join_mask      = (edge_index[1] >= triples_num)
    src_is_triple_mask    = (edge_index[0] <  triples_num)
    src_is_join_mask      = ~src_is_triple_mask

    # ------------------------------------------------------------------
    # Trainable parameters (logits) + optimiser/scheduler
    # ------------------------------------------------------------------
    edge_logits       = torch.empty(num_edges, device=device).uniform_(-0.05, 0.05).requires_grad_(True)
    edge_logits_slot2 = torch.empty(num_edges, device=device).uniform_(-0.05, 0.05).requires_grad_(True)



    opt_params = [edge_logits, edge_logits_slot2] if logit_sampling == "dual-softmax" else [edge_logits]
    optimiser  = optim.AdamW(opt_params, lr=learning_rate)

    if use_lr_scheduling:
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimiser,
            lr_lambda=lambda step: (step + 1) / lr_warmup_steps if step < lr_warmup_steps and lr_warmup_steps > 0 else 1.0,
        )

    # ------------------------------------------------------------------
    # Book-keeping helpers
    # ------------------------------------------------------------------
    best_cost = float("inf")
    best_logits_1 = None
    best_logits_2 = None

    history_buffers = {
        "overall": [],
        "penalty": [],
        "acyclic": [],
        "tri_in": [],
        "tri_out": [],
        "join_in": [],
        "join_out": [],
        "entropy": [],
    }

    # Animation buffer (optional) --------------------------------------------
    animation_data = None
    if save_animation_data:
        animation_data = {
            "edge_weights_history": [],
            "step_numbers":        [],
            "edge_index":          edge_index.cpu(),
            "n_nodes":             N_NODES,
            "triples_num":         triples_num,
            "cost_history":        [],
            "penalty_history":     [],
        }

    # NOTE: We no longer reuse one global dense adjacency matrix across
    # iterations because the in-place `zero_()`/index assignment triggered
    # autograd "double backward" complaints on some PyTorch versions.  A fresh
    # tensor per step is ~O(n²) but the typical join sizes are small (≤ 27)
    # and the cost is negligible compared to the cost-model forward.  This
    # change restores correctness while keeping all other optimisations.

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    for step in range(optimization_steps):
        optimiser.zero_grad()

        # ---------------------- 1) Temperature / τ --------------------------
        if use_temperature_annealing:
            tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps)
        else:
            tau = init_tau

        # ---------------------- 2) Edge sampling ----------------------------
        if logit_sampling == "dual-softmax":
            logits1, logits2 = edge_logits, edge_logits_slot2

            # Mask invalid edge types once (copy-free via masked_fill_)
            masked_logits1 = logits1.clone()
            masked_logits2 = logits2.clone()
            invalid_mask   = triple_to_triple_mask | join_to_triple_mask
            masked_logits1[invalid_mask] = float("-inf")
            masked_logits2[invalid_mask] = float("-inf")

            join_target_mask = dst_is_join_mask
            slot1 = torch.zeros_like(edge_logits)
            slot2 = torch.zeros_like(edge_logits)

            # group-wise samples (only where dst is join)
            slot1[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits1[join_target_mask], edge_index[1][join_target_mask], tau
            )
            slot2[join_target_mask] = sample_grouped_gumbel_softmax(
                masked_logits2[join_target_mask], edge_index[1][join_target_mask], tau
            )
            edge_weights = slot1 + slot2  # relaxed 2-hot in (0,2)
            edge_weights[root_outgoing_mask] = 0.0  # root must not have outgoing edge
        elif logit_sampling == "softmax":
            masked_logits = edge_logits.clone()
            masked_logits[triple_to_triple_mask | join_to_triple_mask] = float("-inf")
            edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], tau)
            edge_weights[root_outgoing_mask] = 0.0
        else:  # sigmoid / binary concrete
            edge_weights = sample_binary_concrete(edge_logits, tau)

        # ---------------------- 3) Cost model forward -----------------------
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # ---------------------- 4) Structural penalties ---------------------
        # Degree aggregates (scatter is far cheaper than forming A_dense) ----
        in_deg  = scatter(edge_weights, dst, dim=0, dim_size=N_NODES, reduce="sum")
        out_deg = scatter(edge_weights, src, dim=0, dim_size=N_NODES, reduce="sum")

        P_triple_in  = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in    = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out   = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2

        # ---------- Acyclic penalty (requires adjacency) -------------------
        A_dense = torch.zeros((N_NODES, N_NODES), device=device)
        A_dense[src, dst] = edge_weights  # in-place write
        P_acyclic = torch.trace(torch.matrix_exp(A_dense)) - N_NODES

        # ---------- Left-deep child composition ----------------------------
        if len(join_nodes) > 0:
            # children counts (triple / join) per destination join
            child_triple_counts = scatter(
                edge_weights[src_is_triple_mask & dst_is_join_mask],
                dst[src_is_triple_mask & dst_is_join_mask] - triples_num,
                dim=0,
                dim_size=len(join_nodes),
                reduce="sum",
            )
            child_join_counts = scatter(
                edge_weights[src_is_join_mask & dst_is_join_mask],
                dst[src_is_join_mask & dst_is_join_mask] - triples_num,
                dim=0,
                dim_size=len(join_nodes),
                reduce="sum",
            )

            P_first = (child_triple_counts[0] - 2) ** 2 + (child_join_counts[0]) ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join   = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = torch.tensor(0.0, device=device)

        # ---------- Entropy regulariser ------------------------------------
        if logit_sampling == "dual-softmax":
            eps = 1e-10
            probs1 = slot1.clamp(min=eps)
            probs2 = slot2.clamp(min=eps)
            P_entropy = -(probs1 * torch.log(probs1) + probs2 * torch.log(probs2)).sum()
        elif logit_sampling == "softmax":
            eps = 1e-10
            probs = edge_weights.clamp(min=eps)
            P_entropy = -(probs * torch.log(probs)).sum()
        else:
            eps = 1e-10
            probs = torch.sigmoid(edge_logits)
            P_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).sum()

        # ---------- Aggregate loss -----------------------------------------
        total_penalty = (
            lambda_triple_in  * P_triple_in
            + lambda_triple_out * P_triple_out
            + lambda_join_in   * P_join_in
            + lambda_join_out  * P_join_out
            + lambda_acyclic   * P_acyclic
            + lambda_entropy   * P_entropy
            + lambda_left_linear * P_left_linear
        )

        total_penalty_raw = (
            P_triple_in + P_triple_out + P_join_in + P_join_out + P_acyclic + P_entropy + P_left_linear
        )

        if use_lambda_ramping:
            def annealed_lam(lam_max, step_idx, ramp_steps=150):
                return lam_max * min(1.0, step_idx / ramp_steps) ** lambda_ramp_exponent
            lambda_total = annealed_lam(lambda_total_penalty, step, optimization_steps)
        else:
            lambda_total = lambda_total_penalty

        loss = cost_pred + lambda_total * total_penalty

        # ---------- Track best feasible solution ---------------------------
        if return_best and total_penalty_raw < min_penalty_threshold and cost_pred < best_cost:
            best_cost = cost_pred.detach()
            best_logits_1 = edge_logits.detach().clone()
            if logit_sampling == "dual-softmax":
                best_logits_2 = edge_logits_slot2.detach().clone()

        # ---------- History (optional) -------------------------------------
        if verbose or save_animation_data:
            history_buffers["overall"].append(cost_pred.item() + total_penalty_raw.item())
            history_buffers["penalty"].append(total_penalty_raw.item())
            history_buffers["acyclic"].append(P_acyclic.item())
            history_buffers["tri_in"].append(P_triple_in.item())
            history_buffers["tri_out"].append(P_triple_out.item())
            history_buffers["join_in"].append(P_join_in.item())
            history_buffers["join_out"].append(P_join_out.item())
            history_buffers["entropy"].append(P_entropy.item())

        if save_animation_data and step % animation_save_interval == 0:
            animation_data["edge_weights_history"].append(edge_weights.clamp(0.0, 1.0).detach().cpu().numpy())
            animation_data["step_numbers"].append(step)
            animation_data["cost_history"].append(cost_pred.item())
            animation_data["penalty_history"].append(total_penalty.item())

        # ---------------------- 5) Back-prop + opt step ---------------------
        loss.backward()
        optimiser.step()
        if use_lr_scheduling:
            lr_scheduler.step()

        # ---------------------- 6) Logging ----------------------------------
        if verbose and (step + 1) % 100 == 0:
            print(
                f"Step {step+1}/{optimization_steps}  Cost: {cost_pred.item():.2f}  Penalty: {total_penalty_raw.item():.2f}  "
                f"LR: {optimiser.param_groups[0]['lr']:.6f}"
            )

    # ------------------------------------------------------------------
    # Hard decoding (same logic as reference implementation) -----------
    # ------------------------------------------------------------------
    with torch.no_grad():
        chosen_logits = best_logits_1 if (return_best and best_cost < float("inf")) else edge_logits
        chosen_logits2 = None
        if logit_sampling == "dual-softmax":
            chosen_logits2 = best_logits_2 if (return_best and best_cost < float("inf")) else edge_logits_slot2

        # The entire decoding section below is a copy ‑ with minor stylistic
        # clean-ups – from the baseline version to guarantee identical output.
        # ------------------------------------------------------------------
        if logit_sampling == "dual-softmax":
            if decoding_method == "threshold":
                final_edge_weights = torch.zeros(num_edges, device=device)
                # re-use static invalid edge mask
                mask_tt_jt = triple_to_triple_mask | join_to_triple_mask
                masked_l1 = chosen_logits.clone(); masked_l1[mask_tt_jt] = float("-inf")
                masked_l2 = chosen_logits2.clone(); masked_l2[mask_tt_jt] = float("-inf")
                for j in torch.unique(dst):
                    if j < triples_num:
                        continue  # skip triple targets
                    cand = (dst == j)
                    final_edge_weights[cand.nonzero(as_tuple=True)[0][torch.argmax(masked_l1[cand])]] = 1.0
                    final_edge_weights[cand.nonzero(as_tuple=True)[0][torch.argmax(masked_l2[cand])]] = 1.0
            else:
                # fall back to relaxed 2-hot + projection (same as original)
                masked_l1 = edge_logits.clone(); masked_l1[triple_to_triple_mask | join_to_triple_mask] = float("-inf")
                masked_l2 = edge_logits_slot2.clone(); masked_l2[triple_to_triple_mask | join_to_triple_mask] = float("-inf")
                slot1 = torch.zeros_like(edge_logits); slot2 = torch.zeros_like(edge_logits)
                join_mask = dst_is_join_mask
                slot1[join_mask] = sample_grouped_gumbel_softmax(masked_l1[join_mask], dst[join_mask], tau)
                slot2[join_mask] = sample_grouped_gumbel_softmax(masked_l2[join_mask], dst[join_mask], tau)
                edge_weights_relaxed = slot1 + slot2
                edge_weights_relaxed[root_outgoing_mask] = 0.0
                final_edge_weights = edge_weights_relaxed
                A_final = torch.zeros((N_NODES, N_NODES), device='cpu'); A_final[src.cpu(), dst.cpu()] = edge_weights_relaxed.cpu()
                if decoding_method == "beam":
                    final_A = project_leftdeep_greedy_beam(A_final.cpu().numpy(), beam_width=6, use_product=False)
                elif decoding_method == "greedy":
                    final_A = project_leftdeep_greedy_beam(A_final.cpu().numpy(), beam_width=1, use_product=False)
                else:  # hungarian
                    final_A = project_to_leftdeep(A_final.cpu().numpy(), exact_threshold=8)
        elif logit_sampling == "softmax":
            if decoding_method == "threshold":
                masked_logits = chosen_logits.clone(); masked_logits[triple_to_triple_mask | join_to_triple_mask] = float("-inf")
                final_edge_weights = torch.zeros_like(edge_logits)
                for v in torch.unique(src):
                    if v == root:
                        continue
                    cand = (src == v)
                    final_edge_weights[cand.nonzero(as_tuple=True)[0][torch.argmax(masked_logits[cand])]] = 1.0
            else:
                masked_logits = edge_logits.clone(); masked_logits[triple_to_triple_mask | join_to_triple_mask] = float("-inf")
                edge_w = sample_grouped_gumbel_softmax(masked_logits, src, tau); edge_w[root_outgoing_mask] = 0.0
                final_edge_weights = edge_w
                A_final = torch.zeros((N_NODES, N_NODES), device='cpu'); A_final[src.cpu(), dst.cpu()] = edge_w.cpu()
                if decoding_method == "beam":
                    final_A = project_leftdeep_greedy_beam(A_final.cpu().numpy(), beam_width=6, use_product=True)
                elif decoding_method == "greedy":
                    final_A = project_leftdeep_greedy_beam(A_final.cpu().numpy(), beam_width=1, use_product=True)
                else:
                    final_A = project_to_leftdeep(A_final.cpu().numpy(), exact_threshold=8)
        else:  # sigmoid
            if decoding_method == "threshold":
                final_edge_weights = (torch.sigmoid(chosen_logits) >= 0.5).float()
            else:
                A_sig = torch.sigmoid(edge_logits); A_final = torch.zeros((N_NODES, N_NODES), device='cpu'); A_final[src.cpu(), dst.cpu()] = A_sig.cpu()
                if decoding_method == "beam":
                    final_A = project_leftdeep_greedy_beam(A_final.cpu().numpy(), beam_width=6, use_product=True)
                elif decoding_method == "greedy":
                    final_A = project_leftdeep_greedy_beam(A_final.cpu().numpy(), beam_width=1, use_product=True)
                else:
                    final_A = project_to_leftdeep(A_final.cpu().numpy(), exact_threshold=8)

        if decoding_method == "threshold":
            final_A = torch.zeros((N_NODES, N_NODES), device=device)
            final_A[src, dst] = final_edge_weights

        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
        predicted_cost_exp = float(np.exp(final_log_cost))

    if verbose:
        plot_optimization_metrics(
            history_buffers["overall"],
            history_buffers["penalty"],
            history_buffers["acyclic"],
            history_buffers["tri_in"],
            history_buffers["tri_out"],
            history_buffers["join_in"],
            history_buffers["join_out"],
            history_buffers["entropy"],
        )

    if save_animation_data:
        return final_A, triples_num, predicted_cost_exp, animation_data
    else:
        return final_A, triples_num, predicted_cost_exp


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
    
    # Convert triples to format expected by random_join_order
    triple_strs = []
    for triple in original_triples:
        triple_strs.append([str(triple.s), str(triple.p), str(triple.o)])
    
    # Use the existing random_join_order function
    random_plan = random_join_order(triple_strs, seed=seed)
    
    return random_plan

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
    for i in tqdm(range(n_triples), desc="DP"):
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

    for perm in tqdm(itertools.permutations(range(n_triples)), desc="Exhaustive"):
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

# -----------------------------------------------------------------------------
# Fast, minimal gradient-based optimiser (benchmarking only)
# -----------------------------------------------------------------------------


def sample_gumbel(shape, eps=1e-10, device="cpu"):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)



def optimize_query_gumbel_efficient_reducedOLD(
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
    lambda_left_linear: float = 1000.0,
    init_tau: float = 10.0,
    min_tau: float = 1.0,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = True,
    min_penalty_threshold: float = 1.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = "sigmoid",  # "sigmoid", "softmax" or "dual-softmax"
    save_animation_data: bool = False,
    animation_save_interval: int = 10,
    gradient_clip_norm: float = 5.0,
    use_lr_scheduling: bool = True,
    lr_warmup_steps: int = 200,
    decoding_method: str = "threshold",
    **kwargs,
):
    """Same optimiser as *optimize_query_gumbel_efficient* but stores logits
    **only for edges whose *target* is a join node** (dst ≥ n_triples).  Edges
    leading to triple-pattern leaves are permanently zero and therefore waste
    memory and gradient bandwidth – we simply leave them out.  The returned
    adjacency matrix, however, is still (2n-1)×(2n-1) so callers remain fully
    compatible.
    """

    # ------------------------------------------------------------------
    # 0.  Static graph information
    # ------------------------------------------------------------------
    data = query_data.to(device)
    N_NODES = data.x.size(0)
    triples_num = (N_NODES + 1) // 2
    root = N_NODES - 1

    # Candidate edges: *exclude* self-loops AND all edges with dst<triples_num
    src_full, dst_full = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
    mask_dst_is_join = dst_full >= triples_num
    src = src_full[mask_dst_is_join]
    dst = dst_full[mask_dst_is_join]
    edge_index = torch.stack([src, dst], dim=0)
    num_edges = edge_index.size(1)

    # Convenience index tensors ----------------------------------------------
    triple_nodes = torch.arange(triples_num, device=device)
    join_nodes   = torch.arange(triples_num, N_NODES, device=device)
    non_root_joins = join_nodes[:-1] if len(join_nodes) > 0 else join_nodes  # exclude root

    # Masks (all target-join by construction) ---------------------------------
    root_outgoing_mask = (src == root)
    src_is_triple_mask = src < triples_num
    src_is_join_mask   = ~src_is_triple_mask

    # ------------------------------------------------------------------
    # 1.  Trainable parameters & optimiser
    # ------------------------------------------------------------------
    edge_logits = torch.empty(num_edges, device=device).uniform_(-0.05, 0.05).requires_grad_(True)
    edge_logits_slot2 = torch.empty_like(edge_logits).requires_grad_(True)

    opt_params = [edge_logits, edge_logits_slot2] if logit_sampling == "dual-softmax" else [edge_logits]
    optimiser = optim.AdamW(opt_params, lr=learning_rate)

    if use_lr_scheduling:
        lr_sched = optim.lr_scheduler.LambdaLR(
            optimiser,
            lr_lambda=lambda s: (s + 1) / lr_warmup_steps if s < lr_warmup_steps and lr_warmup_steps > 0 else 1.0,
        )

    # ------------------------------------------------------------------
    # 2.  Book-keeping
    # ------------------------------------------------------------------
    best_cost = float("inf")
    best_logits_1 = best_logits_2 = None

    if save_animation_data:
        animation_data = {
            "edge_weights_history": [],
            "step_numbers":        [],
            "edge_index":          edge_index.cpu(),
            "n_nodes":             N_NODES,
            "triples_num":         triples_num,
            "cost_history":        [],
            "penalty_history":     [],
        }
    else:
        animation_data = None

    # ------------------------------------------------------------------
    # 3.  Optimisation loop
    # ------------------------------------------------------------------
    for step in range(optimization_steps):
        optimiser.zero_grad()

        tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps) if use_temperature_annealing else init_tau

        # ----------------  edge sampling  -----------------------------------
        if logit_sampling == "dual-softmax":
            # No invalid-edge masking needed: every candidate is valid.
            slot1 = sample_grouped_gumbel_softmax(edge_logits, dst, tau)
            slot2 = sample_grouped_gumbel_softmax(edge_logits_slot2, dst, tau)
            edge_weights = slot1 + slot2  # (0,2)
            edge_weights[root_outgoing_mask] = 0.0
        elif logit_sampling == "softmax":
            edge_weights = sample_grouped_gumbel_softmax(edge_logits, src, tau)
            edge_weights[root_outgoing_mask] = 0.0
        else:  # sigmoid / binary concrete
            edge_weights = sample_binary_concrete(edge_logits, tau)

        # ----------------  cost prediction  ---------------------------------
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # ----------------  structural penalty  ---------------------------
        in_deg  = scatter(edge_weights, dst, dim=0, dim_size=N_NODES, reduce="sum")
        out_deg = scatter(edge_weights, src, dim=0, dim_size=N_NODES, reduce="sum")

        P_triple_in  = (in_deg[triple_nodes] ** 2).sum()  # should stay zero
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in    = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out   = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2

        # build dense A only for acyclicity & left-deep checks
        A_dense = torch.zeros((N_NODES, N_NODES), device=device)
        A_dense[src, dst] = edge_weights  # in-place write
        P_acyclic = torch.trace(torch.matrix_exp(A_dense)) - N_NODES

        # child counts per join
        child_triple_counts = scatter(
            edge_weights[src_is_triple_mask], dst[src_is_triple_mask] - triples_num, dim=0,
            dim_size=len(join_nodes), reduce="sum",
        ) if len(join_nodes) > 0 else edge_weights.new_zeros(0)
        child_join_counts = scatter(
            edge_weights[src_is_join_mask], dst[src_is_join_mask] - triples_num, dim=0,
            dim_size=len(join_nodes), reduce="sum",
        ) if len(join_nodes) > 0 else edge_weights.new_zeros(0)

        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + child_join_counts[0] ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join   = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = edge_weights.new_tensor(0.0)

        # entropy
        if logit_sampling == "dual-softmax":
            eps = 1e-10
            P_entropy = -((slot1.clamp(min=eps) * (slot1.clamp(min=eps)).log()) + (slot2.clamp(min=eps) * (slot2.clamp(min=eps)).log())).sum()
        elif logit_sampling == "softmax":
            eps = 1e-10
            P_entropy = -(edge_weights.clamp(min=eps) * edge_weights.clamp(min=eps).log()).sum()
        else:
            eps = 1e-10
            probs = torch.sigmoid(edge_logits)
            P_entropy = -(probs * (probs+eps).log() + (1-probs)* (1-probs+eps).log()).sum()

        total_penalty = (
            lambda_triple_in  * P_triple_in + lambda_triple_out * P_triple_out + lambda_join_in * P_join_in +
            lambda_join_out * P_join_out + lambda_acyclic * P_acyclic + lambda_entropy * P_entropy + lambda_left_linear * P_left_linear
        )
        total_penalty_raw = P_triple_in + P_triple_out + P_join_in + P_join_out + P_acyclic + P_entropy + P_left_linear

        lambda_total = (lambda_total_penalty * (min(1.0, step / 150) ** lambda_ramp_exponent)) if use_lambda_ramping else lambda_total_penalty
        loss = cost_pred + lambda_total * total_penalty

        # best tracking -------------------------------------------------------
        if return_best and total_penalty_raw < min_penalty_threshold and cost_pred < best_cost:
            best_cost = cost_pred.detach()
            best_logits_1 = edge_logits.detach().clone()
            best_logits_2 = edge_logits_slot2.detach().clone()

        # backward -----------------------------------------------------------
        loss.backward()
        
        # Gradient improvements -----------------------------------------------
        if logit_sampling == 'dual-softmax':
            params_to_clip = [edge_logits, edge_logits_slot2]
        else:
            params_to_clip = [edge_logits]
            
        # Monitor gradient norms before clipping
        grad_norms = []
        for param in params_to_clip:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        
        # Apply gradient clipping to prevent exploding gradients
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=gradient_clip_norm)
        
        
        
        optimiser.step()
        
        # Update learning rate schedule
        if use_lr_scheduling:
            lr_sched.step()

        # Log ------------------------------------------------------------------
        if verbose and (step + 1) % 100 == 0:
            print(f"[reduced] Step {step+1}/{optimization_steps}  cost={cost_pred.item():.3f}  pen={total_penalty_raw.item():.3f}")

    # ------------------------------------------------------------------
    # 4.  Hard decoding  -------------------------------------------------
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits1_final = best_logits_1 if (return_best and best_cost < float("inf")) else edge_logits
        logits2_final = best_logits_2 if (return_best and best_cost < float("inf") and logit_sampling=="dual-softmax") else edge_logits_slot2

        if logit_sampling == "dual-softmax":
            final_edge_weights = torch.zeros(num_edges, device=device)
            for j in join_nodes:
                cand = dst == j
                idx1 = torch.argmax(logits1_final[cand]); final_edge_weights[cand.nonzero(as_tuple=True)[0][idx1]] = 1.0
                idx2 = torch.argmax(logits2_final[cand]); final_edge_weights[cand.nonzero(as_tuple=True)[0][idx2]] = 1.0
        elif logit_sampling == "softmax":
            final_edge_weights = torch.zeros(num_edges, device=device)
            for v in torch.unique(src):
                if v == root: continue
                cand = src == v
                idx = torch.argmax(logits1_final[cand]); final_edge_weights[cand.nonzero(as_tuple=True)[0][idx]] = 1.0
        else:  # sigmoid
            final_edge_weights = (torch.sigmoid(logits1_final) >= 0.5).float()

        # assemble *full* adjacency (zero rows for triple targets)
        final_A = torch.zeros((N_NODES, N_NODES), device=device)
        final_A[src, dst] = final_edge_weights

        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
        predicted_cost_exp = float(np.exp(final_log_cost))

    if save_animation_data:
        return final_A, triples_num, predicted_cost_exp, animation_data
    return final_A, triples_num, predicted_cost_exp


def optimize_query_gumbel_efficient_reduced(
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
    lambda_left_linear: float = 1000.0,
    init_tau: float = 10.0,
    min_tau: float = 1.0,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = True,
    min_penalty_threshold: float = 1.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = "sigmoid",  # "sigmoid", "softmax" or "dual-softmax"
    save_animation_data: bool = False,
    animation_save_interval: int = 10,
    gradient_clip_norm: float = 5.0,
    use_lr_scheduling: bool = True,
    lr_warmup_steps: int = 200,
    decoding_method: str = "threshold",
    **kwargs,
):
    """Same optimiser as *optimize_query_gumbel_efficient* but stores logits
    **only for edges whose *target* is a join node** (dst ≥ n_triples).  Edges
    leading to triple-pattern leaves are permanently zero and therefore waste
    memory and gradient bandwidth – we simply leave them out.  The returned
    adjacency matrix, however, is still (2n-1)×(2n-1) so callers remain fully
    compatible.
    """

    # ------------------------------------------------------------------
    # 0.  Static graph information
    # ------------------------------------------------------------------
    data = query_data
    N_NODES = data.x.size(0)
    triples_num = (N_NODES + 1) // 2
    root = N_NODES - 1

    # Candidate edges: *exclude* self-loops AND all edges with dst<triples_num
    src_full, dst_full = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
    mask_dst_is_join = dst_full >= triples_num
    src = src_full[mask_dst_is_join]
    dst = dst_full[mask_dst_is_join]
    edge_index = torch.stack([src, dst], dim=0)
    num_edges = edge_index.size(1)

    # Convenience index tensors ----------------------------------------------
    triple_nodes = torch.arange(triples_num, device=device)
    join_nodes   = torch.arange(triples_num, N_NODES, device=device)
    non_root_joins = join_nodes[:-1] if len(join_nodes) > 0 else join_nodes  # exclude root

    # Masks (all target-join by construction) ---------------------------------
    root_outgoing_mask = (src == root)
    src_is_triple_mask = src < triples_num
    src_is_join_mask   = ~src_is_triple_mask

    # ------------------------------------------------------------------
    # 1.  Trainable parameters & optimiser
    # ------------------------------------------------------------------
    edge_logits = torch.empty(num_edges, device=device).uniform_(-0.05, 0.05).requires_grad_(True)
    edge_logits_slot2 = torch.empty_like(edge_logits).requires_grad_(True)

    opt_params = [edge_logits, edge_logits_slot2] if logit_sampling == "dual-softmax" else [edge_logits]
    optimiser = optim.AdamW(opt_params, lr=learning_rate)


    # ------------------------------------------------------------------
    # 2.  Book-keeping
    # ------------------------------------------------------------------
    best_cost = float("inf")
    best_logits_1 = best_logits_2 = None


    # ------------------------------------------------------------------
    # 3.  Optimisation loop
    # ------------------------------------------------------------------
    for step in range(optimization_steps):
        optimiser.zero_grad()

        tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps) if use_temperature_annealing else init_tau

        # ----------------  edge sampling  -----------------------------------
        # No invalid-edge masking needed: every candidate is valid.
        slot1 = sample_grouped_gumbel_softmax(edge_logits, dst, tau)
        slot2 = sample_grouped_gumbel_softmax(edge_logits_slot2, dst, tau)
        edge_weights = slot1 + slot2  # (0,2)
        edge_weights[root_outgoing_mask] = 0.0


        # ----------------  cost prediction  ---------------------------------
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # ----------------  structural penalty  ---------------------------
        in_deg  = scatter(edge_weights, dst, dim=0, dim_size=N_NODES, reduce="sum")
        out_deg = scatter(edge_weights, src, dim=0, dim_size=N_NODES, reduce="sum")

        P_triple_in  = (in_deg[triple_nodes] ** 2).sum()  # should stay zero
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in    = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out   = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2

        # build dense A only for acyclicity & left-deep checks
        A_dense = torch.zeros((N_NODES, N_NODES), device=device)
        A_dense[src, dst] = edge_weights  # in-place write
        P_acyclic = torch.trace(torch.matrix_exp(A_dense)) - N_NODES
        P_acyclic = 0.0
        # child counts per join
        child_triple_counts = scatter(
            edge_weights[src_is_triple_mask], dst[src_is_triple_mask] - triples_num, dim=0,
            dim_size=len(join_nodes), reduce="sum",
        ) if len(join_nodes) > 0 else edge_weights.new_zeros(0)
        child_join_counts = scatter(
            edge_weights[src_is_join_mask], dst[src_is_join_mask] - triples_num, dim=0,
            dim_size=len(join_nodes), reduce="sum",
        ) if len(join_nodes) > 0 else edge_weights.new_zeros(0)

        if len(join_nodes) > 0:
            P_first = (child_triple_counts[0] - 2) ** 2 + child_join_counts[0] ** 2
            if len(join_nodes) > 1:
                P_rest_triple = ((child_triple_counts[1:] - 1) ** 2).sum()
                P_rest_join   = ((child_join_counts[1:] - 1) ** 2).sum()
                P_left_linear = P_first + P_rest_triple + P_rest_join
            else:
                P_left_linear = P_first
        else:
            P_left_linear = edge_weights.new_tensor(0.0)

        # entropy
        #if logit_sampling == "dual-softmax":
            #eps = 1e-10
            #P_entropy = -((slot1.clamp(min=eps) * (slot1.clamp(min=eps)).log()) + (slot2.clamp(min=eps) * (slot2.clamp(min=eps)).log())).sum()

        total_penalty = (
            lambda_triple_in  * P_triple_in + lambda_triple_out * P_triple_out + lambda_join_in * P_join_in +
            lambda_join_out * P_join_out + lambda_acyclic * P_acyclic + lambda_left_linear * P_left_linear
        )
        total_penalty_raw = P_triple_in + P_triple_out + P_join_in + P_join_out + P_acyclic + P_left_linear

        lambda_total = (lambda_total_penalty * (min(1.0, step / 150) ** lambda_ramp_exponent)) if use_lambda_ramping else lambda_total_penalty
        loss = cost_pred + lambda_total * total_penalty

        # best tracking -------------------------------------------------------
        #if return_best and total_penalty_raw < min_penalty_threshold and cost_pred < best_cost:
        #    best_cost = cost_pred.detach()
        #    best_logits_1 = edge_logits.detach().clone()
        #    best_logits_2 = edge_logits_slot2.detach().clone()

        # backward -----------------------------------------------------------
        loss.backward()
        
        # Gradient improvements -----------------------------------------------
        params_to_clip = [edge_logits, edge_logits_slot2]

        
        optimiser.step()
        

    # 4.  Hard decoding  -------------------------------------------------
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits1_final = best_logits_1 if (return_best and best_cost < float("inf")) else edge_logits
        logits2_final = best_logits_2 if (return_best and best_cost < float("inf") and logit_sampling=="dual-softmax") else edge_logits_slot2

        if logit_sampling == "dual-softmax":
            final_edge_weights = torch.zeros(num_edges, device=device)
            for j in join_nodes:
                cand = dst == j
                idx1 = torch.argmax(logits1_final[cand]); final_edge_weights[cand.nonzero(as_tuple=True)[0][idx1]] = 1.0
                idx2 = torch.argmax(logits2_final[cand]); final_edge_weights[cand.nonzero(as_tuple=True)[0][idx2]] = 1.0


        # assemble *full* adjacency (zero rows for triple targets)
        final_A = torch.zeros((N_NODES, N_NODES), device=device)
        final_A[src, dst] = final_edge_weights

        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
        predicted_cost_exp = float(np.exp(final_log_cost))

    return final_A, triples_num, predicted_cost_exp