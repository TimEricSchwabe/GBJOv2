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

import torch_optimizer as optim_extra

import matplotlib.pyplot as plt  # Add this import if not present



def GBJO(
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
    min_tau: float = 1.,
    tau_decay: float = 0.999,
    use_temperature_annealing: bool = True,
    return_best: bool = True,
    min_penalty_threshold: float = 1.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = 'sigmoid',  # 'sigmoid', 'softmax' or 'dual-softmax'
    save_animation_data: bool = False,
    animation_save_interval: int = 10,
    gradient_clip_norm: float = 5.0,
    use_lr_scheduling: bool = True,
    lr_warmup_steps: int = 200,
    decoding_method: str = 'greedy', # 'threshold', 'beam', 'greedy', 'hungarian'
    k: int = 1, #not used
    use_gumbel_noise: bool = True,
):

    # Move data 
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n‑1 nodes


    # Enumerate all candidate edges (excluding self‑loops) - we have all-to-all edges because we need to consider all possible plans
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)


    # edge logits = L 
    # Step 1 of the algorithm
    edge_logits = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)

    # Second L is needed only for dual-softmax variant
    edge_logits_slot2 = torch.tensor(0. + 0.1 * (torch.rand(num_edges) - 0.5), requires_grad=True, device=device)

    # Optimiser 
    if logit_sampling == 'dual-softmax':
        #optimiser = optim.AdamW([edge_logits, edge_logits_slot2], lr=learning_rate)
        optimiser = optim.RAdam([edge_logits, edge_logits_slot2], lr=learning_rate)
        optimiser = optim_extra.Lookahead(optimiser, k=10, alpha=0.5)

    else:
        optimiser = optim.RAdam([edge_logits], lr=learning_rate)
        optimiser = optim_extra.Lookahead(optimiser, k=10, alpha=0.5)
    

    # Optional Learning rate scheduler for warmup and decay
    if use_lr_scheduling:
        def lr_schedule(step):
            # This function returns a multiplier for the base learning_rate
            # Actual LR = learning_rate * lr_schedule(step)
            if step < lr_warmup_steps:
                # Linear warmup from 0 to learning_rate
                if lr_warmup_steps == 0:
                    return 1
                else:
                    return (step + 1) / lr_warmup_steps  # 0 -> 1.0
            else:
                return 1
        
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_schedule)
    
    # Track best solution if return_best is True
    best_cost = float('inf')
    best_edge_logits = None
    best_edge_logits_slot2 = None

    # Tracking metrics for plotting 
    cost_history = []
    total_penalty_history = []
    acyclic_penalty_history = []
    triple_in_penalty_history = []
    triple_out_penalty_history = []
    join_in_penalty_history = []
    join_out_penalty_history = []
    entropy_penalty_history = []

    # Animation data storage 
    animation_data = {
        'edge_weights_history': [],
        'step_numbers': [],
        'edge_index': edge_index.cpu(),
        'n_nodes': N_NODES,
        'triples_num': triples_num,
        'cost_history': [],
        'penalty_history': []
    } if save_animation_data else None


    # for t=0 to I-1 do
    for step in range(optimization_steps):
        optimiser.zero_grad()

        # Step 4 in Algorithm 1
        if use_temperature_annealing:
            tau = _temperature_anneal(init_tau, min_tau, tau_decay, step, optimization_steps)
        else:
            tau = init_tau


        # Step 5 and 6 in Algorithm 1
        if logit_sampling == 'dual-softmax':
            # Dual-slot: every join node picks *two* incoming edges
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


        elif logit_sampling == 'softmax':            # Mask out invalid edges before softmax sampling
            
            masked_logits = edge_logits.clone()
            
            # Triple nodes cannot connect to other triple nodes
            triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits[triple_to_triple_mask] = float('-inf')
            
            # Join nodes cannot connect to triple nodes
            join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits[join_to_triple_mask] = float('-inf')
            
            # Use Gumbel-Softmax for exactly one outgoing edge per source node
            edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], tau, use_gumbel_noise)
            # Root (final join) should have *no* outgoing edge
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
        else:
            # Use Binary Concrete (Gumbel-Sigmoid) sampling
            edge_weights = sample_binary_concrete(edge_logits, tau)


        
        # Save animation data if enabled
        if save_animation_data and step % animation_save_interval == 0:
            # Clamp edge weights to [0,1] for consistent visualization
            clamped_weights = torch.clamp(edge_weights, 0.0, 1.0)
            animation_data['edge_weights_history'].append(clamped_weights.detach().cpu().numpy())
            animation_data['step_numbers'].append(step)
        
        # Step 7 in Algorithm 1
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # Build adjacency matrix for penalty calculations
        A = torch.zeros((N_NODES, N_NODES), device=device)
        A[edge_index[0], edge_index[1]] = edge_weights

        in_deg, out_deg = A.sum(0), A.sum(1)
        triple_nodes = torch.arange(triples_num, device=device)
        join_nodes = torch.arange(triples_num, N_NODES, device=device)
        root = N_NODES - 1
        non_root_joins = torch.arange(triples_num, root, device=device)

        # Structural penalties - Step 8 in Algorithm 1
        P_triple_in = (in_deg[triple_nodes] ** 2).sum()
        P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum()
        P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum()
        P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum() + out_deg[root] ** 2
        P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES


        # enforce left-deep / linear join order
        child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)   # (#joins,)
        child_join_counts   = A[join_nodes, :][:, join_nodes].sum(0)      # (#joins,)

        if len(join_nodes) > 0:  # Guard against trivial 0 join queries
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

        # Entropy penalty ( not used anymore - handled by Gumbel-Softmax)
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

        # Aggregate penalties - Step 8 still
        total_penalty = (
            lambda_triple_in * P_triple_in
            + lambda_triple_out * P_triple_out
            + lambda_join_in * P_join_in
            + lambda_join_out * P_join_out
            + lambda_acyclic * P_acyclic
            + lambda_entropy * P_entropy
            + lambda_left_linear * P_left_linear
        )

        # Keeping total penatly for gamma
        total_penalty_raw = (
            P_triple_in
            + P_triple_out
            + P_join_in
            + P_join_out
            + P_acyclic
            + P_entropy
            + P_left_linear
        )

        # Save cost and penalty for animation if enabled 
        if save_animation_data and step % animation_save_interval == 0:
            animation_data['cost_history'].append(cost_pred.item())
            animation_data['penalty_history'].append(total_penalty.item())

        # Step 9 in Algorithm 1
        if use_lambda_ramping:
            frac = min(1.0, step / optimization_steps)
            lambda_total = lambda_total_penalty * (frac ** lambda_ramp_exponent)
        else:
            lambda_total = lambda_total_penalty


        # Step 10 in Algorithm 1
        loss = cost_pred + lambda_total * total_penalty
        # Step 11 in Algorithm 1
        loss.backward()


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
        cost_history.append(cost_pred.item() + total_penalty_raw.item()) 
        total_penalty_history.append(total_penalty_raw.item())
        acyclic_penalty_history.append(P_acyclic.item())
        triple_in_penalty_history.append(P_triple_in.item())
        triple_out_penalty_history.append(P_triple_out.item())
        join_in_penalty_history.append(P_join_in.item())
        join_out_penalty_history.append(P_join_out.item())
        entropy_penalty_history.append(P_entropy.item())


        # Gradient improvements clipping
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
        
        # Apply gradient clipping 
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=gradient_clip_norm)
        
        
        # Step 11 in Algorithm 1
        optimiser.step()
        
        # Update learning rate schedule
        if use_lr_scheduling:
            scheduler.step()

        # Log
        if verbose and (step + 1) % 100 == 0:
            current_lr = optimiser.param_groups[0]['lr']
            print(
                f"Step {step+1}/{optimization_steps}  "
                f"Cost: {cost_pred.item():.2f}  Penalty: {total_penalty_raw.item():.2f}  "
                f"LR: {current_lr:.6f}  Grad: {max_grad_norm:.4f}"
            )

    # Step 14 in Algorithm 1
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
                # Dual-slot: every join node picks *two* incoming edges
                masked_logits_1 = edge_logits.clone()
                masked_logits_2 = edge_logits_slot2.clone()
                # Invalid edge types
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
                    masked_logits_1[join_target_mask], edge_index[1][join_target_mask], tau, use_gumbel_noise)
                slot2[join_target_mask] = sample_grouped_gumbel_softmax(
                    masked_logits_2[join_target_mask], edge_index[1][join_target_mask], tau, use_gumbel_noise)
                
                edge_weights = slot1 + slot2  # relaxed 2-hot (values in (0,2))
                # Ensure root join has no outgoing edges
                edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
                final_edge_weights = edge_weights
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[edge_index[0], edge_index[1]] = edge_weights
                #final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)

                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=False)
                    final_A = torch.tensor(final_A, device=device)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=False)
                    final_A = torch.tensor(final_A, device=device)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)
                    final_A = torch.tensor(final_A, device=device)

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
                edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], tau, use_gumbel_noise)
                # Root (final join) should have *no* outgoing edge
                edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0
                final_edge_weights = edge_weights
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[edge_index[0], edge_index[1]] = edge_weights
                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=False)
                    final_A = torch.tensor(final_A, device=device)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=False)
                    final_A = torch.tensor(final_A, device=device)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)
                    final_A = torch.tensor(final_A, device=device)


        else:

            if decoding_method == 'threshold':
                final_edge_weights = (torch.sigmoid(edge_logits) >= 0.5).float()


            else:
                A_sigmoid = torch.sigmoid(edge_logits)
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[edge_index[0], edge_index[1]] = A_sigmoid
                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=False)
                    final_A = torch.tensor(final_A, device=device)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=False)
                    final_A = torch.tensor(final_A, device=device)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)
                    final_A = torch.tensor(final_A, device=device)

            # original sigmoid threshold

    # Write hard one-hot selection into adjacency matrix
    if decoding_method == 'threshold':
        final_A = torch.zeros((N_NODES, N_NODES), device=device)
        final_A[edge_index[0], edge_index[1]] = final_edge_weights 
    else:
        # Extract edge weights from the projected adjacency matrix
        final_edge_weights = final_A[edge_index[0], edge_index[1]]

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


def GreedySearch(query_data, model, original_triples, device='cpu', verbose=True, choose_random=False):
    """
    Use a greedy heuristic to build a query plan using the cost model.
    After picking the first triple pattern, every further candidate is
    evaluated by creating a new join node that the current (sub-)plan
    root and the candidate triple both point to.
    """
    PICK_FIRST_MODE = "Join" # "Join" or "Triple"


    model.eval()
    data = query_data.to(device)  # Ensure consistent device
    triples_num = len(original_triples)
    
    if triples_num == 0:
        raise ValueError("No triples provided")
    if triples_num == 1:
        # Handle single triple case
        raise ValueError("Single triple case not supported")
    
    if verbose:
        print("Starting greedy query optimization")
        print(f"Number of triple patterns: {triples_num}")
    
    # ---------------------------------------------------------------------------------
    # Helper: build a graph consisting of the current plan + new triple
    def build_join_graph(curr_x, curr_edge_index, curr_root_idx, candidate_feat, join_feat):
        """
        curr_x            : node feature matrix of current plan
        curr_edge_index   : edge index of current plan
        curr_root_idx     : index of the root node of the current plan
        candidate_feat    : (1, F) feature tensor of the triple to be added

        returns:
            new_x, new_edge_index, new_root_idx
        """

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
    # ---------------------------------------------------------------------------------

    # Step 1 : choose the cheapest single triple
    original_features = query_data.x[:triples_num].clone().to(device)

    
    if PICK_FIRST_MODE == "Join":
        best_pair_cost  = float('inf')
        best_i, best_j = -1, -1

        # Fixed Join Node Feature
        join_feat = torch.zeros_like(original_features[-1])
        join_feat[-1] = 1.0
        gaussian = torch.randn((64,), device=device)
        gaussian = gaussian / gaussian.norm(p=2)

        join_feat[:64] = gaussian
        join_feat = join_feat.unsqueeze(0)


        current_edge_index = torch.tensor([[0,1],
                                    [2,2]], dtype=torch.long, device=device)
        for i,j in itertools.combinations(range(triples_num),2):
            x_i = original_features[i:i + 1]
            x_j = original_features[j:j + 1]

            x = torch.cat([x_i, x_j, join_feat], dim=0)



            with torch.no_grad():
                cost = model(x, current_edge_index).item()
            
            if cost < best_pair_cost:
                best_pair_cost = cost
                best_i, best_j = i, j

        # Build start Join Plan
        x_i = original_features[best_i:best_i + 1]
        x_j = original_features[best_j:best_j + 1]


        current_x = torch.cat([x_i, x_j, join_feat], dim=0)
        current_root_idx = 2
        current_plan = Join(left=original_triples[best_i], right=original_triples[best_j])

        remaining_triples = list(range(triples_num))
        remaining_triples.remove(best_i)
        remaining_triples.remove(best_j)


    else:
        raise ValueError("icking first triple not without Join in Greedy")
        best_first_cost, best_first_idx = float('inf'), -1
        for i in range(triples_num):
            with torch.no_grad():
                cost = model(original_features[i:i + 1],
                                torch.zeros((2, 0), dtype=torch.long, device=device)).item()
            if cost < best_first_cost:
                best_first_cost, best_first_idx = cost, i

        if verbose:
            print(f"Initial best triple: {best_first_idx} (cost={best_first_cost:.4f})")

        # initialise current plan
        current_x = original_features[best_first_idx:best_first_idx + 1]           # one node
        current_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)  # no edges yet
        current_root_idx = 0            
        
                                                   # only node is root
        current_plan = original_triples[best_first_idx]

        remaining_triples = list(range(triples_num))
        remaining_triples.remove(best_first_idx)

    

    # Greedily add triples one by one
    while remaining_triples:
        best_cost, best_idx = float('inf'), -1
        best_x = best_edge_index = None
        best_root_idx = None

        # Init fixed new join feature
        join_feat = torch.zeros_like(current_x[-1])
        join_feat[-1] = 1.0
        gaussian = torch.randn((64,), device=device)
        gaussian = gaussian / gaussian.norm(p=2)

        join_feat[:64] = gaussian
        join_feat = join_feat.unsqueeze(0)

        for cand_idx in remaining_triples:
            cand_feat = original_features[cand_idx:cand_idx + 1]

            # build graph with extra join
            new_x, new_edge_index, new_root_idx = build_join_graph(
                current_x, current_edge_index, current_root_idx, cand_feat, join_feat
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

        # update current state with the best candidate
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

def DPLinear(query_data, model, device="cpu"):
    """
    Selinger Style Dynamic Programming for Left-Deep Join Plans, motivated by:
    https://www.cs.emory.edu/~cheung/Courses/554/Syllabus/5-query-opt/dyn-prog-join2.html

    Parameters
    ----------
    query_data : torch_geometric.data.Data
        Node-feature matrix x (nTP + nJoin * F) of *one* random plan plus
        triple-count.
    model      : CostGNN
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
    triple_feats = data.x[:n_triples].clone()
    join_feats = data.x[n_triples:].clone()
    #join_feat    = torch.zeros(F, device=device);  join_feat[-1] = 1.0
    #join_feats   = join_feat.unsqueeze(0).repeat(n_triples - 1, 1)
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
            best_cost, best_A = float("inf"), None #Best cost and best plan (adjancency) to join this subset of triples

            # Try every triple as the *last* right child
            for last in subset:
                left_set = S - {last}
                left_cost, left_A = dp[left_set] # Note, we already computed the best plan to join leftset

                # Build adjacency for (left ⨝ last)
                A = left_A.clone() #optimal plan to join leftset
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

def IterativeImprovement(query_data, model, optimization_steps=100, device="cpu"):
    """
    Iterative Improvement for Left-Deep Join Plans, motivated by:
    """
    model.eval()
    data = query_data.to(device)
    n_triples = (data.x.size(0) + 1) // 2
    F = data.x.size(1)

    pairs = torch.combinations(torch.arange(n_triples, device=device), r=2) # all possible swaps we will consider

    # Start with a random permutation of the triples
    current_plan = torch.randperm(n_triples, device=device)

    n_steps_taken = 0

    best_cost = float('inf')
    best_plan = None

    best_costs = []

    while n_steps_taken < optimization_steps:
        previous_best_cost = best_cost


        for i,j in pairs.tolist():

            if n_steps_taken > optimization_steps:
                break
            n_steps_taken += 1

            candidate_plan = current_plan.clone()
            candidate_plan[i], candidate_plan[j] = current_plan[j], current_plan[i]
            A_candidate = left_deep_adj_from_perm(candidate_plan).to(device)

            src, dst = torch.where(A_candidate > 0.5)
            edge_index = torch.stack([src, dst], dim=0)

            with torch.no_grad():
                cost_pred = model(data.x, edge_index).item()
            
            if cost_pred < best_cost:
                best_cost = cost_pred
                best_plan = candidate_plan
                best_costs.append(best_cost)
        if best_cost < previous_best_cost:
            current_plan = best_plan.clone()
        else:
            # We have found no better plan in this neighborhood
            break

    # print('Did a total of %d steps' % n_steps_taken)
    # print('len of best_costs: %d' % len(best_costs))
    # plt.figure(figsize=(10, 5))
    # plt.plot(best_costs, label='Predicted Cost')
    # plt.xlabel('Optimization Step')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.grid(True)
    # plt.show() 


    return left_deep_adj_from_perm(best_plan), float(np.exp(best_cost))


def GEQO(query_data, model, optimization_steps=100, device="cpu"):
    """
    Genetic Search as implemented in Postgres:
    https://www.postgresql.org/docs/current/geqo-pg-intro.html

    """

    population_size: int = 30 
    generations: int = 50
    mutation_rate: float = 0.05
    elite_fraction: float = 0.1 # top n candidates that survive generation unchanged

    model.eval()
    data = query_data.to(device)
    n_triples = (data.x.size(0)+1) // 2

    base_perm = list(range(n_triples))  #default permutation

    # Generate initial population of plans
    population = [tuple(random.sample(base_perm, n_triples)) for _ in range(population_size)]


    best_plan = None
    best_cost = float('inf')

    
    for gen in range(generations):

        # Turn the permutations into valid adjacency matrices
        A_inds = [left_deep_adj_from_perm(ind) for ind in population]

        edge_indices = [torch.stack(torch.where(A > 0.5), dim=0) for A in A_inds]


        # Calculate costs for all plans in this generation
        costs = [model(data.x, edge_index).item() for edge_index in edge_indices]

        # Get best plan of generation and compare to overall best
        best_idx = min(range(population_size), key=lambda i: costs[i])
        if costs[best_idx] < best_cost:
            best_cost = costs[best_idx]
            best_plan = population[best_idx]

        # Elitism: carry the top plans to next generation unchanged
        elite_count = max(1, int(population_size * elite_fraction))
        elite_indices = sorted(range(population_size), key=lambda i: costs[i])[:elite_count]

        # Start new population
        new_population = [population[i] for i in elite_indices]


        # Generate children plans via parent selection and crossover
        while len(new_population) < population_size:
            # Select parents
            parent1 = parent_selection(population, costs)
            parent2 = parent_selection(population, costs)

            child = order_crossover(parent1, parent2)

            new_population.append(child)

        population = new_population

    return left_deep_adj_from_perm(best_plan), float(np.exp(best_cost))


def parent_selection(population, costs, k=5):
    """
    Tournament style selection of parent. We sample k individuals at random
    and return the one with the lowest cost
    """

    random_candidates = random.sample(range(len(population)), k)
    best = random_candidates[0]
    for cand in random_candidates[1:]:
        if costs[cand] < costs[best]:
            best=cand
    return population[best]

def order_crossover(p1, p2):
    """
    Order crosssover to merge 2 permutations.
    - choose two cut points a < b
    - we copy p1[a:b+1] into child[a:b+1]
    - fill the remaining positions with genes from p2 in their relative order while skipping the
     ones already entered in the first step
    """

    n = len(p1)
    a,b = sorted(random.sample(range(n),2))

    child = [None] * n

    child[a:b+1] = p1[a: b+1]

    pos = (b + 1) % n
    for gene in p2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % n

    return tuple(child)


def exhaustive_leftdeep_best_plan(query_data, model, device="cpu"):
    """
    Return the *predicted-cost–optimal* left-deep join plan for the given
    query under the learnt CostGNN model, using exhaustive search over all
    n! permutations of triple patterns.

    Parameters
    ----------
    query_data : torch_geometric.data.Data
        Node-feature matrix x (nTP + nJoin × F) of *one* random plan plus
        triple-count.
    model      : CostGNNv2
        Trained cost model in eval mode.
    device     : "cpu" | "cuda"

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


#@torch.compile
def calculate_penalties_compiled(
    edge_weights,
    src,
    dst,
    N_NODES,
    triple_nodes,
    join_nodes,
    non_root_joins,
    root,
    src_is_triple_mask,
    src_is_join_mask,
    triples_num,
    device,
    lambda_triple_in,
    lambda_triple_out,
    lambda_join_in,
    lambda_join_out,
    lambda_acyclic,
    lambda_left_linear
):
    """Compiled penalty calculation function."""
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

    total_penalty = (
        lambda_triple_in  * P_triple_in + lambda_triple_out * P_triple_out + lambda_join_in * P_join_in +
        lambda_join_out * P_join_out + lambda_acyclic * P_acyclic + lambda_left_linear * P_left_linear
    )
    total_penalty_raw = P_triple_in + P_triple_out + P_join_in + P_join_out + P_acyclic + P_left_linear
    
    return total_penalty, total_penalty_raw


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
    use_gumbel_noise: bool = True,
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
        slot1 = sample_grouped_gumbel_softmax(edge_logits, dst, tau, use_gumbel_noise)
        slot2 = sample_grouped_gumbel_softmax(edge_logits_slot2, dst, tau, use_gumbel_noise)
        edge_weights = slot1 + slot2  # (0,2)
        edge_weights[root_outgoing_mask] = 0.0


        # ----------------  cost prediction  ---------------------------------
        cost_pred = model(data.x, edge_index, edge_weight=edge_weights)

        # ----------------  compiled penalty calculation  -------------------
        total_penalty, total_penalty_raw = calculate_penalties_compiled(
            edge_weights,
            src,
            dst,
            N_NODES,
            triple_nodes,
            join_nodes,
            non_root_joins,
            root,
            src_is_triple_mask,
            src_is_join_mask,
            triples_num,
            device,
            lambda_triple_in,
            lambda_triple_out,
            lambda_join_in,
            lambda_join_out,
            lambda_acyclic,
            lambda_left_linear
        )

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


def optimize_query_gumbel_sinkhorn(
    query_data,
    model,
    device: str = "cpu",
    *,
    optimization_steps: int = 500,
    learning_rate: float = 0.1,
    init_tau: float = 1.0,
    tau_decay: float = 0.99,
    min_tau: float = 0.05,
    save_animation_data: bool = False,
    return_best: bool = True,
    **kwargs
):
    """
    Optimizes the join order by learning a permutation matrix (Sinkhorn) 
    and mapping it to a fixed Left-Deep topology.
    This avoids invalid plan penalties entirely.
    """
    model.eval()
    data = query_data.to(device)
    N_NODES = data.x.size(0)
    triples_num = (N_NODES + 1) // 2
    
    # We only permute the triples
    # log_alpha[i, j] = Log-Prob that Triple j is at Position i
    log_alpha = torch.zeros((triples_num, triples_num), device=device, requires_grad=True)
    # Initialize near uniform
    torch.nn.init.uniform_(log_alpha, -0.1, 0.1)
    
    optimizer = optim.Adam([log_alpha], lr=learning_rate)
    
    # Pre-compute fixed structure for Left-Deep Tree
    # Nodes in the "Virtual" tree:
    # 0..triples_num-1 : Leaf Positions
    # triples_num..N_NODES-1 : Join Nodes
    
    # We construct the edge_index for the Fixed Left-Deep Tree once
    # (Child -> Parent)
    src_list = []
    dst_list = []
    
    # Join 0 (idx=triples_num) connects Pos 0 and Pos 1
    if triples_num > 1:
        src_list.extend([0, 1])
        dst_list.extend([triples_num, triples_num])
        
        # Subsequent joins
        for k in range(1, triples_num - 1):
            join_idx = triples_num + k
            prev_join_idx = triples_num + k - 1
            leaf_pos = k + 1
            
            # Edges: PrevJoin -> Join, Leaf -> Join
            src_list.extend([prev_join_idx, leaf_pos])
            dst_list.extend([join_idx, join_idx])
            
    fixed_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    
    # Join features (constant)
    join_features = data.x[triples_num:].clone()
    
    best_cost = float('inf')
    best_perm = None
    
    for step in range(optimization_steps):
        optimizer.zero_grad()
        
        # Temperature annealing
        tau = max(min_tau, init_tau * (tau_decay ** step))
        
        # Gumbel-Sinkhorn
        noise = -torch.log(-torch.log(torch.rand_like(log_alpha) + 1e-10) + 1e-10)
        noisy_logits = (log_alpha + noise) / tau
        
        # Sinkhorn Iterations
        log_P = noisy_logits
        for _ in range(10):
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
        P = torch.exp(log_P)
        
        # Permute Triple Features
        # P[i, j] is prob that Position i gets Triple j
        # Feature_at_Pos_i = sum_j P[i, j] * Feature_Triple_j
        # Shape: (triples_num, F)
        permuted_triples = torch.matmul(P, data.x[:triples_num])
        
        # Construct full node features
        node_feats = torch.cat([permuted_triples, join_features], dim=0)
        
        # Predict Cost
        cost_pred = model(node_feats, fixed_edge_index)
        
        # Loss
        loss = cost_pred
        loss.backward()
        optimizer.step()
        
        # Track Best (Hard Evaluation)
        if return_best:
            with torch.no_grad():
                # Check current prediction quality
                if cost_pred.item() < best_cost:
                    best_cost = cost_pred.item()
                    best_perm = log_alpha.detach().clone()

    # Final Decoding
    final_log_alpha = best_perm if best_perm is not None else log_alpha
    
    # Convert to Hard Permutation (Greedy Argmax or Hungarian)
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(final_log_alpha.cpu().numpy(), maximize=True)
        # col_ind[i] is the triple index assigned to position i
        # sort by row_ind to get proper array
        # row_ind is usually 0..N-1 sorted, but good to be safe
        # We want: position 0 has triple X, pos 1 has triple Y...
        # linear_sum_assignment returns (row_ind, col_ind) such that cost is maximized.
        # we want to maximize prob (log_alpha).
        # result: row_ind[k] -> col_ind[k]
        # if row_ind is [0, 1, 2...], then col_ind is [triple_for_pos_0, triple_for_pos_1...]
        
        # We sort by row_ind to ensure order 0..N-1
        zipped = sorted(zip(row_ind, col_ind))
        perm_indices = torch.tensor([c for r, c in zipped], device=device)
        
    except ImportError:
        # Fallback: simple argmax
        _, perm_indices = torch.topk(final_log_alpha, k=1, dim=1)
        perm_indices = perm_indices.squeeze()

    # Reconstruct Adjacency Matrix from Permutation
    final_src = []
    final_dst = []
    
    fixed_src = fixed_edge_index[0].cpu().numpy()
    fixed_dst = fixed_edge_index[1].cpu().numpy()
    perm_map = perm_indices.cpu().numpy()
    
    for s, d in zip(fixed_src, fixed_dst):
        # Map Source
        if s < triples_num: # Leaf
            if triples_num == 1: # Edge case
                 real_s = 0
            else:
                 real_s = perm_map[s]
        else: # Join
            real_s = s
            
        # Map Dest
        if d < triples_num: # Leaf
             # Should not happen for Dst in Left-Deep
             real_d = perm_map[d]
        else: # Join
            real_d = d
            
        final_src.append(real_s)
        final_dst.append(real_d)
        
    final_A = torch.zeros((N_NODES, N_NODES), device=device)
    final_A[final_src, final_dst] = 1.0
    
    # Final Cost
    with torch.no_grad():
        final_edge_idx = torch.tensor([final_src, final_dst], device=device)
        final_log_cost = model(data.x, final_edge_idx).item()
        final_cost = float(np.exp(final_log_cost))

    if save_animation_data:
        return final_A, triples_num, final_cost, None
    else:
        return final_A, triples_num, final_cost


def NeuralSort(
    query_data,
    model,
    device: str = "cpu",
    *,
    optimization_steps: int = 500,
    learning_rate: float = 0.1,
    init_tau: float = 1.0,
    tau_decay: float = 0.99,
    min_tau: float = 0.1,
    save_animation_data: bool = False,
    return_best: bool = True,
    **kwargs
):
    """
    Pure NeuralSort implementation (Grover et al., ICLR 2019).
    
    Instead of learning an n×n matrix, we learn a 1D score vector.
    The permutation matrix is computed deterministically via the NeuralSort formula.
    """
    model.eval()
    data = query_data.to(device)
    N_NODES = data.x.size(0)
    triples_num = (N_NODES + 1) // 2
    

    scores = torch.zeros(triples_num, device=device, requires_grad=True)
    torch.nn.init.uniform_(scores, -0.1, 0.1)
    
    optimizer = optim.Adam([scores], lr=learning_rate)
    
    # Pre-compute fixed Left-Deep topology (same as before)
    src_list = []
    dst_list = []
    
    if triples_num > 1:
        src_list.extend([0, 1])
        dst_list.extend([triples_num, triples_num])
        
        for k in range(1, triples_num - 1):
            join_idx = triples_num + k
            prev_join_idx = triples_num + k - 1
            leaf_pos = k + 1
            src_list.extend([prev_join_idx, leaf_pos])
            dst_list.extend([join_idx, join_idx])
            
    fixed_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    join_features = data.x[triples_num:].clone()
    
    best_cost = float('inf')
    best_scores = None
    
    # Pre-compute position weights: (n+1-2i) for i in 1..n
    # In 0-indexed: (n-1-2i) for i in 0..n-1, but paper uses 1-indexed
    # So for position i (0-indexed): weight = (n + 1 - 2*(i+1)) = (n - 1 - 2i)
    n = triples_num
    position_weights = torch.tensor(
        [n - 1 - 2 * i for i in range(n)], 
        device=device, 
        dtype=torch.float32
    )  # Shape: (n,)


    costs_history = [] 
    soft_costs_history = []
    
    for step in range(optimization_steps):
        optimizer.zero_grad()
        
        # Temperature annealing
        tau = 4
        
        # ========================================
        # NeuralSort Formula (Equation 6)
        # ========================================
        # P[i, j] = softmax(((n+1-2*(i+1)) * s_j - sum_k|s_j - s_k|) / tau)[j]
        
        # 1. Compute pairwise absolute differences: A[j] = sum_k |s_j - s_k|
        # scores: (n,)
        s_vec = scores.unsqueeze(1) # (n, 1)
        A_scores = torch.abs(s_vec - s_vec.t()) # (n, n)
        A = torch.sum(A_scores, dim=1) # (n,)
        
        # 2. Compute logits
        # Term 1: (n+1-2i) * s_j -> Outer product (n, n)
        term1 = torch.outer(position_weights, scores)
        
        # Term 2: -A[j] -> Broadcast to (n, n)
        term2 = A.unsqueeze(0) # (1, n)
        
        # Combine
        logits = (term1 - term2) / tau
        
        # Softmax over columns (which table goes to each position)
        P = torch.softmax(logits, dim=1)  # Shape: (n, n)
        
        # Permute Triple Features
        permuted_triples = torch.matmul(P, data.x[:triples_num])
        
        # Construct full node features
        node_feats = torch.cat([permuted_triples, join_features], dim=0)
        
        # Predict Cost
        cost_pred = model(node_feats, fixed_edge_index)

        
        # Loss = just the cost (no penalties needed!)
        loss = cost_pred
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            # 1. Decode current scores to hard permutation
            current_perm = torch.argsort(scores, descending=True)
            
            # 2. Map to graph structure (same as Final Decoding logic)
            temp_src = []
            temp_dst = []
            perm_map = current_perm.cpu().numpy()
            fixed_src_np = fixed_edge_index[0].cpu().numpy()
            fixed_dst_np = fixed_edge_index[1].cpu().numpy()
            
            for s, d in zip(fixed_src_np, fixed_dst_np):
                real_s = perm_map[s] if s < triples_num else s
                real_d = perm_map[d] if d < triples_num else d
                temp_src.append(real_s)
                temp_dst.append(real_d)
            
            temp_edge_idx = torch.tensor([temp_src, temp_dst], device=device)
            
            # 3. Evaluate Model with HARD edges and ORIGINAL data
            hard_cost = model(data.x, temp_edge_idx).item()
            
            if hard_cost < best_cost:
                best_cost = hard_cost
                best_scores = scores.detach().clone()
        
        # Track Best
        #if return_best and cost_pred.item() < best_cost:
        #    best_cost = cost_pred.item()
        #    best_scores = scores.detach().clone()


        costs_history.append(hard_cost)
        soft_costs_history.append(cost_pred.item())

    # plt.figure(figsize=(10, 5))
    # plt.plot(costs_history, label='Predicted Cost')
    # plt.plot(soft_costs_history, label='Soft Cost')
    # plt.xlabel('Optimization Step')
    # plt.ylabel('Cost')
    # plt.title('NeuralSort Optimization Progress')
    # plt.legend()
    # plt.grid(True)
    # plt.show() 
    
    # ========================================
    # Final Decoding: Simple argsort
    # ========================================
    final_scores = best_scores if best_scores is not None else scores.detach()
    
    # Higher score = earlier position (descending sort)
    perm_indices = torch.argsort(final_scores, descending=True)
    
    # Reconstruct Adjacency Matrix from Permutation
    final_src = []
    final_dst = []
    
    fixed_src = fixed_edge_index[0].cpu().numpy()
    fixed_dst = fixed_edge_index[1].cpu().numpy()
    perm_map = perm_indices.cpu().numpy()
    
    for s, d in zip(fixed_src, fixed_dst):
        if s < triples_num:
            real_s = perm_map[s]
        else:
            real_s = s
            
        if d < triples_num:
            real_d = perm_map[d]
        else:
            real_d = d
            
        final_src.append(real_s)
        final_dst.append(real_d)
        
    final_A = torch.zeros((N_NODES, N_NODES), device=device)
    final_A[final_src, final_dst] = 1.0
    
    # Final Cost
    with torch.no_grad():
        final_edge_idx = torch.tensor([final_src, final_dst], device=device)
        final_log_cost = model(data.x, final_edge_idx).item()
        final_cost = float(np.exp(final_log_cost))



    if save_animation_data:
        return final_A, triples_num, final_cost, None
    else:
        return final_A, triples_num, final_cost


def optimize_query_nevergrad(
    query_data,
    model,
    device: str = "cpu",
    *,
    optimization_steps: int = 500,
    verbose: bool = True,
    lambda_acyclic: float = 1000.0,
    lambda_triple_in: float = 1000.0,
    lambda_triple_out: float = 1000.0,
    lambda_join_in: float = 500.0,
    lambda_join_out: float = 1000.0,
    lambda_left_linear: float = 1000.0,
    lambda_total_penalty: float = 1.0,
    use_lambda_ramping: bool = True,
    lambda_ramp_exponent: float = 2.0,
    logit_sampling: str = 'dual-softmax',
    decoding_method: str = 'greedy',
    save_animation_data: bool = False,
    num_workers: int = 1,
    **kwargs,
):
    """
    Gradient-free query plan optimization using Nevergrad's NGOpt.
    
    Same objective as optimize_query_gumbel but uses evolutionary/derivative-free
    optimization instead of gradient descent.
    """
    import nevergrad as ng
    
    model.eval()
    data = query_data.to(device)
    N_NODES = len(data.x)
    triples_num = (N_NODES + 1) // 2
    
    # Build candidate edges (exclude self-loops)
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
    edge_index = torch.stack([src, dst], dim=0)
    num_edges = edge_index.size(1)
    
    # Precompute masks for valid edges
    triple_to_triple = (src < triples_num) & (dst < triples_num)
    join_to_triple = (src >= triples_num) & (dst < triples_num)
    invalid_mask = triple_to_triple | join_to_triple
    root = N_NODES - 1
    
    triple_nodes = torch.arange(triples_num, device=device)
    join_nodes = torch.arange(triples_num, N_NODES, device=device)
    non_root_joins = torch.arange(triples_num, root, device=device)
    
    call_counter = [0]
    
    def objective(params):
        """Evaluate cost + penalties for given edge logits."""
        # Lambda ramping: landscape changes over time
        if use_lambda_ramping:
            frac = min(1.0, call_counter[0] / optimization_steps)
            lambda_total = lambda_total_penalty * (frac ** lambda_ramp_exponent)
        else:
            lambda_total = lambda_total_penalty
        call_counter[0] += 1
        
        with torch.no_grad():
            if logit_sampling == 'dual-softmax':
                logits1 = torch.tensor(params[:num_edges], dtype=torch.float32, device=device)
                logits2 = torch.tensor(params[num_edges:], dtype=torch.float32, device=device)
                
                # Mask invalid edges
                logits1[invalid_mask] = float('-inf')
                logits2[invalid_mask] = float('-inf')
                
                # Grouped softmax per target join node
                slot1 = torch.zeros(num_edges, device=device)
                slot2 = torch.zeros(num_edges, device=device)
                
                for j in join_nodes:
                    cand = (dst == j)
                    if cand.any():
                        slot1[cand] = torch.softmax(logits1[cand], dim=0)
                        slot2[cand] = torch.softmax(logits2[cand], dim=0)
                
                edge_weights = slot1 + slot2
                edge_weights[src == root] = 0.0
                
            else:  # softmax
                logits = torch.tensor(params, dtype=torch.float32, device=device)
                logits[invalid_mask] = float('-inf')
                
                edge_weights = torch.zeros(num_edges, device=device)
                for v in torch.unique(src):
                    if v == root:
                        continue
                    m = (src == v)
                    edge_weights[m] = torch.softmax(logits[m], dim=0)
            
            # Cost prediction
            cost = model(data.x, edge_index, edge_weight=edge_weights).item()
            
            # Build adjacency for penalties
            A = torch.zeros((N_NODES, N_NODES), device=device)
            A[src, dst] = edge_weights
            
            in_deg = A.sum(0)
            out_deg = A.sum(1)
            
            # Structural penalties
            P_triple_in = (in_deg[triple_nodes] ** 2).sum().item()
            P_triple_out = ((out_deg[triple_nodes] - 1) ** 2).sum().item()
            P_join_in = ((in_deg[join_nodes] - 2) ** 2).sum().item()
            P_join_out = ((out_deg[non_root_joins] - 1) ** 2).sum().item() + (out_deg[root] ** 2).item()
            P_acyclic = (torch.trace(torch.matrix_exp(A)) - N_NODES).item()
            
            # Left-deep penalty
            child_triple_counts = A[:triples_num, :][:, join_nodes].sum(0)
            child_join_counts = A[join_nodes, :][:, join_nodes].sum(0)
            
            if len(join_nodes) > 0:
                P_first = ((child_triple_counts[0] - 2) ** 2 + child_join_counts[0] ** 2).item()
                if len(join_nodes) > 1:
                    P_rest = ((child_triple_counts[1:] - 1) ** 2).sum().item()
                    P_rest += ((child_join_counts[1:] - 1) ** 2).sum().item()
                    P_left_linear = P_first + P_rest
                else:
                    P_left_linear = P_first
            else:
                P_left_linear = 0.0
            
            total_penalty = (
                lambda_triple_in * P_triple_in +
                lambda_triple_out * P_triple_out +
                lambda_join_in * P_join_in +
                lambda_join_out * P_join_out +
                lambda_acyclic * P_acyclic +
                lambda_left_linear * P_left_linear
            )
            
            return cost + lambda_total * total_penalty
    
    # Set up nevergrad parametrization
    if logit_sampling == 'dual-softmax':
        param = ng.p.Array(shape=(2 * num_edges,), lower=-5.0, upper=5.0)
    else:
        param = ng.p.Array(shape=(num_edges,), lower=-5.0, upper=5.0)
    
    # Create optimizer
    optimizer = ng.optimizers.CMA(parametrization=param, budget=optimization_steps, num_workers=num_workers)
    
    # Run optimization
    if verbose:
        pbar = tqdm(range(optimization_steps), desc="Nevergrad optimization")
        for _ in pbar:
            x = optimizer.ask()
            loss = objective(x.value)
            optimizer.tell(x, loss)
            if _ % 50 == 0:
                pbar.set_postfix({"loss": f"{loss:.2f}"})
    else:
        for _ in range(optimization_steps):
            x = optimizer.ask()
            loss = objective(x.value)
            optimizer.tell(x, loss)
    
    # Get best solution
    recommendation = optimizer.provide_recommendation()
    best_params = recommendation.value
    
    # Decode final solution
    with torch.no_grad():
        if logit_sampling == 'dual-softmax':
            logits1 = torch.tensor(best_params[:num_edges], dtype=torch.float32, device=device)
            logits2 = torch.tensor(best_params[num_edges:], dtype=torch.float32, device=device)
            logits1[invalid_mask] = float('-inf')
            logits2[invalid_mask] = float('-inf')
            
            if decoding_method == 'threshold':
                final_edge_weights = torch.zeros(num_edges, device=device)
                for j in join_nodes:
                    cand = (dst == j)
                    if cand.any():
                        idx1 = torch.argmax(logits1[cand])
                        idx2 = torch.argmax(logits2[cand])
                        indices = torch.where(cand)[0]
                        final_edge_weights[indices[idx1]] = 1.0
                        final_edge_weights[indices[idx2]] = 1.0
                
                final_A = torch.zeros((N_NODES, N_NODES), device=device)
                final_A[src, dst] = final_edge_weights
            else:
                # Soft weights for projection-based decoding
                slot1 = torch.zeros(num_edges, device=device)
                slot2 = torch.zeros(num_edges, device=device)
                for j in join_nodes:
                    cand = (dst == j)
                    if cand.any():
                        slot1[cand] = torch.softmax(logits1[cand], dim=0)
                        slot2[cand] = torch.softmax(logits2[cand], dim=0)
                edge_weights = slot1 + slot2
                edge_weights[src == root] = 0.0
                
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[src, dst] = edge_weights
                
                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=False)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=False)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)
                else:
                    final_A = A.cpu().numpy()
                final_A = torch.tensor(final_A, device=device)
        
        else:  # softmax
            logits = torch.tensor(best_params, dtype=torch.float32, device=device)
            logits[invalid_mask] = float('-inf')
            
            if decoding_method == 'threshold':
                final_edge_weights = torch.zeros(num_edges, device=device)
                for v in torch.unique(src):
                    if v == root:
                        continue
                    m = (src == v)
                    idx = torch.argmax(logits[m])
                    final_edge_weights[torch.where(m)[0][idx]] = 1.0
                
                final_A = torch.zeros((N_NODES, N_NODES), device=device)
                final_A[src, dst] = final_edge_weights
            else:
                edge_weights = torch.zeros(num_edges, device=device)
                for v in torch.unique(src):
                    if v == root:
                        continue
                    m = (src == v)
                    edge_weights[m] = torch.softmax(logits[m], dim=0)
                
                A = torch.zeros((N_NODES, N_NODES), device=device)
                A[src, dst] = edge_weights
                
                if decoding_method == 'beam':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=6, use_product=False)
                elif decoding_method == 'greedy':
                    final_A = project_leftdeep_greedy_beam(A.cpu().numpy(), beam_width=1, use_product=False)
                elif decoding_method == 'hungarian':
                    final_A = project_to_leftdeep(A.cpu().numpy(), exact_threshold=8)
                else:
                    final_A = A.cpu().numpy()
                final_A = torch.tensor(final_A, device=device)
        
        # Compute final cost
        final_edge_weights = final_A[src, dst]
        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
        predicted_cost_exp = float(np.exp(final_log_cost))
    
    if save_animation_data:
        return final_A, triples_num, predicted_cost_exp, None
    else:
        return final_A, triples_num, predicted_cost_exp