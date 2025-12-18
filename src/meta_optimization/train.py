import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))



from model import CostGNNv3
from optimization.gumbel_utils import sample_grouped_gumbel_softmax, _temperature_anneal
from utils.data_utils import load_sparql_queries
from src.create_data.create_optimization_data import SPARQLQuery



def compute_structure_penalties(edge_index, edge_weights, N_NODES, triples_num, device):
    """Compute structural penalties for valid query plans."""
    # Build adjacency matrix
    A = torch.zeros((N_NODES, N_NODES), device=device)
    A[edge_index[0], edge_index[1]] = edge_weights
    
    in_deg, out_deg = A.sum(0), A.sum(1)
    root = N_NODES - 1
    n_joins = N_NODES - triples_num
    
    # Triple constraints: out_deg=1  - in-degree is handled by masking
    P_triple = ((out_deg[:triples_num] - 1) ** 2).sum()
    
    # Join constraints: in_deg=2, out_deg=1 (root: out_deg=0)
    P_join_in = ((in_deg[triples_num:] - 2) ** 2).sum()

    P_join_out = ((out_deg[triples_num:root] - 1) ** 2).sum() + out_deg[root] ** 2
    
    # Acyclicity via matrix exponential
    P_acyclic = torch.trace(torch.matrix_exp(A)) - N_NODES
    
    # Left-deep: first join gets 2 triples, rest get 1 triple + 1 join
    child_triples = A[:triples_num, triples_num:].sum(0) # childs of joins that are triples
    child_joins = A[triples_num:, triples_num:].sum(0) # childs of joins that are joins
    
    # Vectorized targets: [2,1,1,...] for triples, [0,1,1,...] for joins
    target_t = torch.ones(n_joins, device=device) # all joins should have 1 triple as child
    target_t[0] = 2 # except the first, he should get 2 triples as child
    target_j = torch.ones(n_joins, device=device) # all joins should have 1 join as child
    target_j[0] = 0 # except the first, he should get 0 joins as child
    
    P_left_linear = ((child_triples - target_t) ** 2).sum() + ((child_joins - target_j) ** 2).sum()


    safe_weights = edge_weights.nan_to_num(0.0)
    P_entropy = -(safe_weights * torch.log(safe_weights.clamp(min=1e-9))).sum()

    
    return P_triple, P_join_in, P_join_out, P_acyclic, P_left_linear, P_entropy


def add_fingerprints_to_query_data(query_data, fingerprint_dim=64):
    """
    Add random Gaussian fingerprints to join nodes in query data.
    Matches AddRandomGaussianFingerprints from data_loader.py
    """
    x = query_data.x.clone()
    
    is_join = (x[:, -1] == 1.0)
    join_indices = torch.where(is_join)[0]
    n_joins = len(join_indices)
    
    if n_joins == 0:
        return query_data
    
    # random fingerprints, normalized (same as training)
    fingerprints = torch.randn(n_joins, fingerprint_dim, device=x.device)
    fingerprints = fingerprints / fingerprints.norm(dim=1, keepdim=True)
    
    for i, join_idx in enumerate(join_indices):
        x[join_idx, :fingerprint_dim] = fingerprints[i]
    
    query_data.x = x
    return query_data


def freeze_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def unfreeze_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(True)

class Hyperparams(nn.Module):
    """
    The hyperparams of C_theta that are to be meta-optimized. 
    Appropriately constrained to sensible domain
    """
    def __init__(self, init_lambda_triple_out=100.0,
     init_lambda_join_in=100.0, init_lambda_join_out=100.0, init_lambda_acyclic=100.0,
      init_lambda_left_linear=100.0, init_lambda_entropy=100.0, init_eta=1e0) -> None:
        super().__init__()

        self._lambda_triple_out = nn.Parameter(torch.tensor(float(init_lambda_triple_out)).clamp(min=-2, max=3000))
        self._lambda_join_in = nn.Parameter(torch.tensor(float(init_lambda_join_in)).clamp(min=-2, max=3000))
        self._lambda_join_out = nn.Parameter(torch.tensor(float(init_lambda_join_out)).clamp(min=-2, max=3000))
        self._lambda_acyclic = nn.Parameter(torch.tensor(float(init_lambda_acyclic)).clamp(min=-2, max=3000))
        self._lambda_left_linear = nn.Parameter(torch.tensor(float(init_lambda_left_linear)).clamp(min=-2, max=3000))
        self._lambda_entropy = nn.Parameter(torch.tensor(float(init_lambda_entropy)).clamp(min=-2, max=3000))
        self._eta = nn.Parameter(torch.tensor(float(init_eta)).clamp(min=1e-4, max=2))

    def lambda_triple_out(self) -> torch.Tensor:
        return F.softplus(self._lambda_triple_out)

    def lambda_join_in(self) -> torch.Tensor:
        return F.softplus(self._lambda_join_in)

    def lambda_join_out(self) -> torch.Tensor:
        return F.softplus(self._lambda_join_out)

    def lambda_acyclic(self) -> torch.Tensor:
        return F.softplus(self._lambda_acyclic)

    def lambda_left_linear(self) -> torch.Tensor:
        return F.softplus(self._lambda_left_linear)

    def lambda_entropy(self) -> torch.Tensor:
        return F.softplus(self._lambda_entropy)

    def eta(self) -> torch.Tensor:
        return self._eta


def gbjo(query, C_theta, hyperparams, device="cpu"):
    """
    GBJO algorithm, adapted to be fully backpropagable

    returns final soft logits
    """

    N_STEPS = 100
    tau = 5
    lambda_triple_out = 1.0
    lambda_join_in = 1.0
    lambda_join_out = 1.0
    lambda_acyclic = 1.0
    lambda_left_linear = 1.0
    lambda_entropy = 0.
    #learning_rate = 0.01

    N_NODES = len(query.x)
    triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n‑1 nodes

    # Enumerate all candidate edges (excluding self‑loops) - we have all-to-all edges because we need to consider all possible plans
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)
    
    logits = torch.zeros(num_edges, requires_grad=True)

    for step in range(N_STEPS):
        tau = _temperature_anneal(5, 1, 0.999, step, N_STEPS)
        learning_rate = hyperparams.eta()

        # First, mask out invalid edges in logits
        masked_logits = logits.clone()
        
        # Triple nodes cannot connect to other triple nodes
        triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
        masked_logits[triple_to_triple_mask] = float('-inf')
        
        # Join nodes cannot connect to triple nodes
        join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
        masked_logits[join_to_triple_mask] = float('-inf')
        
        # Use Gumbel-Softmax for exactly one outgoing edge per source node
        # Note: The grouped softmax is necessary because structural constraints (triple→join only, join→join only) create unequal group sizes per source node
        edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], tau)
        # Root (final join) should have *no* outgoing edge
        edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0


        # Predict Cost
        cost_pred = C_theta(query.x, edge_index, edge_weight=edge_weights)

        # Calculate Penalties
        P_triple_out, P_join_in, P_join_out, P_acyclic, P_left_linear, P_entropy = compute_structure_penalties(edge_index, edge_weights, N_NODES, triples_num, device)

        total_penalty = (
            lambda_triple_out * P_triple_out
            + lambda_join_in * P_join_in
            + lambda_join_out * P_join_out
            + lambda_acyclic * P_acyclic
            + lambda_left_linear * P_left_linear
        )

        # backprop cost to logits

        #cost = (cost_pred + total_penalty).mean()
        cost = cost_pred.mean()
        (g,) = torch.autograd.grad(cost, logits, create_graph=True)

        # Gradient Descent Update
        logits = logits - learning_rate * g

    return logits


def train_step(query, C_theta, C_psi, hyperparams, opt_outer):
    """
    One end-to-end step:
      1) unroll inner searc (GBJO) using theta + current hyperparams
      2) compute outer loss using C_psi (frozen) on final soft plan from GBJO
      4) compute regression loss of found plan (projected to discrete plan) between C_theta and C_psi
      5) Meta Loss = Outer Loss + Regression Loss
      6) update theta+hyper by meta loss

      Parameters:
      query: query object
      C_theta: cost model to be meta-optimized
      C_psi: frozen critic cost model
      hyperparams: hyper parameters to be optimized
      opt_outer: outer optimizer for psi_theta (updates C_theta params and hyperparams)
    """
    pass




if __name__ == "__main__":
    pass

    # Model parameters
    MODEL_PATH = "/home/tim/query_optimization/datasets/models/lubm/6-layers-v3-with-layer-norm/model.pt"
    QUERY_PATH = "/home/tim/query_optimization/datasets/plans/lubm_star_plan_datasets_optimization/optimization_stars_3_to_14/queries.pkl"
    DROPOUT = 0.0
    HIDDEN_DIM = 128
    NODE_FEATURE_DIM = 307
    N_LAYERS = 6
    USE_JK = False
    JK_MODE = 'cat'
    USE_RESIDUAL = False
    USE_LAYER_NORM = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ACCUMULATION_STEPS = 4  # aka batch size, we accumulate gradients over this many steps

    # Load queries
    sparql_queries = load_sparql_queries(QUERY_PATH, 100)
    sparql_queries = [q for q in sparql_queries if len(q.triples) == 3]
    print("INFO")
    # Define C_theta (cost model to be meta-optimized)
    C_theta = CostGNNv3(node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT).to(device)
    C_theta.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # Define C_psi (reezed critic cost model)
    C_psi = CostGNNv3(node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT).to(device)
    C_psi.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Define hyperparams
    hyperparams = Hyperparams(init_lambda_join_in=1.0)


    # define outer optimizer for psi_theta (updates C_theta params and hyperparams)
    opt_theta = torch.optim.AdamW(list(C_theta.parameters()) + list(hyperparams.parameters()), lr=1e-4, weight_decay=1e-2)
    #opt_theta = torch.optim.AdamW(hyperparams.parameters(), lr=1e-1, weight_decay=1e-2)


    EPOCHS = 100
    lambda_triple_out = 1.0
    lambda_join_in = 1.0
    lambda_join_out = 1.0
    lambda_acyclic = 1.0
    lambda_left_linear = 1.0
    lambda_entropy = 1
    query = None

    # Freeze c_psi as we never train it
    freeze_(C_psi)

    average_loss_per_epoch = []
    average_total_penalty_per_epoch = []
    best_loss = float('inf')

    for i in range(0, EPOCHS):
        model_loss = 0
        total_penalties = 0
        total_loss = 0
        opt_theta.zero_grad(set_to_none=True)
        for idx, query in enumerate(sparql_queries):
            #query = sparql_queries[0]
            query = query.torch_data[0]
            # TODO add gaussian fingerprints
            query = add_fingerprints_to_query_data(query, fingerprint_dim=64)



            N_NODES = len(query.x)
            triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n‑1 nodes

            # Enumerate all candidate edges (excluding self‑loops)
            src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
            edge_index = torch.stack([src, dst], dim=0).to(device)
            num_edges = edge_index.size(1)


            final_logits = gbjo(query, C_theta, hyperparams, device=device)

            # mask and softmax final logits
            masked_logits = final_logits.clone()
            
            triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
            masked_logits[triple_to_triple_mask] = float('-inf')
            join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
            masked_logits[join_to_triple_mask] = float('-inf')
            edge_weights = sample_grouped_gumbel_softmax(masked_logits, edge_index[0], temperature=1)
            # Root (final join) should have *no* outgoing edge
            edge_weights[edge_index[0] == (N_NODES - 1)] = 0.0

            # Predict Cost using psi
            cost_pred_psi = C_psi(query.x, edge_index, edge_weight=edge_weights)
            L_outer = cost_pred_psi.mean()

            # predict penalty of final plan
            P_triple_out, P_join_in, P_join_out, P_acyclic, P_left_linear, P_entropy = compute_structure_penalties(edge_index, edge_weights, N_NODES, triples_num, device)
            total_penalty = (
                lambda_triple_out * P_triple_out
                + lambda_join_in * P_join_in
                + lambda_join_out * P_join_out
                + lambda_acyclic * P_acyclic
                + lambda_left_linear * P_left_linear
                + lambda_entropy * P_entropy
            )

            L_outer = (L_outer + total_penalty) / ACCUMULATION_STEPS


            # Supervised Anchor of C_theta on true cost
            pass

            # Backprop Outer-cost through inner gbjo and take gradient step:
            L_outer.backward()
            
            if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len(sparql_queries):
                params = list(C_theta.parameters()) + list(hyperparams.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                opt_theta.step()
                opt_theta.zero_grad(set_to_none=True)

            model_loss += cost_pred_psi.item()
            total_penalties += total_penalty.item()
            total_loss += L_outer.item()

        #### Reporting at the end of the epoch ####
        average_loss_per_epoch.append(model_loss / len(sparql_queries))
        average_total_penalty_per_epoch.append(total_penalties / len(sparql_queries))
        plt.plot(average_loss_per_epoch, label='Average Loss')
        plt.plot(average_total_penalty_per_epoch, label='Average Total Penalty')
        plt.legend()
        plt.savefig('meta_optimization_results.png')
        plt.close()
        if total_loss < best_loss:
            best_loss = total_loss
            # Save the model
            torch.save(C_theta.state_dict(), f'MetaOptimization_Best_Model.pt')
            # save hyperparams to json
            hyperparams_dict = {
                name: getattr(hyperparams, name)().item()
                for name in dir(hyperparams)
                if not name.startswith('_')
                and callable(getattr(hyperparams, name))
                and name not in dir(nn.Module)
            }
            with open('MetaOptimization_Best_Hyperparams.json', 'w') as f:
                json.dump(hyperparams_dict, f, indent=4)

