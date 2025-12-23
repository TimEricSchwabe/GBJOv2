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


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))



from model import CostGNNv3
from optimization.gumbel_utils import sample_grouped_gumbel_softmax
from utils.data_utils import load_sparql_queries
from src.create_data.create_optimization_data import SPARQLQuery


# differentiable temperature annealing
def _temperature_anneal(init_tau: torch.Tensor, min_tau: float, decay: float, step: int, max_step: int, device="cpu") -> torch.Tensor:
    """
    Exponential temperature annealing every step (differentiable version).
    """
    annealed = init_tau - (init_tau - min_tau) * (step / max_step)
    return torch.maximum(annealed, torch.tensor(min_tau, device=device))



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
    def __init__(self, init_lambda_triple_out=1.0,
     init_lambda_join_in=1.0, init_lambda_join_out=1.0, init_lambda_acyclic=1.0,
      init_lambda_left_linear=1.0, init_lambda_entropy=1.0, init_eta=0.8, init_tau=5.0) -> None:
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
        lambda_min, lambda_max = 0, 500
        return lambda_min + F.softplus(self._lambda_triple_out).clamp(min=0, max=500)

    def lambda_join_in(self) -> torch.Tensor:
        lambda_min, lambda_max = 0, 500
        return lambda_min + F.softplus(self._lambda_join_in).clamp(min=0, max=500)

    def lambda_join_out(self) -> torch.Tensor:
        lambda_min, lambda_max = 0, 500
        return lambda_min + F.softplus(self._lambda_join_out).clamp(min=0, max=500)

    def lambda_acyclic(self) -> torch.Tensor:
        lambda_min, lambda_max = 0, 500
        return lambda_min + F.softplus(self._lambda_acyclic).clamp(min=0, max=500)

    def lambda_left_linear(self) -> torch.Tensor:
        lambda_min, lambda_max = 0, 500
        return lambda_min + F.softplus(self._lambda_left_linear).clamp(min=0, max=500)

    def lambda_entropy(self) -> torch.Tensor:
        lambda_min, lambda_max = 0, 500
        return lambda_min + F.softplus(self._lambda_entropy).clamp(min=0, max=500)

    def eta(self) -> torch.Tensor:
        #eta_min, eta_max = 1e-4, 2
        #return eta_min + (eta_max - eta_min) * F.sigmoid(self._eta)
        return F.softplus(self._eta).clamp(min=1e-4, max=2)

    def init_tau(self) -> torch.Tensor:
        return F.softplus(self._init_tau).clamp(min=1, max=10)


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


def gbjo(query, C_theta, hyperparams, device="cpu"):
    """
    GBJO algorithm, adapted to be fully backpropagable

    returns final soft logits
    """

    N_STEPS = 100
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
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
    edge_index = torch.stack([src, dst], dim=0)
    num_edges = edge_index.size(1)
    
    logits = torch.zeros(num_edges, requires_grad=True, device=device)
    v = torch.zeros_like(logits)  # velocity buffer
    g = torch.zeros_like(logits)  # gradient buffer

    for step in range(N_STEPS):
        tau = _temperature_anneal(torch.tensor(5.0, device=device), 1.0, 0.999, step, N_STEPS, device=device)
        learning_rate = hyperparams.eta()
        lambda_triple_out = hyperparams.lambda_triple_out()
        lambda_join_in = hyperparams.lambda_join_in()
        lambda_join_out = hyperparams.lambda_join_out()
        lambda_left_linear = hyperparams.lambda_left_linear()
        lambda_acyclic = hyperparams.lambda_acyclic()

        # First, mask out invalid edges in logits
        masked_logits = logits.clone()
        
        # Triple nodes cannot connect to other triple nodes
        triple_to_triple_mask = (edge_index[0] < triples_num) & (edge_index[1] < triples_num)
        masked_logits[triple_to_triple_mask] = float('-1e9')
        
        # Join nodes cannot connect to triple nodes
        join_to_triple_mask = (edge_index[0] >= triples_num) & (edge_index[1] < triples_num)
        masked_logits[join_to_triple_mask] = float('-1e9')
        
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

        # Calculate the ramping coefficient for the current step
        frac = min(1.0, step / N_STEPS)
        coefficient =  frac ** 3
        #coefficient = 1 # TODO currently no ramping, adapt if necessary

        # backprop cost to logits
        momentum = 0.9  # momentum coefficient


        cost = (cost_pred + coefficient * total_penalty).mean()
        #cost = cost_pred.mean() #  case to not use penalties in inner unroll
        (g,) = torch.autograd.grad(cost, logits, create_graph=True)

        # Gradient Descent Update
        v = momentum * v + g
        logits = logits - learning_rate * (momentum * v + g)

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

    ACCUMULATION_STEPS = 16  # aka batch size, we accumulate gradients over this many steps

    # Create a dedicated output directory for this run (matches evaluation_parallel.py style)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join("meta_optimization_results", f"run_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)
    print(f"Saving all training outputs to: {save_directory}")

    # Load queries
    #sparql_queries = load_sparql_queries(QUERY_PATH, 10)
    sparql_queries = torch.load(QUERY_PATH, weights_only=False)
    sparql_queries = sparql_queries['data']
    random.shuffle(sparql_queries)
    sparql_queries = sparql_queries[:10000]
    #sparql_queries = [q for q in sparql_queries if len(q.triples) == 3]
    # print how many queries are loaded and used
    print(f"INFO: Loaded {len(sparql_queries)} queries")
    # Define C_theta (cost model to be meta-optimized)
    C_theta = CostGNNv3(node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT).to(device)
    C_theta.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # Define C_psi (reezed critic cost model)
    C_psi = CostGNNv3(node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT).to(device)
    C_psi.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Define hyperparams
    hyperparams = Hyperparams()

    # Save training config for reproducibility
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
            "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
            "EPOCHS": 100,
            "outer_optimizer": "AdamW",
            "lr": 1e-2,
            "weight_decay": 1e-2,
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

    hyperparam_history = {
        "lambda_triple_out": [],
        "lambda_join_in": [],
        "lambda_join_out": [],
        "lambda_acyclic": [],
        "lambda_left_linear": [],
        "lambda_entropy": [],
        "eta": [],
        "init_tau": [],
    }

    # Define hyperparams from config
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


    # define outer optimizer for psi_theta (updates C_theta params and hyperparams)
    #opt_theta = torch.optim.Adam(list(C_theta.parameters()) + list(hyperparams.parameters()), lr=1e-4)
    #opt_theta = torch.optim.Adam(hyperparams.parameters(), lr=1e-3)
    opt_theta = torch.optim.AdamW(C_theta.parameters(), lr=1e-5)

    # different lr for hyper and model
    #opt_theta = torch.optim.AdamW(
    #[
    #    
   #     {"params": hyperparams.parameters(), "lr": 1e-5, "weight_decay": 0.0},
        # Cost model
    #    {"params": C_theta.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
    #],
    #eps=1e-8
    #)


    anchor_loss = nn.HuberLoss()


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
    average_anchor_loss_per_epoch = []
    best_loss = float('inf')

    # Step-wise loss tracking (average over every 100 steps)
    avg_loss_per_100_steps = []
    avg_penalty_per_100_steps = []
    avg_anchor_loss_per_100_steps = []
    # Accumulators for 100-step windows
    window_loss = 0.0
    window_penalty = 0.0
    window_anchor_loss = 0.0
    global_step = 0

    for i in range(0, EPOCHS):
        model_loss = 0
        total_penalties = 0 
        total_loss = 0
        total_anchor_loss = 0
        opt_theta.zero_grad(set_to_none=True)
        single_query = sparql_queries[0]
        single_query = add_fingerprints_to_query_data(single_query, fingerprint_dim=64) # TODO remove jsut test for overfitting



        for idx, query in enumerate(tqdm(sparql_queries, desc=f"Epoch {i+1}/{EPOCHS}")):
            #query = sparql_queries[0]
            #query = query.torch_data[0]
            # TODO add gaussian fingerprints
            query = add_fingerprints_to_query_data(query, fingerprint_dim=64)
            #query = single_query # TODO remove

            # Move query data to device
            query.x = query.x.to(device)
            query.edge_index = query.edge_index.to(device)
            query.y = query.y.to(device)

            N_NODES = len(query.x)
            triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n‑1 nodes

            # Enumerate all candidate edges (excluding self‑loops)
            src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool, device=device))
            edge_index = torch.stack([src, dst], dim=0)
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

            ### Just to check the gradinet s
            grads = torch.autograd.grad(L_outer, hyperparams.parameters(), retain_graph=True, allow_unused=True)
            grad_norms = [None if g is None else g.norm().item() for g in grads]
            #m: {grad_norms}")

            # predict penalty of final plan
            P_triple_out, P_join_in, P_join_out, P_acyclic, P_left_linear, P_entropy = compute_structure_penalties(edge_index, edge_weights, N_NODES, triples_num, device)
            L_struct_al = (
               lambda_triple_out * P_triple_out
               + lambda_join_in * P_join_in
               + lambda_join_out * P_join_out
               + lambda_acyclic * P_acyclic
               + lambda_left_linear * P_left_linear
               + lambda_entropy * P_entropy
            )



            # Supervised Anchor of C_theta on true cost
            anchor_plan_pred = C_theta(query.x, query.edge_index)
            L_anchor = anchor_loss(anchor_plan_pred, torch.log(query.y))

            L_outer = (L_outer + L_anchor + 0.1 * L_struct_al) 


            # Checking gradient magnitude
            # Use allow_unused=True and don't unpack with comma (it returns a tuple of gradients)
            #g_m = torch.autograd.grad(L_outer, C_theta.parameters(), retain_graph=True, allow_unused=True)
            #g_s = torch.autograd.grad(L_struct_al, C_theta.parameters(), retain_graph=True, allow_unused=True)
            #g_a = torch.autograd.grad(L_anchor, C_theta.parameters(), retain_graph=True, allow_unused=True)

            # Compute norms, filtering out None gradients (for unused params)
            #nm = torch.sqrt(sum(g.pow(2).sum() for g in g_m if g is not None))
            #ns = torch.sqrt(sum(g.pow(2).sum() for g in g_s if g is not None))
            #na = torch.sqrt(sum(g.pow(2).sum() for g in g_a if g is not None))
            #print(f"INFO: Grad Norm per loss term - L_outer: {nm.item()}, penalty: {ns.item()}, anchor: {na.item()}")





            print(f"INFO: L_cost_pred: {cost_pred_psi.item()}, total_penalty: {L_struct_al.item()} L_anchor: {L_anchor.item()}")


            # Backprop Outer-cost through inner gbjo and take gradient step:
            L_outer.backward()

            if (idx + 1) % ACCUMULATION_STEPS == 0 or (idx + 1) == len(sparql_queries):
                #params = list(C_theta.parameters()) + list(hyperparams.parameters())
                params = list(C_theta.parameters())
                #params = list(hyperparams.parameters())

                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1)
                print(f"INFO: Grad Norm: {grad_norm}")
                opt_theta.step()
                opt_theta.zero_grad(set_to_none=True)



            model_loss += cost_pred_psi.item()
            total_penalties += L_struct_al.item()
            total_loss += L_outer.item()
            total_anchor_loss += L_anchor.item()

            # Step-wise loss tracking (average over 100 steps)
            global_step += 1
            window_loss += cost_pred_psi.item()
            window_penalty += L_struct_al.item()
            window_anchor_loss += L_anchor.item()

            # Log averaged plots every 100 steps
            if global_step % 100 == 0:
                avg_loss_per_100_steps.append(window_loss / 100)
                avg_penalty_per_100_steps.append(window_penalty / 100)
                avg_anchor_loss_per_100_steps.append(window_anchor_loss / 100)
                window_loss = 0.0
                window_penalty = 0.0
                window_anchor_loss = 0.0

                plt.figure()
                plt.plot(avg_loss_per_100_steps, label='Avg Loss (100 steps)')
                plt.xlabel("100-step Window")
                plt.ylabel("Loss")
                plt.title(f"Average Loss per 100 Steps (step {global_step})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_directory, 'step_loss_plot.png'))
                plt.close()

                plt.figure()
                plt.plot(avg_penalty_per_100_steps, label='Avg Penalty (100 steps)')
                plt.xlabel("100-step Window")
                plt.ylabel("Penalty")
                plt.title(f"Average Penalty per 100 Steps (step {global_step})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_directory, 'step_penalty_plot.png'))
                plt.close()

                plt.figure()
                plt.plot(avg_anchor_loss_per_100_steps, label='Avg Anchor Loss (100 steps)')
                plt.xlabel("100-step Window")
                plt.ylabel("Anchor Loss")
                plt.title(f"Average Anchor Loss per 100 Steps (step {global_step})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_directory, 'step_anchor_loss_plot.png'))
                plt.close()
            if global_step % 1000 == 0:
                torch.save(C_theta.state_dict(), os.path.join(save_directory, f'model_epoch_{global_step}.pt'))


        #### Reporting at the end of the epoch ####
        for name in hyperparam_history.keys():
            hyperparam_history[name].append(getattr(hyperparams, name)().item())

        average_loss_per_epoch.append(model_loss / len(sparql_queries))
        average_total_penalty_per_epoch.append(total_penalties / len(sparql_queries))
        average_anchor_loss_per_epoch.append(total_anchor_loss / len(sparql_queries))
        plt.plot(average_loss_per_epoch, label='Average Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'loss_plot.png'))
        plt.close()
        
        plt.plot(average_total_penalty_per_epoch, label='Average Total Penalty')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'penalty_plot.png'))
        plt.close()

        plt.plot(average_anchor_loss_per_epoch, label='Average Anchor Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, 'anchor_loss_plot.png'))
        plt.close()

        plot_hyperparameter_history(hyperparam_history, save_directory)

        torch.save(C_theta.state_dict(), os.path.join(save_directory, f'model_epoch_{i}.pt'))


        if total_loss < best_loss:
            best_loss = total_loss
            # Save the model
            torch.save(C_theta.state_dict(), os.path.join(save_directory, 'best_model.pt'))
            # save hyperparams to json
            hyperparams_dict = {
                name: getattr(hyperparams, name)().item()
                for name in dir(hyperparams)
                if not name.startswith('_')
                and callable(getattr(hyperparams, name))
                and name not in dir(nn.Module)
            }
            with open(os.path.join(save_directory, 'best_hyperparams.json'), 'w') as f:
                json.dump(hyperparams_dict, f, indent=4)

