import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))



from model import CostGNNv3
from optimization.gumbel_utils import sample_grouped_gumbel_softmax
from utils.data_utils import load_sparql_queries
from src.create_data.create_optimization_data import SPARQLQuery



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
    def __init__(self, init_lambda_join_in=1.0, init_eta=1e0) -> None:
        super().__init__()

        self._lambda_join_in = nn.Parameter(torch.tensor(float(init_lambda_join_in)).clamp(min=-2, max=3000))
        self._eta = nn.Parameter(torch.tensor(float(init_eta)).clamp(min=1e-4, max=2))

    def lambda_join_in(self) -> torch.Tensor:
        return F.softplus(self._lambda_join_in)

    def eta(self) -> torch.Tensor:
        return self._eta


def gbjo(query, C_theta, hyperparams, device="cpu"):
    """
    GBJO algorithm, adapted to be fully backpropagable

    returns final soft logits
    """

    N_STEPS = 10
    tau = 1
    #learning_rate = 0.01

    N_NODES = len(query.x)
    triples_num = (N_NODES + 1) // 2  # n triples ➜ 2n‑1 nodes

    # Enumerate all candidate edges (excluding self‑loops) - we have all-to-all edges because we need to consider all possible plans
    src, dst = torch.where(~torch.eye(N_NODES, dtype=torch.bool))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    num_edges = edge_index.size(1)
    
    logits = torch.zeros(num_edges, requires_grad=True)

    for step in range(N_STEPS):
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
        pass

        # backprop cost to logits
        (g,) = torch.autograd.grad(cost_pred, logits, create_graph=True)

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

    # Load queries
    sparql_queries = load_sparql_queries(QUERY_PATH, 100)
    sparql_queries = [q for q in sparql_queries if len(q.triples) == 3]
    # Define C_theta (cost model to be meta-optimized)
    C_theta = CostGNNv3(node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT).to(device)
    C_theta.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # Define C_psi (reezed critic cost model)
    C_psi = CostGNNv3(node_feature_dim=NODE_FEATURE_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, use_jk=USE_JK, jk_mode=JK_MODE, use_residual=USE_RESIDUAL, use_layer_norm=USE_LAYER_NORM, dropout=DROPOUT).to(device)
    C_psi.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Define hyperparams
    hyperparams = Hyperparams(init_lambda_join_in=1.0)


    # define outer optimizer for psi_theta (updates C_theta params and hyperparams)
    opt_theta = torch.optim.AdamW(list(C_theta.parameters()) + list(hyperparams.parameters()), lr=1e-4, weight_decay=1e-4)
    #opt_theta = torch.optim.AdamW(hyperparams.parameters(), lr=1e-2, weight_decay=1e-2)


    EPOCHS = 100

    query = None

    # Freeze c_psi as we never train it
    freeze_(C_psi)

    for i in range(0, EPOCHS):
        for query in sparql_queries:
            query = sparql_queries[0]
            query = query.torch_data[0]
            # TODO add gaussian fingerprints

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

            # Supervised Anchor of C_theta on true cost
            pass

            # Backprop Outer-cost through inner gbjo and take gradient step:
            opt_theta.zero_grad(set_to_none=True)
            L_outer.backward()
            opt_theta.step()

            print(f"Epoch {i} - Eta: {hyperparams.eta().item()} - Outer Cost: {L_outer.item()}")
            print(C_theta.parameters())



            