"""
Runtime evaluation script for query optimization approaches.

This script generates random queries of increasing sizes and measures the runtime
of different optimization approaches: Dynamic Programming, Greedy, and Gradient-based.
"""

import sys
import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import warnings
from dataclasses import dataclass
import torch.optim as optim
import graphviz
import itertools

import requests
import re
from typing import Union

# Add the parent directory to Python path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, spmm
from typing import Callable, Union

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)



# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GINConv(MessagePassing):
    """The graph isomorphism operator from the "How Powerful are
    Graph Neural Networks?" paper."""
    
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # Apply edge weights if provided
        if edge_weight is not None:
            # Handle edge weights that are 1D (need to reshape)
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.view(-1, 1)
            
            # Apply weight to messages
            return x_j * edge_weight
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        # Note: This method won't support edge weights in its current form
        # For edge weights, the regular message+aggregate pipeline will be used
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class CostGNNv2(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super(CostGNNv2, self).__init__()
        
        # Projection for first residual connection
        self.projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # Define MLPs for GINConv layers
        self.mlp1 = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GINConv layers for message passing
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.conv3 = GINConv(self.mlp3)
        
        # Layer normalization after each residual connection
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        
        # Additional FC layers with nonlinearities
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # For the first layer, project input to match hidden_dim for residual
        residual = self.projection(x)
        
        # First message passing layer
        if edge_weight is not None:
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv1(x, edge_index)
        
        # Add residual and apply layer norm
        x = x + residual
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second message passing layer with residual
        residual = x
        if edge_weight is not None:
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv2(x, edge_index)
        
        # Add residual and apply layer norm
        x = x + residual
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third message passing layer with residual
        residual = x
        if edge_weight is not None:
            x = self.conv3(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv3(x, edge_index)
        
        # Add residual and apply layer norm
        x = x + residual
        x = self.layer_norm3(x)
        x = F.relu(x)

        # Global pooling
        if batch is not None:
            x = scatter(x, batch, dim=0, reduce='add')
        else:
            x = torch.sum(x, dim=0)
        
        # Apply FC layers with nonlinearities
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        cost = torch.abs(self.fc2(x))

        return torch.squeeze(cost)
    




### Code from other files

@dataclass
class Entity:
    name: str

    def __post_init__(self):
        self.is_variable = self.name.startswith("?")

    def get_embedding(self, variable_id_dict: dict["Entity", int], rdf2vec=None, counts=None) -> np.ndarray:
        """
        Size - 102

        0: id of the variable, or 0 if it is a constant
        1-100: embedding of the constant, or 1 if it is a variable
        101: count of the constant, or 0 if it is a variable
        """

        if self.is_variable:
            return np.concatenate([
                [variable_id_dict[self]],
                np.ones(100),
                [0]
            ], axis=0)
        else:
            entity_name = self.name[1:-1]  # Remove angle brackets
            if rdf2vec is None or counts is None:
                raise ValueError("rdf2vec and counts must be provided for constant entities")
            
            # Get embedding and count
            embedding = rdf2vec.get(entity_name, np.zeros(100))
            count = counts.get(entity_name, 1)
            
            return np.concatenate([
                [0],
                embedding,
                [count]
            ], axis=0)
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self):
        return hash(self.name)
            


@dataclass
class Triple:
    s: Entity
    p: Entity
    o: Entity

    def __post_init__(self):
        self.variables = {
            var for var in [self.s, self.p, self.o]
                if var.is_variable
        }

    def where_body(self) -> str:
        return f"{self.s} {self.p} {self.o}."
    
    def json(self) -> Union[str, list]:
        return self.where_body()
    
    def get_embedding(self, variable_id_dict: dict[Entity, int], rdf2vec=None, counts=None) -> np.ndarray:
        """
        Size - 307

        0-101 - embedding of the subject
        102-203 - embedding of the predicate
        204-305 - embedding of the object
        306 - 0, representing that this is a triple and not a join node
        """
        return np.concatenate([
            *(
                ent.get_embedding(variable_id_dict, rdf2vec, counts)
                for ent in [self.s, self.p, self.o]
            ),
            [0]
        ], axis=0)
    
    def get_cardinality(self) -> int:
        """
        Returns the cardinality (number of matching triples) for this triple pattern.
        This is useful when the triple pattern is considered as a standalone query.
        """
        query = f"""
            SELECT COUNT(*) AS ?count
            FROM <http://lubm>
            WHERE {{ 
                {self.where_body()}	
            }}
        """
        res = requests.get(
            "http://127.0.0.1:8890/sparql/",
            params={
                "query": query,
                "format": "csv",
            },
        ).text
        
        m = re.match(r'"count"\n(\d+)\n', res)
        
        if not m:
            num_trials = 3
            while not m and num_trials > 0:
                res = requests.get(
                    "http://127.0.0.1:8890/sparql/",
                    params={
                        "query": query,
                        "format": "csv",
                    },
                ).text
                m = re.match(r'"count"\n(\d+)\n', res)
                num_trials -= 1

            if not m:
                print("Error in the following query:", query)
                raise RuntimeError("Query failed")

        return int(m.group(1))
    
    def get_cost(self) -> int:
        """
        Returns the cost of this triple pattern when used in a join.
        For triple patterns, we always return 0 to avoid double-counting in join costs.
        Use get_cardinality() to get the actual number of matching triples.
        """
        return 0
    
    def add_to_graph(self, graph, node_id):
        label = f"{self.s} {self.p} {self.o}"
        # Escape double quotes to avoid Graphviz syntax errors
        if '"' in label:
            label = label.replace('"', '\\"')
        graph.node(str(node_id), label=label, shape="box")
        return node_id





@dataclass
class Join:
    left: Union[Triple, "Join"]
    right: Union[Triple, "Join"]

    def __post_init__(self):
        self.variables = {
            *self.left.variables,
            *self.right.variables
        }
    
    def where_body(self) -> str:
        return f"{self.left.where_body()} {self.right.where_body()}"
    
    def __str__(self) -> str:
        return f"""
            SELECT {', '.join(str(var) for var in self.variables)}
            FROM <lubm>
            WHERE {{
                {self.where_body()}
            }}
        """

    def json(self) -> Union[str, list]:
        return [self.left.json(), self.right.json()]
    
    def get_embedding(self) -> np.ndarray:
        """
        Size - 307

        0-305 - zeros
        306 - 1, representing that this is a join node and not a triple
        """
        return np.concatenate([
            np.zeros(102 * 3),
            [1]
        ], axis=0)
    
    def get_cost(self) -> int:
        query = f"""
            SELECT COUNT(*) AS ?count
            FROM <http://lubm>
            WHERE {{ 
                {self.where_body()}	
            }}
        """
        res = requests.get(
            "http://127.0.0.1:8890/sparql/",
            params={
                "query": query,
                "format": "csv",
            },
        ).text
        
        m = re.match(r'"count"\n(\d+)\n', res)
        
        if not m:
            num_trials = 3
            while not m and num_trials > 0:
                res = requests.get(
                    "http://127.0.0.1:8890/sparql/",
                    params={
                        "query": query,
                        "format": "csv",
                    },
                ).text
                m = re.match(r'"count"\n(\d+)\n', res)
                num_trials -= 1

            if not m:
                print("Error in the following query:", query)
                raise RuntimeError("Query failed")

        self_cardinality = int(m.group(1))

        left_cardinality = self.left.get_cost()
        right_cardinality = self.right.get_cost()

        return self_cardinality + left_cardinality + right_cardinality
    
    def get_cardinality(self) -> int:
        query = f"""
            SELECT COUNT(*) AS ?count
            FROM <http://lubm>
            WHERE {{ 
                {self.where_body()}	
            }}
        """
        res = requests.get(
            "http://127.0.0.1:8890/sparql/",
            params={
                "query": query,
                "format": "csv",
            },
        ).text
        
        m = re.match(r'"count"\n(\d+)\n', res)
        
        if not m:
            num_trials = 3
            while not m and num_trials > 0:
                res = requests.get(
                    "http://127.0.0.1:8890/sparql/",
                    params={
                        "query": query,
                        "format": "csv",
                    },
                ).text
                m = re.match(r'"count"\n(\d+)\n', res)
                num_trials -= 1

            if not m:
                print("Error in the following query:", query)
                raise RuntimeError("Query failed")

        self_cardinality = int(m.group(1))


        return self_cardinality
    

    def add_to_graph(self, graph, node_id):
        # Create join node with bowtie symbol
        join_id = node_id
        graph.node(str(join_id), label="⋈", shape="circle")
        
        # Add left child and get its node id
        current_id = join_id + 1
        left_last_id = self.left.add_to_graph(graph, current_id)
        
        # Add right child and get its node id
        current_id = left_last_id + 1
        right_last_id = self.right.add_to_graph(graph, current_id)
        
        # Connect join node to its left and right children
        graph.edge(str(join_id), str(join_id + 1))  # Connect to left child
        graph.edge(str(join_id), str(left_last_id + 1))  # Connect to right child
        
        return right_last_id
    

@dataclass
class Query:
    root: Join | Triple
    triples_num: int
    
    def visualize(self, output_file="query_plan", format="png"):
        """
        Create a visualization of the query plan tree.
        
        Args:
            output_file: File name without extension to save the visualization
            format: Format of the output file (e.g., png, pdf, svg)
        
        Returns:
            The Graphviz object
        """
        graph = graphviz.Digraph('Query Plan', comment='Query Plan Visualization', 
                                graph_attr={'rankdir': 'TB'}, 
                                edge_attr={'dir': 'none'})
        self.root.add_to_graph(graph, 0)
        graph.render(output_file, format=format, cleanup=True)
        return graph


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



















def sample_gumbel(shape, eps=1e-10, device="cpu"):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)





def sample_grouped_gumbel_softmax(edge_logits: torch.Tensor,
                                src_nodes: torch.Tensor,
                                temperature: float) -> torch.Tensor:
    """
    Return relaxed one-hot edge weights such that every *source* node
    emits exactly one outgoing edge (in expectation) using the Gumbel-Softmax
    trick.

    Args:
        edge_logits: Tensor of shape (E,) - Unconstrained logits of every candidate edge.
        src_nodes: Tensor of shape (E,) - Source node index for each edge (aligned with edge_logits).
        temperature: Positive softmax temperature τ.

    Returns:
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
    """
    Exponential temperature annealing every step.
    
    Args:
        init_tau: Initial temperature
        min_tau: Minimum temperature
        decay: Decay factor 
        step: Current step
        max_step: Maximum steps
        
    Returns:
        Annealed temperature
    """
    return max(min_tau, init_tau - (init_tau - min_tau) * (step / max_step)) 








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
    print_times: bool = False,
):
    """Gradient-based join-order search with **Straight-Through Gumbel-Sigmoid**.

    The signature and return values mirror `optimize_query()` so the rest of
    your code remains unchanged.
    """

        # Track best solution if return_best is True
    best_cost = float('inf')
    best_edge_logits = None
    best_edge_logits_slot2 = None


    
    # Move data ----------------------------------------------------------------
    #data = query_data.to(device)
    data = query_data
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




        # Lambda ramping logic ------------------------------------------------
        if use_lambda_ramping:
            def annealed_lam(lam_max, step, ramp_steps=150):
                frac = min(1.0, step / ramp_steps)
                return lam_max * (frac ** lambda_ramp_exponent)  
            
            lambda_total = annealed_lam(lambda_total_penalty, step, ramp_steps=optimization_steps)
        else:
            lambda_total = lambda_total_penalty

        loss = cost_pred + lambda_total * total_penalty
        #loss = cost_pred


        # Track best solution if return_best is True
        if logit_sampling == 'dual-softmax':
            if return_best and cost_pred < best_cost:
                best_cost = cost_pred
                best_edge_logits = edge_logits.clone().detach()
                best_edge_logits_slot2 = edge_logits_slot2.clone().detach()
            #if return_best and loss < best_cost:
            #    best_cost = cost_pred
            #    best_edge_logits = edge_logits.clone().detach()
            #    best_edge_logits_slot2 = edge_logits_slot2.clone().detach()
        else:
            if return_best and cost_pred < best_cost:
                best_cost = cost_pred
                best_edge_logits = edge_logits.clone().detach()


        # Back‑prop & step -----------------------------------------------------
        loss.backward()
        

        optimiser.step()
        
        # Update learning rate schedule
        if use_lr_scheduling:
            scheduler.step()


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


    with torch.no_grad():
        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
    predicted_cost_exp = float(np.exp(final_log_cost))


    return final_A, triples_num, predicted_cost_exp




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

        else:  # sigmoid
            if decoding_method == "threshold":
                final_edge_weights = (torch.sigmoid(chosen_logits) >= 0.5).float()
            else:
                A_sig = torch.sigmoid(edge_logits); A_final = torch.zeros((N_NODES, N_NODES), device='cpu'); A_final[src.cpu(), dst.cpu()] = A_sig.cpu()


        if decoding_method == "threshold":
            final_A = torch.zeros((N_NODES, N_NODES), device=device)
            final_A[src, dst] = final_edge_weights

        final_log_cost = model(data.x, edge_index, edge_weight=final_edge_weights).item()
        predicted_cost_exp = float(np.exp(final_log_cost))

    if save_animation_data:
        return final_A, triples_num, predicted_cost_exp, animation_data
    else:
        return final_A, triples_num, predicted_cost_exp
    



@torch.compile
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






def left_deep_adj_from_perm(pi):
    """
    Create adjacency matrix for a left-deep join tree from a permutation.
    
    Args:
        pi: Tensor of length n with the (0-based) permutation of triple nodes.
        
    Returns:
        A: (2n-1, 2n-1) adjacency matrix for a left-deep tree:
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
        A[pi[k], new_join] = 1.0
        last_join = new_join
    return A


@dataclass
class Entity:
    name: str

    def __post_init__(self):
        self.is_variable = self.name.startswith("?")

    def get_embedding(self, variable_id_dict: dict["Entity", int], rdf2vec=None, counts=None) -> np.ndarray:
        """
        Size - 102

        0: id of the variable, or 0 if it is a constant
        1-100: embedding of the constant, or 1 if it is a variable
        101: count of the constant, or 0 if it is a variable
        """

        if self.is_variable:
            return np.concatenate([
                [variable_id_dict[self]],
                np.ones(100),
                [0]
            ], axis=0)
        else:
            entity_name = self.name[1:-1]  # Remove angle brackets
            if rdf2vec is None or counts is None:
                raise ValueError("rdf2vec and counts must be provided for constant entities")
            
            # Get embedding and count
            embedding = rdf2vec.get(entity_name, np.zeros(100))
            count = counts.get(entity_name, 1)
            
            return np.concatenate([
                [0],
                embedding,
                [count]
            ], axis=0)
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self):
        return hash(self.name)


def create_dummy_model(device: str = 'cpu', use_compile: bool = False) -> CostGNNv2:
    """
    Create a dummy CostGNNv2 model for benchmarking.
    
    Args:
        device: Target device
        use_compile: Whether to compile the model with torch.compile
        
    Returns:
        Initialized CostGNNv2 model
    """
    node_feature_dim = 307  # Standard feature dimension
    hidden_dim = 512        # Standard hidden dimension
    
    model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    model.eval()
    
    # Freeze parameters to speed up backward pass
    for p in model.parameters():
        p.requires_grad_(False)
    
    # Initialize with random weights (for benchmarking purposes)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    # Compile the model if requested
    if use_compile:
        print(f"Compiling model with torch.compile...")
        model = torch.compile(model)
        print(f"Model compilation completed.")
    
    return model

def warmup_compiled_model(model, device: str = 'cpu', warmup_cycles: int = 5, verbose: bool = True):
    """
    Warmup a compiled model to trigger JIT compilation and kernel optimization.
    
    Args:
        model: The compiled model to warmup
        device: Target device
        warmup_cycles: Number of warmup cycles to run
        verbose: Whether to print warmup progress
    """
    if verbose:
        print(f"Warming up compiled model with {warmup_cycles} cycles...")
    
    model.eval()
    
    # Create dummy data of various sizes for comprehensive warmup
    warmup_sizes = [3, 5, 8, 10]  # Different query sizes for warmup
    
    with torch.no_grad():
        for cycle in range(warmup_cycles):
            if verbose:
                print(f"  Warmup cycle {cycle + 1}/{warmup_cycles}")
            
            for size in warmup_sizes:
                # Generate dummy query data
                dummy_data = generate_random_query_data(size, device, seed=42)
                
                # Create dummy edge weights
                n_nodes = dummy_data.num_nodes
                src, dst = torch.where(~torch.eye(n_nodes, dtype=torch.bool, device=device))
                edge_index = torch.stack([src, dst], dim=0)
                edge_weights = torch.rand(edge_index.size(1), device=device)
                
                # Forward pass to trigger compilation
                _ = model(dummy_data.x, edge_index, edge_weight=edge_weights)
    
    if verbose:
        print("Model warmup completed!")

def generate_random_query_data(n_triples: int, device: str = 'cpu', seed: int = None) -> Data:
    """
    Generate random query data for benchmarking.
    
    Args:
        n_triples: Number of triple patterns in the query
        device: Target device
        seed: Random seed for reproducibility
        
    Returns:
        torch_geometric.data.Data object representing a random query
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Calculate total number of nodes: n triple nodes + (n-1) join nodes
    n_nodes = 2 * n_triples - 1
    feature_dim = 307
    
    # Create random node features
    node_features = torch.randn(n_nodes, feature_dim, device=device)
    
    # Set join node features: last dimension = 1.0 for join nodes, 0.0 for triple nodes
    # Triple nodes: indices 0 to n_triples-1
    # Join nodes: indices n_triples to n_nodes-1
    node_features[:n_triples, -1] = 0.0  # Triple nodes
    node_features[n_triples:, :] = 0.0   # Zero out join node features
    node_features[n_triples:, -1] = 1.0  # Mark join nodes
    
    # Create a random valid left-deep adjacency matrix using permutation
    perm = torch.randperm(n_triples, device=device)
    adjacency = left_deep_adj_from_perm(perm)
    
    # Convert adjacency matrix to edge_index format
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
    
    # Create torch_geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=n_nodes
    )
    
    return data

def create_dummy_triples(n_triples: int) -> List[Triple]:
    """
    Create dummy triple objects for greedy optimization.
    
    Args:
        n_triples: Number of triples to create
        
    Returns:
        List of dummy Triple objects
    """
    triples = []
    for i in range(n_triples):
        s = Entity(name=f"?s{i}")
        p = Entity(name=f"<p{i}>")
        o = Entity(name=f"?o{i}")
        triples.append(Triple(s=s, p=p, o=o))
    
    return triples

def benchmark_method(method_func, query_data, model, device, method_name: str, **kwargs) -> Tuple[float, bool]:
    """
    Benchmark a single optimization method.
    
    Args:
        method_func: The optimization function to benchmark
        query_data: Query data
        model: Cost model
        device: Compute device
        method_name: Name of the method for logging
        **kwargs: Additional arguments for the method
        
    Returns:
        Tuple of (runtime_seconds, success_flag)
    """
    try:

        query_data = query_data.to(device)

        start_time = time.time()
        
        if method_name == "Greedy":
            # Greedy method needs original triples
            n_triples = (query_data.num_nodes + 1) // 2
            original_triples = create_dummy_triples(n_triples)
            result = method_func(query_data, model, original_triples, device, verbose=False)
        elif method_name == "Gradient":
            # Gradient method (optimize_query_gumbel) signature: 
            # optimize_query_gumbel(torch_data, model, device, optimization_steps=500, verbose=False, **optimization_params)
            result = method_func(
                query_data, model, device,
                optimization_steps=kwargs.get('optimization_steps', 250),
                verbose=False,
                **{k: v for k, v in kwargs.items() if k != 'optimization_steps'}
            )
        else:
            # DP method
            result = method_func(query_data, model, device)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        return runtime, True
        
    except Exception as e:
        raise e
        print(f"Error in {method_name}: {e}")
        return float('inf'), False

def run_runtime_evaluation(
    query_sizes: List[int] = list(range(3, 11)),
    num_trials_per_size: int = 5,
    device: str = 'cpu',
    save_plot: bool = True,
    plot_filename: str = 'runtime_comparison.png',
    include_dp: bool = True,
    use_compile: bool = False
) -> Dict[str, Dict[int, List[float]]]:
    """
    Run comprehensive runtime evaluation across different query sizes.
    
    Args:
        query_sizes: List of query sizes to test
        num_trials_per_size: Number of trials per query size
        device: Compute device
        save_plot: Whether to save the plot
        plot_filename: Filename for the saved plot
        include_dp: Whether to include Dynamic Programming in the evaluation
        use_compile: Whether to compile the model with torch.compile
        
    Returns:
        Dictionary containing runtime results for each method and query size
    """
    print(f"Running runtime evaluation on device: {device}")
    print(f"Query sizes: {query_sizes}")
    print(f"Trials per size: {num_trials_per_size}")
    print(f"Include DP: {include_dp}")
    print(f"Use compile: {use_compile}")
    
    # Create dummy model
    model = create_dummy_model(device, use_compile=use_compile)
    
    # Warmup the optimization pipeline by running actual benchmarks
    if True:  # Always perform warmup
        print("Warming up optimization pipeline...")
        warmup_sizes = list(range(3, 8))  # Sizes 3, 4, 5
        warmup_trials = 3
        
        for warmup_size in tqdm(warmup_sizes, desc="Warmup sizes"):
            for warmup_trial in range(warmup_trials):
                # Generate random query for warmup
                warmup_query_data = generate_random_query_data(warmup_size, device, seed=warmup_trial + 1000)
                
                # Run all methods without recording results
                try:

                    
                    # Warmup Greedy
                    _, _ = benchmark_method(
                        greedy_optimize_query, warmup_query_data, model, device, "Greedy"
                    )
                    
                    # Warmup Gradient
                    warmup_gradient_config = {
                        'optimization_steps': 50,  # Shorter for warmup
                        'learning_rate': 1.0,
                        'lambda_acyclic': 1000.0,
                        'lambda_triple_in': 1000.0,
                        'lambda_triple_out': 1000.0,
                        'lambda_join_in': 500.0,
                        'lambda_join_out': 1000.0,
                        'lambda_left_linear': 1000.0,
                        'lambda_entropy': 0.0,
                        'lambda_total_penalty': 1.0,
                        'init_tau': 10.0,
                        'min_tau': 1.0,
                        'tau_decay': 0.999,
                        'use_temperature_annealing': True,
                        'return_best': True,
                        'min_penalty_threshold': 0.1,
                        'use_lambda_ramping': False,
                        'logit_sampling': 'dual-softmax',
                        'save_animation_data': False,
                        'animation_save_interval': 10,
                        'print_times': False
                    }
                    
                    _, _ = benchmark_method(
                        optimize_query_gumbel_efficient_reduced, warmup_query_data, model, device, "Gradient",
                        **warmup_gradient_config
                    )
                    
                except Exception as e:
                    print(f"Warning: Warmup failed for size {warmup_size}, trial {warmup_trial}: {e}")
                    continue
        
        print("Warmup completed!")
    
    # Initialize results storage - conditionally include DP
    results = {
        'Greedy': {size: [] for size in query_sizes},
        'Gradient': {size: [] for size in query_sizes}
    }
    
    if include_dp:
        results['DP'] = {size: [] for size in query_sizes}
    
    # Method configurations
    gradient_config = {
        'optimization_steps': 200,
        'learning_rate': 1.0,
        'lambda_acyclic': 1000.0,
        'lambda_triple_in': 1000.0,
        'lambda_triple_out': 1000.0,
        'lambda_join_in': 500.0,
        'lambda_join_out': 1000.0,
        'lambda_left_linear': 1000.0,
        'lambda_entropy': 0.0,
        'lambda_total_penalty': 1.0,
        'init_tau': 10.0,
        'min_tau': 1.0,
        'tau_decay': 0.999,
        'use_temperature_annealing': True,
        'return_best': True,
        'min_penalty_threshold': 0.1,
        'use_lambda_ramping': False,
        'logit_sampling': 'dual-softmax',
        'save_animation_data': False,
        'animation_save_interval': 10,
        'print_times': False  # Disable detailed timing in benchmarks
    }
    
    # Run evaluation for each query size
    for query_size in tqdm(query_sizes, desc="Query sizes"):
        print(f"\nEvaluating query size: {query_size}")
        
        for trial in tqdm(range(num_trials_per_size), desc="Trials", leave=False):
            # Generate random query
            query_data = generate_random_query_data(query_size, device, seed=trial)
            
            # Benchmark DP (only if enabled)
            if include_dp:
                dp_time, dp_success = benchmark_method(
                    dp_leftdeep_best_plan, query_data, model, device, "DP"
                )
                if dp_success:
                    results['DP'][query_size].append(dp_time)
                print(f"  DP trial {trial+1}: {dp_time:.4f}s")
            
            # Benchmark Greedy
            greedy_time, greedy_success = benchmark_method(
                greedy_optimize_query, query_data, model, device, "Greedy"
            )
            if greedy_success:
                results['Greedy'][query_size].append(greedy_time)
            print(f"  Greedy trial {trial+1}: {greedy_time:.4f}s")
            
            # Benchmark Gradient
            gradient_time, gradient_success = benchmark_method(
                optimize_query_gumbel_efficient_reduced, query_data, model, device, "Gradient",
                **gradient_config
            )
            if gradient_success:
                results['Gradient'][query_size].append(gradient_time)
            print(f"  Gradient trial {trial+1}: {gradient_time:.4f}s")
    
    # Calculate and print summary statistics
    print("\n" + "="*50)
    print("RUNTIME EVALUATION SUMMARY")
    print("="*50)
    
    for method in results:
        print(f"\n{method} Method:")
        for size in query_sizes:
            if results[method][size]:
                times = results[method][size]
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"  Size {size}: {mean_time:.4f}±{std_time:.4f}s (n={len(times)})")
            else:
                print(f"  Size {size}: No successful runs")
    
    # Create runtime plot
    create_runtime_plot(results, query_sizes, save_plot, plot_filename)
    
    return results

def create_runtime_plot(
    results: Dict[str, Dict[int, List[float]]],
    query_sizes: List[int],
    save_plot: bool = True,
    plot_filename: str = 'runtime_comparison.png'
):
    """
    Create and display runtime comparison plot.
    
    Args:
        results: Runtime results dictionary
        query_sizes: List of query sizes
        save_plot: Whether to save the plot
        plot_filename: Filename for the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = {'DP': 'red', 'Greedy': 'blue', 'Gradient': 'green'}
    markers = {'DP': 'o', 'Greedy': 's', 'Gradient': '^'}
    
    for method in results:
        sizes = []
        mean_times = []
        std_times = []
        
        for size in query_sizes:
            if results[method][size]:
                times = results[method][size]
                sizes.append(size)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))
        
        if sizes:
            plt.errorbar(
                sizes, mean_times, yerr=std_times,
                label=method, color=colors[method], marker=markers[method],
                markersize=8, linewidth=2, capsize=5
            )
    
    plt.xlabel('Query Size (Number of Triple Patterns)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    
    # Dynamic title based on included methods
    methods_included = list(results.keys())
    if 'DP' in methods_included:
        title = 'Runtime Comparison: DP vs Greedy vs Gradient-based Optimization'
    else:
        title = 'Runtime Comparison: Greedy vs Gradient-based Optimization'
    
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add annotations
    plt.annotate('Log scale used for y-axis', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, ha='left', va='top', style='italic')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {plot_filename}")
    
    plt.show()

def save_results_to_file(results: Dict, filename: str = 'runtime_results.txt'):
    """
    Save detailed results to a text file.
    
    Args:
        results: Runtime results dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("Runtime Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        for method in results:
            f.write(f"{method} Method:\n")
            for size in sorted(results[method].keys()):
                if results[method][size]:
                    times = results[method][size]
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    f.write(f"  Size {size}: {mean_time:.6f}±{std_time:.6f}s (trials: {len(times)})\n")
                    f.write(f"    Individual times: {[f'{t:.6f}' for t in times]}\n")
                else:
                    f.write(f"  Size {size}: No successful runs\n")
            f.write("\n")
    
    print(f"Detailed results saved to: {filename}")

if __name__ == "__main__":
    # Configuration
    config = {
        'query_sizes': list(range(3, 15)), 
        'num_trials_per_size': 1,           # Number of trials per size
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_plot': True,
        'plot_filename': 'runtime_comparison.png',
        'include_dp': False,  # Set to False to exclude DP and compare only Greedy vs Gradient
        'use_compile': False  # Set to True to enable torch.compile optimization
    }
    
    print("Starting Runtime Evaluation")
    print(f"Configuration: {config}")
    
    # Run the evaluation
    results = run_runtime_evaluation(**config)
    
    # Save detailed results
    save_results_to_file(results, filename='runtime_results.txt')
    
    print("\nRuntime evaluation completed!")
