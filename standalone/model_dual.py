"""
CostGNNDual: CostGNNv3 with a second, fixed adjacency over triples that share
a variable. Each GIN layer aggregates over BOTH graphs; the layer MLP sees
the concatenation, so the model can directly relate a join's children to the
variable-sharing structure (what the old representation made invisible).

Layer:  h <- h + MLP([aggP + (1+eps)h  ;  aggS + (1+eps)h])
        aggP = sum over plan edges (child -> parent, directed)
        aggS = sum over share edges (triple <-> triple, symmetric)

Everything else matches CostGNNv3 (use_residual=True, no norms, aggr='add'):
sign*log1p input transform, 307->H projection, sum pooling, fc1-GELU-fc2.
Implemented with plain index_add_ (no PyG dependency) so it runs anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatCostGNNDual:
    """CostGNNDual as plain tensors with the FlatCostGNN interface, for use
    inside FastGBJO. The share adjacency is fixed per query: call
    bind_share(S_dense) before optimizing (FastGBJO does this when it sees
    this class). Aggregations are dense: aggP = A^T @ h, aggS = S @ h."""

    is_dual = True

    def __init__(self, state_dict, n_layers=6):
        sd = state_dict
        self.proj_w = sd["projection.weight"]
        self.proj_b = sd["projection.bias"]
        self.layers = [(sd[f"mlps.{i}.0.weight"], sd[f"mlps.{i}.0.bias"],
                        sd[f"mlps.{i}.2.weight"], sd[f"mlps.{i}.2.bias"])
                       for i in range(n_layers)]
        self.fc1_w = sd["fc1.weight"]
        self.fc1_b = sd["fc1.bias"]
        self.fc2_w = sd["fc2.weight"]
        self.fc2_b = sd["fc2.bias"]
        self._S = None

    @classmethod
    def load(cls, model_path, n_layers=6):
        sd = torch.load(model_path, map_location="cpu")
        return cls(sd, n_layers=n_layers)

    def bind_share(self, S_dense):
        """S_dense: (N, N) binary symmetric triple-sharing adjacency."""
        self._S = S_dense

    def project_x(self, x):
        xl = torch.sign(x) * torch.log1p(torch.abs(x))
        return F.linear(xl, self.proj_w, self.proj_b)

    def forward_from_h0(self, h, A):
        assert self._S is not None, "call bind_share(S) first"
        S = self._S
        At = A.t()
        for w1, b1, w2, b2 in self.layers:
            z = torch.cat([At @ h + h, S @ h + h], dim=1)
            z = F.linear(z, w1, b1)
            z = F.gelu(z)
            z = F.linear(z, w2, b2)
            h = h + z
        g = h.sum(0)
        g = F.gelu(F.linear(g, self.fc1_w, self.fc1_b))
        return F.linear(g, self.fc2_w, self.fc2_b).squeeze(-1)


class CostGNNDual(nn.Module):
    def __init__(self, node_feature_dim=307, hidden_dim=128, n_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.projection = nn.Linear(node_feature_dim, hidden_dim)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                          nn.GELU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, edge_index_share, batch, num_graphs=None):
        """
        Args:
            x: (N_total, F) raw features (fingerprints already applied)
            edge_index: (2, E) plan edges child -> parent
            edge_index_share: (2, Es) symmetric share edges between triples
            batch: (N_total,) graph id per node
        Returns:
            (num_graphs,) predicted log-cost
        """
        xl = torch.sign(x) * torch.log1p(torch.abs(x))
        h = self.projection(xl)

        sp, dp = edge_index[0], edge_index[1]
        ss, ds = edge_index_share[0], edge_index_share[1]
        for mlp in self.mlps:
            aggP = torch.zeros_like(h).index_add_(0, dp, h[sp])
            aggS = torch.zeros_like(h).index_add_(0, ds, h[ss])
            z = torch.cat([aggP + h, aggS + h], dim=1)   # eps = 0
            h = h + mlp(z)

        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1
        g = torch.zeros(num_graphs, self.hidden_dim, device=h.device,
                        dtype=h.dtype).index_add_(0, batch, h)
        g = F.gelu(self.fc1(g))
        return self.fc2(g).squeeze(-1)
