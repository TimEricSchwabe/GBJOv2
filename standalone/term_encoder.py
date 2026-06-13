"""
FICE-style term encoder for GBJO: produces an embedding for each bound query
term from its sampled factor-graph neighborhood (kg_index.py), replacing the
fixed rdf2vec vectors in the decoder input x. Trained end-to-end with
CostGNNDual through the existing Huber + ranking losses.

Design (per discussion):
- one subgraph PER TERM (disjoint PyG-style batch), so attention/messages
  never cross terms and a term's embedding is independent of batch
  composition -> training-time and offline pack-time embeddings agree.
- subgraph sampling is seeded by the term's node id (kg_index), so the
  neighborhood -- and the cached PE -- is fixed across epochs and runs.
- node features: [0.1 base | log1p (o_s,o_p,o_o) | PE | rdf2vec | center
  flag] -> Linear -> LayerNorm -> SiLU, + node-type embedding (FICE's
  relational mode). counts / rdf2vec inputs are config flags.
- layers: 'gine' (GINEConv) or 'gps' (GPSConv = local GINE + per-subgraph
  self-attention), role-embedded edge attributes; PE: role-aware RWPE
  (default), Laplacian eigenvectors, or none.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE


def compute_pe(edge_index, role_idx, num_nodes, kind, dim):
    """Positional encoding on one subgraph. kind 'rwpe': role-aware random
    walk PE (FICE compute_rwpe), dim must be divisible by 3 (walk length =
    dim/3 per role). kind 'lap': Laplacian eigenvector PE."""
    if kind == "none" or dim <= 0:
        return torch.zeros(num_nodes, 0)
    if kind == "rwpe":
        walk = dim // 3
        transform = AddRandomWalkPE(walk_length=walk)
        parts = []
        for role in range(3):  # subject, predicate, object (idx % 3)
            mask = (role_idx % 3) == role
            d = Data(edge_index=edge_index[:, mask], num_nodes=num_nodes)
            parts.append(transform(d).random_walk_pe)
        return torch.cat(parts, dim=-1)
    if kind == "lap":
        k = min(dim, num_nodes - 2)
        if k <= 0:
            return torch.zeros(num_nodes, dim)
        d = AddLaplacianEigenvectorPE(k=k)(
            Data(edge_index=edge_index, num_nodes=num_nodes))
        pe = d.laplacian_eigenvector_pe
        if pe.shape[1] < dim:
            pe = torch.cat([pe, torch.zeros(num_nodes, dim - pe.shape[1])], -1)
        return pe
    raise ValueError(kind)


class TermEncoder(nn.Module):
    def __init__(self, hidden=128, out_dim=100, n_layers=4, arch="gps",
                 pe="rwpe", pe_dim=24, use_rdf2vec=False, use_counts=True,
                 heads=4, attn="multihead", local_mp=True, use_fanout=True,
                 rel_emb=True, n_relations=0):
        super().__init__()
        if pe == "rwpe" and pe_dim % 3:
            raise ValueError("rwpe pe_dim must be divisible by 3 (role-aware)")
        if rel_emb and n_relations <= 0:
            raise ValueError("rel_emb=True requires n_relations > 0")
        self.arch = arch
        self.pe = pe
        self.pe_dim = pe_dim if pe != "none" else 0
        self.use_rdf2vec = use_rdf2vec
        self.use_counts = use_counts
        self.use_fanout = use_fanout
        self.rel_emb = rel_emb
        raw = 1 + (3 if use_counts else 0) + (2 if use_fanout else 0) \
            + self.pe_dim + (100 if use_rdf2vec else 0) + 1
        self.feature_encoder = nn.Linear(raw, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.node_type_emb = nn.Embedding(3, hidden)
        self.role_emb = nn.Embedding(6, hidden)
        # learnable per-relation identity; id 0 = padding (non-relation nodes)
        self.rel_emb_table = (nn.Embedding(n_relations + 1, hidden,
                                           padding_idx=0) if rel_emb else None)

        def gine_mlp():
            return nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            conv = GINEConv(gine_mlp(), edge_dim=hidden, train_eps=False)
            if arch == "gps":
                self.layers.append(GPSConv(hidden, conv if local_mp else None,
                                           heads=heads, dropout=0.0,
                                           attn_type=attn))
            elif arch == "gine":
                self.layers.append(conv)
            else:
                raise ValueError(arch)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, occ, fanout, pe, emb, center, ntype, rel_id, edge_index,
                role_idx, batch_vec, center_pos):
        parts = [torch.full((occ.shape[0], 1), 0.1, device=occ.device)]
        if self.use_counts:
            parts.append(occ)
        if self.use_fanout:
            parts.append(fanout)
        if self.pe_dim:
            parts.append(pe)
        if self.use_rdf2vec:
            parts.append(emb)
        parts.append(center)
        h = F.silu(self.input_norm(self.feature_encoder(torch.cat(parts, -1))))
        h = h + self.node_type_emb(ntype)
        if self.rel_emb_table is not None:
            h = h + self.rel_emb_table(rel_id)
        edge_attr = self.role_emb(role_idx)
        for layer in self.layers:
            if self.arch == "gps":
                h = layer(h, edge_index, batch_vec, edge_attr=edge_attr)
            else:
                h = h + layer(h, edge_index, edge_attr=edge_attr)
        return self.out_proj(h[center_pos])


class SubgraphProvider:
    """Caches one fixed subgraph (tensors + PE + pack-embedding rows) per
    term and assembles disjoint batches for TermEncoder."""

    def __init__(self, kg, pe="rwpe", pe_dim=24, caps=(16, 8, 4),
                 pack_dir=None, use_rdf2vec=False):
        self.kg = kg
        self.pe_kind = pe
        self.pe_dim = pe_dim if pe != "none" else 0
        self.caps = tuple(caps)
        self.cache = {}
        self.emb = None
        if use_rdf2vec:
            import os
            self.emb = torch.from_numpy(
                np.load(os.path.join(pack_dir, "emb.npy")))
            with open(os.path.join(pack_dir, "keys.txt"),
                      encoding="utf-8") as f:
                pack_idx = {k: i for i, k in enumerate(f.read().splitlines())}
            # node id -> pack row (-1: no rdf2vec, zero row used)
            rows = np.full(kg.nE + kg.nR, -1, dtype=np.int64)
            for keys, off in ((kg.ent_keys, 0), (kg.rel_keys, kg.nE)):
                for i, k in enumerate(keys):
                    r = pack_idx.get(k)
                    if r is not None:
                        rows[off + i] = r
            self.pack_rows = rows

    def _entry(self, node):
        e = self.cache.get(node)
        if e is None:
            node_ids, edge_index, role_idx = self.kg.sample_subgraph(
                node, self.caps)
            occ, ntype = self.kg.node_features(node_ids)
            ei = torch.from_numpy(edge_index)
            ri = torch.from_numpy(role_idx)
            pe = compute_pe(ei, ri, len(node_ids), self.pe_kind, self.pe_dim)
            # relation nodes: id in [nE, nE+nR); local rel index = id - nE
            is_rel = (node_ids >= self.kg.nE) & (node_ids < self.kg.nE + self.kg.nR)
            rel_local = (node_ids - self.kg.nE)
            fanout = np.zeros((len(node_ids), 2), dtype=np.float32)
            rel_id = np.zeros(len(node_ids), dtype=np.int64)  # 0 = padding
            if is_rel.any():
                ri_local = rel_local[is_rel]
                rel_id[is_rel] = ri_local + 1
                if self.kg.max_out is not None:
                    fanout[is_rel, 0] = np.log1p(self.kg.max_out[ri_local])
                    fanout[is_rel, 1] = np.log1p(self.kg.max_in[ri_local])
            emb_rows = None
            if self.emb is not None:
                er = node_ids < (self.kg.nE + self.kg.nR)
                rows = np.full(len(node_ids), -1, dtype=np.int64)
                rows[er] = self.pack_rows[node_ids[er]]
                emb_rows = torch.from_numpy(rows)
            e = self.cache[node] = (
                torch.from_numpy(occ), torch.from_numpy(fanout), pe.float(),
                emb_rows, torch.from_numpy(ntype), torch.from_numpy(rel_id),
                ei, ri, len(node_ids))
        return e

    def batch(self, node_ids, device):
        """Disjoint batch over per-term subgraphs -> kwargs for TermEncoder.
        node_ids: list of factor-graph node ids (one per term)."""
        occs, fans, pes, embs, ntypes, relids, eis, ris, bvec, centers = \
            [], [], [], [], [], [], [], [], [], []
        off = 0
        for gi, node in enumerate(node_ids):
            occ, fanout, pe, emb_rows, ntype, rel_id, ei, ri, n = \
                self._entry(node)
            occs.append(occ)
            fans.append(fanout)
            pes.append(pe)
            ntypes.append(ntype)
            relids.append(rel_id)
            eis.append(ei + off)
            ris.append(ri)
            bvec.append(torch.full((n,), gi, dtype=torch.long))
            centers.append(off)  # seed is local node 0
            if self.emb is not None:
                rows = emb_rows.clamp(min=0)
                e = self.emb[rows]
                e[emb_rows < 0] = 0.0
                embs.append(e)
            off += n
        occ = torch.cat(occs).to(device)
        n_tot = occ.shape[0]
        center = torch.zeros(n_tot, 1)
        center_pos = torch.tensor(centers, dtype=torch.long)
        center[center_pos] = 1.0
        return dict(
            occ=occ,
            fanout=torch.cat(fans).to(device),
            pe=(torch.cat(pes) if self.pe_dim else
                torch.zeros(n_tot, 0)).to(device),
            emb=(torch.cat(embs).to(device) if self.emb is not None
                 else torch.zeros(n_tot, 0, device=device)),
            center=center.to(device),
            ntype=torch.cat(ntypes).to(device),
            rel_id=torch.cat(relids).to(device),
            edge_index=torch.cat(eis, dim=1).to(device),
            role_idx=torch.cat(ris).to(device),
            batch_vec=torch.cat(bvec).to(device),
            center_pos=center_pos.to(device),
        )
