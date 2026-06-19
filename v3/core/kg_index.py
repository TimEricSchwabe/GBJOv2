"""
Factor-graph index of an N-Triples KG for the FICE-style term encoder.

Factor graph (FICE paper, sec. 3.2): nodes are entities [0, nE), relations
[nE, nE+nR) and triple nodes [nE+nR, nE+nR+nT). Each triple (s, p, o) adds
forward edges t->s/t->p/t->o with roles 1/2/3 and reverses s->t/p->t/o->t
with roles -1/-2/-3, stored as one CSR over all directed edges.

Build (one-time, caches to <out>/kg_index.npz + ent_keys.txt + rel_keys.txt):
    uv run python -m v3.core.kg_index \
        --nt /Users/timschwabe/Projects/qlever/wikidata/graph.nt \
        --out v3/artifacts/index/wikidata

KGIndex.load(out_dir) mmaps the arrays; sample_subgraph(node, caps) draws a
fixed-per-node (seeded by node id) capped k-hop neighborhood, mirroring
FICE's GraphSAGE-style sampling but per seed term, so a term's subgraph --
and hence its embedding -- is independent of batch composition.
"""

import argparse
import os
import time

import numpy as np

ROLE_TO_IDX = {1: 0, 2: 1, 3: 2, -1: 3, -2: 4, -3: 5}


def parse_nt(path):
    """-> (ent_keys, rel_keys, s_ids, p_ids, o_ids); terms keep their raw
    token form ('<uri>' / literal), matching the query/pack key format."""
    ent, rel = {}, {}
    s_ids, p_ids, o_ids = [], [], []
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            try:
                s, rest = line.split(" ", 1)
                p, o = rest.split(" ", 1)
                o = o.rstrip()
                if o.endswith(" ."):
                    o = o[:-2]
                elif o.endswith("."):
                    o = o[:-1].rstrip()
            except ValueError:
                continue
            s_ids.append(ent.setdefault(s, len(ent)))
            p_ids.append(rel.setdefault(p, len(rel)))
            o_ids.append(ent.setdefault(o, len(ent)))
            if (ln + 1) % 5_000_000 == 0:
                print(f"  {ln+1:,} lines ({time.time()-t0:.0f}s)", flush=True)
    print(f"parsed {len(s_ids):,} triples, {len(ent):,} entities, "
          f"{len(rel):,} relations ({time.time()-t0:.0f}s)")
    keys_e = list(ent.keys())
    keys_r = list(rel.keys())
    return (keys_e, keys_r,
            np.array(s_ids, dtype=np.int64),
            np.array(p_ids, dtype=np.int64),
            np.array(o_ids, dtype=np.int64))


def build(nt_path, out_dir):
    keys_e, keys_r, s, p, o = parse_nt(nt_path)
    nE, nR, nT = len(keys_e), len(keys_r), len(s)
    N = nE + nR + nT
    t_ids = nE + nR + np.arange(nT, dtype=np.int64)
    p_g = p + nE  # relation node ids

    t0 = time.time()
    src = np.concatenate([t_ids, t_ids, t_ids, s, p_g, o])
    dst = np.concatenate([s, p_g, o, t_ids, t_ids, t_ids])
    roles = np.concatenate([np.full(nT, r, dtype=np.int8)
                            for r in (1, 2, 3, -1, -2, -3)])
    order = np.argsort(src, kind="stable")
    indices = dst[order].astype(np.int32)
    roles = roles[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=indptr[1:])
    del src, dst, order, t_ids, p_g
    print(f"CSR built: {N:,} nodes, {len(indices):,} directed edges "
          f"({time.time()-t0:.0f}s)")

    # occurrence counts (o_s, o_p, o_o) for entities and relations
    occ = np.zeros((nE + nR, 3), dtype=np.float32)
    occ[:nE, 0] = np.bincount(s, minlength=nE)
    occ[nE:, 1] = np.bincount(p, minlength=nR)
    occ[:nE, 2] = np.bincount(o, minlength=nE)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "kg_index.npz"),
             indptr=indptr, indices=indices, roles=roles, occ=occ,
             sizes=np.array([nE, nR, nT], dtype=np.int64))
    for name, keys in (("ent_keys.txt", keys_e), ("rel_keys.txt", keys_r)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(keys))
    print(f"saved index to {out_dir}")


class KGIndex:
    def __init__(self, indptr, indices, roles, occ, sizes, ent_keys, rel_keys,
                 max_out=None, max_in=None):
        self.indptr = indptr
        self.indices = indices
        self.roles = roles
        self.occ = occ  # (nE+nR, 3) occurrence counts
        self.nE, self.nR, self.nT = (int(x) for x in sizes)
        self.ent_keys = ent_keys
        self.rel_keys = rel_keys
        self.ent_idx = {k: i for i, k in enumerate(ent_keys)}
        self.rel_idx = {k: self.nE + i for i, k in enumerate(rel_keys)}
        # per-relation worst-case fan-out bounds (pred_stats.npz), aligned to
        # rel_keys; None if pred_stats was not built for this index.
        self.max_out = max_out
        self.max_in = max_in

    @classmethod
    def load(cls, out_dir):
        z = np.load(os.path.join(out_dir, "kg_index.npz"), mmap_mode="r")
        with open(os.path.join(out_dir, "ent_keys.txt"), encoding="utf-8") as f:
            ent_keys = f.read().splitlines()
        with open(os.path.join(out_dir, "rel_keys.txt"), encoding="utf-8") as f:
            rel_keys = f.read().splitlines()
        max_out = max_in = None
        ps_path = os.path.join(out_dir, "pred_stats.npz")
        if os.path.exists(ps_path):
            ps = np.load(ps_path)
            max_out = ps["max_out"].astype(np.float32)
            max_in = ps["max_in"].astype(np.float32)
        return cls(z["indptr"], z["indices"], z["roles"], z["occ"],
                   z["sizes"], ent_keys, rel_keys, max_out, max_in)

    def node_id(self, term, kind):
        """kind: 'ent' (subject/object slot) or 'rel' (predicate slot)."""
        return (self.rel_idx if kind == "rel" else self.ent_idx).get(term, -1)

    def sample_subgraph(self, node, caps):
        """Capped k-hop neighborhood around `node` (local id 0). Sampling is
        seeded by the node id, so the subgraph is fixed across calls/runs.
        -> (node_ids (n,) int64 global, edge_index (2, E) int64 local,
            role_idx (E,) int64 in [0, 6))."""
        rng = np.random.default_rng(node)
        indptr, indices, roles = self.indptr, self.indices, self.roles
        pos = {node: 0}
        nodes = [node]
        e_src, e_dst, e_role = [], [], []
        frontier = [node]
        for cap in caps:
            nxt = []
            for u in frontier:
                lo, hi = int(indptr[u]), int(indptr[u + 1])
                deg = hi - lo
                if deg <= cap:
                    sel = range(lo, hi)
                else:
                    sel = (rng.choice(deg, size=cap, replace=False) + lo)
                ul = pos[u]
                for e in sel:
                    v = int(indices[e])
                    r = int(roles[e])
                    vl = pos.get(v)
                    if vl is None:
                        vl = pos[v] = len(nodes)
                        nodes.append(v)
                        nxt.append(v)
                    # stored edge u->v (role r) plus its reverse v->u (-r),
                    # so messages flow both ways on the sampled structure
                    e_src.append(ul); e_dst.append(vl); e_role.append(ROLE_TO_IDX[r])
                    e_src.append(vl); e_dst.append(ul); e_role.append(ROLE_TO_IDX[-r])
            frontier = nxt
        edge_index = np.array([e_src, e_dst], dtype=np.int64)
        # dedup (multiple paths can sample the same edge)
        if edge_index.shape[1]:
            key = (edge_index[0] * len(nodes) + edge_index[1]) * 8 + np.array(e_role)
            _, keep = np.unique(key, return_index=True)
            edge_index = edge_index[:, keep]
            role_idx = np.array(e_role, dtype=np.int64)[keep]
        else:
            role_idx = np.zeros(0, dtype=np.int64)
        return np.array(nodes, dtype=np.int64), edge_index, role_idx

    def node_features(self, node_ids):
        """(n, 3) log1p occurrence counts; triple nodes get zeros, plus
        (n,) node type (0=entity, 1=relation, 2=triple)."""
        n = len(node_ids)
        occ = np.zeros((n, 3), dtype=np.float32)
        er = node_ids < (self.nE + self.nR)
        occ[er] = self.occ[node_ids[er]]
        ntype = np.zeros(n, dtype=np.int64)
        ntype[node_ids >= self.nE] = 1
        ntype[node_ids >= self.nE + self.nR] = 2
        return np.log1p(occ), ntype


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.nt, args.out)
