"""
Shared helpers for the dual-adjacency pipeline: derive variable sets, share
edges, and cartesian counts directly from stored sample tensors (x carries
variable ids at slot dims 0/102/204, marked by the all-ones blocks), so every
dataset source is handled uniformly without parsing triple strings.
"""

from collections import defaultdict

import numpy as np
import torch


def triple_var_sets(x, n):
    """Per-triple set of variable ids from the raw (N,307) feature matrix."""
    x_np = x.numpy() if isinstance(x, torch.Tensor) else x
    sets = [set() for _ in range(n)]
    for o in (0, 102, 204):
        isv = (x_np[:n, o + 1:o + 101] == 1).all(axis=1)
        ids = x_np[:n, o]
        for t in np.nonzero(isv)[0]:
            sets[t].add(int(round(float(ids[t]))))
    return sets


def share_edge_index(vsets):
    """Symmetric (2, Es) edge index between triples sharing >=1 variable."""
    by_id = defaultdict(list)
    for t, vs in enumerate(vsets):
        for v in vs:
            by_id[v].append(t)
    pairs = set()
    for ts in by_id.values():
        for a in ts:
            for b in ts:
                if a != b:
                    pairs.add((a, b))
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()


def plan_cartesian_count(edge_index, vsets, n):
    """#cartesian joins of the plan tree in edge_index (child -> parent)."""
    children = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    if src and all(s >= n for s in src) and not all(d >= n for d in dst):
        src, dst = dst, src
    for s, d in zip(src, dst):
        children[d].append(s)
    cover = {t: vsets[t] for t in range(n)}
    cart = 0
    pending = {j: list(cs) for j, cs in children.items()}
    while pending:
        ready = [j for j, cs in pending.items() if all(c in cover for c in cs)]
        if not ready:
            return -1
        for j in ready:
            cs = pending.pop(j)
            if len(cs) == 2 and not (cover[cs[0]] & cover[cs[1]]):
                cart += 1
            u = set()
            for c in cs:
                u |= cover[c]
            cover[j] = u
    return cart


def collate(samples, fp_dim=64, generator=None):
    """Batch (x, edge_index, share_ei, y) tuples; adds fresh normalized
    gaussian fingerprints to join nodes (x[:, -1] == 1), like training."""
    xs, eis, eshs, ys, batch = [], [], [], [], []
    off = 0
    for gi, (x, ei, esh, y) in enumerate(samples):
        xs.append(x)
        eis.append(ei + off)
        eshs.append(esh + off)
        ys.append(y)
        batch.append(torch.full((x.shape[0],), gi, dtype=torch.long))
        off += x.shape[0]
    x = torch.cat(xs).clone()
    join = x[:, -1] == 1.0
    nj = int(join.sum())
    if nj:
        fp = torch.randn(nj, fp_dim, generator=generator)
        x[join, :fp_dim] = fp / fp.norm(dim=1, keepdim=True)
    return (x, torch.cat(eis, dim=1), torch.cat(eshs, dim=1),
            torch.cat(batch), torch.tensor(ys, dtype=torch.float))
