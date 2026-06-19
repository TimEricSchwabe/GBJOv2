"""Shared helpers for per-join intermediate-cardinality (subcost) supervision.

A left-deep plan's C_out is the sum over its join nodes of the cardinality of
that join's partial result (the triples in the join's subtree). Those per-join
cardinalities are NOT stored in the datasets (only the total y) but are
recoverable by component-factorized COUNT:

    card(triple set) = product over connected components of the component COUNT.

Counts are keyed in a shared cache by sha1 of the component's sorted patterns
(mirroring the MRT CStarOracle), so identical sub-patterns are counted once.
`mine_subcosts.py` warms the cache against QLever; the trainer reads it offline.
Node convention (order_to_adjacency): triples are nodes 0..n-1, joins n..2n-2;
join node n+m holds the partial result of the first m+2 triples in plan order.
"""

import hashlib


def parse_atoms(tr_str):
    """'<s> <p> ?o.' -> ('<s>', '<p>', '?o') (matches train_dual.bound_atoms)."""
    s, p, o = tr_str.split(" ", 2)
    return (s, p, o.rstrip(" ."))


def var_set(atoms):
    return set(a for a in atoms if a.startswith("?"))


def var_sets(triples_atoms):
    return [var_set(t) for t in triples_atoms]


def components(idxs, vs):
    """Connected components of the variable-sharing graph restricted to idxs."""
    idxs = list(idxs)
    seen, comps = set(), []
    for start in idxs:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in idxs:
                if v not in seen and (vs[u] & vs[v]):
                    seen.add(v)
                    stack.append(v)
        comps.append(frozenset(comp))
    return comps


def comp_key(comp, triples_atoms):
    """Stable cache key for a connected component: sha1 of its sorted patterns."""
    pats = sorted(" ".join(triples_atoms[i]) for i in comp)
    return hashlib.sha1(" . ".join(pats).encode()).hexdigest()


def join_subtrees(edge_index, n):
    """For each join node j in [n, 2n-2] return (leaf_idxs, node_idxs):
    its descendant triple leaves (< n) and ALL its subtree nodes incl. j.
    edge_index rows are [child, parent] (order_to_adjacency / argwhere(A))."""
    N = 2 * n - 1
    child = edge_index[0].tolist()
    parent = edge_index[1].tolist()
    kids = {}
    for c, p in zip(child, parent):
        kids.setdefault(p, []).append(c)
    out = {}
    for j in range(n, N):
        leaves, nodes, stack = [], [j], [j]
        while stack:
            u = stack.pop()
            for c in kids.get(u, ()):
                nodes.append(c)
                if c < n:
                    leaves.append(c)
                else:
                    stack.append(c)
        out[j] = (leaves, nodes)
    return out


def card_from_cache(leaf_idxs, triples_atoms, vs, cache):
    """Product over connected components of cache[comp_key]; None if any
    component is missing or censored (cache value < 0)."""
    prod = 1
    for comp in components(leaf_idxs, vs):
        c = cache.get(comp_key(comp, triples_atoms))
        if c is None or c < 0:
            return None
        prod *= c
    return prod
