"""
Inspect GBJO cost-model training datasets: sizes, cost stats, and -- crucially
-- how many plans contain cartesian-product joins.

Reconstruction per sample (no metadata needed):
  - n = (N+1)//2 triples; var slots in x identified by the all-ones marker
    block (dims o+1..o+100 == 1 for o in 0/102/204), var id = x[t, o]
  - plan tree from edge_index (child -> parent into join nodes)
  - cartesian join = its two children's subtree variable sets are disjoint

Usage:
    uv run python standalone/inspect_datasets.py <dataset_dir> [...]
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch


def var_sets(x, n):
    """Per-triple set of variable ids, from the raw feature matrix."""
    out = []
    for t in range(n):
        s = set()
        for o in (0, 102, 204):
            block = x[t, o + 1:o + 101]
            if bool((block == 1).all()):
                s.add(round(float(x[t, o])))
        out.append(s)
    return out


def cartesian_count(x, edge_index, n):
    """Number of cartesian joins in the plan encoded by edge_index."""
    N = 2 * n - 1
    vs = var_sets(x, n)
    children = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    # direction: parents are join nodes (>= n); flip if needed
    if all(s >= n for s in src) and not all(d >= n for d in dst):
        src, dst = dst, src
    for s, d in zip(src, dst):
        children[d].append(s)

    cover = {t: vs[t] for t in range(n)}
    cart = 0
    # joins may reference other joins; resolve bottom-up
    pending = {j: list(cs) for j, cs in children.items()}
    while pending:
        ready = [j for j, cs in pending.items() if all(c in cover for c in cs)]
        if not ready:
            return -1  # malformed tree
        for j in ready:
            cs = pending.pop(j)
            if len(cs) == 2:
                a, b = cover[cs[0]], cover[cs[1]]
                if not (a & b):
                    cart += 1
                cover[j] = a | b
            else:
                u = set()
                for c in cs:
                    u |= cover[c]
                cover[j] = u
    return cart


def inspect(path, max_samples=None):
    print(f"\n=== {path} ===")
    d = torch.load(os.path.join(path, "dataset.pt"), weights_only=False)
    data = d["data"]
    print(f"keys: {list(d.keys())}, samples: {len(data)}, "
          f"triples-entries: {len(d.get('triples', []))}")

    idx = range(len(data))
    if max_samples and len(data) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data), max_samples, replace=False)
        print(f"(sampling {max_samples})")

    sizes = Counter()
    carts = Counter()          # n -> #plans with >=1 cartesian join
    cart_dist = Counter()      # #cartesian joins -> count
    logy_cart, logy_free = defaultdict(list), defaultdict(list)
    bad_y = malformed = 0
    for i in idx:
        s = data[i]
        N = s.x.shape[0]
        n = (N + 1) // 2
        sizes[n] += 1
        y = float(s.y.item()) if s.y is not None else float("nan")
        if not np.isfinite(y) or y <= 0:
            bad_y += 1
            continue
        c = cartesian_count(s.x, s.edge_index, n)
        if c < 0:
            malformed += 1
            continue
        cart_dist[c] += 1
        if c > 0:
            carts[n] += 1
            logy_cart[n].append(np.log10(y))
        else:
            logy_free[n].append(np.log10(y))

    print(f"sizes: {dict(sorted(sizes.items()))}")
    print(f"bad y: {bad_y}, malformed trees: {malformed}")
    print(f"#cartesian-joins distribution: {dict(sorted(cart_dist.items()))}")
    total = sum(cart_dist.values())
    with_cart = total - cart_dist.get(0, 0)
    print(f"plans with >=1 cartesian join: {with_cart}/{total} "
          f"({100.0*with_cart/max(total,1):.1f}%)")
    print(f"{'n':>3} {'%cart':>6} {'med log10(y) free':>18} {'cart':>6}")
    for n in sorted(sizes):
        nf, nc = len(logy_free[n]), len(logy_cart[n])
        if nf + nc == 0:
            continue
        mf = np.median(logy_free[n]) if nf else float("nan")
        mc = np.median(logy_cart[n]) if nc else float("nan")
        print(f"{n:>3} {100.0*nc/(nf+nc):>5.1f}% {mf:>18.2f} {mc:>6.2f}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or [
        "data/plans/wikidata_path_plan_datasets_training/new2",
        "data/plans/wikidata_path_plan_datasets_training/new-combined",
        "data/plans/wikidata_path_plan_datasets_training/new3",
    ]
    for p in dirs:
        try:
            inspect(p, max_samples=60_000)
        except Exception as e:
            print(f"FAILED {p}: {e}")
