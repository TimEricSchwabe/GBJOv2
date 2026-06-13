"""Does the cost model itself prefer cartesian plans on path queries?

For each cached path query, compare the model's predicted cost of:
  - the contiguous path order (cartesian-free by construction)
  - its reverse
  - the plan GBJO (lambda_cartesian=0) actually selects
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbjo_fast import (FastGBJO, FlatCostGNN, count_cartesian_joins)
from bench_cartesian import PATH_PARAMS, MODEL, build_query_set, order_to_adjacency

items = build_query_set([5, 8, 11, 14], 25)
flat = FlatCostGNN.load(MODEL)
gbjo = FastGBJO(flat, params=dict(PATH_PARAMS))


def model_cost(h0, A):
    with torch.no_grad():
        return flat.forward_from_h0(h0, torch.tensor(A, dtype=torch.float32)).item()


rows = []
for it in items:
    triples = it["triples"]
    n = len(triples)
    with torch.no_grad():
        h0 = flat.project_x(it["x"])

    # connected (cartesian-free) orders via greedy expansion in the share graph
    S = it["share"].numpy()[:n, :n]

    def connected_order(start):
        order, left = [start], set(range(n)) - {start}
        while left:
            nxt = next((t for t in sorted(left)
                        if any(S[t, u] > 0 for u in order)), None)
            if nxt is None:
                return None  # share graph disconnected
            order.append(nxt)
            left.remove(nxt)
        return order

    fwd = connected_order(0)
    rev = connected_order(n - 1)
    if fwd is None or rev is None:
        continue
    A_fwd = order_to_adjacency(fwd, n)
    A_rev = order_to_adjacency(rev, n)
    assert count_cartesian_joins(A_fwd, triples) == 0

    A_gbjo, _ = gbjo.optimize(it["x"], optimization_steps=10, share=None)
    rows.append({
        "size": n,
        "cart_gbjo": count_cartesian_joins(A_gbjo, triples),
        "c_fwd": model_cost(h0, A_fwd),
        "c_rev": model_cost(h0, A_rev),
        "c_gbjo": model_cost(h0, A_gbjo),
    })

print(f"{'size':>4} {'gbjo cart':>9} {'gbjo<fwd':>9} {'gbjo<rev':>9} "
      f"{'med log10(fwd/gbjo)':>20}")
for s in sorted({r["size"] for r in rows}):
    rs = [r for r in rows if r["size"] == s and r["cart_gbjo"] > 0]
    if not rs:
        print(f"{s:>4}  (no cartesian gbjo plans)")
        continue
    lt_fwd = sum(r["c_gbjo"] < r["c_fwd"] for r in rs)
    lt_rev = sum(r["c_gbjo"] < r["c_rev"] for r in rs)
    med = np.median([(r["c_fwd"] - r["c_gbjo"]) / np.log(10) for r in rs])
    print(f"{s:>4} {len(rs):>6}/25 {lt_fwd:>6}/{len(rs)} {lt_rev:>6}/{len(rs)} "
          f"{med:>20.2f}")
