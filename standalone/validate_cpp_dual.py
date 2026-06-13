"""
C++ kernel vs Python FastGBJO equivalence for the dual (CostGNNDual) model.

For each cached eval path query, runs both implementations with identical
params/steps and compares:
  - the discrete plan adjacency (exact match expected, modulo last-ulp
    drift in the chaotic optimization loop)
  - the predicted cost of the C++ winner, recomputed with the Python
    forward (validates the GNN numerics directly)

Usage:
    uv run python standalone/validate_cpp_dual.py \
        --model standalone/models/dual-v2/model_rank.pt
    uv run python standalone/validate_cpp_dual.py \
        --model standalone/models/lubm-dual-v1/model_rank.pt \
        --queries data/queries/lubm/paths/path_queries.json \
        --emb-dir data/embeddings/lubm \
        --cache standalone/bench_cart_cache_lubm.pt --sizes 3 4 5
"""

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_cartesian import PATH_PARAMS, build_query_set
import bench_cartesian as bc
from gbjo_fast import FastGBJO
from model_dual import FlatCostGNNDual
from gbjo_cpp import CppGBJO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="standalone/models/dual-v2/model_rank.pt")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--per-size", type=int, default=25)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 11, 14])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--emb-dir", default=None)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    items = build_query_set(args.sizes, args.per_size,
                            queries=args.queries or bc.QUERIES,
                            emb_dir=args.emb_dir or bc.EMB_DIR,
                            cache=args.cache or bc.CACHE)

    flat = FlatCostGNNDual.load(args.model)
    fast = FastGBJO(flat, params=PATH_PARAMS)
    cpp = CppGBJO(flat, params=PATH_PARAMS)

    # warmup
    fast.optimize(items[0]["x"], optimization_steps=2, share=items[0]["share"])
    cpp.optimize(items[0]["x"], optimization_steps=2, share=items[0]["share"])

    rows = []
    for it in items:
        x, share = it["x"], it["share"]
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            A_py, cost_py = fast.optimize(x, optimization_steps=args.steps,
                                          share=share)
            ts.append(time.perf_counter() - t0)
        t_py = float(np.median(ts))
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            A_cpp, cost_cpp = cpp.optimize(x, optimization_steps=args.steps,
                                           share=share)
            ts.append(time.perf_counter() - t0)
        t_cpp = float(np.median(ts))

        # recompute the C++ winner's cost with the Python forward
        flat.bind_share((share > 0).float())
        with torch.no_grad():
            h0 = flat.project_x(x)
            ref = float(np.exp(flat.forward_from_h0(
                h0, torch.tensor(A_cpp, dtype=torch.float32)).item()))

        rows.append({
            "size": it["size"],
            "adj_match": bool(np.array_equal(A_py, A_cpp)),
            "cost_rel": abs(cost_cpp - cost_py) / max(cost_py, 1e-12),
            "fwd_rel": abs(cost_cpp - ref) / max(ref, 1e-12),
            "t_py": t_py, "t_cpp": t_cpp,
        })

    print(f"\n=== C++ vs Python (dual model, steps={args.steps}) ===")
    print(f"{'size':>4} {'plans==':>8} {'max fwd rel':>12} {'max cost rel':>13} "
          f"{'py ms':>8} {'cpp ms':>8} {'speedup':>8}")
    for s in sorted({r["size"] for r in rows}):
        rs = [r for r in rows if r["size"] == s]
        m = sum(r["adj_match"] for r in rs)
        t_py = np.median([r["t_py"] for r in rs])
        t_cpp = np.median([r["t_cpp"] for r in rs])
        print(f"{s:>4} {m:>5}/{len(rs)} {max(r['fwd_rel'] for r in rs):>12.2e} "
              f"{max(r['cost_rel'] for r in rs):>13.2e} "
              f"{1e3*t_py:>8.1f} {1e3*t_cpp:>8.1f} {t_py/t_cpp:>7.1f}x")
    total = len(rows)
    match = sum(r["adj_match"] for r in rows)
    print(f"\nplans identical: {match}/{total}")
    for r in rows:
        if not r["adj_match"]:
            print(f"  mismatch size={r['size']}: cost_rel={r['cost_rel']:.2e}")


if __name__ == "__main__":
    main()
