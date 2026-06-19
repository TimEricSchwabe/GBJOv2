"""
Does the differentiable cartesianness penalty reduce cartesian-product joins?

Runs FastGBJO on wikidata path queries (config_wikidata_path params, same
model) with lambda_cartesian in a sweep, counting exact cartesian joins in
the final plan. Also unit-tests the discrete limit:
P_cart(hard plan) == 2 * #cartesian_joins + residual <= 2(n-1)exp(-gamma).

Usage:
    uv run python -m v3.core.featurize [--steps 10] [--per-size 25]
        [--sizes 5 8 11 14] [--lams 0 1 3 10] [--gamma 5]
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

from v3.core.gbjo_fast import (FastGBJO, FlatCostGNN, featurize_query, sharing_matrix,
                       cartesian_penalty, count_cartesian_joins)
from v3 import paths

MODEL = os.path.join(REPO, "models", "wikidata-log1p-plus-cartesian", "model.pt")
QUERIES = os.path.join(REPO, "data", "queries", "wikidata", "path", "path_queries.json")
EMB_DIR = os.path.join(REPO, "data", "embeddings", "wikidata")
CACHE = str(paths.CACHE / "bench_cart_cache.pt")

# config_wikidata_path optimization_params (src/evaluation_parallel.py __main__)
PATH_PARAMS = {
    "learning_rate": 3.9,
    "lambda_acyclic": 1.8,
    "lambda_triple_in": 1.5,
    "lambda_triple_out": 1.27,
    "lambda_join_in": 1.00,
    "lambda_join_out": 1.02,
    "lambda_entropy": 0.0,
    "lambda_total_penalty": 0.87,
    "lambda_left_linear": 5.9,
    "init_tau": 2.55,
    "min_tau": 0.79,
    "use_temperature_annealing": True,
    "return_best": True,
    "use_lambda_ramping": True,
    "lambda_ramp_exponent": 1.06,
    "gradient_clip_norm": 2.27,
    "use_lr_scheduling": True,
    "discrete_beam_width": 6,
}


def build_query_set(sizes, per_size, queries=QUERIES, emb_dir=EMB_DIR,
                    cache=CACHE):
    key = {"sizes": list(sizes), "per_size": per_size, "v": 1,
           **({"queries": queries} if queries != QUERIES else {})}
    if os.path.exists(cache):
        cached = torch.load(cache, weights_only=False)
        if cached.get("key") == key:
            print(f"Loaded {len(cached['items'])} featurized path queries from cache")
            return cached["items"]

    print(f"Loading {os.path.basename(queries)} ...")
    t0 = time.time()
    with open(queries) as f:
        raw = json.load(f)
    print(f"  {len(raw)} queries in {time.time()-t0:.1f}s")
    by_size = {}
    for q in raw:
        by_size.setdefault(len(q["triples"]), []).append(q)
    del raw

    print("Loading rdf2vec + counts ...")
    with open(os.path.join(emb_dir, "rdf2vec100dim.pkl"), "rb") as f:
        rdf2vec = pickle.load(f)
    with open(os.path.join(emb_dir, "counts.pkl"), "rb") as f:
        counts = pickle.load(f)

    items = []
    for size in sizes:
        pool = by_size.get(size, [])
        if not pool:
            print(f"  size {size}: no queries, skipping")
            continue
        for i, q in enumerate(pool[:per_size]):
            triples = [t[:3] for t in q["triples"]]  # lubm has a 4th "."
            rng = torch.Generator().manual_seed(10_000 * size + i)
            x = featurize_query(triples, rdf2vec, counts, rng=rng)
            items.append({"size": size, "triples": triples, "x": x,
                          "share": sharing_matrix(triples)})
    torch.save({"key": key, "items": items}, cache)
    print(f"Featurized + cached {len(items)} queries")
    return items


def order_to_adjacency(order, n):
    """Hard left-deep adjacency for a join order (base join n, chain to root)."""
    N = 2 * n - 1
    A = np.zeros((N, N), dtype=int)
    A[order[0], n] = 1
    A[order[1], n] = 1
    for m in range(2, n):
        A[order[m], n + m - 1] = 1
        A[n + m - 2, n + m - 1] = 1
    return A


def test_discrete_limit(items, gamma):
    """eq (6): P_cart on hard plans == 2*#cartesian joins, up to the residual."""
    rng = np.random.default_rng(0)
    checked, max_err = 0, 0.0
    for it in items:
        n = len(it["triples"])
        if n < 3:
            continue
        for _ in range(5):
            order = [int(v) for v in rng.permutation(n)]
            A = order_to_adjacency(order, n)
            cnt = count_cartesian_joins(A, it["triples"])
            P = float(cartesian_penalty(torch.tensor(A, dtype=torch.float32),
                                        it["share"], n, gamma))
            err = abs(P - cnt)
            bound = (n - 1) * math.exp(-gamma) + 1e-4
            assert err <= bound, f"n={n} count={cnt} P={P} err={err} bound={bound}"
            max_err = max(max_err, err)
            checked += 1
    print(f"discrete-limit test: {checked} random hard plans OK "
          f"(max |P - count| = {max_err:.4f}, within residual bound)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--per-size", type=int, default=25)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 11, 14])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0, 3.0, 10.0])
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--dual", default=None,
                    help="path to a CostGNNDual model.pt (default: old CostGNNv3)")
    ap.add_argument("--queries", default=QUERIES)
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    torch.set_num_threads(1)
    items = build_query_set(args.sizes, args.per_size, args.queries,
                            args.emb_dir, args.cache)
    test_discrete_limit(items, args.gamma)

    if args.dual:
        from v3.core.model_dual import FlatCostGNNDual
        flat = FlatCostGNNDual.load(args.dual)
    else:
        flat = FlatCostGNN.load(MODEL)
    results = {}  # lam -> list of per-query dicts
    for lam in args.lams:
        gbjo = FastGBJO(flat, params={**PATH_PARAMS,
                                      "lambda_cartesian": lam,
                                      "cartesian_gamma": args.gamma})
        gbjo.optimize(items[0]["x"], optimization_steps=2,
                      share=items[0]["share"])  # warmup
        rows = []
        for it in items:
            t0 = time.perf_counter()
            A, cost = gbjo.optimize(it["x"], optimization_steps=args.steps,
                                    share=it["share"])
            dt = time.perf_counter() - t0
            cart = count_cartesian_joins(A, it["triples"])
            cands = gbjo.last_candidates
            cand_carts = [count_cartesian_joins(c, it["triples"]) for c in cands]
            # penalty-consistent selection: fewest cartesian joins, then cost
            with torch.no_grad():
                h0 = flat.project_x(it["x"])
                cand_costs = [flat.forward_from_h0(
                    h0, torch.tensor(c, dtype=torch.float32)).item()
                    for c in cands]
            i_lex = min(range(len(cands)),
                        key=lambda i: (cand_carts[i], cand_costs[i]))
            rows.append({"size": it["size"], "cart": cart, "cost": cost,
                         "s": dt, "n_cand": len(cand_carts),
                         "n_cand_free": sum(c == 0 for c in cand_carts),
                         "cart_lex": cand_carts[i_lex],
                         "cost_lex": float(np.exp(cand_costs[i_lex]))})
        results[lam] = rows
        tot = sum(r["cart"] for r in rows)
        free = sum(r["cart"] == 0 for r in rows)
        print(f"lam_cart={lam:>5}: total cartesian joins {tot:>4}, "
              f"cartesian-free plans {free}/{len(rows)}")

    base = results[args.lams[0]]
    print(f"\n=== SUMMARY (steps={args.steps}, gamma={args.gamma}, "
          f"per-size={args.per_size}) ===")
    hdr = f"{'size':>4} {'lam':>5} | {'cart':>5} {'%free':>6} {'cost':>7} | " \
          f"{'lex cart':>8} {'lex%free':>8} {'lex cost':>8} | " \
          f"{'cand free%':>10} {'med ms':>7}"
    print(hdr)
    for s in args.sizes:
        for lam in args.lams:
            rs = [r for r in results[lam] if r["size"] == s]
            bs = [r for r in base if r["size"] == s]
            if not rs:
                continue
            mean_cart = np.mean([r["cart"] for r in rs])
            pfree = 100.0 * np.mean([r["cart"] == 0 for r in rs])
            logr = np.mean([math.log(r["cost"] / b["cost"])
                            for r, b in zip(rs, bs)])
            mean_cart_lex = np.mean([r["cart_lex"] for r in rs])
            pfree_lex = 100.0 * np.mean([r["cart_lex"] == 0 for r in rs])
            logr_lex = np.mean([math.log(r["cost_lex"] / b["cost"])
                                for r, b in zip(rs, bs)])
            med_ms = 1e3 * np.median([r["s"] for r in rs])
            cfree = 100.0 * (sum(r["n_cand_free"] for r in rs) /
                             max(1, sum(r["n_cand"] for r in rs)))
            print(f"{s:>4} {lam:>5} | {mean_cart:>5.2f} {pfree:>5.0f}% "
                  f"{math.exp(logr):>6.2f}x | {mean_cart_lex:>8.2f} "
                  f"{pfree_lex:>7.0f}% {math.exp(logr_lex):>7.2f}x | "
                  f"{cfree:>9.0f}% {med_ms:>7.1f}")
        print()


if __name__ == "__main__":
    main()
