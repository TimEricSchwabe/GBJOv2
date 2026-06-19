"""
True C_out costs (via QLever count queries) for GBJO plans vs cartesian-free
alternatives on path queries.

Per query, evaluates:
  gbjo : plan selected by GBJO (lambda_cartesian=0, cost-only selection)
  lex  : penalty-consistent selection (fewest cartesian joins, then cost)
         over the same candidate pool
  conn : greedy connected order (cartesian-free by construction)

C_out(left-deep order) = sum over prefixes k=2..n of card(join of first k
triples). Counts are cached by prefix *set* (order-independent) and a timeout
censors the plan's cost as a lower bound (>= partial sum).

Usage:
    uv run python -m v3.eval.true_cost --sizes 5 8 --per-size 2 \
        [--endpoint http://127.0.0.1:7020/] [--timeout 30]
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import requests
import torch

from v3.core.gbjo_fast import (FastGBJO, FlatCostGNN, adjacency_to_join_order,
                       count_cartesian_joins)
from v3.core.featurize import PATH_PARAMS, MODEL, build_query_set


def make_counter(endpoint, timeout):
    session = requests.Session()
    cache = {}

    def count(triples, idx_set):
        key = frozenset(idx_set)
        if key in cache:
            return cache[key]
        body = " . ".join(" ".join(triples[i]) for i in sorted(idx_set)) + " ."
        q = f"SELECT (COUNT(*) AS ?count) WHERE {{ {body} }}"
        t0 = time.time()
        try:
            r = session.get(endpoint, params={"query": q},
                            headers={"Accept": "application/sparql-results+json"},
                            timeout=(5.0, timeout))
            r.raise_for_status()
            val = int(r.json()["results"]["bindings"][0]["count"]["value"])
        except requests.exceptions.Timeout:
            val = None
        except Exception as e:
            print(f"    ! endpoint error ({e}) -- treating as censored")
            val = None
        cache[key] = val
        if time.time() - t0 > 2:
            print(f"    (slow count: {time.time()-t0:.0f}s, "
                  f"{len(idx_set)} triples -> {val})")
        return val

    return count


def c_out(order, triples, count):
    """(cost, censored): sum of prefix cardinalities; censored=True if some
    prefix timed out (cost is then a lower bound)."""
    total, censored = 0, False
    prefix = set(order[:1])
    for t in order[1:]:
        prefix.add(t)
        c = count(triples, prefix)
        if c is None:
            return total, True
        total += c
    return total, censored


def connected_order(triples, S):
    n = len(triples)
    n_const = [sum(not a.startswith("?") for a in t[:3]) for t in triples]
    start = max(range(n), key=lambda i: n_const[i])
    order, left = [start], set(range(n)) - {start}
    while left:
        nxt = next((t for t in sorted(left)
                    if any(S[t, u] > 0 for u in order)), None)
        if nxt is None:
            return None
        order.append(nxt)
        left.remove(nxt)
    return order


def fmt(cost, censored):
    s = f"{cost:.3g}"
    return (">=" + s) if censored else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 8])
    ap.add_argument("--per-size", type=int, default=2)
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--dual", default=None, help="path to CostGNNDual model.pt")
    ap.add_argument("--lam", type=float, default=0.0, help="lambda_cartesian")
    ap.add_argument("--cache-sizes", type=int, nargs="+", default=[5, 8, 11, 14],
                    help="sizes the query cache is built with (superset of --sizes)")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--emb-dir", default=None)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    import featurize as bc
    items = build_query_set(args.cache_sizes, 25,
                            queries=args.queries or bc.QUERIES,
                            emb_dir=args.emb_dir or bc.EMB_DIR,
                            cache=args.cache or bc.CACHE)
    items = [it for s in args.sizes
             for it in [x for x in items if x["size"] == s][:args.per_size]]

    if args.dual:
        from v3.core.model_dual import FlatCostGNNDual
        flat = FlatCostGNNDual.load(args.dual)
    else:
        flat = FlatCostGNN.load(MODEL)
    gbjo = FastGBJO(flat, params={**PATH_PARAMS, "lambda_cartesian": args.lam})
    count = make_counter(args.endpoint, args.timeout)

    wins = {"gbjo": 0, "free": 0, "tie": 0, "unknown": 0}
    wins_pick = {}
    rows = []
    for qi, it in enumerate(items):
        triples, n = it["triples"], it["size"]
        with torch.no_grad():
            h0 = flat.project_x(it["x"])

        def pred(A):
            with torch.no_grad():
                return float(np.exp(flat.forward_from_h0(
                    h0, torch.tensor(A, dtype=torch.float32)).item()))

        A_gbjo, _ = gbjo.optimize(it["x"], optimization_steps=args.steps,
                                  share=it["share"])
        cands = gbjo.last_candidates
        carts = [count_cartesian_joins(c, triples) for c in cands]
        with torch.no_grad():
            costs = [flat.forward_from_h0(
                h0, torch.tensor(c, dtype=torch.float32)).item() for c in cands]
        A_lex = cands[min(range(len(cands)), key=lambda i: (carts[i], costs[i]))]
        conn = connected_order(triples, it["share"].numpy()[:n, :n])

        plans = {"gbjo": adjacency_to_join_order(A_gbjo),
                 "lex": adjacency_to_join_order(A_lex)}
        if conn is not None:
            plans["conn"] = conn
        preds = {"gbjo": pred(A_gbjo), "lex": pred(A_lex)}
        # pick = conn-candidate injection: the model chooses between the
        # GBJO plan and the greedy connected plan
        if conn is not None:
            from v3.core.featurize import order_to_adjacency as o2a
            preds["conn"] = pred(o2a(conn, n))
            pick_src = "conn" if preds["conn"] < preds["gbjo"] else "gbjo"
        else:
            pick_src = "gbjo"
        plans["pick"] = plans[pick_src]
        preds["pick"] = preds[pick_src]

        print(f"\nquery {qi} (n={n}):")
        res, plan_carts = {}, {}
        for name, order in plans.items():
            from v3.core.featurize import order_to_adjacency
            cart = count_cartesian_joins(order_to_adjacency(order, n), triples)
            plan_carts[name] = cart
            t0 = time.time()
            cost, cens = c_out(order, triples, count)
            res[name] = (cost, cens)
            p = preds.get(name)
            print(f"  {name:>4}: cart={cart}  true C_out={fmt(cost, cens):>12}  "
                  f"pred={p and f'{p:.3g}' or '-':>10}  "
                  f"({time.time()-t0:.1f}s)  order={order}")

        # who actually wins: the plan's cartesian choice vs best exact
        # cartesian-free alternative
        def verdict_for(name):
            c, cens = res[name]
            exact_free = [cf for nm, (cf, ce) in res.items()
                          if nm != name and plan_carts[nm] == 0 and not ce]
            if plan_carts[name] == 0:
                return f"{name}-free"  # the plan itself has no cartesian
            if not exact_free:
                return "unknown"
            best_free = min(exact_free)
            if not cens:
                return ("free" if best_free < c else
                        name if c < best_free else "tie")
            return "free" if best_free <= c else "unknown"

        verdict = verdict_for("gbjo")
        verdict_pick = verdict_for("pick")
        wins.setdefault(verdict, 0)
        wins[verdict] += 1
        wins_pick.setdefault(verdict_pick, 0)
        wins_pick[verdict_pick] += 1
        rows.append({"n": n, "verdict": verdict, "verdict_pick": verdict_pick,
                     **{k: v for k, v in res.items()}})
        print(f"  -> truly cheaper: gbjo: {verdict}   pick: {verdict_pick}")

    print(f"\n=== VERDICTS (plan's cartesian choice vs cartesian-free) ===")
    print(f"gbjo: {wins}")
    print(f"pick: {wins_pick}")


if __name__ == "__main__":
    main()
