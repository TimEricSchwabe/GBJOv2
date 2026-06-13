"""
Generate FICE-style sibling plan families for path queries, with exact C_out
costs via component factorization (no cartesian is ever executed).

Per query (greedy connected base order):
  base      : 1 greedy connected order            (0 cartesian)
  siblings  : at sampled depths k, swap in an alternative connected next
              triple and complete greedily        (0 cartesian, shares
              prefix base[:k] with the base -> very related plans)
  cart      : at sampled positions k, insert a NON-sharing triple and
              complete greedily                   (exactly 1 cartesian at
              join k; the base is its cartesian-free twin)
  random    : 2 random permutations               (usually multi-cartesian)

Plans of the same query form a ranking group in training (recovered by the
identical 'triples' strings). Output format matches gen_cartesian_plans.py:
<out>/dataset.pt with {'dataset_size', 'triples', 'data'}, x WITHOUT
fingerprints, y = exact C_out.

Usage:
    uv run python standalone/gen_sibling_plans.py --per-size 150 \
        --sizes 5 6 7 8 9 10 11 12 13 14 15 --out standalone/sib_plans
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbjo_fast import featurize_query
from bench_cartesian import order_to_adjacency
from gen_cartesian_plans import var_sets, components, count_cart, CostOracle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = os.path.join(REPO, "data", "queries", "wikidata", "path", "path_queries.json")
EMB_DIR = os.path.join(REPO, "data", "embeddings", "wikidata")


def greedy_complete(prefix, vs, n, rng):
    """Extend prefix to a full order, picking a var-sharing triple whenever
    one exists. If none does (disconnected share graph, e.g. chains linked
    only by a constant), start the next component -- one unavoidable
    cartesian join."""
    order = list(prefix)
    placed = set(order)
    cur = set()
    for t in order:
        cur |= vs[t]
    while len(order) < n:
        cands = [t for t in range(n) if t not in placed and vs[t] & cur]
        if not cands:
            cands = [t for t in range(n) if t not in placed]
        t = rng.choice(sorted(cands))
        order.append(t)
        placed.add(t)
        cur |= vs[t]
    return order


def make_orders(triples, rng, max_sib, alts, cart_variants):
    """-> list of orders for one query (deduplicated). For queries whose
    share graph is disconnected, min_cart > 0 cartesians are unavoidable;
    'free' siblings then mean count_cart == min_cart and the matched
    cartesian variants add exactly one more."""
    n = len(triples)
    vs = var_sets(triples)
    min_cart = len(components(range(n), vs)) - 1
    base = greedy_complete([rng.randrange(n)], vs, n, rng)
    orders = [tuple(base)]
    seen = {tuple(base)}

    # sibling variants: alternative connected choice at depth k, then greedy
    depths = list(range(1, n))
    rng.shuffle(depths)
    for k in depths:
        if len(orders) - 1 >= max_sib:
            break
        prefix = base[:k]
        cur = set()
        for t in prefix:
            cur |= vs[t]
        cands = [t for t in range(n)
                 if t not in prefix and t != base[k] and vs[t] & cur]
        rng.shuffle(cands)
        for t in cands[:alts]:
            order = greedy_complete(prefix + [t], vs, n, rng)
            if tuple(order) in seen:
                continue
            if count_cart(order, vs) != min_cart:
                continue
            orders.append(tuple(order))
            seen.add(tuple(order))

    # matched single-cartesian variants at varying positions
    ks = []
    for k in range(1, n):
        prefix = base[:k]
        cur = set()
        for t in prefix:
            cur |= vs[t]
        if any(t not in prefix and not (vs[t] & cur) for t in range(n)):
            ks.append(k)
    for k in rng.sample(ks, min(cart_variants, len(ks))):
        prefix = base[:k]
        cur = set()
        for t in prefix:
            cur |= vs[t]
        disc = [t for t in range(n) if t not in prefix and not (vs[t] & cur)]
        t = rng.choice(sorted(disc))
        order = greedy_complete(prefix + [t], vs, n, rng)
        if tuple(order) in seen:
            continue
        if count_cart(order, vs) != min_cart + 1:
            continue
        orders.append(tuple(order))
        seen.add(tuple(order))

    # random permutations (multi-cartesian coverage)
    for _ in range(2):
        order = tuple(rng.sample(range(n), n))
        if order not in seen:
            orders.append(order)
            seen.add(order)
    return [list(o) for o in orders]


def process_query(qi, q, rdf2vec, counts, args):
    triples = [t[:3] for t in q["triples"]]  # lubm triples carry a 4th "."
    n = len(triples)
    rng = random.Random(31_000 + qi)
    orders = make_orders(triples, rng, args.max_sib, args.alts,
                         args.cart_variants)
    if not orders:
        return []
    oracle = CostOracle(triples, args.endpoint, args.timeout)
    vs = var_sets(triples)

    out = []
    for pi, order in enumerate(orders):
        y = oracle.c_out(order)
        if y is None or y <= 0:
            continue
        gen = torch.Generator().manual_seed(888_000_000 + 1000 * qi + pi)
        x = featurize_query(triples, rdf2vec, counts, rng=gen)
        x[n:, :64] = 0.0  # no fingerprints in stored data (training adds them)
        A = order_to_adjacency(order, n)
        edge_index = torch.tensor(np.argwhere(A), dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index,
                    y=torch.tensor([float(y)], dtype=torch.float))
        tr_strs = [" ".join(t) + "." for t in triples]
        out.append((data, tr_strs, count_cart(order, vs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--per-size", type=int, default=150)
    ap.add_argument("--max-sib", type=int, default=12,
                    help="max sibling variants per query")
    ap.add_argument("--alts", type=int, default=2,
                    help="alternative next-triples per depth")
    ap.add_argument("--cart-variants", type=int, default=3,
                    help="single-cartesian variants per query")
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="standalone/sib_plans")
    ap.add_argument("--queries", default=QUERIES)
    ap.add_argument("--emb-dir", default=EMB_DIR)
    args = ap.parse_args()

    with open(args.queries) as f:
        raw = json.load(f)
    by_size = {}
    for q in raw:
        by_size.setdefault(len(q["triples"]), []).append(q)
    del raw
    with open(os.path.join(args.emb_dir, "rdf2vec100dim.pkl"), "rb") as f:
        rdf2vec = pickle.load(f)
    with open(os.path.join(args.emb_dir, "counts.pkl"), "rb") as f:
        counts = pickle.load(f)

    # skip the first 25 per size: those are the benchmark/eval queries
    jobs = []
    for s in args.sizes:
        pool = by_size.get(s, [])[25:25 + args.per_size]
        jobs.extend((s * 100_000 + i, q) for i, q in enumerate(pool))
    print(f"{len(jobs)} queries")

    all_data, all_triples, n_cart, n_plans_hist = [], [], 0, {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_query, qi, q, rdf2vec, counts, args): qi
                for qi, q in jobs}
        for done, fut in enumerate(as_completed(futs)):
            res = fut.result()
            n_plans_hist[len(res)] = n_plans_hist.get(len(res), 0) + 1
            for data, tr, cart in res:
                all_data.append(data)
                all_triples.append(tr)
                n_cart += int(cart > 0)
            if (done + 1) % 50 == 0:
                print(f"  {done+1}/{len(jobs)} queries, {len(all_data)} plans "
                      f"({n_cart} cartesian), {time.time()-t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"dataset_size": len(all_data), "triples": all_triples,
                "data": all_data}, os.path.join(args.out, "dataset.pt"))
    print(f"saved {len(all_data)} plans ({n_cart} with cartesians) "
          f"to {args.out}/dataset.pt in {time.time()-t0:.0f}s")
    print(f"plans-per-query histogram: {dict(sorted(n_plans_hist.items()))}")


if __name__ == "__main__":
    main()
