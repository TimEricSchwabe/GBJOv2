"""
Generate cartesian-containing training plans for path queries (sizes 9-15)
with EXACT C_out costs, without ever executing a cartesian product:

  card(triple set) = product over connected components of card(component)

so only counts of connected sub-patterns are sent to QLever (cached per query).

Per query: 3 plans -- greedy connected (cartesian-free contrast), corrupted
connected (one early cartesian), random permutation (usually cartesian).

Output: <out>/dataset.pt with {'data': [PyG-style Data], 'triples': [...]},
x WITHOUT fingerprints (training adds them), y = exact C_out, plus the same
'triples' string format as the existing datasets.

Usage:  
    uv run python standalone/gen_cartesian_plans.py --per-size 200 \
        --sizes 9 10 11 12 13 14 15 --out standalone/cart_plans_9_15
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
import requests
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbjo_fast import featurize_query
from bench_cartesian import order_to_adjacency

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERIES = os.path.join(REPO, "data", "queries", "wikidata", "path", "path_queries.json")
EMB_DIR = os.path.join(REPO, "data", "embeddings", "wikidata")


def var_sets(triples):
    return [set(a for a in t[:3] if a.startswith("?")) for t in triples]


def components(idxs, vs):
    """Connected components of the share graph restricted to idxs."""
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


class CostOracle:
    def __init__(self, triples, endpoint, timeout):
        self.triples = triples
        self.vs = var_sets(triples)
        self.endpoint = endpoint
        self.timeout = timeout
        self.cache = {}
        self.session = requests.Session()

    def card_connected(self, comp):
        if comp in self.cache:
            return self.cache[comp]
        body = " . ".join(" ".join(self.triples[i]) for i in sorted(comp)) + " ."
        q = f"SELECT (COUNT(*) AS ?count) WHERE {{ {body} }}"
        try:
            r = self.session.get(self.endpoint, params={"query": q},
                                 headers={"Accept": "application/sparql-results+json"},
                                 timeout=(5.0, self.timeout))
            r.raise_for_status()
            val = int(r.json()["results"]["bindings"][0]["count"]["value"])
        except Exception:
            val = None
        self.cache[comp] = val
        return val

    def card(self, idx_set):
        """Exact cardinality of any triple set via component factorization."""
        prod = 1
        for comp in components(idx_set, self.vs):
            c = self.card_connected(comp)
            if c is None:
                return None
            prod *= c
        return prod

    def c_out(self, order):
        total = 0
        prefix = set(order[:1])
        for t in order[1:]:
            prefix.add(t)
            c = self.card(prefix)
            if c is None:
                return None
            total += c
        return total


def connected_order(vs, n, rng):
    start = rng.randrange(n)
    order, left = [start], set(range(n)) - {start}
    while left:
        cands = [t for t in sorted(left) if any(vs[t] & vs[u] for u in order)]
        if not cands:
            return None
        order.append(rng.choice(cands))
        left -= {order[-1]}
    return order


def corrupted_order(vs, n, rng):
    """Connected order with one non-sharing triple forced early -> >=1 cartesian."""
    base = connected_order(vs, n, rng)
    if base is None:
        return None
    for _ in range(20):
        i = rng.randrange(2, n)
        t = base[i]
        if not (vs[t] & (vs[base[0]] | vs[base[1]])):
            order = [base[0], t] + [x for x in base[1:] if x != t]
            return order
    return None


def count_cart(order, vs):
    cur, cart = set(vs[order[0]]), 0
    for t in order[1:]:
        if not (cur & vs[t]):
            cart += 1
        cur |= vs[t]
    return cart


def process_query(qi, q, rdf2vec, counts, endpoint, timeout):
    triples = [t[:3] for t in q["triples"]]  # lubm triples carry a 4th "."
    n = len(triples)
    vs = var_sets(triples)
    rng = random.Random(1000 + qi)
    oracle = CostOracle(triples, endpoint, timeout)

    orders = []
    co = connected_order(vs, n, rng)
    if co is None:
        return []
    orders.append(co)
    cu = corrupted_order(vs, n, rng)
    if cu is not None:
        orders.append(cu)
    orders.append(rng.sample(range(n), n))  # random permutation

    out = []
    for pi, order in enumerate(orders):
        y = oracle.c_out(order)
        if y is None or y <= 0:
            continue
        gen = torch.Generator().manual_seed(777_000_000 + 1000 * qi + pi)
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
    ap.add_argument("--sizes", type=int, nargs="+", default=[9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--per-size", type=int, default=200)
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="standalone/cart_plans_9_15")
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
    print(f"{len(jobs)} queries -> up to {3*len(jobs)} plans")

    all_data, all_triples, n_cart = [], [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_query, qi, q, rdf2vec, counts,
                          args.endpoint, args.timeout): qi for qi, q in jobs}
        for done, fut in enumerate(as_completed(futs)):
            for data, tr, cart in fut.result():
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


if __name__ == "__main__":
    main()
