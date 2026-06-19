"""Mine per-join intermediate cardinalities for the pretraining datasets into a
shared component-count cache (subcost_cache.json), for the auxiliary
intermediate-cardinality loss in train_dual_card.py.

The datasets store only the total C_out, not the per-join breakdown, but they
store the plan tree (edge_index) + triples, so the subcosts are recoverable:
walk each join's subtree, factor its triple set into connected components, COUNT
each missing component via QLever (cached, threaded). Verifies the mapping by
checking sum(per-join card) == stored y per sample.

    cd ~/Projects/GBJOv2 && uv run python -u -m v3.datagen.mine_subcosts \
        --sources cart,sib,addon --endpoint http://127.0.0.1:7020/ --threads 8
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from v3.train.train_dual import load_source
from v3.core import subcost as sc

def count_component(comp, atoms, endpoint, timeout, session):
    body = " . ".join(" ".join(atoms[i]) for i in sorted(comp)) + " ."
    q = f"SELECT (COUNT(*) AS ?count) WHERE {{ {body} }}"
    try:
        r = session.get(endpoint, params={"query": q},
                        headers={"Accept": "application/sparql-results+json"},
                        timeout=(5.0, timeout))
        r.raise_for_status()
        return int(r.json()["results"]["bindings"][0]["count"]["value"])
    except Exception:
        return -1   # censored / failed -> sample masked downstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="cart,sib,addon")
    ap.add_argument("--cart-data", default="v3/artifacts/plans/cart_plans_9_15")
    ap.add_argument("--sib-data", default="v3/artifacts/plans/sib_plans")
    ap.add_argument("--cache", default="v3/artifacts/cache/subcost_cache.json")
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache))
    print(f"cache: {len(cache)} component counts loaded", flush=True)

    # 1. collect every distinct component the datasets need
    needed = {}                          # key -> (comp, atoms)
    samples = []                         # (atoms, vs, subtrees, y) for verify
    for src in args.sources.split(","):
        raw = load_source(src.strip(), args.cart_data, args.sib_data, args.limit)
        for data, tr, _ in raw:
            n = (data.x.shape[0] + 1) // 2
            atoms = [sc.parse_atoms(t) for t in tr]
            vs = sc.var_sets(atoms)
            subtrees = sc.join_subtrees(data.edge_index, n)
            samples.append((atoms, vs, subtrees, float(data.y.item())))
            for leaves, _nodes in subtrees.values():
                for comp in sc.components(leaves, vs):
                    k = sc.comp_key(comp, atoms)
                    if k not in cache and k not in needed:
                        needed[k] = (comp, atoms)
        print(f"  {src}: cumulative {len(samples)} plans, "
              f"{len(needed)} new components to count", flush=True)

    # 2. count the missing components against QLever (threaded, periodic save)
    print(f"counting {len(needed)} components ...", flush=True)
    tl = threading.local()
    lock = threading.Lock()
    done = [0]
    items = list(needed.items())

    def work(item):
        k, (comp, atoms) = item
        s = getattr(tl, "s", None)
        if s is None:
            s = tl.s = requests.Session()
        v = count_component(comp, atoms, args.endpoint, args.timeout, s)
        with lock:
            cache[k] = v
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"   {done[0]}/{len(items)}", flush=True)
                json.dump(cache, open(args.cache, "w"))

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        list(ex.map(work, items))
    json.dump(cache, open(args.cache, "w"))
    n_cens = sum(1 for v in cache.values() if v < 0)
    print(f"cache now {len(cache)} entries ({n_cens} censored)", flush=True)

    # 3. verify the node<->prefix mapping: sum(per-join card) == stored y
    ok = bad = censored = 0
    worst = (0.0, None)
    for atoms, vs, subtrees, y in samples:
        tot, miss = 0, False
        for leaves, _nodes in subtrees.values():
            c = sc.card_from_cache(leaves, atoms, vs, cache)
            if c is None:
                miss = True
                break
            tot += c
        if miss:
            censored += 1
        elif abs(tot - y) <= 1e-3 * max(1.0, y):
            ok += 1
        else:
            bad += 1
            rel = abs(tot - y) / max(1.0, y)
            if rel > worst[0]:
                worst = (rel, (tot, y))
    print(f"verify sum(card)==y:  ok {ok}  MISMATCH {bad}  censored {censored}"
          f"  / {len(samples)} plans")
    if bad:
        print(f"  worst mismatch rel={worst[0]:.4f}  (sum={worst[1][0]}, y={worst[1][1]})")


if __name__ == "__main__":
    main()
