"""
Validate a trained CostGNNDual: does it now rank cartesian plans above
connected (cartesian-free) plans, and how calibrated is it against true C_out?

For each held-out eval path query (the same 100 used by bench_cartesian):
  conn    = greedy connected order      (0 cartesian joins)
  corrupt = connected + 1 early cartesian
  random  = random permutation          (usually multi-cartesian)

Reports, for the new model and the old CostGNNv3 side by side:
  - % queries where pred(corrupt) > pred(conn)  [should be ~100%]
  - % queries where pred(random)  > pred(conn)  [for random with >=1 cart]
  - with --endpoint: median qerr(log) = max(logp/logt, logt/logp) per plan
    type (q-error between log-space values, as in cost_model_training.py)

Usage:
    uv run python standalone/validate_dual.py --model standalone/models/dual-v1/model.pt \
        [--endpoint http://127.0.0.1:7020/] [--no-truth]
"""

import argparse
import math
import os
import random
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_cartesian import build_query_set, order_to_adjacency, MODEL
from dual_data import triple_var_sets, share_edge_index
from model_dual import CostGNNDual
from gbjo_fast import FlatCostGNN
from gen_cartesian_plans import (var_sets, connected_order, corrupted_order,
                                 count_cart, CostOracle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="standalone/models/dual-v1/model.pt")
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--no-truth", action="store_true")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 11, 14])
    ap.add_argument("--queries", default=None, help="path_queries.json override")
    ap.add_argument("--emb-dir", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-old", action="store_true",
                    help="skip the old-CostGNNv3 comparison (e.g. for lubm)")
    args = ap.parse_args()

    torch.set_num_threads(1)
    import bench_cartesian as bc
    items = build_query_set(args.sizes, 25,
                            queries=args.queries or bc.QUERIES,
                            emb_dir=args.emb_dir or bc.EMB_DIR,
                            cache=args.cache or bc.CACHE)

    model = CostGNNDual()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    models = ("new",) if args.no_old else ("new", "old")
    old = None if args.no_old else FlatCostGNN.load(MODEL)

    def pred_new(x, order, n, esh):
        A = order_to_adjacency(order, n)
        ei = torch.tensor(np.argwhere(A), dtype=torch.long).t().contiguous()
        with torch.no_grad():
            return float(model(x, ei, esh,
                               torch.zeros(x.shape[0], dtype=torch.long),
                               num_graphs=1).item())

    def pred_old(x, order, n):
        A = torch.tensor(order_to_adjacency(order, n), dtype=torch.float32)
        with torch.no_grad():
            h0 = old.project_x(x)
            return float(old.forward_from_h0(h0, A).item())

    rank = {"new": {"corrupt": [], "random": []},
            "old": {"corrupt": [], "random": []}}
    qerr = {"new": {}, "old": {}}
    for qi, it in enumerate(items):
        triples, n, x = it["triples"], it["size"], it["x"]
        vs = var_sets(triples)
        rng = random.Random(55_000 + qi)
        orders = {"conn": connected_order(vs, n, rng)}
        cu = corrupted_order(vs, n, rng)
        if cu is not None:
            orders["corrupt"] = cu
        orders["random"] = rng.sample(range(n), n)
        if orders["conn"] is None:
            continue
        esh = share_edge_index(triple_var_sets(x, n))

        preds = {m: {} for m in models}
        for name, order in orders.items():
            preds["new"][name] = pred_new(x, order, n, esh)
            if old is not None:
                preds["old"][name] = pred_old(x, order, n)
        for m in models:
            if "corrupt" in orders:
                rank[m]["corrupt"].append(
                    preds[m]["corrupt"] > preds[m]["conn"])
            if count_cart(orders["random"], vs) > 0:
                rank[m]["random"].append(
                    preds[m]["random"] > preds[m]["conn"])

        if not args.no_truth:
            oracle = CostOracle(triples, args.endpoint, args.timeout)
            for name, order in orders.items():
                y = oracle.c_out(order)
                if y is None or y <= 0:
                    continue
                lt = math.log(y)
                for m in models:
                    lp = preds[m][name]
                    eps = 1e-10
                    qerr[m].setdefault(name, []).append(
                        max(lt / (lp + eps), lp / (lt + eps)))

    print("\n=== ranking: cartesian plan predicted MORE expensive than connected ===")
    for m in models:
        c, r = rank[m]["corrupt"], rank[m]["random"]
        print(f"  {m:>4}: corrupt>conn {100*np.mean(c):.0f}% ({sum(c)}/{len(c)})   "
              f"random>conn {100*np.mean(r):.0f}% ({sum(r)}/{len(r)})")
    if not args.no_truth and qerr["new"]:
        print("\n=== median qerr(log) = max(logp/logt, logt/logp) ===")
        print(f"  {'plan':>8} " + " ".join(f"{m:>6}" for m in models) + "   (n)")
        for name in ("conn", "corrupt", "random"):
            if name in qerr["new"]:
                print(f"  {name:>8} " +
                      " ".join(f"{np.median(qerr[m][name]):>6.2f}"
                               for m in models) +
                      f"   ({len(qerr['new'][name])})")


if __name__ == "__main__":
    main()
