"""Kernel-decode transfer check.

Val regret in finetune_mrt is measured against the diverse training pool
(deterministic mode + Gumbel samples + connected-greedy injection). The
DEPLOYED kernel instead decodes DETERMINISTICALLY: beam search across the
unroll steps, then arg-min predicted cost -- no Gumbel, no injection. A regret
win on the rich pool only matters if it survives this thinner decode.

This runs that exact deterministic path (FastGBJO.optimize, the reference the
C++ kernel replicates) for the pretrained vs the MRT-fine-tuned decoder on the
overfit queries, and compares the TRUE cost (C* via QLever) of the picked plan.
It isolates whether the fine-tuned LANDSCAPE decodes better plans through the
deployed path -- before paying for a full end-to-end rdflib run.

    cd ~/Projects/GBJOv2 && PYTHONPATH=standalone uv run python -u \
      standalone/kernel_transfer_check.py \
      --pre standalone/models/overfit-gps-v2/model_rank.pt \
      --mrt standalone/models/mrt-abl-noTR/model_mrt.pt \
      --pack ~/rdflib-joinordering/gbjo_pack/overfit-gps-v2 \
      --endpoint http://127.0.0.1:7020/ --cache standalone/cstar_cache.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_dual import FlatCostGNNDual
from gbjo_fast import FastGBJO, adjacency_to_join_order
from finetune_mrt import (CStarOracle, load_emb_source, build_items, CENSORED,
                          deploy_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="pretrained decoder .pt")
    ap.add_argument("--mrt", required=True, help="MRT fine-tuned decoder .pt")
    ap.add_argument("--queries", default="standalone/overfit_queries.json")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--cache", default="standalone/cstar_cache.json")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--min-tau", type=float, default=None,
                    help="override the GD-search final anneal floor at DECODE "
                         "time (must match how the decoder was trained)")
    ap.add_argument("--no-deploy-params", action="store_true",
                    help="use FastGBJO stale defaults instead of pack deploy params")
    args = ap.parse_args()

    pre = FastGBJO(FlatCostGNNDual.load(args.pre),
                   params={"lambda_cartesian": 0.0})
    mrt = FastGBJO(FlatCostGNNDual.load(args.mrt),
                   params={"lambda_cartesian": 0.0})
    dp = deploy_params(args.pack)
    for g in (pre, mrt):
        if dp and not args.no_deploy_params:
            g.params.update(dp)
        if args.min_tau is not None:
            g.params["min_tau"] = args.min_tau
        g._sched_cache.clear()
    emb, counts = load_emb_source(args.pack)
    items = build_items(json.load(open(args.queries)), emb, counts)
    oracle = CStarOracle(args.endpoint, args.timeout, args.cache)

    cp, cm, same = [], [], 0
    for it in items:
        Ap, _ = pre.optimize(it["x"], optimization_steps=args.steps,
                             share=it["share"])
        Am, _ = mrt.optimize(it["x"], optimization_steps=args.steps,
                             share=it["share"])
        op = adjacency_to_join_order(Ap)
        om = adjacency_to_join_order(Am)
        same += (op == om)
        cp.append(oracle.c_out(op, it["triples"]))
        cm.append(oracle.c_out(om, it["triples"]))
    oracle.save()

    cp, cm = np.array(cp), np.array(cm)
    lp = np.log10(np.maximum(cp, 1.0))
    lm = np.log10(np.maximum(cm, 1.0))
    cenp, cenm = cp >= CENSORED, cm >= CENSORED
    better = int((lm < lp - 1e-6).sum())
    worse = int((lm > lp + 1e-6).sum())
    n = len(items)
    print(f"\ndeterministic kernel decode on {n} overfit queries "
          f"({same} produced an identical plan):")
    print(f"  pretrained : mean log10 C* {lp.mean():.3f}   median "
          f"{np.median(lp):.3f}   catastrophes(timeout) {int(cenp.sum())}")
    print(f"  MRT        : mean log10 C* {lm.mean():.3f}   median "
          f"{np.median(lm):.3f}   catastrophes(timeout) {int(cenm.sum())}")
    print(f"  MRT better on {better} / worse on {worse} / "
          f"tie on {n - better - worse}")
    print(f"  mean log10 C* improvement (pre - mrt): {(lp - lm).mean():+.3f}  "
          f"(>0 = MRT cheaper)")


if __name__ == "__main__":
    main()
