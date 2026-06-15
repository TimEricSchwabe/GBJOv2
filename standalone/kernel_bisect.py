"""Bisect the FastGBJO(4.5) vs deployed-kernel(5.59) decode gap for the MRT-20k
model @ folded params on the 100 val queries. All three run on the SAME torch
features (build_items), so A/B/C control for featurization; comparing C to the
rdflib GBJOPlanner-det plans isolates featurization as the last variable.

    cd ~/Projects/GBJOv2 && PYTHONPATH=standalone uv run python -u standalone/kernel_bisect.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from model_dual import FlatCostGNNDual
from gbjo_fast import FastGBJO, adjacency_to_join_order
from gbjo_cpp import CppGBJO
from finetune_mrt import (load_emb_source, build_items, deploy_params,
                          CStarOracle, CENSORED)

PACK = os.path.expanduser(
    "~/rdflib-joinordering/gbjo_pack/full-v2-reg-mrt-20k-deploy")
MODEL = "standalone/models/full-v2-reg-mrt-20k/model_mrt.pt"

dp = deploy_params(PACK)
params = {**dp, "lambda_cartesian": 0.0}
fast = FastGBJO(FlatCostGNNDual.load(MODEL), params=params)
cpp = CppGBJO(FlatCostGNNDual.load(MODEL), params=params)
emb, counts = load_emb_source(PACK)
items = build_items(json.load(
    open("standalone/mrt_queries_path20k_val100.json")), emb, counts)
oracle = CStarOracle("http://127.0.0.1:7020/", 10.0, "standalone/cstar_cache.json")


def score(plans):
    c = np.array([oracle.c_out(o, it["triples"]) for o, it in zip(plans, items)])
    l = np.log10(np.maximum(c, 1.0))
    return l.mean(), float(np.median(l)), int((c >= CENSORED).sum())


A = [adjacency_to_join_order(fast.optimize(it["x"], 10, share=it["share"])[0])
     for it in items]
B = [adjacency_to_join_order(
        cpp.optimize(it["x"], 10, share=it["share"], mask_cart=False)[0])
     for it in items]
C = [adjacency_to_join_order(
        cpp.optimize(it["x"], 10, share=it["share"], mask_cart=True)[0])
     for it in items]
oracle.save()

print("\n100 val queries, MRT-20k @ folded params -- true C* of decoded plan:")
for nm, P in [("A FastGBJO (no mask)", A), ("B CppGBJO (no mask)", B),
              ("C CppGBJO (mask_cart)", C)]:
    m, md, cat = score(P)
    print(f"  {nm:>22}: mean {m:.3f}  median {md:.3f}  catastrophes {cat}")

print(f"\nplan agreement:")
print(f"  A==B (FastGBJO vs kernel, no mask): {sum(a==b for a,b in zip(A,B))}/100")
print(f"  B==C (kernel mask off vs on):       {sum(b==c for b,c in zip(B,C))}/100")

gp = json.load(open("standalone/val100_plans.json"))["plans"]["new-det"]
print(f"  C==GBJOPlanner-det (torch vs numpy feats): "
      f"{sum(list(c)==list(g) for c,g in zip(C,gp))}/100")
print(f"  A==GBJOPlanner-det:                        "
      f"{sum(list(a)==list(g) for a,g in zip(A,gp))}/100")
