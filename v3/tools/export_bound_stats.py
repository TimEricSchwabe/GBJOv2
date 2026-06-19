"""
Export per-term cardinality statistics for the overfit queries so the
bound validation (rdflib side) can compute upper bounds without the full
KG index. Combines entity subject/object counts (kg_index.occ) with
per-predicate fan-out (pred_stats).

    cd ~/Projects/GBJOv2 && uv run python -m v3.tools.export_bound_stats
"""

import json
import os
import sys

import numpy as np

from v3.core.kg_index import KGIndex

KG = "v3/artifacts/index/wikidata"
QUERIES = "v3/artifacts/queries/overfit_queries.json"
OUT = "v3/artifacts/stats/query_bound_stats.json"


def main():
    kg = KGIndex.load(KG)
    ps = np.load(os.path.join(KG, "pred_stats.npz"))
    max_out, max_in = ps["max_out"], ps["max_in"]

    with open(QUERIES) as f:
        queries = json.load(f)
    atoms = {a for q in queries for t in q["triples"] for a in t
             if not a.startswith("?")}

    stats = {}
    for a in atoms:
        rec = {}
        e = kg.node_id(a, "ent")
        if e >= 0:
            rec["subj_count"] = int(kg.occ[e][0])
            rec["obj_count"] = int(kg.occ[e][2])
        r = kg.node_id(a, "rel")
        if r >= 0:
            ri = r - kg.nE
            rec["pred_count"] = int(kg.occ[r][1])
            rec["max_out"] = int(max_out[ri])
            rec["max_in"] = int(max_in[ri])
        stats[a] = rec

    out = {"total_triples": int(kg.nT), "stats": stats}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote stats for {len(stats)} terms -> {OUT} "
          f"(total triples {kg.nT})")


if __name__ == "__main__":
    main()
