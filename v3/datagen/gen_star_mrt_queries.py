"""
Build a 20k STAR query set for MRT fine-tuning, analogous to
mrt_queries_path20k.json (same {"triples": [[s,p,o],...]} format, same per-size
distribution) but star-shaped, sampled from the wikidata star query pool. The
path set is for path models; star-v2-* are star models, so MRT must see stars.

    uv run python -m v3.datagen.gen_star_mrt_queries
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from v3 import paths

SRC = str(paths.DATA / "queries/wikidata/star/star_queries.json")
PATH20K = str(paths.QUERIES / "mrt_queries_path20k.json")
OUT = str(paths.QUERIES / "mrt_queries_star20k.json")
OUT_VAL = str(paths.QUERIES / "mrt_queries_star20k_val100.json")
SEED = 0


def sig(triples):
    return tuple(tuple(t) for t in triples)


def main():
    rng = np.random.default_rng(SEED)
    target = Counter(len(q["triples"]) for q in json.load(open(PATH20K)))
    print(f"target size distribution (from path20k): {dict(sorted(target.items()))}"
          f"  total {sum(target.values())}")

    src = json.load(open(SRC))
    # dedup by triple-set signature; group distinct queries by size
    by_size, seen = defaultdict(list), set()
    for q in src:
        tr = q["triples"]
        s = sig(tr)
        if s in seen:
            continue
        seen.add(s)
        by_size[len(tr)].append(tr)
    print(f"distinct source queries: {len(seen):,}; "
          f"per-size available: {dict(sorted((k, len(v)) for k, v in by_size.items()))}")

    # sample per size to match target; redistribute any deficit to surplus sizes
    chosen, deficit = [], 0
    for size, want in sorted(target.items()):
        pool = by_size.get(size, [])
        take = min(want, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        picked = {i for i in idx.tolist()}
        chosen.extend(pool[i] for i in idx.tolist())
        by_size[size] = [t for j, t in enumerate(pool) if j not in picked]  # leftovers
        deficit += want - take
        if want - take:
            print(f"  size {size}: wanted {want}, only {len(pool)} available "
                  f"(short {want - take})")
    if deficit:                                  # fill from any leftover distinct
        leftover = [t for v in by_size.values() for t in v]
        idx = rng.choice(len(leftover), size=min(deficit, len(leftover)), replace=False)
        chosen.extend(leftover[i] for i in idx.tolist())
        print(f"  filled {len(idx)} of {deficit} deficit from other sizes")

    rng.shuffle(chosen)
    queries = [{"triples": tr} for tr in chosen]
    json.dump(queries, open(OUT, "w"))
    realized = Counter(len(q["triples"]) for q in queries)
    print(f"\nwrote {len(queries):,} -> {OUT}")
    print(f"realized size distribution: {dict(sorted(realized.items()))}")

    val_idx = rng.choice(len(queries), size=100, replace=False)
    json.dump([queries[i] for i in val_idx.tolist()], open(OUT_VAL, "w"))
    print(f"wrote 100 (subset) -> {OUT_VAL}")


if __name__ == "__main__":
    main()
