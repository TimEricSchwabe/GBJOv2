"""
Per-predicate degree statistics for the cardinality upper bound (Test A).

For a left-deep path/BGP order, joining pattern (?a P ?b) onto a prefix that
binds ?a multiplies the intermediate by at most the max out-degree of P
(max triples sharing a single subject); joining on ?b uses max in-degree.
These are never-underestimating fan-out bounds. Combined with per-entity
subject/object counts (already in kg_index.occ) and per-predicate totals,
they give an order-sensitive upper bound on each intermediate's size.

Output (aligned to kg_index rel_keys order) -> kg_index/<kg>/pred_stats.npz:
    count    (R,) triples with that predicate
    n_subj   (R,) distinct subjects
    n_obj    (R,) distinct objects
    max_out  (R,) max triples sharing one subject  (out-degree bound)
    max_in   (R,) max triples sharing one object   (in-degree bound)

    cd ~/Projects/GBJOv2 && uv run python standalone/pred_stats.py \
        --nt /Users/timschwabe/Projects/qlever/wikidata/graph.nt \
        --out standalone/kg_index/wikidata
"""

import argparse
import os
import time

import numpy as np

from kg_index import parse_nt


def deg_stats(p, x, R):
    """Per-predicate (max count over distinct x, number of distinct x), where
    each row is a triple with predicate p[i] and key x[i] (subject or object)."""
    order = np.lexsort((x, p))
    ps, xs = p[order], x[order]
    pair_change = np.empty(len(ps), dtype=bool)
    pair_change[0] = True
    pair_change[1:] = (ps[1:] != ps[:-1]) | (xs[1:] != xs[:-1])
    starts = np.nonzero(pair_change)[0]
    counts = np.diff(np.append(starts, len(ps)))  # count per (p, x) pair
    pair_p = ps[starts]
    max_deg = np.zeros(R, dtype=np.int64)
    n_distinct = np.zeros(R, dtype=np.int64)
    np.maximum.at(max_deg, pair_p, counts)
    np.add.at(n_distinct, pair_p, 1)
    return max_deg, n_distinct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", required=True)
    ap.add_argument("--out", required=True, help="kg_index dir (has rel_keys.txt)")
    args = ap.parse_args()

    keys_e, keys_r, s, p, o = parse_nt(args.nt)
    R = len(keys_r)
    # rel_keys.txt order must match parse_nt's keys_r (same builder)
    with open(os.path.join(args.out, "rel_keys.txt"), encoding="utf-8") as f:
        saved = f.read().splitlines()
    assert saved == keys_r, "rel_keys order mismatch; rebuild kg_index first"

    t0 = time.time()
    count = np.bincount(p, minlength=R).astype(np.int64)
    max_out, n_subj = deg_stats(p, s, R)
    max_in, n_obj = deg_stats(p, o, R)
    np.savez(os.path.join(args.out, "pred_stats.npz"),
             count=count, n_subj=n_subj, n_obj=n_obj,
             max_out=max_out, max_in=max_in)
    print(f"pred_stats: {R} predicates ({time.time()-t0:.0f}s)")
    top = np.argsort(max_out)[::-1][:5]
    for i in top:
        print(f"  {keys_r[i][:60]:60} count={count[i]:>9} "
              f"max_out={max_out[i]:>7} max_in={max_in[i]:>7}")


if __name__ == "__main__":
    main()
