"""
Equivalence + runtime comparison: original src/optimization/methods.py::GBJO
vs standalone/gbjo_fast.py::FastGBJO (eager and torch.compile) on identical
featurized wikidata-star queries.

Usage:
    uv run python standalone/bench_compare.py [--steps 10] [--per-size 3] [--sizes 4 6 8 10 12 14]
"""

import argparse
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gbjo_fast import FastGBJO, FlatCostGNN, featurize_query, adjacency_to_join_order

MODEL_DIR = os.path.join(REPO, "models", "wikidata-log1p-plus-cartesian")
QUERIES = os.path.join(REPO, "data", "wikidata-star", "star_queries.json")
EMB_DIR = os.path.join(REPO, "data", "embeddings", "wikidata")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_cache.pt")

# config_wikidata_star optimization_params (src/evaluation_parallel.py __main__)
GBJO_PARAMS = {
    "learning_rate": 4.9,
    "lambda_acyclic": 29,
    "lambda_triple_in": 1.5,
    "lambda_triple_out": 1.4,
    "lambda_join_in": 3.6,
    "lambda_join_out": 4.1,
    "lambda_entropy": 0.0,
    "lambda_total_penalty": 0.99,
    "lambda_left_linear": 60,
    "init_tau": 4,
    "min_tau": 0.49,
    "tau_decay": 0.973,
    "use_temperature_annealing": True,
    "return_best": True,
    "min_penalty_threshold": 9.96,
    "use_lambda_ramping": True,
    "logit_sampling": "softmax",
    "save_animation_data": False,
    "animation_save_interval": 10,
    "lambda_ramp_exponent": 1.01,
    "lr_warmup_steps": 46,
    "gradient_clip_norm": 4.7,
    "use_lr_scheduling": True,
    "decoding_method": "beam",
    "use_gumbel_noise": False,
}


def build_query_set(sizes, per_size):
    """Load star_queries.json once, featurize a per-size sample, cache to disk."""
    key = {"sizes": list(sizes), "per_size": per_size}
    if os.path.exists(CACHE):
        cached = torch.load(CACHE, weights_only=False)
        if cached.get("key") == key:
            print(f"Loaded {len(cached['items'])} featurized queries from cache")
            return cached["items"]

    print("Loading star_queries.json ...")
    t0 = time.time()
    with open(QUERIES) as f:
        raw = json.load(f)
    print(f"  {len(raw)} queries in {time.time()-t0:.1f}s")

    by_size = {}
    for q in raw:
        by_size.setdefault(len(q["triples"]), []).append(q)
    del raw

    print("Loading rdf2vec + counts ...")
    with open(os.path.join(EMB_DIR, "rdf2vec100dim.pkl"), "rb") as f:
        rdf2vec = pickle.load(f)
    with open(os.path.join(EMB_DIR, "counts.pkl"), "rb") as f:
        counts = pickle.load(f)

    items = []
    for size in sizes:
        pool = by_size.get(size, [])
        if not pool:
            print(f"  size {size}: no queries available, skipping")
            continue
        for i, q in enumerate(pool[:per_size]):
            rng = torch.Generator().manual_seed(10_000 * size + i)
            x = featurize_query(q["triples"], rdf2vec, counts, rng=rng)
            items.append({"size": size, "triples": q["triples"], "x": x})
    torch.save({"key": key, "items": items}, CACHE)
    print(f"Featurized + cached {len(items)} queries")
    return items


def load_original_model():
    from model import CostGNNv3
    model = CostGNNv3(node_feature_dim=307, hidden_dim=128, n_layers=6,
                      use_jk=False, jk_mode="cat", use_residual=True,
                      use_layer_norm=False, dropout=0.0)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_original(GBJO, model, x, steps):
    from torch_geometric.data import Data
    data = Data(x=x.clone())
    res = GBJO(data, model, "cpu", optimization_steps=steps, verbose=False,
               **GBJO_PARAMS)
    final_A, triples_num, pred_cost = res[:3]
    return np.asarray(final_A.cpu().numpy() > 0.5, dtype=int), pred_cost


def canon(order):
    """Join order with the first (symmetric) pair canonicalized."""
    return tuple(sorted(order[:2])) + tuple(order[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--per-size", type=int, default=3)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--cpp", action="store_true", help="include C++ kernel variant")
    args = ap.parse_args()

    torch.set_num_threads(1)
    items = build_query_set(args.sizes, args.per_size)

    flat = FlatCostGNN.load(os.path.join(MODEL_DIR, "model.pt"))
    fast = FastGBJO(flat, params={k: GBJO_PARAMS[k] for k in
                                  ("learning_rate", "lambda_acyclic", "lambda_triple_in",
                                   "lambda_triple_out", "lambda_join_in", "lambda_join_out",
                                   "lambda_entropy", "lambda_total_penalty", "lambda_left_linear",
                                   "init_tau", "min_tau", "use_temperature_annealing",
                                   "return_best", "use_lambda_ramping", "lambda_ramp_exponent",
                                   "gradient_clip_norm", "use_lr_scheduling")})
    fast_c = None
    if not args.skip_compile:
        fast_c = FastGBJO(flat, params=fast.params, compile_step=True)

    cpp = None
    if args.cpp:
        from gbjo_cpp import CppGBJO
        cpp = CppGBJO(flat, params=fast.params)

    orig_model = None
    GBJO = None
    if not args.skip_original:
        print("Importing original GBJO (heavy imports) ...")
        from optimization import GBJO as _GBJO
        GBJO = _GBJO
        orig_model = load_original_model()

    # warmups
    wx = items[0]["x"]
    fast.optimize(wx, optimization_steps=2)
    if fast_c is not None:
        print("Warming torch.compile per size ...")
        for size in sorted({it["size"] for it in items}):
            xw = next(it["x"] for it in items if it["size"] == size)
            t0 = time.perf_counter()
            fast_c.optimize(xw, optimization_steps=2)
            print(f"  size {size}: compile warmup {time.perf_counter()-t0:.1f}s")
    if GBJO is not None:
        run_original(GBJO, orig_model, wx, 2)

    rows = []
    for it in items:
        x, size = it["x"], it["size"]
        row = {"size": size}

        # fast eager
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            A_fast, cost_fast = fast.optimize(x, optimization_steps=args.steps)
            ts.append(time.perf_counter() - t0)
        row["fast_s"] = float(np.median(ts))
        row["fast_cost"] = cost_fast
        order_fast = canon(adjacency_to_join_order(A_fast))

        # C++ kernel
        if cpp is not None:
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                A_cpp, cost_cpp = cpp.optimize(x, optimization_steps=args.steps)
                ts.append(time.perf_counter() - t0)
            row["cpp_s"] = float(np.median(ts))
            row["cpp_cost"] = cost_cpp
            row["cpp_match_fast"] = bool(np.array_equal(A_cpp, A_fast))

        # fast compiled
        if fast_c is not None:
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                A_c, cost_c = fast_c.optimize(x, optimization_steps=args.steps)
                ts.append(time.perf_counter() - t0)
            row["compiled_s"] = float(np.median(ts))
            row["compiled_match_fast"] = bool(canon(adjacency_to_join_order(A_c)) == order_fast)

        # original
        if GBJO is not None:
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                A_orig, cost_orig = run_original(GBJO, orig_model, x, args.steps)
                ts.append(time.perf_counter() - t0)
            row["orig_s"] = float(np.median(ts))
            row["orig_cost"] = cost_orig
            order_orig = canon(adjacency_to_join_order(A_orig))
            row["plan_match"] = bool(order_fast == order_orig)
            row["adj_match"] = bool(np.array_equal(A_fast, A_orig))
            row["cost_rel_diff"] = abs(cost_fast - cost_orig) / max(cost_orig, 1e-12)

        rows.append(row)
        print(row)

    # summary
    print("\n=== SUMMARY (steps=%d) ===" % args.steps)
    sizes = sorted({r["size"] for r in rows})
    hdr = (f"{'size':>4} {'orig_s':>9} {'fast_s':>9} {'cpp_s':>9} "
           f"{'orig/fast':>9} {'orig/cpp':>9} {'plan==':>7} {'cpp==':>6}")
    print(hdr)
    for s in sizes:
        rs = [r for r in rows if r["size"] == s]
        o = np.median([r.get("orig_s", np.nan) for r in rs])
        f_ = np.median([r["fast_s"] for r in rs])
        c = np.median([r.get("cpp_s", np.nan) for r in rs])
        match = sum(r.get("plan_match", False) for r in rs)
        cm = sum(r.get("cpp_match_fast", False) for r in rs)
        print(f"{s:>4} {o:>9.4f} {f_:>9.4f} {c:>9.4f} {o/f_:>9.2f} {o/c:>9.2f} "
              f"{match:>4}/{len(rs)} {cm:>3}/{len(rs)}")
    if any("plan_match" in r for r in rows):
        total = sum(1 for r in rows if "plan_match" in r)
        match = sum(r.get("plan_match", False) for r in rows)
        adj = sum(r.get("adj_match", False) for r in rows)
        print(f"\nplan match: {match}/{total}   exact adjacency match: {adj}/{total}")
        mism = [r for r in rows if not r.get("plan_match", True)]
        for r in mism:
            print(f"  mismatch size={r['size']}: fast_cost={r['fast_cost']:.4g} "
                  f"orig_cost={r['orig_cost']:.4g} rel_diff={r['cost_rel_diff']:.2e}")


if __name__ == "__main__":
    main()
