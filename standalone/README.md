# Standalone fast GBJO

Self-contained "query in → plan out" implementation of the GBJO gradient-based
join-order search (`src/optimization/methods.py::GBJO`), optimized for CPU
latency. **Same algorithm, verified identical output** at the production
config (`optimization_steps=10`).

Two tiers:
1. `gbjo_fast.py` — optimized PyTorch (2.5–8× vs original, bit-exact at steps=10)
2. `gbjo_kernel.cpp` + `gbjo_cpp.py` — fully fused C++ loop
   (**13–43× at steps=10, 67× at steps=2500**; 60/60 identical plans)

The C++ kernel is portable: it needs only CBLAS + LAPACK `dgesv` (Accelerate
on macOS, OpenBLAS on Linux) and uses a self-contained vectorizable Cephes
`expf`, so the numerics are identical on both platforms. The wrapper picks
`libgbjo.dylib`/`.so` by platform and pins BLAS to 1 thread.

```bash
# macOS (Accelerate / AMX)
clang++ -O3 -std=c++17 -dynamiclib -framework Accelerate \
    -DACCELERATE_NEW_LAPACK -o standalone/libgbjo.dylib standalone/gbjo_kernel.cpp

# Linux (OpenBLAS; apt install libopenblas-dev)
g++ -O3 -std=c++17 -shared -fPIC -o standalone/libgbjo.so \
    standalone/gbjo_kernel.cpp -lopenblas
```

## Usage

```bash
# one query
uv run python standalone/gbjo_fast.py \
    --model models/wikidata-log1p-plus-cartesian/model.pt \
    --embeddings data/embeddings/wikidata \
    --query '[["?s","<http://www.wikidata.org/prop/direct/P31>","?o0"], ...]' \
    --steps 10

# equivalence + speed comparison vs the original implementation
uv run python standalone/bench_compare.py --steps 10 --per-size 10 --sizes 4 6 8 10 12 14
```

As a library:

```python
from gbjo_fast import FastGBJO, FlatCostGNN, featurize_query, adjacency_to_join_order

model = FlatCostGNN.load("models/.../model.pt")
gbjo = FastGBJO(model)                      # params default to config_wikidata_star
x = featurize_query(triples, rdf2vec, counts)
A, pred_cost = gbjo.optimize(x, optimization_steps=10)
join_order = adjacency_to_join_order(A)
```

## What was changed (and why it is equivalent)

| change | speedup source | equivalence argument |
|---|---|---|
| one `(2n-2, n-1)` logit matrix, single row-softmax | replaces all-pairs edge list + `-inf` masking + 3 scatter ops | masked/invalid edges have softmax weight exactly 0 and zero gradient; dropping them changes nothing |
| dense `Aᵀ@H` message passing, flat weights | replaces PyG `MessagePassing` (~25 python ops/layer) | same math; reduction order differs only in last ulp |
| input projection precomputed once | `sign·log1p` + 307→128 Linear of **constant** features was recomputed 2×/step | identical values |
| `trace(expm(A))` custom backward = `expm(A)ᵀ` | avoids torch's 2N×2N Fréchet-derivative backward | exact analytic gradient |
| discrete plans deduplicated, scored once after the loop | the in-loop scoring (2nd full GNN forward per step) never feeds back into optimization | first-seen order + strict `<` selects the identical winner |
| `beam_exact`: lightweight candidate tuples, only winners materialize sets | original copies 2 sets + 2 lists for **every** candidate (~1300/step) | same float64 op order, same `nsmallest` stability, winners' sets built by the same `parent - {x}` ops → identical iteration order → bit-identical incl. tie-breaking |
| step-0 beam projection cached per query size | step-0 logits are all zero → A is query-independent | identical input → identical output |
| precomputed OneCycleLR schedule + 2-op SGD update | no optimizer/scheduler objects in the loop | schedule extracted from the real `OneCycleLR` |
| no histories / `.item()` / plotting / animation | ~10 syncs/step removed | output-irrelevant |

`torch.compile` was evaluated and adds **nothing** (±1%) on top of the flat
eager implementation — the remaining model cost is matmul-bound, not
dispatch-bound.

## Verification & benchmarks

(Apple Silicon, 1 thread, torch 2.9.1 CPU, `steps=10` = production config,
median over queries, wikidata-star queries, wikidata-log1p-plus-cartesian model)

| n triples | original | python fast | C++ | orig/cpp | plans identical |
|---|---|---|---|---|---|
| 4  | 13.8 ms | 5.4 ms  | 0.9 ms | 15.7× | 10/10 |
| 6  | 17.6 ms | 6.3 ms  | 1.4 ms | 12.9× | 10/10 |
| 8  | 24.7 ms | 6.9 ms  | 1.1 ms | 22.7× | 10/10 |
| 10 | 38.7 ms | 8.0 ms  | 1.4 ms | 27.5× | 10/10 |
| 12 | 55.2 ms | 8.8 ms  | 1.6 ms | 34.0× | 10/10 |
| 14 | 79.6 ms | 10.2 ms | 1.9 ms | 42.9× | 10/10 |

60/60 plans identical across **all three** implementations (python fast is
additionally bit-identical to the original in adjacency and cost).
At `steps=2500` (n=14): 22.5 s → 2.19 s (python) → **0.34 s (C++, 67×)**,
same final plan and cost.

## Why the C++ tier wins (measured, not guessed)

The torch gemms themselves were already optimal — all 18 matmuls of one
forward take 27 µs at 429 GFLOP/s (Accelerate/AMX). But the full torch
fwd+bwd step took 560 µs: **~80% was framework overhead** — autograd graph
construction, ~200 op dispatches at ~0.6 µs each, unfused elementwise, a
55 µs generic `matrix_exp`, allocations. The C++ kernel keeps the same BLAS
calls and removes everything else:

- hand-derived backward w.r.t. the adjacency only (weights are frozen →
  no weight-gradient gemms, no graph construction)
- custom Padé-13 `expm` (double precision, ~µs) with the analytic
  `d trace(expm(A))/dA = expm(A)ᵀ` backward
- GELU via Abramowitz–Stegun erf (|err| < 1.5e-7) with vForce-vectorized exp
- beam projection on bitmask sets (~10 µs vs 300 µs Python)
- softmax/penalties/SGD fused into plain loops; schedules precomputed in Python

C++ tie-breaking on exact beam-score ties is deterministic ascending-index
(Python set order is not reproducible in C++); the all-tied step-0 projection
is computed in Python and passed in. Empirically this + gemm-order ulp drift
still yielded 60/60 identical plans at steps=10; in the diverged steps=100
regime C++ vs python-fast behaves like the python-fast vs original
comparison (14/30 identical, cost-neutral: mean log10 ratio −0.08).

### Caveat: long runs

The loop is chaotic: the implementations agree to ~1e-8 per step (float32 ulp)
but reordered reductions (dense matmul vs scatter) make trajectories diverge
after roughly 30–50 steps for large queries. At `steps=100` (sizes 10–14):
15/30 plans identical; the differing ones are statistically neutral in cost
(4 better / 11 worse, median |log10 ratio| = 0.001, n.s.). Any reordering
(including `torch.compile` of the original) has the same effect. At the
production `steps=10` the output is exactly identical.

## Remaining cost in the C++ tier (n=14, ~0.14 ms/step)

Dominated by the actual gemms (fwd ~27 µs + bwd ~45 µs at AMX speed), the
Padé expm, and the per-step beam. Further gains would require changing the
algorithm (fewer steps, smaller hidden dim, cheaper acyclicity penalty) or
batching multiple queries per call — not better kernels.
