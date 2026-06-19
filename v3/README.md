# GBJO v3 — learned SPARQL join-order optimizer

`v3` is the learned BGP join-order optimizer: a **CostGNNDual** cost model + an
optional **GPS term-encoder**, searched by a differentiable, unrolled
gradient descent over a soft join adjacency, and deployed as a **torch-free
numpy/ctypes runtime** driving a fused **C++ kernel**. It extends the original
implementation under `../src/`.

**Offline / online split.** PyTorch is used *only* for
data generation, pretraining, MRT fine-tuning, and packing. The deployed runtime
(in `~/rdflib-joinordering`) is **numpy + ctypes only**: it loads a *pack*
directory and calls the compiled kernel. The term encoder runs
at train + pack time and writes a frozen `emb.npy`; the runtime never re-encodes.
So shipping a model change means: (a) repack `model.npz`, (b) re-encode
`emb.npy`, (c) drop in the `.dylib`.

## Package layout

```
v3/
  core/      gbjo_fast, gbjo_cpp, model_dual, term_encoder, kg_index, dual_data, subcost, featurize
  datagen/   gen_cartesian_plans, gen_sibling_plans, mine_subcosts, gen_star_mrt_queries
  train/     train_dual, train_dual_card, train_dual_mc
  mrt/       finetune_mrt, kernel_transfer_check, score_plans
  pack/      encode_pack(_mc), build_deploy_packs, overfit_e2e_setup, gbjo_kernel.cpp, build_kernel.sh
  eval/      validate_dual, validate_cpp_dual, bench_compare, true_cost
  tools/     pred_stats, export_bound_stats, inspect_datasets, kernel_bisect, …
  paths.py   single source of truth for all locations
  docs/      EXPERIMENT_PLAN.md (full pipeline reference + run ledger), MRT_FINETUNING.md, NEIGHBORHOOD_SAMPLING.md
  artifacts/ (gitignored) models/ index/ queries/ plans/ cache/ stats/ logs/ lib/
```

Every module is run as `python -m v3.<area>.<module>`. `v3/paths.py` anchors all
artifact locations to the package, so commands work from any CWD (the examples
below assume `cd ~/Projects/GBJOv2`).

---

## Installation

```bash
cd ~/Projects/GBJOv2
uv sync                 # editable-installs the `v3` package into the venv
```

Requires Python ≥3.11 (`.python-version` is pinned to 3.12). Then:

```bash
uv run python -c "import v3.core.gbjo_fast, v3.mrt.finetune_mrt; print('ok')"
```

You also need, depending on stage:

- **A QLever endpoint** serving the target KG on `http://127.0.0.1:7020/` — the
  true-cost oracle for data generation (§1) and MRT (§3). Counts are cached to
  disk, so it is only hit on cache misses.
- **A C++ toolchain** for the kernel (§4): macOS `clang++` + Accelerate, Linux
  `g++` + OpenBLAS (`apt install libopenblas-dev`).

---

## Pipeline Overview

```
 (KG .nt) ──kg_index/pred_stats──┐
 (path queries + rdf2vec) ──┐    │
                            ▼    ▼
 §1 gen plan data ──► §2 pretrain ──► §3 MRT finetune ──► §4 pack ──► deploy
   datagen/*           train/*          mrt/*               pack/*     (rdflib repo)
   → plans/*.pt        → model_rank.pt   → model_mrt.pt      → pack/    → GBJOPlanner
```

Run top→bottom. Each stage names what it produces; `docs/EXPERIMENT_PLAN.md` has
the exact tensor shapes and the full run ledger.

---

## 1. Generate plan data

Supervised `(plan, exact-C*)` samples over path queries. **No cartesian is ever
executed** — a plan's cost is the sum of prefix join cardinalities, and each
prefix is factored into connected components whose counts come from QLever
(`|A×B| = |A|·|B|`), cached per query. Inputs: a path-query pool
(`data/queries/wikidata/path/path_queries.json`) and rdf2vec embeddings
(`data/embeddings/wikidata`).

**Regression plans** (cartesian generator, sizes 9–15) — 3 plans/query: a greedy
connected (cartesian-free) order, a corrupted-connected order with one early
cartesian, and a random permutation:

```bash
uv run python -m v3.datagen.gen_cartesian_plans --per-size 200 \
  --sizes 9 10 11 12 13 14 15 --out v3/artifacts/plans/cart_plans_9_15
```

**Sibling ranking families** — per query a greedy base order, near-identical
*sibling* orders (alternative connected next-triple at sampled depths, sharing
the prefix), *cart* twins (one inserted cartesian), and random permutations.
Plans of one query form a ranking group (recovered by their shared `triples`
string), which the ranking loss consumes:

```bash
uv run python -m v3.datagen.gen_sibling_plans --per-size 150 \
  --sizes 5 6 7 8 9 10 11 12 13 14 15 --out v3/artifacts/plans/sib_plans
```

Both write `<out>/dataset.pt` = `{'data': [PyG Data], 'triples': [...]}`, with
`y` = exact log C_out and `x` **without** fingerprints (training adds them).
The base path-plan sets `new3` / `addon` (under
`data/plans/wikidata_path_plan_datasets_training/`) are pre-existing.

**Optional — per-join subcosts** for the auxiliary intermediate-cardinality loss
(`train_dual_card`): mine the per-join cardinalities into a shared cache (so far we found this didnt improve model performance ).

```bash
uv run python -u -m v3.datagen.mine_subcosts --sources cart,sib,addon \
  --endpoint http://127.0.0.1:7020/ --threads 8     # -> v3/artifacts/cache/subcost_cache.json
```

**Encoder prerequisite — KG index + fan-out stats.** The FICE encoder samples
each term's factor-graph neighborhood from a KG index, and the (optional)
fan-out feature needs per-predicate degree stats. Build both once per KG (the
index mmaps, so training stays fast):

```bash
uv run python -m v3.core.kg_index  --nt path/to/kg.nt --out v3/artifacts/index/<kg>
uv run python -m v3.tools.pred_stats --nt path/to/kg.nt --out v3/artifacts/index/<kg>
```

---

## 2. Pretrain

Supervised cost regression (quantile-Huber) + FICE pairwise ranking. Pretraining
is **schedule-agnostic** (it never runs the GD search), so deploy GD params don't
matter here. **Use `--device cpu`, `--dropout 0`, and watch `val_rank_acc`** —
dropout breaks the ranking head; regularize with weight-decay.

```bash
OMP_NUM_THREADS=8 uv run python -u -m v3.train.train_dual \
  --sources new3,addon,cart,sib --rank-sources sib,cart --rank-weight 3 \
  --cart-weight 10 --quantile-tau 0.6 --epochs 200 --batch 256 --lr 3e-4 \
  --device cpu --encoder gps --encoder-caps 10,10,10,10 \
  --encoder-layers 2 --encoder-hidden 32 --encoder-rel-emb-dim 16 \
  --hidden 32 --layers 3 --weight-decay 1e-4 --dropout 0.0 \
  --kg-index v3/artifacts/index/wikidata \
  --out v3/artifacts/models/<run>
```

Produces in `v3/artifacts/models/<run>/`:

- `model_rank.pt` + `encoder_rank.pt` — best **val rank accuracy** ← **the MRT
  seed** (ranking is what the search + MRT consume).
- `model.pt` / `encoder.pt` — best val median q-error; `*_last.pt` every epoch.
- `config.json` (all args + `kg_index`), `history.json`.

Key flags: `--encoder {off,gine,gps}` (`off` = rdf2vec baseline, no KG index
needed; **keep `layers ≥ hops`** in `--encoder-caps`); `--quantile-tau` (>0.5
penalizes under-pricing → catastrophe-averse); `--encoder-no-{fanout,rel-emb}`,
`--encoder-rdf2vec` opt-outs. test it like:

```bash
uv run python -u -m v3.train.train_dual --sources sib --overfit-groups 5 \
  --epochs 1 --encoder gps --encoder-caps 4,4 --encoder-layers 2 \
  --hidden 32 --layers 3 --device cpu --kg-index v3/artifacts/index/<kg> \
  --encoder-no-fanout --out /tmp/dry
```

**Variants.** `train_dual_mc` adds Monte-Carlo neighborhood sampling for the
encoder (`--mc-m`, `--reweight`, `--sampler`; used by the star models);
`train_dual_card` adds the auxiliary per-join cardinality loss (needs
`mine_subcosts` first).

---

## 3. MRT fine-tuning

On-policy **minimum-risk training**: treat the unrolled-GD + beam decode as a
policy `π_θ(plan | query)` and minimize **expected true cost** (QLever oracle).
Only the **decoder** is fine-tuned; the encoder stays frozen (so the kernel /
runtime are unchanged — just repack `model.npz`). C* is cached by pattern-set
hash in `v3/artifacts/cache/cstar_cache.json`.

First **encode the `emb.npy` the run decodes under** (reuse the wikidata pack's
`keys/counts/meta`, regenerate only `emb`):

`encode_pack` rebuilds the encoder from the run's `config.json`, writes `emb`
aligned to `--pack-in`'s keys, and **makes `--out-emb`'s directory a
self-contained src pack** (symlinks `keys`/`counts`, copies `meta` from
`--pack-in`) plus a `deploy.json` in the run dir — no manual `mkdir`/`cp`:

```bash
PK=~/rdflib-joinordering/gbjo_pack/<run>-mrtsrc
OMP_NUM_THREADS=4 uv run python -m v3.pack.encode_pack \
  --train-out v3/artifacts/models/<run> --encoder-file encoder_rank.pt \
  --queries v3/artifacts/queries/mrt_queries_path20k.json \
  --pack-in ~/rdflib-joinordering/gbjo_pack/wikidata --out-emb $PK/emb.npy
```

Then fine-tune **under the pack's deploy params** (`meta.json` — *not* the
defaults.). Recommended: TBPTT k=4, no trust
region:

```bash
OMP_NUM_THREADS=1 uv run python -u -m v3.mrt.finetune_mrt \
  --model v3/artifacts/models/<run>/model_rank.pt \
  --queries v3/artifacts/queries/mrt_queries_path20k.json --pack $PK \
  --endpoint http://127.0.0.1:7020/ --cache v3/artifacts/cache/cstar_cache.json \
  --pool-samples 32 --sample-temp 4.0 --lr 5e-5 --gamma 0 --tbptt 4 \
  --timeout 10 --epochs 40 --out v3/artifacts/models/<run>-mrt
```

Produces `<run>-mrt/model_mrt.pt` (best val cost), `search_best.json` (learned
λ + `inner_lr_scale`), `deploy.json` (the self-contained manifest `pack --config`
consumes in §4), and the live `mrt_progress.png` dashboard. The MRT query file
**must be pre-shuffled** (val is the deterministic prefix). For a STAR model
build a star query set with `v3.datagen.gen_star_mrt_queries`.

**Cheap evaluation of performance against sparql endpoint:** — deterministic decode, true C*, pretrained
vs MRT:

```bash
uv run python -m v3.mrt.kernel_transfer_check \
  --pre v3/artifacts/models/<run>/model_rank.pt \
  --mrt v3/artifacts/models/<run>-mrt/model_mrt.pt --pack $PK \
  --endpoint http://127.0.0.1:7020/ --cache v3/artifacts/cache/cstar_cache.json
```

---

## 4. Pack to C++

Turn a trained decoder into a deployable pack. `build_deploy_packs` repacks the
torch decoder → `model.npz`, ships the kernel **compiled for this model's hidden
dim** (`-DGBJO_H`, cached), copies `emb`/`keys`/`counts`/`meta` from a source
pack, writes the OneCycle schedule, and auto-detects `n_layers`/`H`. It is
**config-driven — no source edits**.

Common case: the MRT run already wrote a `deploy.json` (model + src-pack +
learned search params), so packing is one flag — `--config` folds the learned
`inner_lr_scale` (→ `learning_rate`) and λ into `meta.params` automatically:

```bash
uv run python -m v3.pack.build_deploy_packs \
  --config v3/artifacts/models/<run>-mrt/deploy.json --out <run>-mrt-deploy
```

**Pretrain-only (no MRT)** is just as easy — `encode_pack` already wrote a
`deploy.json` in the run dir (pointing at `model_rank.pt` + the src pack), so:

```bash
uv run python -m v3.pack.build_deploy_packs \
  --config v3/artifacts/models/<run>/deploy.json --out <run>-pre-deploy
```

Fallback for an ad-hoc decoder with no manifest — point at the model and an
encoded source pack directly (`--search-best <search_best.json>` optional, to
fold learned params):

```bash
uv run python -m v3.pack.build_deploy_packs \
  --model v3/artifacts/models/<run>/model_rank.pt \
  --src ~/rdflib-joinordering/gbjo_pack/<run>-mrtsrc --out <run>-pre-deploy
```

`--out` is a name under `gbjo_pack/` or an absolute path (default:
`<model-run-dir>-deploy`). To compile the kernel standalone (output →
`v3/artifacts/lib/`):

```bash
bash v3/pack/build_kernel.sh 32        # arg = hidden dim; MUST match the model's H
```

A complete pack (`~/rdflib-joinordering/gbjo_pack/<name>/`) holds:

| file | meaning |
|---|---|
| `emb.npy` `(V,100)` | term embeddings (rdf2vec or FICE-encoded), row ↔ `keys.txt[i]` |
| `keys.txt`, `counts.npy` | URI → row index, per-key occurrence count |
| `model.npz` | frozen CostGNNDual weights (dual: `W1` is `H×2H`) |
| `schedule.npz` | OneCycle per-step lr/momentum |
| `meta.json` | deploy GD params, `n_layers`, `emb_dim`, provenance |
| `libgbjo.<ext>` | kernel compiled for this model's `H` |

> ⚠ **The kernel fixes `H` at compile time.** A binary built for a different `H`
> reads weights with the wrong stride → **NaN on the first forward → garbage
> plans, identical for every model**. `build_deploy_packs` ships the H-matched
> dylib; always validate a new pack on the deployed `GBJOPlanner` (the kernel),
> not only `kernel_transfer_check` (FastGBJO reads weights by true shape and
> hides this).

**The C++ kernel** (`gbjo_kernel.cpp` + `gbjo_cpp.py`) is a fully fused unroll —
hand-derived backward w.r.t. the adjacency only, Padé-13 `expm`, bitmask beam —
needing only CBLAS + LAPACK `dgesv`. It is **13–67× faster than the original and
bit-identical at the production `steps=10`** (60/60 plans). `v3.eval.bench_compare`
reproduces the equivalence + speed comparison against `../src`.

---

## 5. Deploy (in `~/rdflib-joinordering`)

The torch-free runtime loads a pack and installs a BGP join-order hook:

```python
from rdflib_joinordering.gbjo import GBJOPlanner, gbjo_join_optimizer
planner = GBJOPlanner.load("gbjo_pack/<run>-deploy",
                           inject=True, gumbel_k=8, gumbel_temp=4.0, select="kernel")
with gbjo_join_optimizer(planner=planner):
    results = list(graph.query(sparql))     # BGPs of size 3..15 get reordered
```

Benchmark with `compare_val.py` / `compare_mrt.py` in that repo.

---

## Notes

- **Offline encoding only** — encoder runs at train + pack time; the runtime/
  kernel never re-encode. Ship a model = repack `model.npz` + re-encode `emb.npy`
  + copy the H-matched `.dylib`.
- **MRT uses deploy params** from the pack `meta.json`, not `DEFAULT_PARAMS`.
- **Use `model_rank.pt`** (best rank-acc) as the MRT seed, not `model.pt`.
- **Pretrain with `--dropout 0`** (dropout breaks the ranking head) and watch
  `val_rank_acc`, not `val_huber`.
- **`encoder_rank.pt` pairs with `model_rank.pt`** (same epoch) — encode
  `emb.npy` with the encoder that matches the packed decoder.
- **`layers ≥ hops`** in the encoder; **per-term subgraph seeded by node id** so
  embeddings match at train and pack time.
- **MRT query file must be pre-shuffled** (val = deterministic prefix).
- **CPU > MPS** for this workload (MPS dense-attention OOMs).
- `x` is width **307** = 3 slots × (1 var-id + 100 emb + 1 count) + 1 join-flag;
  GD `steps=10`.