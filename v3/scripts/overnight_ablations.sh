#!/bin/bash
# ==========================================================================
# Overnight ablation + baseline chain  —  created 2026-06-13
#
# WHY: the v2 overfit model (no-rdf2vec + fanout + rel-emb + asymmetric loss
# tau=0.8) REGRESSED e2e vs old overfit-gps (17 vs 10 timeouts) despite equal
# overfit rank-acc. v2 bundles 4 changes; these ablations turn each OFF one at
# a time (overfit route, identical otherwise) to isolate the culprit. Then a
# full-data retrain of the OLD recipe (the best e2e performer so far) as the
# production baseline.
#
# After it finishes, build a pack + run the e2e for each abl-* model the same
# way overfit-gps-v2 was done (overfit_e2e_setup.build_pack + encode_pack +
# compare_v2.py) and compare timeouts to old-weighted (10) / new-weighted (17).
#
# RUN:  bash standalone/overnight_ablations.sh
# Steps are independent (no set -e) so one failure does not abort the rest.
# Rough budget: 4 x ~40min ablations + 1 full run (hours). Adjust --epochs.
# ==========================================================================
cd /Users/timschwabe/Projects/GBJOv2 || exit 1
export OMP_NUM_THREADS=8

OVERFIT="--sources sib --overfit-groups 100 --epochs 400 --batch 256 \
  --cart-weight 10 --rank-weight 3 --rank-sources sib --device cpu --lr 3e-4 \
  --encoder gps --encoder-caps 10,10,10,10"

echo "### A1  asymmetric loss OFF (tau=0.5), new features ON  $(date)"
uv run python -u standalone/train_dual.py $OVERFIT --quantile-tau 0.5 \
  --out standalone/models/abl-tau05 > standalone/abl_tau05.log 2>&1


echo "### A2  rdf2vec back ON (tau=0.8, fanout+relemb ON)  $(date)"
uv run python -u standalone/train_dual.py $OVERFIT --quantile-tau 0.8 \
  --encoder-rdf2vec --out standalone/models/abl-rdf2vec > standalone/abl_rdf2vec.log 2>&1

echo "### A3  fanout feature OFF (tau=0.8)  $(date)"
uv run python -u standalone/train_dual.py $OVERFIT --quantile-tau 0.8 \
  --encoder-no-fanout --out standalone/models/abl-nofanout > standalone/abl_nofanout.log 2>&1

echo "### A4  learnable predicate emb OFF (tau=0.8)  $(date)"
uv run python -u standalone/train_dual.py $OVERFIT --quantile-tau 0.8 \
  --encoder-no-rel-emb --out standalone/models/abl-norelemb > standalone/abl_norelemb.log 2>&1

echo "### LARGE  full-data retrain, V2 recipe + tau=0.7, at scale  $(date)"
# FULL dataset = new3,addon,cart,sib (363,061 samples), matching dual-v2; NOT
# `sib` alone (15,675). On mps (cpu would be ~days at this scale).
# V2 recipe (ablation-backed, 2026-06-14): DROP rdf2vec (redundant: +0.001 acc,
# blows up features); KEEP fanout + learned rel-emb (removing either costs
# ~0.006 overfit acc). tau=0.7 = moderate catastrophe-aversion without tau0.8's
# heavy calibration cost (huber 0.42 vs 0.17 at 0.5); MRT now does plan
# selection, so the aggressive 0.8 conservatism is no longer needed.
# Fanout uses standalone/kg_index/wikidata/pred_stats.npz (per-predicate, full KG).
# Slim encoder for full-data speed (cpu; mps OOMs from cache growth): hidden 32,
# 2 layers, 2 hops (caps 10,10 = 100 neighbors vs 10,10,10,10 = 10,000), rel-emb
# 16-dim. ~20x fewer encoder params; the 2-hop cap is the big lever.
uv run python -u standalone/train_dual.py --sources new3,addon,cart,sib --epochs 200 --batch 256 \
  --cart-weight 10 --rank-weight 3 --rank-sources sib,cart --device cpu --lr 3e-4 \
  --encoder gps --encoder-caps 10,10 --encoder-layers 2 --encoder-hidden 32 \
  --encoder-rel-emb-dim 16 --quantile-tau 0.7 \
  --out standalone/models/full-v2-tau07 > standalone/full_v2_tau07.log 2>&1

echo "### DONE  $(date)"
