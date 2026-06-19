#!/bin/bash
# ==========================================================================
# GD-search temperature sweep (deploy-params)  —  created 2026-06-13
#
# CORRECTION over the first sweep: MRT minimises E_{P~pi_theta}[C*] where
# pi_theta IS the deployed decode distribution. finetune_mrt previously used
# FastGBJO's stale defaults (init_tau 4.0, lambda_acyclic 29, lr 4.9), which
# differ sharply from the deployed pack (2.55 / 1.8 / 3.9) -- so it optimised a
# different objective and the first sweep lived in the wrong landscape. Now both
# finetune_mrt and kernel_transfer_check load the pack's deploy params, so the
# whole sweep runs in the landscape the kernel actually deploys.
#
# min_tau is a property of the SEARCH: matched train<->deploy each retrain. The
# deploy default is 0.79; sweep brackets sharper (0.49) and softer (1.2, 2.0).
# One variable; everything else fixed at the winner config (tbptt=4, gamma=0,
# lr=5e-5, sample-temp=4.0). Comparable metric = transfer-check abs mean log10 C*.
#
# RUN:  bash standalone/temp_sweep.sh
# ==========================================================================
cd /Users/timschwabe/Projects/GBJOv2 || exit 1
PACK=~/rdflib-joinordering/gbjo_pack/overfit-gps-v2
PRE=standalone/models/overfit-gps-v2/model_rank.pt
EP=http://127.0.0.1:7020/
CACHE=standalone/cstar_cache.json

for T in 0.49 0.79 1.2 2.0; do
  OUT=standalone/models/mrt-dep-tau$T
  echo "### TRAIN min_tau=$T (deploy params)  $(date)"
  PYTHONPATH=standalone uv run python -u standalone/finetune_mrt.py \
    --model $PRE --queries standalone/overfit_queries.json --pack $PACK \
    --endpoint $EP --cache $CACHE \
    --pool-samples 32 --sample-temp 4.0 --lr 5e-5 --gamma 0 --tbptt 4 \
    --epochs 30 --min-tau $T --out $OUT > standalone/sweep_dep_tau$T.log 2>&1

  echo "### TRANSFER-CHECK min_tau=$T (matched deploy)  $(date)"
  PYTHONPATH=standalone uv run python -u standalone/kernel_transfer_check.py \
    --pre $PRE --mrt $OUT/model_mrt.pt --pack $PACK \
    --endpoint $EP --cache $CACHE --min-tau $T \
    >> standalone/sweep_dep_tau$T.log 2>&1
  echo "--- min_tau=$T result ---"
  grep -E "best val regret|MRT *:|MRT better|improvement" standalone/sweep_dep_tau$T.log
done
echo "### SWEEP DONE  $(date)"
