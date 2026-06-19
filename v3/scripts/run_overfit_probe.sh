#!/bin/bash
# Capacity probe: can each variant overfit 100 hard (sib, size-5, cartesian-
# contrast) query groups? Same data + recipe; only the encoder differs.
# Read the floor of val_* in each models/overfit-*/history.json (val == train).
set -e
cd /Users/timschwabe/Projects/GBJOv2
export OMP_NUM_THREADS=4   # gentle: dual-enc-v1 may still be running

COMMON="--sources sib --overfit-groups 100 --epochs 400 --batch 256 \
  --cart-weight 10 --rank-weight 3 --rank-sources sib --device cpu --lr 3e-4"
CAPS="--encoder-caps 10,10,10,10"

echo "######## rdf2vec (encoder off) ########"
uv run python -u standalone/train_dual.py $COMMON --encoder off \
  --out standalone/models/overfit-off

echo "######## GINE encoder ########"
uv run python -u standalone/train_dual.py $COMMON --encoder gine $CAPS \
  --out standalone/models/overfit-gine

echo "######## GPS encoder ########"
uv run python -u standalone/train_dual.py $COMMON --encoder gps $CAPS \
  --out standalone/models/overfit-gps

echo "######## DONE ########"
