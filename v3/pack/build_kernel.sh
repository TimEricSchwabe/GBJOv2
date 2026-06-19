#!/usr/bin/env bash
# Compile the fused GBJO C++ kernel into v3/artifacts/lib/.
#
#   bash v3/pack/build_kernel.sh [HIDDEN_DIM]
#
# HIDDEN_DIM (default 128) sets -DGBJO_H; it MUST match the model's hidden dim
# or the dual layer reads weights with the wrong stride (NaN on first forward).
# Picks Accelerate (macOS) or OpenBLAS (Linux) automatically.
set -euo pipefail

H="${1:-128}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/gbjo_kernel.cpp"
OUT_DIR="$HERE/../artifacts/lib"
mkdir -p "$OUT_DIR"

if [[ "$(uname)" == "Darwin" ]]; then
  OUT="$OUT_DIR/libgbjo.dylib"
  clang++ -O3 -std=c++17 -dynamiclib -framework Accelerate -DACCELERATE_NEW_LAPACK \
    -DGBJO_H="$H" -o "$OUT" "$SRC"
else
  OUT="$OUT_DIR/libgbjo.so"
  g++ -O3 -std=c++17 -shared -fPIC -DGBJO_H="$H" -o "$OUT" "$SRC" -lopenblas
fi
echo "built $OUT (GBJO_H=$H)"
