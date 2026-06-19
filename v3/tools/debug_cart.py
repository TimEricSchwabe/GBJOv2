"""Per-step diagnostics for the cartesianness penalty on one path query."""
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from v3.core.gbjo_fast import (FastGBJO, FlatCostGNN, trace_expm, cartesian_penalty)
from v3.core.featurize import PATH_PARAMS, MODEL, build_query_set

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
GAMMA = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 11

items = build_query_set([5, 8, 11, 14], 25)
it = next(i for i in items if i["size"] == SIZE)
x, share = it["x"], it["share"]
n = (x.shape[0] + 1) // 2
N = 2 * n - 1

flat = FlatCostGNN.load(MODEL)
gbjo = FastGBJO(flat, params={**PATH_PARAMS, "lambda_cartesian": LAM,
                              "cartesian_gamma": GAMMA})
mask, _ = gbjo._size_consts(n)
lrs, moms, taus, lts = gbjo._schedule(STEPS)
with torch.no_grad():
    h0 = flat.project_x(x)

L = torch.zeros(2 * n - 2, n - 1, requires_grad=True)
buf = torch.zeros_like(L)
p = gbjo.params

print(f"n={n} steps={STEPS} lam={LAM} gamma={GAMMA}")
print(f"{'step':>4} {'tau':>5} {'lt':>5} {'P_cart':>8} {'gPcart':>8} "
      f"{'gCost':>8} {'gPen':>8} {'gAcyc':>8} {'Fz_mean':>8} {'R_mean':>7}")
for step in range(STEPS):
    inv_tau = 1.0 / taus[step]
    cost, penalty, A = gbjo._loss(
        L, mask, h0, inv_tau, n, N,
        p["lambda_triple_in"], p["lambda_triple_out"],
        p["lambda_join_in"], p["lambda_join_out"], p["lambda_left_linear"])
    P_ac = trace_expm(A) - N
    P_cart = cartesian_penalty(A, share, n, GAMMA)

    # per-term gradient norms w.r.t. L
    gs = {}
    for name, term in (("cost", cost), ("pen", penalty), ("ac", P_ac),
                       ("cart", P_cart)):
        g = torch.autograd.grad(term, L, retain_graph=True)[0]
        gs[name] = g.norm().item()

    # Fz / R stats (recompute pieces)
    with torch.no_grad():
        M = torch.eye(N) + A
        for _ in range(max(1, math.ceil(math.log2(N)))):
            M = (M @ M).clamp(max=1.0)
        O = M.t() @ (share @ M)
        Fz = torch.exp(-GAMMA * O) * (1.0 - torch.eye(N))

    loss = (cost + lts[step] * (penalty + p["lambda_acyclic"] * P_ac)
            + LAM * P_cart)
    g = torch.autograd.grad(loss, L)[0]
    with torch.no_grad():
        in_deg = A.sum(0)[n:]
        pair_mass = (A[:, n:] * ((1.0 - torch.eye(N)) @ A[:, n:])).sum().item()
    print(f"{step:>4} {taus[step]:>5.2f} {lts[step]:>5.2f} "
          f"{P_cart.item():>8.4f} {LAM*gs['cart']:>8.4f} "
          f"{gs['cost']:>8.3f} {lts[step]*gs['pen']:>8.3f} "
          f"{lts[step]*p['lambda_acyclic']*gs['ac']:>8.3f} "
          f"{Fz.mean().item():>8.5f} {M.mean().item():>7.3f} "
          f"indeg[{in_deg.min().item():.2f},{in_deg.max().item():.2f}] "
          f"pm={pair_mass:.2f}")

    with torch.no_grad():
        total_norm = g.norm(2)
        coef = p["gradient_clip_norm"] / (total_norm + 1e-6)
        if coef < 1.0:
            g = g * coef
        buf.mul_(moms[step]).add_(g)
        L.add_(buf, alpha=-lrs[step])
