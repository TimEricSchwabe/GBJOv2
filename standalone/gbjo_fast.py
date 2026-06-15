"""
Standalone, fast GBJO: query in -> join plan out.

Implements exactly the algorithm of src/optimization/methods.py::GBJO
(softmax sampling, no gumbel noise, SGD+momentum with OneCycleLR,
lambda ramping, linear temperature annealing, per-step discrete beam
projection with best-plan tracking), but restructured for speed:

  * one (2n-2, n-1) logit matrix instead of an all-pairs edge list with
    -inf masking: a single row-softmax replaces the grouped scatter softmax
  * dense adjacency message passing (A^T @ H) instead of PyG MessagePassing
  * node features are constant during optimization, so sign*log1p and the
    input projection are computed once, not 2x per step
  * gradient of trace(expm(A)) is expm(A)^T -> custom autograd Function
    avoids torch's 2Nx2N Frechet-derivative backward
  * discrete plans are deduplicated and scored once after the loop
    (the in-loop scoring never feeds back into the optimization, and
    first-seen-order + strict-min selection picks the identical winner)
  * the OneCycleLR lr/momentum schedule is precomputed; the SGD update is
    two tensor ops, no optimizer/scheduler objects
  * no histories, no .item() tracking, no plotting

The beam projection (project_leftdeep_greedy_beam) is copied verbatim from
src/optimization/plan_decoder.py so tie-breaking is bit-identical.
"""

import heapq
import json
import math
import pickle
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Beam projection -- copied VERBATIM from src/optimization/plan_decoder.py
# --------------------------------------------------------------------------

@dataclass(order=True)
class _State:
    """Partial plan used by the beam search."""
    priority: float
    score:    float = field(compare=False)
    cur_join: int   = field(compare=False)
    unused_js: Set[int] = field(compare=False, repr=False)
    unused_tb: Set[int] = field(compare=False, repr=False)
    jj_edges:  List[Tuple[int, int]] = field(compare=False, repr=False)
    tj_edges:  List[Tuple[int, int]] = field(compare=False, repr=False)


def _best_two_tables(W_row: np.ndarray, tables: Set[int]) -> Tuple[List[int], float]:
    """Pick the two remaining tables with the largest weights into j0."""
    best = heapq.nlargest(2, tables, key=lambda t: W_row[t])
    return best, W_row[best].sum()


def project_leftdeep_greedy_beam(
    W: np.ndarray,
    beam_width: int = 1,
    use_product: bool = False,
) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    assert m == W.shape[1] and (m + 1) % 2 == 0, "shape must be (2n-1, 2n-1)"
    n = (m + 1) // 2

    tables = set(range(n))
    joins_all = set(range(n, 2 * n - 1))
    root = 2 * n - 2

    init_state = _State(
        priority=0.0,
        score=0.0,
        cur_join=root,
        unused_js=joins_all - {root},
        unused_tb=tables.copy(),
        jj_edges=[],
        tj_edges=[],
    )
    beam: List[_State] = [init_state]

    while beam and beam[0].unused_js:
        next_beam: List[_State] = []
        for state in beam:
            cj = state.cur_join
            for j in state.unused_js:
                for t in state.unused_tb:
                    w_t = W[t, cj]
                    w_j = W[j, cj]
                    pair_score = w_t * w_j if use_product else w_t + w_j
                    new_score = state.score + pair_score
                    new_state = _State(
                        priority=-new_score,
                        score=new_score,
                        cur_join=j,
                        unused_js=state.unused_js - {j},
                        unused_tb=state.unused_tb - {t},
                        jj_edges=state.jj_edges + [(j, cj)],
                        tj_edges=state.tj_edges + [(t, cj)],
                    )
                    heapq.heappush(next_beam, new_state)
        beam = heapq.nsmallest(beam_width, next_beam)

    best = beam[0]
    j0 = best.cur_join
    last_tables, add_score = _best_two_tables(W[:, j0], best.unused_tb)
    best.score += add_score
    best.tj_edges += [(last_tables[0], j0), (last_tables[1], j0)]

    A = np.zeros_like(W, dtype=int)
    for j_from, j_to in best.jj_edges:
        A[j_from, j_to] = 1
    for t, j in best.tj_edges:
        A[t, j] = 1
    return A


# --------------------------------------------------------------------------
# Exact fast beam: bit-identical to project_leftdeep_greedy_beam (including
# tie-breaking) but ~10x cheaper. The verbatim version copies two sets and two
# edge lists for EVERY candidate (~beam*|js|*|tb| per level) before selecting
# the top beam_width; here candidates are lightweight (priority, counter,
# parent, j, t) tuples and only the winners materialize sets. Exactness:
# - scores use the same float64 values and the same operation order
#   sc + (w_t + w_j)
# - heapq.nsmallest on (priority, counter) tuples reproduces nsmallest's
#   documented stable sorted()[:k] semantics on _State objects
# - winners' sets are built by the same `parent - {x}` operations on sets
#   with the same construction history, so set iteration order (which defines
#   candidate enumeration order, i.e. tie-breaking) is identical
# --------------------------------------------------------------------------

def beam_exact(W: np.ndarray, beam_width: int = 6) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    n = (m + 1) // 2
    root = 2 * n - 2
    WT = W.T.tolist()  # WT[c][r] == W[r, c] as python floats (same doubles)

    beam = [_State(0.0, 0.0, root, set(range(n, 2 * n - 1)) - {root},
                   set(range(n)), [], [])]

    while beam and beam[0].unused_js:
        cands = []
        ctr = 0
        for si, st in enumerate(beam):
            sc = st.score
            col = WT[st.cur_join]
            for j in st.unused_js:
                for t in st.unused_tb:
                    cands.append((-(sc + (col[t] + col[j])), ctr, si, j, t))
                    ctr += 1
        new_beam = []
        for negsc, _, si, j, t in heapq.nsmallest(beam_width, cands):
            st = beam[si]
            new_beam.append(_State(negsc, -negsc, j,
                                   st.unused_js - {j}, st.unused_tb - {t},
                                   st.jj_edges + [(j, st.cur_join)],
                                   st.tj_edges + [(t, st.cur_join)]))
        beam = new_beam

    best = beam[0]
    j0 = best.cur_join
    last_tables, _ = _best_two_tables(W[:, j0], best.unused_tb)
    tj_edges = best.tj_edges + [(last_tables[0], j0), (last_tables[1], j0)]

    A = np.zeros_like(W, dtype=int)
    for j_from, j_to in best.jj_edges:
        A[j_from, j_to] = 1
    for t, j in tj_edges:
        A[t, j] = 1
    return A


# --------------------------------------------------------------------------
# trace(expm(A)) with the cheap exact backward: d trace(expm(A))/dA = expm(A)^T
# --------------------------------------------------------------------------

class _TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        E = torch.linalg.matrix_exp(A)
        ctx.save_for_backward(E)
        return E.diagonal().sum()

    @staticmethod
    def backward(ctx, grad_out):
        (E,) = ctx.saved_tensors
        return grad_out * E.t()


def trace_expm(A):
    return _TraceExpm.apply(A)


# --------------------------------------------------------------------------
# Differentiable cartesianness penalty
#
# A join is a cartesian product iff its two subtrees share no variable.
# Sharing is exact and fixed per query (S below); only subtree membership is
# combinatorial, and the soft adjacency already parameterizes it: R from
# clamped squaring is the soft reachability (edges point child -> parent), so
# column c of R is the soft membership of c's subtree. O = R^T S R is then the
# soft number of variables shared between any two subtrees, and each pair of
# children softly assigned to the same join is penalized by exp(-gamma * O).
# --------------------------------------------------------------------------

def sharing_matrix(triples):
    """(N, N) matrix S: S[i, j] = number of variables shared by triples i, j
    (i != j); join rows/cols and the diagonal are zero."""
    n = len(triples)
    S = torch.zeros(2 * n - 1, 2 * n - 1)
    occ = {}
    for i, t in enumerate(triples):
        for atom in set(t[:3]):
            if atom.startswith("?"):
                occ.setdefault(atom, []).append(i)
    for idxs in occ.values():
        for a in idxs:
            for b in idxs:
                if a != b:
                    S[a, b] += 1.0
    return S


def cartesian_penalty(A, S, n, gamma, eps=1e-6):
    """Soft count of cartesian-product joins in the relaxed plan A.

    Per join, the pair-weighted average of exp(-gamma * overlap) over its
    soft children pairs. Normalizing by the join's own pair mass makes the
    term scale-invariant: zeroing it by concentrating a join's in-edges on a
    single child (degenerate, structurally invalid) gives no advantage, so
    only the *mix* -- which child pairs win -- carries gradient.

    For a hard valid left-deep plan this equals
    (#cartesian joins) + residual, residual <= (n-1)*exp(-gamma).
    """
    N = A.shape[0]
    M = torch.eye(N) + A
    for _ in range(max(1, math.ceil(math.log2(N)))):
        M = (M @ M).clamp(max=1.0)
    O = M.t() @ (S @ M)
    Fz = torch.exp(-gamma * O) * (1.0 - torch.eye(N))
    AJ = A[:, n:]
    num = (AJ * (Fz @ AJ)).sum(0)                       # per-join cart pair mass
    in_deg = AJ.sum(0)
    den = in_deg * in_deg - (AJ * AJ).sum(0)            # per-join total pair mass
    return (num / (den + eps)).sum()


def count_cartesian_joins(A, triples):
    """Exact number of cartesian-product joins in a hard left-deep plan."""
    order = adjacency_to_join_order(A)
    var_sets = [set(a for a in t[:3] if a.startswith("?")) for t in triples]
    count = 0
    cur = set(var_sets[order[0]])
    if not (cur & var_sets[order[1]]):
        count += 1
    cur |= var_sets[order[1]]
    for t in order[2:]:
        if not (cur & var_sets[t]):
            count += 1
        cur |= var_sets[t]
    return count


# --------------------------------------------------------------------------
# Model weights, flattened out of the CostGNNv3 module
# --------------------------------------------------------------------------

class FlatCostGNN:
    """CostGNNv3 (use_residual=True, no norms, no JK, aggr='add') as plain tensors."""

    def __init__(self, state_dict, n_layers=6):
        sd = state_dict
        self.proj_w = sd["projection.weight"]   # (H, F)
        self.proj_b = sd["projection.bias"]
        self.layers = []
        for i in range(n_layers):
            self.layers.append((
                sd[f"convs.{i}.eps"].item(),
                sd[f"convs.{i}.nn.0.weight"], sd[f"convs.{i}.nn.0.bias"],
                sd[f"convs.{i}.nn.2.weight"], sd[f"convs.{i}.nn.2.bias"],
            ))
        self.fc1_w = sd["fc1.weight"]
        self.fc1_b = sd["fc1.bias"]
        self.fc2_w = sd["fc2.weight"]
        self.fc2_b = sd["fc2.bias"]

    @classmethod
    def load(cls, model_path, n_layers=6):
        sd = torch.load(model_path, map_location="cpu")
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model_state_dict")):
            sd = sd.get("state_dict", sd.get("model_state_dict"))
        return cls(sd, n_layers=n_layers)

    def project_x(self, x):
        """The per-query constant part of the forward pass: sign*log1p + projection."""
        xl = torch.sign(x) * torch.log1p(torch.abs(x))
        return F.linear(xl, self.proj_w, self.proj_b)

    def forward_from_h0(self, h, A):
        """GNN forward given projected features h (N,H) and dense adjacency A (N,N).

        Equivalent to CostGNNv3.forward: per GIN layer
        out = sum_j A[j,i]*h[j] + (1+eps)*h[i] -> MLP -> residual add.
        """
        At = A.t()
        for eps, w1, b1, w2, b2 in self.layers:
            agg = At @ h
            z = agg + (1.0 + eps) * h
            z = F.linear(z, w1, b1)
            z = F.gelu(z)
            z = F.linear(z, w2, b2)
            h = h + z
        g = h.sum(0)
        g = F.gelu(F.linear(g, self.fc1_w, self.fc1_b))
        return F.linear(g, self.fc2_w, self.fc2_b).squeeze(-1)


# --------------------------------------------------------------------------
# Featurization: raw triples -> node feature matrix (matches src/data.py)
# --------------------------------------------------------------------------

def featurize_query(triples, rdf2vec, counts, fingerprint_dim=64, rng=None):
    """
    Build the (2n-1, 307) node feature matrix for a query.

    Matches Entity/Triple/Join.get_embedding from src/data.py plus the
    random gaussian join-node fingerprints from evaluation_parallel.py.

    Args:
        triples: list of [s, p, o] strings ("?var" or "<uri>")
        rdf2vec: dict uri -> 100-dim embedding
        counts:  dict uri -> int count
        rng:     optional torch.Generator for reproducible fingerprints
    """
    n = len(triples)
    N = 2 * n - 1

    variables = []
    seen = set()
    for t in triples:
        for atom in t[:3]:
            if atom.startswith("?") and atom not in seen:
                seen.add(atom)
                variables.append(atom)
    var_ids = list(range(len(variables)))
    if rng is not None:
        perm = torch.randperm(len(variables), generator=rng).tolist()
    else:
        perm = torch.randperm(len(variables)).tolist()
    var_id = {v: var_ids[perm[i]] for i, v in enumerate(variables)}

    def atom_embedding(atom):
        if atom.startswith("?"):
            return np.concatenate([[var_id[atom]], np.ones(100), [0]])
        name = atom[1:-1]  # strip <>
        emb = rdf2vec.get(name)
        if emb is None:
            emb = np.zeros(100)
        return np.concatenate([[0], np.asarray(emb, dtype=np.float64), [counts.get(name, 1)]])

    x = np.zeros((N, 307))
    for i, t in enumerate(triples):
        x[i] = np.concatenate([atom_embedding(t[0]), atom_embedding(t[1]),
                               atom_embedding(t[2]), [0]])
    x[n:, -1] = 1.0  # join-node flag

    x = torch.tensor(x, dtype=torch.float)

    # random gaussian fingerprints on join nodes (same as evaluation_parallel)
    fp = torch.randn(n - 1, fingerprint_dim, generator=rng)
    fp = fp / fp.norm(dim=1, keepdim=True)
    x[n:, :fingerprint_dim] = fp
    return x


# --------------------------------------------------------------------------
# Precomputed OneCycleLR (lr, momentum) schedule -- exact, taken from torch
# --------------------------------------------------------------------------

def onecycle_schedule(max_lr, total_steps):
    """Replicate OneCycleLR exactly by running the real scheduler on a dummy."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=max_lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.2, anneal_strategy="cos",
        div_factor=5.0, final_div_factor=20.0,
        cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
    )
    lrs, moms = [], []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        moms.append(opt.param_groups[0]["momentum"])
        opt.step()
        sched.step()
    return lrs, moms


# --------------------------------------------------------------------------
# The optimizer
# --------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # config_wikidata_star from src/evaluation_parallel.py __main__
    "learning_rate": 4.9,
    "lambda_acyclic": 29.0,
    "lambda_triple_in": 1.5,
    "lambda_triple_out": 1.4,
    "lambda_join_in": 3.6,
    "lambda_join_out": 4.1,
    "lambda_entropy": 0.0,
    "lambda_total_penalty": 0.99,
    "lambda_total_penalty_start": 0.0,
    "lambda_left_linear": 60.0,
    "init_tau": 4.0,
    "min_tau": 0.49,
    "use_temperature_annealing": True,
    "return_best": True,
    "use_lambda_ramping": True,
    "lambda_ramp_exponent": 1.01,
    "gradient_clip_norm": 4.7,
    "use_lr_scheduling": True,
    "discrete_beam_width": 6,
    # cartesianness penalty (off by default; needs share= passed to optimize)
    "lambda_cartesian": 0.0,
    "cartesian_gamma": 5.0,
}


class FastGBJO:
    def __init__(self, model: FlatCostGNN, params=None, compile_step=False):
        self.model = model
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self._size_cache = {}     # n -> (mask, N)
        self._sched_cache = {}    # steps -> (lrs, moms)
        self._beam0_cache = {}    # (n, beam_width) -> step-0 discrete plan
        self._loss_fn = self._loss
        if compile_step:
            self._loss_fn = torch.compile(self._loss, dynamic=False)

    # -- per-size constants ------------------------------------------------
    def _size_consts(self, n):
        c = self._size_cache.get(n)
        if c is None:
            N = 2 * n - 1
            # logits L: rows 0..n-1 = triple sources, rows n..2n-3 = non-root join
            # sources; columns = join nodes n..2n-2. Join self-loops masked -inf.
            mask = torch.zeros(2 * n - 2, n - 1)
            for r in range(n, 2 * n - 2):
                mask[r, r - n] = float("-inf")
            c = (mask, N)
            self._size_cache[n] = c
        return c

    def _schedule(self, steps):
        s = self._sched_cache.get(steps)
        if s is None:
            p = self.params
            if p["use_lr_scheduling"]:
                lrs, moms = onecycle_schedule(p["learning_rate"], steps)
            else:
                lrs = [p["learning_rate"]] * steps
                moms = [0.9] * steps
            taus, lts = [], []
            for step in range(steps):
                if p["use_temperature_annealing"]:
                    taus.append(max(p["min_tau"], p["init_tau"] -
                                    (p["init_tau"] - p["min_tau"]) * (step / steps)))
                else:
                    taus.append(p["init_tau"])
                if p["use_lambda_ramping"]:
                    frac = min(1.0, step / steps)
                    lts.append(p["lambda_total_penalty_start"] +
                               (frac ** p["lambda_ramp_exponent"]) *
                               (p["lambda_total_penalty"] - p["lambda_total_penalty_start"]))
                else:
                    lts.append(p["lambda_total_penalty"])
            s = (lrs, moms, taus, lts)
            self._sched_cache[steps] = s
        return s

    # -- differentiable loss (compilable; acyclicity term added outside) ----
    def _loss(self, L, mask, h0, inv_tau, n, N, lam_ti, lam_to, lam_ji, lam_jo, lam_ll):
        W = torch.softmax((L + mask) * inv_tau, dim=1)        # (2n-2, n-1)
        A = W.new_zeros(N, N)
        A[: 2 * n - 2, n:] = W                                # root row & triple cols stay 0

        cost = self.model.forward_from_h0(h0, A)

        in_deg = A.sum(0)
        out_deg = A.sum(1)
        root = N - 1

        P_triple_in = (in_deg[:n] ** 2).sum()
        P_triple_out = ((out_deg[:n] - 1) ** 2).sum()
        P_join_in = ((in_deg[n:] - 2) ** 2).sum()
        P_join_out = ((out_deg[n:root] - 1) ** 2).sum() + out_deg[root] ** 2

        child_triple = A[:n, n:].sum(0)
        child_join = A[n:, n:].sum(0)
        P_first = (child_triple[0] - 2) ** 2 + child_join[0] ** 2
        if n > 2:
            P_left_linear = (P_first + ((child_triple[1:] - 1) ** 2).sum()
                             + ((child_join[1:] - 1) ** 2).sum())
        else:
            P_left_linear = P_first

        penalty = (lam_ti * P_triple_in + lam_to * P_triple_out +
                   lam_ji * P_join_in + lam_jo * P_join_out +
                   lam_ll * P_left_linear)
        return cost, penalty, A

    # -- main entry ---------------------------------------------------------
    def optimize(self, x, optimization_steps=10, share=None, triples=None):
        """
        Args:
            x: (2n-1, 307) node feature matrix (with fingerprints already set)
            share: optional (2n-1, 2n-1) sharing_matrix(triples); enables the
                cartesianness penalty when lambda_cartesian != 0
            triples: optional [[s,p,o],...]; enables the connected-greedy
                injection in the rich deploy pool (params["rich_pool_inject"]).
        Returns:
            (final_A int8 numpy (2n-1, 2n-1), predicted_cost float)

        Rich deploy pool (off by default -> identical to the C++ kernel):
        params["rich_pool_inject"]=True adds the connected-greedy cartesian-free
        plan; params["rich_pool_samples"]=K adds K Gumbel-max draws from the
        pulled-back final policy (logits / params["rich_pool_temp"]). Both feed
        the same arg-min-predicted-cost selection below. Recovers ~0.8 OOM true
        cost over the per-step trajectory decode (mostly from the injection).
        """
        p = self.params
        n = (x.shape[0] + 1) // 2
        mask, N = self._size_consts(n)
        lrs, moms, taus, lts = self._schedule(optimization_steps)

        if getattr(self.model, "is_dual", False):
            if share is None:
                raise ValueError("dual model requires share=sharing_matrix(triples)")
            self.model.bind_share((share > 0).float())

        with torch.no_grad():
            h0 = self.model.project_x(x)

        L = torch.zeros(2 * n - 2, n - 1, requires_grad=True)
        buf = torch.zeros_like(L)

        lam_ac = p["lambda_acyclic"]
        lam_ent = p["lambda_entropy"]
        lam_cart = p["lambda_cartesian"]
        gamma_cart = p["cartesian_gamma"]
        clip = p["gradient_clip_norm"]
        beam_width = p["discrete_beam_width"]
        return_best = p["return_best"]

        seen = {}        # A.tobytes() -> (first_step, A_int)
        order = []

        for step in range(optimization_steps):
            inv_tau = 1.0 / taus[step]
            lt = lts[step]

            cost, penalty, A = self._loss_fn(
                L, mask, h0, inv_tau, n, N,
                p["lambda_triple_in"], p["lambda_triple_out"],
                p["lambda_join_in"], p["lambda_join_out"], p["lambda_left_linear"],
            )
            P_acyclic = trace_expm(A) - N
            if lam_ent != 0.0:
                w = A[: 2 * n - 2, n:].reshape(-1)
                probs = w.clamp(min=1e-10)
                penalty = penalty + lam_ent * (-(probs * torch.log(probs)).sum())
            loss = cost + lt * (penalty + lam_ac * P_acyclic)
            if lam_cart != 0.0 and share is not None:
                # not ramped: cartesianness is known a priori and defines the
                # feasible basin, so it gets full authority from step 0
                loss = loss + lam_cart * cartesian_penalty(A, share, n, gamma_cart)
            loss.backward()

            if return_best:
                A_np = A.detach().numpy()
                if step == 0:
                    # step-0 logits are all zero -> A is the same uniform matrix
                    # for every query of size n, so its projection can be cached
                    A_hat = self._beam0_cache.get((n, beam_width))
                    if A_hat is None:
                        A_hat = beam_exact(A_np, beam_width=beam_width)
                        self._beam0_cache[(n, beam_width)] = A_hat
                else:
                    A_hat = beam_exact(A_np, beam_width=beam_width)
                key = A_hat.tobytes()
                if key not in seen:
                    seen[key] = A_hat
                    order.append(key)

            with torch.no_grad():
                g = L.grad
                # torch.nn.utils.clip_grad_norm_ semantics
                total_norm = g.norm(2)
                if clip > 0:
                    coef = clip / (total_norm + 1e-6)
                    if coef < 1.0:
                        g.mul_(coef)
                # SGD with momentum (dampening 0, nesterov False)
                buf.mul_(moms[step]).add_(g)
                L.add_(buf, alpha=-lrs[step])
                L.grad = None

        # score the deduplicated discrete candidates; first-seen + strict '<'
        # reproduces the sequential best-tracking of the original loop
        if not order:  # return_best=False fallback: project the final soft plan
            with torch.no_grad():
                inv_tau = 1.0 / taus[-1]
                _, _, A = self._loss_fn(
                    L, mask, h0, inv_tau, n, N,
                    p["lambda_triple_in"], p["lambda_triple_out"],
                    p["lambda_join_in"], p["lambda_join_out"], p["lambda_left_linear"],
                )
                A_hat = beam_exact(A.numpy(), beam_width=beam_width)
                order.append(A_hat.tobytes())
                seen[order[0]] = A_hat

        # optional rich deploy pool: the per-step trajectory modes are already in
        # `order`; add the connected-greedy plan + Gumbel draws from the
        # pulled-back final policy, then the arg-min loop below picks over all.
        rich_k = int(p.get("rich_pool_samples", 0))
        rich_inject = bool(p.get("rich_pool_inject", False))
        if rich_k > 0 or rich_inject:
            def _add(A_hat):
                k = A_hat.tobytes()
                if k not in seen:
                    seen[k] = A_hat
                    order.append(k)

            if rich_inject and triples is not None and share is not None:
                Snp = share[:n, :n]
                Snp = (Snp > 0).numpy() if torch.is_tensor(Snp) else (np.asarray(Snp) > 0)
                conn = connected_order(triples, Snp)
                if conn is not None:
                    _add(order_to_adjacency(conn, n))
            if rich_k > 0:
                inv_tau_f = 1.0 / taus[-1]
                scores = ((L + mask) * inv_tau_f).detach().numpy()  # (2n-2, n-1)
                flat = scores / float(p.get("rich_pool_temp", 4.0))
                rng = np.random.default_rng(0)
                for _ in range(rich_k):
                    gmb = -np.log(-np.log(rng.uniform(size=flat.shape) + 1e-12) + 1e-12)
                    Afull = np.zeros((N, N), dtype=np.float32)
                    Afull[: 2 * n - 2, n:] = flat + gmb
                    _add(beam_exact(Afull, beam_width=beam_width))

        best_cost = float("inf")
        best_A = None
        with torch.no_grad():
            for key in order:
                A_hat = seen[key]
                A_t = torch.tensor(A_hat, dtype=torch.float32)
                log_cost = self.model.forward_from_h0(h0, A_t).item()
                if log_cost < best_cost:
                    best_cost = log_cost
                    best_A = A_hat

        self.last_candidates = [seen[k] for k in order]  # diagnostics
        self.last_L = L.detach().numpy().copy()           # diag: final logits
        return best_A, float(np.exp(best_cost))


# --------------------------------------------------------------------------
# Connected-greedy plan (for the rich deploy pool); mirrors finetune_mrt
# --------------------------------------------------------------------------

def connected_order(triples, S):
    """Greedy cartesian-free left-deep order: most-bound triple first, then
    always extend by a triple sharing a variable. None if disconnected."""
    n = len(triples)
    n_const = [sum(not a.startswith("?") for a in t[:3]) for t in triples]
    start = max(range(n), key=lambda i: n_const[i])
    order, left = [start], set(range(n)) - {start}
    while left:
        nxt = next((t for t in sorted(left)
                    if any(S[t, u] > 0 for u in order)), None)
        if nxt is None:
            return None
        order.append(nxt)
        left.remove(nxt)
    return order


def order_to_adjacency(order, n):
    """Left-deep join order -> discrete (2n-1, 2n-1) adjacency A_hat."""
    N = 2 * n - 1
    A = np.zeros((N, N), dtype=np.int8)
    A[order[0], n] = 1
    A[order[1], n] = 1
    for k in range(2, n):
        j = n + k - 1
        A[n + k - 2, j] = 1
        A[order[k], j] = 1
    return A


# --------------------------------------------------------------------------
# Plan extraction: adjacency -> join order
# --------------------------------------------------------------------------

def adjacency_to_join_order(A):
    """Return the left-deep join order [t_first, t_second, ...] encoded in A."""
    N = A.shape[0]
    n = (N + 1) // 2
    A = np.asarray(A)
    # join chain: join j feeds join A[j] -> build child->parent, walk from base
    triple_child = {j: [t for t in range(n) if A[t, j]] for j in range(n, N)}
    join_child = {j: [k for k in range(n, N) if A[k, j]] for j in range(n, N)}
    base = next(j for j in range(n, N) if len(triple_child[j]) == 2)
    order = list(triple_child[base])
    cur = base
    parent = {c: j for j in range(n, N) for c in join_child[j]}
    while cur in parent:
        cur = parent[cur]
        order.extend(triple_child[cur])
    return order


# --------------------------------------------------------------------------
# CLI: query in -> plan out
# --------------------------------------------------------------------------

def main():
    import argparse, time

    ap = argparse.ArgumentParser(description="Standalone fast GBJO")
    ap.add_argument("--model", required=True, help="path to model.pt (CostGNNv3)")
    ap.add_argument("--embeddings", required=True, help="dir with rdf2vec100dim.pkl + counts.pkl")
    ap.add_argument("--query", required=True,
                    help='JSON file or inline JSON: list of [s,p,o] triples')
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)

    if args.query.strip().startswith("["):
        triples = json.loads(args.query)
    else:
        with open(args.query) as f:
            triples = json.load(f)
        if isinstance(triples, dict) and "triples" in triples:
            triples = triples["triples"]

    with open(f"{args.embeddings}/rdf2vec100dim.pkl", "rb") as f:
        rdf2vec = pickle.load(f)
    with open(f"{args.embeddings}/counts.pkl", "rb") as f:
        counts = pickle.load(f)

    model = FlatCostGNN.load(args.model)
    gbjo = FastGBJO(model, compile_step=args.compile)

    rng = None
    if args.seed is not None:
        rng = torch.Generator().manual_seed(args.seed)
    x = featurize_query(triples, rdf2vec, counts, rng=rng)

    gbjo.optimize(x, optimization_steps=2)  # warm torch kernels / lazy inits

    t0 = time.perf_counter()
    A, pred_cost = gbjo.optimize(x, optimization_steps=args.steps)
    dt = time.perf_counter() - t0

    print(json.dumps({
        "join_order": adjacency_to_join_order(A),
        "predicted_cost": pred_cost,
        "optimize_seconds": dt,
    }, indent=2))


if __name__ == "__main__":
    main()
