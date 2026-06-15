"""
On-policy MRT (minimum-risk training) fine-tuning of the GBJO decoder, to
actively shape the continuous optimization landscape toward plans that are
cheap in TRUE cost -- measured live against a QLever endpoint.

Idea (see EXPERIMENT_PLAN "search-shaping"):
  The pretrained decoder defines a landscape; the 10-step unrolled gradient
  descent rides it and a beam decodes discrete plans. We treat the whole
  pipeline as a POLICY pi_theta(P | query) over discrete plans (the factorized
  softmax over the final soft adjacency), and minimize the expected TRUE cost.
  We never differentiate the (non-diff) decode or the (black-box) true cost;
  we differentiate the plan's log-likelihood
        log pi_theta(P) = sum_r log W[r, c_r]
  through the unroll into the decoder params. Fixing the candidate set to the
  decoded beam and reweighting it makes the gradient a low-variance
  differentiable weighted sum (minimum-risk training, Shen et al. 2016) -- no
  REINFORCE score-function estimator.

Loss per query (over the on-policy beam pool {P_k} with true costs C*_k):
    L_mrt  = sum_k softmax_k(alpha * logpi_k) * regret_k      # shape landscape
    L_rank = sum_{i<j, C*_i<C*_j} softplus(pred_i - pred_j)   # shape selection
    L      = L_mrt + beta*L_rank + gamma * ||theta - theta0||^2  (trust region)
  regret_k = log10 C*_k - min_j log10 C*_j  (>=0; censored plans -> sentinel).

True cost: C*(left-deep order) = sum over prefixes k=2..n of card(join of the
first k triples). A cartesian prefix is split into connected components and the
per-component counts MULTIPLIED (|A x B| = |A|.|B|), so QLever never
materializes a cross product. Counts are cached on disk by the component's
pattern-set hash (re-used across prefixes, plans, queries and runs).

Only the DECODER is fine-tuned; the offline term encoder stays frozen, so the
rdflib runtime and C++ kernel are unchanged -- after fine-tuning, repack
model.npz (overfit_e2e_setup.repack_model) from the saved checkpoint.

    # smoke (mechanics only, no endpoint):
    uv run python standalone/finetune_mrt.py --model <model_rank.pt> \
        --queries standalone/overfit_queries.json --dummy-oracle --epochs 3
    # real run (QLever on :7020, encoder x from the pack):
    uv run python standalone/finetune_mrt.py --model <model_rank.pt> \
        --queries standalone/overfit_queries.json \
        --pack ~/rdflib-joinordering/gbjo_pack/overfit-gps-v2 \
        --endpoint http://127.0.0.1:7020/ --cache standalone/cstar_cache.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbjo_fast import (FastGBJO, cartesian_penalty, beam_exact,
                       adjacency_to_join_order, sharing_matrix, featurize_query)
from model_dual import FlatCostGNNDual

CENSORED = 1e18  # true cost assigned to a plan whose prefix count timed out


# --------------------------------------------------------------------------
# True-cost oracle: QLever counts, cartesian = component product, disk cache
# --------------------------------------------------------------------------

def connected_components(idx_set, triples):
    """Partition triple indices into connected components by shared variables."""
    idx = sorted(idx_set)
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    var_owner = {}
    for i in idx:
        for atom in triples[i][:3]:
            if atom.startswith("?"):
                if atom in var_owner:
                    parent[find(i)] = find(var_owner[atom])
                else:
                    var_owner[atom] = i
    comps = {}
    for i in idx:
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


class CStarOracle:
    """C*(order) = sum of prefix cardinalities; cartesian prefixes via
    component product. Persistent cache keyed by component pattern-set hash."""

    def __init__(self, endpoint=None, timeout=30.0, cache_path=None,
                 dummy=False):
        self.endpoint = endpoint
        self.timeout = timeout
        self.cache_path = cache_path
        self.dummy = dummy
        self.cache = {}                       # hash -> count (persisted)
        self.censored = set()                 # hashes that timed out (not persisted)
        self.calls = 0
        if not dummy:
            import requests
            self.session = requests.Session()
        if cache_path and os.path.exists(cache_path):
            self.cache = json.load(open(cache_path))
        self._dirty = 0

    def _key(self, triples, comp):
        pats = sorted(" ".join(triples[i][:3]) for i in comp)
        return hashlib.sha1("\n".join(pats).encode()).hexdigest()

    def _count_component(self, triples, comp):
        key = self._key(triples, comp)
        if key in self.cache:
            return self.cache[key]
        if key in self.censored:
            return None
        if self.dummy:                        # deterministic proxy for mechanics
            nvars = len({a for i in comp for a in triples[i][:3]
                         if a.startswith("?")})
            val = int(10 ** max(1, nvars))
            self.cache[key] = val
            return val
        body = " . ".join(" ".join(triples[i][:3]) for i in sorted(comp)) + " ."
        q = f"SELECT (COUNT(*) AS ?count) WHERE {{ {body} }}"
        self.calls += 1
        try:
            r = self.session.get(
                self.endpoint, params={"query": q},
                headers={"Accept": "application/sparql-results+json"},
                timeout=(5.0, self.timeout))
            r.raise_for_status()
            val = int(r.json()["results"]["bindings"][0]["count"]["value"])
        except Exception:
            self.censored.add(key)
            return None
        self.cache[key] = val
        self._dirty += 1
        if self._dirty % 50 == 0:
            self.save()
        return val

    def card(self, triples, idx_set):
        """Cardinality of the join of idx_set (component product). None if any
        component timed out."""
        prod = 1
        for comp in connected_components(idx_set, triples):
            c = self._count_component(triples, comp)
            if c is None:
                return None
            prod *= c
        return prod

    def c_out(self, order, triples):
        """Sum of prefix cardinalities (k=2..n). Censored prefix -> CENSORED."""
        total, prefix = 0, {order[0]}
        for t in order[1:]:
            prefix.add(t)
            c = self.card(triples, prefix)
            if c is None:
                return CENSORED
            total += c
        return float(total)

    def save(self):
        if self.cache_path:
            tmp = self.cache_path + ".tmp"
            json.dump(self.cache, open(tmp, "w"))
            os.replace(tmp, self.cache_path)


# --------------------------------------------------------------------------
# Trainable decoder: wrap the flat tensors as autograd leaves
# --------------------------------------------------------------------------

def make_trainable(flat):
    """Turn FlatCostGNNDual's tensor attributes into requires_grad leaves;
    return (params, theta0) where theta0 are detached clones for the anchor."""
    params, theta0 = [], []

    def leaf(t):
        p = t.detach().clone().float().requires_grad_(True)
        params.append(p)
        theta0.append(p.detach().clone())
        return p

    flat.proj_w = leaf(flat.proj_w)
    flat.proj_b = leaf(flat.proj_b)
    flat.fc1_w = leaf(flat.fc1_w)
    flat.fc1_b = leaf(flat.fc1_b)
    flat.fc2_w = leaf(flat.fc2_w)
    flat.fc2_b = leaf(flat.fc2_b)
    flat.layers = [tuple(leaf(t) for t in layer) for layer in flat.layers]
    return params, theta0


def save_finetuned(flat, n_layers, out_path):
    """Write the fine-tuned decoder back in CostGNNDual state_dict format
    (keys match overfit_e2e_setup.repack_model)."""
    sd = {"projection.weight": flat.proj_w.detach(),
          "projection.bias": flat.proj_b.detach(),
          "fc1.weight": flat.fc1_w.detach(), "fc1.bias": flat.fc1_b.detach(),
          "fc2.weight": flat.fc2_w.detach(), "fc2.bias": flat.fc2_b.detach()}
    for i, (w1, b1, w2, b2) in enumerate(flat.layers):
        sd[f"mlps.{i}.0.weight"] = w1.detach()
        sd[f"mlps.{i}.0.bias"] = b1.detach()
        sd[f"mlps.{i}.2.weight"] = w2.detach()
        sd[f"mlps.{i}.2.bias"] = b2.detach()
    torch.save(sd, out_path)


# --------------------------------------------------------------------------
# Differentiable unroll + policy log-prob
# --------------------------------------------------------------------------

def connected_order(triples, S):
    """Greedy cartesian-free left-deep order (most-bound triple first, then
    always extend by a triple sharing a variable). None if disconnected."""
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
        A[n + k - 2, j] = 1      # previous join feeds this one
        A[order[k], j] = 1       # next triple
    return A


LAM_KEYS = ["lambda_triple_in", "lambda_triple_out", "lambda_join_in",
            "lambda_join_out", "lambda_left_linear", "lambda_acyclic"]


def unroll_diff(gbjo, x, steps, share, triples=None, pool_samples=32,
                sample_temp=2.5, tbptt=0, meta=True, lambdas=None, lr_scale=None):
    """Differentiable unroll. Returns (W_final (2n-2,n-1), h0, candidate
    A_hats). The forward trajectory/pool is always the same; what differs is
    how far the META-gradient (to theta / lambdas) flows back:
      meta=False         -> no meta-grad (evaluation)
      meta=True, tbptt<=0 -> full backprop through all `steps` (baseline)
      meta=True, tbptt=k  -> truncated: only the last k inner steps
    `lambdas`: optional dict of trainable penalty weights (real space, e.g.
    exp(log_lambda)); None uses the fixed gbjo.params values.

    Pool = deterministic mode + Gumbel-max samples from the policy + (if
    triples given) the connected-greedy cartesian-free plan, deduped."""
    p = gbjo.params
    n = (x.shape[0] + 1) // 2
    mask, N = gbjo._size_consts(n)
    lrs, moms, taus, lts = gbjo._schedule(steps)
    if getattr(gbjo.model, "is_dual", False):
        gbjo.model.bind_share((share > 0).float())
    h0 = gbjo.model.project_x(x)              # depends on theta (proj_w/b)
    lam = (lambda k: lambdas[k]) if lambdas is not None else (lambda k: p[k])

    L = torch.zeros(2 * n - 2, n - 1, requires_grad=True)
    buf = torch.zeros_like(L)
    clip = p["gradient_clip_norm"]
    lam_cart = p["lambda_cartesian"]
    if not meta:
        grad_start = steps                    # detach all -> no meta-grad
    elif tbptt <= 0:
        grad_start = 0                        # full backprop (all steps)
    else:
        grad_start = max(0, steps - tbptt)    # keep only the last tbptt steps
    for step in range(steps):
        keep = step >= grad_start
        inv_tau, lt = 1.0 / taus[step], lts[step]
        cost, penalty, A = gbjo._loss(
            L, mask, h0, inv_tau, n, N, lam("lambda_triple_in"),
            lam("lambda_triple_out"), lam("lambda_join_in"),
            lam("lambda_join_out"), lam("lambda_left_linear"))
        # native matrix_exp (double-backward safe, unlike the custom _TraceExpm)
        P_ac = torch.trace(torch.linalg.matrix_exp(A)) - N
        loss = cost + lt * (penalty + lam("lambda_acyclic") * P_ac)
        if lam_cart != 0.0 and share is not None:
            loss = loss + lam_cart * cartesian_penalty(
                A, share, n, p["cartesian_gamma"])
        g, = torch.autograd.grad(loss, L, create_graph=keep)
        norm = g.norm(2)
        if clip > 0:
            g = g * (clip / (norm + 1e-6)).clamp(max=1.0)
        buf = buf * moms[step] + g
        step_lr = lrs[step] if lr_scale is None else lr_scale * lrs[step]
        L = L - step_lr * buf
        if not keep:                          # cut graph for the detached prefix
            L = L.detach().requires_grad_(True)
            buf = buf.detach()

    inv_tau_f = 1.0 / taus[-1]
    logits = (L + mask) * inv_tau_f               # factorized-policy logits
    W = torch.softmax(logits, dim=1)
    scores = logits.detach().numpy()              # (2n-2, n-1)
    beam_w = gbjo.params["discrete_beam_width"]

    def decode(sc):
        A = np.zeros((N, N), dtype=np.float32)
        A[: 2 * n - 2, n:] = sc
        return beam_exact(A, beam_width=beam_w)

    pool = {}

    def add(A_hat):
        pool.setdefault(A_hat.tobytes(), A_hat)

    add(decode(scores))                           # deterministic policy mode
    rng = np.random.default_rng(0)
    flat_scores = scores / sample_temp            # flatten the peaked policy
    for _ in range(pool_samples):                 # Gumbel-max samples
        g = -np.log(-np.log(rng.uniform(size=scores.shape) + 1e-12) + 1e-12)
        add(decode(flat_scores + g))
    if triples is not None:                       # inject cartesian-free plan
        Snp = (share[:n, :n] > 0).numpy() if torch.is_tensor(share) \
            else (np.asarray(share)[:n, :n] > 0)
        conn = connected_order(triples, Snp)
        if conn is not None:
            add(order_to_adjacency(conn, n))
    return W, h0, list(pool.values())


def plan_logprob(W, A_hat, n):
    """log pi(P) = sum over source rows of log W[r, chosen join col]."""
    sel = torch.tensor(A_hat[: 2 * n - 2, n:], dtype=W.dtype)   # 0/1, one per row
    return (sel * torch.log(W + 1e-12)).sum()


# --------------------------------------------------------------------------
# Featurization
# --------------------------------------------------------------------------

def deploy_params(pack_dir):
    """The GD-search params the deployed kernel actually uses (pack meta.json).
    MRT minimises E_{P~pi_theta}[C*] where pi_theta IS the deployed decode
    distribution, so fine-tuning/eval MUST run under the deploy params -- using
    FastGBJO's stale defaults (init_tau 4.0, lambda_acyclic 29, lr 4.9 vs the
    tuned 2.55 / 1.8 / 3.9) silently optimises a different objective."""
    if pack_dir is None:
        return {}
    meta_path = os.path.join(os.path.expanduser(pack_dir), "meta.json")
    if not os.path.exists(meta_path):
        return {}
    return dict(json.load(open(meta_path)).get("params", {}))


def load_emb_source(pack_dir):
    """(emb_lookup dict-like, counts dict) from a runtime pack, so x carries the
    ENCODER embeddings (pack emb.npy) rather than rdf2vec."""
    if pack_dir is None:
        return {}, {}
    emb = np.load(os.path.join(pack_dir, "emb.npy"), mmap_mode="r")
    with open(os.path.join(pack_dir, "keys.txt"), encoding="utf-8") as f:
        keys = f.read().splitlines()
    row = {k: i for i, k in enumerate(keys)}
    cpath = os.path.join(pack_dir, "counts.npy")
    carr = np.load(cpath, mmap_mode="r") if os.path.exists(cpath) else None
    if carr is not None and carr.ndim != 1:
        carr = None

    class _Emb:                               # lazy: mmap row on demand
        def get(self, name, default=None):
            i = row.get(name)
            return None if i is None else np.asarray(emb[i], dtype=np.float64)

    class _Counts:
        def get(self, name, default=1):
            i = row.get(name)
            return default if (i is None or carr is None) else int(carr[i])

    return _Emb(), _Counts()


def build_items(queries, emb, counts, rng_seed=0):
    """[{triples, x, share, n}] from a list of {triples:[[s,p,o],...]}."""
    items = []
    g = torch.Generator().manual_seed(rng_seed)
    for q in queries:
        triples = [list(t) for t in q["triples"]]
        n = len(triples)
        if n < 2:
            continue
        x = featurize_query(triples, emb, counts, rng=g)
        share = sharing_matrix(triples)
        items.append({"triples": triples, "x": x, "share": share, "n": n})
    return items


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def mrt_step(gbjo, item, oracle, steps, alpha, beta, pool_samples=32,
             sample_temp=2.5, mrt_weight=1.0, tbptt=0, lambdas=None,
             lr_scale=None):
    """Return (loss, diag) for one query. loss requires grad on theta.
    mrt_weight scales the (through-unroll, 2nd-order) policy-shaping term;
    beta scales the (direct, 1st-order, stable) rank term."""
    triples, n = item["triples"], item["n"]
    W, h0, cands = unroll_diff(gbjo, item["x"], steps, item["share"],
                               triples=triples, pool_samples=pool_samples,
                               sample_temp=sample_temp, tbptt=tbptt,
                               meta=True, lambdas=lambdas, lr_scale=lr_scale)

    costs, logps, preds = [], [], []
    for A_hat in cands:
        order = adjacency_to_join_order(A_hat)
        costs.append(oracle.c_out(order, triples))
        logps.append(plan_logprob(W, A_hat, n))
        preds.append(gbjo.model.forward_from_h0(
            h0, torch.tensor(A_hat, dtype=torch.float32)))
    logc = torch.tensor([math.log10(max(c, 1.0)) for c in costs])
    regret = logc - logc.min()                       # >= 0, constant
    logps = torch.stack(logps)
    w = torch.softmax(alpha * logps, dim=0)
    L_mrt = (w * regret).sum()

    L_rank = torch.zeros(())
    if beta > 0 and len(cands) > 1:
        preds = torch.stack(preds)
        npairs = 0
        for i in range(len(cands)):
            for j in range(len(cands)):
                if logc[i] < logc[j] - 1e-9:         # i truly cheaper than j
                    L_rank = L_rank + F.softplus(preds[i] - preds[j])
                    npairs += 1
        if npairs:
            L_rank = L_rank / npairs

    comp = {"mrt": L_mrt.item(), "rank": L_rank.item(), "ncand": len(cands)}
    return mrt_weight * L_mrt + beta * L_rank, comp


def evaluate(gbjo, items, oracle, steps, pool_samples=32, sample_temp=2.5,
             lambdas=None, lr_scale=None):
    """True-cost diagnostics of the inference-picked plan (argmin predicted
    cost) over each val query. Returns per-query arrays + aggregates."""
    picked, best, censored = [], [], []
    for it in tqdm(items, desc="eval", leave=False):
        W, h0, cands = unroll_diff(gbjo, it["x"], steps, it["share"],
                                   triples=it["triples"], pool_samples=pool_samples,
                                   sample_temp=sample_temp, meta=False,
                                   lambdas=lambdas, lr_scale=lr_scale)
        with torch.no_grad():
            costs = [oracle.c_out(adjacency_to_join_order(A), it["triples"])
                     for A in cands]
            preds = [float(gbjo.model.forward_from_h0(
                h0, torch.tensor(A, dtype=torch.float32))) for A in cands]
        pk = int(np.argmin(preds))
        logc = [math.log10(max(c, 1.0)) for c in costs]
        picked.append(logc[pk])
        best.append(min(logc))
        censored.append(costs[pk] >= CENSORED)
    pk, bl, cen = np.array(picked), np.array(best), np.array(censored)
    return {
        "picked_logc": pk, "best_logc": bl, "censored": cen,
        "mean_regret": float((pk - bl).mean()),
        "subopt": float(((pk - bl) > 1e-6).mean()),
        "n_catastrophe": int(cen.sum()),
        "mean_picked_logc": float(pk.mean()),
    }


def save_plots(out_dir, history, last_eval, base):
    """Live MRT dashboard, rewritten every epoch. `base` = pretrained-model
    eval (reference lines)."""
    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    def ref(ax, y, label):
        ax.axhline(y, ls="--", c="gray", lw=1, label=f"pretrained ({label})")

    ax = axs[0, 0]
    ax.plot(ep, [h["val_regret"] for h in history], "-o", ms=3, c="tab:blue")
    ref(ax, base["mean_regret"], f"{base['mean_regret']:.3f}")
    ax.set_title("val true-cost regret (picked vs pool-best)")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean log10 regret"); ax.legend()

    ax = axs[0, 1]
    ax.plot(ep, [h["val_subopt"] * 100 for h in history], "-o", ms=3, c="tab:orange")
    ref(ax, base["subopt"] * 100, f"{base['subopt']*100:.0f}%")
    ax.set_title("val suboptimal picks"); ax.set_xlabel("epoch")
    ax.set_ylabel("% queries pick != pool-best"); ax.legend()

    ax = axs[0, 2]
    ax.plot(ep, [h["val_catastrophe"] for h in history], "-o", ms=3, c="tab:red")
    ref(ax, base["n_catastrophe"], f"{base['n_catastrophe']}")
    ax.set_title("val catastrophes (picked plan hits a timeout prefix)")
    ax.set_xlabel("epoch"); ax.set_ylabel("# val queries"); ax.legend()

    ax = axs[1, 0]
    ax.plot(ep, [h["train_L"] for h in history], "-o", ms=3, c="k", label="total")
    ax.plot(ep, [h["L_mrt"] for h in history], "-o", ms=3, label="MRT")
    ax.plot(ep, [h["L_rank"] for h in history], "-o", ms=3, label="rank")
    ax.plot(ep, [h["anchor"] for h in history], "-o", ms=3, label="anchor (L2)")
    ax.set_title("train loss components"); ax.set_xlabel("epoch")
    ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend()

    ax = axs[1, 1]
    ax.plot(ep, [h["val_mean_logc"] for h in history], "-o", ms=3, c="tab:green")
    ref(ax, base["mean_picked_logc"], f"{base['mean_picked_logc']:.2f}")
    ax.set_title("val mean picked plan true cost (absolute)")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean log10 C*"); ax.legend()

    ax = axs[1, 2]
    pk, bl = last_eval["picked_logc"], last_eval["best_logc"]
    lo, hi = float(min(bl.min(), pk.min())), float(max(pk.max(), bl.max()))
    ax.plot([lo, hi], [lo, hi], "--", c="gray", lw=1)
    ax.scatter(bl, pk, s=18, alpha=0.6,
               c=np.where(last_eval["censored"], "tab:red", "tab:blue"))
    ax.set_title(f"picked vs best true cost (epoch {ep[-1]}; red=timeout)")
    ax.set_xlabel("pool-best log10 C*"); ax.set_ylabel("picked log10 C*")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mrt_progress.png"), dpi=110)
    plt.close(fig)


def save_search_plots(out_dir, history):
    """Learned search-shaping scalars over epochs (penalty lambdas + the inner
    GD-search lr scale). Only drawn when --train-lambdas / --train-inner-lr is
    on; a no-op otherwise."""
    has_lam = any("lambdas" in h for h in history)
    has_lr = any("inner_lr_scale" in h for h in history)
    if not (has_lam or has_lr):
        return
    ep = [h["epoch"] for h in history]
    ncol = int(has_lam) + int(has_lr)
    fig, axs = plt.subplots(1, ncol, figsize=(7 * ncol, 4.5), squeeze=False)
    col = 0
    if has_lam:
        ax = axs[0, col]; col += 1
        keys = sorted({k for h in history if "lambdas" in h for k in h["lambdas"]})
        for k in keys:
            ax.plot(ep, [h.get("lambdas", {}).get(k, float("nan"))
                         for h in history], "-o", ms=3,
                    label=k.replace("lambda_", "λ_"))
        ax.set_title("learned penalty lambdas"); ax.set_xlabel("epoch")
        ax.set_ylabel("value"); ax.legend(fontsize=8)
    if has_lr:
        ax = axs[0, col]
        ax.plot(ep, [h.get("inner_lr_scale", float("nan")) for h in history],
                "-o", ms=3, c="tab:purple")
        ax.axhline(1.0, ls="--", c="gray", lw=1, label="init (1.0)")
        ax.set_title("learned inner GD-search lr scale"); ax.set_xlabel("epoch")
        ax.set_ylabel("scale x deploy schedule"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mrt_search_params.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pretrained decoder model_rank.pt")
    ap.add_argument("--queries", required=True, help="JSON [{triples:[[s,p,o]...]}]")
    ap.add_argument("--pack", default=None, help="runtime pack for encoder x (emb.npy)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:7020/")
    ap.add_argument("--cache", default=None, help="persistent C* cache json")
    ap.add_argument("--dummy-oracle", action="store_true",
                    help="deterministic proxy C* (mechanics test, no endpoint)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=10, help="unroll length")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mrt-weight", type=float, default=1.0,
                    help="weight on the (through-unroll) policy-shaping term")
    ap.add_argument("--tbptt", type=int, default=0,
                    help="truncated backprop: meta-grad through the last K "
                         "inner steps (0 = full backprop = baseline)")
    ap.add_argument("--clip", type=float, default=0.0,
                    help="theta/lambda grad-norm clip (0 = off)")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine-anneal the outer LR over epochs")
    ap.add_argument("--train-inner-lr", action="store_true",
                    help="also train a scalar multiplier on the inner GD-search "
                         "step size (log-space; shapes the search dynamics, "
                         "deploys as a scaled schedule.npz)")
    ap.add_argument("--search-lr", type=float, default=None,
                    help="separate (higher) lr for the search-shaping scalars "
                         "(lambdas + inner-lr); these are ~handful of scalars "
                         "that need a far bigger step than the decoder weights. "
                         "None = same as --lr")
    ap.add_argument("--train-lambdas", action="store_true",
                    help="also train the penalty lambdas (log-space, positive); "
                         "shapes the GD basins, not just the cost surface")
    ap.add_argument("--pool-samples", type=int, default=32,
                    help="Gumbel-max samples per query (pool diversity; cheap "
                         "because C* is cached per connected sub-pattern)")
    ap.add_argument("--sample-temp", type=float, default=2.5,
                    help="flatten the peaked policy by this factor before "
                         "Gumbel sampling (1=true policy, higher=more diverse)")
    ap.add_argument("--no-deploy-params", action="store_true",
                    help="do NOT load the pack's deploy GD params; use FastGBJO "
                         "stale defaults (old, mismatched behaviour)")
    ap.add_argument("--min-tau", type=float, default=None,
                    help="override the GD-search final anneal floor (default "
                         "0.49). Higher = softer search basin; the policy MRT "
                         "shapes IS the deployed W_T, so set the same value at "
                         "decode/deploy. The deployed decode must match.")
    ap.add_argument("--alpha", type=float, default=1.0, help="MRT policy sharpening")
    ap.add_argument("--beta", type=float, default=1.0, help="rank-term weight")
    ap.add_argument("--gamma", type=float, default=1e-3, help="trust-region L2")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation: average grads over K queries "
                         "before each optimizer step (mini-batch = variance "
                         "reduction; 1 = per-query step = old behaviour)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="standalone/models/mrt",
                    help="output dir (model_mrt.pt + mrt_progress.png + history)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    flat = FlatCostGNNDual.load(args.model)
    n_layers = len(flat.layers)
    params, theta0 = make_trainable(flat)
    gbjo = FastGBJO(flat, params={"lambda_cartesian": 0.0})
    dp = deploy_params(args.pack)
    if dp and not args.no_deploy_params:
        gbjo.params.update(dp)
        gbjo._sched_cache.clear()
        print(f"deploy params from pack: init_tau={dp.get('init_tau')} "
              f"min_tau={dp.get('min_tau')} lr={dp.get('learning_rate')} "
              f"lambda_acyclic={dp.get('lambda_acyclic')} "
              f"lambda_left_linear={dp.get('lambda_left_linear')}")
    if args.min_tau is not None:
        gbjo.params["min_tau"] = args.min_tau
        gbjo._sched_cache.clear()
        print(f"GD-search min_tau (final anneal floor) override = {args.min_tau}")

    # optional: trainable SEARCH-SHAPING scalars in their own (higher-lr) group
    # -- penalty lambdas + a scalar multiplier on the inner GD step size, both
    # log-space (always positive). A handful of scalars reshape the search
    # basins far more per-step than the decoder weights, so they get their own
    # lr; they are NOT in `params`, so the decoder clip/anchor leave them alone.
    log_lams = None
    if args.train_lambdas:
        log_lams = {k: torch.tensor(math.log(max(gbjo.params[k], 1e-3)),
                                    requires_grad=True) for k in LAM_KEYS}
        print("training penalty lambdas:",
              {k: round(gbjo.params[k], 2) for k in LAM_KEYS})
    log_lr = None
    if args.train_inner_lr:
        log_lr = torch.zeros((), requires_grad=True)   # scale starts at 1.0
        print("training inner GD-search lr scale (init 1.0)")
    search_params = (list(log_lams.values()) if log_lams else []) \
        + ([log_lr] if log_lr is not None else [])

    def cur_lambdas():
        return {k: torch.exp(v) for k, v in log_lams.items()} if log_lams else None

    def cur_lr_scale():
        return torch.exp(log_lr) if log_lr is not None else None

    groups = [{"params": params, "lr": args.lr}]
    if search_params:
        groups.append({"params": search_params,
                       "lr": args.search_lr or args.lr})
        print(f"search-shaping group: {len(search_params)} scalars "
              f"@ lr {args.search_lr or args.lr}")
    opt = torch.optim.Adam(groups)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.cosine else None)

    emb, counts = load_emb_source(args.pack)
    queries = json.load(open(args.queries))
    items = build_items(queries, emb, counts)
    nval = max(1, int(args.val_frac * len(items)))
    val, train = items[:nval], items[nval:]
    print(f"{len(items)} queries ({len(train)} train / {len(val)} val); "
          f"oracle={'dummy' if args.dummy_oracle else args.endpoint}", flush=True)

    oracle = CStarOracle(args.endpoint, args.timeout, args.cache,
                         dummy=args.dummy_oracle)
    os.makedirs(args.out, exist_ok=True)

    base = evaluate(gbjo, val, oracle, args.steps, args.pool_samples,
                    args.sample_temp, cur_lambdas(), cur_lr_scale())
    print(f"pretrained val: regret {base['mean_regret']:.3f}  "
          f"suboptimal {base['subopt']*100:.0f}%  "
          f"catastrophes {base['n_catastrophe']}/{len(val)}  "
          f"mean log10 C* {base['mean_picked_logc']:.2f}", flush=True)
    print(f"knobs: lr {args.lr} mrt_w {args.mrt_weight} beta {args.beta} "
          f"gamma {args.gamma} tbptt {args.tbptt or 'full'} clip {args.clip} "
          f"cosine {args.cosine} train_lambdas {args.train_lambdas}", flush=True)

    history, best_regret = [], float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        order = np.random.default_rng(epoch).permutation(len(train))
        agg = {"mrt": 0.0, "rank": 0.0, "anchor": 0.0, "tot": 0.0}
        accum = max(1, args.accum)            # grad accumulation = mini-batch
        pbar = tqdm(order, desc=f"epoch {epoch}", leave=False)
        opt.zero_grad()
        for i, qi in enumerate(pbar):
            loss, comp = mrt_step(gbjo, train[qi], oracle, args.steps,
                                  args.alpha, args.beta, args.pool_samples,
                                  args.sample_temp, args.mrt_weight,
                                  args.tbptt, cur_lambdas(), cur_lr_scale())
            total, anchor_val = loss, 0.0
            if args.gamma > 0:                # trust region (skip if gamma=0)
                anchor = sum(((p - p0) ** 2).sum()
                             for p, p0 in zip(params, theta0))
                total = loss + args.gamma * anchor
                anchor_val = anchor.item()
            (total / accum).backward()        # mean grad over the mini-batch
            if (i + 1) % accum == 0 or (i + 1) == len(order):
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                opt.zero_grad()
            agg["mrt"] += comp["mrt"]; agg["rank"] += comp["rank"]
            agg["anchor"] += anchor_val; agg["tot"] += loss.item()
            pbar.set_postfix(L=f"{agg['tot']/(i+1):.3f}", calls=oracle.calls)
        if sched is not None:
            sched.step()
        nt = len(train)
        ev = evaluate(gbjo, val, oracle, args.steps, args.pool_samples,
                      args.sample_temp, cur_lambdas(), cur_lr_scale())
        rec = {
            "epoch": epoch, "L_mrt": agg["mrt"] / nt, "L_rank": agg["rank"] / nt,
            "anchor": agg["anchor"] / nt, "train_L": agg["tot"] / nt,
            "val_regret": ev["mean_regret"], "val_subopt": ev["subopt"],
            "val_catastrophe": ev["n_catastrophe"],
            "val_mean_logc": ev["mean_picked_logc"]}
        if log_lams:                          # per-epoch learned search scalars
            rec["lambdas"] = {k: torch.exp(v).item() for k, v in log_lams.items()}
        if log_lr is not None:
            rec["inner_lr_scale"] = torch.exp(log_lr).item()
        history.append(rec)
        oracle.save()
        save_plots(args.out, history, ev, base)
        save_search_plots(args.out, history)
        json.dump(history, open(os.path.join(args.out, "history.json"), "w"))
        mark = ""
        if ev["mean_regret"] < best_regret:   # save BEST, not last
            best_regret = ev["mean_regret"]
            save_finetuned(flat, n_layers, os.path.join(args.out, "model_mrt.pt"))
            if log_lams or log_lr is not None:
                best = {k: torch.exp(v).item() for k, v in log_lams.items()} \
                    if log_lams else {}
                if log_lr is not None:
                    best["inner_lr_scale"] = torch.exp(log_lr).item()
                json.dump(best, open(
                    os.path.join(args.out, "search_best.json"), "w"))
            mark = "  <- best"
        print(f"epoch {epoch}: train L {agg['tot']/nt:.4f}  "
              f"val regret {ev['mean_regret']:.3f}  "
              f"subopt {ev['subopt']*100:.0f}%  "
              f"catas {ev['n_catastrophe']}/{len(val)}  "
              f"calls {oracle.calls}  ({time.time()-t0:.0f}s){mark}", flush=True)

    save_finetuned(flat, n_layers, os.path.join(args.out, "model_last.pt"))
    print(f"best val regret {best_regret:.3f}  (vs pretrained "
          f"{base['mean_regret']:.3f}); best -> {args.out}/model_mrt.pt")
    if log_lams:
        print("final lambdas:",
              {k: round(torch.exp(v).item(), 2) for k, v in log_lams.items()})
    if log_lr is not None:
        print(f"final inner-lr scale: {torch.exp(log_lr).item():.3f}")


if __name__ == "__main__":
    main()
