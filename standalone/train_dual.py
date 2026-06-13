"""
Train CostGNNDual (plan adjacency + variable-sharing adjacency) on the path
plan datasets, optionally with the FICE pairwise ranking loss (FICE paper,
appendix H.1) over sibling plans of the same query.

Sources (selectable via --sources):
  new3   : data/plans/wikidata_path_plan_datasets_training/new3 (334k, ~free)
  addon  : new-combined/dataset.pt[:8993] (the cartesian add-on, sizes 3-8)
  cart   : --cart-data dir (sizes 9-15 cartesian + contrast plans)
  sib    : --sib-data dir (sibling families + matched single-cartesian
           variants, gen_sibling_plans.py)

Loss: Huber on log(C_out) over regression batches, plus (if --rank-weight>0)
the FICE RankNet ranking term over plan groups of the same query, sampled
from --rank-sources only:
    L = Huber + rank_weight * sum_pairs(w_size*w_top*BCE(p_j-p_i,1)) / sum(w)
with w_size=(n_i*n_j)^(alpha/2), w_top=1/(1+beta*min(rank_i,rank_j)).

The train/val split is BY QUERY (group-wise), so val ranking pair accuracy
is leak-free. Validation reports q-error between log-space values,
qerr = max(log_pred/log_true, log_true/log_pred), as in the original
cost_model_training.py.

Current-state run (no ranking, as in dual-v1):
    uv run python standalone/train_dual.py --epochs 60 --out .../dual-v1b
FICE-style run:
    uv run python standalone/train_dual.py --epochs 200 --cart-weight 10 \
        --sources new3,addon,cart,sib --rank-weight 3.0 --out .../dual-v2

--encoder gps|gine additionally trains a FICE-style term encoder
(term_encoder.py) end-to-end with the cost model: bound-term embeddings in x
are replaced by encoder outputs computed from each term's sampled
factor-graph neighborhood (kg_index.py; build the index first). rdf2vec and
occurrence counts become encoder INPUTS (--encoder-no-rdf2vec /
--encoder-no-counts to ablate). --encoder off (default) is the exact current
behavior. Deployment stays offline: after training, encode all terms once
and write them as the pack's emb.npy -- the rdflib runtime and C++ kernel
are unchanged.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_data import (triple_var_sets, share_edge_index,
                       plan_cartesian_count, collate)
from model_dual import CostGNNDual

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO, "data", "plans", "wikidata_path_plan_datasets_training")


def load_source(spec, cart_data_dir, sib_data_dir, limit=None):
    """spec is a legacy name (new3/addon/cart/sib) or 'label=path' where
    path is a dataset.pt or a directory containing one (e.g. for lubm:
    base=data/plans/lubm/path-greedy). The label is what --rank-sources
    matches against."""
    name = spec
    if "=" in spec:
        name, path = spec.split("=", 1)
        if os.path.isdir(path):
            path = os.path.join(path, "dataset.pt")
        d = torch.load(path, weights_only=False)
        data, triples = d["data"], d["triples"]
        if limit:
            data, triples = data[:limit], triples[:limit]
        print(f"  {name}: {len(data)} samples ({path})")
        return [(s, tuple(t), name) for s, t in zip(data, triples)]
    if name == "new3":
        path = os.path.join(TRAIN_DIR, "new3", "dataset.pt")
        d = torch.load(path, weights_only=False)
        data, triples = d["data"], d["triples"]
    elif name == "addon":
        path = os.path.join(TRAIN_DIR, "new-combined", "dataset.pt")
        d = torch.load(path, weights_only=False)
        data, triples = d["data"][:8993], d["triples"][:8993]
    elif name == "cart":
        d = torch.load(os.path.join(cart_data_dir, "dataset.pt"),
                       weights_only=False)
        data, triples = d["data"], d["triples"]
    elif name == "sib":
        d = torch.load(os.path.join(sib_data_dir, "dataset.pt"),
                       weights_only=False)
        data, triples = d["data"], d["triples"]
    else:
        raise ValueError(name)
    if limit:
        data, triples = data[:limit], triples[:limit]
    print(f"  {name}: {len(data)} samples")
    return [(s, tuple(t), name) for s, t in zip(data, triples)]


def assemble(raw):
    """-> (samples, gids): samples is a list of (x, edge_index, share_ei,
    log_y, is_cart, gid, n, src); gid groups plans of the same query
    (identical triples); gids maps the triples tuple -> gid."""
    out, skipped = [], 0
    gids = {}
    t0 = time.time()
    for i, (s, tr, src) in enumerate(raw):
        y = float(s.y.item())
        if not np.isfinite(y) or y <= 0:
            skipped += 1
            continue
        n = (s.x.shape[0] + 1) // 2
        vs = triple_var_sets(s.x, n)
        esh = share_edge_index(vs)
        cart = plan_cartesian_count(s.edge_index, vs, n)
        if cart < 0:
            skipped += 1
            continue
        gid = gids.setdefault(tr, len(gids))
        out.append((s.x, s.edge_index.long(), esh, math.log(y), cart > 0,
                    gid, n, src))
        if (i + 1) % 50_000 == 0:
            print(f"  assembled {i+1}/{len(raw)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"assembly: {len(out)} samples, {len(gids)} query groups, "
          f"{skipped} skipped, {sum(s[4] for s in out)} cartesian, "
          f"{time.time()-t0:.0f}s")
    return out, gids


def bound_atoms(triples):
    """Bound (non-variable) atoms of a query: [(row, slot_offset, term,
    kind)] with kind 'ent' for subject/object slots, 'rel' for predicate.
    Triple strings look like '<s> <p> ?o1.'; row i of x is triples[i]."""
    out = []
    for i, t in enumerate(triples):
        s, p, o = t.split(" ", 2)
        o = o.rstrip()
        if o.endswith("."):
            o = o[:-1].rstrip()
        for off, term, kind in ((0, s, "ent"), (102, p, "rel"),
                                (204, o, "ent")):
            if not term.startswith("?"):
                out.append((i, off, term, kind))
    return out


def build_gid_atoms(gids, samples, kg):
    """gid -> [(row, slot_offset, factor-graph node id)] for the encoder.
    Cross-checks the parsed bound slots against the all-ones variable
    markers in x for one sample per query group."""
    first_x = {}
    for s in samples:
        if s[5] not in first_x:
            first_x[s[5]] = (s[0], s[6])
    gid_atoms = {}
    unresolved = 0
    for tr, gid in gids.items():
        atoms = bound_atoms(tr)
        x, n = first_x[gid]
        xn = x.numpy()
        marker = {(r, off) for off in (0, 102, 204)
                  for r in np.nonzero(
                      ~(xn[:n, off + 1:off + 101] == 1).all(axis=1))[0]}
        parsed = {(r, off) for r, off, _, _ in atoms}
        if marker != parsed:
            raise ValueError(f"bound-slot mismatch for gid {gid}: "
                             f"markers {sorted(marker)} vs parsed "
                             f"{sorted(parsed)}\ntriples: {tr}")
        resolved = []
        for r, off, term, kind in atoms:
            nid = kg.node_id(term, kind)
            if nid < 0:
                unresolved += 1  # term not in KG: keep the baked embedding
            else:
                resolved.append((int(r), off, nid))
        gid_atoms[gid] = resolved
    if unresolved:
        print(f"  encoder: {unresolved} bound atoms not in KG index "
              f"(keep baked embeddings)")
    return gid_atoms


def quantile_huber(pred, target, tau, delta=1.0):
    """Asymmetric (quantile) Huber on log-cost. Residual e = target - pred;
    e > 0 means under-prediction (pred too low) -> weighted by tau. With
    tau > 0.5, under-pricing a plan costs ~tau/(1-tau)x more than over-pricing
    it, biasing the model conservative (catastrophe guard). tau = 0.5
    reduces to symmetric Huber (up to a 0.5 scale)."""
    e = target - pred
    h = F.huber_loss(pred, target, reduction="none", delta=delta)
    # weights normalized so tau=0.5 -> 1.0 (exact symmetric Huber); the
    # under/over ratio is tau/(1-tau) regardless (4:1 at tau=0.8).
    w = torch.where(e > 0, 2.0 * tau, 2.0 * (1.0 - tau))
    return (w * h).mean()


def qerror_log(preds, trues):
    """q-error between log-space values (as in cost_model_training.py)."""
    eps = 1e-10
    return np.maximum(trues / (preds + eps), preds / (trues + eps))


def ranking_terms(log_pred, log_true, group_id, sizes, alpha, beta):
    """FICE appendix H.1: for each same-group pair (i, j) with t_i < t_j,
    RankNet term BCE_logits(p_j - p_i, 1), weighted by (n_i*n_j)^(alpha/2)
    and 1/(1 + beta*min(rank_i, rank_j)). Returns (total, total_weight,
    n_correct_pairs, n_pairs) -- loss is total/total_weight."""
    total = log_pred.new_zeros(())
    total_w = log_pred.new_zeros(())
    n_correct, n_pairs = 0, 0
    for g in torch.unique(group_id).tolist():
        m = group_id == g
        if int(m.sum()) < 2:
            continue
        p, t = log_pred[m], log_true[m]
        dt = t.unsqueeze(1) - t.unsqueeze(0)
        pair = dt < 0  # (i, j) with t_i < t_j
        if not bool(pair.any()):
            continue
        dp = p.unsqueeze(1) - p.unsqueeze(0)
        per = F.binary_cross_entropy_with_logits(
            -dp, torch.ones_like(dp), reduction="none")
        w = pair.to(p.dtype)
        if alpha != 0.0:
            s = sizes[m].to(p.dtype)
            w = w * (s.unsqueeze(1) * s.unsqueeze(0)).pow(alpha / 2.0)
        if beta != 0.0:
            r = t.argsort().argsort().to(p.dtype)
            rmin = torch.minimum(r.unsqueeze(1), r.unsqueeze(0))
            w = w * (1.0 / (1.0 + beta * rmin))
        total = total + (per * w).sum()
        total_w = total_w + w.sum()
        n_correct += int(((dp < 0) & pair).sum())
        n_pairs += int(pair.sum())
    return total, total_w, n_correct, n_pairs


def collate_groups(samples, idxs, generator=None):
    """Collate samples idxs and return (model inputs, y, group_id, sizes)."""
    chunk = [samples[i] for i in idxs]
    x, ei, esh, batch, y = collate([(s[0], s[1], s[2], s[3]) for s in chunk],
                                   generator=generator)
    gid = torch.tensor([s[5] for s in chunk], dtype=torch.long)
    sizes = torch.tensor([s[6] for s in chunk], dtype=torch.float32)
    return x, ei, esh, batch, y, gid, sizes


def evaluate(model, samples, device, batch_size=512, embed=None):
    model.eval()
    gen = torch.Generator().manual_seed(123)  # deterministic val fingerprints
    preds, trues, is_cart = [], [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            x, ei, esh, batch, y = collate([(s[0], s[1], s[2], s[3])
                                            for s in chunk], generator=gen)
            xd = embed(x, chunk) if embed else x.to(device)
            out = model(xd, ei.to(device), esh.to(device),
                        batch.to(device), num_graphs=len(chunk))
            preds.extend(out.cpu().tolist())
            trues.extend(y.tolist())
            is_cart.extend(s[4] for s in chunk)
    preds, trues = np.array(preds), np.array(trues)
    is_cart = np.array(is_cart, dtype=bool)
    val_huber = float(F.huber_loss(torch.tensor(preds),
                                   torch.tensor(trues)).item())
    errs = qerror_log(preds, trues)

    def stats(v):
        return (f"med {np.median(v):.3f} p95 {np.percentile(v, 95):.3f}"
                if len(v) else "n/a")
    msg = (f"qerr(log) all[{stats(errs)}] "
           f"free[{stats(errs[~is_cart])}] cart[{stats(errs[is_cart])}]")
    return errs, val_huber, msg, (preds, trues, is_cart)


def evaluate_ranking(model, samples, groups, device, alpha, beta,
                     groups_per_chunk=64, embed=None):
    """Val ranking loss + pair accuracy over whole query groups."""
    if not groups:
        return None, None
    model.eval()
    gen = torch.Generator().manual_seed(456)
    total, total_w = 0.0, 0.0
    n_correct, n_pairs = 0, 0
    with torch.no_grad():
        for i in range(0, len(groups), groups_per_chunk):
            idxs = np.concatenate(groups[i:i + groups_per_chunk])
            x, ei, esh, batch, y, gid, sizes = collate_groups(
                samples, idxs, generator=gen)
            xd = (embed(x, [samples[j] for j in idxs]) if embed
                  else x.to(device))
            out = model(xd, ei.to(device), esh.to(device),
                        batch.to(device), num_graphs=len(idxs)).cpu()
            t, w, c, p = ranking_terms(out, y, gid, sizes, alpha, beta)
            total += float(t)
            total_w += float(w)
            n_correct += c
            n_pairs += p
    if total_w <= 0 or n_pairs == 0:
        return None, None
    return total / total_w, n_correct / n_pairs


def build_rank_groups(samples, idx_list, rank_sources):
    """Query groups (>=2 plans, >=2 distinct costs) from rank sources only."""
    by_gid = {}
    for i in idx_list:
        s = samples[i]
        if s[7] in rank_sources:
            by_gid.setdefault(s[5], []).append(i)
    return [np.array(v) for v in by_gid.values()
            if len(v) >= 2 and len({samples[i][3] for i in v}) >= 2]


def save_plots(out_dir, epoch, preds, trues, is_cart, history):
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    l10 = math.log(10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(trues[~is_cart] / l10, preds[~is_cart] / l10, s=3, alpha=0.3,
               label="cartesian-free", color="tab:blue")
    if is_cart.any():
        ax.scatter(trues[is_cart] / l10, preds[is_cart] / l10, s=4, alpha=0.5,
                   label="with cartesian", color="tab:red")
    lim = [min(trues.min(), preds.min()) / l10 - 0.5,
           max(trues.max(), preds.max()) / l10 + 0.5]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true log10 C_out"); ax.set_ylabel("predicted log10 C_out")
    ax.set_title(f"val epoch {epoch}"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots, f"scatter_epoch_{epoch:03d}.png"), dpi=110)
    fig.savefig(os.path.join(plots, "scatter_latest.png"), dpi=110)
    plt.close(fig)

    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(ep, [h["train_huber"] for h in history], label="train")
    axs[0].plot(ep, [h["val_huber"] for h in history], label="val")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("huber (log C_out)")
    axs[0].set_title("regression loss"); axs[0].legend()

    tr = [h["train_rank"] for h in history]
    vr = [h["val_rank"] for h in history]
    if any(v is not None for v in tr + vr):
        if any(v is not None for v in tr):
            axs[1].plot(ep, tr, label="train")
        if any(v is not None for v in vr):
            axs[1].plot(ep, vr, label="val")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "ranking off", ha="center", va="center",
                    transform=axs[1].transAxes)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("ranking loss")
    axs[1].set_title("ranking loss (FICE)")

    axs[2].plot(ep, [h["val_med_qerr"] for h in history], color="tab:orange",
                label="val med qerr(log)")
    axs[2].set_xlabel("epoch"); axs[2].set_ylabel("median qerr(log)",
                                                  color="tab:orange")
    ra = [h["val_rank_acc"] for h in history]
    if any(v is not None for v in ra):
        ax2 = axs[2].twinx()
        ax2.plot(ep, ra, color="tab:green", label="val pair acc")
        ax2.set_ylabel("val ranking pair accuracy", color="tab:green")
    axs[2].set_title("validation metrics")

    fig.tight_layout()
    fig.savefig(os.path.join(plots, "curves.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="new3,addon,cart")
    ap.add_argument("--cart-data", default="standalone/cart_plans_9_15")
    ap.add_argument("--sib-data", default="standalone/sib_plans")
    ap.add_argument("--out", default="standalone/models/dual-v2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cart-weight", type=float, default=5.0)
    ap.add_argument("--quantile-tau", type=float, default=0.5,
                    help="asymmetric Huber quantile: tau>0.5 penalizes "
                         "under-prediction (catastrophe guard); 0.5 = "
                         "symmetric Huber")
    ap.add_argument("--rank-weight", type=float, default=0.0,
                    help="lambda of the FICE ranking loss (0 = off)")
    ap.add_argument("--rank-groups", type=int, default=8,
                    help="query groups per training step")
    ap.add_argument("--rank-every", type=int, default=1,
                    help="compute the ranking term every k-th step")
    ap.add_argument("--rank-sources", default="sib",
                    help="sources whose groups feed the ranking loss")
    ap.add_argument("--rank-alpha", type=float, default=1.0,
                    help="size-weight exponent (FICE alpha)")
    ap.add_argument("--rank-beta", type=float, default=1.0,
                    help="top-weight strength (FICE beta)")
    ap.add_argument("--limit", type=int, default=None, help="per-source cap (smoke)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["cosine", "constant"],
                    help="constant avoids freezing a slow-converging encoder "
                         "before it reaches its floor (capacity probe)")
    # FICE-style term encoder (off = exact current behavior)
    ap.add_argument("--encoder", default="off", choices=["off", "gine", "gps"],
                    help="encode bound terms from their factor-graph "
                         "neighborhood instead of using baked rdf2vec")
    ap.add_argument("--encoder-pe", default="rwpe",
                    choices=["rwpe", "lap", "none"])
    ap.add_argument("--encoder-pe-dim", type=int, default=24,
                    help="rwpe: 3*walk_length; lap: #eigenvectors")
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--encoder-hidden", type=int, default=128)
    ap.add_argument("--encoder-attn", default="multihead",
                    choices=["multihead", "performer"],
                    help="gps self-attention: O(n^2) softmax or linear FAVOR+")
    ap.add_argument("--encoder-no-local-mp", action="store_true",
                    help="gps: drop the local GINE message-passing, attention only")
    ap.add_argument("--encoder-caps", default="16,8,4",
                    help="per-hop neighbor caps of the term subgraph")
    ap.add_argument("--encoder-rdf2vec", action="store_true",
                    help="add rdf2vec to the encoder input features (off by "
                         "default: blows up feature size, redundant with the "
                         "sampled neighborhood + counts)")
    ap.add_argument("--encoder-no-fanout", action="store_true",
                    help="drop per-relation log max_out/max_in (worst-case "
                         "fan-out) from the encoder input features")
    ap.add_argument("--encoder-no-rel-emb", action="store_true",
                    help="drop the learnable per-predicate identity embedding")
    ap.add_argument("--encoder-no-counts", action="store_true",
                    help="drop occurrence counts from the encoder input")
    ap.add_argument("--kg-index", default="standalone/kg_index/wikidata")
    ap.add_argument("--pack",
                    default=os.path.expanduser(
                        "~/rdflib-joinordering/gbjo_pack/wikidata"),
                    help="pack dir with emb.npy/keys.txt (rdf2vec input)")
    ap.add_argument("--overfit-groups", type=int, default=0,
                    help="capacity probe: train AND eval on the first N query "
                         "groups (no split, no cart oversampling, full pass "
                         "each epoch); val_* columns are then train metrics")
    args = ap.parse_args()

    device = (("mps" if torch.backends.mps.is_available() else "cpu")
              if args.device == "auto" else args.device)
    print(f"device: {device}")

    print("loading sources ...")
    raw = []
    for src in args.sources.split(","):
        raw.extend(load_source(src.strip(), args.cart_data, args.sib_data,
                               args.limit))
    samples, gids = assemble(raw)
    del raw

    if args.overfit_groups:
        keep = set(sorted({s[5] for s in samples})[:args.overfit_groups])
        samples = [s for s in samples if s[5] in keep]
        gids = {tr: g for tr, g in gids.items() if g in keep}
        print(f"overfit probe: {len(samples)} samples / {len(keep)} groups, "
              f"{sum(s[4] for s in samples)} cartesian")

    encoder, provider, embed = None, None, None
    if args.encoder != "off":
        from kg_index import KGIndex
        from term_encoder import SubgraphProvider, TermEncoder
        print(f"loading KG index {args.kg_index} ...")
        kg = KGIndex.load(args.kg_index)
        caps = tuple(int(c) for c in args.encoder_caps.split(","))
        provider = SubgraphProvider(
            kg, pe=args.encoder_pe, pe_dim=args.encoder_pe_dim, caps=caps,
            pack_dir=args.pack, use_rdf2vec=args.encoder_rdf2vec)
        use_fanout = not args.encoder_no_fanout
        if use_fanout and kg.max_out is None:
            print("  WARNING: pred_stats.npz missing; disabling fanout feature")
            use_fanout = False
        encoder = TermEncoder(
            hidden=args.encoder_hidden, out_dim=100,
            n_layers=args.encoder_layers, arch=args.encoder,
            pe=args.encoder_pe, pe_dim=args.encoder_pe_dim,
            use_rdf2vec=args.encoder_rdf2vec,
            use_counts=not args.encoder_no_counts,
            attn=args.encoder_attn, local_mp=not args.encoder_no_local_mp,
            use_fanout=use_fanout, rel_emb=not args.encoder_no_rel_emb,
            n_relations=kg.nR)
        gid_atoms = build_gid_atoms(gids, samples, kg)
        encoder = encoder.to(device)
        print(f"encoder: {args.encoder} {args.encoder_layers}x"
              f"{args.encoder_hidden}, pe {args.encoder_pe}"
              f"({args.encoder_pe_dim}), caps {caps}, "
              f"{sum(p.numel() for p in encoder.parameters()):,} params")

        def embed(x, chunk):
            """Overwrite bound-atom embedding blocks of the collated x with
            encoder outputs (functional index_put, keeps gradients)."""
            x = x.to(device)
            per_off = {0: ([], []), 102: ([], []), 204: ([], [])}
            nodes, node_pos = [], {}
            off_node = 0
            for s in chunk:
                for row, off, nid in gid_atoms[s[5]]:
                    p = node_pos.get(nid)
                    if p is None:
                        p = node_pos[nid] = len(nodes)
                        nodes.append(nid)
                    rows, sel = per_off[off]
                    rows.append(off_node + row)
                    sel.append(p)
                off_node += s[0].shape[0]
            if not nodes:
                return x
            E = encoder(**provider.batch(nodes, device))
            for off, (rows, sel) in per_off.items():
                if rows:
                    r = torch.tensor(rows, device=device)
                    c = torch.arange(off + 1, off + 101, device=device)
                    x = x.index_put((r.unsqueeze(1), c.unsqueeze(0)),
                                    E[torch.tensor(sel, device=device)])
            return x

    rng = np.random.default_rng(0)
    if args.overfit_groups:
        # capacity probe: fit and report on the same set
        train = val = samples
        print(f"overfit: train == val ({len(train)} samples)")
    else:
        # group-wise (by query) 90/10 split
        all_gids = sorted({s[5] for s in samples})
        gperm = rng.permutation(len(all_gids))
        n_val_g = max(1, int(0.1 * len(all_gids)))
        val_gids = {all_gids[i] for i in gperm[:n_val_g]}
        train_idx = [i for i, s in enumerate(samples) if s[5] not in val_gids]
        val_idx = [i for i, s in enumerate(samples) if s[5] in val_gids]
        train = [samples[i] for i in train_idx]
        val = [samples[i] for i in val_idx]
        print(f"train {len(train)} / val {len(val)} "
              f"({len(all_gids) - n_val_g}/{n_val_g} query groups)")

    # ranking groups (indices into train / val lists)
    rank_sources = set(args.rank_sources.split(","))
    rank_on = args.rank_weight > 0
    train_groups = build_rank_groups(train, range(len(train)), rank_sources)
    val_groups = build_rank_groups(val, range(len(val)), rank_sources)
    print(f"ranking groups: train {len(train_groups)} / val {len(val_groups)} "
          f"(sources {sorted(rank_sources)}, "
          f"{'ON' if rank_on else 'off'} w={args.rank_weight})")
    if rank_on and not train_groups:
        sys.exit("--rank-weight > 0 but no ranking groups; check --sources")

    # oversample cartesian-containing samples (uniform full pass in overfit)
    w = np.array([args.cart_weight if s[4] else 1.0 for s in train])
    w /= w.sum()

    model = CostGNNDual(node_feature_dim=307, hidden_dim=args.hidden,
                        n_layers=args.layers).to(device)
    params = list(model.parameters())
    if encoder is not None:
        params += list(encoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.lr_schedule == "cosine" else None)
    crit = lambda out, y: quantile_huber(out, y, args.quantile_tau)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": device, "model": "CostGNNDual",
                   "target": "log(C_out)", "split": "group-wise",
                   "metric": "qerror(log_pred, log_true)"}, f, indent=2)

    best_val = float("inf")
    best_acc = -1.0
    history = []
    if args.overfit_groups:
        n_batches = max(1, math.ceil(len(train) / args.batch))
    else:
        n_batches = max(1, len(train) // args.batch)
    for epoch in range(args.epochs):
        model.train()
        if encoder is not None:
            encoder.train()
        t0 = time.time()
        tot_huber, tot_rank, n_rank = 0.0, 0.0, 0
        if args.overfit_groups:  # full deterministic pass, every sample once
            order = rng.permutation(len(train))
        else:
            order = rng.choice(len(train), size=n_batches * args.batch, p=w)
        for b in tqdm(range(n_batches), desc=f"epoch {epoch}", leave=False,
                      mininterval=5):
            idx = order[b * args.batch:(b + 1) * args.batch]
            chunk = [train[i] for i in idx]
            x, ei, esh, batch, y = collate([(s[0], s[1], s[2], s[3])
                                            for s in chunk])
            opt.zero_grad()
            xd = embed(x, chunk) if embed else x.to(device)
            out = model(xd, ei.to(device), esh.to(device),
                        batch.to(device), num_graphs=len(chunk))
            loss = crit(out, y.to(device))
            tot_huber += loss.item()

            if rank_on and b % args.rank_every == 0:
                gsel = rng.choice(len(train_groups),
                                  size=min(args.rank_groups, len(train_groups)),
                                  replace=False)
                ridx = np.concatenate([train_groups[g] for g in gsel])
                rx, rei, resh, rbatch, ry, rgid, rsz = collate_groups(train, ridx)
                rxd = (embed(rx, [train[i] for i in ridx]) if embed
                       else rx.to(device))
                rout = model(rxd, rei.to(device), resh.to(device),
                             rbatch.to(device), num_graphs=len(ridx))
                t, tw, _, _ = ranking_terms(rout, ry.to(device),
                                            rgid.to(device), rsz.to(device),
                                            args.rank_alpha, args.rank_beta)
                if float(tw) > 0:
                    rloss = t / tw
                    loss = loss + args.rank_weight * rloss
                    tot_rank += float(rloss.detach())
                    n_rank += 1

            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        if device == "mps":
            # varying batch shapes make the MPS caching allocator grow
            # without bound; release the cache once per epoch
            torch.mps.empty_cache()

        errs, val_huber, msg, (preds, trues, is_cart) = evaluate(
            model, val, device, embed=embed)
        val_rank, val_acc = (evaluate_ranking(model, val, val_groups, device,
                                              args.rank_alpha, args.rank_beta,
                                              embed=embed)
                             if val_groups else (None, None))
        val_med = float(np.median(errs))
        history.append({
            "epoch": epoch,
            "train_huber": tot_huber / n_batches,
            "train_rank": (tot_rank / n_rank) if n_rank else None,
            "val_huber": val_huber,
            "val_rank": val_rank,
            "val_med_qerr": val_med,
            "val_rank_acc": val_acc,
        })
        marker = ""

        def save(tag):
            torch.save(model.state_dict(),
                       os.path.join(args.out, f"model{tag}.pt"))
            if encoder is not None:
                torch.save(encoder.state_dict(),
                           os.path.join(args.out, f"encoder{tag}.pt"))

        if val_med < best_val:
            best_val = val_med
            save("")
            marker += "  <- saved"
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            save("_rank")
            marker += " [rank]"
        save("_last")
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f)
        save_plots(args.out, epoch, preds, trues, is_cart, history)
        rk = ""
        if n_rank:
            rk = f"rank {tot_rank/n_rank:.4f}"
            if val_rank is not None:
                rk += f"/{val_rank:.4f} acc {val_acc:.3f}"
            rk += "  "
        print(f"epoch {epoch:>3}: huber {tot_huber/n_batches:.4f}/{val_huber:.4f}  "
              f"{rk}{msg}  ({time.time()-t0:.0f}s){marker}", flush=True)

    print(f"done; best val median qerr(log) {best_val:.3f}"
          + (f", best val pair acc {best_acc:.3f}" if best_acc >= 0 else "")
          + f"; model in {args.out}")


if __name__ == "__main__":
    main()
