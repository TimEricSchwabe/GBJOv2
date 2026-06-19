"""Pretrain CostGNNDual with an AUXILIARY per-join intermediate-cardinality loss
(flavor (a): deep supervision). Reuses train_dual.py's data/loss/eval/plot
machinery verbatim; the only additions are gated behind --card-weight:

  * a card head that predicts each join's partial-result log-cardinality from
    [h_j ; sum over its subtree of h]  (refinement 1: own embedding + subtree
    pool, so the receptive field covers the whole prefix regardless of depth),
  * a masked quantile-Huber loss against the mined subcost cache
    (mine_subcosts.py), added as card_weight * L_card.

The base cost + ranking objective is byte-identical to train_dual.py when
--card-weight 0. The card head is training-only -- never packed or deployed, so
the C++ kernel / pack / rdflib runtime are unaffected.

    cd ~/Projects/GBJOv2 && uv run python -u -m v3.train.train_dual_card \
        --sources cart,sib,addon --rank-sources sib,cart --rank-weight 3 \
        --cart-weight 10 --quantile-tau 0.6 --epochs 40 --batch 256 --lr 3e-4 \
        --device cpu --encoder gps --encoder-layers 2 --encoder-hidden 32 \
        --encoder-rel-emb-dim 16 --hidden 32 --layers 3 --weight-decay 1e-4 \
        --card-weight 0.3 --out v3/artifacts/models/card-on
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v3.core.model_dual import CostGNNDual
from v3.core.dual_data import collate
from v3.train.train_dual import (load_source, assemble, build_gid_atoms, build_rank_groups,
                        quantile_huber, ranking_terms, evaluate, evaluate_ranking,
                        save_plots, TRAIN_DIR)
from v3.core import subcost as sc

# --------------------------------------------------------------------------
# Per-join cardinality supervision
# --------------------------------------------------------------------------

def build_card_data(samples, gid2tr, cache):
    """Per-sample (join_local, memb_slot, memb_node, target, valid) tensors.
    target = natural-log of the join's subtree cardinality; valid masks joins
    whose card is absent/censored in the cache. memb_* map each subtree node to
    its join slot for the pooling sum."""
    out, n_valid, n_join = [], 0, 0
    for s in samples:
        ei, n = s[1], s[6]
        atoms = [sc.parse_atoms(t) for t in gid2tr[s[5]]]
        vs = sc.var_sets(atoms)
        subtrees = sc.join_subtrees(ei, n)
        joins = sorted(subtrees)                          # [n .. 2n-2]
        jl, tgt, val, mslot, mnode = [], [], [], [], []
        for slot, j in enumerate(joins):
            leaves, nodes = subtrees[j]
            card = sc.card_from_cache(leaves, atoms, vs, cache)
            ok = card is not None and card > 0
            jl.append(j)
            tgt.append(math.log(card) if ok else 0.0)
            val.append(ok)
            mslot += [slot] * len(nodes)
            mnode += nodes
            n_join += 1
            n_valid += int(ok)
        out.append((torch.tensor(jl, dtype=torch.long),
                    torch.tensor(mslot, dtype=torch.long),
                    torch.tensor(mnode, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.float),
                    torch.tensor(val, dtype=torch.bool)))
    print(f"  card targets: {n_valid}/{n_join} join nodes valid "
          f"({100 * n_valid / max(1, n_join):.1f}%)")
    return out


def batch_cards(card_list, n_nodes, device):
    """Offset + concat per-sample card tensors to the collated node order
    (collate concatenates samples' nodes in order with running offsets)."""
    jl, ms, mn, tg, vl = [], [], [], [], []
    noff = soff = 0
    for (j, mslot, mnode, tgt, val), nn_nodes in zip(card_list, n_nodes):
        jl.append(j + noff)
        ms.append(mslot + soff)
        mn.append(mnode + noff)
        tg.append(tgt)
        vl.append(val)
        noff += nn_nodes
        soff += j.numel()
    return (torch.cat(jl).to(device), torch.cat(ms).to(device),
            torch.cat(mn).to(device), torch.cat(tg).to(device),
            torch.cat(vl).to(device), soff)


def card_predict(card_head, h, jl, mslot, mnode, n_joins):
    """Per-join prediction from [h_j ; sum_{u in subtree(j)} h_u]."""
    hidden = h.shape[1]
    pooled = torch.zeros(n_joins, hidden, device=h.device, dtype=h.dtype)
    pooled = pooled.index_add_(0, mslot, h[mnode])
    return card_head(torch.cat([h[jl], pooled], dim=1)).squeeze(-1)


@torch.no_grad()
def card_eval(model, card_head, val, val_cards, device, embed, card_tau, bs=512):
    """Val intermediate-cardinality loss + (pred, true) pairs for the scatter."""
    model.eval()
    card_head.eval()
    gen = torch.Generator().manual_seed(123)
    tot, nb, pps, tts = 0.0, 0, [], []
    for i in range(0, len(val), bs):
        chunk, cl = val[i:i + bs], val_cards[i:i + bs]
        x, ei, esh, batch, _ = collate([(s[0], s[1], s[2], s[3]) for s in chunk],
                                       generator=gen)
        xd = embed(x, chunk) if embed else x.to(device)
        _, h = model(xd, ei.to(device), esh.to(device), batch.to(device),
                     num_graphs=len(chunk), return_nodes=True)
        jl, ms, mn, tg, vl, nj = batch_cards(cl, [s[0].shape[0] for s in chunk],
                                             device)
        cp = card_predict(card_head, h, jl, ms, mn, nj)
        if bool(vl.any()):
            tot += float(quantile_huber(cp[vl], tg[vl], card_tau))
            nb += 1
            pps.append(cp[vl].cpu().numpy())
            tts.append(tg[vl].cpu().numpy())
    return ((tot / nb) if nb else None,
            np.concatenate(pps) if pps else np.array([]),
            np.concatenate(tts) if tts else np.array([]))


def save_card_plots(out_dir, history, card_pred, card_true):
    plots = os.path.join(out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    l10 = math.log(10)
    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].plot(ep, [h.get("train_card") for h in history], label="train")
    axs[0].plot(ep, [h.get("val_card") for h in history], label="val")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("quantile-huber (log card)")
    axs[0].set_title("intermediate-cardinality loss"); axs[0].legend()
    if len(card_true):
        axs[1].scatter(card_true / l10, card_pred / l10, s=3, alpha=0.25,
                       color="tab:purple")
        lim = [min(card_true.min(), card_pred.min()) / l10 - 0.5,
               max(card_true.max(), card_pred.max()) / l10 + 0.5]
        axs[1].plot(lim, lim, "k--", lw=0.8)
        axs[1].set_xlim(lim); axs[1].set_ylim(lim)
    axs[1].set_xlabel("true log10 intermediate card")
    axs[1].set_ylabel("predicted log10 intermediate card")
    axs[1].set_title("per-join cardinality (val)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "card.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    # --- mirrors train_dual.py (kept in sync; card args appended) ---
    ap.add_argument("--sources", default="new3,addon,cart")
    ap.add_argument("--cart-data", default="v3/artifacts/plans/cart_plans_9_15")
    ap.add_argument("--sib-data", default="v3/artifacts/plans/sib_plans")
    ap.add_argument("--out", default="v3/artifacts/models/dual-card")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cart-weight", type=float, default=5.0)
    ap.add_argument("--quantile-tau", type=float, default=0.5)
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--rank-groups", type=int, default=8)
    ap.add_argument("--rank-every", type=int, default=1)
    ap.add_argument("--rank-sources", default="sib")
    ap.add_argument("--rank-alpha", type=float, default=1.0)
    ap.add_argument("--rank-beta", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    ap.add_argument("--encoder", default="off", choices=["off", "gine", "gps"])
    ap.add_argument("--encoder-pe", default="rwpe", choices=["rwpe", "lap", "none"])
    ap.add_argument("--encoder-pe-dim", type=int, default=24)
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--encoder-hidden", type=int, default=128)
    ap.add_argument("--encoder-rel-emb-dim", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--encoder-attn", default="multihead",
                    choices=["multihead", "performer"])
    ap.add_argument("--encoder-no-local-mp", action="store_true")
    ap.add_argument("--encoder-caps", default="16,8,4")
    ap.add_argument("--encoder-rdf2vec", action="store_true")
    ap.add_argument("--encoder-no-fanout", action="store_true")
    ap.add_argument("--encoder-no-rel-emb", action="store_true")
    ap.add_argument("--encoder-no-counts", action="store_true")
    ap.add_argument("--kg-index", default="v3/artifacts/index/wikidata")
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/rdflib-joinordering/gbjo_pack/wikidata"))
    ap.add_argument("--overfit-groups", type=int, default=0)
    # --- per-join cardinality supervision (flavor a) ---
    ap.add_argument("--card-weight", type=float, default=0.0,
                    help="weight of the auxiliary per-join cardinality loss "
                         "(0 = exactly train_dual.py)")
    ap.add_argument("--card-tau", type=float, default=0.5,
                    help="quantile for the card head's asymmetric Huber")
    ap.add_argument("--subcost-cache", default="v3/artifacts/cache/subcost_cache.json",
                    help="component-count cache from mine_subcosts.py")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed model init + fingerprint augmentation so the "
                         "card-on/off A/B differs only in the aux loss")
    args = ap.parse_args()
    card_on = args.card_weight > 0

    device = (("mps" if torch.backends.mps.is_available() else "cpu")
              if args.device == "auto" else args.device)
    print(f"device: {device}  card_on: {card_on} (w={args.card_weight})  "
          f"seed: {args.seed}")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    # dedicated generator for training fingerprints: identical across the A/B
    # regardless of how many global-RNG draws the card head's init consumes
    fp_gen = (torch.Generator().manual_seed(args.seed)
              if args.seed is not None else None)

    print("loading sources ...")
    raw = []
    for src in args.sources.split(","):
        raw.extend(load_source(src.strip(), args.cart_data, args.sib_data,
                               args.limit))
    samples, gids = assemble(raw)
    del raw
    gid2tr = {g: tr for tr, g in gids.items()}

    if args.overfit_groups:
        keep = set(sorted({s[5] for s in samples})[:args.overfit_groups])
        samples = [s for s in samples if s[5] in keep]
        gids = {tr: g for tr, g in gids.items() if g in keep}
        print(f"overfit probe: {len(samples)} samples / {len(keep)} groups")

    encoder, provider, embed = None, None, None
    if args.encoder != "off":
        from v3.core.kg_index import KGIndex
        from v3.core.term_encoder import SubgraphProvider, TermEncoder
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
            n_relations=kg.nR, rel_emb_dim=args.encoder_rel_emb_dim,
            dropout=args.dropout)
        gid_atoms = build_gid_atoms(gids, samples, kg)
        encoder = encoder.to(device)
        print(f"encoder: {args.encoder} {args.encoder_layers}x{args.encoder_hidden}, "
              f"{sum(p.numel() for p in encoder.parameters()):,} params")

        def embed(x, chunk):
            x = x.to(device)
            per_off = {0: ([], []), 102: ([], []), 204: ([], [])}
            nodes, node_pos, off_node = [], {}, 0
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
        train = val = samples
        train_idx = val_idx = list(range(len(samples)))
        print(f"overfit: train == val ({len(train)} samples)")
    else:
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

    rank_sources = set(args.rank_sources.split(","))
    rank_on = args.rank_weight > 0
    train_groups = build_rank_groups(train, range(len(train)), rank_sources)
    val_groups = build_rank_groups(val, range(len(val)), rank_sources)
    print(f"ranking groups: train {len(train_groups)} / val {len(val_groups)} "
          f"({'ON' if rank_on else 'off'} w={args.rank_weight})")
    if rank_on and not train_groups:
        sys.exit("--rank-weight > 0 but no ranking groups; check --sources")

    # per-join cardinality targets (only if the aux loss is on); no RNG here
    train_cards = val_cards = None
    if card_on:
        if not os.path.exists(args.subcost_cache):
            sys.exit(f"--card-weight>0 needs {args.subcost_cache} "
                     f"(run mine_subcosts.py)")
        cache = json.load(open(args.subcost_cache))
        print(f"subcost cache: {len(cache)} component counts")
        print("building train card targets ...")
        train_cards = build_card_data(train, gid2tr, cache)
        print("building val card targets ...")
        val_cards = build_card_data(val, gid2tr, cache)

    w = np.array([args.cart_weight if s[4] else 1.0 for s in train])
    w /= w.sum()

    model = CostGNNDual(node_feature_dim=307, hidden_dim=args.hidden,
                        n_layers=args.layers, dropout=args.dropout).to(device)
    # card head created AFTER the model so its init never perturbs the base
    # model's init (keeps the card-on/off A/B controlled under --seed)
    card_head = None
    if card_on:
        card_head = nn.Sequential(nn.Linear(2 * args.hidden, args.hidden),
                                  nn.GELU(),
                                  nn.Linear(args.hidden, 1)).to(device)
        print(f"card head: {sum(p.numel() for p in card_head.parameters()):,} params")
    params = list(model.parameters())
    if encoder is not None:
        params += list(encoder.parameters())
    if card_head is not None:
        params += list(card_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.lr_schedule == "cosine" else None)
    crit = lambda out, y: quantile_huber(out, y, args.quantile_tau)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": device, "model": "CostGNNDual",
                   "aux": "per-join cardinality" if card_on else "none"}, f, indent=2)

    best_val, best_acc, history = float("inf"), -1.0, []
    n_batches = (max(1, math.ceil(len(train) / args.batch)) if args.overfit_groups
                 else max(1, len(train) // args.batch))
    for epoch in range(args.epochs):
        model.train()
        if encoder is not None:
            encoder.train()
        if card_head is not None:
            card_head.train()
        t0 = time.time()
        tot_huber, tot_rank, n_rank, tot_card, n_card = 0.0, 0.0, 0, 0.0, 0
        if args.overfit_groups:
            order = rng.permutation(len(train))
        else:
            order = rng.choice(len(train), size=n_batches * args.batch, p=w)
        for b in range(n_batches):
            idx = order[b * args.batch:(b + 1) * args.batch]
            chunk = [train[i] for i in idx]
            x, ei, esh, batch, y = collate([(s[0], s[1], s[2], s[3])
                                            for s in chunk], generator=fp_gen)
            opt.zero_grad()
            xd = embed(x, chunk) if embed else x.to(device)
            if card_on:
                out, h = model(xd, ei.to(device), esh.to(device),
                               batch.to(device), num_graphs=len(chunk),
                               return_nodes=True)
            else:
                out = model(xd, ei.to(device), esh.to(device),
                            batch.to(device), num_graphs=len(chunk))
            loss = crit(out, y.to(device))
            tot_huber += loss.item()

            if card_on:
                jl, ms, mn, tg, vl, nj = batch_cards(
                    [train_cards[i] for i in idx],
                    [s[0].shape[0] for s in chunk], device)
                cp = card_predict(card_head, h, jl, ms, mn, nj)
                if bool(vl.any()):
                    lc = quantile_huber(cp[vl], tg[vl], args.card_tau)
                    loss = loss + args.card_weight * lc
                    tot_card += float(lc.detach())
                    n_card += 1

            if rank_on and b % args.rank_every == 0:
                gsel = rng.choice(len(train_groups),
                                  size=min(args.rank_groups, len(train_groups)),
                                  replace=False)
                ridx = np.concatenate([train_groups[g] for g in gsel])
                from v3.train.train_dual import collate_groups
                rx, rei, resh, rbatch, ry, rgid, rsz = collate_groups(train, ridx)
                rxd = (embed(rx, [train[i] for i in ridx]) if embed
                       else rx.to(device))
                rout = model(rxd, rei.to(device), resh.to(device),
                             rbatch.to(device), num_graphs=len(ridx))
                t, tw, _, _ = ranking_terms(rout, ry.to(device), rgid.to(device),
                                            rsz.to(device), args.rank_alpha,
                                            args.rank_beta)
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
            torch.mps.empty_cache()

        errs, val_huber, msg, (preds, trues, is_cart) = evaluate(
            model, val, device, embed=embed)
        val_rank, val_acc = (evaluate_ranking(model, val, val_groups, device,
                                              args.rank_alpha, args.rank_beta,
                                              embed=embed)
                             if val_groups else (None, None))
        val_card, cpred, ctrue = (card_eval(model, card_head, val, val_cards,
                                            device, embed, args.card_tau)
                                  if card_on else (None, np.array([]), np.array([])))
        history.append({
            "epoch": epoch,
            "train_huber": tot_huber / n_batches,
            "train_rank": (tot_rank / n_rank) if n_rank else None,
            "train_card": (tot_card / n_card) if n_card else None,
            "val_huber": val_huber,
            "val_rank": val_rank,
            "val_card": val_card,
            "val_med_qerr": float(np.median(errs)),
            "val_rank_acc": val_acc,
        })

        def save(tag):
            torch.save(model.state_dict(),
                       os.path.join(args.out, f"model{tag}.pt"))
            if encoder is not None:
                torch.save(encoder.state_dict(),
                           os.path.join(args.out, f"encoder{tag}.pt"))
            if card_head is not None:
                torch.save(card_head.state_dict(),
                           os.path.join(args.out, f"card_head{tag}.pt"))

        marker = ""
        if float(np.median(errs)) < best_val:
            best_val = float(np.median(errs))
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
        if card_on:
            save_card_plots(args.out, history, cpred, ctrue)

        rk = ""
        if n_rank:
            rk = f"rank {tot_rank/n_rank:.4f}"
            if val_rank is not None:
                rk += f"/{val_rank:.4f} acc {val_acc:.3f}"
            rk += "  "
        cd = f"card {tot_card/n_card:.4f}/{val_card:.4f}  " if n_card and val_card else ""
        print(f"epoch {epoch:>3}: huber {tot_huber/n_batches:.4f}/{val_huber:.4f}  "
              f"{rk}{cd}{msg}  ({time.time()-t0:.0f}s){marker}", flush=True)

    print(f"done; best val median qerr(log) {best_val:.3f}"
          + (f", best val pair acc {best_acc:.3f}" if best_acc >= 0 else "")
          + f"; model in {args.out}")


if __name__ == "__main__":
    main()
