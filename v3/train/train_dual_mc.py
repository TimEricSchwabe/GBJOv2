"""
Small training tests of Monte-Carlo neighborhood sampling for the FICE term
encoder (fice_sampling.pdf). One config per run, on a subsampled set of star
query groups so it runs fast.

Pool: before training we fix `--mc-draws` (default 8) STOCHASTIC subgraph
structures per term (caps from --caps, fresh seed per draw). The expensive
sampling+PE is cached once; the encoder still re-runs each step (weights move).

Configs (--mc-m / --reweight):
  M=8 : average the encoder over all 8 draws each step (Eq 7).
  M=4 : average a random 4-of-8 each step.
  M=1 : a random 1-of-8 each step (fresh resampling; SGD averages over epochs).
  --reweight full : Horvitz-Thompson reweight (Eq 5). In this factor graph the
        sampled neighbors of an entity/relation are all triples of equal
        degree, so alpha and the per-role Z_r drop out and the HT weight on a
        node v's pooled GIN messages collapses to deg_full(v)/k = max(1,
        deg_full(v)/cap). Triple/leaf/fully-enumerated nodes keep weight 1.
        Run at M=1 (the unbiased single sample).

Usage (4 tests):
  for cfg in "1 none" "4 none" "8 none" "1 full"; do set -- $cfg; \
    uv run python -m v3.train.train_dual_mc --mc-m $1 --reweight $2 \
      --max-groups 400 --epochs 40 --out v3/artifacts/models/mc-m$1-$2; done
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from v3.core.dual_data import collate
from v3.core.kg_index import KGIndex, ROLE_TO_IDX
from v3.core.model_dual import CostGNNDual
from v3.core.term_encoder import SubgraphProvider, TermEncoder, compute_pe
from v3.train.train_dual import (load_source, assemble, build_gid_atoms,
                        build_rank_groups, collate_groups, quantile_huber,
                        ranking_terms, evaluate, evaluate_ranking, save_plots)


def sample_subgraph_mc(kg, node, caps, seed):
    """KGIndex.sample_subgraph with an explicit seed, also returning per-node
    HT reweight factors. node_w[i] = max(1, deg_full/cap) for nodes that were
    EXPANDED (their out-neighborhood was subsampled at the cap), else 1.0."""
    rng = np.random.default_rng(seed)
    indptr, indices, roles = kg.indptr, kg.indices, kg.roles
    pos = {node: 0}
    nodes = [node]
    node_w = [1.0]
    e_src, e_dst, e_role = [], [], []
    frontier = [node]
    for cap in caps:
        nxt = []
        for u in frontier:
            lo, hi = int(indptr[u]), int(indptr[u + 1])
            deg = hi - lo
            node_w[pos[u]] = max(1.0, deg / cap)   # =1 when deg<=cap (full)
            sel = range(lo, hi) if deg <= cap else \
                (rng.choice(deg, size=cap, replace=False) + lo)
            ul = pos[u]
            for e in sel:
                v = int(indices[e]); r = int(roles[e])
                vl = pos.get(v)
                if vl is None:
                    vl = pos[v] = len(nodes)
                    nodes.append(v); node_w.append(1.0); nxt.append(v)
                e_src.append(ul); e_dst.append(vl); e_role.append(ROLE_TO_IDX[r])
                e_src.append(vl); e_dst.append(ul); e_role.append(ROLE_TO_IDX[-r])
        frontier = nxt
    edge_index = np.array([e_src, e_dst], dtype=np.int64)
    if edge_index.shape[1]:
        key = (edge_index[0] * len(nodes) + edge_index[1]) * 8 + np.array(e_role)
        _, keep = np.unique(key, return_index=True)
        edge_index = edge_index[:, keep]
        role_idx = np.array(e_role, dtype=np.int64)[keep]
    else:
        role_idx = np.zeros(0, dtype=np.int64)
    return (np.array(nodes, dtype=np.int64), edge_index, role_idx,
            np.array(node_w, dtype=np.float32))


# ---- predicate-stratified + disjoint-block sampling (NEIGHBORHOOD_SAMPLING.md) -

def triple_pred(kg, triple_ids):
    """Predicate (relation node id) of each triple node = its role-2 neighbor.
    A triple node has exactly 3 out-edges (roles 1,2,3), so base..base+2 are
    its s/p/o edges; pick the one with role==2."""
    base = kg.indptr[triple_ids]
    r, ix = kg.roles, kg.indices
    r0, r1, r2 = r[base], r[base + 1], r[base + 2]
    i0, i1, i2 = ix[base], ix[base + 1], ix[base + 2]
    return np.where(r0 == 2, i0, np.where(r1 == 2, i1, i2)).astype(np.int64)


def _allocate(cap, sizes, beta):
    """Tempered-proportional quota: q_h proportional to N_h**beta, rounded by
    largest remainder to sum to `cap`, clipped to stratum size.
      beta = 1 -> q_h proportional to N_h  == uniform over triples (sanity anchor)
      beta = 0 -> equal per stratum        == full flatten (starves the hub)
      beta in (0,1) -> mild hub-dampening  (cf. word2vec freq**0.75)
    The earlier ">=1 per predicate" guarantee is dropped: it is what starved the
    dominant predicate (NEIGHBORHOOD_SAMPLING.md sec. 10); beta<1 lifts rare
    strata smoothly without forcing it."""
    w = sizes.astype(np.float64) ** beta
    target = cap * w / w.sum()
    q = np.floor(target).astype(np.int64)
    left = int(cap - q.sum())
    if left > 0:
        q[np.argsort(-(target - q))[:left]] += 1
    return np.minimum(q, sizes)


def build_partition(kg, u, lo, hi, cap, n_blocks, beta):
    """Group u's incident triples by predicate, shuffle each stratum (seed=u),
    allocate a per-block quota proportional to N_h**beta, and keep only the head
    each stratum needs for n_blocks disjoint draws. Strata with quota 0 are
    dropped. See NEIGHBORHOOD_SAMPLING.md sec. 8 & 10."""
    cand = np.arange(lo, hi, dtype=np.int64)
    pred = triple_pred(kg, kg.indices[cand].astype(np.int64))
    rng = np.random.default_rng(int(u))
    perm = rng.permutation(len(cand))               # shuffle once, seeded by u
    cand_s, pred_s = cand[perm], pred[perm]
    order = np.argsort(pred_s, kind="stable")       # group by pred, keep shuffle
    cand_s, pred_s = cand_s[order], pred_s[order]
    uniq, start = np.unique(pred_s, return_index=True)
    sizes = np.diff(np.append(start, len(pred_s)))
    quota = _allocate(cap, sizes, beta)
    keep = np.minimum(sizes, n_blocks * quota)
    groups, quotas = [], []
    for h in range(len(uniq)):
        q = int(quota[h])
        if q:
            groups.append(cand_s[start[h]:start[h] + int(keep[h])])
            quotas.append(q)
    return {"groups": groups, "quota": quotas}


def take_block(part, block):
    """Select this block's edges from a cached partition. Each retained stratum
    contributes its quota q_h items at offset block*q_h into the shuffled head,
    so blocks 0..B-1 are disjoint where the head is long enough (hub strata) and
    wrap on short strata (the selective predicates we want in every draw)."""
    out = []
    for g, q in zip(part["groups"], part["quota"]):
        L = len(g)
        if L and q:
            s = (block * q) % L
            out.append(g[(s + np.arange(q)) % L])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def sample_subgraph_strat(kg, node, caps, block, n_blocks, pcache, beta):
    """Capped k-hop neighborhood where each over-cap expansion of an
    entity/relation node draws block `block` of a predicate-stratified partition
    (cached per expanded node, so the B blocks are disjoint). reweight is off for
    this sampler, so node weights are all 1."""
    indptr, indices, roles = kg.indptr, kg.indices, kg.roles
    nER = kg.nE + kg.nR
    pos = {node: 0}
    nodes = [node]
    e_src, e_dst, e_role = [], [], []
    frontier = [node]
    for cap in caps:
        nxt = []
        for u in frontier:
            lo, hi = int(indptr[u]), int(indptr[u + 1])
            deg = hi - lo
            if deg <= cap:
                sel = range(lo, hi)
            elif u < nER:                            # entity/relation -> triples
                part = pcache.get(u)
                if part is None:
                    part = pcache[u] = build_partition(kg, u, lo, hi, cap,
                                                       n_blocks, beta)
                sel = take_block(part, block)
            else:                                    # triple deg>cap can't happen
                rng = np.random.default_rng(u * n_blocks + block)
                sel = rng.choice(deg, size=cap, replace=False) + lo
            ul = pos[u]
            for e in sel:
                e = int(e)
                v = int(indices[e]); r = int(roles[e])
                vl = pos.get(v)
                if vl is None:
                    vl = pos[v] = len(nodes)
                    nodes.append(v); nxt.append(v)
                e_src.append(ul); e_dst.append(vl); e_role.append(ROLE_TO_IDX[r])
                e_src.append(vl); e_dst.append(ul); e_role.append(ROLE_TO_IDX[-r])
        frontier = nxt
    edge_index = np.array([e_src, e_dst], dtype=np.int64)
    if edge_index.shape[1]:
        key = (edge_index[0] * len(nodes) + edge_index[1]) * 8 + np.array(e_role)
        _, keep = np.unique(key, return_index=True)
        edge_index = edge_index[:, keep]
        role_idx = np.array(e_role, dtype=np.int64)[keep]
    else:
        role_idx = np.zeros(0, dtype=np.int64)
    return (np.array(nodes, dtype=np.int64), edge_index, role_idx,
            np.ones(len(nodes), dtype=np.float32))


class MCSubgraphProvider(SubgraphProvider):
    """Pool of `n_draws` stochastic subgraphs per term (cached by (node, draw)),
    with optional per-edge HT weights. set_draw(d) selects the active draw."""

    def __init__(self, kg, reweight=False, sampler="uniform", n_blocks=8,
                 strat_beta=0.75, **kw):
        super().__init__(kg, **kw)
        self.reweight = reweight
        self.sampler = sampler
        self.n_blocks = n_blocks
        self.strat_beta = strat_beta
        self._pcache = {}          # per-node predicate partition (stratified)
        self._draw = 1

    def set_draw(self, d):
        self._draw = d

    def _entry(self, node):
        key = (node, self._draw)
        e = self.cache.get(key)
        if e is None:
            if self.sampler == "stratified":
                node_ids, edge_index, role_idx, node_w = sample_subgraph_strat(
                    self.kg, node, self.caps, self._draw - 1, self.n_blocks,
                    self._pcache, self.strat_beta)
            else:
                seed = node * 1_000_003 + self._draw
                node_ids, edge_index, role_idx, node_w = sample_subgraph_mc(
                    self.kg, node, self.caps, seed)
            occ, ntype = self.kg.node_features(node_ids)
            ei = torch.from_numpy(edge_index)
            ri = torch.from_numpy(role_idx)
            pe = compute_pe(ei, ri, len(node_ids), self.pe_kind, self.pe_dim)
            is_rel = (node_ids >= self.kg.nE) & (node_ids < self.kg.nE + self.kg.nR)
            rel_local = (node_ids - self.kg.nE)
            fanout = np.zeros((len(node_ids), 2), dtype=np.float32)
            rel_id = np.zeros(len(node_ids), dtype=np.int64)
            if is_rel.any():
                ri_local = rel_local[is_rel]
                rel_id[is_rel] = ri_local + 1
                if self.kg.max_out is not None:
                    fanout[is_rel, 0] = np.log1p(self.kg.max_out[ri_local])
                    fanout[is_rel, 1] = np.log1p(self.kg.max_in[ri_local])
            emb_rows = None
            if self.emb is not None:
                er = node_ids < (self.kg.nE + self.kg.nR)
                rows = np.full(len(node_ids), -1, dtype=np.int64)
                rows[er] = self.pack_rows[node_ids[er]]
                emb_rows = torch.from_numpy(rows)
            ew = (torch.from_numpy(node_w[edge_index[1]])
                  if (self.reweight and edge_index.shape[1]) else None)
            e = self.cache[key] = (
                torch.from_numpy(occ), torch.from_numpy(fanout), pe.float(),
                emb_rows, torch.from_numpy(ntype), torch.from_numpy(rel_id),
                ei, ri, ew, len(node_ids))
        return e

    def batch(self, node_ids, device):
        occs, fans, pes, embs, ntypes, relids, eis, ris, ews, bvec, centers = \
            [], [], [], [], [], [], [], [], [], [], []
        off = 0
        for gi, node in enumerate(node_ids):
            occ, fanout, pe, emb_rows, ntype, rel_id, ei, ri, ew, n = \
                self._entry(node)
            occs.append(occ); fans.append(fanout); pes.append(pe)
            ntypes.append(ntype); relids.append(rel_id)
            eis.append(ei + off); ris.append(ri)
            if ew is not None:
                ews.append(ew)
            bvec.append(torch.full((n,), gi, dtype=torch.long))
            centers.append(off)
            if self.emb is not None:
                rows = emb_rows.clamp(min=0)
                ee = self.emb[rows]; ee[emb_rows < 0] = 0.0
                embs.append(ee)
            off += n
        occ = torch.cat(occs).to(device)
        n_tot = occ.shape[0]
        center = torch.zeros(n_tot, 1)
        center_pos = torch.tensor(centers, dtype=torch.long)
        center[center_pos] = 1.0
        return dict(
            occ=occ, fanout=torch.cat(fans).to(device),
            pe=(torch.cat(pes) if self.pe_dim else
                torch.zeros(n_tot, 0)).to(device),
            emb=(torch.cat(embs).to(device) if self.emb is not None
                 else torch.zeros(n_tot, 0, device=device)),
            center=center.to(device), ntype=torch.cat(ntypes).to(device),
            rel_id=torch.cat(relids).to(device),
            edge_index=torch.cat(eis, dim=1).to(device),
            role_idx=torch.cat(ris).to(device),
            batch_vec=torch.cat(bvec).to(device),
            center_pos=center_pos.to(device),
            edge_weight=(torch.cat(ews).to(device) if self.reweight else None),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-m", type=int, default=1, help="draws averaged per step")
    ap.add_argument("--mc-draws", type=int, default=8, help="pool size per term")
    ap.add_argument("--reweight", default="none", choices=["none", "full"])
    ap.add_argument("--sampler", default="uniform",
                    choices=["uniform", "stratified"],
                    help="stratified: predicate-stratified + disjoint blocks "
                         "(NEIGHBORHOOD_SAMPLING.md); reweight must be none")
    ap.add_argument("--strat-beta", type=float, default=0.75,
                    help="stratified tempering: q_h ~ N_h**beta. 1=uniform over "
                         "triples (anchor), 0=flatten, 0<beta<1=mild hub-dampen")
    ap.add_argument("--caps", default="10,6,10,6")
    ap.add_argument("--max-groups", type=int, default=400,
                    help="subsample this many query groups (fast tests)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0, help="init + subsample seed")
    ap.add_argument("--out", default="v3/artifacts/models/mc-test")
    ap.add_argument("--sib-data", default="v3/artifacts/plans/sib_star_plans")
    ap.add_argument("--kg-index", default="v3/artifacts/index/wikidata")
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/rdflib-joinordering/gbjo_pack/wikidata"))
    # fixed architecture (matches the star-v2-reg run)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--encoder-hidden", type=int, default=32)
    ap.add_argument("--encoder-pe-dim", type=int, default=24)
    ap.add_argument("--encoder-rel-emb-dim", type=int, default=16)
    ap.add_argument("--quantile-tau", type=float, default=0.6)
    ap.add_argument("--rank-weight", type=float, default=3.0)
    ap.add_argument("--rank-groups", type=int, default=8)
    ap.add_argument("--rank-alpha", type=float, default=1.0)
    ap.add_argument("--rank-beta", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    args = ap.parse_args()
    device = "cpu"
    caps = tuple(int(c) for c in args.caps.split(","))
    print(f"config: M={args.mc_m} sampler={args.sampler}"
          f"{f'(beta={args.strat_beta})' if args.sampler == 'stratified' else ''} "
          f"reweight={args.reweight} caps={caps} draws={args.mc_draws} "
          f"groups={args.max_groups} epochs={args.epochs}")

    # ---- data: subsample query groups, then group-wise 90/10 split ----
    raw = load_source(f"star={args.sib_data}", None, None)
    samples, gids = assemble(raw)
    del raw
    sub = np.random.default_rng(args.seed)
    all_gids = sorted({s[5] for s in samples})
    keep = set(sub.choice(all_gids, size=min(args.max_groups, len(all_gids)),
                          replace=False).tolist())
    samples = [s for s in samples if s[5] in keep]
    gids = {tr: g for tr, g in gids.items() if g in keep}
    kept_gids = sorted(keep)
    gperm = sub.permutation(len(kept_gids))
    n_val_g = max(1, int(0.1 * len(kept_gids)))
    val_gids = {kept_gids[i] for i in gperm[:n_val_g]}
    train = [s for s in samples if s[5] not in val_gids]
    val = [s for s in samples if s[5] in val_gids]
    train_groups = build_rank_groups(train, range(len(train)), {"star"})
    val_groups = build_rank_groups(val, range(len(val)), {"star"})
    print(f"train {len(train)} / val {len(val)} plans "
          f"({len(kept_gids) - n_val_g}/{n_val_g} groups); "
          f"rank groups train {len(train_groups)} / val {len(val_groups)}")

    # ---- encoder + decoder (same init across configs via --seed) ----
    kg = KGIndex.load(args.kg_index)
    use_fanout = kg.max_out is not None
    provider = MCSubgraphProvider(
        kg, reweight=(args.reweight == "full"), sampler=args.sampler,
        n_blocks=args.mc_draws, strat_beta=args.strat_beta, pe="rwpe",
        pe_dim=args.encoder_pe_dim, caps=caps, pack_dir=args.pack,
        use_rdf2vec=False)
    torch.manual_seed(args.seed)
    encoder = TermEncoder(
        hidden=args.encoder_hidden, out_dim=100, n_layers=args.encoder_layers,
        arch="gps", pe="rwpe", pe_dim=args.encoder_pe_dim, use_rdf2vec=False,
        use_counts=True, use_fanout=use_fanout, rel_emb=True, n_relations=kg.nR,
        rel_emb_dim=args.encoder_rel_emb_dim, dropout=0.0).to(device)
    model = CostGNNDual(node_feature_dim=307, hidden_dim=args.hidden,
                        n_layers=args.layers, dropout=0.0).to(device)
    gid_atoms = build_gid_atoms(gids, samples, kg)
    print(f"encoder {sum(p.numel() for p in encoder.parameters()):,} params; "
          f"fanout={use_fanout}")

    rng_draw = np.random.default_rng(args.seed + 7)
    pool = list(range(1, args.mc_draws + 1))
    eval_draws = pool[:args.mc_m]   # deterministic first-M for a stable val signal

    def train_draws():
        return (pool if args.mc_m >= args.mc_draws else
                rng_draw.choice(pool, size=args.mc_m, replace=False).tolist())

    def embed(x, chunk, draws):
        """Average the encoder over `draws` (Eq 7) and splice into x."""
        x = x.to(device)
        per_off = {0: ([], []), 102: ([], []), 204: ([], [])}
        nodes, node_pos, off_node = [], {}, 0
        for s in chunk:
            for row, off, nid in gid_atoms[s[5]]:
                p = node_pos.get(nid)
                if p is None:
                    p = node_pos[nid] = len(nodes); nodes.append(nid)
                rows, sel = per_off[off]
                rows.append(off_node + row); sel.append(p)
            off_node += s[0].shape[0]
        if not nodes:
            return x
        E_sum = None
        for d in draws:
            provider.set_draw(d)
            E = encoder(**provider.batch(nodes, device))
            E_sum = E if E_sum is None else E_sum + E
        E = E_sum / len(draws)
        for off, (rows, sel) in per_off.items():
            if rows:
                r = torch.tensor(rows, device=device)
                c = torch.arange(off + 1, off + 101, device=device)
                x = x.index_put((r.unsqueeze(1), c.unsqueeze(0)),
                                E[torch.tensor(sel, device=device)])
        return x

    emb_eval = lambda x, c: embed(x, c, eval_draws)

    params = list(model.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = lambda out, y: quantile_huber(out, y, args.quantile_tau)

    os.makedirs(args.out, exist_ok=True)
    json.dump({**vars(args), "caps": list(caps), "model": "CostGNNDual+MC"},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)

    rng = np.random.default_rng(args.seed)
    n_batches = max(1, len(train) // args.batch)
    best_acc, best_val, history = -1.0, float("inf"), []
    for epoch in range(args.epochs):
        model.train(); encoder.train()
        t0 = time.time()
        tot_huber, tot_rank, n_rank = 0.0, 0.0, 0
        order = rng.permutation(len(train))
        order = np.resize(order, n_batches * args.batch)
        for b in tqdm(range(n_batches), desc=f"epoch {epoch}", leave=False,
                      mininterval=5):
            idx = order[b * args.batch:(b + 1) * args.batch]
            chunk = [train[i] for i in idx]
            x, ei, esh, batch, y = collate([(s[0], s[1], s[2], s[3])
                                            for s in chunk])
            opt.zero_grad()
            out = model(embed(x, chunk, train_draws()), ei.to(device),
                        esh.to(device), batch.to(device), num_graphs=len(chunk))
            loss = crit(out, y.to(device))
            tot_huber += loss.item()
            if train_groups:
                gsel = rng.choice(len(train_groups),
                                  size=min(args.rank_groups, len(train_groups)),
                                  replace=False)
                ridx = np.concatenate([train_groups[g] for g in gsel])
                rx, rei, resh, rb, ry, rgid, rsz = collate_groups(train, ridx)
                rout = model(embed(rx, [train[i] for i in ridx], train_draws()),
                             rei.to(device), resh.to(device), rb.to(device),
                             num_graphs=len(ridx))
                t, tw, _, _ = ranking_terms(rout, ry.to(device), rgid.to(device),
                                            rsz.to(device), args.rank_alpha,
                                            args.rank_beta)
                if float(tw) > 0:
                    rloss = t / tw
                    loss = loss + args.rank_weight * rloss
                    tot_rank += float(rloss.detach()); n_rank += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
        sched.step()

        errs, val_huber, msg, (preds, trues, is_cart) = evaluate(
            model, val, device, embed=emb_eval)
        val_rank, val_acc = evaluate_ranking(
            model, val, val_groups, device, args.rank_alpha, args.rank_beta,
            embed=emb_eval) if val_groups else (None, None)
        val_med = float(np.median(errs))
        history.append({"epoch": epoch, "train_huber": tot_huber / n_batches,
                        "train_rank": (tot_rank / n_rank) if n_rank else None,
                        "val_huber": val_huber, "val_rank": val_rank,
                        "val_med_qerr": val_med, "val_rank_acc": val_acc})

        def save(tag):
            torch.save(model.state_dict(), os.path.join(args.out, f"model{tag}.pt"))
            torch.save(encoder.state_dict(), os.path.join(args.out, f"encoder{tag}.pt"))
        if val_med < best_val:
            best_val = val_med; save("")
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc; save("_rank")
        save("_last")
        json.dump(history, open(os.path.join(args.out, "history.json"), "w"))
        save_plots(args.out, epoch, preds, trues, is_cart, history)
        print(f"epoch {epoch:>3}: huber {tot_huber/n_batches:.4f}/{val_huber:.4f}"
              f"  rank {(tot_rank/n_rank) if n_rank else 0:.4f}/"
              f"{val_rank if val_rank else 0:.4f} acc {val_acc:.4f}  "
              f"med_q {val_med:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"done; best val med_q {best_val:.4f} acc {best_acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
