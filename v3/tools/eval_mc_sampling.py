"""
Eval-mode probe of Monte-Carlo neighborhood sampling for the FICE term encoder
(fice_sampling.pdf, Eq 7: M-sample averaging).

The trained encoder builds ONE subgraph per term, seeded by the node id and
cached, so the model sees the exact same capped neighborhood every epoch. This
script replaces that single fixed draw with M *independent stochastic* draws
(same per-hop cap budget as training, fresh seed each draw) and averages the M
encoder outputs before the decoder (Eq 7). We then read off how the metrics
move vs. the deterministic baseline as M grows.

What this does NOT do: the doc's Horvitz-Thompson reweight-to-full-neighborhood
(Eq 5). The model was trained on cap-budget *subset* sums with no reweighting;
reweighting to the full neighborhood would scale messages by ~deg/k and wreck a
model that never saw it (the doc's own train/inference-consistency caveat).
Testing reweighting fairly needs a retrain.

Usage:
    OMP_NUM_THREADS=3 uv run python -m v3.tools.eval_mc_sampling \
        --ckpt rank --n-groups 100 --samples 1,2,4,8,16,32
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import torch

from v3.core.kg_index import KGIndex, ROLE_TO_IDX
from v3.core.model_dual import CostGNNDual
from v3.core.term_encoder import SubgraphProvider, TermEncoder, compute_pe
from v3.train.train_dual import (load_source, assemble, build_gid_atoms,
                        build_rank_groups, evaluate, evaluate_ranking)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seeded_sample_subgraph(kg, node, caps, seed):
    """kg_index.KGIndex.sample_subgraph with an explicit rng seed (the only
    change from the original, which hardcodes seed=node). seed=node therefore
    reproduces the cached training subgraph byte-for-byte."""
    rng = np.random.default_rng(seed)
    indptr, indices, roles = kg.indptr, kg.indices, kg.roles
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
            else:
                sel = (rng.choice(deg, size=cap, replace=False) + lo)
            ul = pos[u]
            for e in sel:
                v = int(indices[e])
                r = int(roles[e])
                vl = pos.get(v)
                if vl is None:
                    vl = pos[v] = len(nodes)
                    nodes.append(v)
                    nxt.append(v)
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
    return np.array(nodes, dtype=np.int64), edge_index, role_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="v3/artifacts/models/star-v2-reg")
    ap.add_argument("--ckpt", default="rank", choices=["rank", "best", "last"],
                    help="rank=model_rank.pt (best val rank-acc), best=model.pt "
                         "(best val q-err), last=model_last.pt")
    ap.add_argument("--n-groups", type=int, default=100,
                    help="number of held-out query groups to evaluate on")
    ap.add_argument("--samples", default="1,2,4,8,16,32",
                    help="comma list of M (encoder draws averaged, Eq 7)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = args.device
    Ms = [int(m) for m in args.samples.split(",")]

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    tag = {"rank": "_rank", "best": "", "last": "_last"}[args.ckpt]

    # snapshot the live-training checkpoints so we don't race the writer
    tmp = tempfile.mkdtemp(prefix="mc_eval_")
    shutil.copy(os.path.join(args.model_dir, f"model{tag}.pt"),
                os.path.join(tmp, "model.pt"))
    shutil.copy(os.path.join(args.model_dir, f"encoder{tag}.pt"),
                os.path.join(tmp, "encoder.pt"))
    print(f"ckpt: model{tag}.pt / encoder{tag}.pt  (snapshot -> {tmp})")

    # ---- data: reproduce train_dual's group-wise 90/10 val split exactly ----
    print("loading star plans ...")
    raw = []
    for src in cfg["sources"].split(","):
        raw.extend(load_source(src.strip(), cfg["cart_data"], cfg["sib_data"]))
    samples, gids = assemble(raw)
    del raw
    rng = np.random.default_rng(0)
    all_gids = sorted({s[5] for s in samples})
    gperm = rng.permutation(len(all_gids))
    n_val_g = max(1, int(0.1 * len(all_gids)))
    val_gids = {all_gids[i] for i in gperm[:n_val_g]}
    # first N val groups (sorted by gid) -> the eval subset
    keep = set(sorted(val_gids)[:args.n_groups])
    val = [s for s in samples if s[5] in keep]
    gids_val = {tr: g for tr, g in gids.items() if g in keep}
    rank_groups = build_rank_groups(val, range(len(val)), {"star"})
    print(f"eval subset: {len(val)} plans / {len(keep)} query groups, "
          f"{len(rank_groups)} ranking groups")

    # ---- encoder + decoder from config, load snapshot weights ----
    kg = KGIndex.load(cfg["kg_index"])
    caps = tuple(int(c) for c in cfg["encoder_caps"].split(","))
    use_fanout = (not cfg.get("encoder_no_fanout", False)) and kg.max_out is not None
    use_rdf2vec = cfg.get("encoder_rdf2vec", False)
    provider = SubgraphProvider(kg, pe=cfg["encoder_pe"],
                                pe_dim=cfg["encoder_pe_dim"], caps=caps,
                                pack_dir=cfg["pack"], use_rdf2vec=use_rdf2vec)
    encoder = TermEncoder(
        hidden=cfg["encoder_hidden"], out_dim=100,
        n_layers=cfg["encoder_layers"], arch=cfg["encoder"],
        pe=cfg["encoder_pe"], pe_dim=cfg["encoder_pe_dim"],
        use_rdf2vec=use_rdf2vec, use_counts=not cfg.get("encoder_no_counts", False),
        attn=cfg["encoder_attn"], local_mp=not cfg.get("encoder_no_local_mp", False),
        use_fanout=use_fanout, rel_emb=not cfg.get("encoder_no_rel_emb", False),
        n_relations=kg.nR, rel_emb_dim=cfg["encoder_rel_emb_dim"],
        dropout=0.0).to(device)
    encoder.load_state_dict(torch.load(os.path.join(tmp, "encoder.pt"),
                                       map_location=device))
    encoder.eval()
    model = CostGNNDual(node_feature_dim=307, hidden_dim=cfg["hidden"],
                        n_layers=cfg["layers"], dropout=0.0).to(device)
    model.load_state_dict(torch.load(os.path.join(tmp, "model.pt"),
                                     map_location=device))
    model.eval()
    gid_atoms = build_gid_atoms(gids_val, val, kg)
    print(f"encoder caps {caps}, {sum(p.numel() for p in encoder.parameters()):,} params")

    # ---- monkeypatch the sampler to honor a mutable per-draw seed ----
    state = {"draw": 0}  # 0 = deterministic (seed=node, = training subgraph)
    kg.sample_subgraph = lambda node, c: seeded_sample_subgraph(
        kg, node, c, node if state["draw"] == 0 else node * 1_000_003 + state["draw"])

    # ---- encode each (term, draw) ONCE; every M then averages from the pool.
    # A term's embedding is independent of batch composition, so one batch over
    # all unique terms == the per-chunk path. The M rows share one fixed draw
    # sequence -> a true running average (monotone convergence), not a fresh
    # independent draw set per M.
    unique_nodes = sorted({nid for g in keep for _, _, nid in gid_atoms[g]})
    node_idx = {nid: i for i, nid in enumerate(unique_nodes)}
    Mmax = max(Ms)
    print(f"\nencoding {len(unique_nodes)} unique bound terms x {Mmax + 1} draws "
          f"(1 baseline + {Mmax} MC) ...")

    @torch.no_grad()
    def encode_draw(d):
        state["draw"] = d
        provider.cache.clear()
        return encoder(**provider.batch(unique_nodes, device)).cpu()  # (N,100)

    t0 = time.time()
    base_emb = encode_draw(0)                       # seed=node = training graph
    pool = torch.empty(Mmax, len(unique_nodes), 100)
    for d in range(1, Mmax + 1):
        pool[d - 1] = encode_draw(d)
        print(f"  draw {d}/{Mmax} ({time.time() - t0:.0f}s)", flush=True)
    print(f"encoded in {time.time() - t0:.0f}s")

    def make_embed(emb):
        """Splice a precomputed (N,100) term-embedding matrix into x (no
        encoder call); emb row i corresponds to unique_nodes[i]."""
        emb = emb.to(device)

        def embed(x, chunk):
            x = x.to(device)
            per_off = {0: ([], []), 102: ([], []), 204: ([], [])}
            off_node = 0
            for s in chunk:
                for row, off, nid in gid_atoms[s[5]]:
                    rows, sel = per_off[off]
                    rows.append(off_node + row); sel.append(node_idx[nid])
                off_node += s[0].shape[0]
            for off, (rows, sel) in per_off.items():
                if rows:
                    r = torch.tensor(rows, device=device)
                    c = torch.arange(off + 1, off + 101, device=device)
                    x = x.index_put((r.unsqueeze(1), c.unsqueeze(0)),
                                    emb[torch.tensor(sel, device=device)])
            return x
        return embed

    alpha, beta = cfg["rank_alpha"], cfg["rank_beta"]

    def run(label, M, emb):
        emb_fn = make_embed(emb)
        errs, val_huber, _, _ = evaluate(model, val, device, embed=emb_fn)
        _, acc = evaluate_ranking(model, val, rank_groups, device, alpha, beta,
                                  embed=emb_fn)
        row = {"label": label, "M": M, "med_qerr": float(np.median(errs)),
               "p95_qerr": float(np.percentile(errs, 95)),
               "mean_qerr": float(np.mean(errs)),
               "huber": val_huber, "rank_acc": acc}
        print(f"  {label:>10}: med_q {row['med_qerr']:.4f}  "
              f"p95_q {row['p95_qerr']:.4f}  huber {row['huber']:.4f}  "
              f"rank_acc {acc:.4f}", flush=True)
        return row

    print("\nbaseline = deterministic training subgraph (seed=node)")
    print("M=k      = running average of the first k stochastic draws (Eq 7)\n")
    rows = [run("baseline", 1, base_emb)]
    for M in Ms:
        rows.append(run(f"M={M}", M, pool[:M].mean(0)))

    out = os.path.join(args.model_dir, "mc_sampling_eval.json")
    json.dump({"ckpt": f"model{tag}.pt", "n_groups": len(keep),
               "n_plans": len(val), "caps": list(caps), "rows": rows},
              open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
