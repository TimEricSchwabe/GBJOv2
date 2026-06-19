"""
Offline encoding for an MC-trained term encoder (train_dual_mc.py). Same job as
encode_pack.py -- run the frozen encoder over query terms and write a
pack-aligned emb.npy -- but (i) reads the train_dual_mc config schema (caps as a
list, sampler/mc_m/mc_draws, no encoder_* opt-out keys) and (ii) serves the term
embedding as the MEAN over the first `mc_m` MC draws, exactly the `emb_eval` the
model's rank-acc was selected on. A single deterministic draw would feed the
decoder an input distribution the M=4 encoder never saw.

    uv run python -m v3.pack.encode_pack_mc \
        --train-out v3/artifacts/models/star-v2-m4 \
        --queries v3/artifacts/queries/overfit_queries.json \
        --pack-in ~/rdflib-joinordering/gbjo_pack/wikidata \
        --out-emb ~/rdflib-joinordering/gbjo_pack/star-v2-m4/emb.npy
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from v3.core.kg_index import KGIndex
from v3.core.term_encoder import TermEncoder
from v3.train.train_dual_mc import MCSubgraphProvider
from v3.pack.encode_pack import iter_bound_terms_all, complete_src_and_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", required=True, help="train_dual_mc run dir")
    ap.add_argument("--encoder-file", default="encoder_rank.pt")
    ap.add_argument("--queries", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pack-in", required=True, help="pack whose keys.txt aligns rows")
    ap.add_argument("--out-emb", required=True)
    ap.add_argument("--mc-m", type=int, default=None,
                    help="draws averaged (default: config mc_m); matches emb_eval")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.train_out, "config.json")))
    mc_m = args.mc_m if args.mc_m is not None else cfg["mc_m"]
    caps = tuple(int(c) for c in cfg["caps"])
    draws = list(range(1, mc_m + 1))            # deterministic first-M = emb_eval
    print(f"loading KG index {cfg['kg_index']} ...")
    kg = KGIndex.load(cfg["kg_index"])
    use_fanout = kg.max_out is not None
    provider = MCSubgraphProvider(
        kg, reweight=(cfg["reweight"] == "full"), sampler=cfg["sampler"],
        n_blocks=cfg["mc_draws"], strat_beta=cfg.get("strat_beta", 0.75),
        pe="rwpe", pe_dim=cfg["encoder_pe_dim"], caps=caps,
        pack_dir=cfg["pack"], use_rdf2vec=False)
    encoder = TermEncoder(
        hidden=cfg["encoder_hidden"], out_dim=100, n_layers=cfg["encoder_layers"],
        arch="gps", pe="rwpe", pe_dim=cfg["encoder_pe_dim"], use_rdf2vec=False,
        use_counts=True, use_fanout=use_fanout, rel_emb=True, n_relations=kg.nR,
        rel_emb_dim=cfg["encoder_rel_emb_dim"], dropout=0.0).to(args.device)
    encoder.load_state_dict(torch.load(
        os.path.join(args.train_out, args.encoder_file), map_location="cpu"))
    encoder.eval()
    print(f"sampler={cfg['sampler']} mc_m={mc_m} (draws {draws}) caps={caps}")

    with open(os.path.join(args.pack_in, "keys.txt"), encoding="utf-8") as f:
        pack_keys = f.read().splitlines()
    pack_row = {k: i for i, k in enumerate(pack_keys)}

    def term_pack_row(term):
        if term.startswith("<") and term.endswith(">"):
            term = term[1:-1]
        return pack_row.get(term)

    todo = {}
    if args.all:
        for keys, idx in ((kg.ent_keys, kg.ent_idx), (kg.rel_keys, kg.rel_idx)):
            for k in keys:
                r = term_pack_row(k)
                if r is not None and r not in todo:
                    todo[r] = idx[k]
    else:
        if not args.queries:
            sys.exit("pass --queries ... or --all")
        missing = 0
        for path in args.queries:
            for q in json.load(open(path)):
                for term, kind in iter_bound_terms_all(q):
                    r = term_pack_row(term)
                    nid = kg.node_id(term, kind)
                    if r is None or nid < 0:
                        missing += 1
                    elif r not in todo:
                        todo[r] = nid
        if missing:
            print(f"  {missing} term occurrences not in pack/KG (skipped)")
    rows = list(todo.keys())
    print(f"encoding {len(rows):,} distinct terms (of {len(pack_keys):,} pack keys)")

    emb = np.zeros((len(pack_keys), 100), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), args.chunk):
            chunk = rows[i:i + args.chunk]
            nodes = [todo[r] for r in chunk]
            acc = None
            for d in draws:                       # average the eval draws (Eq. 7)
                provider.set_draw(d)
                out = encoder(**provider.batch(nodes, args.device))
                acc = out if acc is None else acc + out
            emb[np.array(chunk)] = (acc / len(draws)).cpu().numpy()
            if (i // args.chunk) % 10 == 9:
                done = i + len(chunk)
                print(f"  {done:,}/{len(rows):,} "
                      f"({done/(time.time()-t0):.0f} terms/s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_emb)), exist_ok=True)
    np.save(args.out_emb, emb)
    print(f"saved {args.out_emb} {emb.shape} ({time.time()-t0:.0f}s)")
    complete_src_and_manifest(args.train_out, args.pack_in, args.out_emb,
                              args.encoder_file)


if __name__ == "__main__":
    main()
