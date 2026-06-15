"""
Offline encoding step: run a trained term encoder (train_dual.py --encoder)
over query terms and write the resulting embeddings as a pack-compatible
emb.npy, aligned to an existing pack's keys.txt. The rdflib runtime and the
C++ kernel consume it unchanged -- this is FICE's offline/online split.

Terms are taken from query JSON files (entries with a 'triples' key; triples
either '<s> <p> ?o.' strings or [s, p, o] lists). Rows of keys not encoded
stay zero (count-only behavior). Use --all to encode the full vocabulary
instead (slow: ~5ms/term).

    uv run python standalone/encode_pack.py \
        --train-out standalone/models/dual-enc-v1 \
        --queries data/queries/wikidata/path/path_queries.json \
        --pack-in ~/rdflib-joinordering/gbjo_pack/wikidata \
        --out-emb /tmp/emb_encoded.npy

The decoder side of the pack (model.npz, schedule.npz) still comes from
pack_gbjo.py on the retrained model_rank.pt.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_index import KGIndex
from term_encoder import SubgraphProvider, TermEncoder


def iter_bound_terms(triple):
    if isinstance(triple, str):
        s, p, o = triple.split(" ", 2)
        o = o.rstrip()
        if o.endswith("."):
            o = o[:-1].rstrip()
    else:
        s, p, o = triple[:3]
    for term, kind in ((s, "ent"), (p, "rel"), (o, "ent")):
        if not term.startswith("?"):
            yield term, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", required=True,
                    help="train_dual.py --out dir (encoder_rank.pt + config.json)")
    ap.add_argument("--encoder-file", default="encoder_rank.pt")
    ap.add_argument("--queries", nargs="*", default=[],
                    help="query JSON files whose bound terms get encoded")
    ap.add_argument("--all", action="store_true",
                    help="encode the entire vocabulary instead")
    ap.add_argument("--pack-in", required=True,
                    help="existing pack (keys.txt alignment + rdf2vec input)")
    ap.add_argument("--out-emb", required=True, help="output emb.npy path")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(os.path.join(args.train_out, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("encoder", "off") == "off":
        sys.exit("config.json says --encoder off; nothing to encode")

    print(f"loading KG index {cfg['kg_index']} ...")
    kg = KGIndex.load(cfg["kg_index"])
    caps = tuple(int(c) for c in cfg["encoder_caps"].split(","))
    # rdf2vec is opt-in now (--encoder-rdf2vec); fall back to the old opt-out
    # key for checkpoints trained before the flip.
    use_rdf2vec = cfg.get("encoder_rdf2vec",
                          not cfg.get("encoder_no_rdf2vec", False))
    provider = SubgraphProvider(
        kg, pe=cfg["encoder_pe"], pe_dim=cfg["encoder_pe_dim"], caps=caps,
        pack_dir=args.pack_in, use_rdf2vec=use_rdf2vec)
    use_fanout = not cfg.get("encoder_no_fanout", False) and kg.max_out is not None
    encoder = TermEncoder(
        hidden=cfg["encoder_hidden"], out_dim=100,
        n_layers=cfg["encoder_layers"], arch=cfg["encoder"],
        pe=cfg["encoder_pe"], pe_dim=cfg["encoder_pe_dim"],
        use_rdf2vec=use_rdf2vec,
        use_counts=not cfg["encoder_no_counts"],
        attn=cfg.get("encoder_attn", "multihead"),
        local_mp=not cfg.get("encoder_no_local_mp", False),
        use_fanout=use_fanout,
        rel_emb=not cfg.get("encoder_no_rel_emb", False),
        n_relations=kg.nR,
        rel_emb_dim=cfg.get("encoder_rel_emb_dim", 0)).to(args.device)
    encoder.load_state_dict(torch.load(
        os.path.join(args.train_out, args.encoder_file), map_location="cpu"))
    encoder.eval()

    with open(os.path.join(args.pack_in, "keys.txt"), encoding="utf-8") as f:
        pack_keys = f.read().splitlines()
    pack_row = {k: i for i, k in enumerate(pack_keys)}

    def term_pack_row(term):
        # pack keys are plain URIs; query/KG terms are '<uri>' tokens
        # (the rdflib runtime strips the brackets the same way, gbjo.py)
        if term.startswith("<") and term.endswith(">"):
            term = term[1:-1]
        return pack_row.get(term)

    # (pack row, factor-graph node id) per distinct encodable term
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
            with open(path) as f:
                qs = json.load(f)
            for q in qs:
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
    print(f"encoding {len(rows):,} distinct terms "
          f"(of {len(pack_keys):,} pack keys)")

    emb = np.zeros((len(pack_keys), 100), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), args.chunk):
            chunk = rows[i:i + args.chunk]
            out = encoder(**provider.batch([todo[r] for r in chunk],
                                           args.device))
            emb[np.array(chunk)] = out.cpu().numpy()
            if (i // args.chunk) % 20 == 19:
                done = i + len(chunk)
                print(f"  {done:,}/{len(rows):,} "
                      f"({done/(time.time()-t0):.0f} terms/s)", flush=True)
    np.save(args.out_emb, emb)
    print(f"saved {args.out_emb} {emb.shape} ({time.time()-t0:.0f}s)")


def iter_bound_terms_all(q):
    for t in q["triples"]:
        yield from iter_bound_terms(t)


if __name__ == "__main__":
    main()
