"""
Set up the end-to-end rdflib comparison on the EXACT overfit-probe queries.

1. extract the 100 sib query BGPs (first 100 gids, same selection as
   --overfit-groups 100) as [s, p, o] atom lists -> overfit_queries.json
2. assemble three runtime packs (off / gine / gps) that share keys/counts/
   schedule/meta/dylib with the existing wikidata pack (symlinked) and carry
   a freshly packed decoder model.npz from each overfit model_rank.pt.
   emb.npy: off symlinks the rdf2vec matrix; the encoder packs get theirs
   from encode_pack.py (run separately).

    cd ~/Projects/GBJOv2 && uv run python -m v3.pack.overfit_e2e_setup
"""

import json
import os
import sys

import numpy as np
import torch

from v3.train.train_dual import assemble, load_source

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXIST = os.path.expanduser("~/rdflib-joinordering/gbjo_pack/wikidata")
PACK_ROOT = os.path.expanduser("~/rdflib-joinordering/gbjo_pack")
SHARED = ["keys.txt", "counts.npy", "schedule.npz", "meta.json"]


def parse_triple(ts):
    """'<s> <p> ?o.' -> ('<s>', '<p>', '?o') (atoms the planner expects)."""
    return tuple(ts.rstrip(" .").split(" "))


def extract_queries(out_json, n_groups=100):
    raw = load_source("sib", "v3/artifacts/plans/cart_plans_9_15", "v3/artifacts/plans/sib_plans")
    samples, gids = assemble(raw)
    keep = sorted({s[5] for s in samples})[:n_groups]
    keepset = set(keep)
    # gids: triples-tuple -> gid; invert for the kept gids
    queries = []
    for tr, g in gids.items():
        if g in keepset:
            triples = [list(parse_triple(t)) for t in tr]
            queries.append({"gid": g, "triples": triples})
    queries.sort(key=lambda q: q["gid"])
    with open(out_json, "w") as f:
        json.dump(queries, f, indent=1)
    print(f"wrote {len(queries)} queries -> {out_json}")
    nbound = sum(1 for q in queries for t in q["triples"]
                 for a in t if not a.startswith("?"))
    print(f"  {nbound} bound atoms total, sizes "
          f"{sorted({len(q['triples']) for q in queries})}")
    return queries


def repack_model(model_pt, out_npz, n_layers=None):
    sd = torch.load(model_pt, map_location="cpu")
    if n_layers is None:                           # infer from the checkpoint
        n_layers = 1 + max(int(k.split(".")[1])
                           for k in sd if k.startswith("mlps."))
    np.savez(
        out_npz,
        proj_w=sd["projection.weight"].numpy(),
        proj_b=sd["projection.bias"].numpy(),
        W1=np.stack([sd[f"mlps.{i}.0.weight"].numpy() for i in range(n_layers)]),
        B1=np.stack([sd[f"mlps.{i}.0.bias"].numpy() for i in range(n_layers)]),
        W2=np.stack([sd[f"mlps.{i}.2.weight"].numpy() for i in range(n_layers)]),
        B2=np.stack([sd[f"mlps.{i}.2.bias"].numpy() for i in range(n_layers)]),
        fc1=sd["fc1.weight"].numpy(),
        b_fc1=sd["fc1.bias"].numpy(),
        fc2=sd["fc2.weight"].numpy().reshape(-1),
        b_fc2=np.float32(sd["fc2.bias"].item()),
    )


def symlink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build_pack(variant):
    out = os.path.join(PACK_ROOT, f"overfit-{variant}")
    os.makedirs(out, exist_ok=True)
    for name in SHARED:
        symlink(os.path.join(EXIST, name), os.path.join(out, name))
    dylib = next(f for f in os.listdir(EXIST) if f.endswith(".dylib"))
    symlink(os.path.join(EXIST, dylib), os.path.join(out, dylib))
    repack_model(f"v3/artifacts/models/overfit-{variant}/model_rank.pt",
                 os.path.join(out, "model.npz"))
    if variant == "off":
        symlink(os.path.join(EXIST, "emb.npy"), os.path.join(out, "emb.npy"))
        print(f"built pack {out} (emb.npy -> rdf2vec)")
    else:
        print(f"built pack {out} (model.npz only; run encode_pack for emb.npy)")
    return out


if __name__ == "__main__":
    extract_queries("v3/artifacts/queries/overfit_queries.json")
    for v in ("off", "gine", "gps"):
        build_pack(v)
