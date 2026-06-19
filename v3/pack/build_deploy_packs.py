"""Pack a trained decoder into a deployable gbjo_pack/ directory (CLI).

Repacks the torch decoder -> model.npz, ships the kernel compiled for the
model's hidden dim, copies emb/keys/counts/meta from a source pack, and writes
the OneCycle schedule. n_layers and H are auto-detected from the repacked
weights. Optionally folds a finetune_mrt search_best.json (the learned
inner_lr_scale + penalty lambdas) into the deploy params -- inner_lr_scale folds
into learning_rate because OneCycleLR scales linearly with max_lr
(lr_scale*onecycle(lr) == onecycle(lr*lr_scale)).

    # pack an MRT decoder, folding its learned search params:
    uv run python -m v3.pack.build_deploy_packs \
        --model v3/artifacts/models/<run>-mrt/model_mrt.pt \
        --src  ~/rdflib-joinordering/gbjo_pack/<run>-mrtsrc \
        --out  <run>-mrt-deploy \
        --search-best v3/artifacts/models/<run>-mrt/search_best.json

    # pack a pretrained decoder with the source pack's params unchanged:
    uv run python -m v3.pack.build_deploy_packs \
        --model v3/artifacts/models/<run>/model_rank.pt \
        --src ~/rdflib-joinordering/gbjo_pack/<run>-mrtsrc --out <run>-pre-deploy
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

from v3.pack.overfit_e2e_setup import repack_model
from v3.core.gbjo_fast import onecycle_schedule
from v3 import paths

PACK_ROOT = os.path.expanduser("~/rdflib-joinordering/gbjo_pack")
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"


def ensure_kernel(H):
    """Compile (or reuse) the C++ kernel for hidden dim H; return its path.

    The kernel fixes H at compile time (-DGBJO_H); a binary built for a
    different H reads weights with the wrong stride and NaNs (see
    gbjo-kernel-hidden-dim-bug). Cached as libgbjo[_hH].<ext>, rebuilt when
    gbjo_kernel.cpp is newer than the binary so a kernel edit propagates."""
    src = os.path.join(HERE, "gbjo_kernel.cpp")
    os.makedirs(paths.LIB, exist_ok=True)
    out = os.path.join(str(paths.LIB), f"libgbjo{'' if H == 128 else f'_h{H}'}.{LIB_EXT}")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        if platform.system() == "Darwin":
            cmd = ["clang++", "-O3", "-std=c++17", "-dynamiclib",
                   "-framework", "Accelerate", "-DACCELERATE_NEW_LAPACK",
                   f"-DGBJO_H={H}", "-o", out, src]
        else:
            cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                   f"-DGBJO_H={H}", "-o", out, src, "-lopenblas"]
        subprocess.run(cmd, check=True)
        print(f"  compiled kernel H={H} -> {os.path.basename(out)}")
    return out


def build(model_pt, src, out, steps, meta_params):
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, "emb.npy")                    # symlink shared emb
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.join(src, "emb.npy"), link)
    for f in ("keys.txt", "counts.npy"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    repack_model(model_pt, os.path.join(out, "model.npz"))  # n_layers auto
    npz = np.load(os.path.join(out, "model.npz"))
    H, n_layers = int(npz["proj_w"].shape[0]), int(npz["W1"].shape[0])
    # ship a kernel compiled for THIS model's hidden dim (auto-built + cached)
    shutil.copy(ensure_kernel(H), os.path.join(out, f"libgbjo.{LIB_EXT}"))

    meta = json.load(open(os.path.join(src, "meta.json")))
    meta["params"].update(meta_params)
    meta["n_layers"] = n_layers
    meta["model"] = "/".join(model_pt.split("/")[-2:])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    lr = meta["params"]["learning_rate"]
    lrs, moms = onecycle_schedule(lr, steps)
    np.savez(os.path.join(out, "schedule.npz"),
             lrs=np.asarray(lrs, dtype=np.float64),
             moms=np.asarray(moms, dtype=np.float64))
    p = meta["params"]
    print(f"built {out}: n_layers={n_layers} H={H} lr={lr:.4f} "
          f"acyc={p['lambda_acyclic']:.3f} ll={p['lambda_left_linear']:.3f} "
          f"join_in={p['lambda_join_in']:.4f}")
    return out


def folded_params(search_best, base):
    """Fold a finetune_mrt search_best.json into deploy params: inner_lr_scale
    -> learning_rate (OneCycle is linear in max_lr); learned penalty lambdas
    overwrite their meta counterparts."""
    sb = json.load(open(os.path.expanduser(search_best)))
    p = {}
    if "inner_lr_scale" in sb:
        p["learning_rate"] = base["learning_rate"] * sb["inner_lr_scale"]
        print(f"  inner_lr_scale={sb['inner_lr_scale']:.4f} -> learning_rate "
              f"{base['learning_rate']:.4f} -> {p['learning_rate']:.4f}")
    for k in ("lambda_triple_in", "lambda_triple_out", "lambda_join_in",
              "lambda_join_out", "lambda_left_linear", "lambda_acyclic"):
        if k in sb:
            p[k] = sb[k]
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pack a trained decoder into a deployable gbjo_pack/ dir.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="a finetune_mrt deploy.json "
                   "(model + src_pack + search_best, self-contained)")
    g.add_argument("--model", help="decoder .pt to pack (use with --src)")
    ap.add_argument("--src", help="source pack: emb.npy/keys.txt/counts.npy/meta.json")
    ap.add_argument("--search-best", help="search_best.json to fold into deploy params")
    ap.add_argument("--out", help="output pack name under gbjo_pack/ or an absolute "
                    "path (default: <model run dir>-deploy)")
    ap.add_argument("--steps", type=int, default=10, help="OneCycle schedule steps")
    args = ap.parse_args()

    if args.config:
        cfg = json.load(open(os.path.expanduser(args.config)))
        model, src = cfg["model"], cfg["src_pack"]
        search_best = cfg.get("search_best")
        steps = cfg.get("steps", args.steps)
    else:
        if not args.src:
            ap.error("--model requires --src")
        model, src, search_best, steps = args.model, args.src, args.search_best, args.steps

    model, src = os.path.expanduser(model), os.path.expanduser(src)
    out_name = args.out or (os.path.basename(os.path.dirname(os.path.abspath(model)))
                            + "-deploy")
    out = out_name if os.path.isabs(out_name) else os.path.join(PACK_ROOT, out_name)
    base = json.load(open(os.path.join(src, "meta.json")))["params"]
    meta_params = folded_params(search_best, base) if search_best else {}
    build(model, src, out, steps, meta_params)
