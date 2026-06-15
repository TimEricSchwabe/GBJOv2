"""Build two deploy packs for the full-v2-reg-mrt-20k validation.

  full-v2-reg-mrt-20k-deploy : ep97 MRT decoder + FOLDED learned hyperparams.
      inner_lr_scale folds into learning_rate (OneCycleLR scales linearly with
      max_lr, so lr_scale*onecycle(lr) == onecycle(lr*lr_scale)); the learned
      penalty lambdas overwrite meta["params"].
  full-v2-reg-pre-deploy     : pretrained decoder + ORIGINAL deploy params.

Both share the 20k-src encoder emb.npy (the encoder is frozen during MRT) via a
symlink, and both fix meta["n_layers"]=3 (full-v2-reg is 3 layers; the runtime
passes meta["n_layers"] straight to the kernel, the 20k-src meta said 6).

    cd ~/Projects/GBJOv2 && uv run python standalone/build_deploy_packs.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overfit_e2e_setup import repack_model
from gbjo_fast import onecycle_schedule

PACK_ROOT = os.path.expanduser("~/rdflib-joinordering/gbjo_pack")
SRC = os.path.join(PACK_ROOT, "full-v2-reg-mrtsrc-20k")   # emb/keys/counts/meta
STEPS = 10
N_LAYERS = 3
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"


def ensure_kernel(H):
    """Compile (or reuse) the C++ kernel for hidden dim H; return its path.

    The kernel fixes H at compile time (-DGBJO_H); a binary built for a
    different H reads weights with the wrong stride and NaNs (see
    gbjo-kernel-hidden-dim-bug). Cached as libgbjo[_hH].<ext>, rebuilt when
    gbjo_kernel.cpp is newer than the binary so a kernel edit propagates."""
    src = os.path.join(HERE, "gbjo_kernel.cpp")
    out = os.path.join(HERE, f"libgbjo{'' if H == 128 else f'_h{H}'}.{LIB_EXT}")
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


def build(out_name, model_pt, meta_params):
    out = os.path.join(PACK_ROOT, out_name)
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, "emb.npy")                    # symlink shared emb
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.join(SRC, "emb.npy"), link)
    for f in ("keys.txt", "counts.npy"):
        shutil.copy(os.path.join(SRC, f), os.path.join(out, f))
    repack_model(model_pt, os.path.join(out, "model.npz"))  # n_layers auto
    # ship a kernel compiled for THIS model's hidden dim (auto-built + cached)
    H = int(np.load(os.path.join(out, "model.npz"))["proj_w"].shape[0])
    shutil.copy(ensure_kernel(H), os.path.join(out, f"libgbjo.{LIB_EXT}"))

    meta = json.load(open(os.path.join(SRC, "meta.json")))
    meta["params"].update(meta_params)
    meta["n_layers"] = N_LAYERS
    meta["model"] = "/".join(model_pt.split("/")[-2:])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    lr = meta["params"]["learning_rate"]
    lrs, moms = onecycle_schedule(lr, STEPS)
    np.savez(os.path.join(out, "schedule.npz"),
             lrs=np.asarray(lrs, dtype=np.float64),
             moms=np.asarray(moms, dtype=np.float64))
    p = meta["params"]
    print(f"built {out_name}: n_layers={N_LAYERS} lr={lr:.4f} "
          f"acyc={p['lambda_acyclic']:.3f} ll={p['lambda_left_linear']:.3f} "
          f"join_in={p['lambda_join_in']:.4f}")


if __name__ == "__main__":
    sb = json.load(open("standalone/models/full-v2-reg-mrt-20k/search_best.json"))
    base = json.load(open(os.path.join(SRC, "meta.json")))["params"]
    lr_scale = sb["inner_lr_scale"]
    print(f"inner_lr_scale={lr_scale:.4f}  ->  learning_rate "
          f"{base['learning_rate']} -> {base['learning_rate']*lr_scale:.4f}")
    new_params = {
        "learning_rate": base["learning_rate"] * lr_scale,
        "lambda_triple_in": sb["lambda_triple_in"],
        "lambda_triple_out": sb["lambda_triple_out"],
        "lambda_join_in": sb["lambda_join_in"],
        "lambda_join_out": sb["lambda_join_out"],
        "lambda_left_linear": sb["lambda_left_linear"],
        "lambda_acyclic": sb["lambda_acyclic"],
    }
    build("full-v2-reg-mrt-20k-deploy",
          "standalone/models/full-v2-reg-mrt-20k/model_mrt.pt", new_params)
    build("full-v2-reg-pre-deploy",
          "standalone/models/full-v2-reg/model_rank.pt", {})
