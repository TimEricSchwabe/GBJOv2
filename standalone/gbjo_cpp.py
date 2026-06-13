"""ctypes wrapper for the fused C++ GBJO kernel.

Build (macOS):
    clang++ -O3 -std=c++17 -dynamiclib -framework Accelerate \
        -DACCELERATE_NEW_LAPACK -o standalone/libgbjo.dylib standalone/gbjo_kernel.cpp
Build (Linux):
    g++ -O3 -std=c++17 -shared -fPIC -o standalone/libgbjo.so \
        standalone/gbjo_kernel.cpp -lopenblas
"""

import ctypes
import os
import sys

# must be set before OpenBLAS initializes: threading tiny gemms hurts badly
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch

from gbjo_fast import FastGBJO, FlatCostGNN, beam_exact

_EXT = "dylib" if sys.platform == "darwin" else "so"
_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libgbjo.{_EXT}")

_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
_i8p = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")


def _load_lib():
    lib = ctypes.CDLL(_LIB)
    lib.gbjo_create.restype = ctypes.c_void_p
    lib.gbjo_create.argtypes = [ctypes.c_int, _f32p, _f32p, _f32p, _f32p, _f32p,
                                _f32p, _f32p, _f32p, ctypes.c_float]
    lib.gbjo_create_dual.restype = ctypes.c_void_p
    lib.gbjo_create_dual.argtypes = [ctypes.c_int, _f32p, _f32p, _f32p, _f32p,
                                     _f32p, _f32p, _f32p, ctypes.c_float]
    lib.gbjo_destroy.argtypes = [ctypes.c_void_p]
    lib.gbjo_optimize.restype = ctypes.c_double
    lib.gbjo_optimize.argtypes = [
        ctypes.c_void_p, _f32p, _f32p, ctypes.c_int, ctypes.c_int,
        _f64p, _f64p, _f64p, _f64p, _f64p,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, _i8p, _i8p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    return lib


class CppGBJO:
    """Same interface as FastGBJO.optimize, backed by the C++ kernel."""

    def __init__(self, model: FlatCostGNN, params=None):
        self.model = model
        self.dual = bool(getattr(model, "is_dual", False))
        self._py = FastGBJO(model, params=params)  # reuse schedules + beam0 + params
        self.params = self._py.params
        if self.params.get("lambda_cartesian", 0.0):
            raise NotImplementedError(
                "C++ kernel has no cartesian penalty; use lambda_cartesian=0")
        self.lib = _load_lib()

        n_layers = len(model.layers)
        fc1 = np.ascontiguousarray(model.fc1_w.numpy())
        b_fc1 = np.ascontiguousarray(model.fc1_b.numpy())
        fc2 = np.ascontiguousarray(model.fc2_w.numpy().reshape(-1))
        b_fc2 = float(model.fc2_b.item())
        if self.dual:
            # FlatCostGNNDual layers: (w1 (H, 2H), b1, w2, b2), eps = 0
            W1 = np.ascontiguousarray(np.stack([l[0].numpy() for l in model.layers]))
            B1 = np.ascontiguousarray(np.stack([l[1].numpy() for l in model.layers]))
            W2 = np.ascontiguousarray(np.stack([l[2].numpy() for l in model.layers]))
            B2 = np.ascontiguousarray(np.stack([l[3].numpy() for l in model.layers]))
            self._keepalive = (W1, B1, W2, B2, fc1, b_fc1, fc2)
            self.ctx = self.lib.gbjo_create_dual(n_layers, W1, B1, W2, B2,
                                                 fc1, b_fc1, fc2, b_fc2)
        else:
            eps = np.array([l[0] for l in model.layers], dtype=np.float32)
            W1 = np.ascontiguousarray(np.stack([l[1].numpy() for l in model.layers]))
            B1 = np.ascontiguousarray(np.stack([l[2].numpy() for l in model.layers]))
            W2 = np.ascontiguousarray(np.stack([l[3].numpy() for l in model.layers]))
            B2 = np.ascontiguousarray(np.stack([l[4].numpy() for l in model.layers]))
            self._keepalive = (eps, W1, B1, W2, B2, fc1, b_fc1, fc2)
            self.ctx = self.lib.gbjo_create(n_layers, eps, W1, B1, W2, B2,
                                            fc1, b_fc1, fc2, b_fc2)
        self._lambdas = np.array([
            self.params["lambda_triple_in"], self.params["lambda_triple_out"],
            self.params["lambda_join_in"], self.params["lambda_join_out"],
            self.params["lambda_acyclic"], self.params["lambda_left_linear"],
            self.params["lambda_entropy"],
        ], dtype=np.float64)

    def _step0_plan(self, n, beam_width):
        """Step-0 soft adjacency is query-independent (zero logits); project it
        with the reference (Python) beam so step-0 tie-breaking matches."""
        key = (n, beam_width)
        plan = self._py._beam0_cache.get(key)
        if plan is None:
            N = 2 * n - 1
            mask, _ = self._py._size_consts(n)
            W = torch.softmax(torch.zeros(2 * n - 2, n - 1) + mask, dim=1)
            A = torch.zeros(N, N)
            A[: 2 * n - 2, n:] = W
            plan = beam_exact(A.numpy(), beam_width=beam_width)
            self._py._beam0_cache[key] = plan
        return np.ascontiguousarray(plan, dtype=np.int8)

    def optimize(self, x, optimization_steps=10, share=None, lex=False,
                 mask_cart=False):
        """lex=True: pick the candidate with fewest cartesian joins, then
        lowest predicted cost (penalty-consistent selection; dual only).
        mask_cart=True: forbid cartesian joins during beam decoding (dual
        only), so the candidate pool is cartesian-free whenever possible."""
        n = (x.shape[0] + 1) // 2
        N = 2 * n - 1
        lrs, moms, taus, lts = self._py._schedule(optimization_steps)
        with torch.no_grad():
            h0 = self.model.project_x(x)
        h0 = np.ascontiguousarray(h0.numpy(), dtype=np.float32)
        if self.dual:
            if share is None:
                raise ValueError("dual model requires share=sharing_matrix(triples)")
            S = np.ascontiguousarray((share.numpy() > 0).astype(np.float32))
        else:
            if lex or mask_cart:
                raise ValueError(
                    "lex/mask_cart require a dual model (share matrix)")
            S = np.zeros((N, N), dtype=np.float32)  # ignored by the kernel

        beam_width = int(self.params["discrete_beam_width"])
        step0 = self._step0_plan(n, beam_width)
        out = np.zeros((N, N), dtype=np.int8)
        log_cost = self.lib.gbjo_optimize(
            self.ctx, h0, S, n, optimization_steps,
            np.asarray(lrs, dtype=np.float64), np.asarray(moms, dtype=np.float64),
            np.asarray(taus, dtype=np.float64), np.asarray(lts, dtype=np.float64),
            self._lambdas, float(self.params["gradient_clip_norm"]),
            beam_width, int(lex), int(mask_cart), step0, out,
            0, None, None, None,  # no candidate-pool export
        )
        return out.astype(int), float(np.exp(log_cost))

    def __del__(self):
        try:
            self.lib.gbjo_destroy(self.ctx)
        except Exception:
            pass
