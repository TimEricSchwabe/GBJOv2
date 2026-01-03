#!/usr/bin/env python3
"""
Sweep optimization_steps for iterative optimizers and plot median true cost vs steps.

- Dataset/config default: config_lubm_path (from src/evaluation_parallel.py)
- Methods swept (steps-dependent): GBJO, GEQO, IterativeImprovement, NeuralSort, CMA
  (DP and GreedySearch excluded by design: no tunable optimization_steps)
- Random is run once and reused as a constant baseline across all step values.

This script writes a single summary JSON (sweep_results.json) so plots can be
regenerated without rerunning evaluation.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add the parent directory to Python path (match other scripts in this repo)
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from src.create_data.create_optimization_data import SPARQLQuery


from src.evaluation_parallel import evaluate_optimization_parallel  # noqa: E402
from src.utils.data_utils import load_sparql_queries, filter_queries_by_max_uri_atoms  # noqa: E402
from src.visualization.plot_optimization_results import plot_optimization_steps_sweep  # noqa: E402


STEP_VALUES_DEFAULT = [10, 50, 100, 500]

# Map from evaluation_parallel plan keys to display names (used in plots/paper)
PLANKEY_TO_DISPLAY = {
    "gradient": "GBJO",
    "GEQO": "Genetic Search",
    "II": "Iterative Improvement",
    "NeuralSort": "Neural Sort",
    "CMA": "CMA",
    "random": "Random",
}


def _safe_nanmedian(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmedian(np.asarray(values, dtype=float)))


def _extract_real_costs(detailed_results: List[Dict[str, Any]], plan_key: str) -> Tuple[float, int]:
    vals: List[float] = []
    for r in detailed_results:
        try:
            v = r.get("plans", {}).get(plan_key, {}).get("real_cost", None)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not np.isfinite(fv):
            continue
        vals.append(fv)
    return _safe_nanmedian(vals), len(vals)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main(config):
    step_values = config.get("step_values", [10, 50, 100, 500])
    debug_timing = bool(config.get("debug_timing", False))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = config.get("save_root")
    if save_root is None:
        save_root = os.path.join(config["save_path"], "steps_sweep", f"run_{timestamp}")
    save_root = os.path.abspath(save_root)
    _ensure_dir(save_root)

    # Load queries once; reuse exact list across sweeps for comparability.
    sparql_queries = load_sparql_queries(config["queries_file"])
    max_uri_atoms = config.get("max_uri_atoms", 2)
    if max_uri_atoms is not None:
        sparql_queries = filter_queries_by_max_uri_atoms(sparql_queries, max_uri_atoms=max_uri_atoms)
    
    if config.get("num_queries") is not None:
        sparql_queries = sparql_queries[: config["num_queries"]]

    print(f"Saving sweep outputs to: {save_root}")
    print(f"Sweeping optimization_steps: {step_values}")
    print(f"Num queries: {len(sparql_queries)}")

    # Run Random once (baseline) and reuse.
    # Note: The new plotter doesn't automatically use this baseline folder yet,
    # but we generate it for completeness/future use.
    random_dir = _ensure_dir(os.path.join(save_root, "random_baseline"))
    evaluate_optimization_parallel(
        sparql_queries,
        config["model_path"],
        num_queries=len(sparql_queries),
        optimization_steps=step_values[0],  # unused for Random, but required by signature
        optimization_params=deepcopy(config.get("optimization_params", {})),
        optimization_algorithms=["Random"],
        save_directory=random_dir,
        use_exhaustive=False,
        use_true_costs=True,
        num_workers=config.get("num_workers", None),
        dp_limit=config.get("dp_limit", 9),
        model_params=config.get("model_params", None),
        debug_timing=debug_timing,
    )

    # Methods that actually depend on optimization_steps (DP/Greedy excluded).
    #sweep_algorithms = ["GBJO", "GEQO", "IterativeImprovement", "NeuralSort", "CMA"]
    sweep_algorithms = ["GEQO", "IterativeImprovement"]

    # Record metadata
    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "step_values": step_values,
        "dataset": config.get("dataset_name", "unknown"),
        "config": {
            # keep only the essentials for reproducing
            "queries_file": config["queries_file"],
            "model_path": config["model_path"],
            "num_queries": config.get("num_queries"),
            "num_workers": config.get("num_workers", None),
            "max_uri_atoms": config.get("max_uri_atoms", None),
            "optimization_params": config.get("optimization_params", {}),
            "model_params": config.get("model_params", None),
        },
    }

    # Save summary JSON (metadata only)
    results_file = os.path.join(save_root, "sweep_results.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    for steps in step_values:
        run_dir = _ensure_dir(os.path.join(save_root, f"steps_{steps}"))
        print(f"\n=== Running steps={steps} ===")
        t0 = time.perf_counter()
        evaluate_optimization_parallel(
            sparql_queries,
            config["model_path"],
            num_queries=len(sparql_queries),
            optimization_steps=steps,
            optimization_params=deepcopy(config.get("optimization_params", {})),
            optimization_algorithms=sweep_algorithms,
            save_directory=run_dir,
            use_exhaustive=False,
            use_true_costs=True,
            num_workers=config.get("num_workers", None),
            dp_limit=config.get("dp_limit", 9),
            model_params=config.get("model_params", None),
            debug_timing=debug_timing,
        )
        if debug_timing:
            dt = time.perf_counter() - t0
            print(f"=== steps={steps} total wall time: {dt:.3f}s ===")

    print(f"\nSaved sweep summary metadata to: {results_file}")

    # Plot
    plots_dir = _ensure_dir(os.path.join(save_root, "plots"))
    # Pass the ROOT DIRECTORY (save_root) instead of the summary file
    # This allows the plotter to crawl subdirectories and re-aggregate as needed.
    plot_optimization_steps_sweep(save_root, plots_dir)
    plot_optimization_steps_sweep(save_root, plots_dir, metric="mean")
    print(f"Done. Plots saved to {plots_dir}")


if __name__ == "__main__":
    config_lubm_path = {
        "dataset_name": "lubm_path",
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm/path-greedy/dataset.pt",
        "model_path": "/home/tim/query_optimization/training_results/lubm-path-log1p/model.pt",
        "num_queries": 20,
        "step_values": [10, 50, 100, 500],
        "use_exhaustive": False,
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": 10,
        "dp_limit": 9,
        "max_query_size": None,
        "max_uri_atoms": 2,
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": True,
            "use_layer_norm": False,
            "dropout": 0.0,
        },
        "optimization_params": {
            "k": 1,
            "learning_rate": 1.67, # todo 1.67
            "lambda_acyclic": 3.34,
            "lambda_triple_in": 1.36,
            "lambda_triple_out": 11.7,
            "lambda_join_in": 2.07,
            "lambda_join_out": 2.8,
            "lambda_entropy": 0.0,
            "lambda_total_penalty": 0.56,
            "lambda_left_linear": 51.7,
            "init_tau": 1.12,
            "min_tau": 0.12,
            "tau_decay": 0.963,
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 3.9,
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 1.31,
            "lr_warmup_steps": 200,
            "gradient_clip_norm": 3.09,
            "use_lr_scheduling": True,
            "decoding_method": "beam",
            "use_gumbel_noise": False,
            "use_swa": False,
            "gbjo_verbose": False,
        },
    }

    config_wikidata_star = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/wikidata_star_plan_datasets_optimization/queries.pkl",
        "model_path": "/home/tim/query_optimization/training_results/wikidata-star-log1p-add-aggr/model.pt", # current best: "/home/tim/query_optimization/training_results/wikidata-star-log1p-add-aggr/model.pt"
        "num_queries": 80,
        "step_values": [50, 500, 1000, 2500, 10000],
        "use_exhaustive": False,
        "use_dp": False,
        "dp_limit": 9,  # Set the limit here (e.g., 15 for star queries)
        "max_query_size": None,
        "max_uri_atoms": 2,
        "use_true_costs": True,
        "debug_timing": True,
        "save_path": "optimization_results",
        "num_workers": 10,  # Use all available cores
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": True,
            "use_layer_norm": False,
            "dropout": 0.0,
        },
        "optimization_params": {
            "k": 1,  # 1 Number of gradient optimization runs
            "learning_rate": 4.9, # 0.35 or 1; best 0.85; 3 or 50 timesteps
            "lambda_acyclic": 29, # 3391
            "lambda_triple_in": 1.5,# 3334.0
            "lambda_triple_out": 1.4,# 2026.0
            "lambda_join_in": 3.6, # 2150.0
            "lambda_join_out": 4.1,# 1295.0
            "lambda_entropy": 0.0,# 0.0
            "lambda_total_penalty": 0.99,# 0.7
            "lambda_left_linear": 60,# 2157.0
            "init_tau": 4, # 15
            "min_tau": 0.49, # 1.0
            "tau_decay": 0.973,
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 9.96,
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 1.01, # 5.3 best: 1.09
            "lr_warmup_steps": 46,
            "gradient_clip_norm": 4.7,
            "use_lr_scheduling": True,
            "decoding_method": "beam",
            "use_gumbel_noise": False,
            "gbjo_verbose": False,
        }
    }

    config_lubm_star = {
        "queries_file": "/home/tim/query_optimization/datasets/plans/lubm/star-greedy/dataset.pt", # /home/tim/query_optimization/datasets/plans/lubm_star_plan_datasets_optimization/optimization_stars_3_to_14/queries.pkl
        "model_path": "/home/tim/query_optimization/training_results/lubm-star-log1p/model.pt", # /home/tim/query_optimization/datasets/models/lubm/6-layers-v3-with-layer-norm/model.pt
        "num_queries": 100,
        "max_query_size": None,  # Filter queries larger than this (None for no filter)
        "step_values": [10, 50, 100, 500, 1000],
        "use_exhaustive": False,
        "use_dp": True,
        "dp_limit": 9,  # Set the limit here (e.g., 15 for star queries)
        "use_true_costs": True,
        "save_path": "optimization_results",
        "num_workers": 6,  # Use all available cores
        "model_params": {
            "version": "v3",
            "hidden_dim": 128,
            "node_feature_dim": 307,
            "n_layers": 6,
            "use_jk": False,
            "jk_mode": "cat",
            "use_residual": True,
            "use_layer_norm": False,
            "dropout": 0.0,
        },
        "optimization_params": { # params for GBJO
            "k": 1,  # Number of gradient optimization runs
            "learning_rate": 2.26, # 1.7
            "lambda_acyclic": 24.5, # 3081.0
            "lambda_triple_in": 13.5, # 3714.0
            "lambda_triple_out": 60.7, # 135.0
            "lambda_join_in": 18.8, # 1742.0
            "lambda_join_out": 9.3, # 1558.0
            "lambda_entropy": 0.0,
            "lambda_total_penalty": 0.99, # 2.6
            "lambda_left_linear": 28.8, # 2300.0
            "init_tau": 3.2,
            "min_tau": 0.12, #1.0
            "tau_decay": 0.963,
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 3.9, # 5
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 1.7, # 6.5
            "lr_warmup_steps": 50,
            "gradient_clip_norm": 4.1,
            "use_lr_scheduling": True,
            "decoding_method": "beam",
            "use_gumbel_noise": False,
            "use_swa": False,
            "gbjo_verbose": False
        }
    }

    
    # Run the sweep
    main(config_wikidata_star)


