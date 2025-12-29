"""
Hyperparameter search results visualization.

This script loads a JSON "history" file that contains entries like:
[
  {"iteration": 0, "loss": 123.4, "params": {"learning_rate": 0.1, ...}},
  ...
]

It then generates a set of plots:
- loss over time
- best loss over time (running min)
- learning_rate vs lambda_acyclic scatter (colored by loss)
- parameter importance / sensitivity plots (correlations, RF importance, etc.)

No CLI args: edit RESULTS_FILE / OUTPUT_DIR below if needed, then run:
python -m src.visualization.hyperparam_results
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


# ----------------------------- configuration ---------------------------------

# Relative to repo root (resolved from this file's location).
RESULTS_FILE = "hpo_results/cma_20251223_195203/history.json"
OUTPUT_DIR = "hpo_results/cma_20251223_195203/plots"

# Plot style defaults
FIG_DPI = 150
SEED = 42


# ------------------------------ utilities ------------------------------------


def _repo_root() -> Path:
    # /.../query_optimization/src/visualization/hyperparam_results.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def _as_float_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype(float)


def _running_min(values: Iterable[float]) -> np.ndarray:
    out: List[float] = []
    best = math.inf
    for v in values:
        best = min(best, float(v))
        out.append(best)
    return np.asarray(out, dtype=float)


def _normalize_0_1(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise min-max normalization to [0,1], handling constants."""
    out = df.copy()
    for c in out.columns:
        col = _as_float_series(out[c])
        mn, mx = float(np.nanmin(col)), float(np.nanmax(col))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            out[c] = 0.0
        else:
            out[c] = (col - mn) / (mx - mn)
    return out


@dataclass(frozen=True)
class History:
    df: pd.DataFrame
    param_cols: List[str]


def load_history(json_path: Path) -> History:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Expected a non-empty list in {json_path}")

    # Build a wide DataFrame: iteration, loss, <params...>
    rows: List[Dict[str, Any]] = []
    all_param_keys: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        params = entry.get("params", {}) or {}
        if isinstance(params, dict):
            all_param_keys.update(params.keys())
        rows.append(
            {
                "iteration": entry.get("iteration"),
                "loss": entry.get("loss"),
                **(params if isinstance(params, dict) else {}),
            }
        )

    df = pd.DataFrame(rows)
    if "iteration" not in df.columns or "loss" not in df.columns:
        raise ValueError(f"Missing required keys 'iteration'/'loss' in {json_path}")

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").astype("Int64")
    df["loss"] = _as_float_series(df["loss"])

    param_cols = sorted([c for c in all_param_keys if c in df.columns])
    for c in param_cols:
        df[c] = _as_float_series(df[c])

    # Sort by iteration if present; otherwise by index
    if df["iteration"].notna().any():
        df = df.sort_values("iteration", kind="stable")
    df = df.reset_index(drop=True)

    return History(df=df, param_cols=param_cols)


# ------------------------------- plots ---------------------------------------


def plot_loss_over_time(hist: History, out_dir: Path) -> None:
    df = hist.df
    x = df["iteration"].fillna(pd.Series(range(len(df)))).astype(int)
    y = df["loss"]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.grid(True, alpha=0.3)
    _savefig(out_dir / "loss_over_time.png")


def plot_best_loss_over_time(hist: History, out_dir: Path) -> None:
    df = hist.df
    x = df["iteration"].fillna(pd.Series(range(len(df)))).astype(int)
    y_best = _running_min(df["loss"].to_numpy())

    plt.figure(figsize=(10, 4))
    plt.plot(x, y_best, marker="o", linewidth=1.5, color="tab:green")
    plt.xlabel("Iteration")
    plt.ylabel("Best Loss So Far")
    plt.title("Best (Running Min) Loss Over Time")
    plt.grid(True, alpha=0.3)
    _savefig(out_dir / "best_loss_over_time.png")


def plot_lr_vs_lambda_acyclic(hist: History, out_dir: Path) -> None:
    df = hist.df
    if "learning_rate" not in df.columns or "lambda_acyclic" not in df.columns:
        print("Skipping lr_vs_lambda_acyclic: missing 'learning_rate' or 'lambda_acyclic'")
        return

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        df["learning_rate"],
        df["lambda_acyclic"],
        c=df["loss"],
        cmap="viridis",
        s=55,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    plt.xlabel("learning_rate")
    plt.ylabel("lambda_acyclic")
    plt.title("learning_rate vs lambda_acyclic (colored by loss)")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label("loss")

    # If values are strictly positive, log-scale often reveals structure.
    if (df["learning_rate"] > 0).all():
        plt.xscale("log")
    if (df["lambda_acyclic"] > 0).all():
        plt.yscale("log")

    _savefig(out_dir / "lr_vs_lambda_acyclic_scatter.png")


def plot_corr_heatmap(hist: History, out_dir: Path) -> None:
    df = hist.df
    cols = [*hist.param_cols, "loss"]
    corr_df = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(max(10, 0.6 * len(cols)), max(6, 0.45 * len(cols))))
    sns.heatmap(corr_df, cmap="coolwarm", center=0.0, annot=False, linewidths=0.5)
    plt.title("Correlation Heatmap (params + loss)")
    _savefig(out_dir / "correlation_heatmap.png")


def plot_param_vs_loss_grid(hist: History, out_dir: Path, cols_per_row: int = 4) -> None:
    df = hist.df
    param_cols = hist.param_cols
    if len(param_cols) == 0:
        return

    n = len(param_cols)
    rows = int(math.ceil(n / cols_per_row))
    plt.figure(figsize=(4.0 * cols_per_row, 3.2 * rows))

    for i, p in enumerate(param_cols):
        ax = plt.subplot(rows, cols_per_row, i + 1)
        sns.regplot(
            data=df,
            x=p,
            y="loss",
            ax=ax,
            scatter_kws={"s": 18, "alpha": 0.7, "edgecolor": "none"},
            line_kws={"color": "tab:red", "linewidth": 1.4, "alpha": 0.85},
            robust=True,
        )
        ax.set_title(p)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Parameter vs Loss (with robust regression)", y=1.01)
    _savefig(out_dir / "param_vs_loss_grid.png")


def _fit_rf_importance(hist: History) -> Tuple[pd.Series, RandomForestRegressor]:
    df = hist.df.dropna(subset=["loss"])
    X = df[hist.param_cols].copy()
    y = df["loss"].to_numpy()

    # Fill missing params with median (robust for sparse/misaligned histories)
    X = X.fillna(X.median(numeric_only=True))

    model = RandomForestRegressor(
        n_estimators=800,
        random_state=SEED,
        n_jobs=-1,
        max_features="sqrt",
    )
    model.fit(X.to_numpy(), y)
    imp = pd.Series(model.feature_importances_, index=hist.param_cols).sort_values(ascending=False)
    return imp, model


def plot_rf_feature_importance(hist: History, out_dir: Path) -> pd.Series:
    if len(hist.param_cols) == 0:
        return pd.Series(dtype=float)
    imp, _ = _fit_rf_importance(hist)

    plt.figure(figsize=(10, max(4, 0.35 * len(imp))))
    sns.barplot(x=imp.values, y=imp.index, orient="h", color="tab:blue")
    plt.xlabel("RandomForest feature importance")
    plt.ylabel("parameter")
    plt.title("Parameter Importance (RandomForestRegressor)")
    plt.grid(True, axis="x", alpha=0.25)
    _savefig(out_dir / "rf_feature_importance.png")
    return imp


def plot_top_bottom_boxplots(hist: History, out_dir: Path, quantile: float = 0.2, cols_per_row: int = 4) -> None:
    df = hist.df.dropna(subset=["loss"]).copy()
    if len(hist.param_cols) == 0 or len(df) < 5:
        return

    lo = df["loss"].quantile(quantile)
    hi = df["loss"].quantile(1.0 - quantile)
    df["bucket"] = np.where(df["loss"] <= lo, f"best_{int(quantile*100)}%", np.where(df["loss"] >= hi, f"worst_{int(quantile*100)}%", "middle"))
    df = df[df["bucket"] != "middle"]

    n = len(hist.param_cols)
    rows = int(math.ceil(n / cols_per_row))
    plt.figure(figsize=(4.0 * cols_per_row, 3.1 * rows))
    for i, p in enumerate(hist.param_cols):
        ax = plt.subplot(rows, cols_per_row, i + 1)
        sns.boxplot(data=df, x="bucket", y=p, ax=ax, palette="Set2")
        ax.set_title(p)
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.2)
    plt.suptitle(f"Best vs Worst {int(quantile*100)}% Parameter Distributions", y=1.01)
    _savefig(out_dir / "best_vs_worst_param_boxplots.png")


def plot_parallel_coordinates(hist: History, out_dir: Path, n_bins: int = 10) -> None:
    if len(hist.param_cols) == 0:
        return

    df = hist.df.dropna(subset=["loss"]).copy()
    if len(df) < 3:
        return

    params = df[hist.param_cols].copy()
    params = params.fillna(params.median(numeric_only=True))
    params_norm = _normalize_0_1(params)

    # Bin loss into quantiles for discrete coloring
    try:
        loss_bin = pd.qcut(df["loss"], q=min(n_bins, len(df)), duplicates="drop")
    except Exception:
        loss_bin = pd.cut(df["loss"], bins=min(n_bins, len(df)))
    plot_df = params_norm.copy()
    plot_df["loss_bin"] = loss_bin.astype(str)

    # Sample if too many lines (keeps plot legible)
    max_lines = 400
    if len(plot_df) > max_lines:
        plot_df = plot_df.sample(max_lines, random_state=SEED)

    plt.figure(figsize=(max(12, 0.75 * len(hist.param_cols)), 7))
    # Manual parallel coordinates (more controllable than pandas plotting)
    cols = hist.param_cols
    x = np.arange(len(cols))
    categories = plot_df["loss_bin"].unique().tolist()
    palette = sns.color_palette("viridis", n_colors=len(categories))
    color_map = dict(zip(categories, palette))

    for _, row in plot_df.iterrows():
        y = row[cols].to_numpy(dtype=float)
        plt.plot(x, y, color=color_map[row["loss_bin"]], alpha=0.15, linewidth=1.0)

    plt.xticks(x, cols, rotation=35, ha="right")
    plt.ylabel("normalized value (0..1)")
    plt.title("Parallel Coordinates (params normalized, colored by loss quantile bin)")
    plt.grid(True, axis="y", alpha=0.25)

    # Legend (compact)
    handles = [plt.Line2D([0], [0], color=color_map[c], lw=3) for c in categories[:12]]
    labels = categories[:12]
    if len(categories) > 12:
        labels[-1] = labels[-1] + " (…)"
    plt.legend(handles, labels, title="loss_bin", loc="upper right", frameon=True)

    _savefig(out_dir / "parallel_coordinates.png")


def plot_pairplot_top4(hist: History, out_dir: Path, importance: pd.Series | None) -> None:
    if importance is None or importance.empty:
        return
    top = importance.index[:4].tolist()
    df = hist.df.dropna(subset=["loss"]).copy()
    if len(df) < 5:
        return

    # Discretize loss for hue (pairplot wants categorical hue for nicer legends)
    try:
        df["loss_quartile"] = pd.qcut(df["loss"], q=min(4, len(df)), duplicates="drop").astype(str)
    except Exception:
        df["loss_quartile"] = "all"

    sns.pairplot(
        df[top + ["loss_quartile"]],
        hue="loss_quartile",
        diag_kind="kde",
        plot_kws={"alpha": 0.75, "s": 28, "edgecolor": "none"},
    )
    plt.suptitle("Pairplot of Top-4 Important Params (RF) + loss quartile hue", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "pairplot_top4_params.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ------------------------------- main ----------------------------------------


def main() -> None:
    sns.set_theme(style="whitegrid")

    root = _repo_root()
    results_path = root / RESULTS_FILE
    out_dir = root / OUTPUT_DIR
    _ensure_dir(out_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    hist = load_history(results_path)

    df = hist.df
    best_idx = int(df["loss"].idxmin()) if df["loss"].notna().any() else -1
    best_row = df.iloc[best_idx] if best_idx >= 0 else None

    print(f"Loaded {len(df)} trials from {results_path}")
    print(f"Parameters ({len(hist.param_cols)}): {', '.join(hist.param_cols)}")
    if best_row is not None:
        print(f"Best loss: {float(best_row['loss']):.6f} @ iteration={best_row.get('iteration')}")
    print(f"Writing plots to: {out_dir}")

    # Required plots
    plot_loss_over_time(hist, out_dir)
    plot_best_loss_over_time(hist, out_dir)
    plot_lr_vs_lambda_acyclic(hist, out_dir)

    # Importance / sensitivity
    plot_corr_heatmap(hist, out_dir)
    plot_param_vs_loss_grid(hist, out_dir)
    rf_imp = plot_rf_feature_importance(hist, out_dir)
    plot_top_bottom_boxplots(hist, out_dir)
    plot_parallel_coordinates(hist, out_dir)
    plot_pairplot_top4(hist, out_dir, rf_imp)

    print("Done.")


if __name__ == "__main__":
    # Make plots deterministic-ish where relevant
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    main()

