import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from model import CostGNN, CostGNNv2, CostGNNv3
from data_loader import QueryDataset, SingleFileQueryDataset, AddRandomGaussianFingerprints


@dataclass
class LoadedRun:
    run_dir: Path
    config: Dict[str, Any]
    model_path: Path


def _load_run(run_dir: Path, model_path: Optional[Path]) -> LoadedRun:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config.json at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if model_path is None:
        model_path = run_dir / "model.pt"
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    return LoadedRun(run_dir=run_dir, config=config, model_path=model_path)


def _build_model_from_config(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_type = config.get("model_type", "CostGNNv3")
    node_feature_dim = int(config.get("node_feature_dim", 307))
    hidden_dim = int(config.get("hidden_dim", 128))

    if model_type == "CostGNN":
        model = CostGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim)
    elif model_type == "CostGNNv2":
        model = CostGNNv2(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim)
    elif model_type == "CostGNNv3":
        model = CostGNNv3(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            n_layers=int(config.get("n_layers", 6)),
            use_jk=bool(config.get("use_jk", False)),
            jk_mode=str(config.get("jk_mode", "cat")),
            use_residual=bool(config.get("use_residual", False)),
            use_layer_norm=bool(config.get("use_layer_norm", False)),
            dropout=float(config.get("dropout", 0.0)),
            aggr=str(config.get("aggr", "add")),
        )
    else:
        raise ValueError(f"Unknown model_type in config: {model_type}")

    return model.to(device)


def _build_dataset_from_config(config: Dict[str, Any], seed: int):
    root_dir = str(config.get("root_dir", ""))
    dataset_dir = str(config.get("dataset_dir", ""))
    use_single_file = bool(config.get("use_single_file", True))

    dataset_path = os.path.join(root_dir, dataset_dir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Match cost_model_training.py behavior (random Gaussian fingerprints).
    # NOTE: This is stochastic; we set seeds for reproducibility per run.
    torch.manual_seed(seed)
    np.random.seed(seed)
    fingerprint_transform = AddRandomGaussianFingerprints(fingerprint_dim=64)

    if use_single_file:
        dataset = SingleFileQueryDataset(root=dataset_path, transform=fingerprint_transform)
        
        # Match cost_model_training.py: filter out plans with infinite cost
        initial_len = len(dataset)
        indices_to_keep = [i for i, data in enumerate(dataset.data_list) 
                          if not torch.isinf(data.y).any() and data.y > 0]
        if len(indices_to_keep) < initial_len:
            dataset.data_list = [dataset.data_list[i] for i in indices_to_keep]
            if hasattr(dataset, 'data_dict') and 'triples' in dataset.data_dict:
                dataset.data_dict['triples'] = [dataset.data_dict['triples'][i] for i in indices_to_keep]
            print(f"Filtered out {initial_len - len(dataset)} plans with infinite cost. Remaining: {len(dataset)}")
    else:
        dataset = QueryDataset(root=dataset_path)

    return dataset


def _get_umap():
    try:
        import umap  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "UMAP dependency missing. Install `umap-learn` (e.g. `pip install umap-learn`)."
        ) from e
    return umap


@torch.no_grad()
def extract_embeddings_and_costs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      embeddings: (N, D) pooled graph embedding right before fc1
      y_true: (N,)
      y_pred: (N,) predicted cost in original cost space (exp(log_cost))
    """
    model.eval()

    captured: List[torch.Tensor] = []

    def _hook(_module, inputs, _output):
        # inputs is a tuple; for Linear, inputs[0] is (B, D) pooled embedding.
        x_in = inputs[0]
        captured.append(x_in.detach().cpu())

    # We hook fc1 to capture the pooled embedding vector (input to fc1).
    if not hasattr(model, "fc1"):
        raise AttributeError("Model has no attribute `fc1`; cannot hook pooled embedding.")

    handle = model.fc2.register_forward_hook(_hook)

    y_true_all: List[torch.Tensor] = []
    y_pred_all: List[torch.Tensor] = []

    N = 30000  # Maximum number of queries to process
    total_processed = 0

    try:
        pbar = tqdm(loader, desc="Extracting embeddings")
        for data in pbar:
            if total_processed >= N:
                break
                
            # Move data to device - matching cost_model_training.py
            data = data.to(device)
            
            batch_size = data.num_graphs if hasattr(data, "num_graphs") else 1
            
            # Forward pass - matching cost_model_training.py (no edge_weight)
            out = model(data.x, data.edge_index, batch=data.batch)

            # Training uses log(y) as target -> model outputs log(cost)
            y_pred = out.detach().cpu()
            y_true = data.y.detach().cpu()

            # Ensure shapes are (B,)
            y_pred_all.append(y_pred.view(-1))
            y_true_all.append(y_true.view(-1))

            total_processed += batch_size
            pbar.set_postfix({"processed": total_processed})

    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("No embeddings captured. Is the model forward using `fc1`?")

    embeddings = torch.cat(captured, dim=0).numpy()
    y_true_np = torch.cat(y_true_all, dim=0).numpy()
    y_pred_np = torch.cat(y_pred_all, dim=0).numpy()

    if embeddings.shape[0] != y_true_np.shape[0]:
        raise RuntimeError(
            f"Embedding count mismatch: embeddings={embeddings.shape[0]} vs y={y_true_np.shape[0]}"
        )

    return embeddings, y_true_np, y_pred_np


def run_umap(embeddings: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    umap = _get_umap()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _safe_log10(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(x, eps))


def plot_umap(
    coords: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    point_size: float = 6.0,
    alpha: float = 0.65,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    c1 = _safe_log10(y_true)
    c2 = _safe_log10(y_pred)

    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=c1, s=point_size, alpha=alpha, cmap="viridis")
    ax1.set_title("UMAP colored by log10(true_cost)")
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

    sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=c2, s=point_size, alpha=alpha, cmap="viridis")
    ax2.set_title("UMAP colored by log10(pred_cost)")
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

    fig_path_png = out_dir / "umap_true_vs_pred.png"
    fig_path_pdf = out_dir / "umap_true_vs_pred.pdf"
    plt.savefig(fig_path_png, dpi=200)
    plt.savefig(fig_path_pdf)
    plt.close(fig)


def save_outputs(
    out_dir: Path,
    embeddings: np.ndarray,
    coords: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run: LoadedRun,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "embeddings_and_costs.npz",
        embeddings=embeddings,
        umap=coords,
        true_cost=y_true,
        pred_cost=y_pred,
    )

    with open(out_dir / "source_run.json", "w") as f:
        json.dump(
            {
                "run_dir": str(run.run_dir),
                "model_path": str(run.model_path),
                "config": run.config,
            },
            f,
            indent=2,
        )


def main() -> None:
    # --- CONFIGURATION ---
    # Set your paths and parameters here
    config = {
        "run_dir": "/home/tim/query_optimization/training_results/wikidata-path-log1p",
        "model_path": "/home/tim/query_optimization/training_results/wikidata-path-log1p/model.pt",  # Defaults to <run_dir>/model.pt
        "out_dir": None,    # Defaults to <run_dir>/embedding_analysis
        "device": "cpu",   # "cuda" or "cpu"
        "batch_size": 256,
        "seed": 0,
        "n_neighbors": 30,
        "min_dist": 0.1,
    }
    # ---------------------

    run_dir = Path(config["run_dir"])
    model_path = Path(config["model_path"]) if config["model_path"] else None
    
    run = _load_run(run_dir, model_path)

    device_str = config["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = _build_dataset_from_config(run.config, seed=seed)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    model = _build_model_from_config(run.config, device=device)
    state = torch.load(run.model_path, map_location=device)
    model.load_state_dict(state)

    embeddings, y_true, y_pred = extract_embeddings_and_costs(model, loader, device=device)
    coords = run_umap(
        embeddings,
        seed=seed,
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
    )

    out_dir = Path(config["out_dir"]).expanduser().resolve() if config["out_dir"] else (run.run_dir / "embedding_analysis")
    save_outputs(out_dir, embeddings, coords, y_true, y_pred, run)
    plot_umap(coords, y_true, y_pred, out_dir)

    print(f"Saved: {out_dir / 'embeddings_and_costs.npz'}")
    print(f"Saved: {out_dir / 'umap_true_vs_pred.png'}")


if __name__ == "__main__":
    main()


