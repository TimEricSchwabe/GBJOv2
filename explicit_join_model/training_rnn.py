import os
import time
from typing import Dict
import json
from datetime import datetime
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import SingleFileRNNQueryDataset


class CostRNNSeq(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embed = torch.nn.Linear(in_dim, hidden_dim)
        self.rnn   = torch.nn.LSTM(in_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head  = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        if seq_feats.dim() == 2:  # (T, D) → (1, T, D)
            seq_feats = seq_feats.unsqueeze(0)

        #x = torch.relu(self.embed(seq_feats))  # (B, T, H)
        x = seq_feats
        out_seq, _ = self.rnn(x)               # (B, T, H)
        cost_pred = self.head(out_seq).squeeze(-1)  # (B, T)
        
        return cost_pred


def calculate_qerror(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-10
    true_div_pred = true / (pred + epsilon)
    pred_div_true = pred / (true + epsilon)
    qerror = torch.maximum(true_div_pred, pred_div_true)
    return qerror


def collate_fn(batch):
    """Collate function to stack variable dictionaries."""
    x = torch.stack([item["x"] for item in batch], dim=0)  # (B, T, D)
    y = torch.stack([item["y"] for item in batch], dim=0)  # (B, T)
    return {"x": x, "y": y}


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 100,
    device: str = "cpu",
    save_path: str | None = None,
    loss_type: str = "mse",
    target_position: str = "all",  # "all" | "first" | "last"
):
    best_perf = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            x = batch["x"].to(device)
            y_true = batch["y"].to(device)

            optimizer.zero_grad()
            log_pred = model(x)                          # (B, T)

            if target_position == "first":
                y_true_sel = y_true[:, 0]        # (B,)
                log_pred_sel = log_pred[:, 0]    # (B,)
            elif target_position == "last":
                y_true_sel = y_true[:, -1]
                log_pred_sel = log_pred[:, -1]
            elif target_position == "all":
                y_true_sel = y_true            # (B, T)
                log_pred_sel = log_pred        # (B, T)
            else:
                raise ValueError(f"Invalid target_position: {target_position}")

            # ------------------------------------------------------------
            # Compute loss on the selected slice
            # ------------------------------------------------------------
            if loss_type == "mse": 
                loss = criterion(log_pred_sel, torch.log(y_true_sel))
            elif loss_type == "qerror":
                pred_sel = torch.exp(log_pred_sel)
                qerr = calculate_qerror(pred_sel, y_true_sel)
                loss = torch.mean(qerr)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ---------------- validation ---------------------------------------
        if val_loader is not None:
            val_loss, val_mse, val_qerr = validate_model(
                model, criterion, val_loader, device, loss_type, target_position
            )
            if val_loss < best_perf:
                best_perf = val_loss
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved to {save_path}")
            print(
                f"Epoch {epoch+1}: trainLoss={avg_train_loss:.4f} | valLoss={val_loss:.4f} "
                f"(best {best_perf:.4f}) | valMSE={val_mse:.4f} | valQErr={val_qerr:.4f}"
            )
        else:
            print(f"Epoch {epoch+1}: trainLoss={avg_train_loss:.4f}")


@torch.no_grad()
def validate_model(
    model: nn.Module,
    criterion,
    loader: DataLoader,
    device: str = "cpu",
    loss_type: str = "mse",
    target_position: str = "all",
):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_qerr = 0.0
    n_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_true = batch["y"].to(device)
        n_samples += x.size(0)

        log_pred = model(x)

        # Select slice according to `target_position`
        if target_position == "first":
            y_true = y_true[:, 0]
            log_pred = log_pred[:, 0]
        elif target_position == "last":
            y_true = y_true[:, -1]
            log_pred = log_pred[:, -1]
        elif target_position == "all":
            y_true = y_true
            log_pred = log_pred
        else:
            raise ValueError(f"Invalid target_position: {target_position}")

        if loss_type == "mse":
            loss = criterion(log_pred, torch.log(y_true))
        elif loss_type == "qerror":
            pred = torch.exp(log_pred)
            qerr = calculate_qerror(pred, y_true)
            loss = torch.mean(qerr)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        total_loss += loss.item() * x.size(0)

        # metrics irrespective of loss type
        pred = torch.exp(log_pred)
        mse = torch.mean((pred - y_true) ** 2)
        qerr = torch.mean(calculate_qerror(pred, y_true))

        total_mse += mse.item() * x.size(0)
        total_qerr += qerr.item() * x.size(0)

    return (
        total_loss / n_samples,
        total_mse / n_samples,
        total_qerr / n_samples,
    )


def plot_prediction_vs_truth_seq(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    result_dir: Path,
):
    """Create scatter & box plots of predicted vs. true *incremental* costs.

    For every size *k* (1 … n) – where *k* corresponds to having *k* triple
    patterns already joined – we plot separate files and compute metrics.
    Plots are saved directly to result_dir/plots/.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_true = batch["y"].to(device)
            log_pred = model(x)
            y_pred = torch.exp(log_pred)

            all_preds.append(y_pred.cpu())
            all_trues.append(y_true.cpu())

    y_pred_all = torch.cat(all_preds, dim=0)  # (N, T)
    y_true_all = torch.cat(all_trues, dim=0)  # (N, T)

    N, T = y_pred_all.shape
    # Prepare flattened arrays for overall scatter
    y_pred_flat = y_pred_all.view(-1)
    y_true_flat = y_true_all.view(-1)

    # Overall metrics
    qerr_all = calculate_qerror(y_pred_flat, y_true_flat)
    mean_qerr = qerr_all.mean().item()
    median_qerr = qerr_all.median().item()
    mse_all = torch.mean((y_pred_flat - y_true_flat) ** 2).item()

    print(f"Overall metrics – Mean QErr {mean_qerr:.4f} | Median {median_qerr:.4f} | MSE {mse_all:.4f}")

    # Create plots directory
    plots_dir = result_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Overall scatter ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true_flat.numpy(), y_pred_flat.numpy(), alpha=0.1)
    ax.axline((0, 0), slope=1, color="red", alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("True Incremental Cost")
    ax.set_ylabel("Predicted Incremental Cost")
    ax.set_title(f"Overall – mean QErr={mean_qerr:.2f}, median={median_qerr:.2f}")
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_vs_truth_overall.png")
    plt.close(fig)

    # Per-size metrics & scatter ------------------------------------------
    metrics_by_size = {}
    for step in range(T):  # step 0 → size 1 …
        size = step + 1
        pred_s = y_pred_all[:, step]
        true_s = y_true_all[:, step]
        qerr_s = calculate_qerror(pred_s, true_s)
        if pred_s.numel() == 0:
            continue
        mean_q = qerr_s.mean().item()
        med_q = qerr_s.median().item()
        mse_s = torch.mean((pred_s - true_s) ** 2).item()
        metrics_by_size[size] = (mean_q, med_q, mse_s, pred_s.numel())

        print(
            f"Size {size}: n={pred_s.numel()} | meanQErr={mean_q:.4f} | medianQErr={med_q:.4f} | MSE={mse_s:.4f}"
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(true_s.numpy(), pred_s.numpy(), alpha=0.2)
        ax.axline((0, 0), slope=1, color="red", alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("True Incremental Cost")
        ax.set_ylabel("Predicted Incremental Cost")
        ax.set_title(f"Size {size} – meanQErr={mean_q:.2f}, median={med_q:.2f}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"prediction_vs_truth_size_{size}.png")
        plt.close(fig)

    # Boxplot of q-error by size ------------------------------------------
    qerr_boxes, labels = [], []
    for step in range(T):
        size = step + 1
        true_s = y_true_all[:, step]
        pred_s = y_pred_all[:, step]
        qerr_s = calculate_qerror(pred_s, true_s)
        qerr_boxes.append(qerr_s.numpy())
        labels.append(f"Size {size}\n(n={len(qerr_s)})")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(qerr_boxes, labels=labels)
    ax.set_yscale("log")
    ax.set_ylabel("Q-Error (log scale)")
    ax.set_title("Q-Error Distribution by Query Size")
    ax.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(plots_dir / "qerror_boxplot_by_size.png")
    plt.close(fig)

    # Save metrics as JSON ------------------------------------------------
    metrics = {
        "overall": {
            "mean_qerror": mean_qerr,
            "median_qerror": median_qerr,
            "mse": mse_all,
            "n_samples": N * T,
        },
        "by_size": {
            str(size): {
                "mean_qerror": metrics[0],
                "median_qerror": metrics[1],
                "mse": metrics[2],
                "n_samples": metrics[3],
            }
            for size, metrics in metrics_by_size.items()
        }
    }
    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

# -----------------------------------------------------------------------------
# ----------------------------- result saving ---------------------------------
# -----------------------------------------------------------------------------

def setup_result_dir(config: Dict) -> Path:
    """Create and return path to result directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("training_results") / f"rnn_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(result_dir / "config.json", "w") as f:
        # Convert paths to strings for JSON serialization
        config_json = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in config.items()
        }
        json.dump(config_json, f, indent=2)

    return result_dir

def save_training_results(
    model: nn.Module,
    result_dir: Path,
    config: Dict,
):
    """Save model and config to result directory."""
    # Save model weights
    torch.save(model.state_dict(), result_dir / "model.pt")
    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    # ---------------- configuration ----------------------------------------
    config: Dict = {
        "node_feature_dim": 307,
        "hidden_dim": 512,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 50,
        "loss_type": "mse",  # or "qerror"
        "target_position": "last",  # "all" | "first" | "last"
        "root_dir": "/home/tim/query_optimization/",
        "dataset_dir": "dataset_stars_8_tp_rnn",
        "enable_training": True,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------- setup result directory -------------------------------
    result_dir = setup_result_dir(config)
    model_path = result_dir / "model.pt"

    # --------------- dataset ---------------------------------------------
    dataset_path = os.path.join(config["root_dir"], config["dataset_dir"])
    dataset = SingleFileRNNQueryDataset(root=dataset_path)


    print(f"Loaded dataset with {len(dataset)} samples from {dataset_path}")

    # split train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    #train_size=5

    train_indices = list(range(train_size))
    #train_indices = [2, 3]
    val_indices = list(range(len(dataset) - val_size, len(dataset)))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    #val_dataset = train_dataset

    #val_dataset = train_dataset #TODO: remove
    print(f"Train size: {train_size} | Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------- model -----------------------------------------------
    model = CostRNNSeq(in_dim=config["node_feature_dim"], hidden_dim=config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    if config["enable_training"]:
        train_model(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            device=device,
            save_path=model_path,  # save best model in result dir
            loss_type=config["loss_type"],
            target_position=config["target_position"],
        )
        print("Training complete.")
    else:
        # Try to load from previous run
        try:
            # Look for most recent result directory
            result_dirs = "/home/tim/query_optimization/training_results/v1"
            model_path = os.path.join(result_dirs, "model.pt")
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
            
        except Exception as e:
            print(f"Error loading previous model: {e}")

    # ------ evaluation & plots -------------------------------------------
    print("Creating prediction plots …")
    plot_prediction_vs_truth_seq(model, val_dataset, device, result_dir)

    # ------ save results ------------------------------------------------
    save_training_results(model, result_dir, config)
    print(f"Training run completed. Results saved in {result_dir}") 