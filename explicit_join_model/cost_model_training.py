import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from model import CostGNN, CostGNNv2
from data_loader import QueryDataset, SingleFileQueryDataset

import scienceplots
plt.style.use('science')


def calculate_qerror(pred, true):
    """Calculate Q-Error between predicted and true values"""
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    true_div_pred = true / (pred + epsilon)
    pred_div_true = pred / (true + epsilon)
    qerror = torch.maximum(true_div_pred, pred_div_true)
    return qerror


def train_model(model, optimizer, criterion, train_loader, val_loader=None, 
                num_epochs=100, device='cpu', save_path="best_model.pt", 
                loss_type="mse", result_dir=None, dataset_dir=None):
    """
    Train the model and validate on the validation dataset
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        criterion: Loss function
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        device: Device to train on (cpu or cuda)
        save_path: Path to save the best model
        loss_type: Type of loss function to use ('mse' or 'qerror')
        result_dir: Directory to save training results and plots
    """
    model.train()
    best_performance = float('inf')
    
    # Lists to store metrics for plotting
    val_losses = []
    val_qerrors = []
    val_mses = []
    epochs = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        prev_time = time.time()
        model.train()
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for data in pbar:
            # Move data to the specified device
            data = data.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(data.x, data.edge_index, batch=data.batch)
            
            # Calculate loss based on loss_type
            if loss_type == "mse":
                loss = criterion(out, torch.log(data.y))
            elif loss_type == "qerror":
                pred_y = torch.exp(out)
                qerrors = calculate_qerror(pred_y, data.y)
                loss = torch.mean(qerrors)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Validate if validation loader is provided
        perf = 0
        mse_metric = 0
        qerror_metric = 0
        if val_loader:
            perf, mse_metric, qerror_metric = validate_model(model, criterion, val_loader, device, loss_type=loss_type)
            
            # Store metrics for plotting
            epochs.append(epoch + 1)
            val_losses.append(perf)
            val_mses.append(mse_metric)
            val_qerrors.append(qerror_metric)
            
            # Plot validation metrics
            if result_dir:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                ax1.plot(epochs, val_losses)
                ax1.set_title('Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                ax2.plot(epochs, val_mses)
                ax2.set_title('Validation MSE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MSE')
                ax2.grid(True)
                
                ax3.plot(epochs, val_qerrors)
                ax3.set_title('Validation Q-Error')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Q-Error')
                ax3.grid(True)
                
                plt.tight_layout()
                plt.savefig(result_dir / 'validation_metrics.png')
                plt.close()
                
                # Plot predictions vs truth after each epoch
                plot_prediction_vs_truth(model, val_loader.dataset, device, result_dir, dataset_dir=None, debug=False)
            
            if perf < best_performance:
                best_performance = perf
                # Save the model when there's a new best performance
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
                
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}, Time: {time.time() - prev_time:.2f}s, '
              f'Val Loss: {perf:.4f}, Best Val Loss: {best_performance:.4f}, '
              f'MSE: {mse_metric:.4f}, Q-Error: {qerror_metric:.4f}')


def validate_model(model, criterion, val_loader, device='cpu', loss_type="mse"):
    """
    Validate the model on validation data
    
    Args:
        model: The neural network model
        criterion: Loss function
        val_loader: DataLoader for validation data
        device: Device to validate on (cpu or cuda)
        loss_type: Type of loss function to use ('mse' or 'qerror')
        
    Returns:
        tuple: (average loss, average MSE, average Q-Error)
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_qerror = 0
    num_samples = 0
    
    with torch.no_grad():
        for data in val_loader:
            # Move data to the specified device
            data = data.to(device)
            num_samples += data.y.size(0)
            
            out = model(data.x, data.edge_index, batch=data.batch)
            
            # Calculate the primary loss based on loss_type
            if loss_type == "mse":
                loss = criterion(out, torch.log(data.y))
            elif loss_type == "qerror":
                pred_y = torch.exp(out)
                qerrors = calculate_qerror(pred_y, data.y)
                loss = torch.mean(qerrors)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
                
            total_loss += loss.item() * data.y.size(0)
            
            # Always calculate both metrics regardless of loss type
            pred_y = torch.exp(out)
            
            # Calculate MSE
            mse = torch.mean((pred_y - data.y) ** 2)
            total_mse += mse.item() * data.y.size(0)
            
            # Calculate q-error
            qerrors = calculate_qerror(pred_y, data.y)
            qerror = torch.mean(qerrors)
            total_qerror += qerror.item() * data.y.size(0)

    avg_loss = total_loss / num_samples
    avg_mse = total_mse / num_samples
    avg_qerror = total_qerror / num_samples
    
    return avg_loss, avg_mse, avg_qerror


def plot_prediction_vs_truth(model, val_dataset, device, result_dir=None, dataset_dir=None, debug=False):
    """Plot predicted vs true values with expanded visualizations and metrics"""
    model.eval()
    data = next(iter(DataLoader(val_dataset, batch_size=10000, shuffle=True)))
    data = data.to(device)
    
    with torch.no_grad():
        y_pred = torch.exp(model(data.x, data.edge_index, batch=data.batch))
        y_true = data.y
    
    # Calculate metrics
    qerrors = calculate_qerror(y_pred, y_true)
    mse = ((y_pred - y_true) ** 2).mean().item()
    
    mean_qerror = qerrors.mean().item()
    median_qerror = qerrors.median().item()
    
    print(f"Overall Metrics:")
    print(f"  Mean Q-Error: {mean_qerror:.4f}")
    print(f"  Median Q-Error: {median_qerror:.4f}")
    print(f"  MSE: {mse:.4f}")
    
    # Create directory for plots
    if result_dir:
        plots_dir = result_dir / 'plots'
    else:
        plots_dir = Path('prediction_plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot overall scatter plot
    #fig, ax = plt.subplots(figsize=(10, 8))
    fig, ax = plt.subplots()

    # Convert to numpy arrays for correlation calculation
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    correlation = np.corrcoef(y_true_np, y_pred_np)[0,1]

    ax.plot(y_true_np, y_pred_np, alpha=0.1, color='black',
    marker="x", linestyle="none", markersize=2)
    ax.axline((0, 0), slope=1, color="black", alpha=0.5, zorder=1)
    ax.set_xlabel("True cost")
    ax.set_ylabel("Predicted cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.text(0.05, 0.95, f'$r$={correlation:.3f}', 
    #        transform=ax.transAxes, verticalalignment='top')
    
    plt.savefig(plots_dir / 'prediction_vs_truth_overall.pdf')
    plt.close()
    
    # Try to get query sizes from dataset
    try:
        # Extract query size information
        # This assumes query size information is available in the dataset
        # We'll need to check if triples_num is accessible or needs to be loaded separately
        query_sizes = []
        
        # If we have a SingleFileQueryDataset, try to access metadata
        if hasattr(val_dataset.dataset, 'triples'):
            # Access size from triples data directly if available
            for i in range(len(data.y)):
                batch_idx = data.batch[i].item()
                dataset_idx = batch_idx  # This might need adjustment depending on batch ordering
                if dataset_idx < len(val_dataset.dataset.triples):
                    query_sizes.append(len(val_dataset.dataset.triples[dataset_idx]))
                else:
                    # Default fallback - might need better logic based on data structure
                    query_sizes.append(0)
        else:
            # Load metadata from the original data file if needed
            try:
                # This assumes the dataset metadata is stored separately and has size information
                if result_dir and dataset_dir:
                    dataset_path = result_dir.parent.parent / dataset_dir
                    dataset_file = dataset_path / 'dataset.pt'
                    if dataset_file.exists():
                        metadata = torch.load(dataset_file)
                        if 'triples' in metadata:
                            triples_data = metadata['triples']
                            for i in range(len(data.y)):
                                batch_idx = data.batch[i].item()
                                if batch_idx < len(triples_data):
                                    query_sizes.append(len(triples_data[batch_idx]))
                                else:
                                    query_sizes.append(0)
                else:
                    raise ValueError("Result directory or dataset directory not provided")
            except Exception as e:
                print(f"Error loading query size metadata: {e}")
                # Default to uniform distribution for demonstration
                query_sizes = np.ones(len(data.y), dtype=int) * 4  # Assume middle size
        
        # Convert to tensor
        query_sizes = torch.tensor(query_sizes, device=device)
        
        # Create plots for each query size
        size_metrics = {}
        for size in range(1, 14):  # Sizes 1-8
            # Get data for this size
            size_mask = (query_sizes == size)
            if size_mask.sum() > 0:
                y_pred_size = y_pred[size_mask]
                y_true_size = y_true[size_mask]
                qerrors_size = calculate_qerror(y_pred_size, y_true_size)
                
                # Calculate metrics for this size
                mean_qerror_size = qerrors_size.mean().item()
                median_qerror_size = qerrors_size.median().item()
                mse_size = ((y_pred_size - y_true_size) ** 2).mean().item()
                
                size_metrics[size] = {
                    'mean_qerror': mean_qerror_size,
                    'median_qerror': median_qerror_size,
                    'mse': mse_size,
                    'count': size_mask.sum().item()
                }
                
                if debug:
                    print(f"Size {size} Metrics (n={size_mask.sum().item()}):")
                    print(f"  Mean Q-Error: {mean_qerror_size:.4f}")
                    print(f"  Median Q-Error: {median_qerror_size:.4f}")
                    print(f"  MSE: {mse_size:.4f}")
                
                # Plot scatter for this size
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(
                    y_true_size.cpu().detach().numpy(), 
                    y_pred_size.cpu().detach().numpy(), 
                    alpha=0.2
                )
                ax.axline((0, 0), slope=1, color="red", alpha=0.5, zorder=1)
                ax.set_xlabel("True cost")
                ax.set_ylabel("Predicted cost")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"Size {size}: Mean Q-Error={mean_qerror_size:.2f}, Median Q-Error={median_qerror_size:.2f}")
                
                plt.savefig(plots_dir / f'prediction_vs_truth_size_{size}.png')
                plt.close()
        
        # Create boxplot of q-errors by size
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        boxplot_data = []
        labels = []
        
        for size in range(1, 9):
            size_mask = (query_sizes == size)
            if size_mask.sum() > 0:
                size_qerrors = qerrors[size_mask].cpu().detach().numpy()
                boxplot_data.append(size_qerrors)
                labels.append(f"Size {size}\n(n={size_mask.sum().item()})")
        
        # Create boxplot
        plt.boxplot(boxplot_data, labels=labels)
        plt.ylabel("Q-Error (log scale)")
        plt.title("Q-Error Distribution by Query Size")
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add horizontal line at q-error = 1 (perfect prediction)
        plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'qerror_boxplot_by_size.png')
        plt.close()
        
        # Save metrics as JSON if result_dir is provided
        if result_dir:
            metrics = {
                "overall": {
                    "mean_qerror": mean_qerror,
                    "median_qerror": median_qerror,
                    "mse": mse,
                    "n_samples": len(data.y),
                },
                "by_size": {
                    str(size): {
                        "mean_qerror": metrics_data['mean_qerror'],
                        "median_qerror": metrics_data['median_qerror'],
                        "mse": metrics_data['mse'],
                        "n_samples": metrics_data['count'],
                    }
                    for size, metrics_data in size_metrics.items()
                }
            }
            with open(result_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        
    except Exception as e:
        print(f"Error creating size-specific plots: {e}")
        print("Falling back to overall plot only")


def setup_result_dir(config: Dict) -> Path:
    """Create and return path to result directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(config["root_dir"]) / "training_results" / f"gnn_{timestamp}"
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
    # Hyperparameters and configuration
    config = {
        # Model parameters
        'model_type': 'CostGNNv2',  # Options: 'CostGNN', 'CostGNNv2'
        'node_feature_dim': 307,    # Input feature dimension
        'hidden_dim': 512,          # Hidden layer dimension
        
        # Training parameters
        'learning_rate': 0.0001,
        'batch_size': 128,
        'num_epochs': 1,
        'loss_type': 'mse',         # Options: 'mse', 'qerror'
        
        # Dataset parameters
        'use_single_file': True,
        'train_size': 130000,
        'val_size': 20000,
        
        # Paths
        'root_dir': '/home/tim/query_optimization/',
        'dataset_dir': '/home/tim/query_optimization/datasets/star_plan_datasets_training/LUBM_STAR',
        
        # Other settings
        'enable_training': False,    # Set to False to skip training
    }
    
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup result directory
    result_dir = setup_result_dir(config)
    model_save_path = result_dir / "model.pt"
    
    # Load dataset
    if config['use_single_file']:
        dataset_path = os.path.join(config['root_dir'], config['dataset_dir'])
        dataset = SingleFileQueryDataset(root=dataset_path)
        print(f"Using single-file dataset from {dataset_path}")
    else:
        dataset_path = os.path.join(config['root_dir'], config['dataset_dir'])
        dataset = QueryDataset(root=dataset_path)
        print(f"Using regular dataset from {dataset_path}")
    
    # Get total dataset size
    total_size = len(dataset)
    print(f"Dataset loaded: {total_size} samples")
    
    # Ensure train and validation sizes are within bounds
    train_size = min(config['train_size'], total_size - config['val_size'])
    val_size = min(config['val_size'], total_size - train_size)
    
    print(f"Using {train_size} samples for training and {val_size} samples for validation")
    
    # Create indices for training and validation subsets
    train_indices = list(range(train_size))
    val_indices = list(range(total_size - val_size, total_size))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    #val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_dataset = train_dataset
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize model based on config
    if config['model_type'] == 'CostGNN':
        model = CostGNN(node_feature_dim=config['node_feature_dim'], 
                        hidden_dim=config['hidden_dim']).to(device)
    elif config['model_type'] == 'CostGNNv2':
        model = CostGNNv2(node_feature_dim=config['node_feature_dim'], 
                         hidden_dim=config['hidden_dim']).to(device)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Train model if enabled
    if config['enable_training']:
        train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            device=device,
            save_path=str(model_save_path),
            loss_type=config['loss_type'],
            result_dir=result_dir  # Pass result_dir to train_model
        )
        print(f"Model training complete. Best model saved at: {model_save_path}")
    else:
        print("Training skipped as per configuration.")
        
        # If not training, try to load a pre-trained model
        pretrained_model_path = os.path.join(config['root_dir'], 'join_plus_tp_prediction_all_sizes.pt')
        pretrained_model_path = "/home/tim/query_optimization/explicit_join_model/models/star_model.pt"
        try:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Loaded pre-trained model from {pretrained_model_path}")
        except FileNotFoundError:
            print(f"No pre-trained model found at {pretrained_model_path}")
    
    # Plot prediction vs truth
    plot_prediction_vs_truth(model, val_dataset, device, result_dir, config['dataset_dir'], debug=False)
    
    # Save training results
    save_training_results(model, result_dir, config)
    print(f"Training run completed. Results saved in {result_dir}") 