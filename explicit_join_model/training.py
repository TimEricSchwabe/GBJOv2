import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model import CostGNN, CostGNNv2
from data_loader import QueryDataset, SingleFileQueryDataset


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
                loss_type="mse"):
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
    """
    model.train()
    best_performance = float('inf')
    
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


def plot_prediction_vs_truth(model, val_dataset, device, root_dir=None, dataset_dir=None):
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
    
    # Create directory for plots if it doesn't exist
    plots_dir = 'prediction_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot overall scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), alpha=0.1)
    ax.axline((0, 0), slope=1, color="red", alpha=0.5, zorder=1)
    ax.set_xlabel("True cost")
    ax.set_ylabel("Predicted cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Overall: Mean Q-Error={mean_qerror:.2f}, Median Q-Error={median_qerror:.2f}")
    
    plt.savefig(os.path.join(plots_dir, 'prediction_vs_truth_overall.png'))
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
                if root_dir and dataset_dir:
                    dataset_path = os.path.join(root_dir, dataset_dir)
                    dataset_file = os.path.join(dataset_path, 'dataset.pt')
                    if os.path.exists(dataset_file):
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
                    raise ValueError("Root directory or dataset directory not provided")
            except Exception as e:
                print(f"Error loading query size metadata: {e}")
                # Default to uniform distribution for demonstration
                query_sizes = np.ones(len(data.y), dtype=int) * 4  # Assume middle size
        
        # Convert to tensor
        query_sizes = torch.tensor(query_sizes, device=device)
        
        # Create plots for each query size
        size_metrics = {}
        for size in range(1, 9):  # Sizes 1-8
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
                
                plt.savefig(os.path.join(plots_dir, f'prediction_vs_truth_size_{size}.png'))
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
        plt.savefig(os.path.join(plots_dir, 'qerror_boxplot_by_size.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error creating size-specific plots: {e}")
        print("Falling back to overall plot only")


if __name__ == "__main__":
    # Hyperparameters and configuration
    config = {
        # Model parameters
        'model_type': 'CostGNNv2',  # Options: 'CostGNN', 'CostGNNv2'
        'node_feature_dim': 307,    # Input feature dimension
        'hidden_dim': 512,          # Hidden layer dimension
        
        # Training parameters
        'learning_rate': 0.001,
        'batch_size': 1,
        'num_epochs': 2000,
        'loss_type': 'mse',         # Options: 'mse', 'qerror'
        
        # Dataset parameters
        'use_single_file': True,
        'train_size': 15000,
        'val_size': 10000,
        
        # Paths
        'root_dir': '/home/tim/query_optimization/',
        'dataset_dir': 'dataset_stars_8_with_subplans',
        'model_save_path': 'best_model_local.pt',
        
        # Other settings
        'enable_training': False,    # Set to False to skip training
    }
    
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
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
    
    # Full path for model saving
    save_path = os.path.join(config['root_dir'], config['model_save_path'])
    
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
            save_path=save_path,
            loss_type=config['loss_type']
        )
        print(f"Model training complete. Best model saved at: {save_path}")
    else:
        print("Training skipped as per configuration.")
        
        # If not training, try to load a pre-trained model
        pretrained_model_path = os.path.join(config['root_dir'], 'join_plus_tp_prediction_all_sizes.pt')
        try:
            pass
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Loaded pre-trained model from {pretrained_model_path}")
        except FileNotFoundError:
            print(f"No pre-trained model found at {pretrained_model_path}")
    
    # Plot prediction vs truth
    plot_prediction_vs_truth(model, val_dataset, device, config['root_dir'], config['dataset_dir']) 