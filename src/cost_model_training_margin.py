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
from typing import Dict, List, Tuple
import sys
import csv

# Add src to path to import SPARQLQuery if needed by pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model import CostGNN, CostGNNv2, CostGNNv3
from data_loader_new import SPARQLQueryDataset, AddRandomGaussianFingerprints

# Need to ensure SPARQLQuery is available for unpickling
from create_data.create_cost_model_training_data import SPARQLQuery 

import torch_optimizer as optim_extra

def calculate_qerror(pred, true):
    """Calculate Q-Error between predicted and true values"""
    epsilon = 1e-10
    true_div_pred = true / (pred + epsilon)
    pred_div_true = pred / (true + epsilon)
    qerror = torch.maximum(true_div_pred, pred_div_true)
    return qerror

def train_model(model, optimizer, criterion, train_loader, val_loader=None, 
                num_epochs=100, device='cpu', save_path="best_model.pt", 
                loss_type="mse", result_dir=None):
    
    model.train()
    best_performance = float('inf')
    
    # Initialize metric lists
    history = {
        'epochs': [],
        'val_loss_reg': [],
        'val_loss_rank': [],
        'val_qerror_mean': [],
        'val_pairwise_acc': []
    }
    
    # Setup metrics directory and CSV
    metrics_dir = result_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for scatter plots
    scatter_dir = metrics_dir / 'scatter_plots'
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = metrics_dir / 'metrics.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'reg_loss', 'rank_loss', 'qerror_mean', 'pairwise_acc'])
    
    for epoch in range(num_epochs):
        total_loss = 0
        prev_time = time.time()
        model.train()
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for data_pair in pbar:
            # data_pair is [good_batch, bad_batch] (list of Batch objects)
            good_batch, bad_batch = data_pair
            
            good_batch = good_batch.to(device)
            bad_batch = bad_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            # Use .view(-1) to ensure 1D tensor even if batch size is 1 or scalar output
            out_good = model(good_batch.x, good_batch.edge_index, batch=good_batch.batch).view(-1)
            out_bad = model(bad_batch.x, bad_batch.edge_index, batch=bad_batch.batch).view(-1)
            
            # Regression Loss
            if loss_type != "qerror":
                reg_loss_good = criterion(out_good, torch.log(good_batch.y.view(-1)))
                reg_loss_bad = criterion(out_bad, torch.log(bad_batch.y.view(-1)))
            else:
                pred_y_good = torch.exp(out_good)
                reg_loss_good = torch.mean(calculate_qerror(pred_y_good, good_batch.y.view(-1)))
                pred_y_bad = torch.exp(out_bad)
                reg_loss_bad = torch.mean(calculate_qerror(pred_y_bad, bad_batch.y.view(-1)))
            
            reg_loss = (reg_loss_good + reg_loss_bad) / 2
            
            # Ranking Loss: Minimize max(0, -1 * (out_bad - out_good) + margin)
            # => Minimize max(0, out_good - out_bad + margin)
            # Since good < bad (cost), we want out_good < out_bad.
            # Using softplus(out_good - out_bad)
            rank_loss = torch.nn.functional.softplus(out_good - out_bad).mean()
            
            loss = reg_loss + 10 * rank_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Validation
        if val_loader:
            metrics, raw_data = validate_model(model, criterion, val_loader, device, loss_type=loss_type)
            
            # Update history
            epoch_num = epoch + 1
            history['epochs'].append(epoch_num)
            history['val_loss_reg'].append(metrics['reg_loss'])
            history['val_loss_rank'].append(metrics['rank_loss'])
            history['val_qerror_mean'].append(metrics['qerror_mean'])
            history['val_pairwise_acc'].append(metrics['pairwise_acc'])
            
            # Save to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_num, 
                    metrics['reg_loss'], 
                    metrics['rank_loss'], 
                    metrics['qerror_mean'], 
                    metrics['pairwise_acc']
                ])
            
            # Save scatter plot to the dedicated folder
            plot_pred_vs_true(raw_data, scatter_dir / f'epoch_{epoch_num}_pred_vs_true.png')
            
            # Update running plots
            update_running_plots(history, metrics_dir)
            
            # Save best model based on regression loss (or whatever metric you prefer)
            current_perf = metrics['reg_loss'] # using regression loss for model selection
            if current_perf < best_performance:
                best_performance = current_perf
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
                
            print(f'Epoch {epoch_num}, Loss: {total_loss:.4f}, Val Reg Loss: {metrics["reg_loss"]:.4f}, Val Rank Loss: {metrics["rank_loss"]:.4f}, Acc: {metrics["pairwise_acc"]:.4f}')

def validate_model(model, criterion, val_loader, device='cpu', loss_type="mse"):
    """
    Validate on the validation set calculating comprehensive metrics.
    """
    model.eval()
    
    # Store all raw outputs for global metric calculation
    all_pred_good = []
    all_pred_bad = []
    all_true_good = []
    all_true_bad = []
    
    with torch.no_grad():
        for data_pair in val_loader:
            good_batch, bad_batch = data_pair
            
            good_batch = good_batch.to(device)
            bad_batch = bad_batch.to(device)
            
            # Model outputs
            out_good = model(good_batch.x, good_batch.edge_index, batch=good_batch.batch).view(-1)
            out_bad = model(bad_batch.x, bad_batch.edge_index, batch=bad_batch.batch).view(-1)
            
            # Collect data
            all_pred_good.append(out_good.cpu())
            all_pred_bad.append(out_bad.cpu())
            all_true_good.append(good_batch.y.view(-1).cpu())
            all_true_bad.append(bad_batch.y.view(-1).cpu())

    # Concatenate all batches
    pred_good = torch.cat(all_pred_good)
    pred_bad = torch.cat(all_pred_bad)
    true_good = torch.cat(all_true_good)
    true_bad = torch.cat(all_true_bad)
    
    # --- 1. Regression Loss ---
    # Calculate on all data
    pred_all = torch.cat([pred_good, pred_bad])
    true_all = torch.cat([true_good, true_bad])
    
    if loss_type != "qerror":
        reg_loss = criterion(pred_all, torch.log(true_all)).item()
    else:
        qerrors = calculate_qerror(torch.exp(pred_all), true_all)
        reg_loss = torch.mean(qerrors).item()
        
    # --- 2. Ranking Loss ---
    # Using softplus(pred_good - pred_bad)
    rank_loss = torch.nn.functional.softplus(pred_good - pred_bad).mean().item()
    
    # --- 3. Q-Error ---
    pred_cost_all = torch.exp(pred_all)
    qerrors_all = calculate_qerror(pred_cost_all, true_all)
    qerror_mean = torch.mean(qerrors_all).item()
    
    # --- 4. Pairwise Accuracy ---
    # We want to check if the model correctly identified that cost(good) < cost(bad)
    # i.e., pred_good < pred_bad
    # Only consider pairs where true costs are actually different
    mask = true_good != true_bad 
    correct_pairs = (pred_good[mask] < pred_bad[mask])
    pairwise_acc = correct_pairs.float().mean().item() if mask.sum() > 0 else 0.0
    
    metrics = {
        'reg_loss': reg_loss,
        'rank_loss': rank_loss,
        'qerror_mean': qerror_mean,
        'pairwise_acc': pairwise_acc
    }
    
    raw_data = {
        'pred_all': pred_cost_all.numpy(),
        'true_all': true_all.numpy()
    }
    
    return metrics, raw_data

def plot_pred_vs_true(raw_data, save_path):
    """Plot Predicted vs True Cost"""
    y_pred = raw_data['pred_all']
    y_true = raw_data['true_all']
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Plot x=y line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Predicted vs True Cost')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def update_running_plots(history, metrics_dir):
    """Update all running history plots"""
    epochs = history['epochs']
    
    # 1. Loss History
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_loss_reg'], label='Regression Loss')
    plt.plot(epochs, history['val_loss_rank'], label='Ranking Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Losses')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(metrics_dir / 'loss_history.png')
    plt.close()
    
    # 2. Q-Error History
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_qerror_mean'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Q-Error')
    plt.title('Validation Mean Q-Error')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(metrics_dir / 'q_error_history.png')
    plt.close()
    
    # 3. Pairwise Accuracy History
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_pairwise_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Pairwise Accuracy')
    plt.grid(True)
    plt.savefig(metrics_dir / 'pairwise_acc_history.png')
    plt.close()

def setup_result_dir(config: Dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(config["root_dir"]) / "training_results" / f"gnn_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)
    return result_dir

if __name__ == "__main__":
    config = {
        'model_type': 'CostGNNv3',
        'node_feature_dim': 307,
        'hidden_dim': 128,
        'n_layers': 6,
        'use_jk': False,
        'jk_mode': 'cat',
        'use_residual': False,
        'use_layer_norm': True,
        'dropout': 0.0,
        'learning_rate': 0.0001,
        'batch_size': 64, # Batch size of QUERIES (so 32*2 = 64 plans per batch)
        'num_epochs': 1000,
        'loss_type': 'huber',
        'root_dir': '',
        'dataset_dir': 'datasets/plans/lubm/path-greedy/new', # Directory containing queries.pkl
        'enable_training': True,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    result_dir = setup_result_dir(config)
    model_save_path = result_dir / "model.pt"

    # Load Dataset
    dataset_path = os.path.join(config['root_dir'], config['dataset_dir'])
    print(f"Loading dataset from {dataset_path}")
    dataset = SPARQLQueryDataset(root=dataset_path)
    
    print(f"Dataset loaded: {len(dataset)} queries")
    
    # Split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    model = CostGNNv3(
        node_feature_dim=config['node_feature_dim'], 
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        use_jk=config['use_jk'],
        jk_mode=config['jk_mode'],
        use_residual=config['use_residual'],
        use_layer_norm=config['use_layer_norm'],
        dropout=config['dropout']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    optimizer = optim_extra.Lookahead(optimizer, k=10, alpha=0.5)
    
    if config['loss_type'] == 'huber':
        criterion = nn.HuberLoss()
    else:
        criterion = nn.MSELoss()
        
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
            result_dir=result_dir
        )
