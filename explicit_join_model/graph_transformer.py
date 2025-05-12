import argparse
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    MSELoss,
    Dropout
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GPSConv, global_add_pool, global_mean_pool
from torch_geometric.nn.attention import PerformerAttention
import torch.nn.functional as F

from data import random_join_order, join_order_to_adjacency_matrix
from data_loader import QueryDataset, load_dataset_metadata
import time
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
args = parser.parse_args()


class QueryGraphTransformer(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_layers=4, attn_type='multihead', heads=4, dropout=0.2):
        super().__init__()
        
        # Initial node feature projection
        self.node_emb = Sequential(
            Linear(node_feature_dim, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim),
            Dropout(dropout)
        )
        
        # Transformer layers with GPS architecture
        self.convs = ModuleList()
        for _ in range(num_layers):
            # MLP for the local GNN component
            nn = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            
            # GPS layer combines local message passing with global attention
            attn_kwargs = {'dropout': dropout}
            conv = GPSConv(hidden_dim, GINConv(nn), heads=heads,
                          attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)
        
        # MLP for final prediction
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, 1),
        )
        
        # For Performer attention (if used)
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Initial feature transformation
        x = self.node_emb(x)
        
        # Apply GPS transformer layers
        for conv in self.convs:
            x = conv(x, edge_index, batch=batch)
        
        # Find the root node similar to RecursiveCostGNN
        if batch is None:
            # If no batch info, assume we're working with a single graph
            num_nodes = x.size(0)
            
            # Create an adjacency matrix to find the root node
            A = torch.zeros((num_nodes, num_nodes), device=x.device)
            if edge_weight is not None:
                A[edge_index[0], edge_index[1]] = edge_weight
            else:
                A[edge_index[0], edge_index[1]] = 1.0
            
            # Calculate out-degree for each node
            out_degree = torch.sum(A, dim=1)
            
            # Find nodes with zero out-degree (potential root nodes)
            zero_out_degree = (out_degree < 0.01)
            if torch.any(zero_out_degree):
                # Get indices of nodes with zero out-degree
                possible_roots = torch.where(zero_out_degree)[0]
                # Choose the node with highest index (likely the root join node)
                root_idx = torch.max(possible_roots)
                
                # Use the embedding of the root node for prediction
                x = x[root_idx].unsqueeze(0)
            else:
                # Fallback: if no clear root is found, use mean pooling
                x = torch.mean(x, dim=0, keepdim=True)
        else:
            # With batched graphs, use global pooling
            x = global_mean_pool(x, batch)
        
        # Final prediction layer
        cost = self.mlp(x)
        return torch.squeeze(cost)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


def train_model(model, optimizer, criterion, train_loader, val_loader=None, num_epochs=100, device='cpu', save_path="best_transformer_model.pt"):
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
            # Redraw projections if using performer attention
            model.redraw_projection.redraw_projections()
            out = model(data.x, data.edge_index, batch=data.batch)
            loss = criterion(out, torch.log(data.y))
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
            perf, mse_metric, qerror_metric = validate_model(model, criterion, val_loader, device)
            if perf < best_performance:
                best_performance = perf
                # Save the model when there's a new best performance
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
                
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}, Time: {time.time() - prev_time:.2f}s, '
              f'Val Loss: {perf:.4f}, Best Val Loss: {best_performance:.4f}, '
              f'MSE: {mse_metric:.4f}, Q-Error: {qerror_metric:.4f}')


def validate_model(model, criterion, val_loader, device='cpu'):
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
            loss = criterion(out, torch.log(data.y))
            total_loss += loss.item() * data.y.size(0)
            
            # Calculate MSE between true y and predicted (exp(out))
            pred_y = torch.exp(out)
            mse = torch.mean((pred_y - data.y) ** 2)
            total_mse += mse.item() * data.y.size(0)
            
            # Calculate q-error: max(true/pred, pred/true)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            true_div_pred = data.y / (pred_y + epsilon)
            pred_div_true = pred_y / (data.y + epsilon)
            qerror = torch.maximum(true_div_pred, pred_div_true)
            qerror = torch.mean(qerror)
            total_qerror += qerror.item() * data.y.size(0)

    avg_loss = total_loss / num_samples
    avg_mse = total_mse / num_samples
    avg_qerror = total_qerror / num_samples
    
    return avg_loss, avg_mse, avg_qerror


if __name__ == "__main__":
    # Example of using the model with batch-loaded dataset
    dataset_dir = "dataset"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = QueryDataset(root=dataset_dir)
    
    # Get total dataset size
    total_size = len(dataset)

    print(f"Dataset loaded: {total_size} samples")
    
    # Set train and validation sizes for experimentation
    # Make sure they don't exceed the total dataset size
    train_size = 40000
    val_size = 1024

    print(f"Using {train_size} samples for training and {val_size} samples for validation")
    
    # Use the first train_size examples for training and the last val_size for validation
    train_indices = list(range(train_size))
    val_indices = list(range(total_size - val_size, total_size))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize model and move to device
    node_feature_dim = 307  # Based on the data format
    hidden_dim = 64
    model = QueryGraphTransformer(
        node_feature_dim=node_feature_dim, 
        hidden_dim=hidden_dim,
        num_layers=4,  # Can be tuned
        attn_type=args.attn_type,  # 'multihead' or 'performer'
        heads=4,
        dropout=0.2
    ).to(device)
    
    # Training setup
    learning_rate = 0.0005  # Reduced learning rate for transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)

    TRAIN = False
    
    # Train model
    if TRAIN:
        train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=100, device=device)
    else:
        model.load_state_dict(torch.load("best_transformer_model.pt"))
    
    # Calculate validation metrics
    model.eval()
    val_loss, val_mse, val_qerror = validate_model(model, criterion, val_loader, device)
    print(f"Validation metrics - MSE: {val_mse:.4f}, Q-Error: {val_qerror:.4f}")
    
    # Evaluate and visualize results
    data = next(iter(DataLoader(val_dataset, batch_size=min(1024, len(val_dataset)), shuffle=True)))
    data = data.to(device)
    y = torch.exp(model(data.x, data.edge_index, batch=data.batch))
    x = data.y

    fig, ax = plt.subplots()
    ax.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy(), alpha=0.1)
    ax.axline((0, 0), slope=1, color="red", alpha=0.5, zorder=1)
    ax.set_xlabel("True cost")
    ax.set_ylabel("Predicted cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig("transformer_results.png")
    print("Evaluation plot saved to transformer_results.png")