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

import torch_geometric.transforms as T
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

# Configuration flags
USE_PE = True  # Toggle positional encoding usage
ATTN_TYPE = 'multihead'  # 'multihead' or 'performer'
PE_DIM = 16  # Dimension of positional encoding features
WALK_LENGTH = 20  # Length of random walks for positional encoding
HIDDEN_DIM = 64  # Hidden dimension size
NUM_LAYERS = 4  # Number of transformer layers
HEADS = 4  # Number of attention heads
DROPOUT = 0.2  # Dropout rate
LEARNING_RATE = 0.0005  # Learning rate for optimizer
WEIGHT_DECAY = 1e-5  # Weight decay for optimizer
TRAIN_SIZE = 80000  # Number of training samples
VAL_SIZE = 2048  # Number of validation samples
BATCH_SIZE = 32  # Batch size for training and validation
TRAIN = False
DIRECTED = False  # If True, use directed graph. If False, convert to undirected

# Cost scaling configuration
COST_SCALING = 'log'  # Options: 'log', 'minmax', 'logminmax', 'none'
SCALING_MIN = 0  # For min-max scaling (will be computed from data if None)
SCALING_MAX = 1  # For min-max scaling (will be computed from data if None)


class QueryGraphTransformer(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, pe_dim=16, num_layers=4, attn_type='multihead', heads=4, dropout=0.2, use_pe=True, walk_length=12):
        super().__init__()
        
        # Configuration
        self.use_pe = use_pe
        self.node_feature_dim = node_feature_dim
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        
        # Initial node feature projection
        if use_pe:
            # Projection for node features that will be concatenated with PE
            self.node_emb = Sequential(
                Linear(node_feature_dim, hidden_dim - pe_dim),
                ReLU(),
                BatchNorm1d(hidden_dim - pe_dim),
                Dropout(dropout)
            )
            
            # Positional encoding projection - use walk_length as input dimension
            self.pe_linear = Linear(walk_length, pe_dim)  # Convert random walk PE to desired dimension
            self.pe_norm = BatchNorm1d(pe_dim)
        else:
            # Simple projection for node features without PE
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
    
    def forward(self, data, edge_weight=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if self.use_pe and hasattr(data, 'pe'):
            # Process positional encodings
            pe = data.pe
            # Project positional encoding to desired dimension
            pe_projected = self.pe_norm(self.pe_linear(pe))
            
            # Process original node features
            x_projected = self.node_emb(x)
            
            # Combine original features with positional encoding
            x = torch.cat([x_projected, pe_projected], dim=1)
        else:
            # Use regular node features without PE
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
            x = global_add_pool(x, batch)
        
        # Final prediction layer
        cost = torch.abs(self.mlp(x))

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


def add_positional_encoding(data_batch, walk_length=12, directed=True):
    """
    Add random walk positional encodings to a batch of data.
    This is applied on-the-fly during training/inference.
    
    Since random walk PE works better on undirected graphs, we:
    1. Create an undirected copy of each graph
    2. Compute PE on the undirected version
    3. Transfer the PE back to the original directed graph OR return the undirected version
       based on the 'directed' parameter
    
    Args:
        data_batch: A batch of graph data
        walk_length: Length of random walks for the positional encoding
        directed: If True, return directed graph with PE. If False, return undirected graph with PE.
        
    Returns:
        The data batch with added positional encodings (directed or undirected based on parameter)
    """
    transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
    
    # Process each graph in the batch or single graph
    
    # Clone the data to avoid modifying the original
    data_batch_with_pe = data_batch.clone()
    
    # Step 1: Create undirected copy by adding reverse edges
    edge_index = data_batch.edge_index.clone()
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    undirected_edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    # Remove duplicates
    undirected_edge_index = torch.unique(undirected_edge_index, dim=1)
    
    # Create undirected version of the data
    undirected_data = data_batch.clone()
    undirected_data.edge_index = undirected_edge_index
    
    # Step 2: Apply transform to the undirected version
    undirected_data_with_pe = transform(undirected_data)
    
    # Step 3: Either return the undirected version or transfer PE to directed version
    if directed:
        # Transfer PE from undirected to original directed graph
        data_batch_with_pe.pe = undirected_data_with_pe.pe
        return data_batch_with_pe
    else:
        # Return the undirected version with PE
        return undirected_data_with_pe


class CostScaler:
    """
    Utility class for scaling cost values during training and inverting the scaling for evaluation.
    Supports log scaling, min-max scaling, and a combination of both.
    """
    def __init__(self, method='log', min_val=None, max_val=None):
        """
        Initialize the scaler.
        
        Args:
            method: Scaling method - 'log', 'minmax', 'logminmax', or 'none'
            min_val: Minimum value for min-max scaling (computed from data if None)
            max_val: Maximum value for min-max scaling (computed from data if None)
        """
        self.method = method
        self.min_val = min_val
        self.max_val = max_val
        self.log_min = None  # For logminmax: min of log values
        self.log_max = None  # For logminmax: max of log values
        self.fitted = False
        
    def fit(self, values):
        """Compute scaling parameters from data if needed"""
        if self.method == 'minmax' and (self.min_val is None or self.max_val is None):
            self.min_val = float(torch.min(values).item())
            self.max_val = float(torch.max(values).item())
            
        elif self.method == 'logminmax':
            # For logminmax, we compute the min/max of the log-transformed values
            log_values = torch.log(values + 1e-10)
            self.log_min = float(torch.min(log_values).item())
            self.log_max = float(torch.max(log_values).item())
            print(f"Log value range: [{self.log_min:.4f}, {self.log_max:.4f}]")
            
        self.fitted = True
        return self
        
    def scale(self, values):
        """Scale the values according to the chosen method"""
        if not self.fitted and (self.method == 'minmax' or self.method == 'logminmax'):
            raise ValueError(f"Scaler must be fitted before use with {self.method} scaling")
            
        if self.method == 'log':
            # Log scaling (add small epsilon to avoid log(0))
            return torch.log(values + 1e-10)
            
        elif self.method == 'minmax':
            # Min-max scaling to [0,1] range
            return (values - self.min_val) / (self.max_val - self.min_val + 1e-10)
            
        elif self.method == 'logminmax':
            # Combined log + min-max scaling
            # First apply log transform
            log_values = torch.log(values + 1e-10)
            # Then apply min-max scaling to the log values
            return (log_values - self.log_min) / (self.log_max - self.log_min + 1e-10)
            
        else:  # 'none'
            return values
            
    def invert(self, scaled_values):
        """Invert the scaling to get back original values"""
        if not self.fitted and (self.method == 'minmax' or self.method == 'logminmax'):
            raise ValueError(f"Scaler must be fitted before use with {self.method} scaling")
            
        if self.method == 'log':
            # Invert log scaling
            return torch.exp(scaled_values)
            
        elif self.method == 'minmax':
            # Invert min-max scaling
            return scaled_values * (self.max_val - self.min_val + 1e-10) + self.min_val
            
        elif self.method == 'logminmax':
            # Invert combined log + min-max scaling
            # First invert min-max to get back to log space
            log_values = scaled_values * (self.log_max - self.log_min + 1e-10) + self.log_min
            # Then invert log transform
            return torch.exp(log_values)
            
        else:  # 'none'
            return scaled_values


def train_model(model, optimizer, criterion, train_loader, val_loader=None, num_epochs=100, 
               device='cpu', walk_length=12, use_pe=True, directed=True, 
               cost_scaling='log', scaler=None, save_path="best_transformer_model.pt"):
    model.train()
    best_performance = float('inf')
    
    # Initialize cost scaler if not provided
    if scaler is None:
        scaler = CostScaler(method=cost_scaling)
        # Fit the scaler on the training data if using min-max scaling
        if cost_scaling in ['minmax', 'logminmax']:
            print(f"Computing scaling parameters from training data for {cost_scaling} scaling...")
            all_costs = []
            # Only process 10 batches for scaling parameter computation
            for i, data in enumerate(tqdm(train_loader, desc="Computing scaling parameters")):
                all_costs.append(data.y)
                if i >= 9:  # Stop after 10 batches (0-9)
                    break
            all_costs = torch.cat(all_costs)
            scaler.fit(all_costs)
            if cost_scaling == 'minmax':
                print(f"Cost scaling range: [{scaler.min_val:.2e}, {scaler.max_val:.2e}]")
    
    for epoch in range(num_epochs):
        total_loss = 0
        prev_time = time.time()
        model.train()
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for data in pbar:
            # Add positional encodings on-the-fly if enabled
            # This will also handle the directed/undirected conversion if needed
            if use_pe:
                data = add_positional_encoding(data, walk_length=walk_length, directed=directed)
            elif not directed:
                # If not using PE but want undirected, create undirected version
                data = data.clone()
                edge_index = data.edge_index.clone()
                edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
                data.edge_index = torch.unique(torch.cat([edge_index, edge_index_rev], dim=1), dim=1)
            
            # Move data to the specified device
            data = data.to(device)
            
            # Scale target values
            scaled_y = scaler.scale(data.y)
            
            optimizer.zero_grad(set_to_none=True)
            # Redraw projections if using performer attention
            model.redraw_projection.redraw_projections()
            
            # Model predicts the scaled value
            out = model(data)
            
            # Loss computed on scaled values
            loss = criterion(out, scaled_y)
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
            perf, mse_metric, qerror_metric = validate_model(
                model, criterion, val_loader, device, walk_length, 
                use_pe, directed, cost_scaling, scaler
            )
            if perf < best_performance:
                best_performance = perf
                # Save the model when there's a new best performance
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'cost_scaling': cost_scaling
                }, save_path)
                print(f"New best model saved to {save_path}")
                
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}, Time: {time.time() - prev_time:.2f}s, '
              f'Val Loss: {perf:.4f}, Best Val Loss: {best_performance:.4f}, '
              f'MSE: {mse_metric:.4f}, Q-Error: {qerror_metric:.4f}')
    
    return scaler


def validate_model(model, criterion, val_loader, device='cpu', walk_length=12, 
                  use_pe=True, directed=True, cost_scaling='log', scaler=None):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_qerror = 0
    num_samples = 0
    
    # Initialize cost scaler if not provided
    if scaler is None:
        scaler = CostScaler(method=cost_scaling)
    
    with torch.no_grad():
        for data in val_loader:
            # Add positional encodings on-the-fly if enabled
            # This will also handle the directed/undirected conversion if needed
            if use_pe:
                data = add_positional_encoding(data, walk_length=walk_length, directed=directed)
            elif not directed:
                # If not using PE but want undirected, create undirected version
                data = data.clone()
                edge_index = data.edge_index.clone()
                edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
                data.edge_index = torch.unique(torch.cat([edge_index, edge_index_rev], dim=1), dim=1)
            
            # Move data to the specified device
            data = data.to(device)
            num_samples += data.y.size(0)
            
            # Scale target values
            scaled_y = scaler.scale(data.y)
            
            # Get model prediction (in scaled space)
            out = model(data)
            
            # Loss computed on scaled values
            loss = criterion(out, scaled_y)
            total_loss += loss.item() * data.y.size(0)
            
            # Convert prediction back to original scale for metrics
            pred_y = scaler.invert(out)
            
            # Calculate MSE between true y and predicted
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
    dataset_dir = "dataset_stars_3_old"
    
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
    train_size = min(TRAIN_SIZE, total_size - VAL_SIZE)
    val_size = min(VAL_SIZE, total_size - train_size)

    print(f"Using {train_size} samples for training and {val_size} samples for validation")
    
    # Use the first train_size examples for training and the last val_size for validation
    train_indices = list(range(train_size))
    val_indices = list(range(total_size - val_size, total_size))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    #train_dataset = torch.utils.data.Subset(dataset, [0])
    #val_dataset = torch.utils.data.Subset(dataset, [0])
    #val_dataset = train_dataset #todo: remove
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize model and move to device
    node_feature_dim = 307  # Based on the data format
    model = QueryGraphTransformer(
        node_feature_dim=node_feature_dim, 
        hidden_dim=HIDDEN_DIM,
        pe_dim=PE_DIM,
        num_layers=NUM_LAYERS,
        attn_type=ATTN_TYPE,
        heads=HEADS,
        dropout=DROPOUT,
        use_pe=USE_PE,
        walk_length=WALK_LENGTH
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
    
    # Initialize cost scaler
    scaler = None
    
    # Train model or load pre-trained model
    if TRAIN:
        scaler = train_model(
            model, optimizer, criterion, train_loader, val_loader, 
            num_epochs=100, device=device, walk_length=WALK_LENGTH, 
            use_pe=USE_PE, directed=DIRECTED, cost_scaling=COST_SCALING
        )
    else:
        # Load model with scaler
        checkpoint = torch.load("best_transformer_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint.get('scaler', CostScaler(method=COST_SCALING))
        if 'cost_scaling' in checkpoint:
            print(f"Loaded model was trained with {checkpoint['cost_scaling']} scaling")
    
    # Calculate validation metrics
    model.eval()
    val_loss, val_mse, val_qerror = validate_model(
        model, criterion, val_loader, device, WALK_LENGTH, 
        USE_PE, DIRECTED, COST_SCALING, scaler
    )
    print(f"Validation metrics - MSE: {val_mse:.4f}, Q-Error: {val_qerror:.4f}")
    
    # Evaluate and visualize results
    data_batch = next(iter(DataLoader(val_dataset, batch_size=min(1024, len(val_dataset)), shuffle=True)))
        
    # Add positional encodings if enabled and handle directed/undirected conversion
    if USE_PE:
        data_batch = add_positional_encoding(data_batch, walk_length=WALK_LENGTH, directed=DIRECTED)
    elif not DIRECTED:
        # If not using PE but want undirected, create undirected version
        data_batch = data_batch.clone()
        edge_index = data_batch.edge_index.clone()
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        data_batch.edge_index = torch.unique(torch.cat([edge_index, edge_index_rev], dim=1), dim=1)
    
    data_batch = data_batch.to(device)
    
    # Get model predictions (scaled)
    out = model(data_batch)
    # Convert back to original scale
    y_pred = scaler.invert(out)
    x = data_batch.y

    fig, ax = plt.subplots()
    ax.scatter(x.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), alpha=0.1)
    ax.axline((0, 0), slope=1, color="red", alpha=0.5, zorder=1)
    ax.set_xlabel("True cost")
    ax.set_ylabel("Predicted cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig("transformer_results.png")
    print("Evaluation plot saved to transformer_results.png")