from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from torch.nn.utils import clip_grad_norm_
from data import random_join_order, join_order_to_adjacency_matrix
from torch_geometric.data import DataLoader
from data_loader import QueryDataset, load_dataset_metadata
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from message_passing import GINConv

import pickle


class RecursiveCostGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super(RecursiveCostGNN, self).__init__()
        
        # Define initial MLP for projection from node_feature_dim to hidden_dim
        self.initial_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial projection layer
        self.initial_conv = GINConv(self.initial_mlp)
        
        # Define MLP for recursive GINConv layer 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # A single GINConv layer for recursive message passing
        self.conv = GINConv(self.mlp)
        
        # Additional FC layers with nonlinearities
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Get number of nodes in the graph
        num_nodes = x.size(0)
        
        # Calculate number of join nodes: (n-1)/2 for n total nodes
        num_joins = (num_nodes - 1) // 2
        
        # Convert edge_weights to adjacency matrix to find root node
        if edge_weight is not None:
            # Create an adjacency matrix from edge_index and edge_weight
            A = torch.zeros((num_nodes, num_nodes), device=x.device)
            A[edge_index[0], edge_index[1]] = edge_weight
            
            # Calculate out-degree for each node
            out_degree = torch.sum(A, dim=1)
            
            # First projection from initial dimension to hidden_dim
            h = self.initial_conv(x, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = self.dropout(h)
        else:
            # If no edge weights, we can't easily identify the root node
            # So we'll create a simple adjacency matrix from edge_index
            A = torch.zeros((num_nodes, num_nodes), device=x.device)
            A[edge_index[0], edge_index[1]] = 1.0
            
            # Calculate out-degree for each node
            out_degree = torch.sum(A, dim=1)
            
            # First projection from initial dimension to hidden_dim
            h = self.initial_conv(x, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Apply GINConv recursively for 'num_joins-1' times (since we already used one pass for projection)
        for _ in range(max(0, num_joins - 1)): # Make sure we don't have negative iterations
            if edge_weight is not None:
                h = self.conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = self.conv(h, edge_index)
            
            h = F.relu(h)
            h = self.dropout(h)
        
        # Find the root node (node with out-degree = 0)
        # In case there are multiple nodes with zero out-degree, pick the one with highest index
        # which is typically the root join node in our query plans
        zero_out_degree = (out_degree < 0.01)
        if torch.any(zero_out_degree):
            # Get indices of nodes with zero out-degree
            possible_roots = torch.where(zero_out_degree)[0]
            # Choose the node with highest index (which is likely the root join node)
            root_idx = torch.max(possible_roots)
            
            # Use the embedding of the root node for prediction
            root_embedding = h[root_idx]
        else:
            # Fallback: if no clear root is found, use mean pooling as before
            if batch is not None:
                root_embedding = scatter(h, batch, dim=0, reduce='mean')
            else:
                root_embedding = torch.mean(h, dim=0)
        
        # Apply FC layers with nonlinearities
        x = self.fc1(root_embedding)
        x = F.relu(x)
        x = self.dropout(x)
        # Apply absolute value to ensure cost is always positive
        cost = torch.abs(self.fc2(x))
        
        return torch.squeeze(cost)


def train_model(model, optimizer, criterion, train_loader, val_loader=None, num_epochs=100, device='cpu', save_path="best_model.pt"):
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
    train_size = 128
    val_size = 128

    print(f"Using {train_size} samples for training and {val_size} samples for validation")
    
    # Use the first train_size examples for training and the last val_size for validation
    train_indices = list(range(train_size))
    val_indices = list(range(total_size - val_size, total_size))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize model and move to device
    node_feature_dim = 307  # Based on the data format
    hidden_dim = 64
    model = RecursiveCostGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    
    # Training setup
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    TRAIN = True
    
    # Train model
    if TRAIN:
        train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=100, device=device)
    
    # Save trained model
    torch.save(model.state_dict(), "recursive_model.pt")
    print("Model training complete and saved to recursive_model.pt") 