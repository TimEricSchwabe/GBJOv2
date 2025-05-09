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


class CostGNN(nn.Module):
	def __init__(self, node_feature_dim, hidden_dim):
		super(CostGNN, self).__init__()
		
		# Define MLPs for GINConv layers
		self.mlp1 = nn.Sequential(
			nn.Linear(node_feature_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		self.mlp2 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		self.mlp3 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		# GINConv layers for more powerful message passing
		self.conv1 = GINConv(self.mlp1)
		self.conv2 = GINConv(self.mlp2)
		self.conv3 = GINConv(self.mlp3)
		
		# Additional FC layers with nonlinearities
		self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
		self.fc2 = nn.Linear(hidden_dim // 2, 1)
		
		# Dropout for regularization
		self.dropout = nn.Dropout(0.2)

	def forward(self, x, edge_index, edge_weight=None, batch=None):
		# For GINConv, edge_weight needs special handling
		if edge_weight is not None:
			x = self.conv1(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv1(x, edge_index)

		x = F.relu(x)
		x = self.dropout(x)
		
		if edge_weight is not None:
			x = self.conv2(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv2(x, edge_index)
		
		x = F.relu(x)
		x = self.dropout(x)
		
		if edge_weight is not None:
			x = self.conv3(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv3(x, edge_index)
		
		x = F.relu(x)

		# Global pooling
		if batch is not None:
			x = scatter(x, batch, dim=0, reduce='mean')
		else:
			x = torch.mean(x, dim=0)
		
		# Apply FC layers with nonlinearities
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout(x)
		cost = self.fc2(x)
		
		return torch.squeeze(cost)
      

class CostGNN3(nn.Module):
	def __init__(self, node_feature_dim, hidden_dim):
		super(CostGNN, self).__init__()
		self.conv1 = GCNConv(node_feature_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim)
		self.conv3 = GCNConv(hidden_dim, hidden_dim)
		self.fc0 = nn.Linear(hidden_dim, hidden_dim)  # Output a single cost value
		self.fc = nn.Linear(hidden_dim, 1)  # Output a single cost value
		
	def forward(self, x, edge_index, edge_weight=None, batch=None):
		if edge_weight is not None:
			x = self.conv1(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv1(x, edge_index)

		x = F.relu(x)
		if edge_weight is not None:
			x = self.conv2(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv2(x, edge_index)
		
		x = F.relu(x)
		
		if edge_weight is not None:
			x = self.conv3(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv3(x, edge_index)
		
		x = F.relu(x)

		if batch is not None:
			x = scatter(x, batch, dim=0, reduce='mean')
		else:
			x = torch.mean(x, dim=0)
		
		cost = self.fc0(x)
		x = F.relu(x)

		cost = self.fc(x)		
		return torch.squeeze(cost)


class CostGNNbla(nn.Module):
	def __init__(self, node_feature_dim, hidden_dim):
		super(CostGNN, self).__init__()
		layers = [node_feature_dim, hidden_dim, hidden_dim]
		self.convs = nn.ModuleList([
			GINConv(
				torch.nn.Linear(layers[i], layers[i + 1]),
				eps=1.
			)
			for i in range(len(layers) - 1)
		])
		self.fc0 = nn.Linear(hidden_dim, hidden_dim)  # Output a single cost value
		self.fc = nn.Linear(hidden_dim, 1, bias=False)  # Output a single cost value

	def forward(self, x, edge_index, edge_weight=None, batch=None):
		for conv in self.convs:
			if edge_weight is not None:
				x = conv(x, edge_index, edge_weight=edge_weight)
			else:
				x = conv(x, edge_index)
			# x = F.dropout(x, p=0.5, training=self.training)
			x = F.relu(x)

		if batch is not None:
			x = scatter(x, batch, dim=0, reduce='mean')
		else:
			x = torch.mean(x, dim=0)
		
		cost = self.fc0(x)
		x = F.relu(x)

		cost = self.fc(x)
		
		# cost = torch.exp(cost)
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
            #data.y = torch.tensor(1) ## REOMVE !!!!
            # Move data to the specified device
            data = data.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            out = model(data.x, data.edge_index, batch=data.batch)
            loss = criterion(out, torch.log(data.y))
            loss.backward()
            #clip_grad_norm_(model.parameters(), .4)
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
            # Clamp extreme values to avoid skewing the mean
            #qerror = torch.clamp(qerror, 0, 1000)  # Cap at 1000x error
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
    train_size = 60000
    val_size = 1024

    print(f"Using {train_size} samples for training and {val_size} samples for validation")
    
    # # Create indices for the subset we want to use
    indices = torch.randperm(total_size)[:train_size + val_size]
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Then split the subset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        subset, [train_size, val_size]
    )
    
    # Deterministically use the first element of dataset for both train and validation
    #train_dataset = torch.utils.data.Subset(dataset, [0])
    #val_dataset = torch.utils.data.Subset(dataset, [0])
    #val_dataset = train_dataset #todo: remove

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Initialize model and move to device
    node_feature_dim = 307  # Based on the data format
    hidden_dim = 64
    model = CostGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    
    # Training setup
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    TRAIN = False
    
    # Train model
    if TRAIN:
        train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs=100, device=device)
		
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "Path-GIN-40000.pt")
    model.load_state_dict(torch.load(model_path))

    model.eval()


    data = next(iter(DataLoader(train_dataset, batch_size=1024, shuffle=True)))
    y = torch.exp(model(data.x, data.edge_index, batch=data.batch))
    x = data.y

    fig, ax = plt.subplots()
    ax.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy(), alpha=0.1)
    ax.axline((0, 0), slope=1, color="red", alpha=0.5, zorder=1)
    ax.set_xlabel("True cost")
    ax.set_ylabel("Predicted cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()

    exit()

    # Save trained model
    torch.save(model.state_dict(), "trained_model.pt")
    print("Model training complete and saved to trained_model.pt")
    
    # Example of generating a simple query plan (for demonstration)
    triples = [
        ["?x", "?p1", "?z1"],
        ["?x", "?p2", "?z2"],
        ["?x", "?p3", "?z3"],
    ]

    query_plan = random_join_order(triples)
    query_plan.visualize()
    print("Query plan visualization saved as query_plan.png")

    datapoint = join_order_to_adjacency_matrix(query_plan, seed=42)
    print("Example query plan generated")
	