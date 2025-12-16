import os
import pickle
import torch
from torch_geometric.data import Dataset, DataLoader
import shutil

class QueryDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(QueryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        # Check if the processed directory exists and get file names
        if not os.path.exists(self.processed_dir):
            return []
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
    
    def process(self):
        # This method is called if processed_file_names is empty
        pass
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

class SingleFileQueryDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_path = os.path.join(root, 'dataset.pt')
        super(SingleFileQueryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # Load the dataset after initialization
        if os.path.exists(self.dataset_path):
            self.data_dict = torch.load(self.dataset_path, weights_only=False)
            raw_data_list = self.data_dict['data']

            # Filter out samples with zero or invalid costs
            self.data_list = [
                d for d in raw_data_list 
                if hasattr(d, 'y') and d.y is not None 
                and torch.isfinite(d.y).all() and (d.y > 0).all()
            ]
            
            n_filtered = len(raw_data_list) - len(self.data_list)
            if n_filtered > 0:
                print(f"Filtered out {n_filtered} samples with zero/invalid costs")
        else:
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        # Since we're using a single file, just check if it exists
        if os.path.exists(self.dataset_path):
            return ['dataset.pt']
        return []
    
    def process(self):
        # This method is called if processed_file_names is empty
        pass
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

def save_dataset(triples, torch_dataset, output_dir, clear_existing=False):
    """
    Save dataset to disk for batch loading
    
    Args:
        triples: List of triples data
        torch_dataset: PyTorch Geometric dataset
        output_dir: Directory to save the processed data
        clear_existing: Whether to clear existing files in output_dir
    """
    processed_dir = os.path.join(output_dir, 'processed')
    
    # Create directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    elif clear_existing:
        # Clear existing files if requested
        shutil.rmtree(processed_dir)
        os.makedirs(processed_dir)
    
    # Save metadata
    metadata = {
        'dataset_size': len(torch_dataset),
        'triples': triples
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save each data point individually
    for i, data in enumerate(torch_dataset):
        torch.save(data, os.path.join(processed_dir, f'data_{i}.pt'))
    
    print(f"Dataset saved to {output_dir}")
    print(f"Total samples: {len(torch_dataset)}")

def save_dataset_single_file(triples, torch_dataset, output_dir):
    """
    Save dataset to a single file for batch loading
    
    Args:
        triples: List of triples data
        torch_dataset: PyTorch Geometric dataset
        output_dir: Directory to save the processed data
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save metadata and dataset in one file
    data = {
        'dataset_size': len(torch_dataset),
        'triples': triples,
        'data': torch_dataset
    }
    
    torch.save(data, os.path.join(output_dir, 'dataset.pt'))
    
    print(f"Dataset saved to {os.path.join(output_dir, 'dataset.pt')}")
    print(f"Total samples: {len(torch_dataset)}")

def load_dataset_metadata(dataset_dir):
    """Load metadata about the dataset"""
    with open(os.path.join(dataset_dir, 'metadata.pkl'), 'rb') as f:
        return pickle.load(f)

def load_single_file_dataset_metadata(dataset_dir):
    """Load metadata about the single file dataset"""
    dataset_path = os.path.join(dataset_dir, 'dataset.pt')
    if os.path.exists(dataset_path):
        data_dict = torch.load(dataset_path, weights_only=False)
        return {
            'dataset_size': data_dict['dataset_size'],
            'triples': data_dict['triples']
        }
    return None 

# Add this to data_loader.py (or a new transforms.py file)

import torch
import numpy as np

class AddJoinFingerprints:
    """
    Transform that adds orthonormal fingerprint vectors to join nodes.
    Fingerprints are randomly assigned each time a graph is loaded,
    but stay fixed during a single optimization run.
    """
    def __init__(self, fingerprint_dim=14, max_joins=14):
        """
        Args:
            fingerprint_dim: Dimension of fingerprint vectors (default 14 for orthonormal)
            max_joins: Maximum number of join nodes to support (n-1 for n triples)
        """
        self.fingerprint_dim = fingerprint_dim
        self.max_joins = max_joins
        # Pre-compute orthonormal basis (identity matrix rows)
        self.fingerprint_basis = torch.eye(max_joins, fingerprint_dim)
    
    def __call__(self, data):
        """
        Modify node features to include fingerprints for join nodes.
        
        Join nodes are identified by: feature[306] == 1
        Triple nodes have: feature[306] == 0
        """
        x = data.x.clone()
        n_nodes = x.size(0)
        
        # Identify join nodes (last feature dim == 1)
        is_join = (x[:, -1] == 1.0)
        join_indices = torch.where(is_join)[0]
        n_joins = len(join_indices)
        
        if n_joins == 0:
            return data
        
        # Randomly permute fingerprint assignment for this graph
        perm = torch.randperm(self.max_joins)[:n_joins]
        fingerprints = self.fingerprint_basis[perm]  # [n_joins, fingerprint_dim]
        
        # Insert fingerprints into the first `fingerprint_dim` positions of join nodes
        # (these are currently zeros)
        for i, join_idx in enumerate(join_indices):
            x[join_idx, :self.fingerprint_dim] = fingerprints[i]
        
        data.x = x
        return data


class AddRandomGaussianFingerprints:
    """
    Alternative: Fresh random Gaussian fingerprints each time.
    Normalized to unit length for stability.
    """
    def __init__(self, fingerprint_dim=32):
        self.fingerprint_dim = fingerprint_dim
    
    def __call__(self, data):
        x = data.x.clone()
        
        is_join = (x[:, -1] == 1.0)
        join_indices = torch.where(is_join)[0]
        n_joins = len(join_indices)
        
        if n_joins == 0:
            return data
        
        # Fresh random fingerprints, normalized
        fingerprints = torch.randn(n_joins, self.fingerprint_dim)
        fingerprints = fingerprints / fingerprints.norm(dim=1, keepdim=True)
        
        for i, join_idx in enumerate(join_indices):
            x[join_idx, :self.fingerprint_dim] = fingerprints[i]
        
        data.x = x
        return data

class QueryPairDataset(Dataset):
    """
    Wraps a flat dataset of plans (where n consecutive plans belong to one query)
    and exposes it as a dataset of QUERIES.
    
    __getitem__ returns a pair (good_plan, bad_plan) sampled from the query.
    """
    def __init__(self, dataset, plans_per_query=None):
        super().__init__()
        self.dataset = dataset
        
        # Determine plans per query (n)
        if plans_per_query is not None:
            self.plans_per_query = plans_per_query
        elif hasattr(dataset, 'data_dict') and 'triples' in dataset.data_dict:
            # Infer from metadata
            num_queries = len(dataset.data_dict['triples'])
            if len(dataset) % num_queries != 0:
                raise ValueError(f"Dataset size {len(dataset)} not divisible by num_queries {num_queries}")
            self.plans_per_query = len(dataset) // num_queries
        else:
            raise ValueError("Could not infer plans_per_query. Please provide it explicitly.")
            
        self.num_queries = len(dataset) // self.plans_per_query

    def len(self):
        return self.num_queries

    def get(self, idx):
        """
        Returns a pair (good_data, bad_data) for the query at `idx`.
        """
        base_idx = idx * self.plans_per_query
        
        # Naive sampling: pick two random indices from this query's block
        # You can optimize this by pre-calculating costs if speed is an issue
        
        # Try a few times to find a pair with different costs
        for _ in range(10):
            offset1 = torch.randint(0, self.plans_per_query, (1,)).item()
            offset2 = torch.randint(0, self.plans_per_query, (1,)).item()
            
            if offset1 == offset2:
                continue
                
            idx1 = base_idx + offset1
            idx2 = base_idx + offset2
            
            data1 = self.dataset[idx1]
            data2 = self.dataset[idx2]
            
            cost1 = data1.y.item()
            cost2 = data2.y.item()
            
            if cost1 != cost2:
                if cost1 < cost2:
                    return data1, data2 # (good, bad)
                else:
                    return data2, data1 # (good, bad)
        
        # Fallback: if all plans have same cost or we failed to find diff, return any two
        # The ranking loss with margin will naturally be 0 if costs are effectively equal
        return self.dataset[base_idx], self.dataset[base_idx + 1 if self.plans_per_query > 1 else 0]
