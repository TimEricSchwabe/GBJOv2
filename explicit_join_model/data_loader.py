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

def load_dataset_metadata(dataset_dir):
    """Load metadata about the dataset"""
    with open(os.path.join(dataset_dir, 'metadata.pkl'), 'rb') as f:
        return pickle.load(f) 