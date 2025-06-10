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
            self.data_dict = torch.load(self.dataset_path)
            self.data_list = self.data_dict['data']
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

class SingleFileRNNQueryDataset(Dataset):
    """Dataset loader for the RNN training data stored in a single ``dataset.pt`` file.
    Each datapoint is expected to be a dictionary (or any PyTorch container) with at
    least the following keys:

    ``x`` : ``torch.FloatTensor`` with shape ``(seq_len, feature_dim)`` –
        the sequential triple-pattern embeddings in the *join order* that was
        executed/generated for this query.
    ``y`` : ``torch.FloatTensor`` with shape ``(seq_len,)`` –
        the *incremental* cost after every time-step of that join order.  Position
        0 must contain the standalone cardinality of the first triple-pattern;
        positions ``1..seq_len-1`` contain the join cardinality introduced by the
        corresponding join step **without** re-adding triple costs (see the
        dataset generation script).

    Additional keys are ignored by the loader and passed through unchanged so
    that future extensions of the datapoint structure do not require changes to
    this class.
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

        self.dataset_path = os.path.join(root, "dataset.pt")
        super().__init__(root, transform, pre_transform, pre_filter)

        if os.path.exists(self.dataset_path):

            self.data_dict = torch.load(self.dataset_path)
            self.data_list = self.data_dict["data"]
            self.triples = self.data_dict["triples"]
        else:
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # Presence of the single file is sufficient.
        return ["dataset.pt"] if os.path.exists(self.dataset_path) else []

    def process(self):
        # No on-the-fly processing – dataset generation happens offline.
        pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        data["triples"] = self.triples[idx]  # Add triples to the data dictionary
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
        data_dict = torch.load(dataset_path)
        return {
            'dataset_size': data_dict['dataset_size'],
            'triples': data_dict['triples']
        }
    return None 