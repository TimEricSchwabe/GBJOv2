import pickle
import os
from data_loader import save_dataset, QueryDataset
from torch_geometric.data import DataLoader

if __name__ == "__main__":
    # Set paths
    input_file = "/home/tim/Downloads/example_full_dataset_compact.pkl"
    output_dir = "dataset"
    
    # Load the full dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "rb") as f:
        data = pickle.load(f)
        triples, torch_dataset = zip(*data)
    
    print(f"Dataset loaded. Total samples: {len(torch_dataset)}")
    
    # Save in the new format for batch loading
    save_dataset(triples, torch_dataset, output_dir, clear_existing=True)
    
    # Test loading the dataset
    print("\nTesting dataset loading...")
    dataset = QueryDataset(root=output_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test batch loading
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nLoading {len(loader)} batches with batch_size={batch_size}")
    for i, batch in enumerate(loader):
        if i == 0:
            print(f"First batch shape: x={batch.x.shape}, edge_index={batch.edge_index.shape}")
            print(f"First batch y: {batch.y[:5]}")
        
        if i >= 2:  # Just test a few batches
            break
    
    print("\nDataset conversion and testing complete!") 