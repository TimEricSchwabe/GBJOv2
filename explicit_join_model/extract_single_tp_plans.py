import os
import torch
from torch_geometric.data import Data
from data_loader import save_dataset_single_file
from tqdm import tqdm

def extract_single_tp_plans(input_path: str, output_dir: str):
    """Extract plans containing only one triple pattern from a GNN dataset.
    
    Args:
        input_path: Path to the full dataset.pt file
        output_dir: Directory to save the filtered dataset
    """
    print(f"Loading dataset from {input_path}...")
    data_dict = torch.load(input_path)
    
    # Lists to store filtered data
    single_tp_triples = []
    single_tp_data = []
    
    print("Filtering for single-triple-pattern plans...")
    for triples, data in tqdm(zip(data_dict['triples'], data_dict['data']), total=len(data_dict['triples'])):
        # A plan with one triple pattern will have:
        # - exactly one triple in its triples list
        # - x tensor with shape [1, 307] (one node with 307 features)
        # - empty edge_index (no edges in a single-node graph)
        if (len(triples) == 1 and 
            data.x.shape == torch.Size([1, 307]) and 
            data.edge_index.shape[1] == 0):
            
            single_tp_triples.append(triples)
            single_tp_data.append(data)
    
    print(f"\nFound {len(single_tp_data)} single-triple-pattern plans")
    
    # Save filtered dataset
    print(f"Saving filtered dataset to {output_dir}...")
    save_dataset_single_file(single_tp_triples, single_tp_data, output_dir)

if __name__ == "__main__":
    # Paths
    input_path = "/home/tim/query_optimization/datasets/dataset_stars_8_tp_with_subplans/dataset.pt"
    output_dir = "dataset_stars_8_tp_with_subplans_single_tp"
    
    # Extract and save
    extract_single_tp_plans(input_path, output_dir)
    print("Done!") 