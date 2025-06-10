import os
import torch
import argparse
from tqdm import tqdm

def count_triple_patterns(triples_where):
    """Count the number of triple patterns in a plan"""
    return len(triples_where)

def filter_dataset(input_file, output_file):
    """
    Load dataset, remove plans of size 1, and save the filtered dataset
    
    Args:
        input_file: Path to the input dataset file
        output_file: Path to save the filtered dataset
    """
    print(f"Loading dataset from {input_file}...")
    data = torch.load(input_file)
    
    # Extract data components
    dataset_size = data['dataset_size']
    all_triples = data['triples']
    all_torch_data = data['data']
    
    print(f"Original dataset size: {dataset_size} plans")
    
    # Filter out plans of size 1
    filtered_triples = []
    filtered_torch_data = []
    
    print("Filtering out plans of size 1...")
    for i, (triples, torch_data) in enumerate(tqdm(zip(all_triples, all_torch_data), total=dataset_size)):
        # Count the number of triple patterns
        tp_count = count_triple_patterns(triples)
        
        # Only keep plans with more than 1 triple pattern
        if tp_count > 1:
            filtered_triples.append(triples)
            filtered_torch_data.append(torch_data)
    
    # Create filtered dataset
    filtered_data = {
        'dataset_size': len(filtered_triples),
        'triples': filtered_triples,
        'data': filtered_torch_data
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save filtered dataset
    print(f"Saving filtered dataset to {output_file}...")
    torch.save(filtered_data, output_file)
    
    print(f"Filtered dataset size: {len(filtered_triples)} plans")
    print(f"Removed {dataset_size - len(filtered_triples)} plans of size 1")

if __name__ == "__main__":
    # Define input and output paths directly in the script
    input_file = "dataset_stars_8_with_subplans/dataset.pt"
    output_file = "dataset_stars_8_with_subplans_no_1tp/dataset.pt"
    
    filter_dataset(input_file, output_file)