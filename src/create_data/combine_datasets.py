#!/usr/bin/env python3
"""
Script to combine multiple Query datasets (.pt files) into a single dataset file.
"""

import torch
import os
import random


def combine_datasets(input_files, output_file):
    """
    Combine multiple PyTorch dataset files into a single file.
    
    Args:
        input_files: List of paths to input .pt dataset files
        output_file: Path to output combined dataset file
    """
    combined_triples = []
    combined_data = []
    total_size = 0
    
    print(f"Combining {len(input_files)} dataset files...")
    
    for i, input_file in enumerate(input_files):
        print(f"Loading dataset {i+1}/{len(input_files)}: {input_file}")
        
        # Load the dataset
        data = torch.load(input_file)
        
        # Validate structure
        if not all(key in data for key in ['dataset_size', 'triples', 'data']):
            raise ValueError(f"Invalid dataset structure in {input_file}")
        
        # Add to combined data
        combined_triples.extend(data['triples'])
        combined_data.extend(data['data'])
        total_size += data['dataset_size']
        
        print(f"  Added {data['dataset_size']} samples")
    
    # Validate that lengths match before shuffling
    assert len(combined_triples) == total_size, f"Triples length ({len(combined_triples)}) doesn't match dataset_size ({total_size})"
    assert len(combined_data) == total_size, f"Data length ({len(combined_data)}) doesn't match dataset_size ({total_size})"
    
    # Shuffle the combined dataset
    print("Shuffling combined dataset...")
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Apply shuffle to both lists to maintain correspondence
    shuffled_triples = [combined_triples[i] for i in indices]
    shuffled_data = [combined_data[i] for i in indices]
    
    # Create combined dataset
    combined_dataset = {
        'dataset_size': total_size,
        'triples': shuffled_triples,
        'data': shuffled_data
    }
    
    # Validate that lengths still match after shuffling
    assert len(shuffled_triples) == total_size, f"Shuffled triples length ({len(shuffled_triples)}) doesn't match dataset_size ({total_size})"
    assert len(shuffled_data) == total_size, f"Shuffled data length ({len(shuffled_data)}) doesn't match dataset_size ({total_size})"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save combined dataset
    torch.save(combined_dataset, output_file)
    
    print(f"\nCombined and shuffled dataset saved to: {output_file}")
    print(f"Total samples: {total_size}")
    print(f"Combined from {len(input_files)} source datasets")


if __name__ == "__main__":
    # Set random seed for reproducible shuffling
    random.seed(42)
    
    # Define input and output files directly here
    input_files = [
        "datasets/path_plan_datasets_training/dataset_path_4_tp__with_subplans/dataset.pt",
        "datasets/path_plan_datasets_training/dataset_path_5/dataset.pt"
    ]
    output_file = "datasets/LUBM_PATH/dataset.pt"
    
    # Validate input files exist
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print("Input files:")
    for f in input_files:
        print(f"  - {f}")
    print(f"Output file: {output_file}")
    print()
    
    # Combine datasets
    combine_datasets(input_files, output_file)
    
    # Verify the combined dataset
    print("\nVerifying combined dataset...")
    data = torch.load(output_file)
    print(f"Verified: {data['dataset_size']} samples")
    print(f"Triples length: {len(data['triples'])}")
    print(f"Data length: {len(data['data'])}") 
