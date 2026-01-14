import torch
import os

def combine_datasets(path1, path2, output_path):
    """Combine two dataset.pt files into one."""
    
    # Load both datasets
    data1 = torch.load(os.path.join(path1, 'dataset.pt'), weights_only=False)
    data2 = torch.load(os.path.join(path2, 'dataset.pt'), weights_only=False)
    
    print(f"Dataset 1: {len(data1['data'])} samples")
    print(f"Dataset 2: {len(data2['data'])} samples")
    
    # Combine the data lists
    combined_data = data1['data'] + data2['data']
    
    # Combine triples if they exist
    combined_triples = []
    if 'triples' in data1:
        combined_triples.extend(data1['triples'])
    if 'triples' in data2:
        combined_triples.extend(data2['triples'])
    
    # Create combined dataset dict
    combined = {
        'dataset_size': len(combined_data),
        'triples': combined_triples,
        'data': combined_data
    }
    
    # Save combined dataset
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'dataset.pt')
    torch.save(combined, output_file)
    
    print(f"Combined dataset saved to {output_file}")
    print(f"Total samples: {len(combined_data)}")

# Usage
if __name__ == "__main__":

    path1 = "DATASET PATH 1"
    path2 = "DATASET PATH 2"
    output_path = "DATASET PATH COMBINED"
    
    combine_datasets(path1, path2, output_path)