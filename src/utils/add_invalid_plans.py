import os
import torch
import pickle
import sys
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# Add src to path to import necessary modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from create_data.create_cost_model_training_data import SPARQLQuery, generate_invalid_plan

def add_invalid_plans_to_dataset(input_file: str, output_file: str):
    """
    Load a dataset of SPARQLQuery objects, add one invalid plan to each query if missing,
    and save the modified dataset.
    
    Args:
        input_file: Path to the input dataset (queries.pt or pickle file)
        output_file: Path to save the modified dataset
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load dataset
    if input_file.endswith('.pt'):
        try:
            # Try loading with torch.load first (weights_only=False required for custom objects)
            sparql_queries = torch.load(input_file, weights_only=False)
        except (RuntimeError, pickle.UnpicklingError, TypeError):
            # Fallback to pickle
            with open(input_file, 'rb') as f:
                sparql_queries = pickle.load(f)
    else:
        with open(input_file, 'rb') as f:
            sparql_queries = pickle.load(f)
            
    print(f"Loaded {len(sparql_queries)} queries.")
    
    modified_count = 0
    
    for query in tqdm(sparql_queries, desc="Processing queries"):
        # Check if query already has an invalid plan (infinite cost)
        has_invalid = any(c == float('inf') for c in query.costs)
        
        if not has_invalid:
            # Find a valid plan to base the invalid one on
            valid_idx = next((i for i, d in enumerate(query.torch_data) if d is not None), -1)
            
            if valid_idx != -1:
                valid_data = query.torch_data[valid_idx]
                try:
                    invalid_data = generate_invalid_plan(valid_data)
                    
                    # Append invalid plan data
                    query.join_plans.append(None)
                    query.costs.append(float('inf'))
                    query.torch_data.append(invalid_data)
                    
                    # Copy triples_where from the valid plan used as base
                    # (Assuming context/structure is similar enough for where clause purposes, 
                    # though invalid plan has no valid structure)
                    if hasattr(query, 'triples_where'):
                        query.triples_where.append(query.triples_where[valid_idx])
                        
                    modified_count += 1
                except Exception as e:
                    print(f"Error generating invalid plan for a query: {e}")
    
    print(f"Added invalid plans to {modified_count} queries.")
    
    # Save modified dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file.endswith('.pt'):
        torch.save(sparql_queries, output_file)
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(sparql_queries, f)
            
    print(f"Saved modified dataset to {output_file}")

if __name__ == "__main__":
    # Configuration
    input_file = "...queries.pt" # Replace with your input file path
    output_file = "...queries_with_invalid.pt" # Replace with your output file path
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
        
    add_invalid_plans_to_dataset(input_file, output_file)

