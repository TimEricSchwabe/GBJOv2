#!/usr/bin/env python3
"""
Script to visualize and save information about the first plan of each query from a pickle file.

This script loads SPARQL queries from a pickle file, extracts the first plan of each query,
visualizes it using the existing plan.visualize method, and creates text files with 
readable information about costs, triples, and torch data.
"""

import sys
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse
from typing import List, Optional

# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required classes and functions
from src.create_data.create_optimization_data import SPARQLQuery
from data import Triple, Entity
from utils.data_utils import (
    load_sparql_queries,
    plan_to_string
)


def save_plan_info_to_text(query: SPARQLQuery, query_idx: int, plan_idx: int, 
                          output_dir: str) -> None:
    """
    Save detailed information about a query plan to a text file.
    
    Args:
        query: SPARQLQuery object
        query_idx: Index of the query
        plan_idx: Index of the plan (typically 0 for first plan)
        output_dir: Directory to save the text file
    """
    filename = os.path.join(output_dir, f"query_{query_idx}_plan_{plan_idx}_info.txt")
    
    with open(filename, 'w') as f:
        f.write(f"=== Query {query_idx} Plan {plan_idx} Information ===\n\n")
        
        # Basic information
        f.write(f"Number of triple patterns: {len(query.triples)}\n")
        f.write(f"Number of total plans: {len(query.join_plans)}\n")
        f.write(f"Number of cost entries: {len(query.costs)}\n")
        f.write(f"Number of torch_data entries: {len(query.torch_data)}\n\n")
        
        # Triple patterns
        f.write("=== Triple Patterns ===\n")
        for i, triple in enumerate(query.triples):
            f.write(f"Triple {i}: {triple[0]} {triple[1]} {triple[2]}\n")
        f.write("\n")
        
        # Plan information
        if plan_idx < len(query.join_plans) and query.join_plans[plan_idx] is not None:
            plan = query.join_plans[plan_idx]
            f.write(f"=== Plan {plan_idx} Structure ===\n")
            f.write(f"Plan string: {plan_to_string(plan)}\n")
            f.write(f"Plan triples_num: {plan.triples_num}\n")
            
            # Try to get the cost
            try:
                cost = plan.root.get_cost()
                f.write(f"Plan cost (calculated): {cost}\n")
            except Exception as e:
                f.write(f"Plan cost (calculated): Error - {e}\n")
            f.write("\n")
        else:
            f.write(f"=== Plan {plan_idx} Structure ===\n")
            f.write("Plan not available or is None\n\n")
        
        # Cost information
        if plan_idx < len(query.costs) and query.costs[plan_idx] is not None:
            f.write(f"=== Cost Information ===\n")
            f.write(f"Stored cost: {query.costs[plan_idx]}\n\n")
        else:
            f.write(f"=== Cost Information ===\n")
            f.write("No stored cost available\n\n")
        
        # Torch data information
        if plan_idx < len(query.torch_data) and query.torch_data[plan_idx] is not None:
            torch_data = query.torch_data[plan_idx]
            f.write(f"=== Torch Data Information ===\n")
            f.write(f"Node features shape (x): {torch_data.x.shape}\n")
            f.write(f"Edge index shape: {torch_data.edge_index.shape}\n")
            f.write(f"Number of nodes: {torch_data.x.shape[0]}\n")
            f.write(f"Number of edges: {torch_data.edge_index.shape[1]}\n")
            f.write(f"Feature dimension: {torch_data.x.shape[1]}\n\n")
            
            # Edge index details
            f.write("=== Edge Index ===\n")
            edge_index = torch_data.edge_index
            num_edges_to_show = min(20, edge_index.shape[1])
            for i in range(num_edges_to_show):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                f.write(f"Edge {i}: {src} -> {dst}\n")
            if edge_index.shape[1] > 20:
                f.write(f"... and {edge_index.shape[1] - 20} more edges\n")
            f.write("\n")
            
            
        else:
            f.write(f"=== Torch Data Information ===\n")
            f.write("No torch data available\n\n")


def visualize_first_plans(queries_file: str, output_dir: str = "plan_visuals", 
                         max_queries: Optional[int] = None) -> None:
    """
    Visualize and save information about the first plan of each query.
    
    Args:
        queries_file: Path to the pickle file containing SPARQLQuery objects
        output_dir: Directory to save visualizations and text files
        max_queries: Maximum number of queries to process (None for all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading queries from {queries_file}...")
    
    # Load queries
    try:
        sparql_queries = load_sparql_queries(queries_file, max_queries)
        print(f"Loaded {len(sparql_queries)} queries")
    except Exception as e:
        print(f"Error loading queries: {e}")
        return
    
    successful_visualizations = 0
    failed_visualizations = 0
    
    # Process each query
    for query_idx, query in enumerate(tqdm(sparql_queries, desc="Processing queries")):
        try:
            plan_idx = 0  # Always use the first plan
            
            # Check if the first plan exists
            if (plan_idx >= len(query.join_plans) or 
                query.join_plans[plan_idx] is None):
                print(f"Warning: Query {query_idx} has no plan at index {plan_idx}. Skipping.")
                failed_visualizations += 1
                continue
            
            plan = query.join_plans[plan_idx]
            
            # Create base filename
            base_filename = f"query_{query_idx}_plan_{plan_idx}"
            
            # Visualize the plan
            try:
                visualization_path = os.path.join(output_dir, base_filename)
                plan.visualize(output_file=visualization_path, format="png")
                print(f"Saved visualization for query {query_idx} to {visualization_path}.png")
            except Exception as viz_error:
                print(f"Warning: Failed to visualize query {query_idx}: {viz_error}")
                # Continue with text file creation even if visualization fails
            
            # Save detailed information to text file
            try:
                save_plan_info_to_text(query, query_idx, plan_idx, output_dir)
                print(f"Saved info for query {query_idx} to {base_filename}_info.txt")
                successful_visualizations += 1
            except Exception as text_error:
                print(f"Warning: Failed to save text info for query {query_idx}: {text_error}")
                failed_visualizations += 1
                
        except Exception as e:
            print(f"Error processing query {query_idx}: {e}")
            failed_visualizations += 1
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful_visualizations}")
    print(f"Failed to process: {failed_visualizations}")
    print(f"Total queries: {len(sparql_queries)}")
    print(f"Output directory: {output_dir}")


def main():
    """Main function to visualize plans with hardcoded parameters."""
    # Define parameters directly in code
    queries_file = "/home/tim/query_optimization/datasets/plans/wikidata_path_plan_datasets_optimization/queries.pkl"
    output_dir = "plan_visuals"
    max_queries = 20
    
    # Check if queries file exists
    if not os.path.exists(queries_file):
        print(f"Error: Queries file '{queries_file}' not found.")
        return 1
    
    try:
        visualize_first_plans(queries_file, output_dir, max_queries)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 