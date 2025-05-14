import pickle
import os
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from torch_geometric.data import Data, DataLoader
from data import Triple, Join, Query, Entity, join_order_to_adjacency_matrix
from tqdm import tqdm
import shutil
from data_loader import QueryDataset

@dataclass
class SPARQLQuery:
    """Class to hold all possible join plans for a 3-triple pattern query"""
    triples: List[List[str]]
    join_plans: List[Query]
    costs: List[float]
    torch_data: List[Data]  # Store torch_data for each plan
    
    def get_best_plan_index(self) -> int:
        """Return the index of the plan with the lowest cost"""
        return np.argmin(self.costs)
    
    def get_best_plan(self) -> Query:
        """Return the plan with the lowest cost"""
        return self.join_plans[self.get_best_plan_index()]
    
    def get_best_cost(self) -> float:
        """Return the lowest cost among all plans"""
        return self.costs[self.get_best_plan_index()]
    
    def get_best_torch_data(self) -> Data:
        """Return the torch_data for the best plan"""
        return self.torch_data[self.get_best_plan_index()]

def create_all_join_orders(triples: List[List[str]], rdf2vec_dict, counts_dict) -> List[Query]:
    """
    Create all possible join orders for a 3-triple pattern query.
    Returns a list of 3 Query objects.
    """
    assert len(triples) == 3, "This function only works for queries with exactly 3 triple patterns"
    
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    
    # Create all possible join orders
    plans = []
    
    # Plan 1: (tp1 JOIN tp2) JOIN tp3
    join1 = Join(left=triple_objs[0], right=triple_objs[1])
    plan1 = Query(root=Join(left=join1, right=triple_objs[2]), triples_num=3)
    plans.append(plan1)
    
    # Plan 2: (tp1 JOIN tp3) JOIN tp2
    join2 = Join(left=triple_objs[0], right=triple_objs[2])
    plan2 = Query(root=Join(left=join2, right=triple_objs[1]), triples_num=3)
    plans.append(plan2)
    
    # Plan 3: (tp2 JOIN tp3) JOIN tp1
    join3 = Join(left=triple_objs[1], right=triple_objs[2])
    plan3 = Query(root=Join(left=join3, right=triple_objs[0]), triples_num=3)
    plans.append(plan3)
    
    return plans

def query_to_sparql_query(query_data: dict, rdf2vec_dict, counts_dict) -> SPARQLQuery:
    """Convert a raw query to a SPARQLQuery with all join plans and costs"""
    triples = query_data["triples"]
    join_plans = create_all_join_orders(triples, rdf2vec_dict, counts_dict)
    
    # Calculate cost for each plan and create torch_data
    costs = []
    torch_data_list = []
    
    for plan in join_plans:
        try:
            # Calculate cost
            cost = plan.root.get_cost()
            costs.append(cost)
            
            # Create torch_data
            datapoint = join_order_to_adjacency_matrix(plan, rdf2vec=rdf2vec_dict, counts=counts_dict)
            data = datapoint.get_torch_data()
            torch_data_list.append(data)
        except Exception as e:
            print(f"Error calculating cost or creating torch_data: {e}")
            costs.append(float('inf'))
            torch_data_list.append(None)  # Add None for failed plans
    
    return SPARQLQuery(
        triples=triples, 
        join_plans=join_plans, 
        costs=costs,
        torch_data=torch_data_list
    )

def create_datapoints(sparql_query: SPARQLQuery, rdf2vec_dict, counts_dict) -> List[Tuple[List[str], Data]]:
    """Create datapoints for all plans in a SPARQLQuery"""
    results = []
    
    for i, plan in enumerate(sparql_query.join_plans):
        if sparql_query.torch_data[i] is not None:
            try:
                datapoint = join_order_to_adjacency_matrix(plan, rdf2vec=rdf2vec_dict, counts=counts_dict)
                triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
                results.append((triples_where, sparql_query.torch_data[i]))
            except Exception as e:
                print(f"Error creating datapoint: {e}")
    
    return results

def save_sparql_queries(sparql_queries, output_dir):
    """Save individual SPARQLQuery objects to files in the output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, query in enumerate(sparql_queries):
        filename = os.path.join(output_dir, f'query_{i}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(query, f)
    
    print(f"Saved {len(sparql_queries)} SPARQLQuery objects to {output_dir}")

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

if __name__ == "__main__":
    # Load the RDF2Vec embeddings
    with open("/home/tim/query_optimization/queries/rdf2vec100dim.pkl", "rb") as f:
        rdf2vec_dict = pickle.load(f)
    
    with open("/home/tim/query_optimization/queries/counts.pkl", "rb") as f:
        counts_dict = pickle.load(f)

    
    # Set paths
    input_file = "/home/tim/query_optimization/queries/Star_Queries.json"
    dataset_dir = "dataset_stars_3"
    sparql_queries_dir = "sparql_queries_3"
    
    # Load the queries
    print(f"Loading queries from {input_file}...")
    with open(input_file, "r") as f:
        queries = json.load(f)
    
    # Filter queries with exactly 3 triple patterns
    queries_3tp = [q for q in queries if len(q["triples"]) == 3]
    print(f"Found {len(queries_3tp)} queries with exactly 3 triple patterns")
    
    # Process queries
    sparql_queries = []
    all_triples = []
    all_torch_data = []
    
    for i, query in enumerate(tqdm(queries_3tp[:], desc="Processing queries")):  # Process first 10 queries for testing
        try:
            print(f"Processing query {i+1}/{len(queries_3tp)}")
            sparql_query = query_to_sparql_query(query, rdf2vec_dict, counts_dict)
            sparql_queries.append(sparql_query)
            
            # Add datapoints for each plan
            for j, plan in enumerate(sparql_query.join_plans):
                if sparql_query.torch_data[j] is not None:
                    try:
                        datapoint = join_order_to_adjacency_matrix(plan, rdf2vec=rdf2vec_dict, counts=counts_dict)
                        triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
                        all_triples.append(triples_where)
                        all_torch_data.append(sparql_query.torch_data[j])
                    except Exception as e:
                        print(f"Error creating datapoint for plan {j}: {e}")
            
            # Print costs for debugging
            print(f"  Plans costs: {sparql_query.costs}")
            print(f"  Best plan index: {sparql_query.get_best_plan_index()}")
        except Exception as e:
            print(f"Error processing query {i}: {e}")
    
    # Save SPARQLQuery objects
    save_sparql_queries(sparql_queries, sparql_queries_dir)
    
    # Save dataset for PyTorch Geometric
    save_dataset(all_triples, all_torch_data, dataset_dir, clear_existing=True)
    
    # Test loading the dataset
    print("\nTesting dataset loading...")
    dataset = QueryDataset(root=dataset_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Test batch loading
    batch_size = 32
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    
    print(f"\nLoading {len(loader)} batches with batch_size={batch_size}")
    for i, batch in enumerate(loader):
        if i == 0:
            print(f"First batch shape: x={batch.x.shape}, edge_index={batch.edge_index.shape}")
            print(f"First batch y: {batch.y[:5]}")
        
        if i >= 2:  # Just test a few batches
            break
    
    print("\nDataset conversion and testing complete!") 