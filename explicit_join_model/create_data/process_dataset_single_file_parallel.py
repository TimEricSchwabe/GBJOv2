import pickle
import os
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from torch_geometric.data import Data, DataLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import Triple, Join, Query, Entity, join_order_to_adjacency_matrix, random, Datapoint, random_join_order
from tqdm import tqdm
import shutil
from data_loader import QueryDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def has_all_variable_triple_pattern(query_data: dict) -> bool:
    """
    Check if any triple pattern in the query has all variables.
    
    Args:
        query_data: Dictionary containing query data with "triples" key
        
    Returns:
        True if any triple pattern has all variables, False otherwise
    """
    for triple in query_data["triples"]:
        # Check if all components (subject, predicate, object) are variables
        # Variables start with '?' in SPARQL
        if all(component.startswith('?') for component in triple[:3]):  # [:3] to skip the trailing '.'
            return True
    return False

@dataclass
class SPARQLQuery:
    """Class to hold multiple join plans for query"""
    triples: List[List[str]]
    join_plans: List[Query]
    costs: List[float]
    torch_data: List[Data]  # Store torch_data for each plan
    triples_where: List[List[str]]  # Store triples_where for each plan
    
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

def create_random_join_orders(triples: List[List[str]], count: int, rdf2vec_dict, counts_dict) -> List[Query]:
    """
    Create multiple random join orders for a query.
    
    Args:
        triples: List of triple patterns
        count: Number of random join orders to create
        rdf2vec_dict: Dictionary of RDF2Vec embeddings
        counts_dict: Dictionary of entity counts
        
    Returns:
        List of Query objects representing different join orders
    """
    plans = []
    for i in range(count):
        # Use different seeds to ensure diversity in join orders
        seed = i + 1
        try:
            plan = random_join_order(triples, seed=seed)
            plans.append(plan)
        except Exception as e:
            print(f"Error creating random join order {i}: {e}")
    
    return plans

def query_to_sparql_query(query_data: dict, rdf2vec_dict, counts_dict, num_plans: int = 10) -> SPARQLQuery:
    """Convert a raw query to a SPARQLQuery with multiple random join plans and costs"""
    triples = query_data["triples"]
    join_plans = create_random_join_orders(triples, num_plans, rdf2vec_dict, counts_dict)
    
    # Calculate cost for each plan and create torch_data
    costs = []
    torch_data_list = []
    triples_where_list = []
    
    # Create mapping from triple pattern to index (create once and reuse)
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}

    cost_error = False

    for plan in join_plans:
        try:
            # Calculate cost
            try:
                cost = plan.root.get_cost() 
            except RuntimeError as e:
                print(f"Error calculating cost: {e}")
                cost_error = True
                break
            costs.append(cost)
            
            # Create torch_data with consistent triple indices
            datapoint = join_order_to_adjacency_matrix_consistent(plan, triple_to_index, rdf2vec=rdf2vec_dict, counts=counts_dict)
            data = datapoint.get_torch_data(cost=cost)
            torch_data_list.append(data)
            
            # Extract triples_where for this plan
            triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
            triples_where_list.append(triples_where)
        except Exception as e:
            raise e
            print(f"Error calculating cost or creating torch_data: {e}")
            costs.append(float('inf'))
            torch_data_list.append(None)  # Add None for failed plans
            triples_where_list.append([])  # Add empty list for failed plans
    
    if cost_error:
        return None
    
    return SPARQLQuery(
        triples=triples, 
        join_plans=join_plans, 
        costs=costs,
        torch_data=torch_data_list,
        triples_where=triples_where_list
    )

def join_order_to_adjacency_matrix_consistent(join_order: Query, triple_to_index: dict, seed = None, rdf2vec=None, counts=None) -> Datapoint:
    """
    Modified version of join_order_to_adjacency_matrix that ensures 
    consistent triple pattern indexing across different plans.
    
    Args:
        join_order: Query object representing the join order
        triple_to_index: Dictionary mapping triple string representations to indices
        seed: Random seed for variable indexing
        rdf2vec: RDF2Vec embeddings
        counts: Entity counts
        
    Returns:
        Datapoint object with adjacency matrix and embeddings
    """
    # There are len(join_order.triples) triple patterns and len(join_order.triples)-1 join nodes
    triples_num = join_order.triples_num
    nodes_num = triples_num * 2 - 1
    rng = random.Random(seed)

    variable_indexing = list(range(len(join_order.root.variables)))
    rng.shuffle(variable_indexing)
    variable_id_dict = dict(zip(
        join_order.root.variables,
        variable_indexing
    ))

    if isinstance(join_order.root, Triple):
        return Datapoint(
            nodes_order=[join_order.root],
            adjacency_matrix=np.zeros((1, 1)),
            embedding_matrix=join_order.root.get_embedding(variable_id_dict, rdf2vec, counts).reshape(1, 307),
            join_order=join_order
        )

    # Generate join node indices
    join_indexing = iter(range(triples_num, nodes_num))
    
    adjacency_matrix = np.zeros((nodes_num, nodes_num))
    embedding_matrix = np.zeros((nodes_num, 307))
    nodes_order = [None] * nodes_num  # Initialize with None

    def get_triple_index(triple: Triple) -> int:
        """Get consistent index for a triple based on the mapping"""
        return triple_to_index[str(triple)]
    
    def get_join_index(node: Join) -> int:
        """Get next join index"""
        return next(join_indexing)
    
    def get_node_embedding(node: Triple | Join) -> np.ndarray:
        if isinstance(node, Triple):
            return node.get_embedding(variable_id_dict, rdf2vec, counts)
        else:
            return node.get_embedding()

    # Process the join tree 
    root_index = next(join_indexing)
    q = [(join_order.root, root_index)]
    embedding_matrix[root_index] = join_order.root.get_embedding()
    nodes_order[root_index] = join_order.root

    while q:
        node, node_index = q.pop(0)
        
        # Process left child
        if isinstance(node.left, Triple):
            left_index = get_triple_index(node.left)
        else:
            left_index = get_join_index(node.left)
            
        adjacency_matrix[left_index, node_index] = 1
        embedding_matrix[left_index] = get_node_embedding(node.left)
        nodes_order[left_index] = node.left

        # Process right child
        if isinstance(node.right, Triple):
            right_index = get_triple_index(node.right)
        else:
            right_index = get_join_index(node.right)
            
        adjacency_matrix[right_index, node_index] = 1
        embedding_matrix[right_index] = get_node_embedding(node.right)
        nodes_order[right_index] = node.right

        # Add join nodes to the queue
        if isinstance(node.left, Join):
            q.append((node.left, left_index))
        
        if isinstance(node.right, Join):
            q.append((node.right, right_index))
    
    # Ensure all nodes have been assigned
    assert None not in nodes_order, "Some nodes were not assigned"
    
    return Datapoint(
        nodes_order=nodes_order,
        adjacency_matrix=adjacency_matrix,
        embedding_matrix=embedding_matrix,
        join_order=join_order
    )

def create_datapoints(sparql_query: SPARQLQuery, rdf2vec_dict, counts_dict) -> List[Tuple[List[str], Data]]:
    """Create datapoints for all plans in a SPARQLQuery"""
    results = []
    
    # Create mapping from triple pattern to index
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in sparql_query.triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}
    
    for i, plan in enumerate(sparql_query.join_plans):
        if sparql_query.torch_data[i] is not None:
            try:
                # Use the consistent version for datapoint creation
                datapoint = join_order_to_adjacency_matrix_consistent(
                    plan, 
                    triple_to_index, 
                    rdf2vec=rdf2vec_dict, 
                    counts=counts_dict
                )
                triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
                results.append((triples_where, sparql_query.torch_data[i]))
            except Exception as e:
                print(f"Error creating datapoint: {e}")
    
    return results

def save_sparql_queries_single_file(sparql_queries, output_file):
    """Save all SPARQLQuery objects to a single pickle file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(sparql_queries, f)
    
    print(f"Saved {len(sparql_queries)} SPARQLQuery objects to {output_file}")

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

def visualize_and_save_plans(sparql_query: SPARQLQuery, query_idx: int, output_dir: str):
    """
    Visualize and save each join plan for a query
    
    Args:
        sparql_query: SPARQLQuery object with multiple join plans
        query_idx: Index of the query
        output_dir: Directory to save visualizations
    """
    # Create directory for this query's plans
    query_dir = os.path.join(output_dir, f"query_{query_idx}")
    os.makedirs(query_dir, exist_ok=True)
    
    best_plan_idx = sparql_query.get_best_plan_index()
    
    # Visualize each plan
    for i, plan in enumerate(sparql_query.join_plans):
        try:
            # Define output path - mark the best plan with "_best"
            plan_label = f"_best_cost_{sparql_query.costs[i]:.0f}" if i == best_plan_idx else f"_cost_{sparql_query.costs[i]:.0f}"
            output_path = os.path.join(query_dir, f"plan_{i}{plan_label}")
            
            # Visualize and save the plan
            plan.visualize(output_file=output_path, format="png")
            print(f"  Saved visualization for query {query_idx}, plan {i} to {output_path}.png")
        except Exception as e:
            print(f"  Error visualizing plan {i} for query {query_idx}: {e}")

# Worker function for parallel processing
def process_single_query(query_data: dict, rdf2vec_dict, counts_dict, num_plans: int = 2) -> dict:
    """
    Process a single query and return results.
    This function will be executed in parallel workers.
    
    Args:
        query_data: Single query dictionary
        rdf2vec_dict: RDF2Vec embeddings dictionary  
        counts_dict: Entity counts dictionary
        num_plans: Number of random plans to create
        
    Returns:
        Dictionary containing processing results or None if failed
    """
    try:
        sparql_query = query_to_sparql_query(query_data, rdf2vec_dict, counts_dict, num_plans=num_plans)
        if sparql_query is None:
            return None
            
        # Extract datapoints for each plan
        all_triples = []
        all_torch_data = []
        
        for j, plan in enumerate(sparql_query.join_plans):
            if sparql_query.torch_data[j] is not None:
                all_triples.append(sparql_query.triples_where[j])
                all_torch_data.append(sparql_query.torch_data[j])
        
        return {
            'sparql_query': sparql_query,
            'triples': all_triples,
            'torch_data': all_torch_data,
            'costs': sparql_query.costs
        }
    except Exception as e:
        print(f"Error processing query in worker: {e}")
        return None

def process_queries_parallel(queries, rdf2vec_dict, counts_dict, num_workers=4, num_plans=2, batch_size=100):
    """
    Process queries in parallel using ProcessPoolExecutor.
    
    Args:
        queries: List of query dictionaries to process
        rdf2vec_dict: RDF2Vec embeddings dictionary
        counts_dict: Entity counts dictionary  
        num_workers: Number of parallel worker processes
        num_plans: Number of random plans per query
        batch_size: Number of queries to process in each batch
        
    Returns:
        Tuple of (sparql_queries, all_triples, all_torch_data)
    """
    sparql_queries = []
    all_triples = []
    all_torch_data = []
    
    # Process queries in batches to manage memory
    for batch_start in range(0, len(queries), batch_size):
        batch_end = min(batch_start + batch_size, len(queries))
        batch_queries = queries[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: queries {batch_start+1}-{batch_end}")
        
        # Submit batch to worker processes
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all queries in current batch
            future_to_query = {
                executor.submit(process_single_query, query, rdf2vec_dict, counts_dict, num_plans): (i + batch_start, query) 
                for i, query in enumerate(batch_queries)
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_query), total=len(batch_queries), desc=f"Batch {batch_start//batch_size + 1}"):
                query_idx, query = future_to_query[future]
                try:
                    result = future.result()
                    if result is not None:
                        sparql_queries.append(result['sparql_query'])
                        all_triples.extend(result['triples'])
                        all_torch_data.extend(result['torch_data'])
                        print(f"  Query {query_idx}: Plans costs: {result['costs']}")
                    else:
                        print(f"  Query {query_idx}: Failed to process")
                except Exception as e:
                    print(f"  Query {query_idx}: Error - {e}")
        
        print(f"Batch {batch_start//batch_size + 1} completed. Total processed: {len(sparql_queries)} queries")
    
    return sparql_queries, all_triples, all_torch_data


if __name__ == "__main__":
    # Load the RDF2Vec embeddings
    with open("/mnt/data/tim_triple_files/wikidata/rdf2vec100dim.pkl", "rb") as f:
        rdf2vec_dict = pickle.load(f)
        print(f"Loaded RDF2Vec embeddings: {len(rdf2vec_dict)} entities")

    with open("/mnt/data/tim_triple_files/wikidata/counts.pkl", "rb") as f:
        counts_dict = pickle.load(f)
        print(f"Loaded entity counts: {len(counts_dict)} entities")

    # Set paths
    input_file = "/mnt/data/tim_triple_files/wikidata/star/star_queries.json"
    dataset_dir = "/mnt/data/tim_triple_files/wikidata/star/plans/"
    sparql_queries_file = "sparql_queries_path_wikidata/queries.pkl"

    # Parallel processing configuration
    NUM_WORKERS = 4  # Number of parallel processes (adjust based on your CPU cores)
    BATCH_SIZE = 100  # Process queries in batches to manage memory
    MAX_QUERIES = 200000000000000000
    MIN_CARDINALITY = 1
    SAVE_INTERVAL = 1000
    
    # Number of random plans to create per query
    num_random_plans = 2
    
    # Load the queries
    print(f"Loading queries from {input_file}...")
    with open(input_file, "r") as f:
        queries = json.load(f)
    
    # Filter queries for min cardinality and all-variable triple patterns
    queries_filtered = [q for q in queries if q["y"] >= MIN_CARDINALITY and not has_all_variable_triple_pattern(q)]
    print(f"Found {len(queries_filtered)} queries with min cardinality {MIN_CARDINALITY}")
    
    # Shuffle queries
    random.shuffle(queries_filtered)
    
    # Limit to MAX_QUERIES
    queries_to_process = queries_filtered[:MAX_QUERIES]
    print(f"Processing {len(queries_to_process)} queries with {NUM_WORKERS} parallel workers")
    
    # Process queries in parallel
    print("Starting parallel processing...")
    sparql_queries, all_triples, all_torch_data = process_queries_parallel(
        queries_to_process, 
        rdf2vec_dict, 
        counts_dict, 
        num_workers=NUM_WORKERS, 
        num_plans=num_random_plans,
        batch_size=BATCH_SIZE
    )
    
    # Save final results
    print("\nSaving final results...")
    save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
    
    print(f"\nParallel dataset conversion complete!")
    print(f"Processed {len(sparql_queries)} queries successfully")
    print(f"Generated {len(all_torch_data)} datapoints total") 