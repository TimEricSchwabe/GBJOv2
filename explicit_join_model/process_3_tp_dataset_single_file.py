import pickle
import os
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from torch_geometric.data import Data, DataLoader
from data import Triple, Join, Query, Entity, join_order_to_adjacency_matrix, random, Datapoint
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
    
    # Create mapping from triple pattern to index
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}
    
    for plan in join_plans:
        try:
            # Calculate cost
            cost = plan.root.get_cost()
            costs.append(cost)
            
            # Create torch_data with consistent triple indices
            datapoint = join_order_to_adjacency_matrix_consistent(plan, triple_to_index, rdf2vec=rdf2vec_dict, counts=counts_dict)
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

if __name__ == "__main__":
    # Load the RDF2Vec embeddings
    with open("/home/tim/query_optimization/queries/rdf2vec100dim.pkl", "rb") as f:
        rdf2vec_dict = pickle.load(f)
    
    with open("/home/tim/query_optimization/queries/counts.pkl", "rb") as f:
        counts_dict = pickle.load(f)

    
    # Set paths
    input_file = "/home/tim/query_optimization/queries/Star_Queries.json"
    dataset_dir = "dataset_stars_3_single"
    sparql_queries_file = "sparql_queries_3_single/queries.pkl"
    
    # Load the queries
    print(f"Loading queries from {input_file}...")
    with open(input_file, "r") as f:
        queries = json.load(f)
    
    # Filter queries with exactly 3 triple patterns
    queries_3tp = [q for q in queries if len(q["triples"]) == 8]
    print(f"Found {len(queries_3tp)} queries with exactly 3 triple patterns")
    
    # Process queries
    sparql_queries = []
    all_triples = []
    all_torch_data = []
    
    for i, query in enumerate(tqdm(queries_3tp[:], desc="Processing queries")):  # Process first 300 queries as in original
        try:
            print(f"Processing query {i+1}/{len(queries_3tp)}")
            sparql_query = query_to_sparql_query(query, rdf2vec_dict, counts_dict)
            sparql_queries.append(sparql_query)
            
            # Add datapoints for each plan
            for j, plan in enumerate(sparql_query.join_plans):
                if sparql_query.torch_data[j] is not None:
                    try:
                        datapoint = join_order_to_adjacency_matrix_consistent(
                            plan, 
                            {str(triple): i for i, triple in enumerate([Triple(*(Entity(name=name) for name in triple[:3])) for triple in query["triples"]])}, 
                            rdf2vec=rdf2vec_dict, 
                            counts=counts_dict
                        )
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
    
    # Save all SPARQLQuery objects to a single file
    save_sparql_queries_single_file(sparql_queries, sparql_queries_file)
    
    # Save dataset to a single file
    save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
    
    print("\nDataset conversion complete!") 