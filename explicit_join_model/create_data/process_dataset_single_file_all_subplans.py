import pickle
import os
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from torch_geometric.data import Data, DataLoader
from data import Triple, Join, Query, Entity, join_order_to_adjacency_matrix, random, Datapoint, random_join_order
from tqdm import tqdm
import shutil
from data_loader import QueryDataset

@dataclass
class SubPlan:
    """Class to hold a subplan with its cost and representation"""
    root: Triple | Join
    triples_num: int
    cost: float
    torch_data: Optional[Data] = None
    parent_plan_idx: Optional[int] = None
    
    def __str__(self):
        """String representation of the subplan"""
        return f"SubPlan(triples={self.triples_num}, cost={self.cost:.2f})"
    
    def to_query(self):
        """Convert subplan to Query object"""
        return Query(root=self.root, triples_num=self.triples_num)

@dataclass
class SPARQLQuery:
    """Class to hold multiple join plans for a 8-triple pattern query, including subplans"""
    triples: List[List[str]]
    join_plans: List[Query]
    costs: List[float]
    torch_data: List[Data]  # Store torch_data for each full plan
    subplans: List[List[SubPlan]]  # Store subplans for each full plan
    
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
    
    def get_all_subplans(self) -> List[SubPlan]:
        """Return all subplans from all plans flattened into a single list"""
        return [subplan for plan_subplans in self.subplans for subplan in plan_subplans]

def extract_subplans(query: Query, parent_plan_idx: int) -> List[SubPlan]:
    """
    Extract all subplans from a query recursively.
    
    Args:
        query: Query object representing a join plan
        parent_plan_idx: Index of the parent plan this subplan belongs to
        
    Returns:
        List of SubPlan objects representing all subplans in the query
    """
    subplans = []
    seen_subplans = set()  # To avoid duplicates based on string representation
    
    def collect_subplans(node: Triple | Join, triple_count: int):
        """Recursively collect subplans from the query tree"""
        # Skip if we've seen this exact subplan before
        node_str = str(node)
        if node_str in seen_subplans:
            return
        
        seen_subplans.add(node_str)
        
        # Calculate cost with special handling for triples
        try:
            # If it's a standalone triple, use get_cardinality
            # For join nodes, get_cost already uses 0 for triple costs
            if isinstance(node, Triple):
                cost = node.get_cardinality()
            else:
                cost = node.get_cost()
            
            # Create a SubPlan object
            subplan = SubPlan(
                root=node,
                triples_num=triple_count,
                cost=cost,
                parent_plan_idx=parent_plan_idx
            )
            
            subplans.append(subplan)
        except Exception as e:
            print(f"Error calculating cost for subplan: {e}")
        
        # Recursively process join nodes
        if isinstance(node, Join):
            # Process left subtree
            if isinstance(node.left, Triple):
                collect_subplans(node.left, 1)
            else:
                left_triple_count = count_triples_in_node(node.left)
                collect_subplans(node.left, left_triple_count)
            
            # Process right subtree
            if isinstance(node.right, Triple):
                collect_subplans(node.right, 1)
            else:
                right_triple_count = count_triples_in_node(node.right)
                collect_subplans(node.right, right_triple_count)
    
    def count_triples_in_node(node: Triple | Join) -> int:
        """Count the number of triple patterns in a node"""
        if isinstance(node, Triple):
            return 1
        elif isinstance(node, Join):
            return count_triples_in_node(node.left) + count_triples_in_node(node.right)
        return 0
    
    # Start the recursive collection from the root
    collect_subplans(query.root, query.triples_num)
    
    return subplans

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
    """Convert a raw query to a SPARQLQuery with multiple random join plans, costs, and all subplans"""
    triples = query_data["triples"]
    join_plans = create_random_join_orders(triples, num_plans, rdf2vec_dict, counts_dict)
    
    # Calculate cost for each plan and create torch_data
    costs = []
    torch_data_list = []
    all_subplans = []
    
    # Create mapping from triple pattern to index
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}
    
    for i, plan in enumerate(join_plans):
        try:
            # Calculate cost (Triple.get_cost already returns 0)
            cost = plan.root.get_cost()
            costs.append(cost)
            
            # Create datapoint with consistent triple indices
            datapoint = join_order_to_adjacency_matrix_consistent(plan, triple_to_index, rdf2vec=rdf2vec_dict, counts=counts_dict)
            
            # Create torch_data directly from matrices to avoid additional get_cost() calls
            torch_data = Data(
                x=torch.tensor(datapoint.embedding_matrix, dtype=torch.float),
                edge_index=torch.tensor(datapoint.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),
                y=torch.tensor([cost], dtype=torch.float)  # Use pre-calculated cost
            )
            torch_data_list.append(torch_data)
            
            # Extract all subplans
            subplans = extract_subplans(plan, i)
            all_subplans.append(subplans)
            
            print(f"  Plan {i}: cost={cost:.2f}, subplans={len(subplans)}")
            
        except Exception as e:
            print(f"Error calculating cost or creating torch_data: {e}")
            costs.append(float('inf'))
            torch_data_list.append(None)  # Add None for failed plans
            all_subplans.append([])  # Empty list for failed plans
    
    return SPARQLQuery(
        triples=triples, 
        join_plans=join_plans, 
        costs=costs,
        torch_data=torch_data_list,
        subplans=all_subplans
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

def create_subplan_datapoint(subplan: SubPlan, rdf2vec_dict, counts_dict) -> Tuple[List[str], Data]:
    """Create a datapoint for a subplan with its own local indexing"""
    # Convert subplan to Query
    query = subplan.to_query()
    
    try:
        # Create fresh datapoint using regular join_order_to_adjacency_matrix
        # Pass the query object directly instead of just the root
        datapoint = join_order_to_adjacency_matrix(
            query, 
            rdf2vec=rdf2vec_dict, 
            counts=counts_dict
        )
        
        # Get triples in where clause
        triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
        
        # Create torch data directly from the matrices to avoid additional get_cost() calls
        torch_data = Data(
            x=torch.tensor(datapoint.embedding_matrix, dtype=torch.float),
            edge_index=torch.tensor(datapoint.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),
            y=torch.tensor([subplan.cost], dtype=torch.float)  # Use our pre-calculated cost
        )
        
        # Store the torch data in the subplan for later reuse
        subplan.torch_data = torch_data
        
        return triples_where, torch_data
    except Exception as e:
        print(f"  Error in join_order_to_adjacency_matrix: {e}")
        raise

def process_subplans(sparql_query: SPARQLQuery, rdf2vec_dict, counts_dict) -> List[Tuple[List[str], Data]]:
    """
    Process all subplans and convert them to datapoints.
    Group subplans by size and select one from each size,
    excluding subplans that have the same size as the full query (8 triple patterns).
    
    Args:
        sparql_query: SPARQLQuery with subplans
        rdf2vec_dict: Dictionary of RDF2Vec embeddings
        counts_dict: Dictionary of entity counts
        
    Returns:
        List of (triples_where, torch_data) tuples for selected subplans
    """
    results = []
    full_query_size = len(sparql_query.triples)  # Size of the full query (8 in this case)
    
    # Process plans and group subplans by size for each plan
    for i, plan_subplans in enumerate(sparql_query.subplans):
        print(f"  Processing {len(plan_subplans)} subplans for plan {i}")
        
        # Group subplans by size (number of triple patterns)
        subplans_by_size = {}
        for subplan in plan_subplans:
            # Use triples_num as the key
            size = subplan.triples_num
            if size not in subplans_by_size:
                subplans_by_size[size] = []
            subplans_by_size[size].append(subplan)
        
        # Log the sizes found
        print(f"  Plan {i} has subplans with sizes: {sorted(subplans_by_size.keys())}")
        
        # Select one subplan from each size group (the first one)
        # BUT exclude those with size equal to the full query size
        selected_subplans = []
        for size, subplans in subplans_by_size.items():
            if size != full_query_size:  # Skip subplans with the same size as the full query
                selected_subplans.append(subplans[0])
        
        print(f"  Selected {len(selected_subplans)} subplans (one per size, excluding size {full_query_size}) from plan {i}")
        
        # Create datapoints for selected subplans
        for subplan in selected_subplans:
            try:
                # Create datapoint for subplan
                triples_where, torch_data = create_subplan_datapoint(
                    subplan,
                    rdf2vec_dict,
                    counts_dict
                )
                
                results.append((triples_where, torch_data))
            except Exception as e:
                print(f"  Error creating datapoint for subplan of size {subplan.triples_num}: {e}")
    
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

def visualize_and_save_plans(sparql_query: SPARQLQuery, query_idx: int, output_dir: str, include_subplans: bool = False):
    """
    Visualize and save each join plan and optionally subplans for a query
    
    Args:
        sparql_query: SPARQLQuery object with multiple join plans and subplans
        query_idx: Index of the query
        output_dir: Directory to save visualizations
        include_subplans: Whether to visualize subplans as well
    """
    # Create directory for this query's plans
    query_dir = os.path.join(output_dir, f"query_{query_idx}")
    os.makedirs(query_dir, exist_ok=True)
    
    best_plan_idx = sparql_query.get_best_plan_index()
    
    # Visualize each full plan
    for i, plan in enumerate(sparql_query.join_plans):
        try:
            # Define output path - mark the best plan with "_best"
            plan_label = f"_best_cost_{sparql_query.costs[i]:.0f}" if i == best_plan_idx else f"_cost_{sparql_query.costs[i]:.0f}"
            output_path = os.path.join(query_dir, f"plan_{i}{plan_label}")
            
            # Visualize and save the plan
            plan.visualize(output_file=output_path, format="png")
            print(f"  Saved visualization for query {query_idx}, plan {i} to {output_path}.png")
            
            # Visualize subplans if requested
            if include_subplans and i < len(sparql_query.subplans):
                # Create directory for this plan's subplans
                subplans_dir = os.path.join(query_dir, f"plan_{i}_subplans")
                os.makedirs(subplans_dir, exist_ok=True)
                
                for j, subplan in enumerate(sparql_query.subplans[i]):
                    try:
                        subplan_query = subplan.to_query()
                        subplan_output_path = os.path.join(subplans_dir, f"subplan_{j}_tp{subplan.triples_num}_cost_{subplan.cost:.0f}")
                        subplan_query.visualize(output_file=subplan_output_path, format="png")
                    except Exception as e:
                        print(f"  Error visualizing subplan {j} for plan {i}: {e}")
        except Exception as e:
            print(f"  Error visualizing plan {i} for query {query_idx}: {e}")

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

if __name__ == "__main__":
    # Load the RDF2Vec embeddings
    with open("/home/tim/query_optimization/datasets/queries/rdf2vec100dim.pkl", "rb") as f:
        rdf2vec_dict = pickle.load(f)
    
    with open("/home/tim/query_optimization/datasets/queries/counts.pkl", "rb") as f:
        counts_dict = pickle.load(f)

    
    # Set paths
    input_file = "/home/tim/query_optimization/datasets/queries/Path_Queries.json"
    dataset_dir = "dataset_path_8_with_subplans"
    sparql_queries_file = "sparql_path_queries_8_with_subplans/queries.pkl"
    visualization_dir = "join_plan_visualizations_path_8_with_subplans"
    
    # Create visualization directory
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Load the queries
    print(f"Loading queries from {input_file}...")
    with open(input_file, "r") as f:
        queries = json.load(f)
    
    # Filter queries with exactly 8 triple patterns and no all-variable patterns
    queries_8tp = [q for q in queries if len(q["triples"]) == 4 and not has_all_variable_triple_pattern(q)]
    print(f"Found {len(queries_8tp)} valid queries with exactly 8 triple patterns (excluding queries with all-variable patterns)")
    
    # Number of random plans to create per query
    num_random_plans = 3 
    
    # Process queries
    sparql_queries = []
    all_triples = []
    all_torch_data = []
    
    for i, query in enumerate(tqdm(queries_8tp[:30000], desc="Processing queries")):
        try:
            print(f"Processing query {i+1}/{len(queries_8tp[:30000])}")
            sparql_query = query_to_sparql_query(query, rdf2vec_dict, counts_dict, num_plans=num_random_plans)
            sparql_queries.append(sparql_query)
            
            # Visualize and save all plans for this query (uncomment to visualize)
            #visualize_and_save_plans(sparql_query, i, visualization_dir, include_subplans=True)
            
            # Process subplans - one per size per plan
            subplan_results = process_subplans(sparql_query, rdf2vec_dict, counts_dict)
            print(f"  Generated {len(subplan_results)} subplan datapoints (one per size per plan)")
            
            # Add selected subplans to combined list
            for triples_where, torch_data in subplan_results:
                all_triples.append(triples_where)
                all_torch_data.append(torch_data)
            
            # Also add full plans
            for j, plan in enumerate(sparql_query.join_plans):
                # We don't need to check sparql_query.torch_data[j] since we're creating our own torch data
                try:
                    datapoint = join_order_to_adjacency_matrix_consistent(
                        plan, 
                        {str(triple): i for i, triple in enumerate([Triple(*(Entity(name=name) for name in triple[:3])) for triple in query["triples"]])}, 
                        rdf2vec=rdf2vec_dict, 
                        counts=counts_dict
                    )
                    triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
                    
                    # Create torch data with the pre-calculated cost from the SPARQLQuery
                    torch_data = Data(
                        x=torch.tensor(datapoint.embedding_matrix, dtype=torch.float),
                        edge_index=torch.tensor(datapoint.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),
                        y=torch.tensor([sparql_query.costs[j]], dtype=torch.float)
                    )
                    
                    all_triples.append(triples_where)
                    all_torch_data.append(torch_data)
                except Exception as e:
                    print(f"Error creating datapoint for plan {j}: {e}")
            
            # Print costs for debugging
            print(f"  Full plans costs: {sparql_query.costs}")
            print(f"  Best plan index: {sparql_query.get_best_plan_index()}")
            total_subplans_selected = len(subplan_results)
            total_subplans_available = sum(len(subplans) for subplans in sparql_query.subplans)
            print(f"  Selected {total_subplans_selected} out of {total_subplans_available} available subplans")
            print(f"  Total datapoints so far: {len(all_triples)}")
        except Exception as e:
            print(f"Error processing query {i}: {e}")
    
    # Save all SPARQLQuery objects to a single file
    save_sparql_queries_single_file(sparql_queries, sparql_queries_file)
    
    # Save dataset to a single file
    save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
    
    # Print summary statistics
    total_full_plans = sum(len(q.join_plans) for q in sparql_queries)
    total_subplans = sum(sum(len(subplans) for subplans in q.subplans) for q in sparql_queries)
    
    print("\nDataset conversion complete!") 
    print(f"Total queries: {len(sparql_queries)}")
    print(f"Total full plans: {total_full_plans}")
    print(f"Total subplans: {total_subplans}")
    print(f"Total datapoints: {len(all_triples)}") 