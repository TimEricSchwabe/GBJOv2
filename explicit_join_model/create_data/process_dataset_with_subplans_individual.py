import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pickle
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from torch_geometric.data import Data, DataLoader
from explicit_join_model.data import Triple, Join, Query, Entity, join_order_to_adjacency_matrix, random, Datapoint, random_join_order
from tqdm import tqdm
import shutil

@dataclass
class SubPlan:
    """Class to hold a subplan with its cost and representation"""
    root: Triple | Join
    triples_num: int
    cost: Optional[float] = None
    torch_data: Optional[Data] = None
    parent_plan_idx: Optional[int] = None
    
    def __str__(self):
        """String representation of the subplan"""
        return f"SubPlan(triples={self.triples_num}, cost={self.cost})"
    
    def to_query(self):
        """Convert subplan to Query object"""
        return Query(root=self.root, triples_num=self.triples_num)

@dataclass
class SPARQLQuery:
    """Class to hold a single join plan for a query"""
    triples: List[List[str]]
    join_plans: List[Query]
    costs: List[Optional[float]]
    torch_data: List[Data]  # Store torch_data for each plan
    
    def get_best_plan_index(self) -> int:
        """Return the index of the plan with the lowest cost"""
        # Since costs are None, just return the first plan
        return 0
    
    def get_best_plan(self) -> Query:
        """Return the plan with the lowest cost"""
        return self.join_plans[self.get_best_plan_index()]
    
    def get_best_cost(self) -> Optional[float]:
        """Return the lowest cost among all plans"""
        return self.costs[self.get_best_plan_index()]
    
    def get_best_torch_data(self) -> Data:
        """Return the torch_data for the best plan"""
        return self.torch_data[self.get_best_plan_index()]

def extract_subplans_left_linear(query: Query, parent_plan_idx: int, calculate_costs: bool = False) -> List[SubPlan]:
    """
    Generate left-linear subplans of all sizes from 3 to the full query size.
    
    Args:
        query: Query object representing a join plan
        parent_plan_idx: Index of the parent plan this subplan belongs to
        calculate_costs: Whether to calculate real costs using root.get_cost()
        
    Returns:
        List of SubPlan objects representing subplans of each size
    """
    subplans = []
    
    # Get all triple patterns from the query
    def collect_triples_from_query(node: Triple | Join) -> List[Triple]:
        """Collect all Triple objects from the query tree"""
        triples = []
        if isinstance(node, Triple):
            triples.append(node)
        elif isinstance(node, Join):
            triples.extend(collect_triples_from_query(node.left))
            triples.extend(collect_triples_from_query(node.right))
        return triples
    
    all_triples = collect_triples_from_query(query.root)
    full_size = len(all_triples)
    
    # Generate subplans for each size from 3 to full_size-1 (excluding single triples and full query)
    for size in range(3, full_size):
        try:
            # Take the first 'size' triples to create a left-linear plan
            selected_triples = all_triples[:size]
            
            # Create left-linear join tree
            if len(selected_triples) == 1:
                subplan_root = selected_triples[0]
            else:
                # Build left-linear tree: ((T1 ⋈ T2) ⋈ T3) ⋈ ... ⋈ Tn
                subplan_root = selected_triples[0]
                for i in range(1, len(selected_triples)):
                    subplan_root = Join(left=subplan_root, right=selected_triples[i])
            
            # Calculate cost if requested
            cost = None
            if calculate_costs:
                try:
                    print(f"Cost for left-linear subplan of size {size} is", subplan_root.get_cost())
                    cost = subplan_root.get_cost()
                except Exception as e:
                    print(f"    Warning: Could not calculate cost for left-linear subplan of size {size}: {e}")
                    cost = None
            
            # Create SubPlan object
            subplan = SubPlan(
                root=subplan_root,
                triples_num=size,
                cost=cost,
                parent_plan_idx=parent_plan_idx
            )
            
            subplans.append(subplan)
            
        except Exception as e:
            print(f"    Error creating left-linear subplan of size {size}: {e}")
    
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
    
    # Create mapping from triple pattern to index
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}
    
    for i, plan in enumerate(join_plans):
        try:
            # Set cost to None instead of calculating
            cost = None
            costs.append(cost)
            
            # Create datapoint with consistent triple indices
            datapoint = join_order_to_adjacency_matrix_consistent(plan, triple_to_index, rdf2vec=rdf2vec_dict, counts=counts_dict)
            
            # Create torch_data directly from matrices with cost set to None
            torch_data = Data(
                x=torch.tensor(datapoint.embedding_matrix, dtype=torch.float),
                edge_index=torch.tensor(datapoint.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),
                y=torch.tensor([0.0], dtype=torch.float) if cost is None else torch.tensor([cost], dtype=torch.float)  # Use 0.0 when cost is None
            )
            torch_data_list.append(torch_data)
            
        except Exception as e:
            print(f"Error creating torch_data: {e}")
            costs.append(None)
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

def subplan_to_sparql_query(subplan: SubPlan, original_triples: List[List[str]], rdf2vec_dict, counts_dict) -> SPARQLQuery:
    """Convert a SubPlan to a SPARQLQuery object"""
    # Convert subplan to Query object
    query = subplan.to_query()
    
    # Get the triples that are part of this subplan
    subplan_triples = []
    
    def collect_triples(node):
        if isinstance(node, Triple):
            # Convert Triple back to list format using correct attributes (s, p, o)
            triple_list = [node.s.name, node.p.name, node.o.name, '.']
            subplan_triples.append(triple_list)
        elif isinstance(node, Join):
            collect_triples(node.left)
            collect_triples(node.right)
    
    collect_triples(query.root)
    
    # Create torch_data for the subplan
    try:
        datapoint = join_order_to_adjacency_matrix(query, rdf2vec=rdf2vec_dict, counts=counts_dict)
        # Use actual cost from subplan if available, otherwise default to 0.0
        cost_value = subplan.cost if subplan.cost is not None else 0.0
        torch_data = Data(
            x=torch.tensor(datapoint.embedding_matrix, dtype=torch.float),
            edge_index=torch.tensor(datapoint.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),
            y=torch.tensor([cost_value], dtype=torch.float)
        )
    except Exception as e:
        print(f"Error creating torch_data for subplan: {e}")
        torch_data = None
    
    return SPARQLQuery(
        triples=subplan_triples,
        join_plans=[query],
        costs=[subplan.cost],  # Use the actual cost from subplan
        torch_data=[torch_data] if torch_data else [None]
    )

def save_sparql_queries_single_file(sparql_queries, output_file):
    """Save all SPARQLQuery objects to a single pickle file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(sparql_queries, f)
    
    print(f"Saved {len(sparql_queries)} SPARQLQuery objects to {output_file}")

def visualize_and_save_plans(sparql_query: SPARQLQuery, query_idx: int, output_dir: str, include_subplans: bool = False, calculate_costs: bool = False):
    """
    Visualize and save each join plan and optionally subplans for a query
    
    Args:
        sparql_query: SPARQLQuery object with multiple join plans and subplans
        query_idx: Index of the query
        output_dir: Directory to save visualizations
        include_subplans: Whether to visualize subplans as well
        calculate_costs: Whether to calculate real costs for subplans
    """
    # Create directory for this query's plans
    query_dir = os.path.join(output_dir, f"query_{query_idx}")
    os.makedirs(query_dir, exist_ok=True)
    
    best_plan_idx = sparql_query.get_best_plan_index()
    
    # Visualize each full plan
    for i, plan in enumerate(sparql_query.join_plans):
        try:
            # Define output path - mark the best plan with "_best"
            cost_str = str(sparql_query.costs[i]) if sparql_query.costs[i] is not None else "None"
            plan_label = f"_best_cost_{cost_str}" if i == best_plan_idx else f"_cost_{cost_str}"
            output_path = os.path.join(query_dir, f"plan_{i}{plan_label}")
            
            # Visualize and save the plan
            plan.visualize(output_file=output_path, format="png")
            print(f"  Saved visualization for query {query_idx}, plan {i} to {output_path}.png")
            
            # Visualize subplans if requested
            if include_subplans and i < len(sparql_query.join_plans):
                # Create directory for this plan's subplans
                subplans_dir = os.path.join(query_dir, f"plan_{i}_subplans")
                os.makedirs(subplans_dir, exist_ok=True)
                
                for j, subplan in enumerate(extract_subplans_left_linear(plan, i, calculate_costs=calculate_costs)):
                    try:
                        subplan_query = subplan.to_query()
                        cost_str = str(subplan.cost) if subplan.cost is not None else "None"
                        subplan_output_path = os.path.join(subplans_dir, f"subplan_{j}_tp{subplan.triples_num}_cost_{cost_str}")
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
    input_file = "/home/tim/query_optimization/datasets/queries/LUBM_stars_10.json"
    sparql_queries_file = "sparql_star_query_10/queries.pkl"
    visualization_dir = "None"

    # Number of random plans to create per query
    num_random_plans = 1
    # Min Cardinality
    min_cardinality = 0
    n_triple_sizes = [10]  # Multiple triple sizes to process
    n_queries = 20  # Number of queries to collect for each size
    include_subplans = False  # Set to True to generate and include subplans
    calculate_subplan_costs = False  # Set to True to calculate real costs for subplans using root.get_cost()
    
    # Create visualization directory
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Load the queries
    print(f"Loading queries from {input_file}...")
    with open(input_file, "r") as f:
        queries = json.load(f)

    # Print how many queries of each size there are
    for n_triples in range(1, 10):
        queries_n_triples = [q for q in queries if len(q["triples"]) == n_triples]
        print(f"Found {len(queries_n_triples)} queries with {n_triples} triple patterns")
    
    # Collect queries for each specified size
    all_selected_queries = []
    
    for n_triples in n_triple_sizes:
        # Filter queries with exactly n_triples triple patterns and no all-variable patterns
        queries_filtered = [q for q in queries if len(q["triples"]) == n_triples and not has_all_variable_triple_pattern(q)]
        queries_filtered = [q for q in queries_filtered if q["y"] >= min_cardinality]
        print(f"Found {len(queries_filtered)} valid queries with exactly {n_triples} triple patterns and min cardinality {min_cardinality} (excluding queries with all-variable patterns)")

        # Sort queries by cardinality and take the first n_queries
        queries_filtered.sort(key=lambda x: x["y"])
        selected_queries = queries_filtered[:n_queries]
        
        print(f"Selected {len(selected_queries)} queries with {n_triples} triple patterns")
        
        # Add to the overall list with size information
        for query in selected_queries:
            query['n_triples'] = n_triples  # Add size information for later reference
            all_selected_queries.append(query)
    
    print(f"Total selected queries: {len(all_selected_queries)}")
    print(f"Subplan generation: {'ENABLED' if include_subplans else 'DISABLED'}")
    print(f"Subplan cost calculation: {'ENABLED' if calculate_subplan_costs else 'DISABLED'}")
    
    # Process queries
    all_sparql_queries = []
    
    for i, query in enumerate(tqdm(all_selected_queries, desc="Processing queries")):
        try:
            print(f"Processing query {i+1}/{len(all_selected_queries)} (size: {query['n_triples']})")
            
            # Create full query SPARQLQuery objects
            full_sparql_query = query_to_sparql_query(query, rdf2vec_dict, counts_dict, num_plans=num_random_plans)
            
            # Add full plans as individual SPARQLQuery objects
            for j, plan in enumerate(full_sparql_query.join_plans):
                try:
                    # Create individual SPARQLQuery for each full plan
                    individual_sparql_query = SPARQLQuery(
                        triples=query["triples"],
                        join_plans=[plan],
                        costs=[full_sparql_query.costs[j]],
                        torch_data=[full_sparql_query.torch_data[j]]
                    )
                    all_sparql_queries.append(individual_sparql_query)
                    print(f"  Added full plan {j} with cost {full_sparql_query.costs[j]}")
                    
                    # Only process subplans if include_subplans is True
                    if include_subplans:
                        # Extract subplans from this plan and create individual SPARQLQuery objects
                        subplans = extract_subplans_left_linear(plan, j, calculate_costs=calculate_subplan_costs)
                        print(f"  Extracted {len(subplans)} left-linear subplans from plan {j}")
                        
                        # Create SPARQLQuery objects for all subplans
                        for subplan in subplans:
                            try:
                                subplan_sparql_query = subplan_to_sparql_query(
                                    subplan, 
                                    query["triples"], 
                                    rdf2vec_dict, 
                                    counts_dict
                                )
                                all_sparql_queries.append(subplan_sparql_query)
                                print(f"    Added left-linear subplan of size {subplan.triples_num} with cost {subplan.cost}")
                            except Exception as e:
                                print(f"    Error creating SPARQLQuery for left-linear subplan of size {subplan.triples_num}: {e}")
                    else:
                        print(f"  Skipping subplan extraction (include_subplans=False)")
                
                except Exception as e:
                    print(f"  Error processing plan {j}: {e}")
            
            
            print(f"  Total SPARQLQuery objects created so far: {len(all_sparql_queries)}")
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
    
    # Save all SPARQLQuery objects to a single file
    save_sparql_queries_single_file(all_sparql_queries, sparql_queries_file)
    
    # Print summary statistics
    query_counts_by_size = {}
    subplan_counts_by_size = {}
    
    for sparql_query in all_sparql_queries:
        size = len(sparql_query.triples)
        
        # Check if this is a full query or subplan by looking at the original selected queries
        is_full_query = any(len(q["triples"]) == size for q in all_selected_queries)
        
        if is_full_query:
            query_counts_by_size[size] = query_counts_by_size.get(size, 0) + 1
        else:
            subplan_counts_by_size[size] = subplan_counts_by_size.get(size, 0) + 1
    
    print("\nDataset conversion complete!") 
    print(f"Total original queries processed: {len(all_selected_queries)}")
    print(f"Total SPARQLQuery objects created: {len(all_sparql_queries)}")
    print("Full plans by size:")
    for size, count in sorted(query_counts_by_size.items()):
        print(f"  - Size {size}: {count}")
    
    if include_subplans:
        print("Subplans by size:")
        for size, count in sorted(subplan_counts_by_size.items()):
            print(f"  - Size {size}: {count}")
    else:
        print("No subplans generated (include_subplans=False)") 