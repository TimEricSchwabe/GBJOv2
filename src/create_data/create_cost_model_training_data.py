import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pickle
import json
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from torch_geometric.data import Data, DataLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import Triple, Join, Query, Entity, join_order_to_adjacency_matrix, random, Datapoint, random_join_order
from tqdm import tqdm
import shutil
from data_loader import QueryDataset
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        if all(component.startswith('?') for component in triple[:3]):
            return True
    return False

@dataclass
class SPARQLQuery:
    """Class to hold multiple join plans for query"""
    triples: List[List[str]] # list of triples (each is [subject, predicate, object])
    join_plans: List[Query] # list of join plans (of type Query)
    costs: List[float] # list of costs for each plan
    torch_data: List[Data]  # torch_data for each plan
    triples_where: List[List[str]]  # list of the where clauses for each plan
    
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

def create_random_join_orders(triples: List[List[str]], count: int, rdf2vec_dict: dict, counts_dict: dict) -> List[Query]:
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
        seed = i + 1
        try:
            plan = random_join_order(triples, seed=seed)
            plans.append(plan)
        except Exception as e:
            print(f"Error creating random join order {i}: {e}")
    
    return plans


def shares_variables(node1, node2) -> bool:
    """
    Check if two nodes (Triple or Join) share at least one variable.
    Used to avoid Cartesian products in join plans (necessary for path queries on larger graphs).
    """
    return bool(node1.variables & node2.variables)


def beam_search_best_plan(triples: List[List[str]], beam_width: int = 1, 
                          avoid_cartesian: bool = False) -> List[Tuple[Query, float]]:
    """
    Build left-deep plans using beam search to find plans that minimize cost.
    Keeps top beam_width partial plans at each step.
    
    
    Args:
        triples: List of triple patterns (each is [subject, predicate, object]) (as strings)
        beam_width: Number of top plans to keep at each step
        avoid_cartesian: If True, skip joins where subtrees don't share variables
        
    Returns:
        List of (Query object, cost) tuples - up to beam_width best plans sorted by cost ascending.
    """
    triple_objs = [Triple(*(Entity(name=name) for name in t[:3])) for t in triples]
    n = len(triple_objs)
    
    if n == 1:
        # C_out cost for a single triple is 0 (leaves have no cost) ( but there should be no queries with only one triple)
        try:
            # Still need to verify the triple is valid
            triple_objs[0].get_cardinality()
            return [(Query(root=triple_objs[0], triples_num=1), 0)]
        except Exception as e:
            print(f"Error getting cardinality for single triple: {e}")
            return []
    
    # Initialize beam with single triples
    # Each beam entry: (cardinality_for_sorting, plan, used_indices_frozenset)
    # We use cardinality to select the best starting triple, but cost starts at 0 (C_out def)
    beam = []
    for i in range(n):
        try:
            cardinality = triple_objs[i].get_cardinality()
            beam.append((cardinality, triple_objs[i], frozenset({i})))
        except Exception as e:
            print(f"Error getting cardinality for triple {i}, skipping: {e}")
            continue
    
    if not beam:
        print("All triples failed cardinality check, cannot build any plan")
        return []
    
    # Sort by cardinality (ascending) to pick best starting triples, keep top beam_width
    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_width]
    
    # Reset cost to 0 for selected triples (C_out: leaves have cost 0)
    beam = [(0, plan, used) for (_, plan, used) in beam]
    
    # Expand beam n-1 times (add one triple at each step)
    for step in range(n - 1):
        candidates = []
        
        for current_cost, current_plan, used in beam:
            remaining = set(range(n)) - used
            
            for idx in remaining:
                # Skip Cartesian products if requested
                if avoid_cartesian and not shares_variables(current_plan, triple_objs[idx]):
                    continue
                
                new_plan = Join(left=current_plan, right=triple_objs[idx])
                try:
                    # Get cardinality of current join and calculate C_out cost
                    new_cardinality = new_plan.get_cardinality()
                    new_cost = new_cardinality + current_cost
                    candidates.append((new_cost, new_plan, used | {idx}))
                except Exception as e:
                    print(f"Error in beam search step {step}, skipping candidate: {e}")
                    continue
        
        if not candidates:
            if avoid_cartesian:
                print(f"No valid candidates at step {step} (possibly only Cartesian products remain)")
            else:
                print(f"No valid candidates at step {step}, returning partial results")
            break
            
        # Sort by cost and keep top beam_width
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
    
    # Return only complete plans (those that include all n triples)
    complete_plans = [
        (Query(root=plan, triples_num=n), cost) 
        for cost, plan, used in beam 
        if len(used) == n
    ]
    
    return complete_plans


def beam_search_worst_plan(triples: List[List[str]], beam_width: int = 1,
                           avoid_cartesian: bool = False) -> List[Tuple[Query, float]]:
    """
    Build left-deep plans using beam search to find plans that maxiize cost.
    
    Args:
        triples: List of triple patterns (each is [subject, predicate, object])
        beam_width: Number of top plans to keep at each step (1 = greedy)
        avoid_cartesian: If True, skip joins where subtrees don't share variables
        
    Returns:
        List of (Query object, cost) tuples - up to beam_width worst plans sorted by cost descending.
        Returns empty list if no complete plans could be built.
    """
    triple_objs = [Triple(*(Entity(name=name) for name in t[:3])) for t in triples]
    n = len(triple_objs)
    
    if n == 1:
        try:
            triple_objs[0].get_cardinality()
            return [(Query(root=triple_objs[0], triples_num=1), 0)]
        except Exception as e:
            print(f"Error getting cardinality for single triple: {e}")
            return []
    

    beam = []
    for i in range(n):
        try:
            cardinality = triple_objs[i].get_cardinality()
            beam.append((cardinality, triple_objs[i], frozenset({i})))
        except Exception as e:
            print(f"Error getting cardinality for triple {i}, skipping: {e}")
            continue
    
    if not beam:
        print("All triples failed cardinality check, cannot build any plan")
        return []
    
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:beam_width]
    
    beam = [(0, plan, used) for (_, plan, used) in beam]
    
    # Expand beam n-1 times (add one triple at each step)
    for step in range(n - 1):
        candidates = []
        
        for current_cost, current_plan, used in beam:
            remaining = set(range(n)) - used
            
            for idx in remaining:
                if avoid_cartesian and not shares_variables(current_plan, triple_objs[idx]):
                    continue
                
                new_plan = Join(left=current_plan, right=triple_objs[idx])
                try:
                    new_cardinality = new_plan.get_cardinality()
                    new_cost = new_cardinality + current_cost
                    candidates.append((new_cost, new_plan, used | {idx}))
                except Exception as e:
                    print(f"Error in beam search step {step}, skipping candidate: {e}")
                    continue
        
        if not candidates:
            if avoid_cartesian:
                print(f"No valid candidates at step {step} (possibly only Cartesian products remain)")
            else:
                print(f"No valid candidates at step {step}, returning partial results")
            break
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
    
    complete_plans = [
        (Query(root=plan, triples_num=n), cost) 
        for cost, plan, used in beam 
        if len(used) == n
    ]
    
    return complete_plans


def create_diverse_join_orders(triples: List[List[str]], num_random: int = 3, 
                                beam_width: int = 1, include_worst_plans: bool = True,
                                avoid_cartesian: bool = False) -> List[Tuple[Query, Optional[float]]]:
    """
    Create a diverse set of join orders including:
    - beam_width beam-search-best plans (minimize real execution cost)
    - beam_width beam-search-worst plans (maximize real execution cost) - Optional
    - num_random random plans
        
    Args:
        triples: List of triple patterns
        num_random: Number of random plans to generate
        beam_width: Beam width for search (returns this many best and worst plans)
        include_worst_plans: Whether to include worst plans (default: True)
        avoid_cartesian: If True, skip joins where subtrees don't share variables
        
    Returns:
        List of (Query, cost_or_None) tuples.
        Returns None only if no plans could be generated at all.
    """
    plans = []
    
    # 1. Beam search best plans
    try:
        best_plans = beam_search_best_plan(triples, beam_width=beam_width, avoid_cartesian=avoid_cartesian)
        if best_plans:
            plans.extend(best_plans)
            print(f"Generated {len(best_plans)} beam-search-best plans")
        else:
            print("Warning: beam_search_best returned no complete plans")
    except Exception as e:
        print(f"Error creating beam-search-best plans: {e}")
    
    # 2. Beam search worst plans
    if include_worst_plans:
        try:
            worst_plans = beam_search_worst_plan(triples, beam_width=beam_width, avoid_cartesian=avoid_cartesian)
            if worst_plans:
                plans.extend(worst_plans)
                print(f"Generated {len(worst_plans)} beam-search-worst plans")
            else:
                print("Warning: beam_search_worst returned no complete plans")
        except Exception as e:
            print(f"Error creating beam-search-worst plans: {e}")
    
    # 3. Random plans
    random_count = 0
    for i in range(num_random):
        try:
            plan = random_join_order(triples, seed=i + 42, avoid_cartesian=avoid_cartesian)
            plans.append((plan, None))
            random_count += 1
        except Exception as e:
            print(f"Error creating random plan {i}: {e}")
    
    if random_count > 0:
        print(f"Generated {random_count} random plans")
    
    if not plans:
        print("Warning: Could not generate any plans for this query")
        return None
    
    return plans


def generate_invalid_plan(valid_data: Data) -> Data:
    """
    Generate an invalid plan by modifying valid data:
    1. Randomly remove between 0 and num_edges/2 edges
    2. Randomly add between 0 and num_edges/2 edges
    3. Ensure at least one modification makes it invalid (if 0 removed, add at least 1)
    4. Added edges must NOT be present in the original valid data
    5. Set cost to infinity
    
    Args:
        valid_data: valid Data object to base the invalid one on
        
    Returns:
        Modified Data object representing an invalid plan
    """
    invalid_data = valid_data.clone()
    
    num_edges = invalid_data.edge_index.size(1)
    
    max_changes = max(1, num_edges // 2)
    
    # 1. Remove edges
    # Randomly remove between 0 and num_edges/2 edges
    n_remove = np.random.randint(0, max_changes + 1)
    
    if n_remove > 0 and num_edges > 0:
        # Create a mask of all true
        mask = torch.ones(num_edges, dtype=torch.bool)
        
        # Select n_remove unique indices to remove
        if n_remove >= num_edges:
            mask[:] = False
        else:
            remove_indices = np.random.choice(num_edges, size=n_remove, replace=False)
            mask[remove_indices] = False
            
        invalid_data.edge_index = invalid_data.edge_index[:, mask]
    
    # 2. Add edges
    # Randomly add between 0 and num_edges/2 edges
    n_add = np.random.randint(0, max_changes + 1)
    
    # Constraint: if no edge has been removed, add at least one invalid edge
    if n_remove == 0 and n_add == 0:
        n_add = 1
        
    num_nodes = invalid_data.x.size(0)
    
    if n_add > 0 and num_nodes > 1:
        # Get existing edges from the valid data to ensure we add truly new edges
        existing_edges = set()
        edge_index_np = valid_data.edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            existing_edges.add((edge_index_np[0, i], edge_index_np[1, i]))
            
        new_edges_list = []
        attempts = 0
        max_attempts = n_add * 20  
        
        while len(new_edges_list) < n_add and attempts < max_attempts:
            attempts += 1
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            
            if (u, v) not in existing_edges:
                new_edges_list.append([u, v])
                existing_edges.add((u, v))
        
        if new_edges_list:
            new_edges_tensor = torch.tensor(new_edges_list, dtype=torch.long).t()
            
            invalid_data.edge_index = torch.cat([invalid_data.edge_index, new_edges_tensor], dim=1)
    
    # 3. Set cost to infinity (invalid plan)
    invalid_data.y = torch.tensor([float('inf')], dtype=torch.float)
    
    return invalid_data


def query_to_sparql_query(query_data: dict, rdf2vec_dict, counts_dict, num_plans: int = 10, 
                          use_diverse_plans: bool = False, num_random_plans: int = 3,
                          beam_width: int = 1, include_worst_plans: bool = True,
                          include_invalid_plan: bool = True,
                          avoid_cartesian: bool = False) -> SPARQLQuery:
    """
    Convert a raw query to a SPARQLQuery with multiple join plans and costs.
    
    Args:
        query_data: Dictionary containing query data with "triples" key
        rdf2vec_dict: Dictionary of RDF2Vec embeddings
        counts_dict: Dictionary of entity counts
        num_plans: Number of random plans (used when use_diverse_plans=False)
        use_diverse_plans: If True, use beam-search-best, beam-search-worst optionally, and random plans
        num_random_plans: Number of random plans when using diverse mode
        beam_width: Beam width for search
        include_worst_plans: Whether to include worst plans
        include_invalid_plan: Whether to include an invalid plan
        avoid_cartesian: If True, skip joins where subtrees don't share variables
        
    Returns:
        SPARQLQuery object with multiple join plans
    """
    triples = query_data["triples"]
    
    if use_diverse_plans:
        plans_with_costs = create_diverse_join_orders(triples, num_random=num_random_plans, 
                                                       beam_width=beam_width,
                                                       include_worst_plans=include_worst_plans,
                                                       avoid_cartesian=avoid_cartesian)
        if plans_with_costs is None:
            return None
    else:

        raw_plans = create_random_join_orders(triples, num_plans, rdf2vec_dict, counts_dict)
        plans_with_costs = [(plan, None) for plan in raw_plans]
    
    # Calculate cost for each plan and create torch_data
    costs = []
    torch_data_list = []
    triples_where_list = []
    final_join_plans = []  # Store just the Query objects for SPARQLQuery
    
    # Create mapping from triple pattern to index
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}

    plans_processed = 0
    plans_failed = 0

    for plan, precomputed_cost in plans_with_costs:
        try:
            # Use pre-computed cost if available, otherwise calculate
            if precomputed_cost is not None:
                cost = precomputed_cost
            else:
                try:
                    cost = plan.root.get_cost() 
                except RuntimeError as e:
                    # Skip this plan - don't add it to the final data
                    print(f"Error calculating cost for plan, skipping: {e}")
                    plans_failed += 1
                    continue
            
            # Create torch_data with consistent triple indices
            datapoint = join_order_to_adjacency_matrix_consistent(plan, triple_to_index, rdf2vec=rdf2vec_dict, counts=counts_dict)
            data = datapoint.get_torch_data(cost=cost)
            
            triples_where = [triple.where_body() for triple in datapoint.nodes_order if isinstance(triple, Triple)]
            
            costs.append(cost)
            final_join_plans.append(plan)
            torch_data_list.append(data)
            triples_where_list.append(triples_where)
            plans_processed += 1
            
        except Exception as e:
            print(f"Error processing plan, skipping: {e}")
            plans_failed += 1
            continue
    
    if plans_failed > 0:
        print(f"Warning: {plans_failed} plans failed, {plans_processed} plans succeeded")
    
    if include_invalid_plan and torch_data_list:
        try:
            valid_idx = next((i for i, d in enumerate(torch_data_list) if d is not None), -1)
            
            if valid_idx != -1:
                valid_data = torch_data_list[valid_idx]
                invalid_data = generate_invalid_plan(valid_data)
                
                final_join_plans.append(None)  
                costs.append(float('inf'))
                torch_data_list.append(invalid_data)
                triples_where_list.append(triples_where_list[valid_idx])
        except Exception as e:
            print(f"Error generating invalid plan: {e}")

    if not final_join_plans:
        print("No valid plans could be generated for this query")
        return None
    
    return SPARQLQuery(
        triples=triples, 
        join_plans=final_join_plans, 
        costs=costs,
        torch_data=torch_data_list,
        triples_where=triples_where_list
    )

def join_order_to_adjacency_matrix_consistent(join_order: Query, triple_to_index: dict, seed = None, rdf2vec=None, counts=None) -> Datapoint:
    """
    Convert a join order to an adjacency matrix and embeddings.
    Args:
        join_order: Query object representing the join order
        triple_to_index: Dictionary mapping triple string representations to indices
        seed: Random seed for variable indexing
        rdf2vec: RDF2Vec embeddings
        counts: Entity counts
        
    Returns:
        Datapoint object with adjacency matrix and embeddings
    """
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
        if plan is not None and sparql_query.torch_data[i] is not None:
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    torch.save(sparql_queries, output_file)
    
    print(f"Saved {len(sparql_queries)} SPARQLQuery objects to {output_file}")


def save_sparql_queries_human_readable(sparql_queries, output_file, use_diverse_plans=True, 
                                       beam_width=1, include_worst_plans=True):
    """
    Save SPARQLQuery objects to a human-readable JSON file.
    
    Each query is saved with:
    - triples: Original triple patterns
    - plans: List of join plan structures with their costs
    - best_plan_index: Index of the plan with lowest cost (actual best)
    
    When use_diverse_plans=True, plans are labeled based on beam_width:
    - Indices [0:beam_width): beam_search_best plans (ranked 1st, 2nd, etc.)
    - Indices [beam_width:2*beam_width): beam_search_worst plans (ranked 1st, 2nd, etc.) (if included)
    - Remaining: random plans
    
    Args:
        sparql_queries: List of SPARQLQuery objects
        output_file: Path to save the JSON file
        use_diverse_plans: If True, label plans with beam search info
        beam_width: Beam width used for search (to properly label plans)
        include_worst_plans: Whether worst plans were included
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    def get_plan_type(plan_idx, num_plans):
        """Get the type label for a plan based on its index."""
        if not use_diverse_plans:
            return "random"
        
        if include_worst_plans:            
            if plan_idx < beam_width:
                return f"beam_search_best_rank_{plan_idx + 1}"
            elif plan_idx < 2 * beam_width:
                return f"beam_search_worst_rank_{plan_idx - beam_width + 1}"
            else:
                return "random"
        else:
            if plan_idx < beam_width:
                return f"beam_search_best_rank_{plan_idx + 1}"
            else:
                return "random"
    
    def to_python_type(val):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    queries_data = []
    for i, sq in enumerate(sparql_queries):
        actual_best_idx = int(sq.get_best_plan_index())
        
        query_data = {
            "query_index": i,
            "triples": sq.triples,  # Original triple patterns
            "num_plans": len(sq.join_plans),
            "actual_best_plan_index": actual_best_idx,
            "actual_best_cost": to_python_type(sq.get_best_cost()),
            "plans": []
        }
        
        for j, (plan, cost) in enumerate(zip(sq.join_plans, sq.costs)):
            if plan is None:
                 plan_data = {
                    "plan_index": j,
                    "plan_type": "invalid",
                    "cost": to_python_type(cost),
                    "is_actual_best": False,
                    "join_tree": "INVALID",
                    "where_clause": "INVALID"
                }
                 query_data["plans"].append(plan_data)
                 continue

            plan_type = get_plan_type(j, len(sq.join_plans))
            plan_data = {
                "plan_index": j,
                "plan_type": plan_type,
                "cost": to_python_type(cost),
                "is_actual_best": j == actual_best_idx,
                "join_tree": plan.root.json(),  # Nested structure showing join order
                "where_clause": plan.root.where_body()  # Full WHERE body
            }
            query_data["plans"].append(plan_data)
        
        queries_data.append(query_data)
    
    with open(output_file, 'w') as f:
        json.dump(queries_data, f, indent=2)
    
    print(f"Saved {len(sparql_queries)} queries in human-readable format to {output_file}")

def save_dataset_single_file(triples, torch_dataset, output_dir):
    """
    Save dataset to a single file
    
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
        if plan is None:
            continue
        try:
            # Define output path - mark the best plan with "_best"
            plan_label = f"_best_cost_{sparql_query.costs[i]:.0f}" if i == best_plan_idx else f"_cost_{sparql_query.costs[i]:.0f}"
            output_path = os.path.join(query_dir, f"plan_{i}{plan_label}")
            
            # Visualize and save the plan
            plan.visualize(output_file=output_path, format="png")
            print(f"  Saved visualization for query {query_idx}, plan {i} to {output_path}.png")
        except Exception as e:
            print(f"  Error visualizing plan {i} for query {query_idx}: {e}")



if __name__ == "__main__":
    # Load the RDF2Vec embeddings
    with open(".../rdf2vec100dim.pkl", "rb") as f:
        rdf2vec_dict = pickle.load(f)
        print(len(rdf2vec_dict))


    # Load the counts
    with open(".../counts.pkl", "rb") as f:
        counts_dict = pickle.load(f)

    
    # Queries to generate random plans for
    input_file = ".../queries.json"

    # Directory to save the plans
    dataset_dir = ".../greedy"

    #visualization_dir = "join_plan_visualizations_path_wikidata"
    sparql_queries_file = ".../queries.pt"


    # How many queries to process
    MAX_QUERIES = 100000

    # The minimum cardinality of the queries to process
    MIN_CARDINALITY = 1
    #N_TRIPLES = 5
    SAVE_INTERVAL = 1000

    # Plan generation configuration
    USE_DIVERSE_PLANS = True  # If True: generate beam-search-best, beam-search-worst, and random plans
                               # If False: generate only random plans (original behavior)
    
    INCLUDE_WORST_PLANS = True # If True: include beam-search-worst plans in diverse generation
    INCLUDE_INVALID_PLAN = False # If True: include one invalid plan per query (randomly mutated)
    
    # If True: only create joins where subtrees share variables (avoids Cartesian products)
    # This is useful for path queries where random plans can create huge cross products
    AVOID_CARTESIAN_PRODUCTS = True

    # Beam width for beam search (1 = greedy, higher = more exploration)
    BEAM_WIDTH = 3
    
    # Number of random plans to create per query
    # When USE_DIVERSE_PLANS=True: total plans = beam_width (best) + [beam_width (worst)] + NUM_RANDOM_PLANS
    # When USE_DIVERSE_PLANS=False: total plans = NUM_RANDOM_PLANS
    NUM_RANDOM_PLANS = 3
    
    # Number of parallel workers for query processing
    NUM_WORKERS = 4
    
    # Create visualization directory
    #os.makedirs(visualization_dir, exist_ok=True)
    
    # Load the queries
    print(f"Loading queries from {input_file}...")
    with open(input_file, "r") as f:
        queries = json.load(f)
    
    # Filter queries with exactly 8 triple patterns
    #queries_8tp = [q for q in queries if len(q["triples"]) == N_TRIPLES]
    # Filter queries for min cardinality and all-variable triple patterns
    queries = [q for q in queries if q["y"] >= MIN_CARDINALITY and not has_all_variable_triple_pattern(q)]
    #Shuffle queries
    random.shuffle(queries)

    
    
    ############ Process queries ############
    
    def process_single_query(query_data: dict) -> Optional[SPARQLQuery]:
        """Worker function to process a single query in parallel."""
        try:
            sparql_query = query_to_sparql_query(
                query_data, rdf2vec_dict, counts_dict, 
                num_plans=NUM_RANDOM_PLANS,
                use_diverse_plans=USE_DIVERSE_PLANS,
                num_random_plans=NUM_RANDOM_PLANS,
                beam_width=BEAM_WIDTH,
                include_worst_plans=INCLUDE_WORST_PLANS,
                include_invalid_plan=INCLUDE_INVALID_PLAN,
                avoid_cartesian=AVOID_CARTESIAN_PRODUCTS
            )
            if sparql_query is None:
                return None

            # check if all costs are the same (no diversity in plans)
            if all(cost == sparql_query.costs[0] for cost in sparql_query.costs):
                return None

            return sparql_query
        except Exception as e:
            print(f"Error processing query: {e}")
            return None
    
    sparql_queries = []
    all_triples = []
    all_torch_data = []
    n_queries = 0
    
    # Process queries in parallel
    queries_to_process = queries[:MAX_QUERIES]
    
    print(f"Processing {len(queries_to_process)} queries with {NUM_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all queries
        futures = {executor.submit(process_single_query, q): i for i, q in enumerate(queries_to_process)}
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
            query_idx = futures[future]
            try:
                sparql_query = future.result()
                
                if sparql_query is None:
                    continue
                
                sparql_queries.append(sparql_query)
                n_queries += 1
                
                # Add datapoints for each plan
                for j, plan in enumerate(sparql_query.join_plans):
                    if sparql_query.torch_data[j] is not None:
                        all_triples.append(sparql_query.triples_where[j])
                        all_torch_data.append(sparql_query.torch_data[j])
                
                print(f"  Query {query_idx} costs: {sparql_query.costs}")

                if (n_queries % SAVE_INTERVAL) == 0:
                    print(f"\nSaving checkpoint at {n_queries} queries...")
                    save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
                    save_sparql_queries_single_file(sparql_queries, sparql_queries_file)

                    print(f"Checkpoint saved at {n_queries} queries")

            except Exception as e:
                print(f"Error processing query {query_idx}: {e}")
    
    # Save final results
    print("\nSaving final results...")
    save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
    save_sparql_queries_single_file(sparql_queries, sparql_queries_file)
    
    # Save human-readable version
    human_readable_file = sparql_queries_file.replace('.pt', '_readable.json')
    save_sparql_queries_human_readable(sparql_queries, human_readable_file, use_diverse_plans=USE_DIVERSE_PLANS, 
                                       beam_width=BEAM_WIDTH, include_worst_plans=INCLUDE_WORST_PLANS)

    print("\nDataset creation complete!")