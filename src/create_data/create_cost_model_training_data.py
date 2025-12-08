import pickle
import os
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


def beam_search_best_plan(triples: List[List[str]], beam_width: int = 1) -> Tuple[Query, float]:
    """
    Build a left-deep plan using beam search to find plans that minimize cost.
    Keeps top beam_width partial plans at each step.
    
    Args:
        triples: List of triple patterns (each is [subject, predicate, object])
        beam_width: Number of top plans to keep at each step (1 = greedy)
        
    Returns:
        Tuple of (Query object, final_cost) - the best plan and its cost
    """
    triple_objs = [Triple(*(Entity(name=name) for name in t[:3])) for t in triples]
    n = len(triple_objs)
    
    if n == 1:
        # C_out cost for a single triple is 0 (leaves have no cost)
        return Query(root=triple_objs[0], triples_num=1), 0
    
    # Initialize beam with single triples
    # Each beam entry: (cardinality_for_sorting, plan, used_indices_frozenset)
    # We use cardinality to select the best starting triple, but cost starts at 0
    beam = []
    for i in range(n):
        try:
            cardinality = triple_objs[i].get_cardinality()
            beam.append((cardinality, triple_objs[i], frozenset({i})))
        except Exception:
            continue
    
    if not beam:
        # Fallback if all failed
        beam = [(float('inf'), triple_objs[0], frozenset({0}))]
    
    # Sort by cardinality (ascending) to pick best starting triples, keep top beam_width
    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_width]
    
    # Reset cost to 0 for selected triples (C_out: leaves have cost 0)
    beam = [(0, plan, used) for (_, plan, used) in beam]
    
    # Expand beam n-1 times (add one triple at each step)
    for _ in range(n - 1):
        candidates = []
        
        for current_cost, current_plan, used in beam:
            remaining = set(range(n)) - used
            
            for idx in remaining:
                new_plan = Join(left=current_plan, right=triple_objs[idx])
                try:
                    # Incremental cost: only query this join's cardinality, reuse current_cost
                    new_cardinality = new_plan.get_cardinality()
                    new_cost = new_cardinality + current_cost
                    candidates.append((new_cost, new_plan, used | {idx}))
                except Exception:
                    raise
                    continue
        
        if not candidates:
            break
            
        # Sort by cost and keep top beam_width
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
    
    # Return the best complete plan
    best_cost, best_plan, _ = beam[0]
    return Query(root=best_plan, triples_num=n), best_cost


def beam_search_worst_plan(triples: List[List[str]], beam_width: int = 1) -> Tuple[Query, float]:
    """
    Build a left-deep plan using beam search to find plans that MAXIMIZE cost.
    Keeps top beam_width partial plans (by highest cost) at each step.
    
    Args:
        triples: List of triple patterns (each is [subject, predicate, object])
        beam_width: Number of top plans to keep at each step (1 = greedy)
        
    Returns:
        Tuple of (Query object, final_cost) - the worst plan and its cost
    """
    triple_objs = [Triple(*(Entity(name=name) for name in t[:3])) for t in triples]
    n = len(triple_objs)
    
    if n == 1:
        # C_out cost for a single triple is 0 (leaves have no cost)
        return Query(root=triple_objs[0], triples_num=1), 0
    
    # Initialize beam with single triples
    # Each beam entry: (cardinality_for_sorting, plan, used_indices_frozenset)
    # We use cardinality to select the worst starting triple, but cost starts at 0
    beam = []
    for i in range(n):
        try:
            cardinality = triple_objs[i].get_cardinality()
            beam.append((cardinality, triple_objs[i], frozenset({i})))
        except Exception:
            continue
    
    if not beam:
        # Fallback if all failed
        beam = [(0, triple_objs[0], frozenset({0}))]
    
    # Sort by cardinality (descending) to pick worst starting triples, keep top beam_width
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:beam_width]
    
    # Reset cost to 0 for selected triples (C_out: leaves have cost 0)
    beam = [(0, plan, used) for (_, plan, used) in beam]
    
    # Expand beam n-1 times (add one triple at each step)
    for _ in range(n - 1):
        candidates = []
        
        for current_cost, current_plan, used in beam:
            remaining = set(range(n)) - used
            
            for idx in remaining:
                new_plan = Join(left=current_plan, right=triple_objs[idx])
                try:
                    # Incremental cost: only query this join's cardinality, reuse current_cost
                    new_cardinality = new_plan.get_cardinality()
                    new_cost = new_cardinality + current_cost
                    candidates.append((new_cost, new_plan, used | {idx}))
                except Exception:
                    continue
        
        if not candidates:
            break
            
        # Sort by cost (descending) and keep top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]
    
    # Return the worst (highest cost) complete plan
    worst_cost, worst_plan, _ = beam[0]
    return Query(root=worst_plan, triples_num=n), worst_cost


def create_diverse_join_orders(triples: List[List[str]], num_random: int = 3, 
                                beam_width: int = 1) -> List[Tuple[Query, Optional[float]]]:
    """
    Create a diverse set of join orders including:
    - 1 beam-search-best plan (minimizes real execution cost)
    - 1 beam-search-worst plan (maximizes real execution cost)
    - num_random random plans (for coverage of middle ground)
    
    Args:
        triples: List of triple patterns
        num_random: Number of random plans to generate
        beam_width: Beam width for search (1 = greedy, higher = more exploration)
        
    Returns:
        List of (Query, cost_or_None) tuples. Beam search plans include pre-computed costs,
        random plans have None (cost calculated later).
    """
    plans = []
    
    # 1. Beam search best plan (comes with pre-computed cost)
    try:
        best_plan, best_cost = beam_search_best_plan(triples, beam_width=beam_width)
        plans.append((best_plan, best_cost))
    except Exception as e:
        print(f"Error creating beam-search-best plan: {e}")
    
    # 2. Beam search worst plan (comes with pre-computed cost)
    try:
        worst_plan, worst_cost = beam_search_worst_plan(triples, beam_width=beam_width)
        plans.append((worst_plan, worst_cost))
    except Exception as e:
        print(f"Error creating beam-search-worst plan: {e}")
    
    # 3. Random plans (cost will be calculated later)
    for i in range(num_random):
        try:
            plan = random_join_order(triples, seed=i + 42)
            plans.append((plan, None))
        except Exception as e:
            print(f"Error creating random plan {i}: {e}")
    
    return plans

def query_to_sparql_query(query_data: dict, rdf2vec_dict, counts_dict, num_plans: int = 10, 
                          use_diverse_plans: bool = False, num_random_plans: int = 3,
                          beam_width: int = 1) -> SPARQLQuery:
    """
    Convert a raw query to a SPARQLQuery with multiple join plans and costs.
    
    Args:
        query_data: Dictionary containing query data with "triples" key
        rdf2vec_dict: Dictionary of RDF2Vec embeddings
        counts_dict: Dictionary of entity counts
        num_plans: Number of random plans (used when use_diverse_plans=False)
        use_diverse_plans: If True, use beam-search-best, beam-search-worst, and random plans
        num_random_plans: Number of random plans when using diverse mode
        beam_width: Beam width for search (1 = greedy, higher = more exploration)
        
    Returns:
        SPARQLQuery object with multiple join plans
    """
    triples = query_data["triples"]
    
    if use_diverse_plans:
        # Use diverse plan generation: beam-search-best, beam-search-worst, and random plans
        # Returns List[Tuple[Query, Optional[float]]] - plans with pre-computed costs where available
        plans_with_costs = create_diverse_join_orders(triples, num_random=num_random_plans, 
                                                       beam_width=beam_width)
    else:
        # Use purely random plans (original behavior)
        # Wrap in tuples with None cost for uniform handling
        raw_plans = create_random_join_orders(triples, num_plans, rdf2vec_dict, counts_dict)
        plans_with_costs = [(plan, None) for plan in raw_plans]
    
    # Calculate cost for each plan and create torch_data
    costs = []
    torch_data_list = []
    triples_where_list = []
    final_join_plans = []  # Store just the Query objects for SPARQLQuery
    
    # Create mapping from triple pattern to index (create once and reuse)
    triple_objs = [Triple(*(Entity(name=name) for name in triple[:3])) for triple in triples]
    triple_to_index = {str(triple): i for i, triple in enumerate(triple_objs)}


    cost_error = False

    for plan, precomputed_cost in plans_with_costs:
        try:
            # Use pre-computed cost if available, otherwise calculate
            if precomputed_cost is not None:
                cost = precomputed_cost
            else:
                try:
                    cost = plan.root.get_cost() 
                except RuntimeError as e:
                    print(f"Error calculating cost: {e}")
                    cost_error = True
                    break
            
            costs.append(cost)
            final_join_plans.append(plan)
            
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
        join_plans=final_join_plans, 
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



if __name__ == "__main__":
    # Load the RDF2Vec embeddings
    with open("datasets/graphs/lubm/rdf2vec100dim.pkl", "rb") as f:
        rdf2vec_dict = pickle.load(f)
        print(len(rdf2vec_dict))


    # Load the counts
    with open("datasets/graphs/lubm/counts.pkl", "rb") as f:
        counts_dict = pickle.load(f)

    
    # Queries to generate random plans for
    input_file = "datasets/queries/lubm/stars/star_queries.json"

    # Directory to save the plans
    dataset_dir = "datasets/plans/lubm/star/plans/new_dataset"

    #visualization_dir = "join_plan_visualizations_path_wikidata"
    sparql_queries_file = "sparql_queries_star_lubm/queries.pkl"


    # How many queries to process
    MAX_QUERIES = 20

    # The minimum cardinality of the queries to process
    MIN_CARDINALITY = 1
    #N_TRIPLES = 5
    SAVE_INTERVAL = 1000

    # Plan generation configuration
    USE_DIVERSE_PLANS = True  # If True: generate beam-search-best, beam-search-worst, and random plans
                               # If False: generate only random plans (original behavior)
    
    # Beam width for beam search (1 = greedy, higher = more exploration)
    # Complexity: O(n^2 * beam_width) per query
    BEAM_WIDTH = 2
    
    # Number of random plans to create per query
    # When USE_DIVERSE_PLANS=True: total plans = 2 (beam search) + NUM_RANDOM_PLANS
    # When USE_DIVERSE_PLANS=False: total plans = NUM_RANDOM_PLANS
    NUM_RANDOM_PLANS = 3
    
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
    sparql_queries = []
    all_triples = []
    all_torch_data = []

    n_queries = 0
    
    for i, query in enumerate(tqdm(queries[:MAX_QUERIES], desc="Processing queries")):

        try:
            sparql_query = query_to_sparql_query(
                query, rdf2vec_dict, counts_dict, 
                num_plans=NUM_RANDOM_PLANS,
                use_diverse_plans=USE_DIVERSE_PLANS,
                num_random_plans=NUM_RANDOM_PLANS,
                beam_width=BEAM_WIDTH
            )
            if sparql_query is None:
                continue

            # check if all costs are the same
            if all(cost == sparql_query.costs[0] for cost in sparql_query.costs):
                continue

            sparql_queries.append(sparql_query)
            n_queries += 1
            if n_queries > MAX_QUERIES:
                break
            
            # Visualize and save all plans for this query
            #visualize_and_save_plans(sparql_query, i, visualization_dir)
            
            # Add datapoints for each plan (using pre-created data from query_to_sparql_query)
            for j, plan in enumerate(sparql_query.join_plans):
                if sparql_query.torch_data[j] is not None:
                    all_triples.append(sparql_query.triples_where[j])
                    all_torch_data.append(sparql_query.torch_data[j])
            
            # Print costs for debugging
            print(f"  Plans costs: {sparql_query.costs}")

            # Save every SAVE_INTERVAL queries
            if (n_queries % SAVE_INTERVAL) == 0:
                print(f"\nSaving checkpoint at {n_queries} queries...")
                
                # Save SPARQLQuery objects checkpoint
                #save_sparql_queries_single_file(sparql_queries, sparql_queries_file)
                
                # Save dataset checkpoint
                save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
                
                print(f"Checkpoint saved at {n_queries} queries")

        except Exception as e:
            #raise
            print(f"Error processing query {i}: {e}")
    
    # Save final results
    print("\nSaving final results...")
    save_dataset_single_file(all_triples, all_torch_data, dataset_dir)
    #save_sparql_queries_single_file(sparql_queries, sparql_queries_file)

    print("\nDataset creation complete!")