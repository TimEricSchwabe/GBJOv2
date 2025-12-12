import torch
import numpy as np
import itertools
import pickle
from data import Triple, Join, Query, Entity
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', '..'))
from src.create_data.create_cost_model_training_data import SPARQLQuery
import random

# Add module compatibility for old pickle files
import src.data as data_module
sys.modules['explicit_join_model.data'] = data_module
sys.modules['explicit_join_model'] = sys.modules['src']


def left_deep_adj_from_perm(pi):
    """
    Create adjacency matrix for a left-deep join tree from a permutation.
    
    Args:
        pi: Tensor of length n with the (0-based) permutation of triple nodes.
        
    Returns:
        A: (2n-1, 2n-1) adjacency matrix for a left-deep tree:
           (((T_pi0 ▷◁ T_pi1) ▷◁ T_pi2) … )
    """
    n = len(pi)
    N = 2 * n - 1
    A = torch.zeros(N, N, dtype=torch.float32)
    # indices: triple 0..n-1, join nodes n..2n-2 (root = 2n-2)
    # first join joins pi0 and pi1 -> node idx = n
    A[pi[0], n] = 1.0
    A[pi[1], n] = 1.0
    last_join = n
    for k in range(2, n):
        new_join = n + k - 1
        A[last_join, new_join] = 1.0
        A[pi[k], new_join] = 1.0
        last_join = new_join
    return A


def adjacency_to_query_with_real_triples(A, triples_num, original_triples):
    """
    Convert an adjacency matrix to a Query object using the original triples.
    
    Args:
        A: The adjacency matrix (torch tensor or numpy array)
        triples_num: Number of triple nodes
        original_triples: List of original Triple objects
        
    Returns:
        A Query object representing the plan
    """
    if isinstance(A, torch.Tensor):
        A = A.cpu().detach().numpy()
    
    N_NODES = A.shape[0]
    
    if len(original_triples) != triples_num:
        raise ValueError(f"Number of original triples ({len(original_triples)}) doesn't match triples_num ({triples_num})")
    
    def build_tree(node_idx):
        """Recursively build the query tree from the adjacency matrix"""
        # For triple nodes, return the corresponding original triple
        if node_idx < triples_num:
            return original_triples[node_idx]
        
        # For join nodes, find children and build recursively
        children = np.where(A[:, node_idx] > 0.5)[0]
        
        if len(children) != 2:
            raise ValueError(f"Join node {node_idx} has {len(children)} children, expected 2")
        
        left = build_tree(children[0])
        right = build_tree(children[1])
        
        return Join(left=left, right=right)
    
    # Find the root node (join node with no outgoing edges)
    root_idx = N_NODES - 1  # Default to the last node
    for i in range(triples_num, N_NODES):
        if np.sum(A[i, :]) < 0.1:  # No outgoing edges
            root_idx = i
            break
    
    root = build_tree(root_idx)
    return Query(root=root, triples_num=triples_num)


def count_triples_in_plan(plan):
    """
    Count the number of triple patterns in a plan.
    
    Args:
        plan: Query plan representation
        
    Returns:
        Number of triple patterns
    """
    if isinstance(plan, dict):
        if 'triples' in plan:
            return len(plan['triples'])
        elif 'n_triples' in plan:
            return plan['n_triples']
    
    # If plan is a list of triples
    if isinstance(plan, list):
        return len(plan)
    
    return 0


def count_triples_in_plan(plan):
    """
    Count the number of triple patterns in a query plan.
    
    Args:
        plan: Query object representing a join plan
        
    Returns:
        int: The number of triple patterns in the plan
    """
    def traverse_count(node):
        if isinstance(node, Triple):
            return 1
        elif isinstance(node, Join):
            return traverse_count(node.left) + traverse_count(node.right)
        else:
            return 0
    
    return traverse_count(plan.root)


def validate_plan(plan, expected_triples):
    """
    Validate that a query plan contains all expected triple patterns.
    
    Args:
        plan: Query object representing a join plan
        expected_triples: List of Triple objects that should be in the plan
        
    Returns:
        tuple: (is_valid, message) 
               where is_valid is a boolean and message is a description of any issues
    """
    # Check if the plan has the right number of triples
    triples_in_plan = collect_triples_in_plan(plan)
    
    if len(triples_in_plan) != len(expected_triples):
        return False, f"Plan has {len(triples_in_plan)} triples but expected {len(expected_triples)}"
    
    # Check if all expected triples are in the plan
    # Create a simple string representation for comparison
    plan_triple_strs = set(str(t) for t in triples_in_plan)
    expected_triple_strs = set(str(t) for t in expected_triples)
    
    if plan_triple_strs != expected_triple_strs:
        missing = expected_triple_strs - plan_triple_strs
        extra = plan_triple_strs - expected_triple_strs
        message = ""
        if missing:
            message += f"Missing triples: {missing}"
        if extra:
            message += f"Unexpected triples: {extra}"
        return False, message
    
    return True, "Plan is valid"

def collect_triples_in_plan(plan):
    """
    Collect all triple patterns in a query plan.
    
    Args:
        plan: Query object representing a join plan
        
    Returns:
        list: All triple patterns in the plan
    """
    triples = []
    
    def traverse_collect(node):
        if isinstance(node, Triple):
            triples.append(node)
        elif isinstance(node, Join):
            traverse_collect(node.left)
            traverse_collect(node.right)
    
    traverse_collect(plan.root)
    return triples


def plan_to_string(plan):
    """
    Convert a query plan (Query object) to a string representation.
    
    Args:
        plan: Query object representing a join plan
        
    Returns:
        str: String representation of the plan structure
    """
    def node_to_string(node):
        if isinstance(node, Triple):
            return f"({node.s} {node.p} {node.o})"
        elif isinstance(node, Join):
            left_str = node_to_string(node.left)
            right_str = node_to_string(node.right)
            return f"Join({left_str}, {right_str})"
        else:
            return str(node)
    
    if plan is None:
        return "None"
    
    return node_to_string(plan.root)


def plans_are_equivalent(plan1, plan2):
    """
    Check if two query plans are equivalent, considering that joins are symmetric.
    
    Args:
        plan1: First Query object to compare
        plan2: Second Query object to compare
        
    Returns:
        bool: True if the plans are equivalent, False otherwise
    """
    if plan1 is None or plan2 is None:
        return plan1 == plan2
    
    def normalize_node(node):
        """
        Normalize a node to a canonical form for comparison.
        For joins, we sort the children to handle symmetry.
        """
        if isinstance(node, Triple):
            # For triples, create a normalized representation
            return ('Triple', str(node.s), str(node.p), str(node.o))
        elif isinstance(node, Join):
            # For joins, normalize both children and sort them
            left_norm = normalize_node(node.left)
            right_norm = normalize_node(node.right)
            # Sort to handle join symmetry - smaller one first
            children = sorted([left_norm, right_norm])
            return ('Join', children[0], children[1])
        else:
            return str(node)
    
    # Compare the normalized forms
    try:
        norm1 = normalize_node(plan1.root)
        norm2 = normalize_node(plan2.root)
        return norm1 == norm2
    except Exception:
        # If there's any error in comparison, fall back to False
        return False


def load_sparql_queries(queries_file: str, num_queries, seed=42):
    """
    Load all the SPARQL query objects from the given file.
    
    Args:
        queries_file: Path to the file containing saved SPARQLQuery objects
        num_queries: Number of queries to return (None for all)
        seed: Random seed for shuffling (default: 42)
        
    Returns:
        List of SPARQLQuery objects
    """
    with open(queries_file, 'rb') as f:
        sparql_queries = pickle.load(f)
    
    if num_queries is not None:
        print(f"Loaded {num_queries} SPARQL queries from {queries_file}")
        random.seed(seed)  # Set seed for reproducible shuffling
        random.shuffle(sparql_queries)
        return sparql_queries[-num_queries:]
    print(f"Loaded {len(sparql_queries)} SPARQL queries from {queries_file}")
    return sparql_queries



