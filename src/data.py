from dataclasses import dataclass
from typing import Union, Optional, Callable

import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import Timeout, RequestException, ConnectionError
import numpy as np
import random
import graphviz
import json

from torch_geometric.data import Data, Dataset, DataLoader
import torch
import asyncio
from tqdm import tqdm
import pickle

# ============== SPARQL Query Execution Setup ==============
# Shared session with retry logic and connection pooling.
# requests.Session is thread-safe, so one session works for all workers.

QLEVER_ENDPOINT = "http://127.0.0.1:7001/"

_session = requests.Session()
_retry = Retry(
    total=1,
    connect=3,
    read=3,
    backoff_factor=0.2,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def run_count_query(where_body: str,
                    connect_timeout: float = 5.0,
                    read_timeout: float = 60.0) -> int:
    """
    Run SELECT (COUNT(*) AS ?count) WHERE { ... } on QLever and return the integer count.
    
    Args:
        where_body: The WHERE clause body (without the WHERE { } wrapper)
        connect_timeout: Timeout for establishing connection
        read_timeout: Timeout for reading response
        
    Returns:
        Integer count result
        
    Raises:
        RuntimeError("SPARQL timeout") on timeout
        RuntimeError("SPARQL error: ...") on persistent HTTP/JSON errors
    """
    query = f"""
        SELECT (COUNT(*) AS ?count)
        WHERE {{
            {where_body}
        }}
    """
    try:
        resp = _session.get(
            QLEVER_ENDPOINT,
            params={"query": query, "format": "json"},
            timeout=(connect_timeout, read_timeout),
            headers={
                "Accept": "application/sparql-results+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        count_str = data["results"]["bindings"][0]["count"]["value"]
        return int(count_str)
    except Timeout:
        raise RuntimeError("SPARQL timeout")
    except (RequestException, ValueError, KeyError, IndexError) as e:
        raise RuntimeError(f"SPARQL error: {e}")

@dataclass
class Entity:
	"""
	Represents an entity in a triple pattern, i.e. subject, predicate or object, which can be a variable or a constant.
	"""
	name: str

	def __post_init__(self):
		self.is_variable = self.name.startswith("?")

	def get_embedding(self, variable_id_dict: dict["Entity", int], rdf2vec=None, counts=None) -> np.ndarray:
		"""
		Size - 102

		0: id of the variable, or 0 if it is a constant
		1-100: embedding of the constant, or 1 if it is a variable
		101: count of the constant, or 0 if it is a variable
		"""

		if self.is_variable:
			return np.concatenate([
				[variable_id_dict[self]],
				np.ones(100),
				[0]
			], axis=0)
		else:
			entity_name = self.name[1:-1]  # Remove angle brackets
			if rdf2vec is None or counts is None:
				raise ValueError("rdf2vec and counts must be provided for constant entities")
			
			# Get embedding and count
			try:		
				embedding = rdf2vec[entity_name]
			except:
				print(f"Entity {entity_name} not found in rdf2vec")
				embedding = rdf2vec.get(entity_name, np.zeros(100))
			count = counts.get(entity_name, 1)
			#count = counts[entity_name]
			
			return np.concatenate([
				[0],
				embedding,
				[count]
			], axis=0)
	
	def __str__(self) -> str:
		return self.name
	
	def __hash__(self):
		return hash(self.name)
			

@dataclass
class Triple:
	"""
	Represents a triple pattern consisting of a subject, predicate and object.
	"""
	s: Entity
	p: Entity
	o: Entity

	def __post_init__(self):
		self.variables = {
			var for var in [self.s, self.p, self.o]
				if var.is_variable
		}

	def where_body(self) -> str:
		return f"{self.s} {self.p} {self.o}."
	
	def json(self) -> Union[str, list]:
		return self.where_body()
	
	def get_embedding(self, variable_id_dict: dict[Entity, int], rdf2vec=None, counts=None) -> np.ndarray:
		"""
		Size - 307

		0-101 - embedding of the subject
		102-203 - embedding of the predicate
		204-305 - embedding of the object
		306 - 0, representing that this is a triple and not a join node
		"""
		return np.concatenate([
			*(
				ent.get_embedding(variable_id_dict, rdf2vec, counts)
				for ent in [self.s, self.p, self.o]
			),
			[0]
		], axis=0)
	
	def get_cardinality(self) -> int:
		"""
		Returns the cardinality (number of matching triples) for this triple pattern.
		This is useful when the triple pattern is considered as a standalone query.
		"""
		return run_count_query(self.where_body())
	
	def get_cost(self) -> int:
		"""
		For triple patterns, we always return 0 as per definition of c_out cost.
		Use get_cardinality() to get the actual number of matching triples.
		"""
		return 0
	
	def add_to_graph(self, graph, node_id):
		label = f"{self.s} {self.p} {self.o}"
		# Escape double quotes to avoid Graphviz syntax errors
		if '"' in label:
			label = label.replace('"', '\\"')
		graph.node(str(node_id), label=label, shape="box")
		return node_id


@dataclass
class Join:
	left: Union[Triple, "Join"]
	right: Union[Triple, "Join"]

	def __post_init__(self):
		self.variables = {
			*self.left.variables,
			*self.right.variables
		}
	
	def where_body(self) -> str:
		return f"{self.left.where_body()} {self.right.where_body()}"
	
	def __str__(self) -> str:
		return f"""
			SELECT {', '.join(str(var) for var in self.variables)}
			WHERE {{
				{self.where_body()}
			}}
		"""

	def json(self) -> Union[str, list]:
		return [self.left.json(), self.right.json()]
	
	def get_embedding(self) -> np.ndarray:
		"""
		Size - 307

		0-305 - zeros
		306 - 1, representing that this is a join node and not a triple
		"""
		return np.concatenate([
			np.zeros(102 * 3),
			[1]
		], axis=0)
	
	def get_cost(self) -> int:
		"""
		Returns the c_out cost of this join
		"""
		self_cardinality = run_count_query(self.where_body())
		left_cost = self.left.get_cost()
		right_cost = self.right.get_cost()

		return self_cardinality + left_cost + right_cost
	
	def get_cardinality(self) -> int:
		return run_count_query(self.where_body())
	

	def add_to_graph(self, graph, node_id):
		# Create join node with bowtie symbol
		join_id = node_id
		graph.node(str(join_id), label="⋈", shape="circle")
		
		# Add left child and get its node id
		current_id = join_id + 1
		left_last_id = self.left.add_to_graph(graph, current_id)
		
		# Add right child and get its node id
		current_id = left_last_id + 1
		right_last_id = self.right.add_to_graph(graph, current_id)
		
		# Connect join node to its left and right children
		graph.edge(str(join_id), str(join_id + 1))  # Connect to left child
		graph.edge(str(join_id), str(left_last_id + 1))  # Connect to right child
		
		return right_last_id


@dataclass
class Query:
	"""
	Represents a query plan recursively 
	"""
	root: Join | Triple
	triples_num: int
	
	def visualize(self, output_file="query_plan", format="png"):
		"""
		Create a visualization of the query plan tree.
		
		Args:
			output_file: File name without extension to save the visualization
			format: Format of the output file (e.g., png, pdf, svg)
		
		Returns:
			The Graphviz object
		"""
		graph = graphviz.Digraph('Query Plan', comment='Query Plan Visualization', 
		                         graph_attr={'rankdir': 'TB'}, 
		                         edge_attr={'dir': 'none'})
		self.root.add_to_graph(graph, 0)
		graph.render(output_file, format=format, cleanup=True)
		return graph



def random_join_order(triples: list[list[str]], seed = None) -> Query:
	triple_objs: list[Triple | Join] = [
		Triple(
			*(Entity(name=name) for name in triple[:3])
		)
		for triple in triples
	]
	
	rng = random.Random(seed)

	rng.shuffle(triple_objs)
	
	while len(triple_objs) > 1:
		join_index = rng.randint(0, len(triple_objs) - 2)
		join = Join(
			left=triple_objs[join_index],
			right=triple_objs[join_index + 1]
		)
		triple_objs[join_index:join_index + 2] = [join]
	
	return Query(
		triple_objs[0],
		len(triples)
	)

	

@dataclass
class Datapoint:
	"""
	Mapping from the adjencency matrix rows to the nodes in the query.
	"""
	nodes_order: list[Triple | Join]

	"""
	Structure of the adjacency matrix:
	| triple patterns in dfs order | join nodes in dfs order |
	"""
	adjacency_matrix: np.ndarray

	"""
	Structure of the embedding matrix:
	shape = (nodes_num, 307)
	"""
	embedding_matrix: np.ndarray

	join_order: Query

	def get_torch_data(self, cost=None) -> Data:
		if cost is None:
			cost = self.join_order.root.get_cost()
		return Data(
			x=torch.tensor(self.embedding_matrix, dtype=torch.float),
			edge_index=torch.tensor(self.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),
			y=torch.tensor([cost], dtype=torch.float)
		)


def join_order_to_adjacency_matrix(join_order: Query, seed = None, rdf2vec=None, counts=None) -> Datapoint:
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


	triple_indexing = iter(range(triples_num))
	join_indexing = iter(range(triples_num, nodes_num))
	
	adjacency_matrix = np.zeros((nodes_num, nodes_num))
	embedding_matrix = np.zeros((nodes_num, 307))
	nodes_order: list[Triple | Join] = [join_order.root] * nodes_num

	def get_new_node_index(node: Triple | Join) -> int:
		return next(triple_indexing) if isinstance(node, Triple) else next(join_indexing)
	
	def get_node_embedding(node: Triple | Join) -> np.ndarray:
		if isinstance(node, Triple):
			return node.get_embedding(variable_id_dict, rdf2vec, counts)
		else:
			return node.get_embedding()

	root_index = next(join_indexing)
	q = [(join_order.root, root_index)]
	embedding_matrix[root_index] = join_order.root.get_embedding()

	while q:
		node, node_index = q.pop(0)

		left_index = get_new_node_index(node.left)
		adjacency_matrix[left_index, node_index] = 1
		embedding_matrix[left_index] = get_node_embedding(node.left)
		nodes_order[left_index] = node.left

		right_index = get_new_node_index(node.right)
		adjacency_matrix[right_index, node_index] = 1
		embedding_matrix[right_index] = get_node_embedding(node.right)
		nodes_order[right_index] = node.right

		if isinstance(node.left, Join):
			q.append((node.left, left_index))
		
		if isinstance(node.right, Join):
			q.append((node.right, right_index))
		
	return Datapoint(
		nodes_order=nodes_order,
		adjacency_matrix=adjacency_matrix,
		embedding_matrix=embedding_matrix,
		join_order=join_order
	)
		
