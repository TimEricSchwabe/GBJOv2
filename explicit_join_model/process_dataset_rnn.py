import os
import json
import pickle
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from data import Entity, Triple, Join, Query
from data_loader import save_dataset_single_file
from collections import defaultdict

def generate_datapoint(
    triples_raw: List[List[str]],
    permutation: List[int],
    rdf2vec_dict,
    counts_dict,
) -> tuple[List[str], dict]:
    """Create one RNN datapoint for *triples_raw* given a *permutation*.

    Returns
    -------
    tuple (triples_where, data_dict)
        * `triples_where` – textual `s p o .` representation **in join order**.
        * `data_dict` – dict with keys
            * ``x``  – tensor of shape ``(n, 307)``
            * ``y``  – tensor of shape ``(n,)``  (incremental costs)
            * ``perm`` – tensor of indices (n,)
    """
    # ---- construct Triple objects ------------------------------------------
    triple_objs = [
        Triple(*(Entity(name) for name in tp[:3])) for tp in triples_raw
    ]
    n = len(triple_objs)

    # -------- variable ↦ id mapping -----------------------------------------
    variables: list[Entity] = sorted(
        {var for t in triple_objs for var in t.variables},
        key=lambda v: v.name,
    )
    rng = random.Random()
    rng.shuffle(variables)
    variable_id_dict = {var: idx for idx, var in enumerate(variables)}

    # -------- embeddings in permutation order ------------------------------
    embed_matrix: list[np.ndarray] = []
    triples_where: list[str] = []
    for idx in permutation:
        t = triple_objs[idx]
        triples_where.append(t.where_body())
        embed_matrix.append(
            t.get_embedding(variable_id_dict, rdf2vec_dict, counts_dict)
        )
    x_tensor = torch.tensor(np.stack(embed_matrix, axis=0), dtype=torch.float32)

    # -------- cost sequence ------------------------------------
    # first cost: cardinality of the first triple pattern
    first_tp = triple_objs[permutation[0]]
    costs: list[float] = [float(first_tp.get_cardinality())]

    current_tree: Triple | Join = first_tp
    for idx in permutation[1:]:
        next_tp = triple_objs[idx]
        join_node = Join(left=current_tree, right=next_tp)
        costs.append(float(join_node.get_cost()))
        current_tree = join_node

    y_tensor = torch.tensor(costs, dtype=torch.float32)
    perm_tensor = torch.tensor(permutation, dtype=torch.long)

    data_dict = {
        "x": x_tensor,
        "y": y_tensor
                }

    return triples_where, data_dict

if __name__ == "__main__":
    # ---------------- configuration ----------------------------------
    input_json = "/home/tim/query_optimization/datasets/queries/Star_Queries.json"
    rdf2vec = "/home/tim/query_optimization/datasets/queries/rdf2vec100dim.pkl"
    counts = "/home/tim/query_optimization/datasets/queries/counts.pkl"
    output_dir = "dataset_stars_8_tp_rnn"
    num_plans = 3 # Random permutations per query
    max_queries = 30000  # Maximum number of queries to process 
    triples_num = 5  # Only use queries with exactly this number of triple patterns

    # ---------------- load data ----------------------------------
    print("Loading RDF2Vec embeddings …")
    with open(rdf2vec, "rb") as f:
        rdf2vec_dict = pickle.load(f)

    print("Loading entity counts …")
    with open(counts, "rb") as f:
        counts_dict = pickle.load(f)

    # ---------------- load queries -----------------------------------------
    print(f"Loading queries from {input_json} …")
    with open(input_json, "r") as f:
        raw_queries = json.load(f)

    # filter by triple-pattern count
    raw_queries = [q for q in raw_queries if len(q["triples"]) == triples_num]
    if max_queries:
        raw_queries = raw_queries[:max_queries]
    print(f"Using {len(raw_queries)} queries with {triples_num} triple patterns")

    # ---------------- dataset creation -------------------------------------
    all_triples: list[list[str]] = []
    all_data: list[dict] = []

    for qi, q in enumerate(tqdm(raw_queries, desc="Processing queries")):
        triples_raw = q["triples"]
        n = len(triples_raw)

        for p_i in range(num_plans):
            perm = list(range(n))
            random.shuffle(perm)
            try:
                triples_where, data_dict = generate_datapoint(
                    triples_raw, perm, rdf2vec_dict, counts_dict
                )
                all_triples.append(triples_where)
                all_data.append(data_dict)
            except Exception as e:
                print(f"  Error generating datapoint for query {qi} (plan {p_i}): {e}")

    # ---------------- save ---------------------------------------------------
    print(f"Saving {len(all_data)} datapoints → {output_dir}/dataset.pt …")
    save_dataset_single_file(all_triples, all_data, output_dir)

    print("Done.")