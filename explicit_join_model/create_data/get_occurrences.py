import json
from tqdm import tqdm
import os
import pickle

def create_occurrences_file(dataset_name, kg_file):
    """
    This function calculates simple occurrences for entities in an RDF graph
    and saves them to a pickle file.
    
    :param dataset_name: The name of the dataset, used to save the occurrence file
    :param kg_file: path to the .ttl file of the knowledge graph
    :return: None
    """
    
    # Count occurrences of nodes
    occurrences = {}
    
    print("Calculating Occurrences")
    
    with open(kg_file, "r") as file:
        for line in tqdm(file):
            line = line.strip().split(" ")  # Assuming the elements are separated by a space
            s = line[0].replace("<", "").replace(">", "")
            p = line[1].replace("<", "").replace(">", "")
            o = line[2].replace("<", "").replace(">", "")

            # Using dict.get() method to count occurrences
            occurrences[s] = occurrences.get(s, 0) + 1
            occurrences[p] = occurrences.get(p, 0) + 1
            occurrences[o] = occurrences.get(o, 0) + 1

    print(f"Found {len(occurrences)} unique entities")

    # Saving occurrences as pickle file
    output_file = dataset_name + "_counts.pkl"
    with open(output_file, "wb") as fp:
        pickle.dump(occurrences, fp)
    
    print(f"Occurrences saved to {output_file}")
    
    return occurrences

if __name__ == "__main__":
    print('Creating occurrence counts...')
    create_occurrences_file("wikidata", "/home/tim/CQOS-dataset/wikidata/graph/wikidata.ttl")