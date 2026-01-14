from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
import json
import os
import pickle


def generate_embeddings(graph_file, entities):
    kg = KG(graph_file)

    walker = RandomWalker(max_depth=4, max_walks=10, with_reverse=False, n_jobs=24)
    embedder = Word2Vec(epochs=10, vector_size=100)

    model = RDF2VecTransformer(
        walkers=[walker],
        embedder=embedder,
        verbose=1,
    )
    

    embeddings = model.fit_transform(kg, entities)

    with open("rdf2vec100dim.pkl", "wb") as f:
        pickle.dump(
            dict(zip(entities, embeddings)),
            f
        )



if __name__ == "__main__":
    entities = []
    # Example: Only use entities occurring in the queries
    with open('.../Joined_Queries.json', 'r') as f:
       queries = json.load(f)
    for query in queries:
       entities += query['x']
       if 'instantiated_objects' in query:
           entities += [obj.strip('<>') for obj in query['instantiated_objects']]


    entities = list(set(entities))
    entities = entities[:5]

    print('Using ', len(entities), ' entities for RDF2Vec')

    print(entities)

    generate_embeddings(".../...nt", entities) 
