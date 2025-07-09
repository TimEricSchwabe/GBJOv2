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
    
    #walks = model.get_walks(kg, entities)
    #print(walks)


    # Use fit_transform which handles the walk generation internally
    embeddings = model.fit_transform(kg, entities)

    with open("rdf2vec100dim.pkl", "wb") as f:
        pickle.dump(
            dict(zip(entities, embeddings)),
            f
        )



if __name__ == "__main__":
    entities = []
    with open('/home/tim/CQOS-dataset/lubm/star/Joined_Queries.json', 'r') as f:
       queries = json.load(f)
    for query in queries:
       entities += query['x']
    #with open('/home/tim/Datasets/yago/star/Joined_Queries.json', 'r') as f:
    #    queries = json.load(f)
    #for query in queries:
    #    entities += query['x']


    entities = list(set(entities))
    entities = entities[:5]

    print('Using ', len(entities), ' entities for RDF2Vec')

    print(entities)

    #entities = ["http://example.org/122898", "http://example.org/125346", "http://example.org/127818", "http://example.org/238792", "http://example.org/117921"]

    generate_embeddings("/home/tim/CQOS-dataset/lubm/graph/lubm.nt", entities)

    #/home/tim/CQOS-dataset/lubm/graph/lubm.nt
    #http://127.0.0.1:8890/sparql/  