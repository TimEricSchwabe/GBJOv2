# GBJO: Gradient Based Join Ordering

This repository contains the code for the paper on Gradient Based Join Ordering and implements a gradient-based search method for query plans.


## Overview

The overview closely follows the experiments performed in the paper and presents the following:

1. **Datasets and Queries**: How to process SPARQL queries and generate training/evaluation datasets for query plans
2. **Results**: Where and how the raw results from the paper are saved
3. **Cost Model Training**: A Graph Neural Network (CostGNN) that learns to predict query execution costs
4. **Gradient-Based Optimization**:  Optimization procedures for finding optimal join orders
5. **Cost and Runtime Comparison**: Comparing gradient-based optimization to other baselines
6. **Visualization**: Generation of final plots and error landscape visualizations


## Installation

### Python Packages

To install the required python packages to run the experiments (from `pyproject.toml`):

```bash
pipx install uv
uv pip install -e .
```
### SPARQL endpoint

Further, to calculate costs for generated plans (and optionally for generating rdf2vec embeddings) you need to serve the used graphs
via a SPARQL endpoint. We recommend using qlever for this: 
https://github.com/ad-freiburg/qlever


## Datasets and Queries

### Graph Datasets
The datasets on which the queries are based are taken from GNCE[1]. They are stored in .nt format (one triple per line) and can be accessed
in the `/datasets/graphs` folder (this and all other data below is hosted here: https://figshare.com/s/ab9a9ea0b55b647e031e).

### RDF2Vec Embeddings and Entity Counts

Likewise, the pre-computed RDF2Vec embeddings and entity counts are pickled and stored in the `/datasets/graphs` folder. To generate these files yourself, you can run the following scripts:

- **To generate RDF2Vec embeddings:**
  ```bash
  python src/create_data/generate_embeddings.py
  ```
  > **Note**: For larger graphs like Wikidata, `generate_embeddings.py` might consume a large amount of RAM. You might need to load the graph via a SPARQL endpoint to mitigate this.

- **To generate entity counts:**
  ```bash
  python src/create_data/generate_counts.py
  ```



### Query Datasets

The queries that are used to generate random plans and their costs are stored in `/datasets/queries` as.json file.

### Plan Datasets

The plan datasets used in the paper are stored in the `/datasets/plans` folder.

### Plan Dataset Generation

If you want to generate additional plans, they can be generated based on queries via
  ```bash
  python src/create_data/create_cost_model_training_data.py
  ```
  > **Important**: The .nt file for the graph you generate plans for need to be loaded into a SPARQL endpoint in order to calculate the c_out costs for each plan.

##  Results
The raw results and plots shown in the paper are saved as follows:

### Model Training
The results for the 4 cost models are saved under `datasets/models` with the model files as well as plots and metrics


### Optimization Results
The results of the optimizationfor GBJO and baselines are saved under `/optimization_results`, including config, plots and raw query plans

## Cost Model Training
After the necessary datasets have been generated, the next step is to train a cost model on them to predict the (c_out) cost.

### Pretrained Models

The models for LUBM and Wikidata used in the paper are saved in the `/datasets/models` folder, with one model for dataset and query shape.


### Training

A new model training can now be started via
```bash
python src/cost_model_training.py
```

Within the code you can specify the config as follows:

#### Training Configuration
```python
    config = {
        # Model parameters
        'model_type': 'CostGNNv3',  # Options: 'CostGNN', 'CostGNNv2', 'CostGNNv3'
        'node_feature_dim': 307,    # Input feature dimension
        'hidden_dim': 128,          # Hidden layer dimension
        
        # CostGNNv3 architecture parameters
        'n_layers': 6,              # Number of GIN message-passing layers
        'use_jk': False,            # Whether to use Jumping Knowledge
        'jk_mode': 'cat',           # JK mode: 'cat', 'max', or 'lstm'
        'use_residual': True,       # Whether to use residual connections
        'use_layer_norm': False,    # Whether to use layer normalization
        'use_graph_norm': False,     # Whether to use graph normalization instead
        'dropout': 0.,             # Dropout probability
        'aggr': 'add',              # Aggregation function for gin layers: 'add' or 'mean'
        
        # Training parameters
        'learning_rate': 0.0001,
        'batch_size': 32,
        'num_epochs': 500,
        'loss_type': 'huber',         # Options: 'mse', 'qerror', 'huber'
        
        # Dataset parameters
        'use_single_file': True,
        # Paths
        'root_dir': '',
        'dataset_dir': '.../star-greedy', # on which plan dataset to train
        
        # Other settings
        'enable_training': True,    # Set to False to skip training
    }
```
The training will be written to `/training_results` in a new folder which stores the best model, plots of metrics/loss during training and plots for final evaluation
on the validation data.


## Join Order Optimization

The code for the gradient-based optimization and the other baselines in the paper can be found in `src/optimization/methods.py`

### Visualizing the Cost Landscape
To generate the visualizations of cost between random plans, run
```bash
python src/visualization/optimization_space_visualization.py
```

### Comparison to baseline join order algorithms

The results for comparison of GBJO to the baselines from the paper are stored in the `optimization_results` folder. The 
results include the plots as well as the raw data including query triples, predicted cost and predicted plan per query.


To regenerate those results, run
```bash
python src/evaluation_parallel.py
```
The code requires you to define a config (examples are given in the file). The new results will similarly be stored in the `optimization_results`
folder. To generate the final plots run
```bash
python src/visualization/plot_optimization_results.py
```



### Hyperparameter Search
In order to search for hyperparameters for a particular model and plan dataset, run
```bash
python src/hyperparameter_search/hyperparam_search.py
```
Change the SEARCH_SPACE in the code to your requirements


## References
[1] Tim Schwabe, Maribel Acosta:
Cardinality Estimation over Knowledge Graphs with Embeddings and Graph Neural Networks. Proc. ACM Manag. Data 2(1): 44:1-44:26 (2024)