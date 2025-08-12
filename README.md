# Gradient Based Join Ordering

This repository contains the code for the paper on Gradient Based Join Ordering and implements a gradient-based search method for query plans. The following animation shows the initial continuous superposition of query plans and
how it changes while the gradient optimizer tries to minimize the predicted cost (top right) as well as violation of valid plan constraints (bottom right).


![Demo of feature](./optim.gif)


## Overview

The overview closely follows the experiments performed in the paper and presents the following:

1. **Datasets and Queries**: How to process SPARQL queries and generate training/evaluation datasets for query plans
2. **Results**: Where and how the raw results from the paper are saved
3. **Cost Model Training**: A Graph Neural Network (CostGNN) that learns to predict query execution costs
4. **Gradient-Based Optimization**:  Optimization procedures for finding optimal join orders
5. **Cost and Runtime Comparison**: Comparing gradient-based optimization to discrete Search
6. **Visualization**: Generation of final plots and error landscape visualizations


## Installation

### Python Packages

To install the required python packages to run the experiments:
  ```bash
  pip install -r requirements.txt
  ```

Or, create a conda environment using
  ```bash
 conda env create -f environment.yml
  ```
### SPARQL endpoint

Further, to calculate costs for generated plans (and optionally for generating rdf2vec embeddings) you need to serve the used graphs
via a SPARQL endpoint. We recommend using virtuoso for this: 
https://github.com/openlink/virtuoso-opensource


## Datasets and Queries

### Graph Datasets
The datasets on which the queries are based are taken from GNCE[1]. They are stored in .nt format (one triple per line) and can be accessed
in the `/datasets/graphs` folder.

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

### Training Datasets

The datasets used for training the cost model are stored in the `*_training` folders as .pt files, ready to be used by the training script (see below).
The training files can be generated based on the queries via
  ```bash
  python src/create_data/create_cost_model_training_data.py
  ```
  > **Important**: The .nt file for the graph you generate plans for need to be loaded into a SPARQL endpoint in order to calculate the c_out costs for each plan.

### Optimization Datasets

The datasets used to investigate the cost landscape, and perform the optimization evaluations are stored in the `*_optimization` folders as `.pkl` files.
To generate them, run
  ```bash
  python src/create_data/create_optimization_data.py
  ```

##  Results
The raw results and plots shown in the paper are saved as follows:

### Model Training
The results for the 4 cost models are saved under `/training_results` with the model files as well as plots and metrics

### Gradient Search Fronts
Saved under `/k_vs_cost_results`

### Optimization Results
The results of the optimization comparison to greedy search are saved under `/optimization_results`, including config, plots and raw query plans

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
        'model_type': 'CostGNNv2',  
        'node_feature_dim': 307,    # Input feature dimension
        'hidden_dim': 512,          # Hidden layer dimension
        
        # Training parameters
        'learning_rate': 0.0001,
        'batch_size': 128,
        'num_epochs': 1,
        'loss_type': 'mse',         # Options: 'mse', 'qerror'
        
        # Dataset parameters
        'use_single_file': True, 
        # Paths
        'root_dir': '', # Root Dir of the repository
        'dataset_dir': 'datasets/..', # which .pt to use for training
        
        # Other settings
        'enable_training': False,    # Set to False to skip training and only perform evaluation
    }
```
The training will be written to `/training_results` in a new folder which stores the best model, plots of metrics/loss during training and plots for final evaluation
on the validation data.


## Join Order Optimization

The code for the gradient-based optimization as given in Algorithm 1 in the paper can be found as `optimize_query_gumbel` in `src/optimization/methods.py`

### Visualizing the Cost Landscape
To generate the 1-D visualizations of cost between 2 random plans, run
```bash
python src/cost_landscape_visualization.py
```
The script lets you pick a model and a plan pickle file and then generated 2 random left-linear plans for a given query and interpolates N cost estimations between those to generate the final plot.



### Number of Search Fronts $k$

To gnerate the results how the median predicted cost changes with increasing the number of search fronts $k$, run
```bash
python src/k_vs_cost.py
```

Here, you need to specify a configuration for the gradient-based join order optimizer, as follows:
```python
    config = {
        "queries_file": "datasets/wikidata_star_plan_datasets_optimization/queries.pkl", # on which plan file to run the experiment
        "model_path": "datasets/models/wikidata/star_model.pt", # Which cost model to use
        "num_queries": 20, # How many queries to use for the evaluation
        "optimization_steps": 100, # How many gradient steps to take
        "max_nk": 10, # Maximal number of gradient fronts to perform 
        "optimization_params": {
            "learning_rate": 1.7, # Learning rate of the gradient optimizer
            "lambda_acyclic": 3081.0, # penalty weights as per the paper
            "lambda_triple_in": 3714.0,
            "lambda_triple_out": 135.0,
            "lambda_join_in": 1742.0,
            "lambda_join_out": 1558.0,
            "lambda_entropy": 0.0, # not used
            "lambda_total_penalty": 2.6,
            "lambda_left_linear": 2300.0,
            "init_tau": 4.5, 
            "min_tau": 1.0,
            "tau_decay": 0.963, # not used
            "use_temperature_annealing": True,
            "return_best": True,
            "min_penalty_threshold": 5, # gamma
            "use_lambda_ramping": True,
            "logit_sampling": "softmax",
            "save_animation_data": False,
            "animation_save_interval": 10,
            "lambda_ramp_exponent": 6.5, # p
            "lr_warmup_steps": 0, # optional. perform a warmup of the learning rate
            "gradient_clip_norm": 2, # optional, clip gradients
            "use_lr_scheduling": True,
            "decoding_method": "greedy"
        }
    }
```
The results are saved to the `k_vs_cost` folder in a new subfolder including plots and raw results.

### Comparison to discrete local Search

The results for comparison of gradient-based and greedy search from the paper are stored in the `optimization_results` folder. The 
results include the plots as well as the raw data including query triples, predicted cost and predicted plan per query.


Finally, to compare the gradient-based search to greedy search (and optionally dynamic programming) yourself, run
```bash
python src/evaluation.py
```
The code requires you to define a config similar to above. The new results will similarly be stored in the `optimization_results`
folder. To generate the final plots run
```bash
python src/visualization/plot_optimization_results.py /path/to/your/results_directory
```



### Hyperparameter Search
In order to search for hyperparameters for a particular model and plan dataset, run
```bash
python src/hyperparameter_search/hyperparameter_search.py
```
Change the SEARCH_SPACE in the code to your requirements


## References
[1] Tim Schwabe, Maribel Acosta:
Cardinality Estimation over Knowledge Graphs with Embeddings and Graph Neural Networks. Proc. ACM Manag. Data 2(1): 44:1-44:26 (2024)