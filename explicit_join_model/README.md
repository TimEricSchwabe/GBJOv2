# Gradient Based Query Optimization

This repository implements a gradient-based approach for optimizing SPARQL query join orders using a Graph Neural Network (GNN) cost model. Instead of iteratively traversing the discrete search space of query plans, the approach presented here performs a gradient-based search in the continous relaxation of the search space.

## Overview

The project consists of several components:

1. **Data Generation**: Scripts to process SPARQL queries and generate training/evaluation datasets
2. **Cost Model Training**: A Graph Neural Network (CostGNN) that learns to predict query execution costs
3. **Gradient-Based Optimization**: Differentiable optimization procedures for finding optimal join orders
4. **Evaluation Framework**: Comprehensive evaluation comparing gradient-based optimization against baseline methods
5. **Visualization Tools**: Rich plotting and animation capabilities for analyzing optimization results

## Data Generation

The `create_data/` directory contains scripts for generating datasets from SPARQL query files:

### process_dataset_single_file.py

**Purpose**: Generates training data for the Cost GNN by creating multiple random join plans for each query.

**Usage**:
```bash
python create_data/process_dataset_single_file.py
```

**Functionality**:
- Takes raw SPARQL queries and generates multiple random join orders (default: 10 per query)
- Calculates true execution costs for each join plan using `get_cost()` method
- Creates torch geometric `Data` objects with:
  - Node embeddings (307-dimensional features including RDF2Vec embeddings and cardinality estimates)
  - Adjacency matrices representing join trees
  - Cost labels for supervised learning
- Filters out queries with very low cardinality or all-variable triple patterns
- Saves processed queries as pickle files for training

**Key Parameters**:
- `num_plans`: Number of random join orders per query (default: 10)
- `MIN_CARDINALITY`: Minimum query cardinality threshold for inclusion
- `rdf2vec_dict`: Pre-computed RDF2Vec embeddings for entities
- `counts_dict`: Entity frequency statistics

### process_dataset_with_subplans_individual.py

**Purpose**: Generates evaluation data for optimization algorithms by creating single join plans and their subplans.

**Usage**:
```bash
python create_data/process_dataset_with_subplans_individual.py
```

**Functionality**:
- Creates one main join plan per query (no cost calculation to save time)
- Generates left-linear subplans of sizes 3 to n-1 for gradient optimization
- Sets costs to `None` since true costs are expensive to compute during evaluation
- Creates consistent triple-to-index mappings across different plans
- Saves queries individually for efficient loading during optimization

**Key Features**:
- `extract_subplans_left_linear()`: Generates subplans of all sizes for optimization
- `join_order_to_adjacency_matrix_consistent()`: Ensures consistent node indexing
- `calculate_costs=False`: Skips expensive cost calculations for evaluation data


## Cost Model Training

### cost_model_training.py

**Purpose**: Trains the CostGNN model to predict query execution costs from graph representations.

**Usage**:
```bash
python cost_model_training.py
```

**Key Components**:

#### Model Architecture
- **CostGNN**: Graph Convolutional Network with 307-dimensional input features ((s,p,o) embeddings + simple statistics concatenated)
- **Hidden layers**: 512-dimensional by default
- **Output**: Single cost prediction (log-transformed)

#### Training Configuration
```python
config = {
    'dataset_dir': 'path/to/training/data',
    'model_path': 'models/cost_model.pt',
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hidden_dim': 512,
    'loss_type': 'mse',  # or 'qerror'
    'device': 'cuda',
    'train_test_split': 0.8
}
```

#### Loss Functions
- **MSE Loss**: Mean squared error on log-transformed costs
- **Q-Error Loss**: Database-specific metric: `max(pred/true, true/pred)`

#### Model Evaluation
- `validate_model()`: validation with multiple metrics
- `plot_prediction_vs_truth()`: Visualization of model accuracy

## Optimization Evaluation

### evaluation.py

**Purpose**: Main evaluation script comparing gradient-based optimization against Greedy Heuristic, Dynamic Programming, Exhaustive Search and Random Selection.

**Usage**:
```bash
python evaluation.py
```

#### Configuration System

the evaluation is onfigures using a dictionary as follows:

```python
config = {
    # Data and Model
    'queries_file': "path/to/queries.pkl", # Filepath to queries generated using process_dataset_with_subplans_individual.py
    'model_path': "path/to/trained_model.pt", # Path of the weights of the trained CostGNN
    'num_queries': 1000, # on how many queries to evaluate
    'optimization_steps': 1000, # how many steps the gradient based optimizer takes
    'use_true_costs': False,  # Whether to calculate true costs and run the optimized plans on the database
    'use_exhaustive': False,  # Whether to run exhaustive search baseline
    'verbose': False, # Show gradient based cost descent and penalties over time after each query
    'save_path': "optimization_results", 
    
    # Optimization Parameters
    'optimization_params': {
        # Core Algorithm
        'optimization_procedure': 'gumbel',  # 'gumbel' only
        'learning_rate': 1.0, # learning rate of the gradient descent adam optimizer
        
        # Constraint Penalties (enforcing valid join trees)
        'lambda_acyclic': 1000.0,      # Acyclicity constraint
        'lambda_triple_in': 1000.0,    # Triple node in-degree ≤ 1  
        'lambda_triple_out': 1000.0,   # Triple node out-degree ≤ 1
        'lambda_join_in': 500.0,       # Join node in-degree ≤ 1
        'lambda_join_out': 1000.0,     # Join node out-degree = 2
        'lambda_left_linear': 1000.0,  # Left-linear tree structure, set to 0 to allow for bushy plans
        'lambda_entropy': 0.0,          # Edge weight entropy regularization
        'lambda_total_penalty': 1.0,    # Overall penalty weight
        
        # Gumbel-Softmax Parameters
        'init_tau': 10.0,              # Initial temperature
        'min_tau': 1.0,                # Minimum temperature  
        'tau_decay': 0.999,            # Temperature decay rate
        'use_temperature_annealing': True,
        
        # Solution Selection
        'return_best': True,           # Return best feasible solution
        'min_penalty_threshold': 0.1,  # Feasibility threshold (how low do the penalties need to be to be accepted ?)
        'use_lambda_ramping': True,     # Gradually increase penalty weights
        'logit_sampling': 'dual-softmax',  # Sampling method (sigmoid, softmax or dual-softmax)
        
        # Animation and Debugging
        'save_animation_data': False,
        'animation_save_interval': 10
    }
}
```

#### Optimization Methods Compared

1. **Gradient-Based** (`optimize_query_gumbel`):
   - Performs Gradient-Based Search in the continously relaxed search space of plans 
   - Enforces plan validity constraints via penalty functions
   - Supports both left-linear and bushy trees (controlled by `lambda_left_linear`)

2. **Greedy** (`greedy_optimize_query`):
   - Iteratively adds lowest-cost joins
   - Fast but potentially suboptimal

3. **Random** (`random_join_plan`):
   - Random join ordering baseline

4. **Dynamic Programming** (`dp_leftdeep_best_plan`):
   - Uses trained cost model for plan evaluation (similar to greedy)

5. **Exhaustive Search** (`exhaustive_leftdeep_best_plan`):
   - Brute-force optimal solution (optional, expensive)



#### Output Files

The evaluation saves the following results to the save_path specified in the config:

- `config.json`:  configuration used
- `detailed_results.json`: Per-query results with all methods (for visualization, see below)
- `final_statistics.json`: Aggregated performance metrics
- `plan_visualizations/`: Graph visualizations of found plans
- `animation_data/`: Data for creating optimization animations

## Visualization Tools

The `visualization/` directory provides comprehensive analysis and plotting capabilities:



### plot_optimization_results.py

**Plotting script for analyzing saved optimization results:**

**Usage**:
```bash
python visualization/plot_optimization_results.py
```

**Configuration**:
```python
RESULTS_DIR = "path/to/optimization_results/run_timestamp"
# Plot type flags
SKIP_BOXPLOT = False
SKIP_BARPLOT = False
SKIP_SCATTER = False
SKIP_RATIOS = False
SKIP_SIZE_ANALYSIS = False
SKIP_SUMMARY = False
EXCLUDE_TRUE_COSTS = True  # New flag to exclude true costs from plots

# Data inclusion flags
INCLUDE_PREDICTED = True  # Include predicted costs in boxplot
EXCLUDE_EXHAUSTIVE = True  # Exclude exhaustive search from plots
EXCLUDE_GREEDY = False  # Exclude greedy method from plots
EXCLUDE_GRADIENT = False  # Exclude gradient method from plots
EXCLUDE_DP = False  # Exclude DP method from plots
```



### optimization_space_visualization.py

**visualization of optimization landscapes for queries of size 3:**

- plots the cost landscapes with points being superpositions of the 3 possible plans 
- plots the gradient trajectory taken during optimization



### optim_animation.py

**Creates MP4 animations of the optimization process:**



**Animation Features**:
- **Graph Evolution**: Shows how adjacency matrix changes during optimization
- **Cost Tracking**: Cost and Penalty over time -corresponding to currently shown graph
- **Edge Weight Visualization**: Color and thickness indicate edge strengths


## Usage Examples

### Complete Workflow

1. **Generate Training Data**:
```bash
python create_data/process_dataset_single_file.py
```

2. **Train Cost Model**:
```bash
python cost_model_training.py
```

3. **Run Optimization Evaluation**:
```bash
python evaluation.py
```

4. **Generate Visualization**:
```bash
python visualization/plot_optimization_results.py
```


## Dependencies

- **PyTorch**: Neural network implementation
- **PyTorch Geometric**: Graph neural networks
- **NetworkX**: Graph manipulation and visualization
- **NumPy/SciPy**: Numerical computations
- **Matplotlib**: Plotting and animations
- **tqdm**: Progress tracking
- **pickle/json**: Data serialization

## File Structure Summary

```
explicit_join_model/
├── README.md                    # This file
├── data.py                      # Core data structures (Triple, Join, Query)
├── model.py                     # CostGNN architecture
├── data_loader.py               # PyTorch data loading utilities for CostGNN training
├── cost_model_training.py       # CostGNN training
├── evaluation.py                # Main optimization evaluation
├── create_data/                 # Data generation scripts
│   ├── process_dataset_single_file.py           # CostGNN Training data generation
│   ├── process_dataset_with_subplans_individual.py  # Evaluation data
├── optimization/                # Optimization algorithms
├── visualization/               # Plotting and analysis tools
│   ├── evaluation_plots.py                     # Core plotting functions
│   ├── plot_optimization_results.py           # Standalone result analysis
│   ├── optimization_space_visualization.py    # 3-D Cost Landscape Visualization
│   └── optim_animation.py                     # Optimization animation
└── utils/                       # Utility functions
```

## Data structures
### Torch dataset:
[Source](./data.py#L368)
```python
Data(
   # Node features (307-dimensional embeddings)
   x = torch.tensor(self.embedding_matrix, dtype=torch.float), 

   # Graph edges (adjacency matrix)
   edge_index = torch.tensor(self.adjacency_matrix, dtype=torch.float).nonzero(as_tuple=False).t().contiguous(),

   # Cost label (log-transformed cost of the join order)
   y=torch.tensor([self.join_order.root.get_cost()], dtype=torch.float)
)
```
