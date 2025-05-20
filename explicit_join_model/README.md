# Continuous Query Optimization for SPARQL - Explicit Model


### Core Files

- **data.py**: Contains core data structures for representing SPARQL queries and join plans.
  - `Entity`, `Triple`, `Join`, and `Query` classes for query representation
  - Functions for creating random join orders and converting plans to adjacency matrices
  - Utility functions for working with query plans

- **model.py**: Implements the neural network models for cost prediction.
  - `GINConv`: Modified Graph Isomorphism Network layer
  - `CostGNN`: Graph neural network model for cost prediction
  - `CostGNNv2`: Enhanced version with residual connections and layer normalization

- **training.py**: Contains code for training the cost prediction model.
  - Functions for training and validating the model
  - Hyperparameter configuration in the main block
  - Usage: `python training.py` to train the model with configured parameters

- **data_loader.py**: Provides utilities for loading and managing datasets.
  - `QueryDataset`: Dataset class for loading individual query plan samples
  - `SingleFileQueryDataset`: Optimized dataset class that loads all data from a single file
  - Functions for saving and loading dataset metadata

- **process_dataset_single_file.py**: Processes SPARQL queries into a format suitable for training.
  - Converts raw SPARQL queries into various join plans with associated costs
  - Creates PyTorch Geometric graph representations of the plans
  - Saves processed data for efficient training
  - Usage: `python process_dataset_single_file.py` to process raw queries into training data

- **optimization_evaluation.py**: Implements and evaluates the query optimization approach.
  - `optimize_query`: Implements gradient-based optimization of query plans
  - Functions for evaluating optimized plans against greedy and random baselines
  - Visualization and comparison of optimization results
  - Usage: `python optimization_evaluation.py` to run optimization experiments

### Workflow

1. **Data Processing**: Run `process_dataset_single_file.py` to convert raw SPARQL queries into training data
2. **Model Training**: Run `training.py` to train the cost prediction model
3. **Optimization**: Run `optimization_evaluation.py` to optimize query plans and evaluate performance

### Configuration

Both training and optimization phases use a configuration dictionary in their respective main blocks:

```python
# Example from training.py
config = {
    'model_type': 'CostGNNv2',  
    'node_feature_dim': 307,    
    'hidden_dim': 512,          
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 2000,
    # ... other parameters
}

# Example from optimization_evaluation.py
config = {
    'queries_file': "sparql_queries_8_single/queries.pkl",
    'model_path': "/path/to/model.pt",
    'num_queries': 500,
    'optimization_steps': 5000,
    'optimization_params': {
        'learning_rate': 0.01,
        'lambda_acyclic': 1000.0,
        # ... other hyperparameters
    }
}
```

Adjust these configuration dictionaries to experiment with different settings.

---

# Continuous Query Optimization for SPARQL  

---

## 1  Introduction and Motivation  

Modern query optimizers usually rely on either  

* **Dynamic-programming search** – explores every join order at *O(3ⁿ)* cost; quickly intractable for large *n*.  
* **Greedy heuristics** – choose locally optimal sub-plans in *O(n²)* time but can miss the global optimum under realistic cost models.  
* **Reinforcement-learning (RL) methods** – sometimes outperform heuristics but are data-hungry, unstable to train, and often as slow as greedy search.  

This project explores a **continuous, gradient-based alternative**:

1. Represent a query plan as a *continuous* adjacency matrix **A**.  
2. Train a neural **cost model** **Ĉ(·)** (a GNN) that predicts the runtime cost.  
3. Use gradient descent on **A** (plus validity penalties) to minimize the predicted cost and recover a discrete, valid join order.

If successful, the method combines the expressiveness of learned cost models with the efficiency of differentiable optimization, potentially beating DP, greedy, and RL approaches on both quality–cost trade-off and scalability.

---

## 2  Representation of Queries and Plans  

### 2.1  Query Representation  

* Embed every triple-pattern (tp) with vector embeddings (e.g., RDF2Vec).  
* Investigate:  
  * Path or sub-query embeddings.  
  * Jointly learning embeddings while training the cost model. citeturn0file0  

### 2.2  Plan Representation & Continuous Relaxation  

* Store the plan in an *n+(n-1) × n+(n-1)* adjacency matrix **A** (one node per tp and one node per join).  
* Relax binary edges to **Aᵢⱼ ∈ (0,1)** during optimization.  
* After optimization, threshold (e.g., 0.5) or apply Gumbel-Softmax / straight-through tricks to obtain a valid {0,1} plan. 

---

## 3  Learning the Cost Model  

A **Graph Neural Network (GNN)** is trained offline on triples (X_q, A_q, C_q):  

* **X_q** – node features (tp embeddings).  
* **A_q** – adjacency of a known plan.  
* **C_q** – measured execution cost.  

The loss is Mean-Squared Error (or similar) between **Ĉ_q** and **C_q**. See *Algorithm 1* for pseudocode. 

---

## 4  Gradient-Based Optimization of Query Plans  

1. **Initialize** **A** with random values in (0,1).  
2. **Predict cost** **Ĉ(A)** via the trained GNN.  
3. **Add penalties** for invalid plans:  
   * **Acyclicity:** Pₐ = tr(eᴬ) − n  
   * **Triple node constraints:**
     * **No incoming edges:** P_triple_in = Σ(in_degreeₜ)²
     * **Exactly one outgoing edge:** P_triple_out = Σ(out_degreeₜ - 1)²
   * **Join node constraints:**
     * **Exactly two incoming edges:** P_join_in = Σ(in_degreeⱼ - 2)²
     * **One outgoing edge (except root):** P_join_out = Σ(out_degreeⱼ - 1)² + (out_degree_root)²
   * **Entropy penalty (optional):** Encourages weights to be either 0 or 1
   * **L1 penalty (optional):** Encourages keeping only the strongest connections
4. **Update** *Aᵢⱼ ← Aᵢⱼ − α ∂L/∂Aᵢⱼ* with learning rate α.  
5. **Clamp** to [0,1], iterate *I* steps.  
6. **Threshold** to obtain a discrete plan, then extract a join order via topological sort. citeturn0file0  

### 4.1  Guaranteeing Valid Plans  

Penalty terms enforce acyclicity and degree bounds; weights λ₁–λ₃ are meta-optimized. Bushy or multi-way joins can be allowed by relaxing constraints. citeturn0file0  

### 4.2  Discrete Sampling with Gumbel-Softmax  

Soft edges can be sharpened by  

```
zᵢⱼ = σ((αᵢⱼ + gᵢⱼ) / τ),   gᵢⱼ ~ Gumbel(0,1)
```  

Annealing τ→0 gradually moves **A** from continuous to almost binary while retaining gradients. citeturn0file0  

---


## 6  Expected Contributions  

1. A novel **gradient-based join-order optimizer** driven by a learned cost model.  
2. Effective continuous-to-discrete relaxation strategies for query plans.  
3. Empirical insights on query/plan representations and penalty design.  
4. Benchmarks vs. classic and RL optimizers on large knowledge graphs. citeturn0file0  



## Appendix A  Algorithms  

<details>
<summary><strong>Algorithm 1 – Cost GNN Training</strong></summary>

```text
Input :
  Training set {(X_q, A_q, C_q)}_{q=1..Q}
  Epochs E
Output:
  Trained cost model Ĉ(·)

Initialize Θ randomly
for epoch = 1 … E do
    for each query q do
        Ĉ_q ← Ĉ(X_q, A_q ; Θ)          ▷ forward
        L  ← MSE(Ĉ_q , C_q)            ▷ loss
        Θ  ← Θ − η ∂L/∂Θ              ▷ gradient step
    end for
end for
return Θ
```
</details> citeturn0file0  

<details>
<summary><strong>Algorithm 2 – Gradient-Based Join Order Optimization</strong></summary>

```text
Input :
  Trained model Ĉ(·), node features X
  Number of triple patterns triples_num
  Optimization steps I
  Penalty weights λ_acyclic, λ_triple_in, λ_triple_out, λ_join_in, λ_join_out, λ_entropy, λ_l1
Output:
  Discrete plan (adjacency or join order)

Initialize edge_weights randomly in [0.4, 0.6]
Initialize optimizer (Adam)
for step = 1 … I do
    # Convert edge_weights to adjacency matrix A
    A ← zeros(N_NODES, N_NODES)
    A[edge_index[0], edge_index[1]] ← edge_weights
    
    Ĉ ← Ĉ(X, edge_index, edge_weights)          ▷ predicted cost
    L_obj ← Ĉ
    
    # Calculate in-degree and out-degree
    in_degree ← sum(A, dim=0)
    out_degree ← sum(A, dim=1)
    
    # Calculate penalties
    P_triple_in ← sum(in_degree[triple_nodes]²)
    P_triple_out ← sum((out_degree[triple_nodes] - 1)²)
    P_join_in ← sum((in_degree[join_nodes] - 2)²)
    P_join_out ← sum((out_degree[join_nodes_except_root] - 1)²) + out_degree[root]²
    P_acyclic ← trace(matrix_exp(A)) - N_NODES
    
    # Optional penalties (if enabled)
    temperature ← max(0.5, 10.0 * (1.0 - step/I))
    if USE_ENTROPY_PENALTY:
        P_entropy ← entropy_penalty(A, temperature)
    if USE_L1_PENALTY:
        P_l1 ← l1_penalty(A, triple_nodes, join_nodes)
    
    # Total loss
    L_pen ← λ_acyclic·P_acyclic + λ_triple_in·P_triple_in + λ_triple_out·P_triple_out + 
            λ_join_in·P_join_in + λ_join_out·P_join_out
    if USE_ENTROPY_PENALTY:
        L_pen ← L_pen + λ_entropy·P_entropy
    if USE_L1_PENALTY:
        L_pen ← L_pen + λ_l1·P_l1
    
    L ← L_obj + L_pen
    
    # Update using optimizer
    optimizer.step(L)
    
    # Clamp edge weights to [0,1]
    edge_weights ← clamp(edge_weights, 0, 1)
end for

# Thresholding
A[A < 0.5] ← 0
A[A ≥ 0.5] ← 1

return A
```
</details> citeturn0file0  

---

### Notation Key
* **A** – adjacency matrix (plan)  *X* – node features  
* **Ĉ** – learned cost estimator  *C* – true execution cost  
* **P_triple_in**, **P_triple_out**, **P_join_in**, **P_join_out**, **P_acyclic**, **P_entropy**, **P_l1** – penalty terms  
* **α** – learning rate  *λ* – penalty weights  *τ* – temperature for entropy penalty  

---