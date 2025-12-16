import os
import pickle
import torch
from torch_geometric.data import Dataset
import random

# Import SPARQLQuery definition if needed (or ensure it's picklable and available in path)
# Assuming the class definition is needed for unpickling or available in scope.
# If the pickle file relies on a specific module structure, make sure it matches.
# For now, we assume the objects can be unpickled if the class structure is compatible.

class SPARQLQueryDataset(Dataset):
    """
    Dataset that loads a list of SPARQLQuery objects from a pickle file.
    Each item in the dataset is a SPARQLQuery object containing multiple plans.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, hard_negative_prob=0.1):
        self.dataset_path = os.path.join(root, 'queries.pt') # Assumes queries.pkl is the filename
        # FIX: Pass None to super to prevent it from applying transform to the tuple result
        super(SPARQLQueryDataset, self).__init__(root, None, pre_transform, pre_filter)
        self.custom_transform = transform
        self.hard_negative_prob = hard_negative_prob
        
        if os.path.exists(self.dataset_path):
            try:
                # Try loading with torch.load first (faster, handles tensors better)
                # weights_only=False is required because we are loading custom objects (SPARQLQuery)
                self.sparql_queries = torch.load(self.dataset_path, weights_only=False)
            except (RuntimeError, pickle.UnpicklingError, TypeError):
                # Fallback to standard pickle load if torch.load fails
                with open(self.dataset_path, 'rb') as f:
                    self.sparql_queries = pickle.load(f)
        else:
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
            
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['queries.pkl']

    def len(self):
        return len(self.sparql_queries)

    def get(self, idx):
        """

        """

        query = self.sparql_queries[idx]
        return self._sample_pair(query)

    def set_hard_negative_prob(self, hard_negative_prob):
        self.hard_negative_prob = hard_negative_prob

    def _sample_pair(self, query):
        """
        Samples a pair of (good, bad) plans from the query.
        With probability `hard_negative_prob`, samples from pairs with the smallest cost difference.
        """
        costs = torch.tensor(query.costs, dtype=torch.float)
        
        # Create boolean mask for finite costs
        # (N, 1) & (1, N) -> (N, N) matrix where True only if BOTH costs are finite
        is_finite = torch.isfinite(costs.unsqueeze(1)) & torch.isfinite(costs.unsqueeze(0))

        # Create boolean mask for positive costs (filter out zero costs)
        is_positive = (costs.unsqueeze(1) > 0) & (costs.unsqueeze(0) > 0)
            
        # Create boolean mask for strictly increasing costs
        is_valid_order = costs.unsqueeze(1) < costs.unsqueeze(0)
        
        # Combine: must be finite AND cost[i] < cost[j]
        #valid_pairs_mask = is_finite & is_valid_order
        valid_pairs_mask = is_valid_order & is_positive
        
        # Get indices
        valid_pairs = torch.nonzero(valid_pairs_mask)
        
        if len(valid_pairs) > 0:
            # Check if we should sample the hardest pair (smallest cost difference)
            if random.random() < self.hard_negative_prob:
                # Calculate differences for all valid pairs: cost[j] - cost[i]
                diffs = costs[valid_pairs[:, 1]] - costs[valid_pairs[:, 0]]
                
                # Find minimum difference
                min_diff = diffs.min()
                
                # Select pairs that are close to the minimum difference (within small epsilon)
                hard_mask = diffs <= (min_diff + 1e-6)
                
                # Get indices in valid_pairs that satisfy the condition
                candidate_indices = torch.nonzero(hard_mask).flatten()
                
                # Sample randomly from the hardest candidates
                pair_idx = candidate_indices[torch.randint(0, len(candidate_indices), (1,)).item()]
            else:
                # Randomly select one pair from all possible valid pairs
                pair_idx = torch.randint(0, len(valid_pairs), (1,)).item()
            
            good_idx, bad_idx = valid_pairs[pair_idx].tolist()
            
            good_data = query.torch_data[good_idx]
            bad_data = query.torch_data[bad_idx]
            
            # Ensure y (cost) is set
            if not hasattr(good_data, 'y') or good_data.y is None:
                 good_data.y = costs[good_idx].view(1)
            if not hasattr(bad_data, 'y') or bad_data.y is None:
                 bad_data.y = costs[bad_idx].view(1)

            # FIX: Use self.custom_transform instead of self.transform
            if self.custom_transform is not None:
                good_data = self.custom_transform(good_data.clone())
                bad_data = self.custom_transform(bad_data.clone())

            return good_data, bad_data
        
        # Fallback: return first two (or same if only 1 plan)
        num_plans = len(query.costs)
        idx1, idx2 = 0, 1 if num_plans > 1 else 0
        
        good_data = query.torch_data[idx1]
        bad_data = query.torch_data[idx2]
        
        # FIX: Use self.custom_transform instead of self.transform
        if self.custom_transform is not None:
            good_data = self.custom_transform(good_data.clone())
            bad_data = self.custom_transform(bad_data.clone())

        return good_data, bad_data

# Add this class to data_loader.py

class AddJoinFingerprints:
    """
    Transform that adds orthonormal fingerprint vectors to join nodes.
    Fingerprints are randomly assigned each time a graph is loaded,
    but stay fixed during a single optimization run.
    """
    def __init__(self, fingerprint_dim=14, max_joins=14):
        """
        Args:
            fingerprint_dim: Dimension of fingerprint vectors (default 14 for orthonormal)
            max_joins: Maximum number of join nodes to support (n-1 for n triples)
        """
        self.fingerprint_dim = fingerprint_dim
        self.max_joins = max_joins
        # Pre-compute orthonormal basis (identity matrix rows)
        self.fingerprint_basis = torch.eye(max_joins, fingerprint_dim)
    
    def __call__(self, data):
        """
        Modify node features to include fingerprints for join nodes.
        
        Join nodes are identified by: feature[306] == 1
        Triple nodes have: feature[306] == 0
        """
        x = data.x.clone()
        n_nodes = x.size(0)
        
        # Identify join nodes (last feature dim == 1)
        is_join = (x[:, -1] == 1.0)
        join_indices = torch.where(is_join)[0]
        n_joins = len(join_indices)
        
        if n_joins == 0:
            return data
        
        # Randomly permute fingerprint assignment for this graph
        perm = torch.randperm(self.max_joins)[:n_joins]
        fingerprints = self.fingerprint_basis[perm]  # [n_joins, fingerprint_dim]
        
        # Insert fingerprints into the first `fingerprint_dim` positions of join nodes
        # (these are currently zeros)
        for i, join_idx in enumerate(join_indices):
            x[join_idx, :self.fingerprint_dim] = fingerprints[i]
        
        data.x = x
        return data


class AddRandomGaussianFingerprints:
    """
    Alternative: Fresh random Gaussian fingerprints each time.
    Normalized to unit length for stability.
    """
    def __init__(self, fingerprint_dim=32):
        self.fingerprint_dim = fingerprint_dim
    
    def __call__(self, data):
        x = data.x.clone()
        
        is_join = (x[:, -1] == 1.0)
        join_indices = torch.where(is_join)[0]
        n_joins = len(join_indices)
        
        if n_joins == 0:
            return data
        
        # Fresh random fingerprints, normalized
        fingerprints = torch.randn(n_joins, self.fingerprint_dim)
        fingerprints = fingerprints / fingerprints.norm(dim=1, keepdim=True)
        
        for i, join_idx in enumerate(join_indices):
            x[join_idx, :self.fingerprint_dim] = fingerprints[i]
        
        data.x = x
        return data

