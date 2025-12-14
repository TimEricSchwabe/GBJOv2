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
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_path = os.path.join(root, 'queries.pt') # Assumes queries.pkl is the filename
        super(SPARQLQueryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
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
        Returns the SPARQLQuery object at index `idx`.
        Note: The DataLoader will collate these. Since SPARQLQuery is a custom object,
        standard PyG collation might not work directly if we return the object itself.
        We should return what the training loop needs: a pair of (good_plan_data, bad_plan_data).
        """
        # We'll handle sampling here to return compatible data types (torch_geometric.data.Data)
        # However, 'get' is expected to return a single item.
        # If we want to support pairwise sampling, we can do it here.
        query = self.sparql_queries[idx]
        return self._sample_pair(query)

    def _sample_pair(self, query):
        """
        Samples a pair of (good, bad) plans from the query.
        """
        num_plans = len(query.costs)
        
        # Try to find a pair with different costs
        for _ in range(20):
            idx1, idx2 = torch.randint(0, num_plans, (2,)).tolist()
            if idx1 == idx2:
                continue
                
            cost1 = query.costs[idx1]
            cost2 = query.costs[idx2]
            
            if cost1 != cost2:
                if cost1 < cost2:
                    good_idx, bad_idx = idx1, idx2
                else:
                    good_idx, bad_idx = idx2, idx1
                
                good_data = query.torch_data[good_idx]
                bad_data = query.torch_data[bad_idx]
                
                # Ensure y (cost) is correct in the data object (in case it wasn't stored/updated)
                # It seems create_data stores it, but let's be safe if we need it for regression
                if not hasattr(good_data, 'y') or good_data.y is None:
                     good_data.y = torch.tensor([query.costs[good_idx]], dtype=torch.float)
                if not hasattr(bad_data, 'y') or bad_data.y is None:
                     bad_data.y = torch.tensor([query.costs[bad_idx]], dtype=torch.float)

                return good_data, bad_data
        
        # Fallback: return first two (or same if only 1 plan, though create_data filters those)
        idx1, idx2 = 0, 1 if num_plans > 1 else 0
        return query.torch_data[idx1], query.torch_data[idx2]

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

