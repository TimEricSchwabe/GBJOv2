import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, spmm
from typing import Callable, Union, List

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)


class GINConv(MessagePassing):
	"""The graph isomorphism operator from the "How Powerful are
	Graph Neural Networks?" paper."""
	
	def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
				 **kwargs):
		kwargs.setdefault('aggr', 'add')
		super().__init__(**kwargs)
		self.nn = nn
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.empty(1))
		else:
			self.register_buffer('eps', torch.empty(1))
		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()
		reset(self.nn)
		self.eps.data.fill_(self.initial_eps)

	def forward(
		self,
		x: Union[Tensor, OptPairTensor],
		edge_index: Adj,
		edge_attr: OptTensor = None,
		edge_weight: OptTensor = None,
		size: Size = None,
	) -> Tensor:
		if isinstance(x, Tensor):
			x = (x, x)

		# propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
		out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

		x_r = x[1]
		if x_r is not None:
			out = out + (1 + self.eps) * x_r

		return self.nn(out)

	def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
		# Apply edge weights if provided
		if edge_weight is not None:
			if edge_weight.dim() == 1:
				edge_weight = edge_weight.view(-1, 1)
			
			# Apply weight to messages
			return x_j * edge_weight
		return x_j

	def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
		# Note: This method won't support edge weights in its current form
		# For edge weights, the regular message+aggregate pipeline will be used
		if isinstance(adj_t, SparseTensor):
			adj_t = adj_t.set_value(None, layout=None)
		return spmm(adj_t, x[0], reduce=self.aggr)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(nn={self.nn})'


class CostGNN(nn.Module):
	def __init__(self, node_feature_dim, hidden_dim):
		super(CostGNN, self).__init__()
		
		# Define MLPs for GINConv layers
		self.mlp1 = nn.Sequential(
			nn.Linear(node_feature_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		self.mlp2 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		self.mlp3 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		# GINConv layers for more powerful message passing
		self.conv1 = GINConv(self.mlp1)
		self.conv2 = GINConv(self.mlp2)
		self.conv3 = GINConv(self.mlp3)
		
		# Additional FC layers with nonlinearities
		self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
		self.fc2 = nn.Linear(hidden_dim // 2, 1)
		
		# Dropout for regularization
		self.dropout = nn.Dropout(0.2)

	def forward(self, x, edge_index, edge_weight=None, batch=None):
		if edge_weight is not None:
			x = self.conv1(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv1(x, edge_index)

		x = F.relu(x)
		x = self.dropout(x)
		
		if edge_weight is not None:
			x = self.conv2(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv2(x, edge_index)
		
		x = F.relu(x)
		x = self.dropout(x)
		
		
		if edge_weight is not None:
			x = self.conv3(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv3(x, edge_index)
		
		x = F.relu(x)

		# Global pooling
		if batch is not None:
			x = scatter(x, batch, dim=0, reduce='add')
		else:
			x = torch.sum(x, dim=0)
		
		# Apply FC layers with nonlinearities
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout(x)
		cost = torch.abs(self.fc2(x))

		return torch.squeeze(cost)


class CostGNNv2(nn.Module):
	def __init__(self, node_feature_dim, hidden_dim):
		super(CostGNNv2, self).__init__()
		
		self.projection = nn.Linear(node_feature_dim, hidden_dim)
		
		self.mlp1 = nn.Sequential(
			nn.Linear(node_feature_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		self.mlp2 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		self.mlp3 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		
		# GINConv layers for message passing
		self.conv1 = GINConv(self.mlp1)
		self.conv2 = GINConv(self.mlp2)
		self.conv3 = GINConv(self.mlp3)
		
		# Layer normalization after each residual connection
		self.layer_norm1 = nn.LayerNorm(hidden_dim)
		self.layer_norm2 = nn.LayerNorm(hidden_dim)
		self.layer_norm3 = nn.LayerNorm(hidden_dim)
		
		# Additional FC layers with nonlinearities
		self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
		self.fc2 = nn.Linear(hidden_dim // 2, 1)
		
		# Dropout for regularization
		self.dropout = nn.Dropout(0.2)

	def forward(self, x, edge_index, edge_weight=None, batch=None):
		# For the first layer, project input to match hidden_dim for residual
		residual = self.projection(x)
		
		# First message passing layer
		if edge_weight is not None:
			x = self.conv1(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv1(x, edge_index)
		
		# Add residual and apply layer norm
		x = x + residual
		x = self.layer_norm1(x) 
		x = F.relu(x)
		x = self.dropout(x)
		
		# Second message passing layer with residual
		residual = x
		if edge_weight is not None:
			x = self.conv2(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv2(x, edge_index)
		
		# Add residual and apply layer norm
		x = x + residual
		x = self.layer_norm2(x)
		x = F.relu(x)
		x = self.dropout(x)
		
		# Third message passing layer with residual
		residual = x
		if edge_weight is not None:
			x = self.conv3(x, edge_index, edge_weight=edge_weight)
		else:
			x = self.conv3(x, edge_index)
		
		# Add residual and apply layer norm
		x = x + residual
		x = self.layer_norm3(x)
		x = F.relu(x)

		# Global pooling
		if batch is not None:
			x = scatter(x, batch, dim=0, reduce='add')
		else:
			x = torch.sum(x, dim=0)
		
		# Apply FC layers with nonlinearities
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout(x)
		cost = torch.abs(self.fc2(x))

		return torch.squeeze(cost)
	

class CostGNNv3(nn.Module):
    """
    Improved CostGNN with smoother gradient flow for optimization.
    
    Features:
    - Dynamic number of GIN layers (n_layers parameter)
    - Optional Jumping Knowledge (JK) aggregation using PyG's JumpingKnowledge module
    - GELU activation instead of ReLU (smooth gradients)
    - Configurable residual connections and layer normalization
    
    Args:
        node_feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of GIN message-passing layers (default: 3)
        use_jk: Whether to use Jumping Knowledge (default: False)
        jk_mode: JK aggregation mode - 'cat', 'max', or 'lstm' (default: 'cat')
        use_residual: Whether to use residual connections (default: False)
        use_layer_norm: Whether to use layer normalization (default: False)
        dropout: Dropout probability (default: 0.1)
    
    Reference:
        Jumping Knowledge: https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.nn.models.JumpingKnowledge.html
    """
    def __init__(
        self, 
        node_feature_dim, 
        hidden_dim, 
        n_layers=6,
        use_jk=False,
        jk_mode='cat',
        use_residual=False,
        use_layer_norm=False,
        dropout=0.0001,
        **kwargs
    ):
        super(CostGNNv3, self).__init__()
        
        self.n_layers = n_layers
        self.use_jk = use_jk
        self.jk_mode = jk_mode
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        
        self.projection = nn.Linear(node_feature_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, aggr='add')) #add or mean
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # ReZero: learnable scales starting at 0 for each layer
        # At init, conv output is multiplied by 0, so only residual passes through
        # The model learns to gradually incorporate conv layers during training
        #self.layer_scales = nn.ParameterList([
        #    nn.Parameter(torch.zeros(1)) for _ in range(n_layers)
        #])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Jumping Knowledge module from PyG
        # Aggregates representations from all layers
        if use_jk:
            # For LSTM mode, we need to pass channels and num_layers
            if jk_mode == 'lstm':
                self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=n_layers)
            else:
                self.jk = JumpingKnowledge(mode=jk_mode)
        else:
            self.jk = None
        
        if use_jk and jk_mode == 'cat':
            # Concatenate all layer outputs
            jk_output_dim = hidden_dim * n_layers
        else:
            jk_output_dim = hidden_dim
        
        # Output layers
        self.fc1 = nn.Linear(jk_output_dim, jk_output_dim // 2)
        self.fc2 = nn.Linear(jk_output_dim // 2, 1)
        
        # Apply safe initialization to prevent explosion
        self._init_weights()
    
    def _init_weights(self):
        """
        Safe initialization to prevent count explosion.
        
        Strategy:
        - Use Xavier/Glorot initialization with reduced gain for stability
        - Initialize the second linear layer of each GIN MLP with small weights
          (similar to ReZero/FixUp) so residual connections dominate early
        - Scale down by 1/sqrt(n_layers) to account for depth
        """
        depth_scale = 1.0 / math.sqrt(self.n_layers)
        
        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight, gain=0.5)
        nn.init.zeros_(self.projection.bias)
        
        # Initialize GIN MLP layers
        for i, conv in enumerate(self.convs):
            mlp = conv.nn
            # First linear layer in MLP: standard init with reduced gain
            nn.init.xavier_uniform_(mlp[0].weight, gain=0.5)
            nn.init.zeros_(mlp[0].bias)
            
            # Second linear layer in MLP: small initialization
            # This ensures the conv output starts small, letting residuals dominate
            nn.init.xavier_uniform_(mlp[2].weight, gain=0.1 * depth_scale)
            nn.init.zeros_(mlp[2].bias)
        
        # Initialize output layers
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        
        # Final layer: small init for stable initial predictions
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Project input
        x = self.projection(x)
        
        # Store layer outputs for Jumping Knowledge
        layer_outputs: List[Tensor] = []
        
        # Message passing layers
        for i, conv in enumerate(self.convs):

						# Layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            residual = x if self.use_residual else None



            
            # Graph convolution
            if edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
    
            
            # Activation
            x = F.gelu(x)


            if self.use_residual:
                x = residual + x
            # Dropout (skip on last layer if not using JK)
            if i < self.n_layers - 1:
                pass
                #x = self.dropout(x)
            
            # Store for JK
            layer_outputs.append(x)
        
        # Jumping Knowledge aggregation using PyG module
        if self.jk is not None:
            x = self.jk(layer_outputs)

        # Global pooling
        if batch is not None:
            x = scatter(x, batch, dim=0, reduce='add')
        else:
            x = torch.sum(x, dim=0)
        
        # Output MLP
        x = self.fc1(x)
        x = F.gelu(x)
        cost = self.fc2(x)

        return torch.squeeze(cost)


class AddLearnableFingerprints(torch.nn.Module):
    """
    Learnable fingerprint embeddings for join nodes.
    Handles batched graphs correctly - assigns fingerprints per-graph.
    """
    def __init__(self, num_fingerprints=15, fingerprint_dim=32):
        super().__init__()
        self.num_fingerprints = num_fingerprints
        self.fingerprint_dim = fingerprint_dim
        
        # Learnable embedding table
        self.fingerprint_embeddings = torch.nn.Parameter(
            torch.randn(num_fingerprints, fingerprint_dim) * 0.1
        )
    
    def forward(self, x, batch):
        """
        Add fingerprints to join nodes with random assignment PER GRAPH.
        
        Args:
            x: Node features [total_nodes, feature_dim]
            batch: Graph membership [total_nodes] - which graph each node belongs to
        
        Returns:
            Modified x with fingerprints added to join nodes
        """
        x = x.clone()
        device = x.device
        
        # Identify join nodes (last dim == 1)
        is_join = (x[:, -1] == 1.0)
        
        if not is_join.any():
            return x
        
        # Get number of graphs in batch
        num_graphs = batch.max().item() + 1
        
        # Process each graph separately
        for graph_idx in range(num_graphs):
            # Mask for nodes in this graph that are join nodes
            graph_mask = (batch == graph_idx)
            graph_join_mask = graph_mask & is_join
            
            join_indices = torch.where(graph_join_mask)[0]
            n_joins = len(join_indices)
            
            if n_joins == 0:
                continue
            
            # Random assignment for THIS graph
            perm = torch.randperm(self.num_fingerprints, device=device)[:n_joins]
            fingerprints = self.fingerprint_embeddings[perm]  # [n_joins, fingerprint_dim]
            
            # Insert fingerprints
            x[join_indices, :self.fingerprint_dim] = fingerprints
        
        return x