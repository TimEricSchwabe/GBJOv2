import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, spmm
from typing import Callable, Union

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
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
			# Handle edge weights that are 1D (need to reshape)
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
		# For GINConv, edge_weight needs special handling
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
		
		# Projection for first residual connection
		self.projection = nn.Linear(node_feature_dim, hidden_dim)
		
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
	