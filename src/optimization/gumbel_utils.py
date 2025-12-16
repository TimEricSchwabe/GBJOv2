
import torch


@torch.no_grad()
def _temperature_anneal(init_tau: float, min_tau: float, decay: float, step: int, max_step: int) -> float:
    """
    Exponential temperature annealing every step.
    
    Args:
        init_tau: Initial temperature
        min_tau: Minimum temperature
        decay: Decay factor 
        step: Current step
        max_step: Maximum steps
        
    Returns:
        Annealed temperature
    """
    return max(min_tau, init_tau - (init_tau - min_tau) * (step / max_step)) 


def sample_gumbel(shape, eps=1e-10, device="cpu"):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def sample_binary_concrete(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample from the Binary Concrete (Gumbel‑Sigmoid) distribution using the
    re‑parameterisation trick and return a straight‑through sample.
    
    Args:
        logits: Raw, unconstrained edge logits (shape: [num_edges]).
        temperature: Positive temperature τ controlling smoothness.
        
    Returns:
        edge_weights: Straight‑through hard sample in [0,1] with gradients.
    """
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    y_soft = torch.sigmoid((logits + gumbel) / temperature)
    hard = False
    if hard:
        y_hard = (y_soft >= 0.5).float()
        # Straight‑through estimator: replace forward value with hard, keep soft gradient
        return y_hard.detach() - y_soft.detach() + y_soft
    else:
        return y_soft


def sample_grouped_gumbel_softmax(edge_logits: torch.Tensor,
                                  src_nodes: torch.Tensor,
                                  temperature: float,
                                  use_gumbel_noise: bool = True) -> torch.Tensor:
    """
    Return relaxed one-hot edge weights such that every *source* node
    emits exactly one outgoing edge (in expectation) using the Gumbel-Softmax
    trick.

    Args:
        edge_logits: Tensor of shape (E,) - Unconstrained logits of every candidate edge.
        src_nodes: Tensor of shape (E,) - Source node index for each edge (aligned with edge_logits).
        temperature: Positive softmax temperature τ.
        use_gumbel_noise: Whether to add Gumbel noise for stochastic sampling.

    Returns:
        Tensor of shape (E,) – edge weights in (0,1) summing to 1 for every
        set of edges that share the same source node.
    """

    device = edge_logits.device
    edge_weights = torch.empty_like(edge_logits)

    for v in torch.unique(src_nodes):
        mask = (src_nodes == v)
        logits_group = edge_logits[mask]
        if use_gumbel_noise:
            g = sample_gumbel(logits_group.shape, device=device)
            edge_weights[mask] = torch.softmax((logits_group + g) / temperature, dim=0)
        else:
            edge_weights[mask] = torch.softmax(logits_group / temperature, dim=0)



    return edge_weights 