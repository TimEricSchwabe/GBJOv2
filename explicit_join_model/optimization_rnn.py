import torch



def sample_gumbel(shape, eps=1e-10, device="cpu"):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def sinkhorn(log_alpha, iters=20):          # log_alpha: (n,n)
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()                  # doubly-stochastic

def gumbel_sinkhorn(L, tau, iters=20):
    g = sample_gumbel(L.shape, device=L.device)
    return sinkhorn((L + g) / tau, iters)

def left_deep_adj_from_perm(pi):
    """
    pi: Tensor of length n with the (0-based) permutation of triple nodes.
    Returns A (2n-1, 2n-1) adjacency for a left-deep tree:
       (((T_pi0 ▷◁ T_pi1) ▷◁ T_pi2) … )
    """
    n = len(pi)
    N = 2 * n - 1
    A = torch.zeros(N, N, dtype=torch.float32)
    # indices: triple 0..n-1, join nodes n..2n-2 (root = 2n-2)
    # first join joins pi0 and pi1 -> node idx = n
    A[pi[0], n] = 1.0
    A[pi[1], n] = 1.0
    last_join = n
    for k in range(2, n):
        new_join = n + k - 1
        A[last_join, new_join] = 1.0
        A[pi[k],  new_join] = 1.0
        last_join = new_join
    return A

class CostRNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embed = torch.nn.Linear(in_dim, hidden_dim)
        self.rnn   = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head  = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_feats):           # (1, n, d) → (scalar)
        x = torch.relu(self.embed(seq_feats))
        _, h_n = self.rnn(x)                # h_n: (1,1,H)
        cost_pred = torch.abs(self.head(h_n.squeeze(0)).squeeze(-1))
        return cost_pred                    # shape ()

@torch.no_grad()
def _anneal_tau(init_tau, min_tau, step, max_step):
    return max(min_tau, init_tau * (0.95 ** step))

def optimize_query_gumbel_rnn(
        query_data,                     # torch_geometric.Data
        model,                          # CostRNN
        device="cpu",
        *,
        optimization_steps=500,
        learning_rate=1e-2,
        init_tau=5.0,
        min_tau=0.5,
        lambda_perm=1.0,
        verbose=True,
):

    data = query_data.to(device)
    triple_feats = data.x                          # assume triples first
    n = triple_feats.size(0) // 2 + 1              # same rule: n triples
    triple_feats = triple_feats[:n]                # (n, d)
    d = triple_feats.size(1)

    # ordering logits L
    L = torch.zeros(n, n, device=device, requires_grad=True)
    opt = torch.optim.AdamW([L], lr=learning_rate)

    # histories for optional plotting
    cost_hist, row_pen_hist = [], []

    for step in range(optimization_steps):
        tau = _anneal_tau(init_tau, min_tau, step, optimization_steps)
        P_soft = gumbel_sinkhorn(L, tau)           # (n,n)

        # reorder features and predict cost
        S = P_soft @ triple_feats                 # (n,d)
        cost_pred = model(S.unsqueeze(0))         # scalar

        # permutation penalties
        row_pen = ((P_soft.sum(1) - 1.) ** 2).sum()
        col_pen = ((P_soft.sum(0) - 1.) ** 2).sum()
        loss = cost_pred + lambda_perm * (row_pen + col_pen)

        opt.zero_grad()
        loss.backward()
        opt.step()

        cost_hist.append(cost_pred.item())
        row_pen_hist.append(row_pen.item())

        if verbose and (step+1) % 100 == 0:
            print(f"[{step+1}/{optimization_steps}] "
                  f"cost={cost_pred.item():.2f}  "
                  f"τ={tau:.3f}")

    with torch.no_grad():
        P_final = P_soft                         # last soft matrix
        pi = P_final.argmax(dim=1)               # (n,) permutation
        A = left_deep_adj_from_perm(pi)

    return A, n


if __name__ == "__main__":
    from torch_geometric.data import Data
    # fake 3-TP query: 3 triples → 5 nodes total (3 triples + 2 joins)
    n_triples = 3
    dummy_feats = torch.randn(2 * n_triples - 1, 307)   # same dim as before
    dummy_data = Data(x=dummy_feats)
    cost_model = CostRNN(in_dim=307, hidden_dim=512)
    cost_model.load_state_dict(torch.load("/home/tim/query_optimization/training_results/rnn_20250528_115042/model.pt"))

    A, n = optimize_query_gumbel_rnn(
        dummy_data, cost_model,
        optimization_steps=3000, verbose=False
    )
    print("n =", n)
    print("Adjacency\n", A)
