from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) Utilities: EMA target copy
# ----------------------------
@torch.no_grad()
def ema_update_(target: nn.Module, source: nn.Module, decay: float) -> None:
    """target <- decay*target + (1-decay)*source"""
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)


def freeze_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def unfreeze_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(True)


# ----------------------------
# 2) Hyperparameters to learn (lambdas, eta, tau)
# ----------------------------
class LearnableHyper(nn.Module):
    """
    Learnable scalars (or vectors) that appear inside the inner-loop loss and update.
    We store raw params and map to constrained domain.
    """
    def __init__(self, init_lambda: float = 1.0, init_eta: float = 1e-2, init_tau: float = 1.0):
        super().__init__()
        # Ensure positivity via softplus; store inverse-softplus-ish in raw form:
        self.lambda_raw = nn.Parameter(torch.tensor(float(init_lambda)).log().clamp(min=-10, max=10))
        self.eta_raw    = nn.Parameter(torch.tensor(float(init_eta)).log().clamp(min=-20, max=5))
        self.tau_raw    = nn.Parameter(torch.tensor(float(init_tau)).log().clamp(min=-10, max=10))

    def lam(self) -> torch.Tensor:
        # positive
        return F.softplus(self.lambda_raw)

    def eta(self) -> torch.Tensor:
        # positive step size
        return F.softplus(self.eta_raw)

    def tau(self) -> torch.Tensor:
        # positive temperature
        return F.softplus(self.tau_raw)


# ----------------------------
# 3) Placeholder encoders / cost models
# ----------------------------
class CostModel(nn.Module):
    """
    Replace with your GNN / Transformer / whatever.
    Must be differentiable wrt plan_soft.
    """
    def __init__(self, plan_dim: int, query_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(plan_dim + query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, plan_feat: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        # plan_feat: [B, plan_dim], query_feat: [B, query_dim]
        x = torch.cat([plan_feat, query_feat], dim=-1)
        return self.net(x).squeeze(-1)  # [B]


# ----------------------------
# 4) Plan parameterization (z -> soft plan), penalties, projection, execution
# ----------------------------
def relax_to_soft_plan(z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Map logits z to a differentiable 'soft plan feature'.

    You must replace this with your actual relaxation:
      - Gumbel-Softmax over parent choices,
      - or soft adjacency matrix,
      - or whatever GBJO-style relaxation you use.

    Here: we treat z as plan logits and return a "soft plan feature" via softmax.
    """
    # z: [B, D]
    # tau: scalar tensor
    return F.softmax(z / tau.clamp_min(1e-6), dim=-1)  # [B, D]


def validity_penalty(plan_soft: torch.Tensor) -> torch.Tensor:
    """
    Replace with your differentiable validity penalties.
    Example: encourage sparsity or constraint satisfaction.
    """
    # simple example: penalize entropy (encourage peaky distributions)
    ent = -(plan_soft.clamp_min(1e-9) * plan_soft.clamp_min(1e-9).log()).sum(dim=-1)  # [B]
    return ent  # [B]


def project_to_discrete_plan(plan_soft: torch.Tensor) -> torch.Tensor:
    """
    Non-differentiable projection for execution (argmax).
    Returns a discrete representation (placeholder).
    """
    # Example: argmax one-hot
    idx = plan_soft.argmax(dim=-1)  # [B]
    one_hot = F.one_hot(idx, num_classes=plan_soft.shape[-1]).float()
    return one_hot  # [B, D]


def execute_plan(discrete_plan: torch.Tensor, query_obj: Any) -> torch.Tensor:
    """
    You implement this. It must return true cost/runtime for each element in batch.
    Must be torch.Tensor on CPU/GPU; but gradients are NOT needed.
    """
    raise NotImplementedError("Connect this to your DB engine / executor.")


# ----------------------------
# 5) Inner unroll (differentiable)
# ----------------------------
@dataclass
class UnrollResult:
    z_T: torch.Tensor            # final logits after inner steps (kept in graph)
    plan_soft_T: torch.Tensor    # relaxed plan at final step (in graph)


def inner_unroll(
    z0: torch.Tensor,
    query_feat: torch.Tensor,
    theta_model: nn.Module,
    hyper: LearnableHyper,
    T: int,
    add_noise_std: float = 0.0,
) -> UnrollResult:
    """
    Differentiable unrolled optimization:
      z_{t+1} = z_t - eta * d/dz [ C_theta(relax(z_t,tau), q) + lam * penalty(...) ].

    Key: create_graph=True to allow "backprop through backprop".
    """
    # Ensure z participates in autograd
    z = z0
    if add_noise_std > 0:
        z = z + add_noise_std * torch.randn_like(z)

    for _ in range(T):
        eta = hyper.eta()
        tau = hyper.tau()
        lam = hyper.lam()

        plan_soft = relax_to_soft_plan(z, tau)  # differentiable

        # Inner loss: cost + penalties
        cost_pred = theta_model(plan_soft, query_feat)               # [B]
        pen = validity_penalty(plan_soft)                            # [B]
        Lin = (cost_pred + lam * pen).mean()                         # scalar

        # Inner gradient wrt z, but keep graph for higher-order grads
        (g,) = torch.autograd.grad(Lin, z, create_graph=True)

        z = z - eta * g

    plan_soft_T = relax_to_soft_plan(z, hyper.tau())
    return UnrollResult(z_T=z, plan_soft_T=plan_soft_T)


# ----------------------------
# 6) Full outer training step
# ----------------------------
@dataclass
class TrainConfig:
    T_inner: int = 20
    beta_sup_theta: float = 1.0      # weight of supervised anchor for theta
    gamma_outer: float = 1.0         # weight of outer meta loss
    ema_decay_psi: float = 0.99
    grad_clip: float = 1.0


def train_step(
    *,
    query_obj: Any,
    query_feat: torch.Tensor,          # [B, query_dim]
    z0: torch.Tensor,                  # [B, plan_dim] initial plan logits (requires_grad=True)
    theta: CostModel,
    psi: CostModel,
    psi_target: CostModel,
    hyper: LearnableHyper,
    opt_outer: torch.optim.Optimizer,  # updates theta + hyper (NOT psi)
    opt_psi: torch.optim.Optimizer,    # updates psi
    cfg: TrainConfig,
) -> Dict[str, float]:
    """
    One end-to-end step:
      1) unroll inner search using theta + hyper
      2) compute outer loss using psi_target (frozen) on final soft plan
      3) execute discrete plan -> true cost
      4) update psi supervised on true cost
      5) update theta+hyper by meta loss + supervised anchor (to prevent collapse)
      6) EMA-update psi_target from psi
    """
    B = query_feat.shape[0]
    assert z0.shape[0] == B, "Batch size mismatch."

    # ----- (A) Inner unroll (differentiable) -----
    # z0 must require grad because we differentiate through z updates.
    if not z0.requires_grad:
        z0 = z0.detach().requires_grad_(True)

    unroll = inner_unroll(
        z0=z0,
        query_feat=query_feat,
        theta_model=theta,
        hyper=hyper,
        T=cfg.T_inner,
        add_noise_std=0.0,
    )
    plan_soft_T = unroll.plan_soft_T  # in-graph

    # ----- (B) Outer loss (teacher / target) -----
    # Freeze psi_target parameters: we want grad wrt plan_soft_T -> z -> theta/hyper, but not wrt psi_target params.
    freeze_(psi_target)
    outer_pred = psi_target(plan_soft_T, query_feat)   # [B]
    L_outer = outer_pred.mean()                        # scalar

    # ----- (C) Execute discrete plan for true cost (no grads) -----
    with torch.no_grad():
        plan_disc = project_to_discrete_plan(plan_soft_T)   # [B, plan_dim]
        c_true = execute_plan(plan_disc, query_obj)         # [B] tensor
        # Ensure shape/device are sane
        c_true = c_true.to(query_feat.device).view(-1)

    # ----- (D) Update psi supervised on true costs -----
    unfreeze_(psi)
    opt_psi.zero_grad(set_to_none=True)
    # Train psi on the executed discrete plan (detached, since projection is non-diff anyway)
    psi_pred = psi(plan_disc.detach(), query_feat)          # [B]
    L_psi = F.mse_loss(psi_pred, c_true)
    L_psi.backward()
    opt_psi.step()

    # ----- (E) Update theta + hyper from (outer meta) + (supervised anchor) -----
    # Supervised anchor for theta to prevent collapse:
    # Evaluate theta on discrete plan (detached plan), regress to c_true.
    theta_pred_sup = theta(plan_disc.detach(), query_feat)  # [B], grads only to theta
    L_sup_theta = F.mse_loss(theta_pred_sup, c_true)

    L_total_outer = cfg.gamma_outer * L_outer + cfg.beta_sup_theta * L_sup_theta

    opt_outer.zero_grad(set_to_none=True)
    # This backward carries "backprop through unroll":
    # L_outer -> plan_soft_T -> z_T -> ... -> theta params / hyper params (via create_graph=True in inner_unroll).
    L_total_outer.backward()

    if cfg.grad_clip is not None and cfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(list(theta.parameters()) + list(hyper.parameters()), cfg.grad_clip)

    opt_outer.step()

    # ----- (F) EMA update psi_target from psi -----
    ema_update_(psi_target, psi, decay=cfg.ema_decay_psi)

    return {
        "L_outer": float(L_outer.detach().cpu()),
        "L_sup_theta": float(L_sup_theta.detach().cpu()),
        "L_psi": float(L_psi.detach().cpu()),
        "lambda": float(hyper.lam().detach().cpu()),
        "eta": float(hyper.eta().detach().cpu()),
        "tau": float(hyper.tau().detach().cpu()),
        "true_cost_mean": float(c_true.mean().detach().cpu()),
        "outer_pred_mean": float(outer_pred.mean().detach().cpu()),
    }


# ----------------------------
# 7) Example usage (wiring)
# ----------------------------
def make_optimizers(theta: nn.Module, psi: nn.Module, hyper: nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    # Outer optimizer updates theta + hyper only
    opt_outer = torch.optim.AdamW(
        list(theta.parameters()) + list(hyper.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    # Teacher optimizer updates psi only
    opt_psi = torch.optim.AdamW(psi.parameters(), lr=1e-4, weight_decay=1e-4)
    return opt_outer, opt_psi

