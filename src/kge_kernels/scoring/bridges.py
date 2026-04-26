"""Learnable RL-symbolic + KGE score fusion modules.

A "bridge" combines two parallel score streams — typically RL log-probs
from a proof-search policy and KGE log-scores from an embedding model —
into a single ranking score per candidate. The four bridges in this
module differ in how the blend weight is parameterized:

- :class:`LinearBridge` — single learnable α.
- :class:`GatedBridge` — separate α for proven / unproven candidates.
- :class:`PerPredicateBridge` — α per predicate.
- :class:`MLPBridge` — full MLP over ``(rl, kge)`` features.

All four expose the same ``forward(rl_logprobs, kge_logprobs, ...)``
contract returning ``[B, K]`` fused scores. Use :class:`NeuralBridgeTrainer`
to fit the bridge on a held-out validation set against an MRR-style
ranking objective.

Dataset-specific factories (e.g. predicate-type masks built from rule
heuristics) live in the consumer repo, not here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..losses.ranking_losses import differentiable_mrr_loss, pairwise_ranking_loss


class LinearBridge(nn.Module):
    """Linear combination of RL and KGE log-probs.

    Formula: ``score = sigmoid(α) * rl + (1 - sigmoid(α)) * kge``.
    """

    def __init__(
        self,
        init_alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # sigmoid(α) = init_alpha => α = logit(init_alpha)
        init_value = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
        self.alpha = nn.Parameter(init_value)
        if device is not None:
            self.to(device)

    @property
    def effective_alpha(self) -> float:
        """Effective α after sigmoid (the actual RL weight)."""
        return torch.sigmoid(self.alpha).item()

    def forward(
        self,
        rl_logprobs: Tensor,    # [B, K]
        kge_logprobs: Tensor,   # [B, K]
        success_mask: Optional[Tensor] = None,  # unused, kept for API compat
    ) -> Tensor:
        alpha = torch.sigmoid(self.alpha)
        return alpha * rl_logprobs + (1 - alpha) * kge_logprobs

    def __repr__(self) -> str:
        return f"LinearBridge(alpha={self.effective_alpha:.4f})"


class GatedBridge(nn.Module):
    """Separate-α blend gated on a proof-success mask.

    Successful proofs (RL signal reliable) and failed proofs (lean on KGE)
    use different α values, both learned.
    """

    def __init__(
        self,
        init_alpha_success: float = 0.7,
        init_alpha_fail: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.alpha_success = nn.Parameter(
            torch.logit(torch.tensor(init_alpha_success, dtype=torch.float32))
        )
        self.alpha_fail = nn.Parameter(
            torch.logit(torch.tensor(init_alpha_fail, dtype=torch.float32))
        )
        if device is not None:
            self.to(device)

    @property
    def effective_alpha_success(self) -> float:
        return torch.sigmoid(self.alpha_success).item()

    @property
    def effective_alpha_fail(self) -> float:
        return torch.sigmoid(self.alpha_fail).item()

    @property
    def effective_alpha(self) -> float:
        """Average effective α (compatibility with single-α bridges)."""
        return (self.effective_alpha_success + self.effective_alpha_fail) / 2

    def forward(
        self,
        rl_logprobs: Tensor,    # [B, K]
        kge_logprobs: Tensor,   # [B, K]
        success_mask: Optional[Tensor] = None,  # [B, K]
    ) -> Tensor:
        alpha_s = torch.sigmoid(self.alpha_success)
        alpha_f = torch.sigmoid(self.alpha_fail)
        scores_success = alpha_s * rl_logprobs + (1 - alpha_s) * kge_logprobs
        scores_fail = alpha_f * rl_logprobs + (1 - alpha_f) * kge_logprobs
        if success_mask is not None:
            return torch.where(success_mask, scores_success, scores_fail)
        return (scores_success + scores_fail) / 2

    def __repr__(self) -> str:
        return (
            f"GatedBridge(alpha_success={self.effective_alpha_success:.4f}, "
            f"alpha_fail={self.effective_alpha_fail:.4f})"
        )


class PerPredicateBridge(nn.Module):
    """One learnable α per predicate.

    ``score = sigmoid(α[pred]) * rl + (1 - sigmoid(α[pred])) * kge``.
    Falls back to the mean α when ``pred_indices`` is not provided.
    """

    def __init__(
        self,
        n_predicates: int,
        init_alpha: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.n_predicates = n_predicates
        init_value = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
        self.alphas = nn.Parameter(torch.full((n_predicates,), init_value.item()))
        if device is not None:
            self.to(device)

    def get_effective_alphas(self) -> Tensor:
        """Per-predicate α after sigmoid."""
        return torch.sigmoid(self.alphas)

    def forward(
        self,
        rl_logprobs: Tensor,    # [B, K]
        kge_logprobs: Tensor,   # [B, K]
        pred_indices: Optional[Tensor] = None,  # [B, K]
        success_mask: Optional[Tensor] = None,  # unused
    ) -> Tensor:
        if pred_indices is None:
            alpha = torch.sigmoid(self.alphas.mean())
            return alpha * rl_logprobs + (1 - alpha) * kge_logprobs
        alpha = torch.sigmoid(self.alphas[pred_indices])
        return alpha * rl_logprobs + (1 - alpha) * kge_logprobs

    def __repr__(self) -> str:
        alphas = self.get_effective_alphas().detach().cpu().numpy()
        return (
            f"PerPredicateBridge(n_predicates={self.n_predicates}, "
            f"alphas_range=[{alphas.min():.3f}, {alphas.max():.3f}])"
        )


class MLPBridge(nn.Module):
    """MLP fusion of ``(rl_logprobs, kge_logprobs)`` features.

    More expressive than the α-blend bridges — can learn non-monotonic
    combinations — but easier to overfit. Prefer for large validation
    sets.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if device is not None:
            self.to(device)

    def forward(
        self,
        rl_logprobs: Tensor,    # [B, K]
        kge_logprobs: Tensor,   # [B, K]
        success_mask: Optional[Tensor] = None,  # unused
    ) -> Tensor:
        B, K = rl_logprobs.shape
        features = torch.stack([rl_logprobs, kge_logprobs], dim=-1)  # [B, K, 2]
        scores_flat = self.mlp(features.view(-1, 2))  # [B*K, 1]
        return scores_flat.view(B, K)


class NeuralBridgeTrainer:
    """Train a bridge module on collected validation data.

    Workflow:

    1. During an evaluation pass, accumulate per-batch
       ``(rl_logprobs, kge_logprobs, success_mask, target_idx)`` via
       :meth:`add_validation_batch`.
    2. Call :meth:`train` once to fit the bridge against an MRR-style
       (default) or pairwise-margin ranking loss.
    3. The fitted bridge can then be used to score candidates during the
       next eval pass.

    The trainer is loss-/bridge-agnostic; any ``nn.Module`` with a
    ``forward(rl, kge, success_mask)`` signature works.
    """

    def __init__(
        self,
        bridge: nn.Module,
        lr: float = 0.01,
        epochs: int = 100,
        loss_type: str = "mrr",  # "mrr" or "pairwise"
        verbose: bool = True,
    ) -> None:
        self.bridge = bridge
        self.lr = lr
        self.epochs = epochs
        self.loss_type = loss_type
        self.verbose = verbose
        self.optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)
        self.train_data: List[Dict[str, Tensor]] = []

    def add_validation_batch(
        self,
        rl_logprobs: Tensor,    # [B, K]
        kge_logprobs: Tensor,   # [B, K]
        success_mask: Tensor,   # [B, K]
        target_idx: Optional[Tensor] = None,  # [B]
    ) -> None:
        """Buffer a validation batch for the next ``train()`` call."""
        B, K = rl_logprobs.shape
        if target_idx is None:
            target_idx = torch.zeros(B, dtype=torch.long, device=rl_logprobs.device)
        self.train_data.append({
            "rl_logprobs": rl_logprobs.detach(),
            "kge_logprobs": kge_logprobs.detach(),
            "success_mask": success_mask.detach(),
            "target_idx": target_idx.detach(),
        })

    def train(self) -> Dict[str, float]:
        """Fit the bridge against the buffered validation data.

        Returns a dict with the final loss and (when available) the
        bridge's ``effective_alpha`` for telemetry.
        """
        if not self.train_data:
            print("[NeuralBridge] No validation data collected, skipping training")
            return {
                "loss": 0.0,
                "alpha": getattr(self.bridge, "effective_alpha", 0.5),
            }

        all_rl = torch.cat([d["rl_logprobs"] for d in self.train_data], dim=0)
        all_kge = torch.cat([d["kge_logprobs"] for d in self.train_data], dim=0)
        all_success = torch.cat([d["success_mask"] for d in self.train_data], dim=0)
        all_target = torch.cat([d["target_idx"] for d in self.train_data], dim=0)

        if self.verbose:
            print(
                f"[NeuralBridge] Training on {all_rl.shape[0]} queries, "
                f"{all_rl.shape[1]} candidates each"
            )

        self.bridge.train()
        final_loss = 0.0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            scores = self.bridge(all_rl, all_kge, all_success)
            if self.loss_type == "mrr":
                loss = differentiable_mrr_loss(scores, all_target)
            else:
                loss = pairwise_ranking_loss(scores, all_target)
            loss.backward()
            self.optimizer.step()
            final_loss = loss.item()
            if self.verbose and (epoch + 1) % 10 == 0:
                alpha_str = (
                    f", alpha={self.bridge.effective_alpha:.4f}"
                    if hasattr(self.bridge, "effective_alpha")
                    else ""
                )
                print(f"  Epoch {epoch + 1}/{self.epochs}: loss={final_loss:.4f}{alpha_str}")
        self.bridge.eval()

        result = {"loss": final_loss}
        if hasattr(self.bridge, "effective_alpha"):
            result["alpha"] = self.bridge.effective_alpha
        return result

    def clear_data(self) -> None:
        """Drop the buffered validation batches."""
        self.train_data.clear()


__all__ = [
    "GatedBridge",
    "LinearBridge",
    "MLPBridge",
    "NeuralBridgeTrainer",
    "PerPredicateBridge",
]
