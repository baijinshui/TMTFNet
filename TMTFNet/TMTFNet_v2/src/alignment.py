"""Domain-adaptive alignment losses for TMTFNet.

Three complementary mechanisms that jointly back the "domain-adaptive alignment"
term in the manuscript:

1. ``coral_loss``         -- second-order feature alignment (Sun and Saenko, 2016).
2. ``ModalityConsistency`` -- cross-modal pooled-feature agreement inside a single
   batch; regularizes against domain-specific inter-modality correlations.
3. ``GradientReversalLayer`` + ``DomainClassifier`` -- optional DANN-style adversarial
   domain invariance (Ganin and Lempitsky, 2015).

All three are stateless / weight-light and plug on top of ``aux['representation']``
and ``aux['modality_features']`` produced by TMTFNet_v2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_loss(source_feat: torch.Tensor, target_feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Deep CORAL (Sun & Saenko, ECCV 2016) between two feature batches.

    Both tensors must be shape ``(B, D)``. Returns a scalar tensor.
    """
    if source_feat.ndim != 2 or target_feat.ndim != 2:
        raise ValueError("coral_loss expects 2-D features (B, D).")
    d = source_feat.size(1)
    ns = max(source_feat.size(0), 2)
    nt = max(target_feat.size(0), 2)

    sf = source_feat - source_feat.mean(dim=0, keepdim=True)
    tf = target_feat - target_feat.mean(dim=0, keepdim=True)
    cs = (sf.t() @ sf) / (ns - 1) + eps * torch.eye(d, device=sf.device, dtype=sf.dtype)
    ct = (tf.t() @ tf) / (nt - 1) + eps * torch.eye(d, device=tf.device, dtype=tf.dtype)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


class ModalityConsistency(nn.Module):
    """Cosine-similarity agreement between every pair of modality pooled features.

    Encourages each modality branch to encode the *same* underlying temporal
    phenomenon after fusion, which removes domain-specific inter-modality
    correlations that would otherwise leak across domains. The loss is
    ``1 - mean_pairwise_cosine``, so minimising it pushes modalities toward
    a shared subspace without collapsing them (each branch keeps its own
    parameters).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, modality_features: list[torch.Tensor]) -> torch.Tensor:
        if len(modality_features) < 2:
            return torch.zeros((), device=modality_features[0].device) if modality_features else torch.zeros(())
        normed = [F.normalize(m, dim=-1) for m in modality_features]
        pairs = []
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                pairs.append((normed[i] * normed[j]).sum(dim=-1))
        stacked = torch.stack(pairs, dim=0)
        agreement = stacked.mean() if self.reduction == "mean" else stacked.sum()
        return 1.0 - agreement


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Identity on forward, sign-flipped (and scaled) gradient on backward."""
    return _GradReverse.apply(x, alpha)


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reverse(x, self.alpha)


class DomainClassifier(nn.Module):
    """Small MLP that predicts a domain label from a pooled feature."""

    def __init__(self, d_model: int, n_domains: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_domains),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def dann_schedule(progress: float, gamma: float = 10.0) -> float:
    """The canonical DANN lambda(p) = 2/(1+exp(-gamma*p)) - 1 schedule."""
    progress = max(0.0, min(1.0, float(progress)))
    return 2.0 / (1.0 + float(torch.exp(torch.tensor(-gamma * progress)).item())) - 1.0


__all__ = [
    "coral_loss",
    "ModalityConsistency",
    "GradientReversalLayer",
    "gradient_reverse",
    "DomainClassifier",
    "dann_schedule",
]
