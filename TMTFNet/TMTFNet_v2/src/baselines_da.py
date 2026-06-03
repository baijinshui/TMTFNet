"""Time-series domain-adaptation baselines.

Reviewer 1 required direct comparison with adaptation methods that are cited
in Related Work. We implement four representative families, all on top of a
shared 1D-CNN feature extractor so every method is compared on its adaptation
strategy rather than on backbone capacity:

* ``DANN``      -- Ganin and Lempitsky (ICML 2015). Source CE + gradient-
                   reversed domain classifier.
* ``CoDATS``    -- Wilson et al. (KDD 2020). DANN over a stronger CNN
                   encoder with label-smoothed source loss. Adapted to match
                   parameter budget here.
* ``AdvSKM``    -- Liu and Xue (IJCAI 2021). Adversarial spectral-kernel
                   matching. We implement its core adversarial + MMD-on-
                   random-features objective.
* ``RAINCOAT``  -- He et al. (ICML 2023). Time + frequency fusion with a
                   Sinkhorn/MMD alignment. We include a lightweight
                   reproduction (time+frequency CNN + CORAL alignment).

Each class exposes the same ``(modalities, aux) -> (logits, aux)`` signature,
plus ``aux["shared_feature"]`` and ``aux["domain_logits"]`` when applicable,
so they plug into ``DomainAdaptTrainer`` without extra plumbing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import DomainClassifier, gradient_reverse
from .models import ClassificationHead, ForecastHead, _concat_modalities


class _CNN1DBackbone(nn.Module):
    def __init__(self, in_channels, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x).mean(dim=-1)


class _DAHeadMixin(nn.Module):
    def __init__(self, d_model, n_classes, pred_len, dropout):
        super().__init__()
        if n_classes > 0:
            self.head = ClassificationHead(d_model, d_model, n_classes, dropout=dropout)
        elif pred_len > 0:
            self.head = ForecastHead(d_model, d_model, pred_len, dropout=dropout)
        else:
            raise ValueError("Either n_classes or pred_len must be positive.")


class DANN(_DAHeadMixin):
    def __init__(self, modality_dims, d_model=128, n_classes=0, pred_len=0, dropout=0.1, n_domains=2, **kwargs):
        super().__init__(d_model, n_classes, pred_len, dropout)
        del kwargs
        self.backbone = _CNN1DBackbone(sum(modality_dims), d_model, dropout=dropout)
        self.domain_classifier = DomainClassifier(d_model, n_domains=n_domains, dropout=dropout)

    def forward(self, modality_inputs, domain_alpha=1.0, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        feat = self.backbone(x)
        logits = self.head(feat)
        aux = {
            "representation": feat.detach(),
            "shared_feature": feat,
            "domain_logits": self.domain_classifier(gradient_reverse(feat, domain_alpha)),
        }
        return logits, aux


class CoDATS(DANN):
    """Wilson et al. (KDD 2020). Stronger CNN encoder; same adversarial head."""

    def __init__(self, modality_dims, d_model=128, n_classes=0, pred_len=0, dropout=0.2, n_domains=2, **kwargs):
        super().__init__(
            modality_dims=modality_dims,
            d_model=d_model,
            n_classes=n_classes,
            pred_len=pred_len,
            dropout=dropout,
            n_domains=n_domains,
            **kwargs,
        )
        in_ch = sum(modality_dims)
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, modality_inputs, domain_alpha=1.0, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        feat = self.backbone(x)
        logits = self.head(feat)
        aux = {
            "representation": feat.detach(),
            "shared_feature": feat,
            "domain_logits": self.domain_classifier(gradient_reverse(feat, domain_alpha)),
        }
        return logits, aux


class AdvSKM(_DAHeadMixin):
    """Adversarial spectral kernel matching (Liu & Xue, IJCAI 2021).

    We use a random-features kernel head as the critic; minimising the
    critic-reported MMD under the kernel that *maximises* it yields
    distribution alignment in an adversarial fashion.
    """

    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        n_random_features=64,
        **kwargs,
    ):
        super().__init__(d_model, n_classes, pred_len, dropout)
        del kwargs
        self.backbone = _CNN1DBackbone(sum(modality_dims), d_model, dropout=dropout)
        self.random_feature = nn.Linear(d_model, n_random_features, bias=True)

    def forward(self, modality_inputs, domain_alpha=1.0, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        feat = self.backbone(x)
        logits = self.head(feat)
        phi = torch.cos(self.random_feature(feat))
        aux = {
            "representation": feat.detach(),
            "shared_feature": feat,
            "spectral_features": phi,
            "domain_alpha": float(domain_alpha),
        }
        return logits, aux


class RAINCOAT(_DAHeadMixin):
    """Time+frequency dual-branch encoder (He et al., ICML 2023)."""

    def __init__(self, modality_dims, d_model=128, n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        super().__init__(d_model, n_classes, pred_len, dropout)
        del kwargs
        in_ch = sum(modality_dims)
        self.time_cnn = _CNN1DBackbone(in_ch, d_model // 2, dropout=dropout)
        self.freq_cnn = _CNN1DBackbone(in_ch * 2, d_model // 2, dropout=dropout)
        self.fuse = nn.Linear(d_model, d_model)

    def _freq_input(self, x):
        spec = torch.fft.rfft(x, dim=-1)
        return torch.cat([spec.real, spec.imag], dim=1)

    def forward(self, modality_inputs, domain_alpha=1.0, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        time_feat = self.time_cnn(x)
        freq_feat = self.freq_cnn(self._freq_input(x))
        feat = F.gelu(self.fuse(torch.cat([time_feat, freq_feat], dim=-1)))
        logits = self.head(feat)
        aux = {
            "representation": feat.detach(),
            "shared_feature": feat,
            "domain_alpha": float(domain_alpha),
        }
        return logits, aux


DA_MODEL_REGISTRY = {
    "DANN": DANN,
    "CoDATS": CoDATS,
    "AdvSKM": AdvSKM,
    "RAINCOAT": RAINCOAT,
}


__all__ = ["DANN", "CoDATS", "AdvSKM", "RAINCOAT", "DA_MODEL_REGISTRY"]
