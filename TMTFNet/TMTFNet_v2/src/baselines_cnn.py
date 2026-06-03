"""1D-CNN and DeepConvLSTM HAR baselines.

Adds two HAR architectures that Reviewer 2 explicitly asked for:

* ``CNN1D`` -- a standard 1D ConvNet (three blocks of Conv-BN-ReLU-Pool
  followed by global average pooling and a linear head). This is the
  Ignatov (2018) recipe, a strong HAR baseline.
* ``DeepConvLSTM`` -- Ordonez and Roggen (2016): four 1D convolutional
  feature extractors followed by two stacked LSTM layers. The canonical
  deep HAR architecture against which every modern method is benchmarked.

Both follow the ``(modality_list, aux) -> (logits, aux)`` signature of the
other baselines in ``models.py`` so the same trainer can be used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import ClassificationHead, ForecastHead, _concat_modalities


class CNN1D(nn.Module):
    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        total_dim = sum(modality_dims)
        self.n_classes = n_classes
        self.pred_len = pred_len

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.features = nn.Sequential(
            block(total_dim, 64),
            nn.MaxPool1d(2),
            block(64, 128),
            nn.MaxPool1d(2),
            block(128, d_model),
        )
        if n_classes > 0:
            self.head = ClassificationHead(d_model, d_model, n_classes, dropout=dropout)
        elif pred_len > 0:
            self.head = ForecastHead(d_model, d_model, pred_len, dropout=dropout)
        else:
            raise ValueError("Either n_classes or pred_len must be positive.")

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        feat = self.features(x).mean(dim=-1)
        return self.head(feat), {"representation": feat.detach()}


class DeepConvLSTM(nn.Module):
    """Ordonez & Roggen (2016) DeepConvLSTM."""

    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        n_conv_filters=64,
        n_lstm_layers=2,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        total_dim = sum(modality_dims)
        self.n_classes = n_classes
        self.pred_len = pred_len

        conv_layers = []
        in_ch = total_dim
        for _ in range(4):
            conv_layers.append(nn.Conv1d(in_ch, n_conv_filters, kernel_size=5, padding=2))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(dropout))
            in_ch = n_conv_filters
        self.conv = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(
            input_size=n_conv_filters,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        if n_classes > 0:
            self.head = ClassificationHead(d_model, d_model, n_classes, dropout=dropout)
        elif pred_len > 0:
            self.head = ForecastHead(d_model, d_model, pred_len, dropout=dropout)
        else:
            raise ValueError("Either n_classes or pred_len must be positive.")

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        out, _ = self.lstm(x)
        feat = out[:, -1]
        return self.head(feat), {"representation": feat.detach()}


__all__ = ["CNN1D", "DeepConvLSTM"]
