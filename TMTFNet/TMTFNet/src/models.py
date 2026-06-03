"""
TMTFNet: Transformer-Based Multi-Modal Temporal Fusion Network
==============================================================
Core model architecture and baseline implementations.

Components:
  1. Modality-Specific Temporal Encoder (MSTE)
  2. Cross-Modal Temporal Attention (CMTA)
  3. Adaptive Modality Gating (AMG)
  4. Hierarchical Temporal Fusion (HTF)
  5. Domain-Adaptive Contrastive Alignment (DACA)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================
# Building Blocks
# ============================================================

class PositionalEncoding(nn.Module):
    """Learnable + sinusoidal hybrid positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]
        return self.dropout(x)


class ModalitySpecificEncoder(nn.Module):
    """Lightweight transformer encoder for a single modality."""
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.norm_in = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm_in(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.norm_out(x)


class CrossModalTemporalAttention(nn.Module):
    """
    Cross-Modal Temporal Attention (CMTA):
    Computes attention between modalities at each temporal position.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_query, h_key):
        """
        h_query: [B, T, D] - query modality
        h_key:   [B, T, D] - key/value modality
        """
        B, T, D = h_query.shape
        residual = h_query

        Q = self.W_q(h_query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(h_key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(h_key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.W_o(attn_out)

        x = self.norm(residual + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class AdaptiveModalityGating(nn.Module):
    """Dynamic gating mechanism to weight modality contributions."""
    def __init__(self, d_model, n_modalities):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * n_modalities, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, n_modalities),
            nn.Softmax(dim=-1)
        )
        self.n_modalities = n_modalities

    def forward(self, modality_hiddens):
        """
        modality_hiddens: list of [B, T, D] tensors
        Returns: fused [B, T, D], gate_weights [B, T, M]
        """
        concat = torch.cat(modality_hiddens, dim=-1)  # [B, T, D*M]
        gate_weights = self.gate_net(concat)  # [B, T, M]

        fused = torch.zeros_like(modality_hiddens[0])
        for i, h in enumerate(modality_hiddens):
            fused = fused + gate_weights[:, :, i:i+1] * h

        return fused, gate_weights


class HierarchicalTemporalFusion(nn.Module):
    """Multi-scale temporal fusion with local and global contexts."""
    def __init__(self, d_model, n_scales=3, dropout=0.1):
        super().__init__()
        self.n_scales = n_scales
        # Keep odd kernel sizes so padding can preserve temporal length exactly.
        self.scale_kernel_sizes = [2 * i + 3 for i in range(n_scales)]  # 3, 5, 7, ...
        self.local_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model)
            for k in self.scale_kernel_sizes
        ])
        self.scale_attention = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_scales),
            nn.Softmax(dim=-1)
        )
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, T, D]"""
        B, T, D = x.shape
        x_t = x.transpose(1, 2)  # [B, D, T]

        multi_scale = []
        for conv in self.local_convs:
            out = conv(x_t)
            if out.size(2) != T:
                # Safety fallback for unexpected shape drift (e.g., custom kernels).
                out = out[:, :, :T]
            multi_scale.append(out.transpose(1, 2))  # [B, T, D]

        concat = torch.cat(multi_scale, dim=-1)  # [B, T, D*S]
        scale_w = self.scale_attention(concat)  # [B, T, S]

        fused = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        for i, s in enumerate(multi_scale):
            fused = fused + scale_w[:, :, i:i+1] * s

        return self.norm(x + self.dropout(self.output_proj(fused)))


class GradientReversal(Function):
    """Gradient reversal layer for domain adaptation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainAdaptiveLayer(nn.Module):
    """Domain classifier with gradient reversal for domain-invariant features."""
    def __init__(self, d_model, n_domains=2):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_domains)
        )

    def forward(self, x, alpha=1.0):
        """x: [B, D] pooled representation"""
        reversed_x = GradientReversal.apply(x, alpha)
        domain_pred = self.domain_classifier(reversed_x)
        return domain_pred


# ============================================================
# TMTFNet Main Model
# ============================================================

class TMTFNet(nn.Module):
    """
    Transformer-Based Multi-Modal Temporal Fusion Network.

    Args:
        modality_dims: list of input dimensions per modality
        d_model: hidden dimension
        n_heads: number of attention heads
        n_enc_layers: encoder layers per modality
        n_classes: number of output classes (0 for regression)
        pred_len: prediction length for forecasting (0 for classification)
        dropout: dropout rate
        use_domain_adapt: enable domain adaptation
        n_domains: number of domains
    """
    def __init__(self, modality_dims, d_model=64, n_heads=4, n_enc_layers=2,
                 n_classes=0, pred_len=0, dropout=0.1, use_domain_adapt=False,
                 n_domains=2, max_len=512):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.n_classes = n_classes
        self.pred_len = pred_len
        self.d_model = d_model
        self.use_domain_adapt = use_domain_adapt

        # 1. Modality-Specific Encoders
        self.encoders = nn.ModuleList([
            ModalitySpecificEncoder(dim, d_model, n_heads, n_enc_layers, dropout, max_len)
            for dim in modality_dims
        ])

        # 2. Cross-Modal Temporal Attention (bidirectional for each pair)
        self.cmta_layers = nn.ModuleList()
        for i in range(self.n_modalities):
            row = nn.ModuleList()
            for j in range(self.n_modalities):
                if i != j:
                    row.append(CrossModalTemporalAttention(d_model, n_heads, dropout))
                else:
                    row.append(None)
            self.cmta_layers.append(row)

        # 3. Adaptive Modality Gating
        self.amg = AdaptiveModalityGating(d_model, self.n_modalities)

        # 4. Hierarchical Temporal Fusion
        self.htf = HierarchicalTemporalFusion(d_model, n_scales=3, dropout=dropout)

        # 5. Domain-Adaptive Layer (optional)
        if use_domain_adapt:
            self.domain_layer = DomainAdaptiveLayer(d_model, n_domains)

        # 6. Task Head
        if n_classes > 0:  # Classification
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes)
            )
        elif pred_len > 0:  # Forecasting
            self.forecaster = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, pred_len)
            )

    def forward(self, modality_inputs, domain_alpha=1.0):
        """
        modality_inputs: list of [B, T, d_i] tensors
        Returns: predictions, aux_outputs dict
        """
        aux = {}

        # Step 1: Encode each modality
        encoded = [enc(x) for enc, x in zip(self.encoders, modality_inputs)]

        # Step 2: Cross-Modal Temporal Attention
        cross_attended = []
        all_attn_weights = []
        for i in range(self.n_modalities):
            cross_out = encoded[i]
            for j in range(self.n_modalities):
                if i != j:
                    attended, attn_w = self.cmta_layers[i][j](cross_out, encoded[j])
                    cross_out = cross_out + attended
                    all_attn_weights.append(attn_w)
            cross_attended.append(cross_out)
        aux['cross_attn_weights'] = all_attn_weights

        # Step 3: Adaptive Modality Gating
        fused, gate_weights = self.amg(cross_attended)
        aux['gate_weights'] = gate_weights

        # Step 4: Hierarchical Temporal Fusion
        fused = self.htf(fused)

        # Step 5: Domain adaptation (if enabled)
        if self.use_domain_adapt:
            pooled = fused.mean(dim=1)
            domain_pred = self.domain_layer(pooled, domain_alpha)
            aux['domain_pred'] = domain_pred
        
        # Step 6: Task prediction
        if self.n_classes > 0:
            pooled = fused.mean(dim=1)
            output = self.classifier(pooled)
        elif self.pred_len > 0:
            pooled = fused.mean(dim=1)
            output = self.forecaster(pooled)
        else:
            output = fused

        aux['representation'] = fused.mean(dim=1).detach()
        return output, aux


# ============================================================
# Ablation Variants
# ============================================================

class TMTFNet_NoCMTA(TMTFNet):
    """Ablation: without Cross-Modal Temporal Attention."""
    def forward(self, modality_inputs, domain_alpha=1.0):
        aux = {}
        encoded = [enc(x) for enc, x in zip(self.encoders, modality_inputs)]
        # Skip CMTA, go directly to gating
        fused, gate_weights = self.amg(encoded)
        aux['gate_weights'] = gate_weights
        fused = self.htf(fused)
        if self.n_classes > 0:
            output = self.classifier(fused.mean(dim=1))
        elif self.pred_len > 0:
            output = self.forecaster(fused.mean(dim=1))
        else:
            output = fused
        aux['representation'] = fused.mean(dim=1).detach()
        return output, aux


class TMTFNet_NoAMG(TMTFNet):
    """Ablation: without Adaptive Modality Gating (use mean fusion)."""
    def forward(self, modality_inputs, domain_alpha=1.0):
        aux = {}
        encoded = [enc(x) for enc, x in zip(self.encoders, modality_inputs)]
        cross_attended = []
        for i in range(self.n_modalities):
            cross_out = encoded[i]
            for j in range(self.n_modalities):
                if i != j:
                    attended, _ = self.cmta_layers[i][j](cross_out, encoded[j])
                    cross_out = cross_out + attended
            cross_attended.append(cross_out)
        # Mean fusion instead of AMG
        fused = torch.stack(cross_attended, dim=0).mean(dim=0)
        fused = self.htf(fused)
        if self.n_classes > 0:
            output = self.classifier(fused.mean(dim=1))
        elif self.pred_len > 0:
            output = self.forecaster(fused.mean(dim=1))
        else:
            output = fused
        aux['representation'] = fused.mean(dim=1).detach()
        return output, aux


class TMTFNet_NoHTF(TMTFNet):
    """Ablation: without Hierarchical Temporal Fusion."""
    def forward(self, modality_inputs, domain_alpha=1.0):
        aux = {}
        encoded = [enc(x) for enc, x in zip(self.encoders, modality_inputs)]
        cross_attended = []
        for i in range(self.n_modalities):
            cross_out = encoded[i]
            for j in range(self.n_modalities):
                if i != j:
                    attended, _ = self.cmta_layers[i][j](cross_out, encoded[j])
                    cross_out = cross_out + attended
            cross_attended.append(cross_out)
        fused, gate_weights = self.amg(cross_attended)
        aux['gate_weights'] = gate_weights
        # Skip HTF
        if self.n_classes > 0:
            output = self.classifier(fused.mean(dim=1))
        elif self.pred_len > 0:
            output = self.forecaster(fused.mean(dim=1))
        else:
            output = fused
        aux['representation'] = fused.mean(dim=1).detach()
        return output, aux


# ============================================================
# Baseline Models
# ============================================================

class LSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline with multi-modal concatenation."""
    def __init__(self, modality_dims, d_model=64, n_layers=2, n_classes=0,
                 pred_len=0, dropout=0.1, **kwargs):
        super().__init__()
        total_dim = sum(modality_dims)
        self.lstm = nn.LSTM(total_dim, d_model, n_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.n_classes = n_classes
        self.pred_len = pred_len
        out_dim = d_model * 2
        if n_classes > 0:
            self.head = nn.Sequential(nn.Linear(out_dim, d_model), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        elif pred_len > 0:
            self.head = nn.Sequential(nn.Linear(out_dim, d_model), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, modality_inputs, **kwargs):
        x = torch.cat(modality_inputs, dim=-1)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.head(pooled), {'representation': pooled.detach()}


class GRUBaseline(nn.Module):
    """Bidirectional GRU baseline."""
    def __init__(self, modality_dims, d_model=64, n_layers=2, n_classes=0,
                 pred_len=0, dropout=0.1, **kwargs):
        super().__init__()
        total_dim = sum(modality_dims)
        self.gru = nn.GRU(total_dim, d_model, n_layers, batch_first=True,
                         bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.n_classes = n_classes
        self.pred_len = pred_len
        out_dim = d_model * 2
        if n_classes > 0:
            self.head = nn.Sequential(nn.Linear(out_dim, d_model), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        elif pred_len > 0:
            self.head = nn.Sequential(nn.Linear(out_dim, d_model), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, modality_inputs, **kwargs):
        x = torch.cat(modality_inputs, dim=-1)
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled), {'representation': pooled.detach()}


class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.norm1(self.conv1(x)[:, :, :x.size(2)])))
        out = self.dropout(self.relu(self.norm2(self.conv2(out)[:, :, :x.size(2)])))
        if self.downsample:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNBaseline(nn.Module):
    """Temporal Convolutional Network baseline."""
    def __init__(self, modality_dims, d_model=64, n_layers=4, n_classes=0,
                 pred_len=0, dropout=0.1, **kwargs):
        super().__init__()
        total_dim = sum(modality_dims)
        channels = [d_model] * n_layers
        layers = []
        in_ch = total_dim
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size=3, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.n_classes = n_classes
        self.pred_len = pred_len
        if n_classes > 0:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        elif pred_len > 0:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, modality_inputs, **kwargs):
        x = torch.cat(modality_inputs, dim=-1)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        pooled = out.mean(dim=2)
        return self.head(pooled), {'representation': pooled.detach()}


class VanillaTransformerBaseline(nn.Module):
    """Standard Transformer with concatenated modalities."""
    def __init__(self, modality_dims, d_model=64, n_heads=4, n_layers=2,
                 n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        super().__init__()
        total_dim = sum(modality_dims)
        self.input_proj = nn.Linear(total_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, 512, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.n_classes = n_classes
        self.pred_len = pred_len
        if n_classes > 0:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        elif pred_len > 0:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, modality_inputs, **kwargs):
        x = torch.cat(modality_inputs, dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.head(pooled), {'representation': pooled.detach()}


class CrossformerBaseline(nn.Module):
    """Simplified Crossformer: separate encoders + cross-attention fusion."""
    def __init__(self, modality_dims, d_model=64, n_heads=4, n_layers=2,
                 n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([
            ModalitySpecificEncoder(dim, d_model, n_heads, n_layers, dropout)
            for dim in modality_dims
        ])
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.n_classes = n_classes
        self.pred_len = pred_len
        if n_classes > 0:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        elif pred_len > 0:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_model, pred_len))

    def forward(self, modality_inputs, **kwargs):
        encoded = [enc(x) for enc, x in zip(self.encoders, modality_inputs)]
        concat_kv = torch.cat(encoded, dim=1)  # [B, T*M, D]
        query = encoded[0]
        out, _ = self.cross_attn(query, concat_kv, concat_kv)
        out = self.norm(out + query)
        pooled = out.mean(dim=1)
        return self.head(pooled), {'representation': pooled.detach()}


# ============================================================
# Model Registry
# ============================================================

MODEL_REGISTRY = {
    'TMTFNet': TMTFNet,
    'TMTFNet_NoCMTA': TMTFNet_NoCMTA,
    'TMTFNet_NoAMG': TMTFNet_NoAMG,
    'TMTFNet_NoHTF': TMTFNet_NoHTF,
    'LSTM': LSTMBaseline,
    'GRU': GRUBaseline,
    'TCN': TCNBaseline,
    'Transformer': VanillaTransformerBaseline,
    'Crossformer': CrossformerBaseline,
}


def build_model(model_name, **kwargs):
    """Factory function to build models."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)
