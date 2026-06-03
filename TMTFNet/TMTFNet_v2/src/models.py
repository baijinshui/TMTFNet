import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import DomainClassifier, gradient_reverse


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]
        return self.dropout(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ModalitySpecificEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.output_norm(x)


class GatedCrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_query, h_key):
        h_cross, attn_weights = self.attn(h_query, h_key, h_key, need_weights=False)
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_query, h_cross], dim=-1)))
        mixed = gate * h_cross + (1.0 - gate) * h_query
        mixed = self.norm1(mixed)
        out = self.norm2(mixed + self.ffn(mixed))
        return out, attn_weights


class AdaptiveModalityGating(nn.Module):
    def __init__(self, d_model, n_modalities):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * n_modalities, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, n_modalities),
            nn.Softmax(dim=-1),
        )
        self.n_modalities = n_modalities

    def forward(self, modality_hiddens):
        concat = torch.cat(modality_hiddens, dim=-1)
        gate_weights = self.gate_net(concat)
        fused = torch.zeros_like(modality_hiddens[0])
        for idx, hidden in enumerate(modality_hiddens):
            fused = fused + gate_weights[:, :, idx:idx + 1] * hidden
        return fused, gate_weights


class AdaptiveHierarchicalTemporalFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1, n_scales=3):
        super().__init__()
        kernels = [3 + 2 * idx for idx in range(n_scales)]
        self.depthwise_convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model) for k in kernels]
        )
        self.scale_attention = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_scales),
            nn.Softmax(dim=-1),
        )
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_t = x.transpose(1, 2)
        multi_scale = [conv(x_t).transpose(1, 2) for conv in self.depthwise_convs]
        concat = torch.cat(multi_scale, dim=-1)
        scale_weights = self.scale_attention(concat)
        fused = torch.zeros_like(x)
        for idx, scale in enumerate(multi_scale):
            fused = fused + scale_weights[:, :, idx:idx + 1] * scale
        htf_out = self.output_proj(fused)
        alpha = torch.sigmoid(self.alpha)
        out = self.norm(x + alpha * self.dropout(htf_out))
        return out, alpha


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x):
        scores = torch.matmul(x, self.query) / math.sqrt(x.size(-1))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class LastKAttentionPooling(nn.Module):
    def __init__(self, d_model, last_k=16):
        super().__init__()
        self.last_k = last_k
        self.pool = AttentionPooling(d_model)

    def forward(self, x):
        last_k = min(self.last_k, x.size(1))
        return self.pool(x[:, -last_k:, :])


class ClassificationHead(nn.Module):
    def __init__(self, rep_dim, hidden_dim, n_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(rep_dim),
            nn.Linear(rep_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ForecastHead(nn.Module):
    def __init__(self, rep_dim, hidden_dim, pred_len, dropout=0.1):
        super().__init__()
        self.residual = nn.Linear(rep_dim, pred_len)
        self.mlp = nn.Sequential(
            nn.LayerNorm(rep_dim),
            nn.Linear(rep_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, pred_len),
        )

    def forward(self, x):
        return self.residual(x) + self.mlp(x)


def _concat_modalities(modality_inputs):
    return torch.cat(modality_inputs, dim=-1)


class HybridPositionalEncoding(PositionalEncoding):
    pass


class TMTFNet_v2(nn.Module):
    """
    Improved TMTFNet with:
    1. modality dropout
    2. gated cross-modal attention
    3. adaptive modality gating
    4. adaptive hierarchical temporal fusion
    5. attention pooling for classification
    6. last-k attention pooling with residual MLP for forecasting
    """

    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_heads=8,
        n_enc_layers=3,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        modality_dropout=0.1,
        use_domain_adapt=False,
        n_domains=2,
        max_len=512,
        **kwargs,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.n_classes = n_classes
        self.pred_len = pred_len
        self.modality_dropout = modality_dropout
        self.use_domain_adapt = use_domain_adapt
        self.d_model = d_model

        self.encoders = nn.ModuleList(
            [ModalitySpecificEncoder(dim, d_model, n_heads, n_enc_layers, dropout=dropout, max_len=max_len) for dim in modality_dims]
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        None if i == j else GatedCrossModalAttention(d_model, n_heads, dropout=dropout)
                        for j in range(self.n_modalities)
                    ]
                )
                for i in range(self.n_modalities)
            ]
        )
        self.amg = AdaptiveModalityGating(d_model, self.n_modalities)
        self.ahtf = AdaptiveHierarchicalTemporalFusion(d_model, dropout=dropout)
        self.attn_pool = AttentionPooling(d_model)
        self.lastk_pool = LastKAttentionPooling(d_model, last_k=16)

        if n_classes > 0:
            self.classifier = ClassificationHead(d_model, d_model, n_classes, dropout=dropout)
        elif pred_len > 0:
            self.forecaster = ForecastHead(d_model, d_model, pred_len, dropout=dropout)
        else:
            raise ValueError("Either n_classes or pred_len must be positive.")

        if use_domain_adapt:
            self.domain_classifier = DomainClassifier(d_model, n_domains=n_domains, dropout=dropout)
        else:
            self.domain_classifier = None

    def _apply_modality_dropout(self, modality_inputs):
        if not self.training or self.modality_dropout <= 0.0:
            return modality_inputs
        batch_size = modality_inputs[0].size(0)
        keep_mask = torch.rand(batch_size, self.n_modalities, device=modality_inputs[0].device) > self.modality_dropout
        all_dropped = keep_mask.sum(dim=1) == 0
        if all_dropped.any():
            restore_idx = torch.randint(0, self.n_modalities, (int(all_dropped.sum().item()),), device=keep_mask.device)
            keep_mask[all_dropped] = False
            keep_mask[all_dropped, restore_idx] = True
        outputs = []
        for idx, modality in enumerate(modality_inputs):
            mask = keep_mask[:, idx].view(batch_size, 1, 1).to(modality.dtype)
            outputs.append(modality * mask)
        return outputs

    def _cross_modal_fusion(self, encoded_modalities):
        fused_modalities = []
        for i in range(self.n_modalities):
            cross_outputs = []
            for j in range(self.n_modalities):
                if i == j:
                    continue
                cross_out, _ = self.cross_attn[i][j](encoded_modalities[i], encoded_modalities[j])
                cross_outputs.append(cross_out)
            if cross_outputs:
                fused_modalities.append(torch.stack(cross_outputs, dim=0).mean(dim=0))
            else:
                fused_modalities.append(encoded_modalities[i])
        return fused_modalities

    def _classification_representation(self, sequence):
        rep, weights = self.attn_pool(sequence)
        return rep, {"attn_pool_weights": weights}

    def _forecast_representation(self, sequence):
        rep, weights = self.lastk_pool(sequence)
        return rep, {"lastk_attn_weights": weights}

    def forward(self, modality_inputs, domain_alpha=1.0):
        aux = {}
        modality_inputs = self._apply_modality_dropout(modality_inputs)
        encoded = [encoder(modality) for encoder, modality in zip(self.encoders, modality_inputs)]
        cross_encoded = self._cross_modal_fusion(encoded)
        fused, gate_weights = self.amg(cross_encoded)
        fused, htf_alpha = self.ahtf(fused)
        aux["gate_weights"] = gate_weights
        aux["htf_alpha"] = float(htf_alpha.detach().cpu())

        modality_features = [h.mean(dim=1) for h in cross_encoded]
        aux["modality_features"] = modality_features

        if self.n_classes > 0:
            rep, rep_aux = self._classification_representation(fused)
            logits = self.classifier(rep)
            aux.update(rep_aux)
            aux["shared_feature"] = rep
            aux["representation"] = rep.detach()
            if self.domain_classifier is not None:
                aux["domain_logits"] = self.domain_classifier(gradient_reverse(rep, domain_alpha))
            return logits, aux

        rep, rep_aux = self._forecast_representation(fused)
        preds = self.forecaster(rep)
        aux.update(rep_aux)
        aux["shared_feature"] = rep
        aux["representation"] = rep.detach()
        if self.domain_classifier is not None:
            aux["domain_logits"] = self.domain_classifier(gradient_reverse(rep, domain_alpha))
        return preds, aux


class TMTFNetRevIN(TMTFNet_v2):
    """TMTFNet wrapped with reversible instance normalization (RevIN).

    Each forecasting input window is per-channel zero-meaned and unit-scaled
    before the model forward, and the forecast head's output is rescaled
    back using the target channel's input statistics. RevIN is widely used
    in modern long-horizon forecasting (PatchTST, iTransformer, FEDformer).
    """

    def __init__(self, *args, target_channel=-1, revin_eps=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_channel = int(target_channel)
        self.revin_eps = float(revin_eps)

    def forward(self, modality_inputs, domain_alpha=1.0):
        if self.pred_len <= 0 or self.n_classes > 0:
            return super().forward(modality_inputs, domain_alpha=domain_alpha)
        normalised = []
        means, stds = [], []
        for m in modality_inputs:
            mu = m.mean(dim=1, keepdim=True)
            sd = (m.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps).sqrt()
            normalised.append((m - mu) / sd)
            means.append(mu)
            stds.append(sd)
        all_concat = torch.cat([m for m in modality_inputs], dim=-1)
        last_mu = all_concat[:, :, self.target_channel].mean(dim=1, keepdim=True)
        last_sd = (all_concat[:, :, self.target_channel].var(dim=1, keepdim=True, unbiased=False) + self.revin_eps).sqrt()
        preds, aux = super().forward(normalised, domain_alpha=domain_alpha)
        preds = preds * last_sd + last_mu
        return preds, aux


class PatchModalityEncoder(nn.Module):
    """Replaces ``ModalitySpecificEncoder`` with PatchTST-style per-channel patch
    encoding followed by channel-wise mean pooling. Keeps the per-modality output
    interface ``(B, T_eff, d_model)`` so downstream G-CMTA / AMG / A-HTF modules
    are unchanged."""

    def __init__(self, input_dim, d_model, n_heads, n_layers, patch_len=8, stride=4,
                 dropout=0.1, max_len=512):
        super().__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, C_in)
        bsz, T, C = x.shape
        # patch over T per channel -> (B*C, n_patches, patch_len)
        xt = x.transpose(1, 2)  # (B, C, T)
        patches = xt.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        bsz2, ch, n_patches, plen = patches.shape
        tokens = patches.contiguous().view(bsz2 * ch, n_patches, plen)
        tokens = self.proj(tokens)
        tokens = self.pos(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        # reshape and average across channels: -> (B, n_patches, d_model)
        per_ch = tokens.view(bsz2, ch, n_patches, self.d_model).mean(dim=1)
        return per_ch


class TMTFNetUltra(TMTFNet_v2):
    """TMTFNet with PatchTST-style per-channel encoders replacing the original
    ModalitySpecificEncoder. Combines PatchTST's sample efficiency on dense
    multi-channel signals with TMTFNet's gated cross-modal fusion."""

    def __init__(
        self,
        modality_dims,
        d_model=64,
        n_heads=8,
        n_enc_layers=3,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        modality_dropout=0.1,
        use_domain_adapt=False,
        n_domains=2,
        max_len=512,
        seq_len=128,
        patch_len=8,
        patch_stride=4,
        **kwargs,
    ):
        super().__init__(
            modality_dims=modality_dims, d_model=d_model, n_heads=n_heads,
            n_enc_layers=n_enc_layers, n_classes=n_classes, pred_len=pred_len,
            dropout=dropout, modality_dropout=modality_dropout,
            use_domain_adapt=use_domain_adapt, n_domains=n_domains, max_len=max_len,
            **kwargs,
        )
        # Replace per-modality encoders with patch-style ones
        self.encoders = nn.ModuleList([
            PatchModalityEncoder(dim, d_model, n_heads, n_enc_layers,
                                  patch_len=patch_len, stride=patch_stride,
                                  dropout=dropout, max_len=max_len)
            for dim in modality_dims
        ])


class TMTFNetPlus(TMTFNet_v2):
    """TMTFNet enhanced with a parallel patch-token branch (PatchTST-style)
    fused via a learnable branch gate.

    The standard MSTE branch captures explicit cross-modal interactions, while
    the patch-token branch processes each input channel independently as a
    sequence of overlapping patches and is known to be highly sample-efficient
    on dense multi-IMU data such as PAMAP2 cross-subject HAR.

    The two branch representations are produced at the same hidden dimension
    and combined with a per-batch sigmoid gate. We add the patch branch
    *after* MSTE and AMG so all original ablation conclusions about TMTFNet's
    fusion modules continue to hold; the patch branch is therefore a strict
    additive enhancement.
    """

    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_heads=8,
        n_enc_layers=3,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        modality_dropout=0.1,
        use_domain_adapt=False,
        n_domains=2,
        max_len=512,
        seq_len=128,
        patch_len=8,
        patch_stride=4,
        n_patch_layers=2,
        **kwargs,
    ):
        super().__init__(
            modality_dims=modality_dims, d_model=d_model, n_heads=n_heads,
            n_enc_layers=n_enc_layers, n_classes=n_classes, pred_len=pred_len,
            dropout=dropout, modality_dropout=modality_dropout,
            use_domain_adapt=use_domain_adapt, n_domains=n_domains, max_len=max_len,
            **kwargs,
        )
        # parallel patch-token branch
        total_dim = sum(modality_dims)
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.patch_pos = PositionalEncoding(d_model, max_len=max(64, seq_len), dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(layer, num_layers=n_patch_layers)
        self.patch_norm = nn.LayerNorm(d_model)
        self.patch_pool = AttentionPooling(d_model)
        self.branch_gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        # per-channel forecasting head when forecasting; classification head
        # already projects from d_model so no change needed there.
        self._total_input_dim = total_dim

    def _patch_branch(self, modality_inputs):
        # concatenate modalities along feature axis -> (B, T, C); per-channel patch tokens
        x = torch.cat(modality_inputs, dim=-1).transpose(1, 2)  # (B, C, T)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        bsz, ch, n_patches, plen = patches.shape
        tokens = patches.contiguous().view(bsz * ch, n_patches, plen)
        tokens = self.patch_proj(tokens)
        tokens = self.patch_pos(tokens)
        encoded = self.patch_encoder(tokens)
        encoded = self.patch_norm(encoded)
        # average over patches per channel, then mean-pool over channels
        per_channel = encoded.mean(dim=1).view(bsz, ch, -1)
        rep = per_channel.mean(dim=1)
        return rep

    def forward(self, modality_inputs, domain_alpha=1.0):
        aux = {}
        modality_inputs_raw = modality_inputs
        modality_inputs = self._apply_modality_dropout(modality_inputs)
        encoded = [encoder(modality) for encoder, modality in zip(self.encoders, modality_inputs)]
        cross_encoded = self._cross_modal_fusion(encoded)
        fused, gate_weights = self.amg(cross_encoded)
        fused, htf_alpha = self.ahtf(fused)
        aux["gate_weights"] = gate_weights
        aux["htf_alpha"] = float(htf_alpha.detach().cpu())
        aux["modality_features"] = [h.mean(dim=1) for h in cross_encoded]

        if self.n_classes > 0:
            rep_msft, rep_aux = self._classification_representation(fused)
            rep_patch = self._patch_branch(modality_inputs_raw)
            gate = self.branch_gate_net(torch.cat([rep_msft, rep_patch], dim=-1))
            rep = gate * rep_msft + (1.0 - gate) * rep_patch
            logits = self.classifier(rep)
            aux.update(rep_aux)
            aux["branch_gate"] = gate.detach().mean().item()
            aux["shared_feature"] = rep
            aux["representation"] = rep.detach()
            if self.domain_classifier is not None:
                aux["domain_logits"] = self.domain_classifier(gradient_reverse(rep, domain_alpha))
            return logits, aux

        # forecasting path: use MSTE branch only (patch branch needs separate forecast head)
        rep, rep_aux = self._forecast_representation(fused)
        preds = self.forecaster(rep)
        aux.update(rep_aux)
        aux["shared_feature"] = rep
        aux["representation"] = rep.detach()
        if self.domain_classifier is not None:
            aux["domain_logits"] = self.domain_classifier(gradient_reverse(rep, domain_alpha))
        return preds, aux


class TMTFNet_v2_NoGCMTA(TMTFNet_v2):
    def _cross_modal_fusion(self, encoded_modalities):
        return encoded_modalities


class TMTFNet_v2_NoAMG(TMTFNet_v2):
    def forward(self, modality_inputs, domain_alpha=1.0):
        del domain_alpha
        aux = {}
        modality_inputs = self._apply_modality_dropout(modality_inputs)
        encoded = [encoder(modality) for encoder, modality in zip(self.encoders, modality_inputs)]
        cross_encoded = self._cross_modal_fusion(encoded)
        fused = torch.stack(cross_encoded, dim=0).mean(dim=0)
        fused, htf_alpha = self.ahtf(fused)
        aux["htf_alpha"] = float(htf_alpha.detach().cpu())

        if self.n_classes > 0:
            rep, rep_aux = self._classification_representation(fused)
            logits = self.classifier(rep)
            aux.update(rep_aux)
            aux["representation"] = rep.detach()
            return logits, aux

        rep, rep_aux = self._forecast_representation(fused)
        preds = self.forecaster(rep)
        aux.update(rep_aux)
        aux["representation"] = rep.detach()
        return preds, aux


class TMTFNet_v2_NoAHTF(TMTFNet_v2):
    def forward(self, modality_inputs, domain_alpha=1.0):
        del domain_alpha
        aux = {}
        modality_inputs = self._apply_modality_dropout(modality_inputs)
        encoded = [encoder(modality) for encoder, modality in zip(self.encoders, modality_inputs)]
        cross_encoded = self._cross_modal_fusion(encoded)
        fused, gate_weights = self.amg(cross_encoded)
        aux["gate_weights"] = gate_weights
        aux["htf_alpha"] = 0.0

        if self.n_classes > 0:
            rep, rep_aux = self._classification_representation(fused)
            logits = self.classifier(rep)
            aux.update(rep_aux)
            aux["representation"] = rep.detach()
            return logits, aux

        rep, rep_aux = self._forecast_representation(fused)
        preds = self.forecaster(rep)
        aux.update(rep_aux)
        aux["representation"] = rep.detach()
        return preds, aux


class TMTFNet_v2_NoAttnPool(TMTFNet_v2):
    def _classification_representation(self, sequence):
        return sequence.mean(dim=1), {}

    def _forecast_representation(self, sequence):
        last_k = min(16, sequence.size(1))
        return sequence[:, -last_k:, :].mean(dim=1), {}


class TMTFNet_v2_NoModDrop(TMTFNet_v2):
    def __init__(self, *args, **kwargs):
        kwargs["modality_dropout"] = 0.0
        super().__init__(*args, **kwargs)


class TMTFNetLite(nn.Module):
    def __init__(
        self,
        modality_dims,
        d_model=64,
        n_heads=4,
        n_enc_layers=2,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        modality_dropout=0.1,
        max_len=512,
        use_hub=True,
        use_gate=True,
        use_temp_conv=True,
        use_attn_pool=True,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.n_modalities = len(modality_dims)
        self.d_model = d_model
        self.n_classes = n_classes
        self.pred_len = pred_len
        self.modality_dropout_p = modality_dropout
        self.use_hub = use_hub
        self.use_gate = use_gate and use_hub
        self.use_temp_conv = use_temp_conv
        self.use_attn_pool = use_attn_pool

        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                )
                for dim in modality_dims
            ]
        )
        self.pos_enc = HybridPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        if self.use_hub:
            self.hub = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.gather_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.gather_norm = nn.LayerNorm(d_model)
            self.broadcast_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.broadcast_norm = nn.LayerNorm(d_model)
        else:
            self.register_parameter("hub", None)
            self.gather_attn = None
            self.gather_norm = None
            self.broadcast_attn = None
            self.broadcast_norm = None

        self.gate_proj = nn.Linear(d_model * self.n_modalities, self.n_modalities) if self.use_gate else None

        if self.use_temp_conv:
            self.temp_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            self.temp_gate = nn.Parameter(torch.zeros(1))
            self.temp_norm = nn.LayerNorm(d_model)
        else:
            self.temp_conv = None
            self.register_parameter("temp_gate", None)
            self.temp_norm = None

        if self.use_attn_pool:
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        else:
            self.register_parameter("pool_query", None)
            self.pool_attn = None

        if n_classes > 0:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif pred_len > 0:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, pred_len),
            )
        else:
            raise ValueError("Either n_classes or pred_len must be positive.")

    def _apply_modality_dropout(self, modality_inputs):
        if not self.training or self.modality_dropout_p <= 0.0:
            return modality_inputs
        batch_size = modality_inputs[0].size(0)
        keep_mask = torch.rand(batch_size, self.n_modalities, device=modality_inputs[0].device) > self.modality_dropout_p
        all_dropped = keep_mask.sum(dim=1) == 0
        if all_dropped.any():
            restore_idx = torch.randint(0, self.n_modalities, (int(all_dropped.sum().item()),), device=keep_mask.device)
            keep_mask[all_dropped] = False
            keep_mask[all_dropped, restore_idx] = True
        outputs = []
        for idx, modality in enumerate(modality_inputs):
            mask = keep_mask[:, idx].view(batch_size, 1, 1).to(modality.dtype)
            outputs.append(modality * mask)
        return outputs

    def _encode_modalities(self, modality_inputs):
        encoded = []
        for adapter, x in zip(self.adapters, modality_inputs):
            h = adapter(x)
            h = self.pos_enc(h)
            h = self.shared_encoder(h)
            encoded.append(self.encoder_norm(h))
        return encoded

    def _hub_cross_modal(self, encoded):
        if not self.use_hub:
            return encoded
        batch_size, seq_len, _ = encoded[0].shape
        hub = self.hub.expand(batch_size, seq_len, -1)
        all_modalities = torch.cat(encoded, dim=1)
        hub_out, _ = self.gather_attn(hub, all_modalities, all_modalities, need_weights=False)
        hub = self.gather_norm(hub + hub_out)

        cross_attended = []
        for h in encoded:
            h_out, _ = self.broadcast_attn(h, hub, hub, need_weights=False)
            cross_attended.append(self.broadcast_norm(h + h_out))
        return cross_attended

    def _fuse_modalities(self, encoded):
        batch_size, seq_len, _ = encoded[0].shape
        if not self.use_hub:
            fused = torch.stack(encoded, dim=0).mean(dim=0)
            gate_weights = torch.full((batch_size, self.n_modalities), 1.0 / self.n_modalities, device=fused.device)
            return fused, gate_weights

        pooled_modalities = [h.mean(dim=1) for h in encoded]
        if self.use_gate:
            gate_input = torch.cat(pooled_modalities, dim=-1)
            gate_weights = torch.softmax(self.gate_proj(gate_input), dim=-1)
        else:
            gate_weights = torch.full((batch_size, self.n_modalities), 1.0 / self.n_modalities, device=encoded[0].device)

        fused = torch.zeros(batch_size, seq_len, self.d_model, device=encoded[0].device)
        for idx, h in enumerate(encoded):
            fused = fused + gate_weights[:, idx : idx + 1].unsqueeze(-1) * h
        return fused, gate_weights

    def _temporal_shortcut(self, fused):
        if not self.use_temp_conv:
            return fused, 0.0
        conv_out = self.temp_conv(fused.transpose(1, 2)).transpose(1, 2)
        gate_value = torch.sigmoid(self.temp_gate)
        return self.temp_norm(fused + gate_value * conv_out), float(gate_value.detach().cpu())

    def _pool_sequence(self, fused):
        if not self.use_attn_pool:
            return fused.mean(dim=1), None
        query = self.pool_query.expand(fused.size(0), -1, -1)
        pooled, weights = self.pool_attn(query, fused, fused)
        return pooled.squeeze(1), weights

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        aux = {}
        modality_inputs = self._apply_modality_dropout(modality_inputs)
        encoded = self._encode_modalities(modality_inputs)
        cross_attended = self._hub_cross_modal(encoded)
        fused, gate_weights = self._fuse_modalities(cross_attended)
        fused, temp_gate = self._temporal_shortcut(fused)
        pooled, pool_weights = self._pool_sequence(fused)

        aux["representation"] = pooled.detach()
        aux["gate_weights"] = gate_weights.detach()
        aux["temp_gate"] = temp_gate
        if pool_weights is not None:
            aux["pool_weights"] = pool_weights.detach()

        return self.head(pooled), aux


class TMTFNetLite_NoHubCMA(TMTFNetLite):
    def __init__(self, *args, **kwargs):
        kwargs["use_hub"] = False
        kwargs["use_gate"] = False
        super().__init__(*args, **kwargs)


class TMTFNetLite_NoGate(TMTFNetLite):
    def __init__(self, *args, **kwargs):
        kwargs["use_gate"] = False
        super().__init__(*args, **kwargs)


class TMTFNetLite_NoTempConv(TMTFNetLite):
    def __init__(self, *args, **kwargs):
        kwargs["use_temp_conv"] = False
        super().__init__(*args, **kwargs)


class TMTFNetLite_NoAttnPool(TMTFNetLite):
    def __init__(self, *args, **kwargs):
        kwargs["use_attn_pool"] = False
        super().__init__(*args, **kwargs)


class TMTFNetLite_NoModDrop(TMTFNetLite):
    def __init__(self, *args, **kwargs):
        kwargs["modality_dropout"] = 0.0
        super().__init__(*args, **kwargs)


class SequenceBackboneBase(nn.Module):
    def __init__(self, rep_dim, d_model, n_classes=0, pred_len=0, dropout=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.pred_len = pred_len
        if n_classes > 0:
            self.classifier = ClassificationHead(rep_dim, d_model, n_classes, dropout=dropout)
        elif pred_len > 0:
            self.forecaster = ForecastHead(rep_dim, d_model, pred_len, dropout=dropout)
        else:
            raise ValueError("Either n_classes or pred_len must be positive.")

    def _finalize(self, rep):
        if self.n_classes > 0:
            return self.classifier(rep), {"representation": rep.detach()}
        return self.forecaster(rep), {"representation": rep.detach()}


class LSTMBaseline(SequenceBackboneBase):
    def __init__(self, modality_dims, d_model=128, n_layers=3, n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        total_dim = sum(modality_dims)
        super().__init__(rep_dim=d_model * 2, d_model=d_model, n_classes=n_classes, pred_len=pred_len, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs)
        out, _ = self.lstm(x)
        rep = out.mean(dim=1)
        return self._finalize(rep)


class GRUBaseline(SequenceBackboneBase):
    def __init__(self, modality_dims, d_model=128, n_layers=3, n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        total_dim = sum(modality_dims)
        super().__init__(rep_dim=d_model * 2, d_model=d_model, n_classes=n_classes, pred_len=pred_len, dropout=dropout)
        self.gru = nn.GRU(
            input_size=total_dim,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs)
        out, _ = self.gru(x)
        rep = out.mean(dim=1)
        return self._finalize(rep)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNBaseline(SequenceBackboneBase):
    def __init__(self, modality_dims, d_model=128, n_layers=4, n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        total_dim = sum(modality_dims)
        super().__init__(rep_dim=d_model, d_model=d_model, n_classes=n_classes, pred_len=pred_len, dropout=dropout)
        blocks = []
        in_ch = total_dim
        for layer_idx in range(n_layers):
            blocks.append(TemporalBlock(in_ch, d_model, kernel_size=3, dilation=2 ** layer_idx, dropout=dropout))
            in_ch = d_model
        self.network = nn.Sequential(*blocks)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        out = self.network(x).transpose(1, 2)
        rep = out.mean(dim=1)
        return self._finalize(rep)


class TransformerBaseline(SequenceBackboneBase):
    def __init__(self, modality_dims, d_model=128, n_heads=8, n_layers=3, n_classes=0, pred_len=0, dropout=0.1, max_len=512, **kwargs):
        total_dim = sum(modality_dims)
        super().__init__(rep_dim=d_model, d_model=d_model, n_classes=n_classes, pred_len=pred_len, dropout=dropout)
        self.input_proj = nn.Linear(total_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs)
        x = self.pos_enc(self.input_proj(x))
        x = self.norm(self.encoder(x))
        rep = x.mean(dim=1)
        return self._finalize(rep)


class CrossformerBaseline(SequenceBackboneBase):
    def __init__(self, modality_dims, d_model=128, n_heads=8, n_layers=3, n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        super().__init__(rep_dim=d_model, d_model=d_model, n_classes=n_classes, pred_len=pred_len, dropout=dropout)
        self.encoders = nn.ModuleList(
            [ModalitySpecificEncoder(dim, d_model, n_heads, n_layers, dropout=dropout) for dim in modality_dims]
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        encoded = [encoder(modality) for encoder, modality in zip(self.encoders, modality_inputs)]
        query = torch.stack(encoded, dim=0).mean(dim=0)
        key_value = torch.cat(encoded, dim=1)
        cross, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        cross = self.norm(query + cross)
        rep = cross.mean(dim=1)
        return self._finalize(rep)


class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_heads=8,
        n_layers=3,
        n_classes=0,
        pred_len=0,
        dropout=0.1,
        seq_len=128,
        patch_len=8,
        stride=4,
        max_len=512,
        **kwargs,
    ):
        super().__init__()
        self.total_dim = sum(modality_dims)
        self.n_classes = n_classes
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        rep_dim = self.total_dim * d_model
        if n_classes > 0:
            self.head = ClassificationHead(rep_dim, d_model, n_classes, dropout=dropout)
        else:
            self.head = ForecastHead(rep_dim, d_model, pred_len, dropout=dropout)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch_size, channels, num_patches, patch_len = patches.shape
        patches = patches.contiguous().view(batch_size * channels, num_patches, patch_len)
        tokens = self.patch_proj(patches)
        tokens = self.pos_enc(tokens)
        encoded = self.encoder(tokens)
        rep = encoded.mean(dim=1).view(batch_size, channels * encoded.size(-1))
        out = self.head(rep)
        return out, {"representation": rep.detach()}


class DLinearBaseline(nn.Module):
    def __init__(self, modality_dims, n_classes=0, pred_len=0, seq_len=128, d_model=128, dropout=0.1, **kwargs):
        super().__init__()
        self.total_dim = sum(modality_dims)
        self.n_classes = n_classes
        self.pred_len = pred_len
        self.cls_proj = nn.Linear(seq_len, 1)
        self.forecast_proj = nn.Linear(seq_len, pred_len)
        self.channel_mixer = nn.Linear(self.total_dim, 1)
        if n_classes > 0:
            self.classifier = ClassificationHead(self.total_dim, d_model, n_classes, dropout=dropout)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        if self.n_classes > 0:
            rep = self.cls_proj(x).squeeze(-1)
            logits = self.classifier(rep)
            return logits, {"representation": rep.detach()}
        sequence = self.forecast_proj(x).transpose(1, 2)
        out = self.channel_mixer(sequence).squeeze(-1)
        rep = sequence.mean(dim=1)
        return out, {"representation": rep.detach()}


class TimeMixerBaseline(SequenceBackboneBase):
    def __init__(self, modality_dims, d_model=128, n_classes=0, pred_len=0, dropout=0.1, **kwargs):
        super().__init__(rep_dim=d_model, d_model=d_model, n_classes=n_classes, pred_len=pred_len, dropout=dropout)
        total_dim = sum(modality_dims)
        self.input_proj = nn.Linear(total_dim, d_model)
        self.scale_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(3)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = self.input_proj(_concat_modalities(modality_inputs))
        scales = []
        for idx, factor in enumerate([1, 2, 4]):
            if factor == 1:
                scaled = x
            else:
                pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=factor, stride=factor)
                scaled = pooled.transpose(1, 2)
            scaled = self.scale_mlps[idx](scaled)
            if scaled.size(1) != x.size(1):
                scaled = F.interpolate(scaled.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False).transpose(1, 2)
            scales.append(scaled)
        mixed = self.norm(sum(scales) / len(scales))
        rep = mixed.mean(dim=1)
        return self._finalize(rep)


class ITransformerBaseline(nn.Module):
    def __init__(
        self,
        modality_dims,
        d_model=128,
        n_heads=8,
        n_layers=3,
        n_classes=0,
        pred_len=0,
        seq_len=128,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pred_len = pred_len
        self.token_proj = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        if n_classes > 0:
            self.head = ClassificationHead(d_model, d_model, n_classes, dropout=dropout)
        else:
            self.head = ForecastHead(d_model, d_model, pred_len, dropout=dropout)

    def forward(self, modality_inputs, **kwargs):
        del kwargs
        x = _concat_modalities(modality_inputs).transpose(1, 2)
        tokens = self.token_proj(x)
        encoded = self.encoder(tokens)
        rep = encoded.mean(dim=1)
        out = self.head(rep)
        return out, {"representation": rep.detach()}


from .baselines_cnn import CNN1D, DeepConvLSTM  # noqa: E402  (registered below)

MODEL_REGISTRY = {
    "TMTFNet_v2": TMTFNet_v2,
    "TMTFNetPlus": TMTFNetPlus,
    "TMTFNetUltra": TMTFNetUltra,
    "TMTFNetRevIN": TMTFNetRevIN,
    "TMTFNet_v2_NoGCMTA": TMTFNet_v2_NoGCMTA,
    "TMTFNet_v2_NoAMG": TMTFNet_v2_NoAMG,
    "TMTFNet_v2_NoAHTF": TMTFNet_v2_NoAHTF,
    "TMTFNet_v2_NoAttnPool": TMTFNet_v2_NoAttnPool,
    "TMTFNet_v2_NoModDrop": TMTFNet_v2_NoModDrop,
    "TMTFNetLite": TMTFNetLite,
    "TMTFNetLite_NoHubCMA": TMTFNetLite_NoHubCMA,
    "TMTFNetLite_NoGate": TMTFNetLite_NoGate,
    "TMTFNetLite_NoTempConv": TMTFNetLite_NoTempConv,
    "TMTFNetLite_NoAttnPool": TMTFNetLite_NoAttnPool,
    "TMTFNetLite_NoModDrop": TMTFNetLite_NoModDrop,
    "LSTM": LSTMBaseline,
    "GRU": GRUBaseline,
    "TCN": TCNBaseline,
    "Transformer": TransformerBaseline,
    "Crossformer": CrossformerBaseline,
    "PatchTST": PatchTSTBaseline,
    "DLinear": DLinearBaseline,
    "TimeMixer": TimeMixerBaseline,
    "iTransformer": ITransformerBaseline,
    "CNN1D": CNN1D,
    "DeepConvLSTM": DeepConvLSTM,
}


def build_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)
