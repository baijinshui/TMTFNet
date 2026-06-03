"""GPU-resident loaders that bypass torch DataLoader.

For our datasets the entire training corpus fits in a few hundred MB on the
RTX 5090, so the cleanest way to remove the CPU/Python bottleneck is to push
*everything* to the GPU once at start of training and slice tensors directly
each step. This eliminates worker-process spawn cost, ``collate_multimodal``
overhead, and CPU<->GPU pinned-memory transfers.

This module mirrors the slow-path APIs in ``datasets.py`` but returns a
``GPUResidentBundle`` carrying ready-to-use tensors plus an iterator that
behaves like a DataLoader. The slow path remains authoritative for the
already-completed 5-seed rebuttal matrix; the fast path is opt-in via the
``--fast`` flag in ``run_rebuttal_experiments.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .datasets import (HAR_CLASS_NAMES, PAMAP2_MAIN_ACTIVITIES,
                       PAMAP2_CLASS_NAMES, _load_ett_frame, _load_har_split,
                       _load_pamap2_subject, _resample_to_hourly,
                       _resolve_har_base, _segment_pamap2_subject,
                       _split_by_ratio)


@dataclass
class GPUSplit:
    """Single train/val/test split living on the GPU."""

    modalities: list[torch.Tensor]   # each (N, T, d_i)
    targets: torch.Tensor            # (N,) long for classification, (N, P) float for forecasting
    n: int


@dataclass
class GPUResidentBundle:
    train: GPUSplit
    val: GPUSplit
    test: GPUSplit
    target_train: GPUSplit | None
    modality_dims: list[int]
    n_classes: int = 0
    class_names: list | None = None


def _to_gpu(arrays, device, dtype):
    return [torch.as_tensor(a, dtype=dtype, device=device) for a in arrays]


class GPUBatchIterator:
    """Yields ``(modalities_list, targets)`` batches by indexing the resident tensors.

    Reproduces the behaviour of ``DataLoader(dataset, batch_size=B, shuffle=True)``
    using a single permutation tensor on the GPU. No worker processes, no
    pinned memory transfers, no collate.
    """

    def __init__(self, split: GPUSplit, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
        self.split = split
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.generator = generator
        self.n_batches = (split.n + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.split.n, device=self.split.modalities[0].device, generator=self.generator)
        else:
            perm = torch.arange(self.split.n, device=self.split.modalities[0].device)
        for start in range(0, self.split.n, self.batch_size):
            idx = perm[start:start + self.batch_size]
            mods = [m.index_select(0, idx) for m in self.split.modalities]
            tgts = self.split.targets.index_select(0, idx)
            yield mods, tgts


def _split_har_modalities(array):
    return [array[:, :, 0:3], array[:, :, 3:6], array[:, :, 6:9]]


def get_har_bundle_gpu(data_dir, batch_size=64, seq_len=128, split_seed=42, device=torch.device("cuda")):
    base = _resolve_har_base(data_dir)
    train_x, train_y = _load_har_split(base, "train")
    test_x, test_y = _load_har_split(base, "test")

    scaler = StandardScaler()
    n_feat = train_x.shape[-1]
    scaler.fit(train_x.reshape(-1, n_feat))
    train_x = scaler.transform(train_x.reshape(-1, n_feat)).reshape(train_x.shape).astype(np.float32)
    test_x = scaler.transform(test_x.reshape(-1, n_feat)).reshape(test_x.shape).astype(np.float32)
    if train_x.shape[1] > seq_len:
        train_x = train_x[:, :seq_len]
        test_x = test_x[:, :seq_len]

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(train_y))
    n_val = int(0.2 * len(perm))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_mods = _to_gpu(_split_har_modalities(train_x[tr_idx]), device, torch.float32)
    val_mods = _to_gpu(_split_har_modalities(train_x[val_idx]), device, torch.float32)
    test_mods = _to_gpu(_split_har_modalities(test_x), device, torch.float32)
    train_y_t = torch.as_tensor(train_y[tr_idx], dtype=torch.long, device=device)
    val_y_t = torch.as_tensor(train_y[val_idx], dtype=torch.long, device=device)
    test_y_t = torch.as_tensor(test_y, dtype=torch.long, device=device)

    return GPUResidentBundle(
        train=GPUSplit(train_mods, train_y_t, train_y_t.size(0)),
        val=GPUSplit(val_mods, val_y_t, val_y_t.size(0)),
        test=GPUSplit(test_mods, test_y_t, test_y_t.size(0)),
        target_train=None,
        modality_dims=[3, 3, 3],
        n_classes=6,
        class_names=list(HAR_CLASS_NAMES),
    )


def _ett_windows(series, seq_len, pred_len):
    """Vectorised sliding-window expansion. Returns (N, T+P, F) blocks."""
    n = series.shape[0] - seq_len - pred_len + 1
    if n <= 0:
        return np.zeros((0, seq_len + pred_len, series.shape[1]), dtype=np.float32)
    starts = np.arange(n)
    idx = starts[:, None] + np.arange(seq_len + pred_len)[None, :]
    return series[idx].astype(np.float32)


def get_ett_bundle_gpu(
    data_dir,
    train_dataset="ETTh1",
    test_dataset=None,
    fit_dataset=None,
    batch_size=64,
    seq_len=96,
    pred_len=24,
    resample_test_to_hourly=False,
    emit_target_train=False,
    device=torch.device("cuda"),
):
    test_dataset = test_dataset or train_dataset
    fit_dataset = fit_dataset or train_dataset

    fit_series = _load_ett_frame(data_dir, fit_dataset)
    train_series = _load_ett_frame(data_dir, train_dataset)
    test_series = _load_ett_frame(data_dir, test_dataset)
    if resample_test_to_hourly:
        test_series = _resample_to_hourly(test_series)

    fit_train, _, _ = _split_by_ratio(fit_series)
    scaler = StandardScaler().fit(fit_train)

    train_all = scaler.transform(train_series).astype(np.float32)
    test_all = scaler.transform(test_series).astype(np.float32)
    train_split, val_split, _ = _split_by_ratio(train_all)
    target_train_split, _, test_split = _split_by_ratio(test_all)

    def to_gpu_split(series):
        win = _ett_windows(series, seq_len, pred_len)
        if win.size == 0:
            zeros = torch.zeros(0, seq_len, 7, dtype=torch.float32, device=device)
            target = torch.zeros(0, pred_len, dtype=torch.float32, device=device)
            return GPUSplit([zeros[:, :, :3], zeros[:, :, 3:]], target, 0)
        windows = torch.as_tensor(win, dtype=torch.float32, device=device)
        inputs = windows[:, :seq_len, :]
        targets = windows[:, seq_len:, -1]
        return GPUSplit([inputs[:, :, :3].contiguous(), inputs[:, :, 3:].contiguous()], targets, inputs.size(0))

    bundle = GPUResidentBundle(
        train=to_gpu_split(train_split),
        val=to_gpu_split(val_split),
        test=to_gpu_split(test_split),
        target_train=to_gpu_split(target_train_split) if emit_target_train else None,
        modality_dims=[3, 4],
        n_classes=0,
        class_names=None,
    )
    return bundle


def get_pamap2_bundle_gpu(
    data_dir,
    batch_size=64,
    window=171,
    stride=85,
    source_subjects=(1, 2, 3, 4, 5),
    target_subjects=(6, 7, 8),
    split_seed=42,
    device=torch.device("cuda"),
):
    import glob
    protocol_dir = os.path.join(data_dir, "Protocol")
    subject_data = {}
    for path in sorted(glob.glob(os.path.join(protocol_dir, "subject10?.dat"))):
        sid = int(os.path.basename(path)[7:10]) - 100
        mods, labels = _load_pamap2_subject(path)
        segs, seg_labels = _segment_pamap2_subject(mods, labels, window=window, stride=stride)
        if seg_labels.size:
            subject_data[sid] = (segs, seg_labels)

    def pick(ids):
        mods = {"hand": [], "chest": [], "ankle": []}
        labels = []
        for sid in ids:
            if sid not in subject_data:
                continue
            segs, seg_labels = subject_data[sid]
            for k in mods:
                mods[k].append(segs[k])
            labels.append(seg_labels)
        if not labels:
            return None, None
        return {k: np.concatenate(v, axis=0) for k, v in mods.items()}, np.concatenate(labels, axis=0)

    src_mods, src_labels = pick(source_subjects)
    tgt_mods, tgt_labels = pick(target_subjects)
    for k in src_mods:
        shape = src_mods[k].shape
        scaler = StandardScaler().fit(src_mods[k].reshape(-1, shape[-1]))
        src_mods[k] = scaler.transform(src_mods[k].reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        tshape = tgt_mods[k].shape
        tgt_mods[k] = scaler.transform(tgt_mods[k].reshape(-1, tshape[-1])).reshape(tshape).astype(np.float32)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(src_labels.shape[0])
    n_val = int(0.2 * src_labels.shape[0])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    keys = ("hand", "chest", "ankle")
    train_mods = _to_gpu([src_mods[k][tr_idx] for k in keys], device, torch.float32)
    val_mods = _to_gpu([src_mods[k][val_idx] for k in keys], device, torch.float32)
    test_mods = _to_gpu([tgt_mods[k] for k in keys], device, torch.float32)
    train_y = torch.as_tensor(src_labels[tr_idx], dtype=torch.long, device=device)
    val_y = torch.as_tensor(src_labels[val_idx], dtype=torch.long, device=device)
    test_y = torch.as_tensor(tgt_labels, dtype=torch.long, device=device)
    target_y = torch.zeros(test_y.size(0), dtype=torch.long, device=device)

    return GPUResidentBundle(
        train=GPUSplit(train_mods, train_y, train_y.size(0)),
        val=GPUSplit(val_mods, val_y, val_y.size(0)),
        test=GPUSplit(test_mods, test_y, test_y.size(0)),
        target_train=GPUSplit(test_mods, target_y, target_y.size(0)),
        modality_dims=[9, 9, 9],
        n_classes=len(PAMAP2_MAIN_ACTIVITIES),
        class_names=list(PAMAP2_CLASS_NAMES),
    )
