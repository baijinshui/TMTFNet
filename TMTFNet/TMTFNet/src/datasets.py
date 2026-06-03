"""
Dataset loaders for TMTFNet experiments.
=========================================
1. UCI HAR - Human Activity Recognition (multi-sensor classification)
2. ETT - Electricity Transformer Temperature (cross-domain forecasting)
3. Synthetic - For code verification and additional experiments
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ============================================================
# UCI HAR Dataset
# ============================================================

class HARDataset(Dataset):
    """
    UCI Human Activity Recognition dataset.
    Splits 9 sensor channels into 3 modalities:
      - Modality 0: body_acc (x, y, z)
      - Modality 1: body_gyro (x, y, z)
      - Modality 2: total_acc (x, y, z)
    """
    def __init__(self, data_dir, split='train', seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        base = os.path.join(data_dir, 'UCI HAR Dataset', 'UCI HAR Dataset')
        if not os.path.exists(base):
            base = os.path.join(data_dir, 'UCI HAR Dataset')
        if not os.path.exists(base):
            # Try flat structure
            base = data_dir

        split_dir = os.path.join(base, split)

        # Load signals
        signals = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                   'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                   'total_acc_x', 'total_acc_y', 'total_acc_z']

        inertial_dir = os.path.join(split_dir, 'Inertial Signals')
        data_list = []
        for sig in signals:
            fpath = os.path.join(inertial_dir, f'{sig}_{split}.txt')
            arr = pd.read_csv(fpath, sep=r'\s+', header=None).values
            data_list.append(arr)

        # Stack: [N, 128, 9]
        self.data = np.stack(data_list, axis=-1).astype(np.float32)

        # Labels
        label_path = os.path.join(split_dir, f'y_{split}.txt')
        self.labels = pd.read_csv(label_path, header=None).values.flatten() - 1  # 0-indexed

        # Normalize per channel
        N, T, C = self.data.shape
        self.data = self.data.reshape(-1, C)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data).reshape(N, T, C)

        # Truncate/pad to seq_len
        if T > seq_len:
            self.data = self.data[:, :seq_len, :]
        elif T < seq_len:
            pad = np.zeros((N, seq_len - T, C), dtype=np.float32)
            self.data = np.concatenate([self.data, pad], axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        # Split into 3 modalities
        mod1 = x[:, 0:3]  # body_acc
        mod2 = x[:, 3:6]  # body_gyro
        mod3 = x[:, 6:9]  # total_acc
        y = torch.LongTensor([self.labels[idx]])[0]
        return [mod1, mod2, mod3], y

    @property
    def modality_dims(self):
        return [3, 3, 3]

    @property
    def n_classes(self):
        return 6


# ============================================================
# ETT Dataset
# ============================================================

class ETTDataset(Dataset):
    """
    Electricity Transformer Temperature dataset for forecasting.
    Multi-variate → split into 2 modalities.
    """
    def __init__(self, data_path, split='train', seq_len=96, pred_len=24,
                 train_ratio=0.6, val_ratio=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(data_path)
        # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        df = df.drop('date', axis=1)
        data = df.values.astype(np.float32)

        # Split train/val/test
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == 'train':
            data = data[:train_end]
        elif split == 'val':
            data = data[train_end:val_end]
        else:
            data = data[val_end:]

        # Normalize
        self.scaler = StandardScaler()
        if split == 'train':
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.fit_transform(data)  # In practice, use train scaler

        self.data = data
        self.target_col = -1  # OT column

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        # Split into 2 modalities: [HUFL, HULL, MUFL] and [MULL, LUFL, LULL, OT]
        mod1 = x[:, :3]
        mod2 = x[:, 3:]

        return [mod1, mod2], y

    @property
    def modality_dims(self):
        return [3, 4]


# ============================================================
# Synthetic Dataset (for code verification + additional exp)
# ============================================================

class SyntheticMultiModalDataset(Dataset):
    """
    Synthetic multi-modal temporal data with controllable complexity.
    Generates data with inter-modal dependencies and domain shifts.
    """
    def __init__(self, n_samples=5000, seq_len=64, n_modalities=3,
                 dims_per_mod=4, n_classes=5, domain=0, noise_level=0.1, seed=42):
        super().__init__()
        np.random.seed(seed + domain * 1000)

        self.n_modalities = n_modalities
        self._modality_dims = [dims_per_mod] * n_modalities
        self._n_classes = n_classes

        # Generate base patterns per class
        freqs = np.linspace(0.5, 3.0, n_classes)
        t = np.linspace(0, 4 * np.pi, seq_len)

        data_mods = [[] for _ in range(n_modalities)]
        labels = []

        for i in range(n_samples):
            cls = i % n_classes
            labels.append(cls)

            freq = freqs[cls]
            # Domain shift: phase and amplitude
            phase_shift = domain * np.pi / 4
            amp_scale = 1.0 + domain * 0.3

            for m in range(n_modalities):
                mod_data = np.zeros((seq_len, dims_per_mod), dtype=np.float32)
                for d in range(dims_per_mod):
                    base = amp_scale * np.sin(freq * t + phase_shift + m * 0.5 + d * 0.3)
                    # Inter-modal dependency
                    if m > 0:
                        interaction = 0.3 * np.sin((freq + m * 0.2) * t + d)
                    else:
                        interaction = 0
                    mod_data[:, d] = base + interaction + noise_level * np.random.randn(seq_len)
                data_mods[m].append(mod_data)

        self.data = [np.array(d, dtype=np.float32) for d in data_mods]
        self.labels = np.array(labels, dtype=np.int64)

        # Normalize
        for m in range(n_modalities):
            N, T, D = self.data[m].shape
            reshaped = self.data[m].reshape(-1, D)
            scaler = StandardScaler()
            self.data[m] = scaler.fit_transform(reshaped).reshape(N, T, D)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mods = [torch.FloatTensor(self.data[m][idx]) for m in range(self.n_modalities)]
        y = torch.LongTensor([self.labels[idx]])[0]
        return mods, y

    @property
    def modality_dims(self):
        return self._modality_dims

    @property
    def n_classes(self):
        return self._n_classes


class SyntheticForecastDataset(Dataset):
    """Synthetic multi-modal forecasting data."""
    def __init__(self, n_total=10000, seq_len=96, pred_len=24,
                 n_modalities=2, dims_per_mod=4, domain=0, seed=42):
        super().__init__()
        np.random.seed(seed + domain * 1000)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_modalities = n_modalities
        self._modality_dims = [dims_per_mod] * n_modalities

        t = np.arange(n_total, dtype=np.float32)
        phase = domain * np.pi / 3

        # Generate multi-variate time series with cross-modal dependencies
        self.all_mods = []
        for m in range(n_modalities):
            mod = np.zeros((n_total, dims_per_mod), dtype=np.float32)
            for d in range(dims_per_mod):
                trend = 0.001 * t * (1 + domain * 0.2)
                seasonal = np.sin(2 * np.pi * t / (24 + m * 6) + phase + d * 0.4)
                weekly = 0.5 * np.sin(2 * np.pi * t / (168 + m * 12))
                noise = 0.1 * np.random.randn(n_total)
                mod[:, d] = trend + seasonal + weekly + noise
            scaler = StandardScaler()
            mod = scaler.fit_transform(mod)
            self.all_mods.append(mod.astype(np.float32))

        # Target: weighted sum of last dim of each modality
        self.target = sum(mod[:, -1] for mod in self.all_mods) / n_modalities

    def __len__(self):
        return max(0, len(self.target) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        mods = [torch.FloatTensor(self.all_mods[m][idx:idx + self.seq_len])
                for m in range(self.n_modalities)]
        y = torch.FloatTensor(self.target[idx + self.seq_len:idx + self.seq_len + self.pred_len])
        return mods, y

    @property
    def modality_dims(self):
        return self._modality_dims


# ============================================================
# Data Loading Utilities
# ============================================================

def collate_multimodal(batch):
    """Custom collate for multi-modal data."""
    n_mods = len(batch[0][0])
    modalities = []
    for m in range(n_mods):
        mod_batch = torch.stack([item[0][m] for item in batch])
        modalities.append(mod_batch)
    labels = torch.stack([item[1] for item in batch])
    return modalities, labels


def get_har_loaders(data_dir, batch_size=128, num_workers=4, seq_len=128):
    """Get UCI HAR data loaders."""
    train_ds = HARDataset(data_dir, split='train', seq_len=seq_len)
    test_ds = HARDataset(data_dir, split='test', seq_len=seq_len)

    # Split train into train/val (80/20)
    n_train = int(0.8 * len(train_ds))
    n_val = len(train_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
    }
    return loaders, [3, 3, 3], 6


def get_ett_loaders(data_dir, dataset_name='ETTh1', batch_size=128,
                    num_workers=4, seq_len=96, pred_len=24):
    """Get ETT data loaders."""
    data_path = os.path.join(data_dir, f'{dataset_name}.csv')

    train_ds = ETTDataset(data_path, 'train', seq_len, pred_len)
    val_ds = ETTDataset(data_path, 'val', seq_len, pred_len)
    test_ds = ETTDataset(data_path, 'test', seq_len, pred_len)

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
    }
    return loaders, train_ds.modality_dims, 0  # 0 classes = regression


def get_synthetic_loaders(task='classification', batch_size=128, num_workers=4,
                          n_modalities=3, dims_per_mod=4, seq_len=64,
                          n_classes=5, domain=0, seed=42, pred_len=24):
    """Get synthetic data loaders."""
    if task == 'classification':
        full_ds = SyntheticMultiModalDataset(
            n_samples=8000, seq_len=seq_len, n_modalities=n_modalities,
            dims_per_mod=dims_per_mod, n_classes=n_classes, domain=domain, seed=seed
        )
        mod_dims = full_ds.modality_dims
        nc = full_ds.n_classes

        n_train = int(0.7 * len(full_ds))
        n_val = int(0.15 * len(full_ds))
        n_test = len(full_ds) - n_train - n_val
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test])
    else:
        full_ds = SyntheticForecastDataset(
            n_total=10000, seq_len=seq_len, pred_len=pred_len,
            n_modalities=n_modalities, dims_per_mod=dims_per_mod, domain=domain, seed=seed
        )
        mod_dims = full_ds.modality_dims
        nc = 0
        n = len(full_ds)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test])

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_multimodal, pin_memory=True),
    }
    return loaders, mod_dims, nc
