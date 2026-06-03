import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


HAR_CLASS_NAMES = [
    "Walking",
    "WalkingUpstairs",
    "WalkingDownstairs",
    "Sitting",
    "Standing",
    "Laying",
]


def collate_multimodal(batch):
    n_modalities = len(batch[0][0])
    modalities = [torch.stack([sample[0][m] for sample in batch], dim=0) for m in range(n_modalities)]
    targets = torch.stack([sample[1] for sample in batch], dim=0)
    return modalities, targets


class ArrayMultiModalDataset(Dataset):
    def __init__(self, modalities, targets):
        self.modalities = [torch.as_tensor(mod, dtype=torch.float32) for mod in modalities]
        self.targets = torch.as_tensor(targets)

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, idx):
        mods = [mod[idx] for mod in self.modalities]
        target = self.targets[idx]
        return mods, target


class ForecastWindowDataset(Dataset):
    def __init__(self, data, seq_len=96, pred_len=24):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = self.data.size(1) - 1

    def __len__(self):
        return max(0, self.data.size(0) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.seq_len]
        target = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col]
        return [window[:, :3], window[:, 3:]], target


@dataclass
class DataBundle:
    loaders: dict
    modality_dims: list
    n_classes: int = 0
    class_names: list | None = None


def _resolve_har_base(data_dir):
    candidates = [
        os.path.join(data_dir, "UCI HAR Dataset", "UCI HAR Dataset"),
        os.path.join(data_dir, "UCI HAR Dataset"),
        data_dir,
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "train")):
            return candidate
    raise FileNotFoundError(f"Could not find UCI HAR files under {data_dir}")


def _load_har_split(base_dir, split):
    signals = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]
    inertial_dir = os.path.join(base_dir, split, "Inertial Signals")
    arrays = []
    for signal in signals:
        path = os.path.join(inertial_dir, f"{signal}_{split}.txt")
        arrays.append(pd.read_csv(path, sep=r"\s+", header=None).values)
    features = np.stack(arrays, axis=-1).astype(np.float32)
    labels = pd.read_csv(os.path.join(base_dir, split, f"y_{split}.txt"), header=None).values.flatten().astype(np.int64) - 1
    return features, labels


def get_har_loaders(data_dir, batch_size=64, num_workers=4, seq_len=128, split_seed=42):
    base_dir = _resolve_har_base(data_dir)
    train_x, train_y = _load_har_split(base_dir, "train")
    test_x, test_y = _load_har_split(base_dir, "test")

    scaler = StandardScaler()
    train_shape = train_x.shape
    test_shape = test_x.shape
    scaler.fit(train_x.reshape(-1, train_shape[-1]))
    train_x = scaler.transform(train_x.reshape(-1, train_shape[-1])).reshape(train_shape)
    test_x = scaler.transform(test_x.reshape(-1, test_shape[-1])).reshape(test_shape)

    if train_x.shape[1] > seq_len:
        train_x = train_x[:, :seq_len]
        test_x = test_x[:, :seq_len]

    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(train_y))
    n_val = int(0.2 * len(indices))
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]

    def split_modalities(array):
        return [
            array[:, :, 0:3],
            array[:, :, 3:6],
            array[:, :, 6:9],
        ]

    train_ds = ArrayMultiModalDataset(split_modalities(train_x[tr_idx]), train_y[tr_idx])
    val_ds = ArrayMultiModalDataset(split_modalities(train_x[val_idx]), train_y[val_idx])
    test_ds = ArrayMultiModalDataset(split_modalities(test_x), test_y)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_multimodal,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_multimodal,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_multimodal,
        ),
    }
    return DataBundle(loaders=loaders, modality_dims=[3, 3, 3], n_classes=6, class_names=HAR_CLASS_NAMES)


# PAMAP2 protocol: 54 columns = timestamp + activityID + heart_rate + 3 x 17 IMU channels.
# Each IMU block: temp(1) + acc16g(3) + acc6g(3) + gyro(3) + mag(3) + orient(4).
# We keep acc16g + gyro + mag (9 channels per IMU) to match a 3-modality layout.
PAMAP2_IMU_BLOCKS = {
    "hand": slice(4 - 1, 21 - 1),   # columns 4..20 (0-indexed 3..19)
    "chest": slice(21 - 1, 38 - 1),
    "ankle": slice(38 - 1, 55 - 1),
}
PAMAP2_IMU_KEEP_OFFSETS = [1, 2, 3, 7, 8, 9, 10, 11, 12]  # acc16g + gyro + mag
PAMAP2_MAIN_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]  # the 12 canonical PAMAP2 activities
PAMAP2_CLASS_NAMES = [
    "Lying", "Sitting", "Standing", "Walking", "Running", "Cycling",
    "NordicWalking", "Ascending", "Descending", "VacuumCleaning",
    "Ironing", "RopeJumping",
]


def _load_pamap2_subject(path):
    """Load one PAMAP2 subject .dat file and return (imu_windows_per_modality, labels).

    Returns ``(hand, chest, ankle), labels`` where each modality array has
    shape ``(N, T, 9)`` after windowing.
    """
    frame = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN", "nan"])
    activity = frame.iloc[:, 1].values.astype(np.int64)
    valid_mask = np.isin(activity, PAMAP2_MAIN_ACTIVITIES)
    frame = frame.iloc[valid_mask].reset_index(drop=True)
    activity = activity[valid_mask]
    # linear-interpolate missing IMU channels
    frame = frame.interpolate(limit_direction="both")
    label_map = {v: i for i, v in enumerate(PAMAP2_MAIN_ACTIVITIES)}
    labels_all = np.array([label_map[int(a)] for a in activity], dtype=np.int64)
    modality_cols = {}
    for name, sl in PAMAP2_IMU_BLOCKS.items():
        block = frame.iloc[:, sl].values.astype(np.float32)
        modality_cols[name] = block[:, PAMAP2_IMU_KEEP_OFFSETS]
    return modality_cols, labels_all


def _sliding_window(sig, window=171, stride=85):  # ~1.71s @ 100Hz, 50% overlap (keeps 128-like length after downsample)
    n = sig.shape[0]
    if n < window:
        return np.zeros((0, window, sig.shape[1]), dtype=sig.dtype)
    starts = np.arange(0, n - window + 1, stride)
    return np.stack([sig[s:s + window] for s in starts], axis=0)


def _segment_pamap2_subject(modality_cols, labels, window=171, stride=85):
    cuts = np.concatenate([[0], np.where(np.diff(labels) != 0)[0] + 1, [labels.size]])
    segments = {"hand": [], "chest": [], "ankle": []}
    seg_labels = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        lbl = labels[start]
        for name in segments:
            block = modality_cols[name][start:end]
            windows = _sliding_window(block, window=window, stride=stride)
            segments[name].append(windows)
        seg_labels.append(np.full(sum(w.shape[0] for w in [segments["hand"][-1]]), lbl, dtype=np.int64))
    cat = {name: np.concatenate(parts, axis=0) for name, parts in segments.items()}
    lbls = np.concatenate(seg_labels, axis=0)
    return cat, lbls


def get_pamap2_loaders(
    data_dir,
    batch_size=64,
    num_workers=4,
    window=171,
    stride=85,
    split="cross_subject",
    source_subjects=(1, 2, 3, 4, 5),
    target_subjects=(6, 7, 8),
    split_seed=42,
):
    """Load PAMAP2 with either a standard in-subject split or a cross-subject split.

    split='cross_subject' uses source_subjects for train+val, target_subjects
    for test (Reviewer-1 friendly cross-domain HAR setting). The caller can
    also request just labeled source loaders by passing an empty target tuple.
    """
    protocol_dir = os.path.join(data_dir, "Protocol")
    subject_data = {}
    for path in sorted(glob.glob(os.path.join(protocol_dir, "subject10?.dat"))):
        sid = int(os.path.basename(path)[7:10]) - 100  # subject101 -> 1
        mods, labels = _load_pamap2_subject(path)
        segs, seg_labels = _segment_pamap2_subject(mods, labels, window=window, stride=stride)
        if seg_labels.size == 0:
            continue
        subject_data[sid] = (segs, seg_labels)

    def pick(subject_ids):
        mods = {"hand": [], "chest": [], "ankle": []}
        labels = []
        for sid in subject_ids:
            if sid not in subject_data:
                continue
            segs, seg_labels = subject_data[sid]
            for key in mods:
                mods[key].append(segs[key])
            labels.append(seg_labels)
        if not labels:
            return None, None
        cat = {k: np.concatenate(v, axis=0) for k, v in mods.items()}
        return cat, np.concatenate(labels, axis=0)

    if split == "cross_subject":
        src_mods, src_labels = pick(source_subjects)
        tgt_mods, tgt_labels = pick(target_subjects)
        if src_mods is None or tgt_mods is None:
            raise RuntimeError("PAMAP2 cross-subject split produced empty arrays.")
    else:
        raise ValueError(f"Unknown PAMAP2 split '{split}'")

    scalers = {}
    for name in src_mods:
        shape = src_mods[name].shape
        flat = src_mods[name].reshape(-1, shape[-1])
        scaler = StandardScaler().fit(flat)
        src_mods[name] = scaler.transform(flat).reshape(shape)
        tgt_shape = tgt_mods[name].shape
        tgt_mods[name] = scaler.transform(tgt_mods[name].reshape(-1, tgt_shape[-1])).reshape(tgt_shape)
        scalers[name] = scaler

    rng = np.random.default_rng(split_seed)
    n_src = src_labels.shape[0]
    perm = rng.permutation(n_src)
    n_val = int(0.2 * n_src)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    src_train = [src_mods[name][tr_idx] for name in ("hand", "chest", "ankle")]
    src_val = [src_mods[name][val_idx] for name in ("hand", "chest", "ankle")]
    tgt_full = [tgt_mods[name] for name in ("hand", "chest", "ankle")]

    train_ds = ArrayMultiModalDataset(src_train, src_labels[tr_idx])
    val_ds = ArrayMultiModalDataset(src_val, src_labels[val_idx])
    test_ds = ArrayMultiModalDataset(tgt_full, tgt_labels)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_multimodal),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_multimodal),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_multimodal),
    }
    # also expose an unlabeled target-train loader for DA methods
    tgt_train_ds = ArrayMultiModalDataset(tgt_full, np.zeros(tgt_labels.shape[0], dtype=np.int64))
    loaders["target_train"] = DataLoader(tgt_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_multimodal)
    return DataBundle(loaders=loaders, modality_dims=[9, 9, 9], n_classes=len(PAMAP2_MAIN_ACTIVITIES), class_names=list(PAMAP2_CLASS_NAMES))


def _load_ett_frame(data_dir, dataset_name):
    path = os.path.join(data_dir, f"{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ETT file: {path}")
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame = frame.drop(columns=["date"])
    return frame.values.astype(np.float32)


def _split_by_ratio(array, train_ratio=0.6, val_ratio=0.2):
    n_total = len(array)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    return array[:train_end], array[train_end:val_end], array[val_end:]


def _resample_to_hourly(series):
    """Average every 4 consecutive rows -- turns 15-min ETTm* into hourly."""
    trunc = (series.shape[0] // 4) * 4
    return series[:trunc].reshape(-1, 4, series.shape[1]).mean(axis=1)


def get_ett_loaders(
    data_dir,
    train_dataset="ETTh1",
    test_dataset=None,
    fit_dataset=None,
    batch_size=64,
    num_workers=4,
    seq_len=96,
    pred_len=24,
    resample_test_to_hourly=False,
    emit_target_train=False,
):
    """Build ETT loaders.

    Setting ``train_dataset`` != ``test_dataset`` yields a cross-domain split.
    Setting ``resample_test_to_hourly=True`` downsamples ``test_dataset`` from
    15-min to hourly granularity before windowing -- enables cross-granularity
    experiments such as ETTh1 -> ETTm1 (ETTm1 collapsed to hourly so the
    temporal semantics match). When ``emit_target_train=True`` a target-train
    loader is also exposed (unlabeled target windows drawn from the target's
    train split) for the DA trainer's CORAL/DANN objectives.
    """
    test_dataset = test_dataset or train_dataset
    fit_dataset = fit_dataset or train_dataset

    fit_series = _load_ett_frame(data_dir, fit_dataset)
    train_series = _load_ett_frame(data_dir, train_dataset)
    test_series = _load_ett_frame(data_dir, test_dataset)
    if resample_test_to_hourly:
        test_series = _resample_to_hourly(test_series)

    fit_train, _, _ = _split_by_ratio(fit_series)
    scaler = StandardScaler()
    scaler.fit(fit_train)

    train_all = scaler.transform(train_series)
    test_all = scaler.transform(test_series)
    train_split, val_split, _ = _split_by_ratio(train_all)
    target_train_split, _, test_split = _split_by_ratio(test_all)

    train_ds = ForecastWindowDataset(train_split, seq_len=seq_len, pred_len=pred_len)
    val_ds = ForecastWindowDataset(val_split, seq_len=seq_len, pred_len=pred_len)
    test_ds = ForecastWindowDataset(test_split, seq_len=seq_len, pred_len=pred_len)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_multimodal,
        )

    loaders = {"train": make_loader(train_ds, True), "val": make_loader(val_ds, False), "test": make_loader(test_ds, False)}
    if emit_target_train:
        target_ds = ForecastWindowDataset(target_train_split, seq_len=seq_len, pred_len=pred_len)
        loaders["target_train"] = make_loader(target_ds, True)
    return DataBundle(loaders=loaders, modality_dims=[3, 4], n_classes=0, class_names=None)
