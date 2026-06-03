"""TMTFNet-Ultra: TMTFNet with PatchTST-style per-channel encoders inside each
modality, combined with SWA + mixup. Targeting PAMAP2 cross-subject HAR
where PatchTST's per-channel sample-efficiency is the dominant factor."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_har_loaders, get_pamap2_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import (count_parameters, set_global_seed,  # noqa: E402
                         train_and_evaluate_swa)

DATA_DIR = str(REPO / "data")
SEEDS = (42, 123, 456, 789, 2024)


def run_har(out_dir, max_epochs=120, mixup=0.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    accs = []
    for seed in SEEDS:
        path = out_dir / f"TMTFNet_ultra_seed{seed}.json"
        if path.exists() and path.stat().st_size > 0:
            with path.open() as fh:
                accs.append(json.load(fh).get("accuracy"))
            print(f"  HAR ultra seed={seed} cached acc={accs[-1]:.4f}")
            continue
        set_global_seed(int(seed))
        bundle = get_har_loaders(DATA_DIR, batch_size=64, num_workers=2, seq_len=128, split_seed=int(seed))
        set_global_seed(int(seed))
        model = build_model(
            "TMTFNetUltra",
            modality_dims=bundle.modality_dims,
            d_model=64, n_heads=8, n_enc_layers=2,
            modality_dropout=0.1, n_classes=bundle.n_classes,
            dropout=0.1, seq_len=128,
            patch_len=8, patch_stride=4,
        )
        n_params = count_parameters(model)
        t0 = time.time()
        result = train_and_evaluate_swa(
            model=model, loaders=bundle.loaders, task="classification",
            device=torch.device("cuda"), lr=5e-4,
            max_epochs=max_epochs, patience=20, label_smoothing=0.1,
            swa_start_frac=0.7, mixup_alpha=mixup,
        )
        result.update({"seed": int(seed), "model_name": "TMTFNet_ultra", "n_params": int(n_params),
                       "wall_time": time.time() - t0})
        with path.open("w") as fh:
            json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        accs.append(result["accuracy"])
        print(f"  HAR ultra seed={seed} acc={result['accuracy']:.4f} (n_params={n_params}, t={result['wall_time']:.1f}s)")

    import numpy as np
    arr = np.asarray(accs, dtype=np.float64)
    print(f"\n=== UCI HAR TMTFNet-Ultra 5-seed ===")
    print(f"mean={arr.mean()*100:.2f}, std={arr.std(ddof=1)*100:.2f}")
    print(f"vs PatchTST baseline 94.18 ± 0.47")
    if arr.mean() > 0.9418:
        print(">>> WIN: TMTFNet-Ultra > PatchTST <<<")
    else:
        print(">>> trails PatchTST <<<")


def run_pamap2(out_dir, max_epochs=80, mixup=0.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    accs = []
    for seed in SEEDS:
        path = out_dir / f"TMTFNet_ultra_seed{seed}.json"
        if path.exists() and path.stat().st_size > 0:
            with path.open() as fh:
                accs.append(json.load(fh).get("accuracy"))
            print(f"  PAMAP2 ultra seed={seed} cached acc={accs[-1]:.4f}")
            continue
        set_global_seed(int(seed))
        bundle = get_pamap2_loaders(os.path.join(DATA_DIR, "PAMAP2_Dataset"),
                                     batch_size=64, num_workers=2, split_seed=int(seed))
        set_global_seed(int(seed))
        model = build_model(
            "TMTFNetUltra",
            modality_dims=bundle.modality_dims,
            d_model=64, n_heads=8, n_enc_layers=2,
            modality_dropout=0.1, n_classes=bundle.n_classes,
            dropout=0.1, seq_len=171,
            patch_len=16, patch_stride=8,
        )
        n_params = count_parameters(model)
        t0 = time.time()
        result = train_and_evaluate_swa(
            model=model, loaders=bundle.loaders, task="classification",
            device=torch.device("cuda"), lr=5e-4,
            max_epochs=max_epochs, patience=20, label_smoothing=0.1,
            swa_start_frac=0.7, mixup_alpha=mixup,
        )
        result.update({"seed": int(seed), "model_name": "TMTFNet_ultra", "n_params": int(n_params),
                       "wall_time": time.time() - t0})
        with path.open("w") as fh:
            json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        accs.append(result["accuracy"])
        print(f"  PAMAP2 ultra seed={seed} acc={result['accuracy']:.4f} (n_params={n_params}, t={result['wall_time']:.1f}s)")

    import numpy as np
    arr = np.asarray(accs, dtype=np.float64)
    print(f"\n=== PAMAP2 TMTFNet-Ultra 5-seed ===")
    print(f"mean={arr.mean()*100:.2f}, std={arr.std(ddof=1)*100:.2f}")
    print(f"vs PatchTST baseline 87.53 ± 1.44")
    if arr.mean() > 0.8753:
        print(">>> WIN: TMTFNet-Ultra > PatchTST <<<")
    else:
        print(">>> trails PatchTST <<<")


if __name__ == "__main__":
    print(f"started: {time.strftime('%F %T')}")
    # PAMAP2 first since it's the harder benchmark
    run_pamap2(REPO / "results" / "rebuttal_exp1c_har_pamap2_ultra")
    run_har(REPO / "results" / "rebuttal_exp1a_har_uci_ultra")
    print(f"finished: {time.strftime('%F %T')}")
