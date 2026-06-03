"""Targeted retraining for the rebuttal weak points.

Two independent jobs:
1. Lambda sweep: rerun TMTFNet$_{\text{DA}}$ on every cross-domain split with a
   weaker alignment configuration (lambda_C=0.1, lambda_M=0.05, lambda_D=0.05),
   to test whether the over-regularisation observed on m-direction transfers
   can be cured without a per-direction grid search. Adds 5 seeds x 5 splits
   = 25 new runs to ``rebuttal_exp3_cross_da_pred24__<split>/``.
2. DeepConvLSTM uprated: rerun the canonical HAR baseline with the larger
   filter / layer configuration that the original Ordonez and Roggen paper
   reports, to bring our DeepConvLSTM number in line with the published
   ~94% baseline rather than the current ~91%.

Both jobs are append-only and use the same ``_done(path)`` skip logic as the
main runner, so safe to restart.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_ett_loaders, get_har_loaders, get_pamap2_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import (count_parameters, set_global_seed,  # noqa: E402
                         train_and_evaluate)
from src.trainer_da import train_and_evaluate_da  # noqa: E402

DATA_DIR = str(REPO / "data")
RESULTS = REPO / "results"

SEEDS = (42, 123, 456, 789, 2024)

# (src, tgt, resample, label)
SPLITS = [
    ("ETTh1", "ETTh2", False, "h1_to_h2"),
    ("ETTh2", "ETTh1", False, "h2_to_h1"),
    ("ETTm1", "ETTm2", False, "m1_to_m2"),
    ("ETTm2", "ETTm1", False, "m2_to_m1"),
    ("ETTh1", "ETTm1", True, "h1_to_m1"),
]


def _done(path):
    return path.exists() and path.stat().st_size > 0


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))


def lambda_sweep_da(lam_c=0.1, lam_m=0.05, lam_d=0.05, h=24):
    """Run TMTFNet_DA with weaker alignment lambdas on all 5 splits."""
    out_dir_base = RESULTS / f"rebuttal_exp3_cross_da_pred{h}"
    suffix = f"weak_lam_C{lam_c}_M{lam_m}_D{lam_d}".replace(".", "p")
    print(f"\n=== Lambda sweep ({suffix}) ===")
    for src, tgt, resample, label in SPLITS:
        for seed in SEEDS:
            split_dir = RESULTS / f"rebuttal_exp3_cross_da_pred{h}__{label}"
            tag = f"{label}_TMTFNet_v2_DA_{suffix}_seed{seed}"
            path = split_dir / f"{tag}.json"
            if _done(path):
                continue
            set_global_seed(int(seed))
            bundle = get_ett_loaders(
                DATA_DIR, train_dataset=src, test_dataset=tgt, fit_dataset=src,
                seq_len=96, pred_len=h, num_workers=2,
                resample_test_to_hourly=resample, emit_target_train=True,
            )
            set_global_seed(int(seed))
            model = build_model(
                "TMTFNet_v2",
                modality_dims=bundle.modality_dims,
                d_model=64, n_heads=8, n_enc_layers=2,
                pred_len=h, seq_len=96, dropout=0.1,
                modality_dropout=0.1, use_domain_adapt=True,
            )
            n_params = count_parameters(model)
            t0 = time.time()
            result = train_and_evaluate_da(
                model=model,
                loaders=bundle.loaders,
                target_loader=bundle.loaders.get("target_train"),
                task="forecast",
                device=torch.device("cuda"),
                lr=1e-3,
                max_epochs=100,
                patience=15,
                lambda_coral=lam_c,
                lambda_consistency=lam_m,
                lambda_dann=lam_d,
            )
            result.update({
                "seed": int(seed), "model_name": "TMTFNet_v2_DA",
                "n_params": int(n_params), "split": label,
                "source": src, "target": tgt,
                "lambda_coral": lam_c, "lambda_consistency": lam_m, "lambda_dann": lam_d,
                "wall_time": time.time() - t0,
            })
            _save_json(path, result)
            print(f"  {label} seed={seed} mse={result.get('mse', float('nan')):.4f} ({suffix})")


def deepconvlstm_uprated():
    """Bigger DeepConvLSTM (n_conv_filters=128, n_lstm_layers=3) on UCI HAR + PAMAP2."""
    print("\n=== DeepConvLSTM uprated ===")
    for dataset_name, exp_name, n_classes, build_fn, max_epochs in [
        ("UCI HAR", "exp1a_har_uci", 6, lambda seed: get_har_loaders(DATA_DIR, batch_size=64, num_workers=2, seq_len=128, split_seed=int(seed)), 100),
        ("PAMAP2", "exp1c_har_pamap2", 12, lambda seed: get_pamap2_loaders(os.path.join(DATA_DIR, "PAMAP2_Dataset"), batch_size=64, num_workers=2, split_seed=int(seed)), 60),
    ]:
        for seed in SEEDS:
            tag = f"DeepConvLSTM_uprated_seed{seed}"
            path = RESULTS / f"rebuttal_{exp_name}" / f"{tag}.json"
            if _done(path):
                continue
            set_global_seed(int(seed))
            bundle = build_fn(seed)
            set_global_seed(int(seed))
            model = build_model(
                "DeepConvLSTM",
                modality_dims=bundle.modality_dims,
                d_model=128,
                n_classes=bundle.n_classes,
                dropout=0.1,
                n_conv_filters=128,
                n_lstm_layers=3,
            )
            n_params = count_parameters(model)
            t0 = time.time()
            result = train_and_evaluate(
                model=model,
                loaders=bundle.loaders,
                task="classification",
                device=torch.device("cuda"),
                lr=5e-4,
                max_epochs=max_epochs,
                patience=15,
                label_smoothing=0.1,
            )
            result.update({"seed": int(seed), "model_name": "DeepConvLSTM_uprated",
                           "n_params": int(n_params), "wall_time": time.time() - t0})
            _save_json(path, result)
            print(f"  {dataset_name} seed={seed} acc={result['accuracy']:.4f}")


if __name__ == "__main__":
    print("Started:", time.strftime("%F %T"))
    deepconvlstm_uprated()  # cheaper, run first
    lambda_sweep_da()
    print("Finished:", time.strftime("%F %T"))
