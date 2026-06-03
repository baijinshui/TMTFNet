"""TMTFNet-RevIN on cross-domain ETT forecasting (5 splits, H=24).

RevIN's instance-level normalisation should help most on cross-domain
because per-window mean and std are exactly the moments that shift
between source and target domains. Includes both vanilla TMTFNet-RevIN
(no alignment) and the DA variant for validated-lambda comparisons.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_ett_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import (count_parameters, set_global_seed,  # noqa: E402
                         train_and_evaluate)
from src.trainer_da import train_and_evaluate_da  # noqa: E402

DATA_DIR = str(REPO / "data")
SEEDS = (42, 123, 456, 789, 2024)

CROSS = [
    ("ETTh1", "ETTh2", False, "h1_to_h2"),
    ("ETTh2", "ETTh1", False, "h2_to_h1"),
    ("ETTm1", "ETTm2", False, "m1_to_m2"),
    ("ETTm2", "ETTm1", False, "m2_to_m1"),
    ("ETTh1", "ETTm1", True, "h1_to_m1"),
]

# Per-split lambda configurations chosen by the validation-driven selector
LAMBDA_BY_SPLIT = {
    "h1_to_h2": (0.0, 0.0, 0.0),
    "h2_to_h1": (1.0, 0.1, 0.1),
    "m1_to_m2": (0.0, 0.0, 0.0),
    "m2_to_m1": (0.1, 0.05, 0.05),
    "h1_to_m1": (1.0, 0.1, 0.1),
}


def run_one(src, tgt, resample, label, h=24, max_epochs=100):
    out_dir = REPO / "results" / f"rebuttal_exp3_cross_forecast_pred{h}__{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    accs = []
    lam_c, lam_m, lam_d = LAMBDA_BY_SPLIT[label]
    use_da = (lam_c > 0 or lam_m > 0 or lam_d > 0)
    for seed in SEEDS:
        path = out_dir / f"TMTFNet_v2_revin_seed{seed}.json"
        if path.exists() and path.stat().st_size > 0:
            with path.open() as fh:
                accs.append(json.load(fh).get("mse"))
            continue
        set_global_seed(int(seed))
        bundle = get_ett_loaders(
            DATA_DIR, train_dataset=src, test_dataset=tgt, fit_dataset=src,
            seq_len=96, pred_len=h, num_workers=2,
            resample_test_to_hourly=resample, emit_target_train=use_da,
        )
        set_global_seed(int(seed))
        model = build_model(
            "TMTFNetRevIN",
            modality_dims=bundle.modality_dims,
            d_model=64, n_heads=8, n_enc_layers=2,
            modality_dropout=0.1, pred_len=h, seq_len=96,
            dropout=0.1, target_channel=-1,
            use_domain_adapt=use_da,
        )
        n_params = count_parameters(model)
        t0 = time.time()
        if use_da:
            result = train_and_evaluate_da(
                model=model, loaders=bundle.loaders,
                target_loader=bundle.loaders.get("target_train"),
                task="forecast", device=torch.device("cuda"),
                lr=1e-3, max_epochs=max_epochs, patience=15,
                lambda_coral=lam_c, lambda_consistency=lam_m, lambda_dann=lam_d,
            )
        else:
            result = train_and_evaluate(
                model=model, loaders=bundle.loaders, task="forecast",
                device=torch.device("cuda"), lr=1e-3,
                max_epochs=max_epochs, patience=15,
            )
        result.update({
            "seed": int(seed), "model_name": "TMTFNet_v2_revin",
            "n_params": int(n_params), "wall_time": time.time() - t0,
            "split": label, "lam_c": lam_c, "lam_m": lam_m, "lam_d": lam_d,
        })
        with path.open("w") as fh:
            json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        accs.append(result["mse"])
        print(f"  {label} seed={seed} mse={result['mse']:.4f} (lam={lam_c}/{lam_m}/{lam_d}, t={result['wall_time']:.1f}s)")
    import numpy as np
    arr = np.asarray(accs, dtype=np.float64)
    print(f"  >>> {label} TMTFNet-RevIN 5-seed MSE = {arr.mean():.4f} +- {arr.std(ddof=1):.4f}\n")


def main():
    print(f"started: {time.strftime('%F %T')}")
    for src, tgt, resample, label in CROSS:
        run_one(src, tgt, resample, label)
    print(f"finished: {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
