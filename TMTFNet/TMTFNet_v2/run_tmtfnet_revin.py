"""TMTFNet with reversible instance normalization (RevIN) on 9 ETT cells.

RevIN is a standard preprocessing trick used by modern long-horizon
forecasting methods (PatchTST, iTransformer, FEDformer): every input
window is per-channel zero-meaned and unit-scaled before the model
forward, and the prediction is rescaled back using the input
statistics. Expected gain on ETT: 5-10% MSE reduction.
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

DATA_DIR = str(REPO / "data")
SEEDS = (42, 123, 456, 789, 2024)
DATASETS = ("ETTh1", "ETTh2", "ETTm1")
HORIZONS = (24, 48, 96)


def run_one(dataset, horizon, max_epochs=100):
    out_dir = REPO / "results" / f"rebuttal_exp2_forecast_{dataset.lower()}_pred{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    accs = []
    for seed in SEEDS:
        path = out_dir / f"TMTFNet_v2_revin_seed{seed}.json"
        if path.exists() and path.stat().st_size > 0:
            with path.open() as fh:
                accs.append(json.load(fh).get("mse"))
            continue
        set_global_seed(int(seed))
        bundle = get_ett_loaders(DATA_DIR, train_dataset=dataset,
                                  seq_len=96, pred_len=horizon, num_workers=2)
        set_global_seed(int(seed))
        model = build_model(
            "TMTFNetRevIN",
            modality_dims=bundle.modality_dims,
            d_model=64, n_heads=8, n_enc_layers=2,
            modality_dropout=0.1, pred_len=horizon,
            seq_len=96, dropout=0.1,
            target_channel=-1,
        )
        n_params = count_parameters(model)
        t0 = time.time()
        result = train_and_evaluate(
            model=model, loaders=bundle.loaders, task="forecast",
            device=torch.device("cuda"), lr=1e-3,
            max_epochs=max_epochs, patience=15,
        )
        result.update({"seed": int(seed), "model_name": "TMTFNet_v2_revin",
                       "n_params": int(n_params), "wall_time": time.time() - t0})
        with path.open("w") as fh:
            json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        accs.append(result["mse"])
        print(f"  {dataset} H={horizon} seed={seed} mse={result['mse']:.4f} (t={result['wall_time']:.1f}s)")
    import numpy as np
    arr = np.asarray(accs, dtype=np.float64)
    print(f"  >>> {dataset} H={horizon} TMTFNet-RevIN 5-seed MSE = {arr.mean():.4f} +- {arr.std(ddof=1):.4f}\n")


def main():
    print(f"started: {time.strftime('%F %T')}")
    for ds in DATASETS:
        for h in HORIZONS:
            run_one(ds, h)
    print(f"finished: {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
