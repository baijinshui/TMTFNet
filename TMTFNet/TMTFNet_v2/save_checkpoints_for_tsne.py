"""Train two TMTFNet checkpoints on ETTh2->ETTh1 (vanilla and strong-alignment)
and save them to disk so the t-SNE source/target-alignment figure can extract
features from both. Reuses the same hyperparameters and 5-seed protocol as
the main rebuttal matrix; we save only the seed=42 model since the t-SNE
plot uses a single representative model.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_ett_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import set_global_seed, train_and_evaluate  # noqa: E402
from src.trainer_da import train_and_evaluate_da  # noqa: E402

OUT = REPO / "results" / "tsne_checkpoints"
OUT.mkdir(parents=True, exist_ok=True)


def train_one(name, use_da, lam_c, lam_m, lam_d):
    print(f"\n=== {name} ===")
    set_global_seed(42)
    bundle = get_ett_loaders(
        str(REPO / "data"),
        train_dataset="ETTh2", test_dataset="ETTh1", fit_dataset="ETTh2",
        seq_len=96, pred_len=24, num_workers=2, emit_target_train=use_da,
    )
    set_global_seed(42)
    model = build_model(
        "TMTFNet_v2",
        modality_dims=bundle.modality_dims,
        d_model=64, n_heads=8, n_enc_layers=2,
        pred_len=24, seq_len=96, dropout=0.1,
        modality_dropout=0.1, use_domain_adapt=use_da,
    )
    t0 = time.time()
    if use_da:
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
    else:
        result = train_and_evaluate(
            model=model,
            loaders=bundle.loaders,
            task="forecast",
            device=torch.device("cuda"),
            lr=1e-3,
            max_epochs=100,
            patience=15,
        )
    elapsed = time.time() - t0
    out_path = OUT / f"{name}_h2_to_h1_seed42.pt"
    torch.save({"state_dict": model.state_dict(), "result": {k: v for k, v in result.items() if k != "history"},
                "config": dict(d_model=64, n_heads=8, n_enc_layers=2, modality_dropout=0.1, use_domain_adapt=use_da)},
               out_path)
    print(f"  test mse={result.get('mse', float('nan')):.4f} elapsed={elapsed:.1f}s saved={out_path}")


if __name__ == "__main__":
    train_one("vanilla", use_da=False, lam_c=0.0, lam_m=0.0, lam_d=0.0)
    train_one("strong_alignment", use_da=True, lam_c=1.0, lam_m=0.1, lam_d=0.1)
    print("done")
