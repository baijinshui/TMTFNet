"""Train TMTFNet (RevIN), DLinear, and a strong deep baseline once each on
ETTh1, ETTh2, ETTm1 at H=24 (seed 42), then dump per-window ground truth
and predictions for the figure ``forecast_predictions.pdf``.

Output: ``results/forecast_predictions/{dataset}_{model}.npz`` with arrays
``y_true`` (N, H), ``y_pred`` (N, H), ``mse`` (scalar).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets import get_ett_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import set_global_seed, count_parameters  # noqa: E402

DATA_DIR = str(REPO / "data")
OUT = REPO / "results" / "forecast_predictions"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
HORIZON = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_predictions(model, loader):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            modalities, target = batch
            modalities = [m.to(DEVICE) for m in modalities]
            target = target.to(DEVICE)
            out, _ = model(modalities)
            ys.append(target.cpu().numpy())
            preds.append(out.cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(preds, axis=0)
    if y.ndim == 3:  # (N, H, C); take last channel as target if not already
        y = y[:, :, -1]
    if p.ndim == 3:
        p = p[:, :, -1]
    mse = float(((y - p) ** 2).mean())
    return y, p, mse


def train_one(dataset, model_name, model_kwargs, max_epochs=60, lr=1e-3):
    out_path = OUT / f"{dataset}_{model_name}.npz"
    if out_path.exists():
        d = np.load(out_path)
        print(f"  cache: {dataset} {model_name} mse={float(d['mse']):.4f}")
        return
    set_global_seed(SEED)
    bundle = get_ett_loaders(DATA_DIR, train_dataset=dataset,
                              seq_len=96, pred_len=HORIZON, num_workers=2)
    set_global_seed(SEED)
    model = build_model(model_name, modality_dims=bundle.modality_dims,
                        n_classes=bundle.n_classes, **model_kwargs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_left = 12
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        for batch in bundle.loaders["train"]:
            modalities, target = batch
            modalities = [m.to(DEVICE) for m in modalities]
            target = target.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out, _ = model(modalities)
            if out.shape != target.shape and target.ndim == 3:
                target = target[:, :, -1]
            loss = crit(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # validation
        model.eval()
        with torch.no_grad():
            vsum, vcnt = 0.0, 0
            for batch in bundle.loaders["val"]:
                modalities, target = batch
                modalities = [m.to(DEVICE) for m in modalities]
                target = target.to(DEVICE)
                out, _ = model(modalities)
                if out.shape != target.shape and target.ndim == 3:
                    target = target[:, :, -1]
                vsum += float(((out - target) ** 2).mean()) * target.size(0)
                vcnt += target.size(0)
            v = vsum / max(1, vcnt)
        if v < best_val - 1e-5:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            patience_left = 12
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred, mse = collect_predictions(model, bundle.loaders["test"])
    np.savez(out_path, y_true=y_true, y_pred=y_pred, mse=np.array(mse, dtype=np.float64),
             params=count_parameters(model), wall_time=time.time() - t0)
    print(f"  {dataset} {model_name} mse={mse:.4f} t={time.time()-t0:.1f}s -> {out_path.name}")


def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    runs = [
        ("ETTh1", "TMTFNetRevIN", dict(d_model=64, n_heads=8, n_enc_layers=2,
                                        modality_dropout=0.1, pred_len=HORIZON,
                                        seq_len=96, dropout=0.1, target_channel=-1)),
        ("ETTh1", "DLinear", dict(seq_len=96, pred_len=HORIZON)),
        ("ETTh1", "Crossformer", dict(d_model=64, n_heads=8, n_enc_layers=2,
                                       seq_len=96, pred_len=HORIZON)),
        ("ETTh2", "TMTFNetRevIN", dict(d_model=64, n_heads=8, n_enc_layers=2,
                                        modality_dropout=0.1, pred_len=HORIZON,
                                        seq_len=96, dropout=0.1, target_channel=-1)),
        ("ETTh2", "DLinear", dict(seq_len=96, pred_len=HORIZON)),
        ("ETTh2", "Crossformer", dict(d_model=64, n_heads=8, n_enc_layers=2,
                                       seq_len=96, pred_len=HORIZON)),
        ("ETTm1", "TMTFNetRevIN", dict(d_model=64, n_heads=8, n_enc_layers=2,
                                        modality_dropout=0.1, pred_len=HORIZON,
                                        seq_len=96, dropout=0.1, target_channel=-1)),
        ("ETTm1", "DLinear", dict(seq_len=96, pred_len=HORIZON)),
        ("ETTm1", "Crossformer", dict(d_model=64, n_heads=8, n_enc_layers=2,
                                       seq_len=96, pred_len=HORIZON)),
    ]
    print(f"started: {time.strftime('%F %T')}")
    for ds, model_name, kwargs in runs:
        try:
            train_one(ds, model_name, kwargs)
        except Exception as e:
            print(f"FAILED {ds} {model_name}: {e}")
    print(f"finished: {time.strftime('%F %T')}")


if __name__ == "__main__":
    main()
