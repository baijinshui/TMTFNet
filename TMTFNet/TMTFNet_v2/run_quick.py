"""Quick single-(model, dataset, seed) trainer using the GPU-resident fast path.

Use this for any future ad-hoc reruns -- adding a new baseline, replying to
a reviewer who asks for a different hyperparameter, etc. -- without touching
the validated 5-seed rebuttal matrix.

Examples::

    # train TMTFNet_v2 on UCI HAR, 1 seed, fast path
    python run_quick.py --model TMTFNet_v2 --dataset har_uci --seed 42

    # train PatchTST on PAMAP2 cross-subject
    python run_quick.py --model PatchTST --dataset pamap2 --seed 123

    # train TMTFNet_v2 forecast on ETTh1 cross-domain ETTh1->ETTh2
    python run_quick.py --model TMTFNet_v2 --dataset ett --src ETTh1 --tgt ETTh2 --pred-len 24 --seed 42

Outputs are saved under ``results/quick_runs/<dataset>_<model>_seed<seed>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.datasets_fast import (get_ett_bundle_gpu, get_har_bundle_gpu,  # noqa: E402
                                get_pamap2_bundle_gpu)
from src.models import build_model  # noqa: E402
from src.trainer import count_parameters, set_global_seed  # noqa: E402
from src.trainer_fast import train_and_evaluate_fast  # noqa: E402

OUT_DIR = REPO / "results" / "quick_runs"
DATA_DIR = REPO / "data"


def _build_bundle(args, device):
    if args.dataset == "har_uci":
        return get_har_bundle_gpu(str(DATA_DIR), batch_size=args.batch_size, seq_len=128,
                                  split_seed=args.seed, device=device)
    if args.dataset == "pamap2":
        return get_pamap2_bundle_gpu(str(DATA_DIR / "PAMAP2_Dataset"),
                                     batch_size=args.batch_size, split_seed=args.seed,
                                     device=device)
    if args.dataset == "ett":
        return get_ett_bundle_gpu(
            str(DATA_DIR), train_dataset=args.src, test_dataset=args.tgt,
            fit_dataset=args.src, batch_size=args.batch_size, seq_len=96,
            pred_len=args.pred_len, resample_test_to_hourly=args.resample,
            emit_target_train=False, device=device,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name from MODEL_REGISTRY")
    parser.add_argument("--dataset", required=True, choices=["har_uci", "pamap2", "ett"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-enc-layers", type=int, default=2)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    # ETT-only
    parser.add_argument("--src", default="ETTh1")
    parser.add_argument("--tgt", default=None)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    bundle = _build_bundle(args, device)

    is_classification = bundle.n_classes > 0
    task = "classification" if is_classification else "forecast"
    if args.lr is None:
        args.lr = 5e-4 if is_classification else 1e-3

    d_model = args.d_model or (64 if args.model == "TMTFNet_v2" else 128)
    cfg = dict(
        modality_dims=bundle.modality_dims,
        d_model=d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers,
        modality_dropout=args.modality_dropout,
        dropout=0.1,
        seq_len=bundle.train.modalities[0].size(1),
    )
    if is_classification:
        cfg["n_classes"] = bundle.n_classes
    else:
        cfg["pred_len"] = args.pred_len

    set_global_seed(args.seed)
    model = build_model(args.model, **cfg)
    n_params = count_parameters(model)

    start = time.time()
    result = train_and_evaluate_fast(
        model=model,
        bundle=bundle,
        task=task,
        device=device,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        label_smoothing=0.1 if is_classification else 0.0,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=True,
    )
    wall = time.time() - start

    result.update(
        {
            "wall_time": wall,
            "model_name": args.model,
            "dataset": args.dataset,
            "seed": args.seed,
            "n_params": n_params,
            "config": cfg,
        }
    )
    if args.dataset == "ett":
        result.update({"src": args.src, "tgt": args.tgt or args.src,
                       "pred_len": args.pred_len, "resample": args.resample})

    name = args.out or f"{args.dataset}_{args.model}_seed{args.seed}.json"
    path = OUT_DIR / name
    with path.open("w") as fh:
        json.dump(result, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\nDone. wall={wall:.1f}s, epochs={result['epochs_trained']}, wrote {path}")
    if is_classification:
        print(f"  test_acc={result['accuracy']:.4f}, f1_macro={result['f1_macro']:.4f}")
    else:
        print(f"  test_mse={result['mse']:.4f}, test_mae={result['mae']:.4f}")


if __name__ == "__main__":
    main()
