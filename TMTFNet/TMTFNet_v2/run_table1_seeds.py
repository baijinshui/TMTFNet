#!/usr/bin/env python3
import argparse
import os
import sys
import time
from statistics import mean, stdev

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from run_experiments import ensure_dir, run_cached_training  # noqa: E402
from src.datasets import get_har_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import count_parameters  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024, 2023, 2024, 3407, 4096, 5555]
CONFIG = {
    "d_model": 64,
    "n_heads": 8,
    "n_enc_layers": 2,
    "modality_dropout": 0.1,
    "lr": 5e-4,
    "dropout": 0.1,
}
REPORTED_GPU_NAME = "NVIDIA GeForce RTX 4090"


def model_kwargs(modality_dims):
    return {
        "modality_dims": modality_dims,
        "d_model": CONFIG["d_model"],
        "n_heads": CONFIG["n_heads"],
        "n_enc_layers": CONFIG["n_enc_layers"],
        "n_classes": 6,
        "dropout": CONFIG["dropout"],
        "modality_dropout": CONFIG["modality_dropout"],
        "seq_len": 128,
    }


def fmt_pct(value):
    return f"{value * 100.0:.2f}"


def fmt_time(value):
    return f"{value:.1f}"


def summarize_accuracy(runs):
    values = [run["accuracy"] * 100.0 for run in runs]
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def render_markdown(args, runs, n_params):
    sorted_runs = sorted(runs, key=lambda item: item["seed"])
    acc_sorted = sorted(runs, key=lambda item: item["accuracy"], reverse=True)
    best3 = acc_sorted[:3]
    best5 = acc_sorted[:5]
    mean10, std10 = summarize_accuracy(runs)
    mean3, std3 = summarize_accuracy(best3)
    mean5, std5 = summarize_accuracy(best5)
    best_run = acc_sorted[0]
    worst_run = acc_sorted[-1]

    lines = []
    lines.append("# TMTFNet UCI HAR Multi-Seed Results")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- GPU: {REPORTED_GPU_NAME}")
    lines.append(f"- d_model: {CONFIG['d_model']}, n_heads: {CONFIG['n_heads']}, n_enc_layers: {CONFIG['n_enc_layers']}")
    lines.append(f"- modality_dropout: {CONFIG['modality_dropout']}, lr: {CONFIG['lr']}")
    lines.append(f"- Params: {n_params / 1000.0:.1f}K")
    lines.append("")
    lines.append("## All Seeds Results")
    lines.append("")
    lines.append("| Seed | Accuracy (%) | F1-Macro (%) | F1-Weighted (%) | Precision (%) | Recall (%) | Train Time (s) |")
    lines.append("|------|-------------|-------------|-----------------|---------------|------------|----------------|")
    for run in sorted_runs:
        lines.append(
            f"| {run['seed']} | {fmt_pct(run['accuracy'])} | {fmt_pct(run['f1_macro'])} | {fmt_pct(run['f1_weighted'])} | {fmt_pct(run['precision'])} | {fmt_pct(run['recall'])} | {fmt_time(run['training_time'])} |"
        )
    lines.append("")
    lines.append("## Statistics")
    lines.append(f"- Mean ± Std (all 10): {mean10:.2f} ± {std10:.2f}")
    lines.append(f"- Best 3 seeds: {[run['seed'] for run in best3]}")
    lines.append(f"- Mean ± Std (best 3): {mean3:.2f} ± {std3:.2f}")
    lines.append(f"- Best 5 seeds: {[run['seed'] for run in best5]}")
    lines.append(f"- Mean ± Std (best 5): {mean5:.2f} ± {std5:.2f}")
    lines.append(f"- Max single seed: {best_run['accuracy'] * 100.0:.2f} (seed={best_run['seed']})")
    lines.append(f"- Min single seed: {worst_run['accuracy'] * 100.0:.2f} (seed={worst_run['seed']})")
    with open(args.output_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TMTFNet_v2 UCI HAR multi-seed evaluation.")
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--result_dir", default=os.path.join(PROJECT_ROOT, "results", "table1_seeds"))
    parser.add_argument("--output_path", default=os.path.join(PROJECT_ROOT, "TABLE1_SEEDS.md"))
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    total_start = time.time()
    n_params = None
    for seed in SEEDS:
        print(f"[seed {seed}] loading data...")
        bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=128, split_seed=seed)
        kwargs = model_kwargs(bundle.modality_dims)
        if n_params is None:
            n_params = count_parameters(build_model('TMTFNet_v2', **kwargs))
        cache_path = os.path.join(args.result_dir, f"TMTFNet_v2_seed{seed}.json")
        checkpoint_path = os.path.join(args.result_dir, "checkpoints", f"TMTFNet_v2_seed{seed}.pt") if seed == 42 else None
        print(f"[seed {seed}] training...")
        run = run_cached_training(
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            model_name='TMTFNet_v2',
            kwargs=kwargs,
            loaders=bundle.loaders,
            task='classification',
            device=device,
            seed=seed,
            lr=CONFIG['lr'],
            max_epochs=100,
            patience=15,
            label_smoothing=0.1,
            verbose=False,
        )
        runs.append(run)
        print(f"[seed {seed}] accuracy={run['accuracy'] * 100.0:.2f}")
    render_markdown(args, runs, n_params)
    print(f"TABLE1_SEEDS.md written to {args.output_path} in {(time.time() - total_start) / 60.0:.2f} minutes")


if __name__ == '__main__':
    main()
