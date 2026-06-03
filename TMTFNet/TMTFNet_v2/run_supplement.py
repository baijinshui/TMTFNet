#!/usr/bin/env python3
import argparse
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from run_experiments import (  # noqa: E402
    CLASSIFICATION_MODELS,
    cache_tag,
    ensure_dir,
    load_json,
    model_kwargs,
    run_cached_training,
    save_json,
    setup_device,
)
from src.datasets import get_ett_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import Trainer, count_parameters, set_global_seed  # noqa: E402


SEED = 42
LR = 1e-3
BATCH_SIZE = 64
SEQ_LEN = 96
PRED_LEN = 24
MAX_EPOCHS = 100
PATIENCE = 15


def format_value(value):
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def config_tag(config):
    ordered = ["d_model", "n_heads", "n_enc_layers", "modality_dropout"]
    return "_".join(f"{key}{format_value(config[key])}" for key in ordered)


def tmt_kwargs(modality_dims, config, pred_len=PRED_LEN):
    return {
        "modality_dims": modality_dims,
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_enc_layers": config["n_enc_layers"],
        "pred_len": pred_len,
        "dropout": 0.1,
        "seq_len": SEQ_LEN,
        "modality_dropout": config["modality_dropout"],
    }


def sensitivity_sections():
    return {
        "d_model": [
            {"d_model": 32, "n_heads": 4, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 64, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 256, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1},
        ],
        "n_heads": [
            {"d_model": 128, "n_heads": 2, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 4, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 16, "n_enc_layers": 3, "modality_dropout": 0.1},
        ],
        "n_enc_layers": [
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 1, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 2, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 4, "modality_dropout": 0.1},
        ],
        "modality_dropout": [
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.0},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.05},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.15},
            {"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.2},
        ],
    }


def fmt_float(value):
    return f"{value:.4f}"


def run_training_with_checkpoint(
    cache_path,
    checkpoint_path,
    kwargs,
    loaders,
    device,
):
    cached = load_json(cache_path)
    if cached is not None and os.path.exists(checkpoint_path):
        return cached
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return run_cached_training(
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        model_name="TMTFNet_v2",
        kwargs=kwargs,
        loaders=loaders,
        task="forecasting",
        device=device,
        seed=SEED,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        verbose=False,
    )


def cross_evaluate(cache_path, checkpoint_path, kwargs, loaders, device):
    cached = load_json(cache_path)
    if cached is not None:
        return cached

    set_global_seed(SEED)
    model = build_model("TMTFNet_v2", **kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    trainer = Trainer(model, task="forecasting", device=device, lr=LR)
    metrics = trainer.evaluate(loaders["test"])
    result = {
        "seed": SEED,
        "n_params": count_parameters(model),
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }
    save_json(cache_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_sensitivity_ab(args, device):
    root_dir = os.path.join(args.result_dir, "supplement_sensitivity")
    run_dir = os.path.join(root_dir, "runs")
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    source_bundle = get_ett_loaders(
        args.data_dir,
        train_dataset="ETTh1",
        test_dataset="ETTh1",
        fit_dataset="ETTh1",
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )
    cross_bundle = get_ett_loaders(
        args.data_dir,
        train_dataset="ETTh1",
        test_dataset="ETTh2",
        fit_dataset="ETTh1",
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
    )

    sections = sensitivity_sections()
    unique_configs = {}
    for configs in sections.values():
        for config in configs:
            unique_configs[config_tag(config)] = config

    results_by_tag = {}
    for tag, config in unique_configs.items():
        kwargs = tmt_kwargs(source_bundle.modality_dims, config)
        source_cache = os.path.join(run_dir, f"{tag}_etth1_source.json")
        cross_cache = os.path.join(run_dir, f"{tag}_etth1_to_etth2.json")
        checkpoint_path = os.path.join(ckpt_dir, f"{tag}.pt")
        source_result = run_training_with_checkpoint(
            cache_path=source_cache,
            checkpoint_path=checkpoint_path,
            kwargs=kwargs,
            loaders=source_bundle.loaders,
            device=device,
        )
        cross_result = cross_evaluate(
            cache_path=cross_cache,
            checkpoint_path=checkpoint_path,
            kwargs=kwargs,
            loaders=cross_bundle.loaders,
            device=device,
        )
        results_by_tag[tag] = {"config": config, "source": source_result, "cross": cross_result}

    save_json(os.path.join(root_dir, "summary.json"), results_by_tag)
    return sections, results_by_tag


def run_experiment_c(args, device):
    root_dir = os.path.join(args.result_dir, "supplement_ettm1")
    ensure_dir(root_dir)
    pred_lens = [24, 48, 96]
    results = {model_name: {} for model_name in CLASSIFICATION_MODELS}
    for pred_len in pred_lens:
        bundle = get_ett_loaders(
            args.data_dir,
            train_dataset="ETTm1",
            test_dataset="ETTm1",
            fit_dataset="ETTm1",
            batch_size=BATCH_SIZE,
            num_workers=args.num_workers,
            seq_len=SEQ_LEN,
            pred_len=pred_len,
        )
        horizon_dir = os.path.join(root_dir, f"H{pred_len}")
        ensure_dir(horizon_dir)
        for model_name in CLASSIFICATION_MODELS:
            run_path = os.path.join(horizon_dir, f"{model_name}_seed{SEED}_lr{cache_tag(LR)}.json")
            result = run_cached_training(
                cache_path=run_path,
                checkpoint_path=None,
                model_name=model_name,
                kwargs=model_kwargs(model_name, bundle.modality_dims, "forecasting", seq_len=SEQ_LEN, pred_len=pred_len),
                loaders=bundle.loaders,
                task="forecasting",
                device=device,
                seed=SEED,
                lr=LR,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                verbose=False,
            )
            results[model_name][f"H{pred_len}"] = result
    save_json(os.path.join(root_dir, "summary.json"), results)
    return results


def render_markdown(args, gpu_name, total_time_s, sections, results_by_tag, exp_c_results):
    lines = []
    lines.append("# TMTFNet_v2 Supplementary Experimental Results")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- GPU: {gpu_name}")
    lines.append(f"- Total time: {total_time_s / 60.0:.2f} minutes")
    lines.append("")

    lines.append("## Experiment A: Hyperparameter Sensitivity on ETTh1 Forecasting (H=24, seed=42)")
    lines.append("")
    lines.append("### A1: Effect of d_model")
    lines.append("| d_model | Params (K) | MSE | MAE |")
    lines.append("|---------|-----------|-----|-----|")
    for config in sections["d_model"]:
        result = results_by_tag[config_tag(config)]["source"]
        lines.append(f"| {config['d_model']} | {result['n_params'] / 1000.0:.1f} | {fmt_float(result['mse'])} | {fmt_float(result['mae'])} |")
    lines.append("")

    lines.append("### A2: Effect of n_heads")
    lines.append("| n_heads | MSE | MAE |")
    lines.append("|---------|-----|-----|")
    for config in sections["n_heads"]:
        result = results_by_tag[config_tag(config)]["source"]
        lines.append(f"| {config['n_heads']} | {fmt_float(result['mse'])} | {fmt_float(result['mae'])} |")
    lines.append("")

    lines.append("### A3: Effect of n_enc_layers")
    lines.append("| n_enc_layers | Params (K) | MSE | MAE |")
    lines.append("|-------------|-----------|-----|-----|")
    for config in sections["n_enc_layers"]:
        result = results_by_tag[config_tag(config)]["source"]
        lines.append(f"| {config['n_enc_layers']} | {result['n_params'] / 1000.0:.1f} | {fmt_float(result['mse'])} | {fmt_float(result['mae'])} |")
    lines.append("")

    lines.append("### A4: Effect of modality_dropout")
    lines.append("| modality_dropout | MSE | MAE |")
    lines.append("|-----------------|-----|-----|")
    for config in sections["modality_dropout"]:
        result = results_by_tag[config_tag(config)]["source"]
        lines.append(f"| {config['modality_dropout']:.2f} | {fmt_float(result['mse'])} | {fmt_float(result['mae'])} |")
    lines.append("")

    lines.append("## Experiment B: Hyperparameter Sensitivity on ETTh1→ETTh2 Cross-Domain (H=24, seed=42)")
    lines.append("")
    lines.append("### B1: Effect of d_model")
    lines.append("| d_model | Cross-Domain MSE |")
    lines.append("|---------|-----------------|")
    for config in sections["d_model"]:
        result = results_by_tag[config_tag(config)]["cross"]
        lines.append(f"| {config['d_model']} | {fmt_float(result['mse'])} |")
    lines.append("")

    lines.append("### B2: Effect of n_heads")
    lines.append("| n_heads | Cross-Domain MSE |")
    lines.append("|---------|-----------------|")
    for config in sections["n_heads"]:
        result = results_by_tag[config_tag(config)]["cross"]
        lines.append(f"| {config['n_heads']} | {fmt_float(result['mse'])} |")
    lines.append("")

    lines.append("### B3: Effect of n_enc_layers")
    lines.append("| n_enc_layers | Cross-Domain MSE |")
    lines.append("|-------------|-----------------|")
    for config in sections["n_enc_layers"]:
        result = results_by_tag[config_tag(config)]["cross"]
        lines.append(f"| {config['n_enc_layers']} | {fmt_float(result['mse'])} |")
    lines.append("")

    lines.append("### B4: Effect of modality_dropout")
    lines.append("| modality_dropout | Cross-Domain MSE |")
    lines.append("|-----------------|-----------------|")
    for config in sections["modality_dropout"]:
        result = results_by_tag[config_tag(config)]["cross"]
        lines.append(f"| {config['modality_dropout']:.2f} | {fmt_float(result['mse'])} |")
    lines.append("")

    lines.append("## Experiment C: ETTm1 Forecasting (optional, seed=42)")
    lines.append("")
    if exp_c_results is None:
        lines.append("Skipped because the supplementary run exceeded the time budget before Experiment C.")
    else:
        lines.append("### ETTm1 Results")
        lines.append("| Model | H=24 MSE | H=24 MAE | H=48 MSE | H=48 MAE | H=96 MSE | H=96 MAE |")
        lines.append("|-------|---------|---------|---------|---------|---------|---------|")
        for model_name in CLASSIFICATION_MODELS:
            h24 = exp_c_results[model_name]["H24"]
            h48 = exp_c_results[model_name]["H48"]
            h96 = exp_c_results[model_name]["H96"]
            lines.append(
                f"| {model_name} | {fmt_float(h24['mse'])} | {fmt_float(h24['mae'])} | "
                f"{fmt_float(h48['mse'])} | {fmt_float(h48['mae'])} | "
                f"{fmt_float(h96['mse'])} | {fmt_float(h96['mae'])} |"
            )
    lines.append("")

    with open(args.supplement_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Run supplementary TMTFNet_v2 experiments.")
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--result_dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--supplement_path", default=os.path.join(PROJECT_ROOT, "SUPPLEMENT_RESULTS.md"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_exp_c", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.result_dir)
    device, gpu_name, _ = setup_device()
    total_start = time.time()

    sections, results_by_tag = run_sensitivity_ab(args, device)
    elapsed_s = time.time() - total_start

    exp_c_results = None
    if not args.skip_exp_c and elapsed_s < 30 * 60:
        exp_c_results = run_experiment_c(args, device)

    total_time_s = time.time() - total_start
    render_markdown(args, gpu_name, total_time_s, sections, results_by_tag, exp_c_results)
    print(f"SUPPLEMENT_RESULTS.md written to {args.supplement_path}")


if __name__ == "__main__":
    main()
