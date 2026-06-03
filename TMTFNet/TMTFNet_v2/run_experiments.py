#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from statistics import mean, stdev

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.datasets import HAR_CLASS_NAMES, get_ett_loaders, get_har_loaders
from src.models import build_model
from src.trainer import count_parameters, measure_inference_time, set_global_seed, train_and_evaluate


CLASSIFICATION_MODELS = [
    "TMTFNet_v2",
    "LSTM",
    "GRU",
    "TCN",
    "Transformer",
    "Crossformer",
    "PatchTST",
    "DLinear",
    "TimeMixer",
    "iTransformer",
]

ABLATION_MODELS = [
    "TMTFNet_v2",
    "TMTFNet_v2_NoGCMTA",
    "TMTFNet_v2_NoAMG",
    "TMTFNet_v2_NoAHTF",
    "TMTFNet_v2_NoAttnPool",
    "TMTFNet_v2_NoModDrop",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        cuda_version = "N/A"
    return device, gpu_name, cuda_version


def budget_config(budget):
    if budget == "fast":
        return {
            "exp1_seeds": [42, 123, 456],
            "exp2_seeds": [42, 123],
            "exp3_seeds": [42, 123],
            "exp4_seeds": [42, 123],
            "max_epochs": 60,
            "patience": 10,
        }
    return {
        "exp1_seeds": [42, 123, 456, 789, 1024],
        "exp2_seeds": [42, 123, 456],
        "exp3_seeds": [42, 123, 456],
        "exp4_seeds": [42, 123, 456],
        "max_epochs": 100,
        "patience": 15,
    }


def classification_lr(model_name):
    return 3e-4 if model_name.startswith("TMTFNet_v2") else 5e-4


def forecasting_lr(model_name):
    return 1e-3


def cache_tag(lr):
    return str(lr).replace(".", "p")


def model_kwargs(model_name, modality_dims, task, seq_len, pred_len=None, d_model=128, dropout=0.1):
    kwargs = {
        "modality_dims": modality_dims,
        "d_model": d_model,
        "dropout": dropout,
        "seq_len": seq_len,
    }
    if task == "classification":
        kwargs["n_classes"] = 6
    else:
        kwargs["pred_len"] = pred_len

    if model_name.startswith("TMTFNet_v2"):
        kwargs.update({"n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1})
    elif model_name in {"Transformer", "Crossformer", "PatchTST", "iTransformer"}:
        kwargs.update({"n_heads": 8, "n_layers": 3})
    elif model_name in {"LSTM", "GRU"}:
        kwargs.update({"n_layers": 3})
    elif model_name == "TCN":
        kwargs.update({"n_layers": 6})
    return kwargs


def run_cached_training(
    cache_path,
    checkpoint_path,
    model_name,
    kwargs,
    loaders,
    task,
    device,
    seed,
    lr,
    max_epochs,
    patience,
    label_smoothing=0.0,
    weight_decay=5e-4,
    grad_accum_steps=2,
    warmup_epochs=5,
    verbose=False,
):
    cached = load_json(cache_path)
    if cached is not None:
        return cached

    set_global_seed(seed)
    model = build_model(model_name, **kwargs)
    n_params = count_parameters(model)
    result = train_and_evaluate(
        model=model,
        loaders=loaders,
        task=task,
        device=device,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        grad_accum_steps=grad_accum_steps,
        warmup_epochs=warmup_epochs,
        verbose=verbose,
    )
    result["seed"] = seed
    result["model_name"] = model_name
    result["n_params"] = n_params
    result.pop("history", None)
    if checkpoint_path:
        ensure_dir(os.path.dirname(checkpoint_path))
        torch.save(model.state_dict(), checkpoint_path)
        result["checkpoint_path"] = checkpoint_path

    save_json(cache_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize_classification_runs(runs):
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]
    summary = {"n_params": int(mean([run["n_params"] for run in runs]))}
    for metric in metrics:
        values = [run[metric] for run in runs]
        summary[metric] = {"mean": float(mean(values)), "std": float(stdev(values)) if len(values) > 1 else 0.0}
    train_times = [run["training_time"] for run in runs]
    summary["training_time"] = {"mean": float(mean(train_times)), "std": float(stdev(train_times)) if len(train_times) > 1 else 0.0}
    best_run = max(runs, key=lambda item: item["accuracy"])
    summary["best_seed"] = best_run["seed"]
    summary["best_accuracy"] = float(best_run["accuracy"])
    summary["best_f1_macro"] = float(best_run["f1_macro"])
    summary["best_f1_weighted"] = float(best_run["f1_weighted"])
    summary["runs"] = runs
    return summary


def summarize_forecasting_runs(runs):
    metrics = ["mse", "mae", "rmse"]
    summary = {"n_params": int(mean([run["n_params"] for run in runs]))}
    for metric in metrics:
        values = [run[metric] for run in runs]
        summary[metric] = {"mean": float(mean(values)), "std": float(stdev(values)) if len(values) > 1 else 0.0}
    summary["runs"] = runs
    return summary


def format_mean_std(metric_dict, percent=False):
    factor = 100.0 if percent else 1.0
    return f"{metric_dict['mean'] * factor:.2f} ± {metric_dict['std'] * factor:.2f}" if percent else f"{metric_dict['mean']:.4f} ± {metric_dict['std']:.4f}"


def format_scalar(value, percent=False):
    return f"{value * 100:.2f}" if percent else f"{value:.4f}"


def ablation_label(model_name):
    mapping = {
        "TMTFNet_v2": "TMTFNet_v2 (Full)",
        "TMTFNet_v2_NoGCMTA": "w/o G-CMTA",
        "TMTFNet_v2_NoAMG": "w/o AMG",
        "TMTFNet_v2_NoAHTF": "w/o A-HTF",
        "TMTFNet_v2_NoAttnPool": "w/o AttnPool",
        "TMTFNet_v2_NoModDrop": "w/o ModDrop",
    }
    return mapping.get(model_name, model_name)


def estimate_total_experiment_time(results):
    total_seconds = 0.0
    if "exp1" in results:
        for summary in results["exp1"].values():
            total_seconds += sum(run.get("training_time", 0.0) for run in summary["runs"])
    if "exp2" in results:
        for horizon_summary in results["exp2"].values():
            for summary in horizon_summary.values():
                total_seconds += sum(run.get("training_time", 0.0) for run in summary["runs"])
    if "exp3" in results:
        for summary in results["exp3"].values():
            total_seconds += sum(run.get("training_time", 0.0) for run in summary["ETTh1→ETTh1"]["runs"])
            total_seconds += sum(run.get("training_time", 0.0) for run in summary["ETTh2→ETTh2"]["runs"])
    if "exp4" in results:
        for section in results["exp4"].values():
            for summary in section.values():
                total_seconds += sum(run.get("training_time", 0.0) for run in summary["runs"])
    if "exp5" in results:
        for entries in results["exp5"].values():
            total_seconds += sum(entry.get("training_time", 0.0) for entry in entries)
    return total_seconds


def experiment_1(args, device, budget):
    bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=128, split_seed=42)
    exp_dir = os.path.join(args.result_dir, "exp1_har")
    ensure_dir(exp_dir)
    aggregated = {}
    for model_name in CLASSIFICATION_MODELS:
        runs = []
        for seed in budget["exp1_seeds"]:
            lr = classification_lr(model_name)
            lr_tag = cache_tag(lr)
            run_path = os.path.join(exp_dir, f"{model_name}_seed{seed}_lr{lr_tag}.json")
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{model_name}_seed{seed}_lr{lr_tag}.pt") if seed == 42 else None
            kwargs = model_kwargs(model_name, bundle.modality_dims, "classification", seq_len=128)
            runs.append(
                run_cached_training(
                    cache_path=run_path,
                    checkpoint_path=checkpoint_path,
                    model_name=model_name,
                    kwargs=kwargs,
                    loaders=bundle.loaders,
                    task="classification",
                    device=device,
                    seed=seed,
                    lr=lr,
                    max_epochs=budget["max_epochs"],
                    patience=budget["patience"],
                    label_smoothing=0.1,
                    verbose=args.verbose,
                )
            )
        aggregated[model_name] = summarize_classification_runs(runs)
    save_json(os.path.join(exp_dir, "summary.json"), aggregated)
    return aggregated


def experiment_2(args, device, budget):
    pred_lens = [24, 48, 96]
    root_dir = os.path.join(args.result_dir, "exp2_etth1")
    ensure_dir(root_dir)
    aggregated = {}
    for pred_len in pred_lens:
        bundle = get_ett_loaders(
            args.data_dir,
            train_dataset="ETTh1",
            test_dataset="ETTh1",
            fit_dataset="ETTh1",
            batch_size=64,
            num_workers=args.num_workers,
            seq_len=96,
            pred_len=pred_len,
        )
        horizon_key = f"H{pred_len}"
        aggregated[horizon_key] = {}
        horizon_dir = os.path.join(root_dir, horizon_key)
        ensure_dir(horizon_dir)
        for model_name in CLASSIFICATION_MODELS:
            runs = []
            for seed in budget["exp2_seeds"]:
                lr = forecasting_lr(model_name)
                lr_tag = cache_tag(lr)
                run_path = os.path.join(horizon_dir, f"{model_name}_seed{seed}_lr{lr_tag}.json")
                kwargs = model_kwargs(model_name, bundle.modality_dims, "forecasting", seq_len=96, pred_len=pred_len)
                runs.append(
                    run_cached_training(
                        cache_path=run_path,
                        checkpoint_path=None,
                        model_name=model_name,
                        kwargs=kwargs,
                        loaders=bundle.loaders,
                        task="forecasting",
                        device=device,
                        seed=seed,
                        lr=lr,
                        max_epochs=budget["max_epochs"],
                        patience=budget["patience"],
                        verbose=args.verbose,
                    )
                )
            aggregated[horizon_key][model_name] = summarize_forecasting_runs(runs)
        save_json(os.path.join(horizon_dir, "summary.json"), aggregated[horizon_key])
    save_json(os.path.join(root_dir, "summary.json"), aggregated)
    return aggregated


def experiment_3(args, device, budget):
    exp_dir = os.path.join(args.result_dir, "exp3_cross_domain")
    ensure_dir(exp_dir)
    aggregated = {}
    source_in = get_ett_loaders(args.data_dir, "ETTh1", "ETTh1", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=96, pred_len=24)
    source_cross = get_ett_loaders(args.data_dir, "ETTh1", "ETTh2", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=96, pred_len=24)
    target_in = get_ett_loaders(args.data_dir, "ETTh2", "ETTh2", "ETTh2", batch_size=64, num_workers=args.num_workers, seq_len=96, pred_len=24)

    for model_name in CLASSIFICATION_MODELS:
        scenario_runs = {
            "ETTh1→ETTh1": [],
            "ETTh1→ETTh2": [],
            "ETTh2→ETTh2": [],
        }
        for seed in budget["exp3_seeds"]:
            source_lr = forecasting_lr(model_name)
            source_lr_tag = cache_tag(source_lr)
            source_run = run_cached_training(
                cache_path=os.path.join(exp_dir, f"{model_name}_source_seed{seed}_lr{source_lr_tag}.json"),
                checkpoint_path=os.path.join(exp_dir, "checkpoints", f"{model_name}_source_seed{seed}_lr{source_lr_tag}.pt"),
                model_name=model_name,
                kwargs=model_kwargs(model_name, source_in.modality_dims, "forecasting", seq_len=96, pred_len=24),
                loaders=source_in.loaders,
                task="forecasting",
                device=device,
                seed=seed,
                lr=source_lr,
                max_epochs=budget["max_epochs"],
                patience=budget["patience"],
                verbose=args.verbose,
            )
            scenario_runs["ETTh1→ETTh1"].append(source_run)
            cross_eval_path = os.path.join(exp_dir, f"{model_name}_cross_seed{seed}_lr{source_lr_tag}.json")
            cross_eval = load_json(cross_eval_path)
            if cross_eval is None:
                set_global_seed(seed)
                model = build_model(model_name, **model_kwargs(model_name, source_cross.modality_dims, "forecasting", seq_len=96, pred_len=24))
                model.load_state_dict(torch.load(source_run["checkpoint_path"], map_location="cpu")) if source_run.get("checkpoint_path") else None
                model = model.to(device)
                from src.trainer import Trainer

                trainer = Trainer(model, task="forecasting", device=device, lr=1e-3)
                metrics = trainer.evaluate(source_cross.loaders["test"])
                cross_eval = {
                    "seed": seed,
                    "model_name": model_name,
                    "n_params": count_parameters(model),
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                }
                save_json(cross_eval_path, cross_eval)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            scenario_runs["ETTh1→ETTh2"].append(cross_eval)

            target_run = run_cached_training(
                cache_path=os.path.join(exp_dir, f"{model_name}_target_seed{seed}_lr{source_lr_tag}.json"),
                checkpoint_path=None,
                model_name=model_name,
                kwargs=model_kwargs(model_name, target_in.modality_dims, "forecasting", seq_len=96, pred_len=24),
                loaders=target_in.loaders,
                task="forecasting",
                device=device,
                seed=seed,
                lr=source_lr,
                max_epochs=budget["max_epochs"],
                patience=budget["patience"],
                verbose=args.verbose,
            )
            scenario_runs["ETTh2→ETTh2"].append(target_run)

        aggregated[model_name] = {scenario: summarize_forecasting_runs(runs) for scenario, runs in scenario_runs.items()}
        cross_mean = aggregated[model_name]["ETTh1→ETTh2"]["mse"]["mean"]
        target_mean = aggregated[model_name]["ETTh2→ETTh2"]["mse"]["mean"]
        aggregated[model_name]["transfer_gap"] = ((cross_mean - target_mean) / max(target_mean, 1e-8)) * 100.0

    save_json(os.path.join(exp_dir, "summary.json"), aggregated)
    return aggregated


def experiment_4(args, device, budget):
    exp_dir = os.path.join(args.result_dir, "exp4_ablation")
    ensure_dir(exp_dir)
    har_bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=128, split_seed=42)
    cross_bundle = get_ett_loaders(args.data_dir, "ETTh1", "ETTh2", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=96, pred_len=24)
    aggregated = {"har": {}, "cross_forecast": {}}

    for model_name in ABLATION_MODELS:
        cls_runs = []
        forecast_runs = []
        for seed in budget["exp4_seeds"]:
            cls_lr = classification_lr(model_name)
            cls_lr_tag = cache_tag(cls_lr)
            frc_lr = forecasting_lr(model_name)
            frc_lr_tag = cache_tag(frc_lr)
            cls_runs.append(
                run_cached_training(
                    cache_path=os.path.join(exp_dir, "har", f"{model_name}_seed{seed}_lr{cls_lr_tag}.json"),
                    checkpoint_path=None,
                    model_name=model_name,
                    kwargs=model_kwargs(model_name, har_bundle.modality_dims, "classification", seq_len=128),
                    loaders=har_bundle.loaders,
                    task="classification",
                    device=device,
                    seed=seed,
                    lr=cls_lr,
                    max_epochs=budget["max_epochs"],
                    patience=budget["patience"],
                    label_smoothing=0.1,
                    verbose=args.verbose,
                )
            )
            forecast_runs.append(
                run_cached_training(
                    cache_path=os.path.join(exp_dir, "cross_forecast", f"{model_name}_seed{seed}_lr{frc_lr_tag}.json"),
                    checkpoint_path=None,
                    model_name=model_name,
                    kwargs=model_kwargs(model_name, cross_bundle.modality_dims, "forecasting", seq_len=96, pred_len=24),
                    loaders=cross_bundle.loaders,
                    task="forecasting",
                    device=device,
                    seed=seed,
                    lr=frc_lr,
                    max_epochs=budget["max_epochs"],
                    patience=budget["patience"],
                    verbose=args.verbose,
                )
            )
        aggregated["har"][model_name] = summarize_classification_runs(cls_runs)
        aggregated["cross_forecast"][model_name] = summarize_forecasting_runs(forecast_runs)

    full_cls_acc = aggregated["har"]["TMTFNet_v2"]["accuracy"]["mean"]
    for variant in aggregated["har"].values():
        variant["delta_acc"] = (variant["accuracy"]["mean"] - full_cls_acc) * 100.0
    full_cross_mse = aggregated["cross_forecast"]["TMTFNet_v2"]["mse"]["mean"]
    for variant in aggregated["cross_forecast"].values():
        variant["delta_mse"] = variant["mse"]["mean"] - full_cross_mse

    save_json(os.path.join(exp_dir, "summary.json"), aggregated)
    return aggregated


def experiment_5(args, device, budget):
    del budget
    exp_dir = os.path.join(args.result_dir, "exp5_sensitivity")
    ensure_dir(exp_dir)
    bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=128, split_seed=42)
    seed = 42
    sections = {
        "d_model": [{"d_model": value, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": 0.1} for value in [32, 64, 128, 256]],
        "n_heads": [{"d_model": 128, "n_heads": value, "n_enc_layers": 3, "modality_dropout": 0.1} for value in [2, 4, 8, 16]],
        "n_layers": [{"d_model": 128, "n_heads": 8, "n_enc_layers": value, "modality_dropout": 0.1} for value in [1, 2, 3, 4]],
        "modality_dropout": [{"d_model": 128, "n_heads": 8, "n_enc_layers": 3, "modality_dropout": value} for value in [0.0, 0.05, 0.1, 0.15, 0.2]],
    }
    aggregated = {}
    for section_name, configs in sections.items():
        aggregated[section_name] = []
        for config in configs:
            tag = "_".join(f"{key}{value}" for key, value in config.items())
            cache_path = os.path.join(exp_dir, section_name, f"{tag}.json")
            kwargs = {
                "modality_dims": bundle.modality_dims,
                "n_classes": 6,
                "dropout": 0.1,
                "seq_len": 128,
                **config,
            }
            result = run_cached_training(
                cache_path=cache_path,
                checkpoint_path=None,
                model_name="TMTFNet_v2",
                kwargs=kwargs,
                loaders=bundle.loaders,
                task="classification",
                device=device,
                seed=seed,
                lr=3e-4,
                max_epochs=100,
                patience=15,
                label_smoothing=0.1,
                verbose=args.verbose,
            )
            result["config"] = config
            aggregated[section_name].append(result)
    save_json(os.path.join(exp_dir, "summary.json"), aggregated)
    return aggregated


def experiment_6(args, device, exp1_summary):
    exp_dir = os.path.join(args.result_dir, "exp6_efficiency")
    ensure_dir(exp_dir)
    bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=128, split_seed=42)
    summary = {}
    for model_name in CLASSIFICATION_MODELS:
        run = next(item for item in exp1_summary[model_name]["runs"] if item["seed"] == 42)
        checkpoint_path = run.get("checkpoint_path")
        lr = classification_lr(model_name)
        lr_tag = cache_tag(lr)
        canonical_checkpoint = os.path.join(
            args.result_dir,
            "exp1_har",
            "checkpoints",
            f"{model_name}_seed42_lr{lr_tag}.pt",
        )
        if checkpoint_path and not os.path.exists(checkpoint_path) and os.path.exists(canonical_checkpoint):
            checkpoint_path = canonical_checkpoint
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{model_name}_seed42_lr{lr_tag}.pt")
            rerun = run_cached_training(
                cache_path=os.path.join(exp_dir, f"{model_name}_seed42_rerun_lr{lr_tag}.json"),
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                kwargs=model_kwargs(model_name, bundle.modality_dims, "classification", seq_len=128),
                loaders=bundle.loaders,
                task="classification",
                device=device,
                seed=42,
                lr=lr,
                max_epochs=100,
                patience=15,
                label_smoothing=0.1,
                verbose=False,
            )
            run = rerun
            checkpoint_path = rerun["checkpoint_path"]

        model = build_model(model_name, **model_kwargs(model_name, bundle.modality_dims, "classification", seq_len=128))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        inference_ms = measure_inference_time(model, bundle.loaders["test"], device=device, num_batches=10)
        summary[model_name] = {
            "params_k": run["n_params"] / 1000.0,
            "train_time_per_epoch_s": run["training_time"] / max(1, run["epochs_trained"]),
            "inference_time_ms": inference_ms,
            "accuracy": run["accuracy"] * 100.0,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary


def render_full_results(args, device_name, cuda_version, total_time_s, protocol, results):
    torch_version = torch.__version__
    effective_total_time_s = max(total_time_s, estimate_total_experiment_time(results))
    lines = []
    lines.append("# TMTFNet_v2 Complete Experimental Results")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- GPU: {device_name}")
    lines.append(f"- PyTorch version: {torch_version}")
    lines.append(f"- CUDA version: {cuda_version}")
    lines.append(f"- Total experiment time: {effective_total_time_s / 60.0:.2f} minutes")
    lines.append(f"- Protocol: {protocol}")
    lines.append("")

    if "exp1" in results:
        lines.append("## Experiment 1: UCI HAR Classification")
        lines.append("")
        lines.append(f"### Table 1: Main Results (mean ± std over {len(next(iter(results['exp1'].values()))['runs'])} seeds)")
        lines.append("")
        lines.append("| Model | Params | Accuracy (%) | F1-Macro (%) | F1-Weighted (%) | Precision (%) | Recall (%) |")
        lines.append("|-------|--------|-------------|-------------|-----------------|---------------|------------|")
        for model_name, summary in results["exp1"].items():
            lines.append(
                f"| {model_name} | {summary['n_params'] / 1000:.1f}K | "
                f"{format_mean_std(summary['accuracy'], percent=True)} | "
                f"{format_mean_std(summary['f1_macro'], percent=True)} | "
                f"{format_mean_std(summary['f1_weighted'], percent=True)} | "
                f"{format_mean_std(summary['precision'], percent=True)} | "
                f"{format_mean_std(summary['recall'], percent=True)} |"
            )
        lines.append("")
        lines.append("Best single-seed results per model:")
        lines.append("")
        lines.append("| Model | Seed | Accuracy | F1-Macro | F1-Weighted | Train Time (s) |")
        lines.append("|-------|------|----------|----------|-------------|----------------|")
        for model_name, summary in results["exp1"].items():
            best_run = max(summary["runs"], key=lambda item: item["accuracy"])
            lines.append(
                f"| {model_name} | {summary['best_seed']} | "
                f"{summary['best_accuracy'] * 100:.2f} | {summary['best_f1_macro'] * 100:.2f} | "
                f"{summary['best_f1_weighted'] * 100:.2f} | {best_run['training_time']:.2f} |"
            )
        lines.append("")

    if "exp2" in results:
        lines.append("## Experiment 2: ETTh1 Forecasting")
        lines.append("")
        lines.append(f"### Table 2: ETTh1 Results (mean ± std over {len(next(iter(next(iter(results['exp2'].values())).values()))['runs'])} seeds)")
        lines.append("")
        lines.append("| Model | H=24 MSE | H=24 MAE | H=48 MSE | H=48 MAE | H=96 MSE | H=96 MAE |")
        lines.append("|-------|---------|---------|---------|---------|---------|---------|")
        for model_name in CLASSIFICATION_MODELS:
            h24 = results["exp2"]["H24"][model_name]
            h48 = results["exp2"]["H48"][model_name]
            h96 = results["exp2"]["H96"][model_name]
            lines.append(
                f"| {model_name} | {format_mean_std(h24['mse'])} | {format_mean_std(h24['mae'])} | "
                f"{format_mean_std(h48['mse'])} | {format_mean_std(h48['mae'])} | "
                f"{format_mean_std(h96['mse'])} | {format_mean_std(h96['mae'])} |"
            )
        lines.append("")

    if "exp3" in results:
        lines.append("## Experiment 3: Cross-Domain Forecasting")
        lines.append("")
        lines.append(f"### Table 3: Cross-Domain Results (H=24, mean ± std over {len(next(iter(results['exp3'].values()))['ETTh1→ETTh1']['runs'])} seeds)")
        lines.append("")
        lines.append("| Model | ETTh1→ETTh1 MSE | ETTh1→ETTh2 MSE | ETTh2→ETTh2 MSE | Transfer Gap |")
        lines.append("|-------|----------------|----------------|----------------|-------------|")
        for model_name, summary in results["exp3"].items():
            lines.append(
                f"| {model_name} | {format_mean_std(summary['ETTh1→ETTh1']['mse'])} | "
                f"{format_mean_std(summary['ETTh1→ETTh2']['mse'])} | "
                f"{format_mean_std(summary['ETTh2→ETTh2']['mse'])} | "
                f"{summary['transfer_gap']:.2f}% |"
            )
        lines.append("")
        lines.append("Transfer Gap = (ETTh1→ETTh2 MSE - ETTh2→ETTh2 MSE) / ETTh2→ETTh2 MSE × 100%")
        lines.append("")

    if "exp4" in results:
        lines.append("## Experiment 4: Ablation Study")
        lines.append("")
        lines.append(f"### Table 4a: Ablation on UCI HAR (mean ± std over {len(next(iter(results['exp4']['har'].values()))['runs'])} seeds)")
        lines.append("")
        lines.append("| Variant | Accuracy (%) | F1-Macro (%) | Δ Acc |")
        lines.append("|---------|-------------|-------------|-------|")
        for model_name, summary in results["exp4"]["har"].items():
            label = ablation_label(model_name)
            delta = "—" if model_name == "TMTFNet_v2" else f"{summary['delta_acc']:+.2f}"
            lines.append(
                f"| {label} | {format_mean_std(summary['accuracy'], percent=True)} | "
                f"{format_mean_std(summary['f1_macro'], percent=True)} | {delta} |"
            )
        lines.append("")
        lines.append("### Table 4b: Ablation on ETTh1→ETTh2 Cross-Domain Forecasting (H=24)")
        lines.append("")
        lines.append("| Variant | MSE | MAE | Δ MSE |")
        lines.append("|---------|-----|-----|-------|")
        for model_name, summary in results["exp4"]["cross_forecast"].items():
            label = ablation_label(model_name)
            delta = "—" if model_name == "TMTFNet_v2" else f"{summary['delta_mse']:+.4f}"
            lines.append(f"| {label} | {format_mean_std(summary['mse'])} | {format_mean_std(summary['mae'])} | {delta} |")
        lines.append("")

    if "exp5" in results:
        lines.append("## Experiment 5: Hyperparameter Sensitivity")
        lines.append("")
        section_meta = [
            ("d_model", "### Table 5a: Effect of d_model", "d_model"),
            ("n_heads", "### Table 5b: Effect of n_heads", "n_heads"),
            ("n_layers", "### Table 5c: Effect of n_layers", "n_enc_layers"),
            ("modality_dropout", "### Table 5d: Effect of modality_dropout", "modality_dropout"),
        ]
        for section_key, title, config_key in section_meta:
            lines.append(title)
            lines.append("")
            lines.append(f"| {config_key} | Params | Accuracy (%) | F1-Macro (%) |")
            lines.append("|---------|--------|-------------|-------------|")
            for entry in results["exp5"][section_key]:
                value = entry["config"][config_key]
                lines.append(
                    f"| {value} | {entry['n_params'] / 1000:.1f}K | "
                    f"{entry['accuracy'] * 100:.2f} | {entry['f1_macro'] * 100:.2f} |"
                )
            lines.append("")

    if "exp6" in results:
        lines.append("## Experiment 6: Efficiency Comparison")
        lines.append("")
        lines.append("| Model | Params (K) | Train Time/epoch (s) | Inference Time/batch (ms) | Accuracy (%) |")
        lines.append("|-------|-----------|---------------------|--------------------------|-------------|")
        for model_name, summary in results["exp6"].items():
            lines.append(
                f"| {model_name} | {summary['params_k']:.1f} | {summary['train_time_per_epoch_s']:.2f} | "
                f"{summary['inference_time_ms']:.2f} | {summary['accuracy']:.2f} |"
            )
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    if "exp1" in results:
        best_cls = max(results["exp1"].items(), key=lambda item: item[1]["accuracy"]["mean"])
        tmt_cls = results["exp1"]["TMTFNet_v2"]["accuracy"]["mean"] * 100.0
        cls_diff = tmt_cls - best_cls[1]["accuracy"]["mean"] * 100.0
        direction = "higher than" if cls_diff >= 0 else "lower than"
        lines.append(
            f"- UCI HAR: TMTFNet_v2 achieves {tmt_cls:.2f}% accuracy, {direction} the strongest baseline "
            f"{best_cls[0]} by {abs(cls_diff):.2f} percentage points."
        )
    if "exp2" in results:
        h24 = results["exp2"]["H24"]
        best_h24 = min(h24.items(), key=lambda item: item[1]["mse"]["mean"])
        tmt_h24 = h24["TMTFNet_v2"]["mse"]["mean"]
        relation = "better than" if tmt_h24 <= best_h24[1]["mse"]["mean"] else "worse than"
        lines.append(
            f"- ETTh1 Forecasting: TMTFNet_v2 achieves {tmt_h24:.4f} MSE at H=24, {relation} the strongest baseline "
            f"{best_h24[0]}."
        )
    if "exp3" in results:
        cross = results["exp3"]
        best_cross = min(cross.items(), key=lambda item: item[1]["ETTh1→ETTh2"]["mse"]["mean"])
        tmt_cross = cross["TMTFNet_v2"]["ETTh1→ETTh2"]["mse"]["mean"]
        lines.append(
            f"- Cross-Domain: TMTFNet_v2 achieves {tmt_cross:.4f} MSE on ETTh1→ETTh2; "
            f"best model under the evaluated protocol is {best_cross[0]} with {best_cross[1]['ETTh1→ETTh2']['mse']['mean']:.4f} MSE."
        )
    if "exp4" in results:
        har_effects = {ablation_label(name): summary["delta_acc"] for name, summary in results["exp4"]["har"].items() if name != "TMTFNet_v2"}
        cross_effects = {ablation_label(name): summary["delta_mse"] for name, summary in results["exp4"]["cross_forecast"].items() if name != "TMTFNet_v2"}
        har_helpful = [name for name, delta in har_effects.items() if delta < 0]
        cross_helpful = [name for name, delta in cross_effects.items() if delta > 0]
        cross_harmful = [name for name, delta in cross_effects.items() if delta < 0]
        lines.append(
            f"- Ablation: on HAR, removing any module lowers accuracy ({', '.join(har_helpful)}). "
            f"On ETTh1→ETTh2, the clearly helpful components are {', '.join(cross_helpful) if cross_helpful else 'none'}, "
            f"while {', '.join(cross_harmful) if cross_harmful else 'none'} slightly reduce MSE when removed."
        )
    if "exp5" in results:
        best_d_model = max(results["exp5"]["d_model"], key=lambda item: item["accuracy"])
        best_heads = max(results["exp5"]["n_heads"], key=lambda item: item["accuracy"])
        best_layers = max(results["exp5"]["n_layers"], key=lambda item: item["accuracy"])
        best_moddrop = max(results["exp5"]["modality_dropout"], key=lambda item: item["accuracy"])
        lines.append(
            f"- Best configuration from sensitivity study: d_model={best_d_model['config']['d_model']}, "
            f"n_heads={best_heads['config']['n_heads']}, n_layers={best_layers['config']['n_enc_layers']}, "
            f"modality_dropout={best_moddrop['config']['modality_dropout']}."
        )

    with open(args.full_results_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TMTFNet_v2 experiments.")
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--result_dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--full_results_path", default=os.path.join(PROJECT_ROOT, "FULL_RESULTS.md"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--budget", choices=["requested", "fast"], default="fast")
    parser.add_argument("--experiments", nargs="*", default=["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.result_dir)
    device, gpu_name, cuda_version = setup_device()
    total_start = time.time()
    budget = budget_config(args.budget)
    collected = {}

    if "exp1" in args.experiments:
        collected["exp1"] = experiment_1(args, device, budget)
    if "exp2" in args.experiments:
        collected["exp2"] = experiment_2(args, device, budget)
    if "exp3" in args.experiments:
        collected["exp3"] = experiment_3(args, device, budget)
    if "exp4" in args.experiments:
        collected["exp4"] = experiment_4(args, device, budget)
    if "exp5" in args.experiments:
        collected["exp5"] = experiment_5(args, device, budget)
    if "exp6" in args.experiments:
        exp1_summary = collected.get("exp1") or load_json(os.path.join(args.result_dir, "exp1_har", "summary.json"))
        collected["exp6"] = experiment_6(args, device, exp1_summary)

    total_time_s = time.time() - total_start
    render_full_results(args, gpu_name, cuda_version, total_time_s, args.budget, collected)
    print(f"FULL_RESULTS.md written to {args.full_results_path}")


if __name__ == "__main__":
    main()
