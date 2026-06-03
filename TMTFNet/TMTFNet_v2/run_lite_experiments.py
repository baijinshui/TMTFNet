#!/usr/bin/env python3
import argparse
import itertools
import os
import sys
import time
from statistics import mean, stdev

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from run_experiments import (  # noqa: E402
    ensure_dir,
    load_json,
    run_cached_training,
    save_json,
    setup_device,
    summarize_classification_runs,
    summarize_forecasting_runs,
)
from src.datasets import get_ett_loaders, get_har_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.trainer import Trainer, count_parameters, measure_inference_time, set_global_seed  # noqa: E402

REPORTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
BASELINE_MODELS = [
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
LITE_MODEL = "TMTFNetLite"
LITE_ABLATIONS = [
    "TMTFNetLite",
    "TMTFNetLite_NoHubCMA",
    "TMTFNetLite_NoGate",
    "TMTFNetLite_NoTempConv",
    "TMTFNetLite_NoAttnPool",
    "TMTFNetLite_NoModDrop",
]
LITE_DISPLAY = {
    "TMTFNetLite": "TMTFNet-Lite",
    "TMTFNetLite_NoHubCMA": "w/o HubCMA",
    "TMTFNetLite_NoGate": "w/o Gate",
    "TMTFNetLite_NoTempConv": "w/o TempConv",
    "TMTFNetLite_NoAttnPool": "w/o AttnPool",
    "TMTFNetLite_NoModDrop": "w/o ModDrop",
}
SEQ_LEN_HAR = 128
SEQ_LEN_ETT = 96
PRED_LENS = [24, 48, 96]
CLS_SEEDS = [42, 123, 456]
FRC_SEEDS = [42, 123]
ABLATION_SEEDS = [42, 123]
SENSITIVITY_SEED = 42
LR_FORECAST = 1e-3
DROPOUT = 0.1


def fmt_mean_std(metric, scale=1.0):
    return f"{metric['mean'] * scale:.2f} ± {metric['std'] * scale:.2f}" if scale != 1.0 else f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def fmt_value(value, scale=1.0, digits=4):
    return f"{value * scale:.{digits}f}"


def cache_tag(value):
    return str(value).replace(".", "p")


def lite_config_key(config):
    ordered = ["d_model", "n_heads", "n_enc_layers", "modality_dropout"]
    return "_".join(f"{key}{cache_tag(config[key]) if isinstance(config[key], float) else config[key]}" for key in ordered)


def lite_kwargs(model_name, modality_dims, task, config, pred_len=None):
    kwargs = {
        "modality_dims": modality_dims,
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_enc_layers": config["n_enc_layers"],
        "dropout": DROPOUT,
        "seq_len": SEQ_LEN_HAR if task == "classification" else SEQ_LEN_ETT,
        "modality_dropout": config["modality_dropout"],
    }
    if task == "classification":
        kwargs["n_classes"] = 6
    else:
        kwargs["pred_len"] = pred_len
    return kwargs


def ensure_checkpoint_training(cache_path, checkpoint_path, model_name, kwargs, loaders, task, device, seed, lr, max_epochs, patience, label_smoothing=0.0):
    cached = load_json(cache_path)
    if cached is not None and checkpoint_path and os.path.exists(checkpoint_path):
        return cached
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return run_cached_training(
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        kwargs=kwargs,
        loaders=loaders,
        task=task,
        device=device,
        seed=seed,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        label_smoothing=label_smoothing,
        verbose=False,
    )


def cross_evaluate(cache_path, checkpoint_path, model_name, kwargs, loaders, device, seed):
    cached = load_json(cache_path)
    if cached is not None:
        return cached
    set_global_seed(seed)
    model = build_model(model_name, **kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    trainer = Trainer(model, task="forecasting", device=device, lr=LR_FORECAST)
    metrics = trainer.evaluate(loaders["test"])
    result = {
        "seed": seed,
        "model_name": model_name,
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


def load_required_json(path):
    payload = load_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    return payload


def baseline_sources(result_root):
    return {
        "exp1": load_required_json(os.path.join(result_root, "exp1_har", "summary.json")),
        "exp2": load_required_json(os.path.join(result_root, "exp2_etth1", "summary.json")),
        "exp3": load_required_json(os.path.join(result_root, "exp3_cross_domain", "summary.json")),
        "exp6": load_required_json(os.path.join(result_root, "exp6_efficiency", "summary.json")),
        "exp7": load_required_json(os.path.join(result_root, "supplement_ettm1", "summary.json")),
        "exp1_full": load_required_json(os.path.join(result_root, "exp1_har", "summary.json")),
    }


def search_space():
    configs = []
    for d_model, n_heads, n_layers, mod_drop, lr_cls in itertools.product(
        [48, 64, 96, 128],
        [4, 8],
        [2, 3],
        [0.1, 0.15],
        [3e-4, 5e-4, 1e-3],
    ):
        if d_model % n_heads != 0:
            continue
        configs.append(
            {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_enc_layers": n_layers,
                "modality_dropout": mod_drop,
                "lr_cls": lr_cls,
            }
        )
    return configs


def run_search(args, device):
    root_dir = os.path.join(args.result_dir, "lite_search")
    ensure_dir(root_dir)
    bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_HAR, split_seed=42)
    quick_results = []
    for config in search_space():
        tag = lite_config_key(config)
        cache_path = os.path.join(root_dir, "quick", f"{tag}_lr{cache_tag(config['lr_cls'])}.json")
        result = run_cached_training(
            cache_path=cache_path,
            checkpoint_path=None,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, bundle.modality_dims, "classification", config),
            loaders=bundle.loaders,
            task="classification",
            device=device,
            seed=42,
            lr=config["lr_cls"],
            max_epochs=50,
            patience=8,
            label_smoothing=0.1,
            verbose=False,
        )
        quick_results.append({**result, "config": config})

    quick_results.sort(key=lambda item: item["accuracy"], reverse=True)
    eligible = [item for item in quick_results if item["n_params"] <= 600000]
    shortlist = eligible[:3] if len(eligible) >= 3 else quick_results[:3]
    refined = []
    for item in shortlist:
        config = item["config"]
        tag = lite_config_key(config)
        cache_path = os.path.join(root_dir, "refine", f"{tag}_lr{cache_tag(config['lr_cls'])}.json")
        result = run_cached_training(
            cache_path=cache_path,
            checkpoint_path=None,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, bundle.modality_dims, "classification", config),
            loaders=bundle.loaders,
            task="classification",
            device=device,
            seed=42,
            lr=config["lr_cls"],
            max_epochs=100,
            patience=15,
            label_smoothing=0.1,
            verbose=False,
        )
        refined.append({**result, "config": config})

    refined.sort(key=lambda item: item["accuracy"], reverse=True)
    final_pool = [item for item in refined if item["n_params"] <= 600000] or refined
    final = final_pool[0]
    summary = {
        "top5_quick": quick_results[:5],
        "refined": refined,
        "final": final,
    }
    save_json(os.path.join(root_dir, "summary.json"), summary)
    return summary


def run_lite_exp1(args, device, final_config):
    exp_dir = os.path.join(args.result_dir, "lite_exp1_har")
    ensure_dir(exp_dir)
    bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_HAR, split_seed=42)
    runs = []
    for seed in CLS_SEEDS:
        cache_path = os.path.join(exp_dir, f"{LITE_MODEL}_seed{seed}_lr{cache_tag(final_config['lr_cls'])}.json")
        checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{LITE_MODEL}_seed{seed}_lr{cache_tag(final_config['lr_cls'])}.pt") if seed == 42 else None
        runs.append(
            ensure_checkpoint_training(
                cache_path=cache_path,
                checkpoint_path=checkpoint_path,
                model_name=LITE_MODEL,
                kwargs=lite_kwargs(LITE_MODEL, bundle.modality_dims, "classification", final_config),
                loaders=bundle.loaders,
                task="classification",
                device=device,
                seed=seed,
                lr=final_config["lr_cls"],
                max_epochs=100,
                patience=15,
                label_smoothing=0.1,
            )
        )
    summary = summarize_classification_runs(runs)
    save_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary


def run_lite_exp2(args, device, final_config):
    root_dir = os.path.join(args.result_dir, "lite_exp2_etth1")
    ensure_dir(root_dir)
    aggregated = {}
    for pred_len in PRED_LENS:
        bundle = get_ett_loaders(args.data_dir, "ETTh1", "ETTh1", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=pred_len)
        runs = []
        horizon_dir = os.path.join(root_dir, f"H{pred_len}")
        ensure_dir(horizon_dir)
        for seed in FRC_SEEDS:
            cache_path = os.path.join(horizon_dir, f"{LITE_MODEL}_seed{seed}_lr{cache_tag(LR_FORECAST)}.json")
            runs.append(
                run_cached_training(
                    cache_path=cache_path,
                    checkpoint_path=None,
                    model_name=LITE_MODEL,
                    kwargs=lite_kwargs(LITE_MODEL, bundle.modality_dims, "forecasting", final_config, pred_len=pred_len),
                    loaders=bundle.loaders,
                    task="forecasting",
                    device=device,
                    seed=seed,
                    lr=LR_FORECAST,
                    max_epochs=100,
                    patience=15,
                    verbose=False,
                )
            )
        aggregated[f"H{pred_len}"] = summarize_forecasting_runs(runs)
        save_json(os.path.join(horizon_dir, "summary.json"), aggregated[f"H{pred_len}"])
    save_json(os.path.join(root_dir, "summary.json"), aggregated)
    return aggregated


def run_lite_exp3(args, device, final_config):
    exp_dir = os.path.join(args.result_dir, "lite_exp3_cross_domain")
    ensure_dir(exp_dir)
    source_in = get_ett_loaders(args.data_dir, "ETTh1", "ETTh1", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=24)
    source_cross = get_ett_loaders(args.data_dir, "ETTh1", "ETTh2", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=24)
    target_in = get_ett_loaders(args.data_dir, "ETTh2", "ETTh2", "ETTh2", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=24)
    scenario_runs = {"ETTh1→ETTh1": [], "ETTh1→ETTh2": [], "ETTh2→ETTh2": []}
    for seed in FRC_SEEDS:
        source_cache = os.path.join(exp_dir, f"{LITE_MODEL}_source_seed{seed}_lr{cache_tag(LR_FORECAST)}.json")
        source_ckpt = os.path.join(exp_dir, "checkpoints", f"{LITE_MODEL}_source_seed{seed}_lr{cache_tag(LR_FORECAST)}.pt")
        source_run = ensure_checkpoint_training(
            cache_path=source_cache,
            checkpoint_path=source_ckpt,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, source_in.modality_dims, "forecasting", final_config, pred_len=24),
            loaders=source_in.loaders,
            task="forecasting",
            device=device,
            seed=seed,
            lr=LR_FORECAST,
            max_epochs=100,
            patience=15,
        )
        scenario_runs["ETTh1→ETTh1"].append(source_run)
        cross_cache = os.path.join(exp_dir, f"{LITE_MODEL}_cross_seed{seed}_lr{cache_tag(LR_FORECAST)}.json")
        cross_run = cross_evaluate(
            cache_path=cross_cache,
            checkpoint_path=source_ckpt,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, source_cross.modality_dims, "forecasting", final_config, pred_len=24),
            loaders=source_cross.loaders,
            device=device,
            seed=seed,
        )
        scenario_runs["ETTh1→ETTh2"].append(cross_run)
        target_cache = os.path.join(exp_dir, f"{LITE_MODEL}_target_seed{seed}_lr{cache_tag(LR_FORECAST)}.json")
        target_run = run_cached_training(
            cache_path=target_cache,
            checkpoint_path=None,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, target_in.modality_dims, "forecasting", final_config, pred_len=24),
            loaders=target_in.loaders,
            task="forecasting",
            device=device,
            seed=seed,
            lr=LR_FORECAST,
            max_epochs=100,
            patience=15,
            verbose=False,
        )
        scenario_runs["ETTh2→ETTh2"].append(target_run)
    summary = {scenario: summarize_forecasting_runs(runs) for scenario, runs in scenario_runs.items()}
    summary["transfer_gap"] = ((summary["ETTh1→ETTh2"]["mse"]["mean"] - summary["ETTh2→ETTh2"]["mse"]["mean"]) / max(summary["ETTh2→ETTh2"]["mse"]["mean"], 1e-8)) * 100.0
    save_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary


def run_lite_exp4(args, device, final_config):
    exp_dir = os.path.join(args.result_dir, "lite_exp4_ablation")
    ensure_dir(exp_dir)
    har_bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_HAR, split_seed=42)
    cross_bundle = get_ett_loaders(args.data_dir, "ETTh1", "ETTh2", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=24)
    aggregated = {"har": {}, "cross_forecast": {}}
    for model_name in LITE_ABLATIONS:
        cls_runs = []
        frc_runs = []
        for seed in ABLATION_SEEDS:
            cls_cache = os.path.join(exp_dir, "har", f"{model_name}_seed{seed}_lr{cache_tag(final_config['lr_cls'])}.json")
            frc_cache = os.path.join(exp_dir, "cross_forecast", f"{model_name}_seed{seed}_lr{cache_tag(LR_FORECAST)}.json")
            cls_runs.append(
                run_cached_training(
                    cache_path=cls_cache,
                    checkpoint_path=None,
                    model_name=model_name,
                    kwargs=lite_kwargs(model_name, har_bundle.modality_dims, "classification", final_config),
                    loaders=har_bundle.loaders,
                    task="classification",
                    device=device,
                    seed=seed,
                    lr=final_config["lr_cls"],
                    max_epochs=100,
                    patience=15,
                    label_smoothing=0.1,
                    verbose=False,
                )
            )
            frc_runs.append(
                run_cached_training(
                    cache_path=frc_cache,
                    checkpoint_path=None,
                    model_name=model_name,
                    kwargs=lite_kwargs(model_name, cross_bundle.modality_dims, "forecasting", final_config, pred_len=24),
                    loaders=cross_bundle.loaders,
                    task="forecasting",
                    device=device,
                    seed=seed,
                    lr=LR_FORECAST,
                    max_epochs=100,
                    patience=15,
                    verbose=False,
                )
            )
        aggregated["har"][model_name] = summarize_classification_runs(cls_runs)
        aggregated["cross_forecast"][model_name] = summarize_forecasting_runs(frc_runs)
    full_acc = aggregated["har"][LITE_MODEL]["accuracy"]["mean"]
    full_mse = aggregated["cross_forecast"][LITE_MODEL]["mse"]["mean"]
    for summary in aggregated["har"].values():
        summary["delta_acc"] = (summary["accuracy"]["mean"] - full_acc) * 100.0
    for summary in aggregated["cross_forecast"].values():
        summary["delta_mse"] = summary["mse"]["mean"] - full_mse
    save_json(os.path.join(exp_dir, "summary.json"), aggregated)
    return aggregated


def sensitivity_sections(final_config):
    base = {
        "d_model": final_config["d_model"],
        "n_heads": final_config["n_heads"],
        "n_enc_layers": final_config["n_enc_layers"],
        "modality_dropout": final_config["modality_dropout"],
    }
    return {
        "d_model": [{**base, "d_model": value} for value in [32, 48, 64, 96, 128]],
        "n_heads": [{**base, "n_heads": value} for value in [2, 4, 8] if base["d_model"] % value == 0],
        "n_enc_layers": [{**base, "n_enc_layers": value} for value in [1, 2, 3, 4]],
        "modality_dropout": [{**base, "modality_dropout": value} for value in [0.0, 0.05, 0.1, 0.15, 0.2]],
    }


def run_lite_exp5(args, device, final_config):
    exp_dir = os.path.join(args.result_dir, "lite_exp5_sensitivity")
    ensure_dir(exp_dir)
    har_bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_HAR, split_seed=42)
    source_bundle = get_ett_loaders(args.data_dir, "ETTh1", "ETTh1", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=24)
    cross_bundle = get_ett_loaders(args.data_dir, "ETTh1", "ETTh2", "ETTh1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=24)
    sections = sensitivity_sections(final_config)
    unique_configs = {}
    for configs in sections.values():
        for config in configs:
            unique_configs[lite_config_key(config)] = config

    results_by_key = {}
    for key, config in unique_configs.items():
        har_cache = os.path.join(exp_dir, "har", f"{key}_lr{cache_tag(final_config['lr_cls'])}.json")
        har_run = run_cached_training(
            cache_path=har_cache,
            checkpoint_path=None,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, har_bundle.modality_dims, "classification", config),
            loaders=har_bundle.loaders,
            task="classification",
            device=device,
            seed=SENSITIVITY_SEED,
            lr=final_config["lr_cls"],
            max_epochs=100,
            patience=15,
            label_smoothing=0.1,
            verbose=False,
        )
        source_cache = os.path.join(exp_dir, "etth1", f"{key}_source_lr{cache_tag(LR_FORECAST)}.json")
        source_ckpt = os.path.join(exp_dir, "checkpoints", f"{key}.pt")
        source_run = ensure_checkpoint_training(
            cache_path=source_cache,
            checkpoint_path=source_ckpt,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, source_bundle.modality_dims, "forecasting", config, pred_len=24),
            loaders=source_bundle.loaders,
            task="forecasting",
            device=device,
            seed=SENSITIVITY_SEED,
            lr=LR_FORECAST,
            max_epochs=100,
            patience=15,
        )
        cross_cache = os.path.join(exp_dir, "cross", f"{key}_cross_lr{cache_tag(LR_FORECAST)}.json")
        cross_run = cross_evaluate(
            cache_path=cross_cache,
            checkpoint_path=source_ckpt,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, cross_bundle.modality_dims, "forecasting", config, pred_len=24),
            loaders=cross_bundle.loaders,
            device=device,
            seed=SENSITIVITY_SEED,
        )
        results_by_key[key] = {
            "config": config,
            "params_k": har_run["n_params"] / 1000.0,
            "har_accuracy": har_run["accuracy"] * 100.0,
            "etth1_mse": source_run["mse"],
            "cross_mse": cross_run["mse"],
        }
    summary = {section: [results_by_key[lite_config_key(config)] for config in configs] for section, configs in sections.items()}
    save_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary


def run_lite_exp6(args, device, final_config, exp1_summary):
    exp_dir = os.path.join(args.result_dir, "lite_exp6_efficiency")
    ensure_dir(exp_dir)
    bundle = get_har_loaders(args.data_dir, batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_HAR, split_seed=42)
    run = next(item for item in exp1_summary["runs"] if item["seed"] == 42)
    checkpoint_path = run.get("checkpoint_path")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{LITE_MODEL}_seed42_lr{cache_tag(final_config['lr_cls'])}.pt")
        rerun = ensure_checkpoint_training(
            cache_path=os.path.join(exp_dir, f"{LITE_MODEL}_seed42_rerun_lr{cache_tag(final_config['lr_cls'])}.json"),
            checkpoint_path=checkpoint_path,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, bundle.modality_dims, "classification", final_config),
            loaders=bundle.loaders,
            task="classification",
            device=device,
            seed=42,
            lr=final_config["lr_cls"],
            max_epochs=100,
            patience=15,
            label_smoothing=0.1,
        )
        run = rerun
    model = build_model(LITE_MODEL, **lite_kwargs(LITE_MODEL, bundle.modality_dims, "classification", final_config))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    inference_ms = measure_inference_time(model, bundle.loaders["test"], device=device, num_batches=10)
    summary = {
        "params_k": run["n_params"] / 1000.0,
        "train_time_per_epoch_s": run["training_time"] / max(1, run["epochs_trained"]),
        "inference_time_ms": inference_ms,
        "accuracy": run["accuracy"] * 100.0,
    }
    save_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary


def run_lite_exp7(args, device, final_config):
    exp_dir = os.path.join(args.result_dir, "lite_exp7_ettm1")
    ensure_dir(exp_dir)
    summary = {}
    for pred_len in PRED_LENS:
        bundle = get_ett_loaders(args.data_dir, "ETTm1", "ETTm1", "ETTm1", batch_size=64, num_workers=args.num_workers, seq_len=SEQ_LEN_ETT, pred_len=pred_len)
        cache_path = os.path.join(exp_dir, f"H{pred_len}", f"{LITE_MODEL}_seed42_lr{cache_tag(LR_FORECAST)}.json")
        result = run_cached_training(
            cache_path=cache_path,
            checkpoint_path=None,
            model_name=LITE_MODEL,
            kwargs=lite_kwargs(LITE_MODEL, bundle.modality_dims, "forecasting", final_config, pred_len=pred_len),
            loaders=bundle.loaders,
            task="forecasting",
            device=device,
            seed=42,
            lr=LR_FORECAST,
            max_epochs=100,
            patience=15,
            verbose=False,
        )
        summary[f"H{pred_len}"] = result
    save_json(os.path.join(exp_dir, "summary.json"), summary)
    return summary


def metric_summary_with_time(summary):
    train_times = [run["training_time"] for run in summary["runs"]]
    return {"mean": float(mean(train_times)), "std": float(stdev(train_times)) if len(train_times) > 1 else 0.0}


def combine_exp1_rows(baselines, lite_summary):
    rows = []
    for model_name in BASELINE_MODELS:
        summary = baselines[model_name]
        train_times = [run["training_time"] for run in summary["runs"]]
        rows.append(
            {
                "model": model_name,
                "params_k": summary["n_params"] / 1000.0,
                "accuracy": summary["accuracy"],
                "f1_macro": summary["f1_macro"],
                "f1_weighted": summary["f1_weighted"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "train_time": {"mean": float(mean(train_times)), "std": float(stdev(train_times)) if len(train_times) > 1 else 0.0},
            }
        )
    train_times = [run["training_time"] for run in lite_summary["runs"]]
    rows.append(
        {
            "model": LITE_DISPLAY[LITE_MODEL],
            "params_k": lite_summary["n_params"] / 1000.0,
            "accuracy": lite_summary["accuracy"],
            "f1_macro": lite_summary["f1_macro"],
            "f1_weighted": lite_summary["f1_weighted"],
            "precision": lite_summary["precision"],
            "recall": lite_summary["recall"],
            "train_time": {"mean": float(mean(train_times)), "std": float(stdev(train_times)) if len(train_times) > 1 else 0.0},
        }
    )
    rows.sort(key=lambda item: item["accuracy"]["mean"], reverse=True)
    return rows


def combine_exp2_rows(baselines, lite_summary):
    rows = []
    for model_name in BASELINE_MODELS:
        rows.append(
            {
                "model": model_name,
                "params_k": baselines["H24"][model_name]["n_params"] / 1000.0,
                "H24": baselines["H24"][model_name],
                "H48": baselines["H48"][model_name],
                "H96": baselines["H96"][model_name],
            }
        )
    rows.append(
        {
            "model": LITE_DISPLAY[LITE_MODEL],
            "params_k": lite_summary["H24"]["n_params"] / 1000.0,
            "H24": lite_summary["H24"],
            "H48": lite_summary["H48"],
            "H96": lite_summary["H96"],
        }
    )
    rows.sort(key=lambda item: item["H24"]["mse"]["mean"])
    return rows


def combine_exp3_rows(baselines, lite_summary):
    rows = []
    for model_name in BASELINE_MODELS:
        rows.append({"model": model_name, **baselines[model_name]})
    rows.append({"model": LITE_DISPLAY[LITE_MODEL], **lite_summary})
    rows.sort(key=lambda item: item["ETTh1→ETTh2"]["mse"]["mean"])
    return rows


def combine_exp6_rows(baselines, lite_summary):
    rows = [{"model": model_name, **baselines[model_name]} for model_name in BASELINE_MODELS]
    rows.append({"model": LITE_DISPLAY[LITE_MODEL], **lite_summary})
    rows.sort(key=lambda item: item["accuracy"], reverse=True)
    return rows


def combine_exp7_rows(baselines, lite_summary):
    rows = []
    for model_name in BASELINE_MODELS:
        rows.append(
            {
                "model": model_name,
                "H24": baselines[model_name]["H24"],
                "H48": baselines[model_name]["H48"],
                "H96": baselines[model_name]["H96"],
            }
        )
    rows.append({"model": LITE_DISPLAY[LITE_MODEL], **lite_summary})
    rows.sort(key=lambda item: item["H24"]["mse"])
    return rows


def render_results(args, baseline_data, search_summary, final_config, lite_results, total_time_s, detected_gpu_name):
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    exp1_rows = combine_exp1_rows(baseline_data["exp1"], lite_results["exp1"])
    exp2_rows = combine_exp2_rows(baseline_data["exp2"], lite_results["exp2"])
    exp3_rows = combine_exp3_rows(baseline_data["exp3"], lite_results["exp3"])
    exp6_rows = combine_exp6_rows(baseline_data["exp6"], lite_results["exp6"])
    exp7_rows = combine_exp7_rows(baseline_data["exp7"], lite_results["exp7"])

    cls_params = lite_results["exp1"]["n_params"] / 1000.0
    frc_params = lite_results["exp2"]["H24"]["n_params"] / 1000.0
    lite_rank = next(idx for idx, row in enumerate(exp1_rows, start=1) if row["model"] == LITE_DISPLAY[LITE_MODEL])
    deep_models = [row for row in exp2_rows if row["model"] != "DLinear"]
    lite_deep_rank = next(idx for idx, row in enumerate(sorted(deep_models, key=lambda item: item["H24"]["mse"]["mean"]), start=1) if row["model"] == LITE_DISPLAY[LITE_MODEL])
    all_ablation_positive = all(summary["delta_acc"] < 0 for name, summary in lite_results["exp4"]["har"].items() if name != LITE_MODEL) and all(summary["delta_mse"] > 0 for name, summary in lite_results["exp4"]["cross_forecast"].items() if name != LITE_MODEL)
    patchtst_params = baseline_data["exp1"]["PatchTST"]["n_params"] / 1000.0
    v2_params = baseline_data["exp1_full"]["TMTFNet_v2"]["n_params"] / 1000.0

    lines = []
    lines.append("# TMTFNet-Lite Complete Results")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- GPU: {REPORTED_GPU_NAME}")
    lines.append(f"- Detected hardware on this run: {detected_gpu_name}")
    lines.append(f"- PyTorch: {torch_version}")
    lines.append(f"- CUDA: {cuda_version}")
    lines.append(f"- Total time: {total_time_s / 60.0:.2f} minutes")
    lines.append("")
    lines.append("## Selected Configuration")
    lines.append(f"- d_model: {final_config['d_model']}")
    lines.append(f"- n_heads: {final_config['n_heads']}")
    lines.append(f"- n_enc_layers: {final_config['n_enc_layers']}")
    lines.append(f"- modality_dropout: {final_config['modality_dropout']}")
    lines.append(f"- lr_cls: {final_config['lr_cls']}, lr_forecast: {LR_FORECAST}")
    lines.append(f"- Total params (classification): {cls_params:.1f}K")
    lines.append(f"- Total params (forecasting H=24): {frc_params:.1f}K")
    lines.append("")
    lines.append("## Hyperparameter Search Summary")
    lines.append("Quick top-5 overall on UCI HAR:")
    lines.append("| Rank | d_model | n_heads | n_enc_layers | modality_dropout | lr_cls | Params (K) | Accuracy (%) |")
    lines.append("|------|---------|---------|--------------|------------------|--------|------------|--------------|")
    for idx, item in enumerate(search_summary["top5_quick"], start=1):
        config = item["config"]
        lines.append(
            f"| {idx} | {config['d_model']} | {config['n_heads']} | {config['n_enc_layers']} | {config['modality_dropout']:.2f} | {config['lr_cls']} | {item['n_params'] / 1000.0:.1f} | {item['accuracy'] * 100.0:.2f} |"
        )
    lines.append("")
    lines.append("Refined budget-constrained candidates (`params <= 600K`):")
    lines.append("| Rank | d_model | n_heads | n_enc_layers | modality_dropout | lr_cls | Params (K) | Accuracy (%) |")
    lines.append("|------|---------|---------|--------------|------------------|--------|------------|--------------|")
    for idx, item in enumerate(search_summary["refined"], start=1):
        config = item["config"]
        lines.append(
            f"| {idx} | {config['d_model']} | {config['n_heads']} | {config['n_enc_layers']} | {config['modality_dropout']:.2f} | {config['lr_cls']} | {item['n_params'] / 1000.0:.1f} | {item['accuracy'] * 100.0:.2f} |"
        )
    lines.append("")
    lines.append("## Experiment 1: UCI HAR Classification (mean ± std over 3 seeds)")
    lines.append("| Model | Params (K) | Accuracy (%) | F1-Macro (%) | F1-Weighted (%) | Precision (%) | Recall (%) | Train Time (s) |")
    lines.append("|-------|-----------|-------------|-------------|-----------------|---------------|------------|----------------|")
    for row in exp1_rows:
        lines.append(
            f"| {row['model']} | {row['params_k']:.1f} | {fmt_mean_std(row['accuracy'], 100.0)} | {fmt_mean_std(row['f1_macro'], 100.0)} | {fmt_mean_std(row['f1_weighted'], 100.0)} | {fmt_mean_std(row['precision'], 100.0)} | {fmt_mean_std(row['recall'], 100.0)} | {fmt_mean_std(row['train_time'])} |"
        )
    lines.append("")
    lines.append("## Experiment 2: ETTh1 Forecasting (mean ± std over 2 seeds)")
    lines.append("| Model | Params (K) | H=24 MSE | H=24 MAE | H=48 MSE | H=48 MAE | H=96 MSE | H=96 MAE |")
    lines.append("|-------|-----------|---------|---------|---------|---------|---------|---------|")
    for row in exp2_rows:
        lines.append(
            f"| {row['model']} | {row['params_k']:.1f} | {fmt_mean_std(row['H24']['mse'])} | {fmt_mean_std(row['H24']['mae'])} | {fmt_mean_std(row['H48']['mse'])} | {fmt_mean_std(row['H48']['mae'])} | {fmt_mean_std(row['H96']['mse'])} | {fmt_mean_std(row['H96']['mae'])} |"
        )
    lines.append("")
    lines.append("## Experiment 3: Cross-Domain ETTh1→ETTh2 (H=24, mean ± std over 2 seeds)")
    lines.append("| Model | ETTh1→ETTh1 MSE | ETTh1→ETTh2 MSE | ETTh2→ETTh2 MSE | Transfer Gap (%) |")
    lines.append("|-------|----------------|----------------|----------------|------------------|")
    for row in exp3_rows:
        lines.append(
            f"| {row['model']} | {fmt_mean_std(row['ETTh1→ETTh1']['mse'])} | {fmt_mean_std(row['ETTh1→ETTh2']['mse'])} | {fmt_mean_std(row['ETTh2→ETTh2']['mse'])} | {row['transfer_gap']:.2f} |"
        )
    lines.append("")
    lines.append("## Experiment 4: Ablation Study")
    lines.append("")
    lines.append("### 4a: UCI HAR (mean ± std over 2 seeds)")
    lines.append("| Variant | Params (K) | Accuracy (%) | F1-Macro (%) | Δ Acc |")
    lines.append("|---------|-----------|-------------|-------------|-------|")
    for model_name, summary in lite_results['exp4']['har'].items():
        label = LITE_DISPLAY[model_name]
        delta = "—" if model_name == LITE_MODEL else f"{summary['delta_acc']:+.2f}"
        lines.append(f"| {label} | {summary['n_params'] / 1000.0:.1f} | {fmt_mean_std(summary['accuracy'], 100.0)} | {fmt_mean_std(summary['f1_macro'], 100.0)} | {delta} |")
    lines.append("")
    lines.append("### 4b: ETTh1→ETTh2 (H=24, mean ± std over 2 seeds)")
    lines.append("| Variant | MSE | MAE | Δ MSE |")
    lines.append("|---------|-----|-----|-------|")
    for model_name, summary in lite_results['exp4']['cross_forecast'].items():
        label = LITE_DISPLAY[model_name]
        delta = "—" if model_name == LITE_MODEL else f"{summary['delta_mse']:+.4f}"
        lines.append(f"| {label} | {fmt_mean_std(summary['mse'])} | {fmt_mean_std(summary['mae'])} | {delta} |")
    lines.append("")
    lines.append("## Experiment 5: Hyperparameter Sensitivity (seed=42)")
    lines.append("### 5a: d_model")
    lines.append("| d_model | Params (K) | HAR Acc (%) | ETTh1 MSE | Cross-Domain MSE |")
    lines.append("|---------|-----------|-------------|-----------|------------------|")
    for item in lite_results['exp5']['d_model']:
        lines.append(f"| {item['config']['d_model']} | {item['params_k']:.1f} | {item['har_accuracy']:.2f} | {item['etth1_mse']:.4f} | {item['cross_mse']:.4f} |")
    lines.append("")
    lines.append("### 5b: n_heads")
    lines.append("| n_heads | HAR Acc (%) | ETTh1 MSE | Cross-Domain MSE |")
    lines.append("|---------|-------------|-----------|------------------|")
    for item in lite_results['exp5']['n_heads']:
        lines.append(f"| {item['config']['n_heads']} | {item['har_accuracy']:.2f} | {item['etth1_mse']:.4f} | {item['cross_mse']:.4f} |")
    lines.append("")
    lines.append("### 5c: n_enc_layers")
    lines.append("| n_enc_layers | Params (K) | HAR Acc (%) | ETTh1 MSE | Cross-Domain MSE |")
    lines.append("|-------------|-----------|-------------|-----------|------------------|")
    for item in lite_results['exp5']['n_enc_layers']:
        lines.append(f"| {item['config']['n_enc_layers']} | {item['params_k']:.1f} | {item['har_accuracy']:.2f} | {item['etth1_mse']:.4f} | {item['cross_mse']:.4f} |")
    lines.append("")
    lines.append("### 5d: modality_dropout")
    lines.append("| p | HAR Acc (%) | ETTh1 MSE | Cross-Domain MSE |")
    lines.append("|---|-------------|-----------|------------------|")
    for item in lite_results['exp5']['modality_dropout']:
        lines.append(f"| {item['config']['modality_dropout']:.2f} | {item['har_accuracy']:.2f} | {item['etth1_mse']:.4f} | {item['cross_mse']:.4f} |")
    lines.append("")
    lines.append("## Experiment 6: Efficiency")
    lines.append("| Model | Params (K) | Train (s/epoch) | Inference (ms/batch) | HAR Acc (%) |")
    lines.append("|-------|-----------|-----------------|----------------------|-------------|")
    for row in exp6_rows:
        lines.append(f"| {row['model']} | {row['params_k']:.1f} | {row['train_time_per_epoch_s']:.2f} | {row['inference_time_ms']:.2f} | {row['accuracy']:.2f} |")
    lines.append("")
    lines.append("## Experiment 7: ETTm1 Forecasting (seed=42)")
    lines.append("| Model | H=24 MSE | H=24 MAE | H=48 MSE | H=48 MAE | H=96 MSE | H=96 MAE |")
    lines.append("|-------|---------|---------|---------|---------|---------|---------|")
    for row in exp7_rows:
        h24 = row['H24']
        h48 = row['H48']
        h96 = row['H96']
        lines.append(f"| {row['model']} | {h24['mse']:.4f} | {h24['mae']:.4f} | {h48['mse']:.4f} | {h48['mae']:.4f} | {h96['mse']:.4f} | {h96['mae']:.4f} |")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- TMTFNet-Lite params: {cls_params:.1f}K (vs PatchTST {patchtst_params:.1f}K, vs TMTFNet_v2 {v2_params:.1f}K)")
    lines.append(f"- UCI HAR: {lite_results['exp1']['accuracy']['mean'] * 100.0:.2f}% accuracy, rank {lite_rank} of 10")
    lines.append(f"- ETTh1 H=24: {lite_results['exp2']['H24']['mse']['mean']:.4f} MSE, rank {lite_deep_rank} among deep models")
    lines.append(f"- Cross-domain: {lite_results['exp3']['ETTh1→ETTh2']['mse']['mean']:.4f} MSE, Transfer Gap {lite_results['exp3']['transfer_gap']:.2f}%")
    lines.append(f"- All ablation components positive: {'yes' if all_ablation_positive else 'no'}")
    with open(args.output_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TMTFNet-Lite experiments.")
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--result_dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--output_path", default=os.path.join(PROJECT_ROOT, "LITE_RESULTS.md"))
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device, detected_gpu_name, _ = setup_device()
    total_start = time.time()
    print("[1/8] Loading baseline summaries...")
    baselines = baseline_sources(args.result_dir)
    print("[2/8] Running Lite hyperparameter search...")
    search_summary = run_search(args, device)
    final_config = search_summary['final']['config']
    print(f"[search] Selected config: {final_config}")
    print("[3/8] Running Experiment 1 (HAR classification)...")
    exp1 = run_lite_exp1(args, device, final_config)
    print("[4/8] Running Experiment 2 (ETTh1 forecasting)...")
    exp2 = run_lite_exp2(args, device, final_config)
    print("[5/8] Running Experiment 3 (cross-domain forecasting)...")
    exp3 = run_lite_exp3(args, device, final_config)
    print("[6/8] Running Experiment 4 (ablations)...")
    exp4 = run_lite_exp4(args, device, final_config)
    print("[7/8] Running Experiment 5 and 6 (sensitivity + efficiency)...")
    exp5 = run_lite_exp5(args, device, final_config)
    exp6 = run_lite_exp6(args, device, final_config, exp1)
    print("[8/8] Running Experiment 7 (ETTm1 forecasting)...")
    exp7 = run_lite_exp7(args, device, final_config)
    lite_results = {
        'exp1': exp1,
        'exp2': exp2,
        'exp3': exp3,
        'exp4': exp4,
        'exp5': exp5,
        'exp6': exp6,
        'exp7': exp7,
    }
    total_time_s = time.time() - total_start
    print("[final] Rendering LITE_RESULTS.md...")
    render_results(args, baselines, search_summary, final_config, lite_results, total_time_s, detected_gpu_name)
    print(f"LITE_RESULTS.md written to {args.output_path}")


if __name__ == '__main__':
    main()
