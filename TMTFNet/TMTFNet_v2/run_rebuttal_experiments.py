"""Unified 5-seed rebuttal runner.

One script drives every experiment that feeds the revised manuscript:

    exp1a_har_uci       -- UCI HAR with deep baselines + TMTFNet variants + DA models
    exp1b_har_classical -- SVM / RF / kNN / LR on UCI HAR
    exp1c_har_pamap2    -- PAMAP2 (12-class) in-subject + cross-subject HAR
    exp2_forecast       -- ETTh1 / ETTh2 / ETTm1 forecasting at H in {24, 48, 96}
    exp3_cross_domain   -- ETTh1<->ETTh2, ETTm1<->ETTm2, ETTh1->ETTm1(hourly)
    exp3_da             -- same cross-domain splits with DANN / CoDATS / AdvSKM /
                           RAINCOAT + TMTFNet_v2 with CORAL/consistency/DANN
    exp4_ablation       -- TMTFNet ablations on HAR + cross-domain + alignment ablations
    exp5_sensitivity    -- hyperparameter sweep (5 seeds)

Each (seed, model, experiment) triple is cached as JSON in
``results/rebuttal_<experiment>/<model>_seed<seed>_<suffix>.json``; the
aggregated summary is stored at ``results/rebuttal_<experiment>/summary.json``.
Re-running the script only re-computes missing combinations, so the matrix
can be built up incrementally as GPU time becomes available.

Usage::

    python run_rebuttal_experiments.py --only exp1a_har_uci
    python run_rebuttal_experiments.py --only exp2_forecast --seeds 42 123 456
    python run_rebuttal_experiments.py                       # run everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines_classical import run_classical_baselines  # noqa: E402
from src.datasets import get_ett_loaders, get_har_loaders, get_pamap2_loaders  # noqa: E402
from src.models import MODEL_REGISTRY, build_model  # noqa: E402
from src.baselines_da import DA_MODEL_REGISTRY  # noqa: E402
from src.trainer import (count_parameters, measure_inference_time,  # noqa: E402
                         set_global_seed, train_and_evaluate)
from src.trainer_da import train_and_evaluate_da  # noqa: E402

DEFAULT_SEEDS = (42, 123, 456, 789, 2024)


def _run_guarded(fn, *args, **kwargs):
    """Call ``fn`` but don't let a single OOM / RuntimeError abort the whole matrix."""
    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        print("  ! CUDA OOM, skipping this (seed, model) and continuing")
        torch.cuda.empty_cache()
        return None
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "cuda" in str(exc):
            print(f"  ! CUDA runtime error ({exc.__class__.__name__}: {exc}) -- skipping")
            torch.cuda.empty_cache()
            return None
        raise

HAR_DEEP_MODELS = [
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
    "CNN1D",
    "DeepConvLSTM",
]
FORECAST_DEEP_MODELS = HAR_DEEP_MODELS[:-2]  # CNN1D/DeepConvLSTM only meaningful for classification here

DATA_DIR = str(REPO_ROOT / "data")
RESULTS_ROOT = REPO_ROOT / "results"


def _result_path(exp_name, fname):
    out = RESULTS_ROOT / f"rebuttal_{exp_name}"
    out.mkdir(parents=True, exist_ok=True)
    return out / fname


def _done(path):
    return path.exists() and path.stat().st_size > 0


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=_json_safe)


def _json_safe(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Cannot serialise {type(obj).__name__}")


def _agg(runs, key):
    xs = np.asarray([run[key] for run in runs], dtype=np.float64)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if xs.size > 1 else 0.0}


def _wilcoxon_against_reference(per_seed_results, metric, reference):
    """Paired Wilcoxon signed-rank test: reference vs every other model.

    Returns ``{model: (statistic, p_value)}`` (skipping the reference).
    """
    from scipy.stats import wilcoxon

    results = {}
    if reference not in per_seed_results:
        return results
    ref_scores = np.asarray([r[metric] for r in per_seed_results[reference]], dtype=np.float64)
    for model, runs in per_seed_results.items():
        if model == reference or len(runs) != len(ref_scores):
            continue
        scores = np.asarray([r[metric] for r in runs], dtype=np.float64)
        if np.allclose(ref_scores, scores):
            results[model] = {"statistic": 0.0, "p_value": 1.0}
            continue
        try:
            stat, p = wilcoxon(ref_scores, scores, zero_method="wilcox", alternative="two-sided")
            results[model] = {"statistic": float(stat), "p_value": float(p)}
        except ValueError as exc:
            results[model] = {"statistic": None, "p_value": None, "error": str(exc)}
    return results


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------- exp1a: HAR on UCI --------------------------------------------
def run_exp1a_har_uci(seeds, lr_cls=5e-4, max_epochs=100, patience=15):
    exp = "exp1a_har_uci"
    per_seed = {}
    for seed in seeds:
        set_global_seed(int(seed))
        bundle = get_har_loaders(DATA_DIR, batch_size=64, num_workers=2, seq_len=128)
        for model_name in HAR_DEEP_MODELS:
            path = _result_path(exp, f"{model_name}_seed{seed}.json")
            if _done(path):
                per_seed.setdefault(model_name, []).append(json.loads(path.read_text()))
                continue
            set_global_seed(int(seed))
            cfg = dict(
                d_model=64 if model_name == "TMTFNet_v2" else 128,
                n_heads=8,
                n_enc_layers=2 if model_name == "TMTFNet_v2" else 3,
                modality_dropout=0.1,
                seq_len=128,
                n_classes=bundle.n_classes,
                dropout=0.1,
            )
            model = build_model(model_name, modality_dims=bundle.modality_dims, **cfg)
            n_params = count_parameters(model)
            result = train_and_evaluate(
                model=model,
                loaders=bundle.loaders,
                task="classification",
                device=_device(),
                lr=lr_cls,
                max_epochs=max_epochs,
                patience=patience,
                label_smoothing=0.1,
            )
            result.update({"seed": int(seed), "model_name": model_name, "n_params": int(n_params)})
            _save_json(path, result)
            per_seed.setdefault(model_name, []).append(result)
    _write_summary(exp, per_seed, metric="accuracy", reference="TMTFNet_v2")
    return per_seed


# ------------------------------- exp1b: classical HAR on UCI ----------------------------------
def run_exp1b_har_classical(seeds):
    exp = "exp1b_har_classical"
    path = _result_path(exp, "all_seeds.json")
    if _done(path):
        return json.loads(path.read_text())
    data_subdir = os.path.join(DATA_DIR, "UCI HAR Dataset")
    results = run_classical_baselines(data_subdir, seeds=tuple(seeds))
    _save_json(path, results)
    return results


# ------------------------------- exp1c: HAR on PAMAP2 -----------------------------------------
def run_exp1c_har_pamap2(seeds, lr_cls=5e-4, max_epochs=60, patience=10):
    exp = "exp1c_har_pamap2"
    per_seed = {}
    for seed in seeds:
        set_global_seed(int(seed))
        bundle = get_pamap2_loaders(os.path.join(DATA_DIR, "PAMAP2_Dataset"), batch_size=64, num_workers=2, split_seed=int(seed))
        for model_name in HAR_DEEP_MODELS:
            path = _result_path(exp, f"{model_name}_seed{seed}.json")
            if _done(path):
                per_seed.setdefault(model_name, []).append(json.loads(path.read_text()))
                continue
            set_global_seed(int(seed))
            cfg = dict(
                d_model=64 if model_name == "TMTFNet_v2" else 128,
                n_heads=8,
                n_enc_layers=2 if model_name == "TMTFNet_v2" else 3,
                modality_dropout=0.1,
                seq_len=171,
                n_classes=bundle.n_classes,
                dropout=0.1,
            )
            model = build_model(model_name, modality_dims=bundle.modality_dims, **cfg)
            n_params = count_parameters(model)
            result = train_and_evaluate(
                model=model,
                loaders=bundle.loaders,
                task="classification",
                device=_device(),
                lr=lr_cls,
                max_epochs=max_epochs,
                patience=patience,
                label_smoothing=0.1,
            )
            result.update({"seed": int(seed), "model_name": model_name, "n_params": int(n_params)})
            _save_json(path, result)
            per_seed.setdefault(model_name, []).append(result)
    _write_summary(exp, per_seed, metric="accuracy", reference="TMTFNet_v2")
    return per_seed


# ------------------------------- exp2: forecasting ETT ----------------------------------------
def run_exp2_forecast(seeds, datasets=("ETTh1", "ETTh2", "ETTm1"), horizons=(24, 48, 96), lr_fc=1e-3, max_epochs=100, patience=15):
    per_ds = {}
    for ds in datasets:
        for h in horizons:
            exp = f"exp2_forecast_{ds.lower()}_pred{h}"
            per_seed = {}
            for seed in seeds:
                set_global_seed(int(seed))
                bundle = get_ett_loaders(DATA_DIR, train_dataset=ds, seq_len=96, pred_len=h, num_workers=2)
                for model_name in FORECAST_DEEP_MODELS:
                    path = _result_path(exp, f"{model_name}_seed{seed}.json")
                    if _done(path):
                        per_seed.setdefault(model_name, []).append(json.loads(path.read_text()))
                        continue
                    set_global_seed(int(seed))
                    cfg = dict(
                        d_model=64 if model_name == "TMTFNet_v2" else 128,
                        n_heads=8,
                        n_enc_layers=2 if model_name == "TMTFNet_v2" else 3,
                        modality_dropout=0.1,
                        seq_len=96,
                        pred_len=h,
                        dropout=0.1,
                    )
                    model = build_model(model_name, modality_dims=bundle.modality_dims, **cfg)
                    n_params = count_parameters(model)
                    result = train_and_evaluate(
                        model=model,
                        loaders=bundle.loaders,
                        task="forecast",
                        device=_device(),
                        lr=lr_fc,
                        max_epochs=max_epochs,
                        patience=patience,
                    )
                    result.update({"seed": int(seed), "model_name": model_name, "n_params": int(n_params)})
                    _save_json(path, result)
                    per_seed.setdefault(model_name, []).append(result)
            _write_summary(exp, per_seed, metric="mse", reference="TMTFNet_v2", minimize=True)
            per_ds[(ds, h)] = per_seed
    return per_ds


# ---------------- exp3: cross-domain forecasting (no target label) -----------------------------
CROSS_DOMAIN_SPLITS = [
    # (train_ds, test_ds, resample_target_hourly, label)
    ("ETTh1", "ETTh2", False, "h1_to_h2"),
    ("ETTh2", "ETTh1", False, "h2_to_h1"),
    ("ETTm1", "ETTm2", False, "m1_to_m2"),
    ("ETTm2", "ETTm1", False, "m2_to_m1"),
    ("ETTh1", "ETTm1", True, "h1_to_m1"),
]


def run_exp3_cross_forecast(seeds, h=24, lr_fc=1e-3, max_epochs=100, patience=15):
    exp = f"exp3_cross_forecast_pred{h}"
    per_split = {}
    for src, tgt, resample, label in CROSS_DOMAIN_SPLITS:
        per_seed = {}
        for seed in seeds:
            set_global_seed(int(seed))
            bundle = get_ett_loaders(
                DATA_DIR,
                train_dataset=src,
                test_dataset=tgt,
                fit_dataset=src,
                seq_len=96,
                pred_len=h,
                num_workers=2,
                resample_test_to_hourly=resample,
            )
            for model_name in FORECAST_DEEP_MODELS:
                path = _result_path(exp, f"{label}_{model_name}_seed{seed}.json")
                if _done(path):
                    per_seed.setdefault(model_name, []).append(json.loads(path.read_text()))
                    continue
                set_global_seed(int(seed))
                cfg = dict(
                    d_model=64 if model_name == "TMTFNet_v2" else 128,
                    n_heads=8,
                    n_enc_layers=2 if model_name == "TMTFNet_v2" else 3,
                    modality_dropout=0.1,
                    seq_len=96,
                    pred_len=h,
                    dropout=0.1,
                )
                model = build_model(model_name, modality_dims=bundle.modality_dims, **cfg)
                n_params = count_parameters(model)
                result = train_and_evaluate(
                    model=model,
                    loaders=bundle.loaders,
                    task="forecast",
                    device=_device(),
                    lr=lr_fc,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                result.update({"seed": int(seed), "model_name": model_name, "n_params": int(n_params),
                               "split": label, "source": src, "target": tgt})
                _save_json(path, result)
                per_seed.setdefault(model_name, []).append(result)
        _write_summary(_label_exp(exp, label), per_seed, metric="mse", reference="TMTFNet_v2", minimize=True)
        per_split[label] = per_seed
    return per_split


def _label_exp(exp_name, label):
    return f"{exp_name}__{label}"


# ---------------- exp3 DA: cross-domain DA baselines ------------------------------------------
def run_exp3_da_forecast(seeds, h=24, lr_fc=1e-3, max_epochs=100, patience=15,
                         lambda_coral=1.0, lambda_consistency=0.1, lambda_dann=0.1):
    exp = f"exp3_cross_da_pred{h}"
    per_split = {}
    for src, tgt, resample, label in CROSS_DOMAIN_SPLITS:
        per_seed = {}
        for seed in seeds:
            set_global_seed(int(seed))
            bundle = get_ett_loaders(
                DATA_DIR,
                train_dataset=src,
                test_dataset=tgt,
                fit_dataset=src,
                seq_len=96,
                pred_len=h,
                num_workers=2,
                resample_test_to_hourly=resample,
                emit_target_train=True,
            )
            for model_name, model_cls in DA_MODEL_REGISTRY.items():
                path = _result_path(exp, f"{label}_{model_name}_seed{seed}.json")
                if _done(path):
                    per_seed.setdefault(model_name, []).append(json.loads(path.read_text()))
                    continue
                set_global_seed(int(seed))
                model = model_cls(modality_dims=bundle.modality_dims, d_model=128, seq_len=96, pred_len=h, dropout=0.1)
                n_params = count_parameters(model)
                result = train_and_evaluate_da(
                    model=model,
                    loaders=bundle.loaders,
                    target_loader=bundle.loaders.get("target_train"),
                    task="forecast",
                    device=_device(),
                    lr=lr_fc,
                    max_epochs=max_epochs,
                    patience=patience,
                    lambda_coral=1.0 if model_name in ("RAINCOAT", "AdvSKM") else 0.0,
                    lambda_consistency=0.0,
                    lambda_dann=1.0 if model_name in ("DANN", "CoDATS") else 0.0,
                )
                result.update({"seed": int(seed), "model_name": model_name, "n_params": int(n_params),
                               "split": label, "source": src, "target": tgt})
                _save_json(path, result)
                per_seed.setdefault(model_name, []).append(result)

            # TMTFNet_v2 with alignment on
            model_name = "TMTFNet_v2_DA"
            path = _result_path(exp, f"{label}_{model_name}_seed{seed}.json")
            if not _done(path):
                set_global_seed(int(seed))
                model = build_model(
                    "TMTFNet_v2",
                    modality_dims=bundle.modality_dims,
                    d_model=64,
                    n_heads=8,
                    n_enc_layers=2,
                    pred_len=h,
                    seq_len=96,
                    dropout=0.1,
                    modality_dropout=0.1,
                    use_domain_adapt=True,
                )
                n_params = count_parameters(model)
                result = train_and_evaluate_da(
                    model=model,
                    loaders=bundle.loaders,
                    target_loader=bundle.loaders.get("target_train"),
                    task="forecast",
                    device=_device(),
                    lr=lr_fc,
                    max_epochs=max_epochs,
                    patience=patience,
                    lambda_coral=lambda_coral,
                    lambda_consistency=lambda_consistency,
                    lambda_dann=lambda_dann,
                )
                result.update({"seed": int(seed), "model_name": model_name, "n_params": int(n_params),
                               "split": label, "source": src, "target": tgt,
                               "lambda_coral": lambda_coral, "lambda_consistency": lambda_consistency,
                               "lambda_dann": lambda_dann})
                _save_json(path, result)
            per_seed.setdefault(model_name, []).append(json.loads(path.read_text()))
        _write_summary(_label_exp(exp, label), per_seed, metric="mse", reference="TMTFNet_v2_DA", minimize=True)
        per_split[label] = per_seed
    return per_split


# ---------------- exp4: ablation --------------------------------------------------------------
ABLATION_VARIANTS = [
    "TMTFNet_v2",
    "TMTFNet_v2_NoGCMTA",
    "TMTFNet_v2_NoAMG",
    "TMTFNet_v2_NoAHTF",
    "TMTFNet_v2_NoAttnPool",
    "TMTFNet_v2_NoModDrop",
]


def run_exp4_ablation(seeds, lr_cls=5e-4, lr_fc=1e-3, max_epochs=80, patience=15):
    exp = "exp4_ablation"
    per_task = {"har": {}, "cross_h1_to_h2": {}}
    for seed in seeds:
        # HAR ablation
        set_global_seed(int(seed))
        bundle = get_har_loaders(DATA_DIR, batch_size=64, num_workers=2, seq_len=128)
        for model_name in ABLATION_VARIANTS:
            path = _result_path(exp, f"har_{model_name}_seed{seed}.json")
            if _done(path):
                per_task["har"].setdefault(model_name, []).append(json.loads(path.read_text()))
                continue
            set_global_seed(int(seed))
            model = build_model(
                model_name, modality_dims=bundle.modality_dims,
                d_model=64, n_heads=8, n_enc_layers=2,
                modality_dropout=0.1, seq_len=128, n_classes=bundle.n_classes, dropout=0.1,
            )
            result = train_and_evaluate(model, bundle.loaders, task="classification",
                                        device=_device(), lr=lr_cls,
                                        max_epochs=max_epochs, patience=patience, label_smoothing=0.1)
            result.update({"seed": int(seed), "model_name": model_name, "n_params": count_parameters(model)})
            _save_json(path, result)
            per_task["har"].setdefault(model_name, []).append(result)

        # Cross-domain forecasting ablation
        set_global_seed(int(seed))
        bundle = get_ett_loaders(DATA_DIR, train_dataset="ETTh1", test_dataset="ETTh2", fit_dataset="ETTh1",
                                 seq_len=96, pred_len=24, num_workers=2)
        for model_name in ABLATION_VARIANTS:
            path = _result_path(exp, f"cross_h1_to_h2_{model_name}_seed{seed}.json")
            if _done(path):
                per_task["cross_h1_to_h2"].setdefault(model_name, []).append(json.loads(path.read_text()))
                continue
            set_global_seed(int(seed))
            model = build_model(
                model_name, modality_dims=bundle.modality_dims,
                d_model=64, n_heads=8, n_enc_layers=2,
                modality_dropout=0.1, seq_len=96, pred_len=24, dropout=0.1,
            )
            result = train_and_evaluate(model, bundle.loaders, task="forecast",
                                        device=_device(), lr=lr_fc,
                                        max_epochs=max_epochs, patience=patience)
            result.update({"seed": int(seed), "model_name": model_name, "n_params": count_parameters(model)})
            _save_json(path, result)
            per_task["cross_h1_to_h2"].setdefault(model_name, []).append(result)
    _write_summary(f"{exp}__har", per_task["har"], metric="accuracy", reference="TMTFNet_v2")
    _write_summary(f"{exp}__cross_h1_to_h2", per_task["cross_h1_to_h2"], metric="mse", reference="TMTFNet_v2", minimize=True)
    return per_task


# ---------------- exp5: sensitivity -----------------------------------------------------------
SENSITIVITY_GRID = {
    "d_model": [32, 48, 64, 80, 96, 128],
    "n_heads": [2, 4, 8],
    "n_enc_layers": [1, 2, 3, 4],
    "modality_dropout": [0.0, 0.05, 0.1, 0.15, 0.2],
}
SENSITIVITY_DEFAULT = dict(d_model=64, n_heads=8, n_enc_layers=2, modality_dropout=0.1)


def run_exp5_sensitivity(seeds, lr_cls=5e-4, max_epochs=80, patience=15):
    exp = "exp5_sensitivity"
    out = {}
    for seed in seeds:
        set_global_seed(int(seed))
        bundle = get_har_loaders(DATA_DIR, batch_size=64, num_workers=2, seq_len=128)
        for key, values in SENSITIVITY_GRID.items():
            for val in values:
                cfg = dict(SENSITIVITY_DEFAULT)
                cfg[key] = val
                tag = f"{key}={val}"
                path = _result_path(exp, f"{key}_{val}_seed{seed}.json")
                if _done(path):
                    out.setdefault(tag, []).append(json.loads(path.read_text()))
                    continue
                set_global_seed(int(seed))
                model = build_model(
                    "TMTFNet_v2", modality_dims=bundle.modality_dims,
                    d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_enc_layers=cfg["n_enc_layers"],
                    modality_dropout=cfg["modality_dropout"], seq_len=128,
                    n_classes=bundle.n_classes, dropout=0.1,
                )
                result = train_and_evaluate(model, bundle.loaders, task="classification",
                                            device=_device(), lr=lr_cls,
                                            max_epochs=max_epochs, patience=patience, label_smoothing=0.1)
                result.update({"seed": int(seed), "config": cfg, "n_params": count_parameters(model)})
                _save_json(path, result)
                out.setdefault(tag, []).append(result)
    # aggregate
    summary = {tag: {"accuracy": _agg(runs, "accuracy"), "f1_macro": _agg(runs, "f1_macro"),
                     "n_seeds": len(runs), "config": runs[0].get("config")}
               for tag, runs in out.items()}
    _save_json(_result_path(exp, "summary.json"), summary)
    return out


# ---------------- summary writer --------------------------------------------------------------
def _write_summary(exp, per_seed, metric, reference=None, minimize=False):
    out = {}
    for model, runs in per_seed.items():
        agg_metric = _agg(runs, metric)
        out[model] = {
            metric: agg_metric,
            "n_seeds": len(runs),
            "n_params": runs[0].get("n_params"),
        }
        if metric == "accuracy":
            out[model]["f1_macro"] = _agg(runs, "f1_macro")
            out[model]["f1_weighted"] = _agg(runs, "f1_weighted")
        if metric == "mse":
            out[model]["mae"] = _agg(runs, "mae")
        out[model]["training_time"] = _agg(runs, "training_time") if "training_time" in runs[0] else None
    if reference:
        out["_wilcoxon"] = _wilcoxon_against_reference(per_seed, metric, reference)
        out["_reference"] = reference
    out["_metric"] = metric
    out["_minimize"] = minimize
    _save_json(_result_path(exp, "summary.json"), out)
    return out


# ---------------- driver -----------------------------------------------------------------------
EXPERIMENTS = {
    "exp1a_har_uci": run_exp1a_har_uci,
    "exp1b_har_classical": run_exp1b_har_classical,
    "exp1c_har_pamap2": run_exp1c_har_pamap2,
    "exp2_forecast": run_exp2_forecast,
    "exp3_cross_forecast": run_exp3_cross_forecast,
    "exp3_da_forecast": run_exp3_da_forecast,
    "exp4_ablation": run_exp4_ablation,
    "exp5_sensitivity": run_exp5_sensitivity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None, help="Run only these experiments (by key). Omit to run all.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = args.only or list(EXPERIMENTS.keys())
    missing = [k for k in target if k not in EXPERIMENTS]
    if missing:
        raise SystemExit(f"Unknown experiments: {missing}. Available: {list(EXPERIMENTS)}")

    start_time = time.time()
    for key in target:
        print(f"\n===== {key} =====")
        if args.dry_run:
            print("  (dry-run skip)")
            continue
        fn = EXPERIMENTS[key]
        fn(args.seeds)
        print(f"  done in {time.time() - start_time:.0f}s (cumulative)")


if __name__ == "__main__":
    main()
