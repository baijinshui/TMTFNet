#!/usr/bin/env python3
"""
TMTFNet: Full Experiment Pipeline
==================================
Runs all experiments for the paper:
  Exp 1: Multi-Modal Classification (UCI HAR / Synthetic)
  Exp 2: Cross-Domain Time Series Forecasting (ETT / Synthetic)
  Exp 3: Ablation Study
  Exp 4: Cross-Domain Transfer Analysis
  Exp 5: Scalability & Efficiency Analysis

Usage:
  conda activate boluo_huo
  python run_experiments.py --use_real_data    # with downloaded datasets
  python run_experiments.py                     # synthetic data only (fast test)

Optimized for RTX 5090 with mixed precision (fp16).
Estimated time: ~15-30 min (synthetic), ~30-60 min (real data)
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models import build_model, MODEL_REGISTRY
from src.datasets import (get_har_loaders, get_ett_loaders, get_synthetic_loaders,
                          collate_multimodal)
from src.trainer import run_single_experiment, count_parameters
from src.visualize import (generate_all_figures, plot_cross_domain_results,
                           plot_training_curves, plot_ablation_study)


def setup_device():
    """Setup CUDA device with optimal settings."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        gpu_mem = total_mem / 1e9 if total_mem else 0
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected, using CPU (will be slow)")
    return device


def print_banner(text):
    print(f"\n{'#'*70}")
    print(f"# {text}")
    print(f"{'#'*70}\n")


# ============================================================
# Experiment 1: Multi-Modal Classification
# ============================================================

def run_exp1_classification(args, device):
    """Experiment 1: Multi-modal sensor fusion for activity recognition."""
    print_banner("Experiment 1: Multi-Modal Classification")

    if args.use_real_data and os.path.exists(os.path.join(args.data_dir, 'UCI HAR Dataset')):
        print("Using UCI HAR dataset...")
        loaders, mod_dims, n_classes = get_har_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
            seq_len=128
        )
        class_names = ['Walking', 'WalkingUp', 'WalkingDown', 'Sitting', 'Standing', 'Laying']
        exp_name = 'har'
    else:
        print("Using synthetic classification data...")
        loaders, mod_dims, n_classes = get_synthetic_loaders(
            task='classification', batch_size=args.batch_size, num_workers=args.num_workers,
            n_modalities=3, dims_per_mod=3, seq_len=64, n_classes=6, seed=42
        )
        class_names = [f'Activity_{i}' for i in range(n_classes)]
        exp_name = 'synthetic_cls'

    print(f"Modality dims: {mod_dims}, Classes: {n_classes}")

    # Models to evaluate
    model_configs = {
        'TMTFNet': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                    'n_enc_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout},
        'LSTM': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
                'n_classes': n_classes, 'dropout': args.dropout},
        'GRU': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
               'n_classes': n_classes, 'dropout': args.dropout},
        'TCN': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': 4,
               'n_classes': n_classes, 'dropout': args.dropout},
        'Transformer': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                       'n_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout},
        'Crossformer': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                       'n_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout},
    }

    results = []
    save_dir = os.path.join(args.result_dir, 'exp1_classification')
    os.makedirs(save_dir, exist_ok=True)

    for model_name, kwargs in model_configs.items():
        result = run_single_experiment(
            model_name, kwargs, loaders, task='classification', device=device,
            max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
            seed=args.seed, save_dir=save_dir, verbose=True
        )
        results.append(result)

    # Save results
    serializable = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    # Generate figures
    fig_dir = os.path.join(args.fig_dir, 'exp1')
    generate_all_figures(results, fig_dir, task='classification', class_names=class_names)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<20} {'Params':>10} {'Accuracy':>10} {'F1-Macro':>10} {'Precision':>10} {'Recall':>10} {'Time(s)':>10}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['model_name']:<20} {r['n_params']:>10,} {r['test_accuracy']:>10.4f} "
              f"{r['test_f1_macro']:>10.4f} {r['test_precision']:>10.4f} "
              f"{r['test_recall']:>10.4f} {r['training_time']:>10.1f}")
    print(f"{'='*80}")

    return results


# ============================================================
# Experiment 2: Time Series Forecasting
# ============================================================

def run_exp2_forecasting(args, device):
    """Experiment 2: Multi-modal time series forecasting."""
    print_banner("Experiment 2: Multi-Modal Time Series Forecasting")

    all_results = {}

    # Test multiple prediction lengths
    pred_lens = [24, 48, 96]

    for pred_len in pred_lens:
        print(f"\n--- Prediction Length: {pred_len} ---")

        if args.use_real_data and os.path.exists(os.path.join(args.data_dir, 'ETTh1.csv')):
            print(f"Using ETTh1 dataset (pred_len={pred_len})...")
            loaders, mod_dims, _ = get_ett_loaders(
                args.data_dir, 'ETTh1', batch_size=args.batch_size,
                num_workers=args.num_workers, seq_len=96, pred_len=pred_len
            )
            exp_tag = f'etth1_pred{pred_len}'
        else:
            print(f"Using synthetic forecasting data (pred_len={pred_len})...")
            loaders, mod_dims, _ = get_synthetic_loaders(
                task='forecasting', batch_size=args.batch_size, num_workers=args.num_workers,
                n_modalities=2, dims_per_mod=4, seq_len=96, pred_len=pred_len, seed=42
            )
            exp_tag = f'synthetic_pred{pred_len}'

        model_configs = {
            'TMTFNet': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                       'n_enc_layers': args.n_layers, 'pred_len': pred_len, 'dropout': args.dropout},
            'LSTM': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
                    'pred_len': pred_len, 'dropout': args.dropout},
            'GRU': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
                   'pred_len': pred_len, 'dropout': args.dropout},
            'TCN': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': 4,
                   'pred_len': pred_len, 'dropout': args.dropout},
            'Transformer': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                           'n_layers': args.n_layers, 'pred_len': pred_len, 'dropout': args.dropout},
            'Crossformer': {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                           'n_layers': args.n_layers, 'pred_len': pred_len, 'dropout': args.dropout},
        }

        results = []
        save_dir = os.path.join(args.result_dir, f'exp2_forecast_{exp_tag}')
        os.makedirs(save_dir, exist_ok=True)

        for model_name, kwargs in model_configs.items():
            result = run_single_experiment(
                model_name, kwargs, loaders, task='forecasting', device=device,
                max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
                seed=args.seed, save_dir=save_dir, verbose=True
            )
            results.append(result)

        all_results[exp_tag] = results

        # Save results
        serializable = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        # Generate figures
        fig_dir = os.path.join(args.fig_dir, f'exp2_{exp_tag}')
        generate_all_figures(results, fig_dir, task='forecasting')

        # Print summary
        print(f"\n{'='*80}")
        print(f"{'Model':<20} {'Params':>10} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'Time(s)':>10}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['model_name']:<20} {r['n_params']:>10,} {r['test_mse']:>10.4f} "
                  f"{r['test_mae']:>10.4f} {r['test_rmse']:>10.4f} {r['training_time']:>10.1f}")
        print(f"{'='*80}")

    return all_results


# ============================================================
# Experiment 3: Ablation Study
# ============================================================

def run_exp3_ablation(args, device):
    """Experiment 3: Ablation study removing each component."""
    print_banner("Experiment 3: Ablation Study")

    # Use same data as Exp 1
    if args.use_real_data and os.path.exists(os.path.join(args.data_dir, 'UCI HAR Dataset')):
        loaders, mod_dims, n_classes = get_har_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, seq_len=128
        )
    else:
        loaders, mod_dims, n_classes = get_synthetic_loaders(
            task='classification', batch_size=args.batch_size, num_workers=args.num_workers,
            n_modalities=3, dims_per_mod=3, seq_len=64, n_classes=6, seed=42
        )

    base_kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                   'n_enc_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout}

    ablation_models = {
        'TMTFNet': 'TMTFNet',
        'TMTFNet_NoCMTA': 'TMTFNet_NoCMTA',    # w/o Cross-Modal Temporal Attention
        'TMTFNet_NoAMG': 'TMTFNet_NoAMG',      # w/o Adaptive Modality Gating
        'TMTFNet_NoHTF': 'TMTFNet_NoHTF',      # w/o Hierarchical Temporal Fusion
    }

    results = []
    save_dir = os.path.join(args.result_dir, 'exp3_ablation')
    os.makedirs(save_dir, exist_ok=True)

    for display_name, model_name in ablation_models.items():
        result = run_single_experiment(
            model_name, base_kwargs, loaders, task='classification', device=device,
            max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
            seed=args.seed, save_dir=save_dir, verbose=True
        )
        result['model_name'] = display_name
        results.append(result)

    serializable = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    fig_dir = os.path.join(args.fig_dir, 'exp3')
    os.makedirs(fig_dir, exist_ok=True)
    plot_ablation_study(results, os.path.join(fig_dir, 'ablation.png'), task='classification')

    print(f"\n{'='*70}")
    print(f"{'Variant':<25} {'Accuracy':>10} {'F1-Macro':>10} {'Delta':>10}")
    print(f"{'-'*70}")
    full_acc = results[0]['test_accuracy']
    for r in results:
        delta = r['test_accuracy'] - full_acc
        print(f"{r['model_name']:<25} {r['test_accuracy']:>10.4f} {r['test_f1_macro']:>10.4f} {delta:>+10.4f}")
    print(f"{'='*70}")

    return results


# ============================================================
# Experiment 4: Cross-Domain Transfer
# ============================================================

def run_exp4_cross_domain(args, device):
    """Experiment 4: Cross-domain transfer analysis."""
    print_banner("Experiment 4: Cross-Domain Transfer Analysis")

    scenarios = {
        'Domain_0→0 (In-domain)': (0, 0),
        'Domain_0→1 (Near)': (0, 1),
        'Domain_0→2 (Far)': (0, 2),
    }

    cross_results = {}
    models_to_test = ['TMTFNet', 'Transformer', 'LSTM']

    for scenario_name, (src_domain, tgt_domain) in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        cross_results[scenario_name] = {}

        # Source data
        src_loaders, mod_dims, n_classes = get_synthetic_loaders(
            task='classification', batch_size=args.batch_size, num_workers=args.num_workers,
            n_modalities=3, dims_per_mod=3, seq_len=64, n_classes=6,
            domain=src_domain, seed=42
        )

        # Target data (for testing)
        tgt_loaders, _, _ = get_synthetic_loaders(
            task='classification', batch_size=args.batch_size, num_workers=args.num_workers,
            n_modalities=3, dims_per_mod=3, seq_len=64, n_classes=6,
            domain=tgt_domain, seed=123
        )

        for model_name in models_to_test:
            if model_name == 'TMTFNet':
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                         'n_enc_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout}
            elif model_name == 'Transformer':
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                         'n_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout}
            else:
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
                         'n_classes': n_classes, 'dropout': args.dropout}

            # Train on source
            torch.manual_seed(args.seed)
            model = build_model(model_name, **kwargs)
            from src.trainer import Trainer
            trainer = Trainer(model, task='classification', device=device, lr=args.lr)
            trainer.train(src_loaders['train'], src_loaders['val'],
                        max_epochs=args.max_epochs, patience=args.patience, verbose=False)

            # Test on target
            test_metrics = trainer.evaluate(tgt_loaders['test'])
            cross_results[scenario_name][model_name] = float(test_metrics['accuracy'])
            print(f"  {model_name}: Accuracy = {test_metrics['accuracy']:.4f}")

    # Save and visualize
    save_dir = os.path.join(args.result_dir, 'exp4_cross_domain')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(cross_results, f, indent=2)

    fig_dir = os.path.join(args.fig_dir, 'exp4')
    os.makedirs(fig_dir, exist_ok=True)
    plot_cross_domain_results(
        cross_results,
        os.path.join(fig_dir, 'cross_domain.png'),
        metric_name='Accuracy',
        lower_is_better=False
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Scenario':<30}", end='')
    for m in models_to_test:
        print(f"{m:>12}", end='')
    print()
    print(f"{'-'*70}")
    for scenario, scores in cross_results.items():
        print(f"{scenario:<30}", end='')
        for m in models_to_test:
            print(f"{scores[m]:>12.4f}", end='')
        print()
    print(f"{'='*70}")

    return cross_results


# ============================================================
# Experiment 5: Scalability Analysis
# ============================================================

def run_exp5_scalability(args, device):
    """Experiment 5: Model scalability and efficiency analysis."""
    print_banner("Experiment 5: Scalability & Efficiency Analysis")

    d_models = [32, 64, 128, 256]
    results = []

    for d in d_models:
        print(f"\n--- d_model = {d} ---")
        loaders, mod_dims, n_classes = get_synthetic_loaders(
            task='classification', batch_size=args.batch_size, num_workers=args.num_workers,
            n_modalities=3, dims_per_mod=3, seq_len=64, n_classes=6, seed=42
        )

        kwargs = {'modality_dims': mod_dims, 'd_model': d, 'n_heads': min(args.n_heads, d // 16 or 1),
                 'n_enc_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout}

        result = run_single_experiment(
            'TMTFNet', kwargs, loaders, task='classification', device=device,
            max_epochs=30, patience=10, lr=args.lr, seed=args.seed, verbose=False
        )
        result['d_model'] = d
        results.append(result)
        print(f"  d_model={d}: Acc={result['test_accuracy']:.4f}, "
              f"Params={result['n_params']:,}, Time={result['training_time']:.1f}s")

    save_dir = os.path.join(args.result_dir, 'exp5_scalability')
    os.makedirs(save_dir, exist_ok=True)
    serializable = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    # Plot scalability
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    params = [r['n_params'] / 1000 for r in results]
    accs = [r['test_accuracy'] for r in results]
    times = [r['training_time'] for r in results]

    ax1.plot(params, accs, 'o-', color='#2196F3', markersize=8)
    ax1.set_xlabel('Parameters (K)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Model Size')

    ax2.plot(params, times, 's-', color='#FF5722', markersize=8)
    ax2.set_xlabel('Parameters (K)')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time vs Model Size')

    plt.tight_layout()
    fig_dir = os.path.join(args.fig_dir, 'exp5')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'scalability.png'), dpi=300)
    plt.close()

    return results


# ============================================================
# Experiment 6: Cross-Domain Forecasting (ETTh1 → ETTh2)
# ============================================================

def run_exp6_cross_domain_forecast(args, device):
    """Experiment 6: Cross-domain forecasting transfer."""
    print_banner("Experiment 6: Cross-Domain Forecasting Transfer")

    pred_len = 24

    # Check if real ETT data exists
    has_etth1 = os.path.exists(os.path.join(args.data_dir, 'ETTh1.csv'))
    has_etth2 = os.path.exists(os.path.join(args.data_dir, 'ETTh2.csv'))

    scenarios = {}

    if args.use_real_data and has_etth1 and has_etth2:
        print("Using real ETT datasets for cross-domain forecasting...")
        src_loaders, mod_dims, _ = get_ett_loaders(
            args.data_dir, 'ETTh1', args.batch_size, args.num_workers, 96, pred_len)
        tgt_loaders, _, _ = get_ett_loaders(
            args.data_dir, 'ETTh2', args.batch_size, args.num_workers, 96, pred_len)
        scenarios['ETTh1→ETTh1'] = (src_loaders, src_loaders)
        scenarios['ETTh1→ETTh2'] = (src_loaders, tgt_loaders)
    else:
        print("Using synthetic data for cross-domain forecasting...")
        for src_d, tgt_d, name in [(0, 0, 'Dom0→Dom0'), (0, 1, 'Dom0→Dom1'), (0, 2, 'Dom0→Dom2')]:
            src_l, mod_dims, _ = get_synthetic_loaders(
                'forecasting', args.batch_size, args.num_workers, 2, 4, 96, pred_len=pred_len, domain=src_d, seed=42)
            tgt_l, _, _ = get_synthetic_loaders(
                'forecasting', args.batch_size, args.num_workers, 2, 4, 96, pred_len=pred_len, domain=tgt_d, seed=99)
            scenarios[name] = (src_l, tgt_l)

    cross_results = {}
    models_to_test = ['TMTFNet', 'Transformer', 'LSTM']

    for scenario_name, (src_loaders, tgt_loaders) in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        cross_results[scenario_name] = {}

        for model_name in models_to_test:
            kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                     'n_enc_layers' if 'TMTFNet' in model_name else 'n_layers': args.n_layers,
                     'pred_len': pred_len, 'dropout': args.dropout}
            if model_name == 'LSTM':
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
                         'pred_len': pred_len, 'dropout': args.dropout}

            torch.manual_seed(args.seed)
            model = build_model(model_name, **kwargs)
            from src.trainer import Trainer
            trainer = Trainer(model, task='forecasting', device=device, lr=args.lr)
            trainer.train(src_loaders['train'], src_loaders['val'],
                        max_epochs=args.max_epochs, patience=args.patience, verbose=False)

            test_metrics = trainer.evaluate(tgt_loaders['test'])
            cross_results[scenario_name][model_name] = float(test_metrics['mse'])
            print(f"  {model_name}: MSE = {test_metrics['mse']:.4f}")

    save_dir = os.path.join(args.result_dir, 'exp6_cross_forecast')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(cross_results, f, indent=2)

    fig_dir = os.path.join(args.fig_dir, 'exp6')
    os.makedirs(fig_dir, exist_ok=True)
    plot_cross_domain_results(
        cross_results,
        os.path.join(fig_dir, 'cross_domain_forecast.png'),
        metric_name='MSE',
        lower_is_better=True
    )

    return cross_results


# ============================================================
# Multiple Seeds for Statistical Significance
# ============================================================

def run_multi_seed(args, device):
    """Run key experiments with multiple seeds for statistical significance."""
    print_banner("Statistical Significance Testing (Multi-Seed)")

    seeds = [42, 123, 456, 789, 1024]
    model_names = ['TMTFNet', 'LSTM', 'GRU', 'TCN', 'Transformer', 'Crossformer']

    loaders, mod_dims, n_classes = get_synthetic_loaders(
        task='classification', batch_size=args.batch_size, num_workers=args.num_workers,
        n_modalities=3, dims_per_mod=3, seq_len=64, n_classes=6, seed=42
    )

    all_accs = {m: [] for m in model_names}

    for seed in seeds:
        print(f"\n--- Seed: {seed} ---")
        for model_name in model_names:
            if model_name in ('TMTFNet', 'TMTFNet_NoCMTA', 'TMTFNet_NoAMG', 'TMTFNet_NoHTF'):
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                         'n_enc_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout}
            elif model_name in ('LSTM', 'GRU'):
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': args.n_layers,
                         'n_classes': n_classes, 'dropout': args.dropout}
            elif model_name == 'TCN':
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_layers': 4,
                         'n_classes': n_classes, 'dropout': args.dropout}
            else:
                kwargs = {'modality_dims': mod_dims, 'd_model': args.d_model, 'n_heads': args.n_heads,
                         'n_layers': args.n_layers, 'n_classes': n_classes, 'dropout': args.dropout}

            result = run_single_experiment(
                model_name, kwargs, loaders, task='classification', device=device,
                max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
                seed=seed, verbose=False
            )
            all_accs[model_name].append(result['test_accuracy'])
            print(f"  {model_name}: Acc={result['test_accuracy']:.4f}")

    # Print with mean ± std
    save_dir = os.path.join(args.result_dir, 'multi_seed')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Mean Acc':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*60}")
    stats = {}
    for m in model_names:
        accs = np.array(all_accs[m])
        mean, std = accs.mean(), accs.std()
        stats[m] = {'mean': float(mean), 'std': float(std),
                    'min': float(accs.min()), 'max': float(accs.max()),
                    'all': [float(a) for a in accs]}
        print(f"{m:<20} {mean:>10.4f} {std:>10.4f} {accs.min():>10.4f} {accs.max():>10.4f}")
    print(f"{'='*60}")

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TMTFNet Experiments')
    # Data
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--use_real_data', action='store_true', help='Use real datasets if available')
    parser.add_argument('--result_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--fig_dir', type=str, default='./figures', help='Figures directory')
    # Model
    parser.add_argument('--d_model', type=int, default=64, help='Model hidden dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    # Experiment selection
    parser.add_argument('--exp', type=str, default='all',
                       choices=['all', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'multi_seed'],
                       help='Which experiment to run')

    args = parser.parse_args()

    # Setup
    device = setup_device()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    total_start = time.time()

    # Run experiments
    if args.exp in ('all', 'exp1'):
        run_exp1_classification(args, device)

    if args.exp in ('all', 'exp2'):
        run_exp2_forecasting(args, device)

    if args.exp in ('all', 'exp3'):
        run_exp3_ablation(args, device)

    if args.exp in ('all', 'exp4'):
        run_exp4_cross_domain(args, device)

    if args.exp in ('all', 'exp5'):
        run_exp5_scalability(args, device)

    if args.exp in ('all', 'exp6'):
        run_exp6_cross_domain_forecast(args, device)

    if args.exp in ('all', 'multi_seed'):
        run_multi_seed(args, device)

    total_time = time.time() - total_start
    print(f"\n{'#'*70}")
    print(f"# ALL EXPERIMENTS COMPLETED in {total_time/60:.1f} minutes")
    print(f"# Results: {args.result_dir}")
    print(f"# Figures: {args.fig_dir}")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
