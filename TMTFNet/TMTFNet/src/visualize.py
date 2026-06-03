"""
Visualization utilities for TMTFNet paper figures.
===================================================
Generates publication-quality figures for:
1. Training curves (loss, accuracy)
2. t-SNE embeddings
3. Attention heatmaps
4. Confusion matrices
5. Performance comparison bar charts
6. Ablation study charts
7. Gate weight distributions
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800',
          '#00BCD4', '#E91E63', '#795548', '#607D8B', '#CDDC39']
MODEL_COLORS = {
    'TMTFNet': '#2196F3', 'LSTM': '#FF5722', 'GRU': '#4CAF50',
    'TCN': '#9C27B0', 'Transformer': '#FF9800', 'Crossformer': '#00BCD4',
    'TMTFNet_NoCMTA': '#E91E63', 'TMTFNet_NoAMG': '#795548', 'TMTFNet_NoHTF': '#607D8B'
}


def plot_training_curves(all_histories, save_path, task='classification'):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, hist in all_histories.items():
        color = MODEL_COLORS.get(name, '#333333')
        axes[0].plot(hist['train_loss'], label=f'{name} (train)', color=color, linestyle='-', alpha=0.8)
        axes[0].plot(hist['val_loss'], label=f'{name} (val)', color=color, linestyle='--', alpha=0.6)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(fontsize=7, ncol=2, loc='upper right')

    if task == 'classification':
        for name, hist in all_histories.items():
            if 'val_acc' in hist:
                color = MODEL_COLORS.get(name, '#333333')
                axes[1].plot(hist['train_acc'], label=f'{name} (train)', color=color, linestyle='-', alpha=0.8)
                axes[1].plot(hist['val_acc'], label=f'{name} (val)', color=color, linestyle='--', alpha=0.6)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend(fontsize=7, ncol=2, loc='lower right')
    else:
        # For forecasting, plot val MSE
        for name, hist in all_histories.items():
            color = MODEL_COLORS.get(name, '#333333')
            axes[1].plot(hist['val_loss'], label=name, color=color)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation MSE')
        axes[1].set_title('Validation Loss Convergence')
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne(representations_dict, labels_dict, save_path, class_names=None):
    """Plot t-SNE visualization of learned representations."""
    n_models = len(representations_dict)
    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(4.5 * min(n_models, 4), 4))
    if n_models == 1:
        axes = [axes]

    for idx, (name, reps) in enumerate(representations_dict.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        labels = labels_dict[name]

        # Subsample if too many points
        if len(reps) > 2000:
            indices = np.random.choice(len(reps), 2000, replace=False)
            reps = reps[indices]
            labels = labels[indices]

        perplexity = min(30, max(5, len(reps) - 1))
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        except TypeError:
            # sklearn>=1.5 uses max_iter instead of n_iter.
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        embedded = tsne.fit_transform(reps)

        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name_label = class_names[label] if class_names else f'Class {label}'
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=COLORS[i % len(COLORS)], s=8, alpha=0.6, label=name_label)

        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=7, markerscale=2, loc='best')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, save_path, class_names=None, title=''):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if class_names is None:
        class_names = [f'C{i}' for i in range(len(cm))]

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix{" - " + title if title else ""}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_performance_comparison(results, save_path, task='classification'):
    """Bar chart comparing model performance."""
    if task == 'classification':
        metrics = ['test_accuracy', 'test_f1_macro', 'test_precision', 'test_recall']
        metric_labels = ['Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
    else:
        metrics = ['test_mse', 'test_mae', 'test_rmse']
        metric_labels = ['MSE', 'MAE', 'RMSE']

    model_names = [r['model_name'] for r in results]
    n_metrics = len(metrics)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 5))
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [r.get(metric, 0) for r in results]
        bars = ax.bar(x + i * width - 0.4 + width/2, values, width,
                     label=label, color=COLORS[i], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance Comparison ({task.title()})')
    ax.legend(loc='upper right')

    if task == 'classification':
        ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_study(results, save_path, task='classification'):
    """Ablation study visualization."""
    if task == 'classification':
        metric = 'test_accuracy'
        ylabel = 'Accuracy'
    else:
        metric = 'test_mse'
        ylabel = 'MSE'

    model_names = [r['model_name'] for r in results]
    values = [r.get(metric, 0) for r in results]

    # Highlight full model
    colors = []
    for name in model_names:
        if name == 'TMTFNet':
            colors.append('#2196F3')
        else:
            colors.append('#90CAF9')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(model_names)), values, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', ha='left', va='center', fontsize=10)

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(ylabel)
    ax.set_title('Ablation Study')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cross_domain_results(results_dict, save_path, metric_name='MSE', lower_is_better=True):
    """Visualize cross-domain transfer results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = list(results_dict.keys())
    model_names = list(results_dict[scenarios[0]].keys())

    x = np.arange(len(scenarios))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        values = [results_dict[s][model] for s in scenarios]
        color = MODEL_COLORS.get(model, COLORS[i])
        bars = ax.bar(x + i * width - 0.4 + width/2, values, width,
                     label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    direction = 'lower is better' if lower_is_better else 'higher is better'
    ax.set_ylabel(f'{metric_name} ({direction})')
    ax.set_title('Cross-Domain Transfer Performance')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_parameter_efficiency(results, save_path, task='classification'):
    """Scatter plot: params vs performance."""
    metric = 'test_accuracy' if task == 'classification' else 'test_mse'
    ylabel = 'Accuracy' if task == 'classification' else 'MSE'

    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        name = r['model_name']
        color = MODEL_COLORS.get(name, '#333333')
        ax.scatter(r['n_params'] / 1000, r[metric], s=120, c=color,
                  edgecolors='white', linewidth=1.5, zorder=3)
        ax.annotate(name, (r['n_params'] / 1000, r[metric]),
                   textcoords="offset points", xytext=(5, 8), fontsize=8)

    ax.set_xlabel('Parameters (K)')
    ax.set_ylabel(ylabel)
    ax.set_title('Parameter Efficiency')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_figures(all_results, fig_dir, task='classification', class_names=None):
    """Generate all paper figures from experiment results."""
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Training curves
    histories = {r['model_name']: r['history'] for r in all_results if 'history' in r}
    if histories:
        plot_training_curves(histories, os.path.join(fig_dir, 'training_curves.png'), task)

    # 2. Performance comparison
    plot_performance_comparison(all_results, os.path.join(fig_dir, 'performance_comparison.png'), task)

    # 3. t-SNE (for classification)
    if task == 'classification':
        reps_dict, labels_dict = {}, {}
        for r in all_results:
            if '_representations' in r and r['_representations'] is not None:
                reps_dict[r['model_name']] = r['_representations']
                labels_dict[r['model_name']] = r['_labels']
        if reps_dict:
            plot_tsne(reps_dict, labels_dict, os.path.join(fig_dir, 'tsne.png'), class_names)

    # 4. Confusion matrix (TMTFNet only)
    for r in all_results:
        if r['model_name'] == 'TMTFNet' and 'confusion_matrix' in r:
            cm = np.array(r['confusion_matrix'])
            if cm.size > 0:
                plot_confusion_matrix(cm, os.path.join(fig_dir, 'confusion_matrix.png'),
                                    class_names, 'TMTFNet')

    # 5. Ablation study
    ablation_models = ['TMTFNet', 'TMTFNet_NoCMTA', 'TMTFNet_NoAMG', 'TMTFNet_NoHTF']
    ablation_results = [r for r in all_results if r['model_name'] in ablation_models]
    if len(ablation_results) > 1:
        plot_ablation_study(ablation_results, os.path.join(fig_dir, 'ablation.png'), task)

    # 6. Parameter efficiency
    plot_parameter_efficiency(all_results, os.path.join(fig_dir, 'param_efficiency.png'), task)

    print(f"\nAll figures saved to: {fig_dir}")
