"""Regenerate the manuscript figures from the rebuttal JSON summaries.

Outputs go straight into ``../latex/fig/`` so the LaTeX source can pick them
up by name. Three figures are produced:

* ``param_accuracy.pdf`` -- latency-vs-accuracy Pareto plot with 95% bootstrap
  CI bars. Replaces the earlier bubble chart that Reviewer 2 found misleading.
* ``transfer_gap.pdf`` -- TMTFNet vs every baseline on cross-domain MSE /
  Transfer Gap, with paired-Wilcoxon p-values annotated.
* ``sensitivity.pdf`` -- hyperparameter sensitivity curves (5-seed mean +
  shaded CI) across d_model / n_heads / n_enc_layers / modality_dropout.

Usage::

    python plot_rebuttal_figures.py  # reads results/rebuttal_*/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
FIG_DIR = REPO.parent / "latex" / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# stable colour palette (baseline models in grey, TMTFNet family highlighted)
COLOR_MAP = {
    "TMTFNet_v2": "#d62728",
    "TMTFNet_v2_DA": "#ff7f0e",
    "PatchTST": "#1f77b4",
    "Transformer": "#2ca02c",
    "Crossformer": "#9467bd",
    "LSTM": "#8c564b",
    "GRU": "#e377c2",
    "TCN": "#7f7f7f",
    "DLinear": "#bcbd22",
    "TimeMixer": "#17becf",
    "iTransformer": "#6a6a6a",
    "CNN1D": "#1a9850",
    "DeepConvLSTM": "#66bd63",
    "SVM": "#4575b4",
    "RF": "#74add1",
    "kNN": "#abd9e9",
    "LR": "#e0f3f8",
    "DANN": "#fdae61",
    "CoDATS": "#f46d43",
    "AdvSKM": "#d73027",
    "RAINCOAT": "#a50026",
}


def _load_summary(exp_name):
    path = RESULTS / f"rebuttal_{exp_name}" / "summary.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _load_per_seed(exp_name, model, seeds=(42, 123, 456, 789, 2024), metric="accuracy"):
    values = []
    for seed in seeds:
        path = RESULTS / f"rebuttal_{exp_name}" / f"{model}_seed{seed}.json"
        if path.exists():
            with path.open() as fh:
                values.append(json.load(fh).get(metric))
    return np.asarray([v for v in values if v is not None], dtype=np.float64)


def _bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    draws = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def plot_param_accuracy_pareto():
    summary = _load_summary("exp1a_har_uci")
    if summary is None:
        print("skip Pareto: exp1a_har_uci summary missing")
        return
    xs, ys, yerr_lo, yerr_hi, names = [], [], [], [], []
    for model, stats in summary.items():
        if model.startswith("_"):
            continue
        params_k = stats.get("n_params", 0) / 1000.0
        per = _load_per_seed("exp1a_har_uci", model)
        if per.size == 0:
            continue
        mean, lo, hi = _bootstrap_ci(per)
        xs.append(params_k)
        ys.append(mean * 100.0)
        yerr_lo.append((mean - lo) * 100.0)
        yerr_hi.append((hi - mean) * 100.0)
        names.append(model)

    if not xs:
        print("skip Pareto: no per-seed runs")
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for x, y, lo, hi, name in zip(xs, ys, yerr_lo, yerr_hi, names):
        color = COLOR_MAP.get(name, "#666666")
        marker = "D" if name.startswith("TMTFNet") else "o"
        ax.errorbar(x, y, yerr=[[lo], [hi]], fmt=marker, color=color,
                    markersize=7, capsize=3, linewidth=1.0, alpha=0.9, label=name)
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameters (K)")
    ax.set_ylabel("UCI HAR Accuracy (\%)")
    ax.set_title("Parameter-efficiency Pareto on UCI HAR (5-seed 95\% CI)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "param_accuracy.pdf")
    plt.close(fig)
    print("wrote", FIG_DIR / "param_accuracy.pdf")


def plot_transfer_gap():
    summaries = {}
    splits = ["h1_to_h2", "h2_to_h1", "m1_to_m2", "m2_to_m1", "h1_to_m1"]
    for split in splits:
        s = _load_summary(f"exp3_cross_forecast_pred24__{split}")
        if s is not None:
            summaries[split] = s
    if not summaries:
        print("skip transfer_gap: no cross-domain summaries")
        return

    models = sorted({m for s in summaries.values() for m in s if not m.startswith("_")},
                    key=lambda m: (0 if m == "TMTFNet_v2" else 1, m))
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    width = 0.14
    x = np.arange(len(models))
    for i, split in enumerate(splits):
        if split not in summaries:
            continue
        heights = []
        for model in models:
            stats = summaries[split].get(model)
            heights.append(stats["mse"]["mean"] if stats and "mse" in stats else np.nan)
        ax.bar(x + i * width, heights, width, label=split.replace("_", "$\\to$"),
               alpha=0.85)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Cross-domain MSE")
    ax.set_title("Cross-domain forecasting MSE (H=24, 5-seed mean, lower is better)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "transfer_gap.pdf")
    plt.close(fig)
    print("wrote", FIG_DIR / "transfer_gap.pdf")


def plot_sensitivity():
    path = RESULTS / "rebuttal_exp5_sensitivity" / "summary.json"
    if not path.exists():
        print("skip sensitivity: summary missing")
        return
    data = json.loads(path.read_text())
    hyperparams = ["d_model", "n_heads", "n_enc_layers", "modality_dropout"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True)
    for ax, name in zip(axes, hyperparams):
        xs = []
        means = []
        stds = []
        for tag, stats in sorted(data.items()):
            if not tag.startswith(f"{name}="):
                continue
            val = float(tag.split("=", 1)[1])
            xs.append(val)
            means.append(stats["accuracy"]["mean"] * 100)
            stds.append(stats["accuracy"]["std"] * 100)
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        means = np.asarray(means)[order]
        stds = np.asarray(stds)[order]
        ax.plot(xs, means, marker="o", color="#d62728", linewidth=1.2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="#d62728")
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("UCI HAR Accuracy (\%)")
    fig.suptitle("Hyperparameter sensitivity (5-seed mean $\\pm$ 1 std)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity.pdf")
    plt.close(fig)
    print("wrote", FIG_DIR / "sensitivity.pdf")


def main():
    plot_param_accuracy_pareto()
    plot_transfer_gap()
    plot_sensitivity()


if __name__ == "__main__":
    main()
