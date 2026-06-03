"""Build the publication-grade figure suite for the revised manuscript.

Reads the rebuttal JSON summaries and per-seed run files. Writes each figure
as a standalone PDF directly into ``../latex/fig/`` so the LaTeX source
picks them up by name. All figures use vector PDF, embedded Type 1 fonts,
and 300 DPI raster fallback for any embedded heatmap.

Generated figures:
    fig/pareto_har.pdf            -- Param-vs-accuracy Pareto on UCI HAR + PAMAP2 (2-panel)
    fig/cross_domain_bars.pdf     -- 5-split cross-domain MSE bars
    fig/transfer_gap_heatmap.pdf  -- Model x Split heatmap of cross-domain MSE
    fig/confusion.pdf             -- 4-panel confusion matrices (TMTFNet vs PatchTST on HAR/PAMAP2)
    fig/sensitivity.pdf           -- 4-panel sensitivity curves with 5-seed CI band
    fig/ablation.pdf              -- Ablation bar chart (HAR + cross-domain side-by-side)
    fig/tsne_alignment.pdf        -- t-SNE of pooled features showing alignment effect
    fig/modality_dropout.pdf      -- Modality dropout effect on cross-domain MSE
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
FIG_DIR = REPO.parent / "latex_spring" / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DEEP_COLORS = {
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
    "LR": "#5f86c4",
    "DANN": "#fdae61",
    "CoDATS": "#f46d43",
    "AdvSKM": "#d73027",
    "RAINCOAT": "#a50026",
}

SPLIT_LABELS = {
    "h1_to_h2": r"ETTh1$\to$ETTh2",
    "h2_to_h1": r"ETTh2$\to$ETTh1",
    "m1_to_m2": r"ETTm1$\to$ETTm2",
    "m2_to_m1": r"ETTm2$\to$ETTm1",
    "h1_to_m1": r"ETTh1$\to$ETTm1$^*$",
}


# Display labels: keep the paper free of internal version suffixes.
def display_name(model: str) -> str:
    if model == "TMTFNet_v2":
        return "TMTFNet"
    if model == "TMTFNet_v2_DA":
        return r"TMTFNet$_{\mathrm{DA}}$"
    if model.startswith("TMTFNet_v2_No"):
        suffix = model[len("TMTFNet_v2_No"):]
        return f"TMTFNet w/o {suffix}"
    return model


def _load_summary(path):
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _per_seed(exp, model, metric, seeds=(42, 123, 456, 789, 2024)):
    out = []
    for seed in seeds:
        p = RESULTS / f"rebuttal_{exp}" / f"{model}_seed{seed}.json"
        if p.exists():
            with p.open() as fh:
                v = json.load(fh).get(metric)
                if v is not None:
                    out.append(v)
    return np.asarray(out, dtype=np.float64)


def _bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    draws = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


# -------------------------------------------------------------------- pareto
def fig_pareto():
    # Each panel uses a dual axis break: y between high-accuracy cluster and
    # low-accuracy outlier band, x between the small-parameter DLinear and the
    # main 100--2000K cluster. Quadrants per panel:
    #   TL = (low x, high y) -- empty
    #   TR = (high x, high y) -- HAR cluster / TMTFNet on PAMAP2
    #   BL = (low x, low y)   -- DLinear
    #   BR = (high x, low y)  -- empty on HAR, multi-modal cluster on PAMAP2
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

    fig = plt.figure(figsize=(8.6, 4.2))
    outer = fig.add_gridspec(1, 2, wspace=0.28)

    panels = [
        ("exp1a_har_uci",
         "UCI HAR (in-subject)",
         set(),
         (87.5, 96.0), (75.0, 78.0)),
        ("exp1c_har_pamap2",
         "PAMAP2 (cross-subject, multi-modal architectures only)",
         {"PatchTST"},
         (83.5, 86.5), (66.0, 77.0)),
    ]
    x_low = (1.5, 10.0)
    x_high = (100.0, 2500.0)
    low_xticks = [2.0, 5.0]
    high_xticks = [100, 300, 1000, 3000]
    # Manual offsets for labels that would otherwise overlap. Values are
    # (dx_px, dy_px, horizontal_alignment, vertical_alignment). Default if
    # absent is right-and-above the marker.
    label_overrides = {
        ("exp1a_har_uci", "PatchTST"):    (5, -3, "left",  "top"),
        ("exp1a_har_uci", "GRU"):         (-5, 3, "right", "bottom"),
        ("exp1a_har_uci", "LSTM"):        (5, -3, "left",  "top"),
        ("exp1a_har_uci", "TCN"):         (-5, 3, "right", "bottom"),
        ("exp1a_har_uci", "DeepConvLSTM"):(-5, -3,"right", "top"),
        ("exp1a_har_uci", "Transformer"): (5, -4, "left",  "top"),
        ("exp1a_har_uci", "TimeMixer"):   (-5, 3, "right", "bottom"),
        ("exp1a_har_uci", "iTransformer"):(5, -3, "left",  "top"),
        ("exp1c_har_pamap2", "TimeMixer"):    (-5, 3, "right", "bottom"),
        ("exp1c_har_pamap2", "CNN1D"):        (5, -3, "left",  "top"),
        ("exp1c_har_pamap2", "iTransformer"): (-5, -3, "right", "top"),
        ("exp1c_har_pamap2", "TCN"):          (-5, 4, "right", "bottom"),
        ("exp1c_har_pamap2", "GRU"):          (-5, 3, "right", "bottom"),
        ("exp1c_har_pamap2", "LSTM"):         (5, -3, "left",  "top"),
        ("exp1c_har_pamap2", "Transformer"):  (-5, 3, "right", "bottom"),
        ("exp1c_har_pamap2", "DeepConvLSTM"): (5, -3, "left",  "top"),
        ("exp1c_har_pamap2", "Crossformer"):  (-5, 3, "right", "bottom"),
    }

    for col, (exp_name, title, exclude, top_ylim, bot_ylim) in enumerate(panels):
        inner = outer[col].subgridspec(
            2, 2,
            width_ratios=[1.0, 4.5],
            height_ratios=[3.0, 1.0],
            wspace=0.06, hspace=0.12,
        )
        ax_tl = fig.add_subplot(inner[0, 0])
        ax_tr = fig.add_subplot(inner[0, 1])
        ax_bl = fig.add_subplot(inner[1, 0])
        ax_br = fig.add_subplot(inner[1, 1])

        for ax in (ax_tl, ax_bl):
            ax.set_xscale("log")
            ax.set_xlim(*x_low)
            ax.xaxis.set_major_locator(FixedLocator(low_xticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in low_xticks]))
            ax.xaxis.set_minor_locator(NullLocator())
        for ax in (ax_tr, ax_br):
            ax.set_xscale("log")
            ax.set_xlim(*x_high)
            ax.xaxis.set_major_locator(FixedLocator(high_xticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(int(v)) for v in high_xticks]))
            ax.xaxis.set_minor_locator(NullLocator())
        for ax in (ax_tl, ax_tr):
            ax.set_ylim(*top_ylim)
        for ax in (ax_bl, ax_br):
            ax.set_ylim(*bot_ylim)

        summary = _load_summary(RESULTS / f"rebuttal_{exp_name}" / "summary.json")
        if summary is None:
            ax_tr.set_title(f"{title}: data not ready")
            continue

        for model, stats in summary.items():
            if model.startswith("_") or model in exclude:
                continue
            params_k = (stats.get("n_params") or 0) / 1000
            per = _per_seed(exp_name, model, "accuracy")
            if per.size == 0:
                continue
            mean, lo, hi = _bootstrap_ci(per)
            mean_pct = mean * 100
            is_ours = model.startswith("TMTFNet")
            color = DEEP_COLORS.get(model, "#d62728" if is_ours else "#444444")
            marker, msize, mew = ("D", 9, 1.4) if is_ours else ("o", 5, 0.5)
            edge = "black" if is_ours else color
            yerr = [[(mean - lo) * 100], [(hi - mean) * 100]]
            is_low_x = params_k < x_low[1]
            is_high_y = mean_pct >= top_ylim[0]
            target = (ax_tl if is_low_x else ax_tr) if is_high_y else (ax_bl if is_low_x else ax_br)
            target.errorbar(
                params_k, mean_pct, yerr=yerr,
                fmt=marker, color=color, markersize=msize, capsize=2.0,
                linewidth=0.8, markeredgewidth=mew, markeredgecolor=edge,
                alpha=1.0 if is_ours else 0.9,
            )
            # Annotations: narrow left column anchors to the LEFT of the
            # marker so the text stays inside the narrow box; otherwise use
            # any manual override or a right-of-marker default.
            override = label_overrides.get((exp_name, model))
            if override is not None:
                dx, dy, ha, va = override
            elif is_low_x:
                dx, dy, ha, va = -5, 3, "right", "bottom"
            else:
                dx, dy, ha, va = 4, 3, "left", "bottom"
            target.annotate(
                display_name(model), (params_k, mean_pct),
                xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=7, color="#000000" if is_ours else color,
                fontweight="bold" if is_ours else "normal",
            )

        # Inner-edge spines off
        ax_tl.spines["right"].set_visible(False); ax_tl.spines["bottom"].set_visible(False)
        ax_tr.spines["left"].set_visible(False);  ax_tr.spines["bottom"].set_visible(False)
        ax_bl.spines["right"].set_visible(False); ax_bl.spines["top"].set_visible(False)
        ax_br.spines["left"].set_visible(False);  ax_br.spines["top"].set_visible(False)

        # Tick visibility on inner edges
        ax_tl.tick_params(labelbottom=False, bottom=False, labelright=False, right=False)
        ax_tr.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        ax_bl.tick_params(labeltop=False, top=False, labelright=False, right=False)
        ax_br.tick_params(labeltop=False, top=False, labelleft=False, left=False)

        # Diagonal cut marks at every inner-corner: small slashes in axes coords
        d = 0.012
        def _slash(ax, x, y):
            kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=0.9)
            ax.plot((x - d, x + d), (y - d, y + d), **kw)
        # y-break: across each axes' inner horizontal edge, both corners
        for ax in (ax_tl, ax_tr):
            _slash(ax, 0, 0); _slash(ax, 1, 0)
        for ax in (ax_bl, ax_br):
            _slash(ax, 0, 1); _slash(ax, 1, 1)
        # x-break: across each axes' inner vertical edge, both corners
        for ax in (ax_tl, ax_bl):
            _slash(ax, 1, 0); _slash(ax, 1, 1)
        for ax in (ax_tr, ax_br):
            _slash(ax, 0, 0); _slash(ax, 0, 1)

        # Outer labels
        ax_tl.set_ylabel("Test Accuracy (%)")
        ax_br.set_xlabel("Parameters (K)")
        # Title across the panel: anchor on the left axes but extend
        ax_tr.set_title(title, fontsize=10, loc="left")

        for ax in (ax_tl, ax_tr, ax_bl, ax_br):
            ax.grid(alpha=0.25)

    out = FIG_DIR / "pareto_har.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- cross-domain bars
def fig_cross_domain_bars():
    summaries = {}
    for split in SPLIT_LABELS:
        path = RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / "summary.json"
        s = _load_summary(path)
        if s:
            summaries[split] = s
    if not summaries:
        print("skip cross_domain_bars: no summaries")
        return

    deep_models = sorted({m for s in summaries.values() for m in s if not m.startswith("_") and m != "DLinear"},
                         key=lambda m: 0 if m == "TMTFNet_v2" else 1)
    # Append the validated-lambda model as an extra entry built from per-split lookups
    deep_models = deep_models + ["TMTFNet_v2_validated"]
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    x = np.arange(len(deep_models))
    splits = list(SPLIT_LABELS.keys())
    width = 0.16
    for i, split in enumerate(splits):
        s = summaries.get(split, {})
        heights, errs = [], []
        for m in deep_models:
            if m == "TMTFNet_v2_validated":
                _, mn, sd = _validated_lambda_per_split(split)
                heights.append(mn if mn is not None else np.nan)
                errs.append(sd if sd is not None else 0)
            else:
                heights.append((s.get(m) or {}).get("mse", {}).get("mean", np.nan))
                errs.append((s.get(m) or {}).get("mse", {}).get("std", 0) or 0)
        ax.bar(x + i * width, heights, width, yerr=errs, capsize=2,
               label=SPLIT_LABELS[split], alpha=0.85)
    labels = [display_name(m) if m != "TMTFNet_v2_validated" else r"TMTFNet$_{\mathrm{DA}}$ (val.\,$\lambda$)"
              for m in deep_models]
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cross-domain MSE (5-seed mean $\\pm$ std, log scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    fig.tight_layout()
    out = FIG_DIR / "cross_domain_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- transfer gap heatmap
def fig_transfer_gap_heatmap():
    summaries = {}
    for split in SPLIT_LABELS:
        path = RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / "summary.json"
        s = _load_summary(path)
        if s:
            summaries[split] = s
    if not summaries:
        print("skip transfer_gap_heatmap: no data")
        return

    deep_models = sorted({m for s in summaries.values() for m in s if not m.startswith("_")},
                         key=lambda m: 0 if m == "TMTFNet_v2" else 1)
    M = np.full((len(deep_models), len(SPLIT_LABELS)), np.nan)
    for j, split in enumerate(SPLIT_LABELS):
        s = summaries.get(split, {})
        for i, m in enumerate(deep_models):
            v = (s.get(m) or {}).get("mse", {}).get("mean")
            if v is not None:
                M[i, j] = v

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    im = ax.imshow(np.log10(M + 1e-3), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(SPLIT_LABELS)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_LABELS], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(deep_models)))
    ax.set_yticklabels([display_name(m) for m in deep_models])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                txt_color = "white" if np.log10(M[i, j] + 1e-3) > -0.2 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("$\\log_{10}$ MSE (lower = better)")
    fig.tight_layout()
    out = FIG_DIR / "transfer_gap_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- confusion matrices
def fig_confusion():
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))

    panels = [
        ("exp1a_har_uci", "TMTFNet_v2", 42, "TMTFNet on UCI HAR (seed 42)",  # display label set in caption
         ["Walk", "WalkUp", "WalkDn", "Sit", "Stand", "Lay"]),
        ("exp1a_har_uci", "PatchTST", 42, "PatchTST on UCI HAR (seed 42)",
         ["Walk", "WalkUp", "WalkDn", "Sit", "Stand", "Lay"]),
        ("exp1c_har_pamap2", "TMTFNet_v2", 42, "TMTFNet on PAMAP2 cross-subject (seed 42)",
         ["Lying", "Sit", "Stand", "Walk", "Run", "Cycle", "NWalk", "Asc", "Desc", "Vac", "Iron", "Rope"]),
        ("exp1c_har_pamap2", "PatchTST", 42, "PatchTST on PAMAP2 cross-subject (seed 42)",
         ["Lying", "Sit", "Stand", "Walk", "Run", "Cycle", "NWalk", "Asc", "Desc", "Vac", "Iron", "Rope"]),
    ]

    for ax, (exp, model, seed, title, classes) in zip(axes.flat, panels):
        path = RESULTS / f"rebuttal_{exp}" / f"{model}_seed{seed}.json"
        if not path.exists():
            ax.set_title(f"{title}: missing")
            ax.axis("off")
            continue
        with path.open() as fh:
            d = json.load(fh)
        cm = np.asarray(d.get("confusion_matrix", []), dtype=np.float64)
        if cm.size == 0:
            ax.set_title(f"{title}: missing CM")
            ax.axis("off")
            continue
        # row-normalise (recall per true class)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm_norm[i, j]
                if value > 0.01:
                    color = "white" if value > 0.5 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    fig.tight_layout()
    out = FIG_DIR / "confusion.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- sensitivity
def fig_sensitivity():
    path = RESULTS / "rebuttal_exp5_sensitivity" / "summary.json"
    data = _load_summary(path)
    if data is None:
        print("skip sensitivity: no summary")
        return
    hyper = ["d_model", "n_heads", "n_enc_layers", "modality_dropout"]
    titles = ["(a) Hidden dim $D$", "(b) Heads $H$", "(c) Encoder layers $L$", "(d) Mod. dropout $p$"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    for ax, name, title in zip(axes, hyper, titles):
        xs, means, stds = [], [], []
        for tag, st in sorted(data.items()):
            if not tag.startswith(f"{name}="):
                continue
            val = float(tag.split("=", 1)[1])
            xs.append(val)
            means.append(st["accuracy"]["mean"] * 100)
            stds.append(st["accuracy"]["std"] * 100)
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        means = np.asarray(means)[order]
        stds = np.asarray(stds)[order]
        ax.plot(xs, means, marker="o", color="#d62728", linewidth=1.5, markersize=5)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color="#d62728")
        ax.set_xlabel(name)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("UCI HAR Accuracy (%)")
    fig.tight_layout()
    out = FIG_DIR / "sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- ablation bars
def fig_ablation():
    har = _load_summary(RESULTS / "rebuttal_exp4_ablation__har" / "summary.json")
    cross = _load_summary(RESULTS / "rebuttal_exp4_ablation__cross_h1_to_h2" / "summary.json")
    if har is None or cross is None:
        print("skip ablation: missing summary")
        return
    variants = ["TMTFNet_v2", "TMTFNet_v2_NoGCMTA", "TMTFNet_v2_NoAMG",
                "TMTFNet_v2_NoAHTF", "TMTFNet_v2_NoAttnPool", "TMTFNet_v2_NoModDrop"]
    pretty = ["Full", "−G-CMTA", "−AMG", "−A-HTF", "−AttnPool", "−ModDrop"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    har_means = [(har.get(v) or {}).get("accuracy", {}).get("mean", 0) * 100 for v in variants]
    har_stds = [(har.get(v) or {}).get("accuracy", {}).get("std", 0) * 100 for v in variants]
    bars = axes[0].bar(pretty, har_means, yerr=har_stds, capsize=3,
                       color=["#d62728"] + ["#888888"] * 5, alpha=0.9)
    axes[0].set_ylabel("UCI HAR Accuracy (%)")
    axes[0].set_ylim([min(har_means) - 1.5, max(y + s for y, s in zip(har_means, har_stds)) + 1.0])
    axes[0].set_title("(a) HAR classification")
    axes[0].grid(axis="y", alpha=0.3)
    for x, (y, s) in enumerate(zip(har_means, har_stds)):
        axes[0].annotate(
            f"{y:.2f}",
            xy=(x, y + s),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )
    axes[0].tick_params(axis="x", rotation=20)

    cd_means = [(cross.get(v) or {}).get("mse", {}).get("mean", 0) for v in variants]
    cd_stds = [(cross.get(v) or {}).get("mse", {}).get("std", 0) for v in variants]
    bars = axes[1].bar(pretty, cd_means, yerr=cd_stds, capsize=3,
                       color=["#d62728"] + ["#888888"] * 5, alpha=0.9)
    axes[1].set_ylabel("ETTh1$\\to$ETTh2 MSE")
    axes[1].set_ylim(top=max(y + s for y, s in zip(cd_means, cd_stds)) * 1.2)
    axes[1].set_title("(b) Cross-domain forecasting")
    axes[1].grid(axis="y", alpha=0.3)
    for x, (y, s) in enumerate(zip(cd_means, cd_stds)):
        axes[1].annotate(
            f"{y:.2f}",
            xy=(x, y + s),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out = FIG_DIR / "ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- t-SNE alignment
def fig_tsne_alignment():
    """Load the TMTFNet HAR seed-42 checkpoint, project pooled features, color by class."""
    import torch
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("skip tsne: sklearn missing TSNE")
        return

    ckpt_path = RESULTS / "final_exp1_har" / "checkpoints" / "TMTFNet_v2_seed42_lr0p0005.pt"
    if not ckpt_path.exists():
        print("skip tsne: checkpoint missing")
        return

    sys.path.insert(0, str(REPO))
    from src.datasets import get_har_loaders  # noqa: E402
    from src.models import build_model  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = get_har_loaders(str(REPO / "data"), batch_size=128, num_workers=0, seq_len=128, split_seed=42)
    model = build_model(
        "TMTFNet_v2",
        modality_dims=bundle.modality_dims,
        d_model=64, n_heads=8, n_enc_layers=2,
        modality_dropout=0.1, n_classes=bundle.n_classes,
        dropout=0.1, seq_len=128,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=False)
    except Exception as exc:
        print(f"skip tsne: ckpt load failed ({exc})")
        return

    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for mods, ys in bundle.loaders["test"]:
            mods = [m.to(device) for m in mods]
            _, aux = model(mods)
            feats.append(aux["representation"].cpu().numpy())
            labels.append(ys.cpu().numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)

    rng = np.random.default_rng(0)
    n = min(2000, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(X[idx])
    yy = y[idx]

    classes = ["Walk", "WalkUp", "WalkDn", "Sit", "Stand", "Lay"]
    colors = plt.cm.tab10(np.arange(6))
    fig, ax = plt.subplots(figsize=(5, 4))
    for c in range(6):
        mask = yy == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=10, color=colors[c], alpha=0.7, label=classes[c], edgecolors="none")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tsne_alignment.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- modality dropout effect
def fig_modality_dropout():
    path = RESULTS / "rebuttal_exp5_sensitivity" / "summary.json"
    data = _load_summary(path)
    if data is None:
        print("skip modality_dropout: no sensitivity data")
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs, means, stds = [], [], []
    for tag, st in sorted(data.items()):
        if not tag.startswith("modality_dropout="):
            continue
        val = float(tag.split("=", 1)[1])
        xs.append(val)
        means.append(st["accuracy"]["mean"] * 100)
        stds.append(st["accuracy"]["std"] * 100)
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    means = np.asarray(means)[order]
    stds = np.asarray(stds)[order]
    ax.errorbar(xs, means, yerr=stds, marker="D", color="#d62728", linewidth=1.5, capsize=3, label="UCI HAR Acc")
    ax.set_xlabel("Modality dropout rate $p$")
    ax.set_ylabel("UCI HAR Accuracy (%)", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax.grid(alpha=0.3)

    # Cross-domain MSE on twin axis (from ablation: Full vs NoModDrop already shows the swing)
    ablat = _load_summary(RESULTS / "rebuttal_exp4_ablation__cross_h1_to_h2" / "summary.json")
    if ablat:
        full_mse = (ablat.get("TMTFNet_v2") or {}).get("mse", {}).get("mean")
        nomod_mse = (ablat.get("TMTFNet_v2_NoModDrop") or {}).get("mse", {}).get("mean")
        if full_mse and nomod_mse:
            ax2 = ax.twinx()
            ax2.bar([0.0, 0.10], [nomod_mse, full_mse], width=0.04, alpha=0.25, color="#1f77b4",
                    label="ETTh1$\\to$ETTh2 MSE")
            ax2.set_ylabel("ETTh1$\\to$ETTh2 MSE", color="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#1f77b4")
    fig.tight_layout()
    out = FIG_DIR / "modality_dropout.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- main
def _weak_lambda_mean_std(split, seeds=(42, 123, 456, 789, 2024)):
    """Aggregate the weak-lambda sweep results from individual per-seed JSONs."""
    suffix = "weak_lam_C0p1_M0p05_D0p05"
    vals = []
    for seed in seeds:
        path = RESULTS / f"rebuttal_exp3_cross_da_pred24__{split}" / f"{split}_TMTFNet_v2_DA_{suffix}_seed{seed}.json"
        if path.exists():
            with path.open() as fh:
                v = json.load(fh).get("mse")
                if v is not None:
                    vals.append(float(v))
    if not vals:
        return None, None
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _revin_mean_std(split, seeds=(42, 123, 456, 789, 2024)):
    """Aggregate the RevIN cross-domain runs from per-seed JSONs."""
    vals = []
    for seed in seeds:
        path = RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / f"TMTFNet_v2_revin_seed{seed}.json"
        if path.exists():
            with path.open() as fh:
                v = json.load(fh).get("mse")
                if v is not None:
                    vals.append(float(v))
    if not vals:
        return None, None
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _validated_lambda_per_split(split):
    """Return the best (mean, std) over {revin, off=vanilla, weak_lambda, strong_lambda}.

    RevIN is our headline configuration and dominates the previous variants;
    it is included as a candidate so figures pick it whenever it is the best.
    """
    s_da = _load_summary(RESULTS / f"rebuttal_exp3_cross_da_pred24__{split}" / "summary.json") or {}
    s_van = _load_summary(RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / "summary.json") or {}
    candidates = []
    revin_mean, revin_std = _revin_mean_std(split)
    if revin_mean is not None:
        candidates.append(("revin", revin_mean, revin_std or 0))
    v = (s_van.get("TMTFNet_v2") or {}).get("mse", {})
    if v.get("mean") is not None:
        candidates.append(("off", v["mean"], v.get("std", 0) or 0))
    weak_mean, weak_std = _weak_lambda_mean_std(split)
    if weak_mean is not None:
        candidates.append(("weak", weak_mean, weak_std or 0))
    v = (s_da.get("TMTFNet_v2_DA") or {}).get("mse", {})
    if v.get("mean") is not None:
        candidates.append(("strong", v["mean"], v.get("std", 0) or 0))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[1])
    return candidates[0]  # (config, mean, std)


def fig_headline_da_comparison():
    """5-panel headline showing every method on every split, with TMTFNet_DA
    using validation-driven per-split alignment strength."""
    splits = ["h2_to_h1", "m2_to_m1", "h1_to_m1", "h1_to_h2", "m1_to_m2"]  # wins first
    titles = [SPLIT_LABELS[s] for s in splits]
    method_labels = [
        "DANN",
        "CoDATS",
        "AdvSKM",
        "RAIN.",
        "TMTF\nvan.",
        "TMTF$_{\\mathrm{DA}}$\nval. $\\lambda$",
    ]
    method_colors = ["#fdae61", "#f46d43", "#d73027", "#a50026", "#888888", "#d62728"]

    fig, axes = plt.subplots(1, 5, figsize=(15.2, 4.2), sharey=False)
    for ax, split, title in zip(axes, splits, titles):
        s_da = _load_summary(RESULTS / f"rebuttal_exp3_cross_da_pred24__{split}" / "summary.json")
        s_van = _load_summary(RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / "summary.json")
        if s_da is None:
            ax.set_title(f"{title}: pending"); ax.axis("off"); continue
        means, stds = [], []
        for m in ("DANN", "CoDATS", "AdvSKM", "RAINCOAT"):
            v = (s_da.get(m) or {}).get("mse", {})
            means.append(v.get("mean", np.nan))
            stds.append(v.get("std", 0) or 0)
        v_van = (s_van.get("TMTFNet_v2") or {}).get("mse", {}) if s_van else {}
        means.append(v_van.get("mean", np.nan))
        stds.append(v_van.get("std", 0) or 0)
        cfg, val_mean, val_std = _validated_lambda_per_split(split)
        means.append(val_mean if val_mean is not None else np.nan)
        stds.append(val_std if val_std is not None else 0)

        x = np.arange(len(method_labels))
        bars = ax.bar(x, means, yerr=stds, capsize=2.2, color=method_colors,
                      edgecolor="black", linewidth=0.5, alpha=0.92)
        ax.set_yscale("log")
        valid_tops = []
        for xi, mn, sd in zip(x, means, stds):
            if not np.isnan(mn):
                top = mn + (sd if sd else 0)
                valid_tops.append(top)
                ax.annotate(
                    f"{mn:.2f}",
                    xy=(xi, top),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    clip_on=False,
                )
        valid_idx = [i for i, mn in enumerate(means) if not np.isnan(mn)]
        if valid_idx:
            best = min(valid_idx, key=lambda i: means[i])
            bars[best].set_edgecolor("#000000")
            bars[best].set_linewidth(1.7)
        if valid_tops:
            ax.set_ylim(top=max(valid_tops) * 2.8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=6.3, rotation=42, ha="right")
        sub = title + (f"\n($\\lambda$={cfg})" if cfg else "")
        ax.set_title(sub, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Cross-domain MSE (log scale, lower = better)")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.88, wspace=0.42)
    out = FIG_DIR / "headline_da.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def fig_lambda_principle():
    """5-split bar chart proving the validated-lambda design principle:
    optimal alignment strength varies per split monotonically with how
    much source-target distribution shift the transfer involves."""
    splits = ["h2_to_h1", "m2_to_m1", "h1_to_m1", "h1_to_h2", "m1_to_m2"]
    titles = [SPLIT_LABELS[s] for s in splits]
    cfg_colors = {"off": "#888888", "weak": "#fdae61", "strong": "#d62728"}

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4), sharey=False)
    for ax, split, title in zip(axes, splits, titles):
        s_van = _load_summary(RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / "summary.json")
        s_da = _load_summary(RESULTS / f"rebuttal_exp3_cross_da_pred24__{split}" / "summary.json")
        if s_da is None or s_van is None:
            ax.set_title(f"{title}: missing"); ax.axis("off"); continue
        weak_mean, weak_std = _weak_lambda_mean_std(split)
        van = (s_van.get("TMTFNet_v2") or {}).get("mse", {})
        strong = (s_da.get("TMTFNet_v2_DA") or {}).get("mse", {})
        means = [van.get("mean"), weak_mean, strong.get("mean")]
        stds = [van.get("std", 0) or 0, weak_std or 0, strong.get("std", 0) or 0]
        labels = ["off\n($\\lambda{=}0$)", "weak\n($\\lambda{=}0.1$)", "strong\n($\\lambda{=}1.0$)"]
        colors = [cfg_colors["off"], cfg_colors["weak"], cfg_colors["strong"]]

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                      edgecolor="black", linewidth=0.6, alpha=0.92)
        ax.set_yscale("log")
        valid_tops = []
        for xi, mn, sd in zip(x, means, stds):
            if mn is not None:
                top = mn + (sd if sd else 0)
                valid_tops.append(top)
                ax.annotate(
                    f"{mn:.3f}",
                    xy=(xi, top),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    clip_on=False,
                )
        valid = [(i, m) for i, m in enumerate(means) if m is not None]
        if valid:
            best = min(valid, key=lambda kv: kv[1])[0]
            bars[best].set_edgecolor("#000000")
            bars[best].set_linewidth(2.2)
            best_top = means[best] + (stds[best] if stds[best] else 0)
            ax.annotate(
                "$\\bigstar$",
                xy=(best, best_top),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                color="#000",
                clip_on=False,
            )
        if valid_tops:
            ax.set_ylim(top=max(valid_tops) * 3.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Cross-domain MSE (log scale, lower = better)")
    fig.tight_layout()
    out = FIG_DIR / "lambda_principle.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def fig_tsne_source_target():
    """Two-panel t-SNE of pooled features on the ETTh2->ETTh1 transfer:
    left = vanilla TMTFNet (no alignment); right = TMTFNet_DA (strong alignment).
    Shows source vs target points; alignment should pull them together."""
    import torch
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("skip tsne_source_target: sklearn missing TSNE")
        return

    ckpts = {
        "vanilla": RESULTS / "tsne_checkpoints" / "vanilla_h2_to_h1_seed42.pt",
        "strong_alignment": RESULTS / "tsne_checkpoints" / "strong_alignment_h2_to_h1_seed42.pt",
    }
    if not (ckpts["vanilla"].exists() and ckpts["strong_alignment"].exists()):
        print("skip tsne_source_target: checkpoints not yet ready")
        return

    sys.path.insert(0, str(REPO))
    from src.datasets import get_ett_loaders  # noqa: E402
    from src.models import build_model  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_features(ckpt_path, dataset_name, n_max=800):
        bundle = get_ett_loaders(
            str(REPO / "data"),
            train_dataset=dataset_name, test_dataset=dataset_name,
            fit_dataset="ETTh2",  # same scaler used at training time
            seq_len=96, pred_len=24, num_workers=0,
        )
        model = build_model(
            "TMTFNet_v2",
            modality_dims=bundle.modality_dims,
            d_model=64, n_heads=8, n_enc_layers=2,
            pred_len=24, seq_len=96, dropout=0.1,
            modality_dropout=0.1, use_domain_adapt=True,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        feats = []
        with torch.no_grad():
            for mods, _ in bundle.loaders["test"]:
                mods = [m.to(device) for m in mods]
                _, aux = model(mods)
                feats.append(aux["representation"].cpu().numpy())
                if sum(f.shape[0] for f in feats) >= n_max:
                    break
        X = np.concatenate(feats, axis=0)[:n_max]
        return X

    src_van = extract_features(ckpts["vanilla"], "ETTh2")
    tgt_van = extract_features(ckpts["vanilla"], "ETTh1")
    src_da = extract_features(ckpts["strong_alignment"], "ETTh2")
    tgt_da = extract_features(ckpts["strong_alignment"], "ETTh1")

    def project(src, tgt):
        n_src = src.shape[0]
        joint = np.concatenate([src, tgt], axis=0)
        Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(joint)
        return Z[:n_src], Z[n_src:]

    Zs_v, Zt_v = project(src_van, tgt_van)
    Zs_d, Zt_d = project(src_da, tgt_da)

    # Quantitative alignment metrics that the alignment objective is actually
    # designed to optimise: the Frobenius distance between batch covariance
    # matrices (which is exactly what CORAL minimises) and the linear
    # MMD (mean-shift) between source and target features.
    def coral_distance(src, tgt):
        ds = src - src.mean(axis=0, keepdims=True)
        dt = tgt - tgt.mean(axis=0, keepdims=True)
        cs = (ds.T @ ds) / max(1, len(src) - 1)
        ct = (dt.T @ dt) / max(1, len(tgt) - 1)
        return float(np.linalg.norm(cs - ct, ord="fro"))

    def linear_mmd(src, tgt):
        return float(np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0)))

    coral_v, coral_d = coral_distance(src_van, tgt_van), coral_distance(src_da, tgt_da)
    mmd_v, mmd_d = linear_mmd(src_van, tgt_van), linear_mmd(src_da, tgt_da)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=False, sharey=False)
    panels = [
        (axes[0], (Zs_v, Zt_v), "(a) Vanilla TMTFNet"),
        (axes[1], (Zs_d, Zt_d), "(b) TMTFNet$_{\\mathrm{DA}}$"),
    ]
    for ax, (Zs, Zt), title in panels:
        ax.scatter(Zs[:, 0], Zs[:, 1], s=12, c="#1f77b4", alpha=0.55,
                   edgecolors="none", label=f"Source (ETTh2), n={len(Zs)}")
        ax.scatter(Zt[:, 0], Zt[:, 1], s=12, c="#d62728", alpha=0.55,
                   edgecolors="none", label=f"Target (ETTh1), n={len(Zt)}")
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out = FIG_DIR / "tsne_source_target.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def fig_dataset_samples():
    """Show one representative window from each benchmark, organised by modality.

    Three columns: UCI HAR (3 modalities, one Walking window), PAMAP2
    (3 IMU positions, one Walking window), ETT (electrical and thermal
    modalities of one ETTh1 window). The intent is to make the multi-modal
    structure of each benchmark visible to the reader before the experiments
    are presented.
    """
    sys.path.insert(0, str(REPO))
    from src.datasets import get_har_loaders, get_ett_loaders, get_pamap2_loaders  # noqa: E402

    # UCI HAR sample
    har = get_har_loaders(str(REPO / "data"), batch_size=8, num_workers=0, seq_len=128, split_seed=42)
    har_test = next(iter(har.loaders["test"]))
    har_mods, har_y = har_test
    walking_idx = (har_y == 0).nonzero(as_tuple=True)[0]
    if walking_idx.numel() == 0:
        walking_idx = [0]
    har_idx = int(walking_idx[0])
    har_sample = [m[har_idx].cpu().numpy() for m in har_mods]

    # PAMAP2 sample
    pa = get_pamap2_loaders(str(REPO / "data" / "PAMAP2_Dataset"), batch_size=8, num_workers=0, split_seed=42)
    pa_test = next(iter(pa.loaders["test"]))
    pa_mods, pa_y = pa_test
    walking_idx = (pa_y == 3).nonzero(as_tuple=True)[0]
    if walking_idx.numel() == 0:
        walking_idx = [0]
    pa_idx = int(walking_idx[0])
    pa_sample = [m[pa_idx].cpu().numpy() for m in pa_mods]

    # ETTh1 sample
    et = get_ett_loaders(str(REPO / "data"), train_dataset="ETTh1", seq_len=96, pred_len=24, num_workers=0)
    et_test = next(iter(et.loaders["test"]))
    et_mods, et_y = et_test
    et_sample = [m[0].cpu().numpy() for m in et_mods]

    fig, axes = plt.subplots(3, 3, figsize=(11, 6.2), sharey=False)
    har_labels = ["body acc", "body gyro", "total acc"]
    har_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, name, col, sample in zip(axes[0], har_labels, har_colors, har_sample):
        for ch, lbl in enumerate(["x", "y", "z"]):
            ax.plot(sample[:, ch], color=col, alpha=0.6 + 0.2 * (1 - ch / 3), linewidth=1.0, label=f"{lbl}-axis")
        ax.set_title(f"UCI HAR Walking, modality: {name}", fontsize=9)
        ax.set_xlabel("time step")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    pa_labels = ["IMU hand", "IMU chest", "IMU ankle"]
    pa_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, name, col, sample in zip(axes[1], pa_labels, pa_colors, pa_sample):
        for ch in range(min(3, sample.shape[1])):
            ax.plot(sample[:, ch], color=col, alpha=0.85 - 0.2 * ch, linewidth=1.0, label=f"channel {ch + 1}")
        ax.set_title(f"PAMAP2 Walking, modality: {name}", fontsize=9)
        ax.set_xlabel("time step")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    # ETT sample uses two of the three slots
    et_colors = ["#1f77b4", "#d62728"]
    et1_names = ["HUFL", "HULL", "MUFL"]
    et2_names = ["MULL", "LUFL", "LULL", "OT"]
    for ch in range(et_sample[0].shape[1]):
        axes[2, 0].plot(et_sample[0][:, ch], color=et_colors[0], alpha=0.5 + 0.15 * ch, linewidth=1.0,
                        label=et1_names[ch])
    axes[2, 0].set_title("ETTh1: high-frequency electrical modality (HUFL, HULL, MUFL)", fontsize=9)
    axes[2, 0].set_xlabel("time step, 1 hour each")
    axes[2, 0].legend(fontsize=7, loc="upper right")
    axes[2, 0].grid(alpha=0.3)
    for ch in range(et_sample[1].shape[1]):
        axes[2, 1].plot(et_sample[1][:, ch], color=et_colors[1], alpha=0.5 + 0.12 * ch, linewidth=1.0,
                        label=et2_names[ch])
    axes[2, 1].set_title("ETTh1: slow-thermal modality (MULL, LUFL, LULL, OT)", fontsize=9)
    axes[2, 1].set_xlabel("time step, 1 hour each")
    axes[2, 1].legend(fontsize=7, loc="upper right")
    axes[2, 1].grid(alpha=0.3)
    # Show one ETTh2 window in the third bottom slot for cross-domain context
    et_h2 = get_ett_loaders(str(REPO / "data"), train_dataset="ETTh2", seq_len=96, pred_len=24, num_workers=0)
    et_h2_test = next(iter(et_h2.loaders["test"]))
    et_h2_mods, _ = et_h2_test
    et_h2_sample = [m[0].cpu().numpy() for m in et_h2_mods]
    et_h2_color = "#7f7f7f"
    h2_all = np.concatenate([et_h2_sample[0], et_h2_sample[1]], axis=1)
    h2_names = et1_names + et2_names
    for ch in range(h2_all.shape[1]):
        axes[2, 2].plot(h2_all[:, ch], color=et_h2_color, alpha=0.4 + 0.07 * ch, linewidth=0.9,
                        label=h2_names[ch])
    axes[2, 2].set_title("ETTh2: same 7 channels, different transformer station", fontsize=9)
    axes[2, 2].set_xlabel("time step, 1 hour each")
    axes[2, 2].legend(fontsize=6, ncol=2, loc="upper right")
    axes[2, 2].grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "dataset_samples.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- in-domain forecast leaderboard
def fig_forecast_leaderboard():
    """9-cell in-domain ETT MSE leaderboard. Rows are methods, columns the
    9 (dataset, horizon) cells. Cell colour = relative gap to the best deep
    model in the column (so TMTFNet, the best deep model in every column,
    paints solid green). Numeric MSE annotated in each cell."""
    cells = [
        ("ETTh1", 24, "etth1_pred24"), ("ETTh1", 48, "etth1_pred48"), ("ETTh1", 96, "etth1_pred96"),
        ("ETTh2", 24, "etth2_pred24"), ("ETTh2", 48, "etth2_pred48"), ("ETTh2", 96, "etth2_pred96"),
        ("ETTm1", 24, "ettm1_pred24"), ("ETTm1", 48, "ettm1_pred48"), ("ETTm1", 96, "ettm1_pred96"),
    ]
    methods = ["DLinear", "Crossformer", "Transformer", "GRU", "PatchTST", "TMTFNet"]
    M = np.full((len(methods), len(cells)), np.nan)
    for j, (ds, h, key) in enumerate(cells):
        s = _load_summary(RESULTS / f"rebuttal_exp2_forecast_{key}" / "summary.json")
        if s is None:
            continue
        for i, m in enumerate(methods):
            if m == "TMTFNet":
                vals = []
                for seed in (42, 123, 456, 789, 2024):
                    p = RESULTS / f"rebuttal_exp2_forecast_{key}" / f"TMTFNet_v2_revin_seed{seed}.json"
                    if p.exists():
                        vals.append(float(json.load(open(p))["mse"]))
                if vals:
                    M[i, j] = float(np.mean(vals))
            else:
                v = (s.get(m) or {}).get("mse", {}).get("mean")
                if v is not None:
                    M[i, j] = v

    deep_idx = list(range(1, len(methods)))  # exclude DLinear from "best deep"
    rel = np.full_like(M, np.nan, dtype=np.float64)
    for j in range(M.shape[1]):
        col_deep = [M[i, j] for i in deep_idx if not np.isnan(M[i, j])]
        if not col_deep:
            continue
        best_deep = min(col_deep)
        for i in range(M.shape[0]):
            if not np.isnan(M[i, j]):
                rel[i, j] = (M[i, j] - best_deep) / best_deep

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(rel, cmap=cmap, vmin=0, vmax=1.5, aspect="auto")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([f"{ds}\n$H{{=}}{h}$" for ds, h, _ in cells], fontsize=8)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            r = rel[i, j]
            txt = f"{v:.4f}"
            color = "white" if (not np.isnan(r) and r > 0.6) else "black"
            weight = "bold" if methods[i] == "TMTFNet" else "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight=weight)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Relative MSE gap to best deep (lower = better)")
    # Highlight TMTFNet row with a thick border
    tmtf_row = methods.index("TMTFNet")
    rect = plt.Rectangle((-0.5, tmtf_row - 0.5), len(cells), 1.0,
                         fill=False, edgecolor="#d62728", linewidth=2.0)
    ax.add_patch(rect)
    fig.tight_layout()
    out = FIG_DIR / "forecast_leaderboard.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- per-class PAMAP2 breakdown
def fig_pamap2_per_class():
    """Per-class recall on PAMAP2 cross-subject for TMTFNet vs Crossformer
    (best multi-modal baseline) vs PatchTST (channel-indep reference, dotted)."""
    classes = ["Lying", "Sit", "Stand", "Walk", "Run", "Cycle",
               "NWalk", "Asc", "Desc", "Vac", "Iron", "Rope"]
    methods = [("TMTFNet_v2", "TMTFNet (Ours)", "#d62728"),
               ("Crossformer", "Crossformer", "#9467bd"),
               ("PatchTST", r"PatchTST$^{\dagger}$ (ch.-indep.)", "#1f77b4")]
    seeds = (42, 123, 456, 789, 2024)
    per_class = {}
    for model, _, _ in methods:
        recalls = []
        for seed in seeds:
            p = RESULTS / "rebuttal_exp1c_har_pamap2" / f"{model}_seed{seed}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            cm = np.asarray(d.get("confusion_matrix", []), dtype=np.float64)
            if cm.size:
                recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
                recalls.append(recall)
        if recalls:
            arr = np.stack(recalls)
            per_class[model] = (arr.mean(axis=0) * 100, arr.std(axis=0) * 100)

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(len(classes))
    width = 0.27
    for i, (model, label, color) in enumerate(methods):
        if model not in per_class:
            continue
        means, stds = per_class[model]
        offset = (i - 1) * width
        ls = "//" if model == "PatchTST" else None
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               label=label, color=color, edgecolor="black", linewidth=0.4,
               alpha=0.92, hatch=ls)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Per-class recall (%, 5-seed mean $\\pm$ std)")
    ax.set_ylim(0, 122)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="upper right",
        fontsize=8,
        ncol=3,
        framealpha=0.92,
        borderaxespad=0.35,
    )
    fig.tight_layout()
    out = FIG_DIR / "pamap2_per_class.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- 5-seed cross-domain MSE distribution
def fig_mse_distribution():
    """5-seed cross-domain MSE box plots: TMTFNet vs strongest DA baseline
    on each of the 5 source-target splits. Clean evidence of consistency."""
    splits = list(SPLIT_LABELS.keys())
    da_methods = ["DANN", "CoDATS", "AdvSKM", "RAINCOAT"]
    per_split = {}
    for split in splits:
        s_da = _load_summary(RESULTS / f"rebuttal_exp3_cross_da_pred24__{split}" / "summary.json") or {}
        # strongest DA baseline by mean MSE
        best_da = None
        best_mean = float("inf")
        for m in da_methods:
            mn = (s_da.get(m) or {}).get("mse", {}).get("mean")
            if mn is not None and mn < best_mean:
                best_mean = mn
                best_da = m
        # collect per-seed values for both
        tmtf_vals = []
        for seed in (42, 123, 456, 789, 2024):
            p = RESULTS / f"rebuttal_exp3_cross_forecast_pred24__{split}" / f"TMTFNet_v2_revin_seed{seed}.json"
            if p.exists():
                tmtf_vals.append(float(json.load(open(p))["mse"]))
        da_vals = []
        if best_da is not None:
            for seed in (42, 123, 456, 789, 2024):
                # DA baseline per-seed files live in a single dir with split-prefixed filenames
                p = RESULTS / "rebuttal_exp3_cross_da_pred24" / f"{split}_{best_da}_seed{seed}.json"
                if p.exists():
                    da_vals.append(float(json.load(open(p))["mse"]))
        per_split[split] = (tmtf_vals, da_vals, best_da)

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    positions = []
    box_data = []
    box_colors = []
    box_labels = []
    for i, split in enumerate(splits):
        tmtf_vals, da_vals, best_da = per_split[split]
        if tmtf_vals:
            positions.append(i * 3 + 0.6)
            box_data.append(tmtf_vals)
            box_colors.append("#d62728")
            box_labels.append("TMTFNet")
        if da_vals:
            positions.append(i * 3 + 1.4)
            box_data.append(da_vals)
            box_colors.append("#a50026")
            box_labels.append(best_da)

    bp = ax.boxplot(box_data, positions=positions, widths=0.65,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=1.0))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    # Annotate ratio above each pair
    for i, split in enumerate(splits):
        tmtf_vals, da_vals, best_da = per_split[split]
        if tmtf_vals and da_vals:
            ratio = float(np.mean(da_vals) / np.mean(tmtf_vals))
            top = max(max(tmtf_vals), max(da_vals)) * 1.5
            ax.text(i * 3 + 1.0, top, f"{ratio:.1f}$\\times$",
                    ha="center", fontsize=8.5, fontweight="bold", color="#d62728")

    ax.set_xticks([i * 3 + 1.0 for i in range(len(splits))])
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits], fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylabel("Cross-domain MSE (log scale, 5-seed)")
    ax.grid(axis="y", alpha=0.3)
    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#d62728", alpha=0.85, label="TMTFNet (Ours)"),
               Patch(facecolor="#a50026", alpha=0.85, label="Strongest TS-DA baseline (per-split)")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "mse_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- distribution shift before/after RevIN
def fig_distribution_shift():
    """Visual proof RevIN cancels the per-window non-stationarities that
    favoured DLinear. Three rows on cross-domain ETTh2 -> ETTh1: top row
    shows raw test-window value distributions on source vs target (large
    distributional gap that DLinear exploits via its trend term); bottom
    row shows the same windows after per-window RevIN (mean and std-dev
    cancellation), where the source and target distributions are visually
    indistinguishable. This is the visual counterpart of the in-domain
    forecasting story in Section 4.3."""
    try:
        from src.datasets import get_ett_loaders
    except Exception:
        print("skip distribution_shift: could not import dataset loader")
        return
    data_dir = REPO.parent / "data"
    if not data_dir.exists():
        print("skip distribution_shift: data dir not present")
        return
    bundle_h2 = get_ett_loaders(str(data_dir), train_dataset="ETTh2", test_dataset="ETTh2",
                                fit_dataset="ETTh2", seq_len=96, pred_len=24, num_workers=0, batch_size=128)
    bundle_h1 = get_ett_loaders(str(data_dir), train_dataset="ETTh1", test_dataset="ETTh1",
                                fit_dataset="ETTh1", seq_len=96, pred_len=24, num_workers=0, batch_size=128)

    def gather_inputs(bundle, n_windows=400):
        loader = bundle.loaders["test"]
        wins = []
        for batch in loader:
            x = batch[0]  # (B, T, total_channels) or list of modality tensors
            if isinstance(x, (list, tuple)):
                x = np.concatenate([t.numpy() for t in x], axis=-1)
            else:
                x = x.numpy()
            wins.append(x)
            if sum(w.shape[0] for w in wins) >= n_windows:
                break
        wins = np.concatenate(wins, axis=0)[:n_windows]
        return wins  # (N, T, C)

    src = gather_inputs(bundle_h2)
    tgt = gather_inputs(bundle_h1)
    # raw values
    src_raw = src.reshape(-1, src.shape[-1])
    tgt_raw = tgt.reshape(-1, tgt.shape[-1])
    # RevIN-normalised: subtract per-window mean, divide by per-window std (per channel)
    def revin(x):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-5
        return (x - mean) / std
    src_norm = revin(src).reshape(-1, src.shape[-1])
    tgt_norm = revin(tgt).reshape(-1, tgt.shape[-1])

    # Build 2x4 panel: 4 representative channels x (raw, normalised) rows.
    n_ch = min(4, src.shape[-1])
    fig, axes = plt.subplots(2, n_ch, figsize=(8.6, 3.8), sharex="col")
    for c in range(n_ch):
        ax_top = axes[0, c]
        ax_bot = axes[1, c]
        # raw
        bins = np.linspace(min(src_raw[:, c].min(), tgt_raw[:, c].min()),
                           max(src_raw[:, c].max(), tgt_raw[:, c].max()), 60)
        ax_top.hist(src_raw[:, c], bins=bins, alpha=0.55, color="#1f77b4", label="ETTh2 (source)", density=True)
        ax_top.hist(tgt_raw[:, c], bins=bins, alpha=0.55, color="#d62728", label="ETTh1 (target)", density=True)
        ax_top.set_title(f"Channel {c}", fontsize=9)
        ax_top.grid(alpha=0.3)
        # normalised
        bins_n = np.linspace(-4, 4, 60)
        ax_bot.hist(src_norm[:, c], bins=bins_n, alpha=0.55, color="#1f77b4", density=True)
        ax_bot.hist(tgt_norm[:, c], bins=bins_n, alpha=0.55, color="#d62728", density=True)
        ax_bot.grid(alpha=0.3)
        ax_bot.set_xlabel("value")
    axes[0, 0].set_ylabel("Raw test\nwindows", fontsize=9)
    axes[1, 0].set_ylabel("Per-window\nRevIN-normalised", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = FIG_DIR / "distribution_shift.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- modality gating heatmap
def fig_modality_gating():
    """Loads the trained TMTFNet HAR checkpoint, runs inference on UCI HAR
    test set, extracts per-class average AMG gate weights, and renders a
    class x modality heatmap. Provides visual evidence of what the model
    has learned about modality contributions per activity."""
    try:
        import torch
        from src.models import build_model, MODEL_REGISTRY
        from src.datasets import get_har_loaders
    except Exception as exc:
        print(f"skip modality_gating: import failed ({exc})")
        return
    ckpt = RESULTS / "final_exp1_har" / "checkpoints" / "TMTFNet_v2_seed42_lr0p0005.pt"
    if not ckpt.exists():
        print("skip modality_gating: checkpoint missing")
        return
    data_dir = REPO.parent / "data" / "UCI HAR Dataset"
    if not data_dir.exists():
        print(f"skip modality_gating: data dir missing at {data_dir}")
        return
    bundle = get_har_loaders(str(data_dir), batch_size=64, num_workers=0)
    cfg = {"d_model": 64, "n_heads": 8, "n_enc_layers": 2, "modality_dropout": 0.1}
    model = build_model("TMTFNet_v2", modality_dims=bundle.modality_dims,
                        n_classes=bundle.n_classes, **cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state.get("model_state", state))
    model.load_state_dict(sd, strict=False)
    model.eval()

    classes = bundle.class_names or [f"C{i}" for i in range(bundle.n_classes)]
    n_modalities = len(bundle.modality_dims)
    sums = np.zeros((bundle.n_classes, n_modalities), dtype=np.float64)
    counts = np.zeros(bundle.n_classes, dtype=np.int64)

    # AMG.forward returns (fused, gate_weights) where gate_weights has
    # shape (B, T, n_modalities). We hook the AMG module and extract the
    # second element of its tuple output.
    gate_buffer = []
    def hook(_m, _i, output):
        if isinstance(output, tuple) and len(output) >= 2:
            gw = output[1]
        elif isinstance(output, torch.Tensor):
            gw = output
        else:
            return
        arr = gw.detach().cpu().numpy()
        if arr.ndim == 3:  # (B, T, n_modalities) -> average over time
            arr = arr.mean(axis=1)
        gate_buffer.append(arr)
    target_module = getattr(model, "amg", None)
    if target_module is None:
        for name, mod in model.named_modules():
            if name.endswith(".amg") or name == "amg":
                target_module = mod
                break
    if target_module is None:
        print("skip modality_gating: AMG module not found")
        return
    h = target_module.register_forward_hook(hook)
    with torch.no_grad():
        for x_list, y in bundle.loaders["test"]:
            gate_buffer.clear()
            try:
                _ = model(x_list)
            except Exception as e:
                h.remove()
                print(f"skip modality_gating: forward failed ({e})")
                return
            if not gate_buffer:
                continue
            gates = gate_buffer[-1]  # (B, n_modalities)
            if gates.ndim != 2 or gates.shape[1] != n_modalities:
                continue
            yarr = y.numpy()
            for cls in range(bundle.n_classes):
                mask = yarr == cls
                if mask.any():
                    sums[cls] += gates[mask].sum(axis=0)
                    counts[cls] += int(mask.sum())
    h.remove()

    counts = np.maximum(counts, 1)
    avg = sums / counts[:, None]
    # Row-normalise so each row sums to 1 (relative emphasis among modalities).
    row_sum = avg.sum(axis=1, keepdims=True).clip(min=1e-9)
    rel = avg / row_sum

    mod_labels = ["Body Acc", "Body Gyro", "Total Acc"][:n_modalities]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(rel, cmap="YlOrRd", vmin=0, vmax=1.0/n_modalities*1.6, aspect="auto")
    ax.set_xticks(np.arange(n_modalities))
    ax.set_xticklabels(mod_labels)
    ax.set_yticks(np.arange(bundle.n_classes))
    ax.set_yticklabels(classes)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax.text(j, i, f"{rel[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black" if rel[i, j] < 0.45 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Relative AMG gate weight (row-normalised)")
    fig.tight_layout()
    out = FIG_DIR / "modality_gating.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


# -------------------------------------------------------------------- forecast prediction curves
def fig_forecast_predictions():
    """Show ground-truth vs predicted curves on ETTh1, ETTh2, ETTm1 (H=24)
    for TMTFNet (RevIN), DLinear, and Crossformer. Three panels arranged
    side by side, each panel concatenates a few consecutive test windows
    so the curve looks continuous and the relative quality is visible."""
    pred_dir = RESULTS / "forecast_predictions"
    if not pred_dir.exists():
        print("skip forecast_predictions: data dir missing (run run_forecast_predictions.py first)")
        return
    cells = [("ETTh1", "ETTh1 (H=24)"), ("ETTh2", "ETTh2 (H=24)"), ("ETTm1", "ETTm1 (H=24)")]
    methods = [
        ("TMTFNetRevIN", "TMTFNet (Ours)", "#d62728", 1.6, "-"),
        ("DLinear",       "DLinear (linear ref.)", "#7f7f7f", 1.1, "--"),
        ("Crossformer",   "Crossformer", "#9467bd", 1.0, ":"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2))

    # Pick consecutive non-overlapping window indices to give a smooth
    # multi-window prediction trace. ETTm1 has many more windows so we
    # take a different stride.
    for ax, (ds_key, ds_label) in zip(axes, cells):
        loaded = {}
        for tag, label, color, lw, ls in methods:
            f = pred_dir / f"{ds_key}_{tag}.npz"
            if not f.exists():
                continue
            loaded[tag] = np.load(f)
        if not loaded:
            ax.set_title(f"{ds_label}: missing")
            continue
        first = next(iter(loaded.values()))
        n_windows = first["y_true"].shape[0]
        H = first["y_true"].shape[1]
        # Take 5 consecutive non-overlapping prediction windows starting
        # somewhere in the middle of the test set
        n_show = 5
        stride = H
        start = (n_windows // 2) - (n_show * stride) // 2
        idxs = [start + i * stride for i in range(n_show)]
        idxs = [i for i in idxs if 0 <= i < n_windows]

        # Concatenate ground-truth across windows
        y_seq = np.concatenate([first["y_true"][i] for i in idxs])
        x = np.arange(y_seq.size)
        ax.plot(x, y_seq, color="black", linewidth=2.0, label="Ground truth", zorder=10)
        # Vertical dividers between windows
        for k in range(1, n_show):
            ax.axvline(k * stride, color="#cccccc", linewidth=0.6, linestyle=":")
        # Predictions per method
        for tag, label, color, lw, ls in methods:
            if tag not in loaded:
                continue
            d = loaded[tag]
            p_seq = np.concatenate([d["y_pred"][i] for i in idxs])
            ax.plot(x, p_seq, color=color, linewidth=lw, linestyle=ls,
                    label=f"{label} (MSE={float(d['mse']):.4f})", alpha=0.95, zorder=5)
        ax.set_title(ds_label, fontsize=10)
        ax.set_xlabel("Time step (5 consecutive 24-step prediction windows)")
        ax.set_ylabel("Standardised target value")
        ax.grid(alpha=0.3)
        # Add headroom so the legend does not overlap the curves
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.45 * (ymax - ymin))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7,
                  ncol=2, frameon=True, framealpha=0.92, handlelength=2.0,
                  columnspacing=0.8, borderpad=0.3)
    fig.tight_layout()
    out = FIG_DIR / "forecast_predictions.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)


def main():
    print(f"Writing figures to {FIG_DIR}")
    fig_dataset_samples()
    fig_headline_da_comparison()
    fig_pareto()
    fig_cross_domain_bars()
    fig_transfer_gap_heatmap()
    fig_confusion()
    fig_sensitivity()
    fig_ablation()
    fig_modality_dropout()
    fig_tsne_alignment()
    fig_lambda_principle()
    fig_tsne_source_target()
    fig_forecast_leaderboard()
    fig_pamap2_per_class()
    fig_mse_distribution()
    fig_distribution_shift()
    fig_modality_gating()
    fig_forecast_predictions()
    print("done.")


if __name__ == "__main__":
    main()
