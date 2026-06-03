#!/usr/bin/env python3
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_PNG = FIG_DIR / "intro_motivation_params_acc.png"
OUT_PDF = FIG_DIR / "intro_motivation_params_acc.pdf"
OUT_CSV = FIG_DIR / "intro_motivation_params_acc.csv"


@dataclass
class Point:
    label: str
    short_label: str
    params_k: float
    acc_mean: float
    acc_std: float
    family: str


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def point_from_summary(summary, key, label, short_label, family):
    entry = summary[key]
    return Point(
        label=label,
        short_label=short_label,
        params_k=entry["n_params"] / 1000.0,
        acc_mean=entry["accuracy"]["mean"] * 100.0,
        acc_std=entry["accuracy"]["std"] * 100.0,
        family=family,
    )


def build_points():
    base = load_json(ROOT / "results" / "exp1_har" / "summary.json")
    final = load_json(ROOT / "results" / "final_exp1_har" / "summary.json")
    lite = load_json(ROOT / "results" / "lite_exp1_har" / "summary.json")

    points = [
        point_from_summary(base, "DLinear", "DLinear", "DLinear", "baseline"),
        point_from_summary(base, "TimeMixer", "TimeMixer", "TimeMixer", "baseline"),
        point_from_summary(base, "TCN", "TCN", "TCN", "baseline"),
        point_from_summary(base, "iTransformer", "iTransformer", "iTransformer", "baseline"),
        point_from_summary(base, "Transformer", "Transformer", "Transformer", "baseline"),
        point_from_summary(base, "GRU", "GRU", "GRU", "baseline"),
        point_from_summary(base, "LSTM", "LSTM", "LSTM", "baseline"),
        point_from_summary(base, "Crossformer", "Crossformer", "Crossformer", "baseline"),
        point_from_summary(base, "PatchTST", "PatchTST", "PatchTST", "patchtst"),
        point_from_summary(base, "TMTFNet_v2", "TMTFNet_v2 (Original)", "TMTFNet_v2\noriginal", "ours_old"),
        Point(
            label="TMTFNet-Lite",
            short_label="TMTFNet-Lite",
            params_k=lite["n_params"] / 1000.0,
            acc_mean=lite["accuracy"]["mean"] * 100.0,
            acc_std=lite["accuracy"]["std"] * 100.0,
            family="ours_lite",
        ),
        Point(
            label="TMTFNet_v2 (Final)",
            short_label="TMTFNet_v2\nfinal",
            params_k=final["n_params"] / 1000.0,
            acc_mean=final["accuracy"]["mean"] * 100.0,
            acc_std=final["accuracy"]["std"] * 100.0,
            family="ours_final",
        ),
    ]
    return points


def pareto_frontier(points):
    sorted_points = sorted(points, key=lambda item: item.params_k)
    frontier = []
    best_acc = float("-inf")
    for point in sorted_points:
        if point.acc_mean > best_acc:
            frontier.append(point)
            best_acc = point.acc_mean
    return frontier


def write_csv(points):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "params_k", "accuracy_mean", "accuracy_std", "family"])
        for point in sorted(points, key=lambda item: item.params_k):
            writer.writerow([point.label, f"{point.params_k:.1f}", f"{point.acc_mean:.2f}", f"{point.acc_std:.2f}", point.family])


def draw(points):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )

    colors = {
        "baseline": "#94A3B8",
        "patchtst": "#2563EB",
        "ours_old": "#B91C1C",
        "ours_lite": "#F97316",
        "ours_final": "#DC2626",
    }
    markers = {
        "baseline": "o",
        "patchtst": "D",
        "ours_old": "^",
        "ours_lite": "s",
        "ours_final": "h",
    }
    sizes = {
        "baseline": 70,
        "patchtst": 135,
        "ours_old": 145,
        "ours_lite": 135,
        "ours_final": 170,
    }
    offsets = {
        "DLinear": (12, 16),
        "TimeMixer": (12, 4),
        "TCN": (12, 2),
        "iTransformer": (10, -6),
        "Transformer": (12, -10),
        "GRU": (8, -26),
        "LSTM": (50, -8),
        "Crossformer": (10, -12),
        "PatchTST": (18, 8),
        "TMTFNet_v2\noriginal": (108, -2),
        "TMTFNet-Lite": (12, -12),
        "TMTFNet_v2\nfinal": (-44, -4),
    }
    label_style = {
        "PatchTST": {"ha": "left", "va": "bottom"},
        "TMTFNet_v2\noriginal": {"ha": "left", "va": "center"},
        "TMTFNet_v2\nfinal": {"ha": "right", "va": "top"},
    }

    fig, ax = plt.subplots(figsize=(10.6, 7.0), dpi=200)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    ax.axvspan(350, 1000, color="#FEE2E2", alpha=0.45, zorder=0)
    ax.axhspan(93.5, 95.6, xmin=0.43, xmax=0.79, color="#FEF3C7", alpha=0.45, zorder=0)
    ax.text(115, 94.62, "Desired regime:\nsub-1M params with >93.5% accuracy", fontsize=10, color="#7C2D12")

    frontier = pareto_frontier(points)
    ax.plot(
        [point.params_k for point in frontier],
        [point.acc_mean for point in frontier],
        linestyle=(0, (4, 4)),
        linewidth=2.0,
        color="#0F766E",
        alpha=0.9,
        zorder=1,
    )
    ax.text(10, 94.9, "Pareto frontier", color="#0F766E", fontsize=10)

    for point in points:
        ax.errorbar(
            point.params_k,
            point.acc_mean,
            yerr=point.acc_std,
            fmt="none",
            ecolor=colors[point.family],
            elinewidth=1.4,
            alpha=0.85,
            capsize=3,
            zorder=2,
        )
        edge = "#7F1D1D" if point.family == "ours_final" else "white"
        ax.scatter(
            point.params_k,
            point.acc_mean,
            s=sizes[point.family],
            c=colors[point.family],
            marker=markers[point.family],
            edgecolors=edge,
            linewidths=1.4,
            alpha=0.96,
            zorder=3,
        )
        dx, dy = offsets[point.short_label]
        text_color = "#111827" if point.family == "baseline" else colors[point.family]
        ax.annotate(
            f"{point.short_label}\n{point.params_k:.0f}K, {point.acc_mean:.2f}%",
            xy=(point.params_k, point.acc_mean),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=text_color,
            weight="bold" if point.family != "baseline" else None,
            zorder=4,
            **label_style.get(point.short_label, {}),
        )

    ax.set_xscale("log")
    ax.set_xlim(1.6, 4200)
    ax.set_ylim(74.0, 95.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.18)
    ax.set_xlabel("Trainable Parameters (K, log scale)", fontsize=12, labelpad=10)
    ax.set_ylabel("UCI HAR Accuracy (%)", fontsize=12, labelpad=10)
    ax.set_xticks([2, 10, 100, 1000, 4000])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}" if value >= 10 else f"{value:g}"))
    ax.tick_params(labelsize=10)

    ax.text(
        0.30,
        0.03,
        "Motivation: brute-force multimodal fusion grows quickly in size,\n"
        "while aggressive compression loses accuracy. The final TMTFNet_v2\n"
        "keeps a PatchTST-scale budget and stays near the high-accuracy frontier.",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#1F2937",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.92),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    points = build_points()
    write_csv(points)
    draw(points)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
