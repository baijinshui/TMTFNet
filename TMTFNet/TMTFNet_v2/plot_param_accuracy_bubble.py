#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_PNG = FIG_DIR / "param_accuracy.png"
OUT_PDF = FIG_DIR / "param_accuracy.pdf"


MODELS = [
    {"name": "DLinear", "params_k": 2.2, "acc": 75.06, "std": 2.94, "group": "baseline"},
    {"name": "TimeMixer", "params_k": 118.2, "acc": 91.41, "std": 0.39, "group": "baseline"},
    {"name": "TCN", "params_k": 564.5, "acc": 91.51, "std": 0.40, "group": "baseline"},
    {"name": "iTransformer", "params_k": 628.9, "acc": 88.59, "std": 0.33, "group": "baseline"},
    {"name": "Transformer", "params_k": 679.4, "acc": 90.32, "std": 0.80, "group": "baseline"},
    {"name": "GRU", "params_k": 733.8, "acc": 92.30, "std": 0.41, "group": "baseline"},
    {"name": "TMTFNet (Ours)", "params_k": 797.1, "acc": 94.78, "std": 0.50, "group": "ours"},
    {"name": "PatchTST", "params_k": 812.2, "acc": 94.28, "std": 0.58, "group": "patchtst"},
    {"name": "LSTM", "params_k": 967.0, "acc": 92.69, "std": 0.12, "group": "baseline"},
    {"name": "Crossformer", "params_k": 2068.0, "acc": 90.26, "std": 1.12, "group": "baseline"},
]


COLORS = {
    "ours": "#E74C3C",
    "patchtst": "#4A90D9",
    "baseline": "#B0BEC5",
}

EDGES = {
    "ours": "black",
    "patchtst": "#4A90D9",
    "baseline": "#90A4AE",
}

LINE_WIDTHS = {
    "ours": 2.0,
    "patchtst": 0.8,
    "baseline": 0.5,
}

LABEL_POS = {
    "DLinear": ((16, 18), "left", "bottom"),
    "TimeMixer": ((10, 8), "left", "bottom"),
    "TCN": ((-30, 18), "right", "bottom"),
    "iTransformer": ((-12, -2), "right", "center"),
    "Transformer": ((10, -10), "left", "top"),
    "GRU": ((-10, -20), "right", "top"),
    "TMTFNet (Ours)": ((-18, 8), "right", "bottom"),
    "PatchTST": ((16, -6), "left", "top"),
    "LSTM": ((12, 2), "left", "center"),
    "Crossformer": ((-14, 0), "right", "center"),
}


def bubble_area(std_value):
    min_std, max_std = 0.12, 2.94
    min_area, max_area = 8 ** 2, 40 ** 2
    ratio = (std_value - min_std) / (max_std - min_std)
    return min_area + ratio * (max_area - min_area)


def build_size_handles():
    refs = [0.2, 1.0, 2.0]
    handles = [
        plt.scatter(
            [],
            [],
            s=bubble_area(std),
            facecolor="#CFD8DC",
            edgecolor="#90A4AE",
            linewidth=0.7,
            alpha=0.75,
        )
        for std in refs
    ]
    labels = [f"Std={std:.1f}" for std in refs]
    return handles, labels


def style_matplotlib():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw():
    style_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 6.3), dpi=300)

    ideal_zone = Rectangle(
        (400, 92.0),
        600,
        5.0,
        facecolor="#6FCF97",
        edgecolor="none",
        alpha=0.05,
        zorder=0,
    )
    ax.add_patch(ideal_zone)
    ax.text(
        420,
        96.82,
        "High accuracy, moderate params",
        fontsize=12.5,
        style="italic",
        color="#2E7D32",
        ha="left",
        va="top",
    )

    for item in MODELS:
        if item["group"] == "ours":
            continue
        ax.scatter(
            item["params_k"],
            item["acc"],
            s=bubble_area(item["std"]),
            c=COLORS[item["group"]],
            edgecolors=EDGES[item["group"]],
            linewidths=LINE_WIDTHS[item["group"]],
            alpha=0.75,
            zorder=2 if item["group"] == "baseline" else 3,
        )

    ours = next(item for item in MODELS if item["group"] == "ours")
    ax.scatter(
        ours["params_k"],
        ours["acc"],
        s=bubble_area(ours["std"]) * 1.06,
        c="white",
        edgecolors="white",
        linewidths=2.0,
        alpha=0.95,
        zorder=4,
    )
    ax.scatter(
        ours["params_k"],
        ours["acc"],
        s=bubble_area(ours["std"]) * 0.92,
        c=COLORS["ours"],
        edgecolors="black",
        linewidths=2.0,
        alpha=0.75,
        zorder=5,
    )

    for item in MODELS:
        offset, ha, va = LABEL_POS[item["name"]]
        text_kwargs = {
            "fontsize": 15 if item["name"] == "TMTFNet (Ours)" else (12.5 if item["name"] == "PatchTST" else 11),
            "ha": ha,
            "va": va,
            "color": "black",
            "zorder": 5,
        }
        if item["name"] == "TMTFNet (Ours)":
            text_kwargs["fontweight"] = "bold"
        arrowprops = None
        if item["name"] == "TMTFNet (Ours)":
            arrowprops = {
                "arrowstyle": "-",
                "color": "black",
                "lw": 0.8,
                "shrinkA": 4,
                "shrinkB": 4,
                "alpha": 0.7,
            }
        ax.annotate(
            item["name"],
            xy=(item["params_k"], item["acc"]),
            xytext=offset,
            textcoords="offset points",
            arrowprops=arrowprops,
            **text_kwargs,
        )

    ax.set_xscale("log")
    ax.set_xlim(1.8, 2400)
    ax.set_ylim(73.0, 97.2)

    xticks = [2, 10, 50, 100, 500, 1000, 2000]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in xticks]))
    ax.set_yticks(list(range(73, 97, 2)))

    ax.set_xlabel("Parameters (K)", fontsize=17, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=17, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.grid(axis="y", color="#D0D7DE", linewidth=0.5, alpha=0.10)

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["ours"], markeredgecolor="black", markeredgewidth=1.5, markersize=8, alpha=0.9),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["patchtst"], markeredgecolor=COLORS["patchtst"], markersize=8, alpha=0.9),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=COLORS["baseline"], markeredgecolor=EDGES["baseline"], markeredgewidth=0.7, markersize=8, alpha=0.9),
    ]
    color_labels = ["TMTFNet (Ours)", "PatchTST", "Other baselines"]
    color_legend = ax.legend(
        color_handles,
        color_labels,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.98),
        frameon=False,
        fontsize=12,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    ax.add_artist(color_legend)

    size_handles, size_labels = build_size_handles()
    size_legend = ax.legend(
        size_handles,
        size_labels,
        loc="lower right",
        bbox_to_anchor=(1.01, 0.02),
        frameon=False,
        scatterpoints=1,
        fontsize=11.5,
        title="Bubble size = Std\n(\u2191 = less stable)",
        title_fontsize=12,
        labelspacing=1.2,
        handletextpad=1.0,
        borderaxespad=0.0,
    )
    ax.add_artist(size_legend)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    draw()
