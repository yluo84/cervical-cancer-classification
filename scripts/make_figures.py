"""Regenerate the figures in results/figures/ from the saved summary JSONs.

    python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path("results")
FIGURES = RESULTS / "figures"

CLASSES = ["Normal", "Precancerous", "Cancerous"]

# Categorical slots 1 and 2; validated for CVD separation on a light surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# Single-hue sequential ramp, light to dark, for the confusion matrices.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
)

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": "#d8d7d2",
    "axes.linewidth": 0.8,
})


def load(name):
    return json.loads((RESULTS / name / "summary.json").read_text())


def fig_comparison(baseline, effnet):
    """Point estimates with 95% bootstrap intervals, one row per metric."""
    metrics = [
        ("Macro-F1", "test_macro_f1", "test_macro_f1_ci95"),
        ("Accuracy", "test_accuracy", "test_accuracy_ci95"),
    ]
    models = [("EfficientNet-B0 (transfer)", effnet, BLUE),
              ("SmallCNN (from scratch)", baseline, ORANGE)]

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    offsets = [0.17, -0.17]

    for row, (_, key, ci_key) in enumerate(metrics):
        for (label, data, color), off in zip(models, offsets):
            y = row + off
            lo, hi = data[ci_key]
            point = data[key]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2,
                    solid_capstyle="round",
                    label=label if row == 0 else None, zorder=2)
            ax.plot([point], [y], "o", color=color, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.text(hi + 0.012, y, f"{point:.2f}", va="center", ha="left",
                    fontsize=9, color=MUTED)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlim(0.15, 0.88)
    ax.set_ylim(-0.55, len(metrics) - 0.45)
    ax.set_xlabel("Score on the held-out test split (point estimate and 95% bootstrap CI)")
    ax.grid(axis="x", color="#eceae4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=9, handlelength=1.6,
              columnspacing=2.0, borderpad=0)
    ax.set_title("Transfer learning versus a from-scratch baseline (n = 81 test images)",
                 loc="left", fontsize=11, color=INK, pad=34)

    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(baseline, effnet):
    """Row-normalised confusion matrices with raw counts annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))

    for ax, (title, data) in zip(axes, [
        ("SmallCNN (from scratch)", baseline),
        ("EfficientNet-B0 (transfer)", effnet),
    ]):
        cm = np.array(data["confusion_matrix"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)

        ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}",
                        ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)

        ax.set_xticks(range(3), CLASSES, fontsize=9)
        ax.set_yticks(range(3), CLASSES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Confusion matrices: counts and row-normalised recall",
                 fontsize=11, color=INK, x=0.015, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    baseline = load("smallcnn")
    effnet = load("efficientnet_b0")
    fig_comparison(baseline, effnet)
    fig_confusion(baseline, effnet)
    print(f"Wrote figures to {FIGURES}/")


if __name__ == "__main__":
    main()
