#!/usr/bin/env python3
"""Generate publication figures for the NEDOQwen canonical technical report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "figure_data.json").read_text(encoding="utf-8"))

COLORS = {
    "reference": "#8C8C8C",
    "pre": "#4C78A8",
    "label": "#F2CF5B",
    "full": "#2A9D8F",
    "strong": "#E76F51",
}
LABEL_COLORS = {
    "A": "#4C78A8",
    "B": "#72B7B2",
    "C": "#F2CF5B",
    "D": "#E76F51",
    "E": "#B279A2",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ROOT / f"{stem}.pdf")
    fig.savefig(ROOT / f"{stem}.png", dpi=300)
    fig.savefig(ROOT / f"{stem}.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def figure_benchmark_accuracy() -> None:
    model_keys = [
        "Qwen2.5-0.5B",
        "NEDO pre-repair",
        "NEDO label-only",
        "NEDO full repair",
        "Qwen2.5-1.5B",
    ]
    display = ["Qwen\n0.5B", "NEDO\npre", "NEDO\nlabel-only", "NEDO\nfull", "Qwen\n1.5B"]
    colors = [COLORS["reference"], COLORS["pre"], COLORS["label"], COLORS["full"], COLORS["strong"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    for ax, (benchmark, values) in zip(axes, DATA["benchmarks"].items(), strict=True):
        scores = [values[key] for key in model_keys]
        x = np.arange(len(scores))
        bars = ax.bar(x, scores, color=colors, width=0.72, edgecolor="white", linewidth=0.7)
        ax.axhline(
            values["chance"],
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label=f"Chance ({values['chance']:.0f}%)",
        )
        ax.set_xticks(x, display)
        ax.set_title(benchmark, weight="bold")
        ax.set_ylim(0, 47)
        ax.set_ylabel("Accuracy (%)" if ax is axes[0] else "")
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, score in zip(bars, scores, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.7,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                weight="bold" if bar.get_facecolor()[:3] == mpl.colors.to_rgb(COLORS["full"]) else "normal",
            )
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.suptitle("NEDOQwen diagnosis-and-repair benchmark results", fontsize=14, weight="bold", y=1.03)
    fig.text(0.5, -0.03, "Zero-shot answer-label log-probability diagnostic protocol", ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    save_all(fig, "figure_1_benchmark_accuracy")


def figure_label_distributions() -> None:
    stages = ["NEDO pre-repair", "NEDO label-only", "NEDO full repair"]
    display = ["Pre-repair", "Label-only", "Full repair"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True)
    for ax, (benchmark, values) in zip(axes, DATA["label_distributions"].items(), strict=True):
        letters = list(next(iter(values.values())).keys())
        y = np.arange(len(stages))
        left = np.zeros(len(stages))
        for letter in letters:
            shares = np.array([values[stage][letter] for stage in stages])
            ax.barh(y, shares, left=left, color=LABEL_COLORS[letter], label=letter, height=0.58)
            for idx, (start, share) in enumerate(zip(left, shares, strict=True)):
                if share >= 7:
                    ax.text(start + share / 2, idx, f"{letter} {share:.1f}%", ha="center", va="center", fontsize=8, color="#1D1D1D")
            left += shares
        ax.set_yticks(y, display)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of predictions (%)")
        ax.set_title(benchmark, weight="bold")
        ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)

    handles = [mpl.patches.Patch(color=LABEL_COLORS[k], label=k) for k in ["A", "B", "C", "D", "E"]]
    fig.legend(handles=handles, title="Predicted label", loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("Repair reduces C-only collapse but leaves strong C/D skew", fontsize=14, weight="bold", y=1.18)
    fig.tight_layout()
    save_all(fig, "figure_2_label_distributions")


def figure_tokenizer_coverage() -> None:
    values = DATA["tokenizers_on_turkishmmlu_sub"]
    tokenizers = list(values.keys())
    tok_per_word = [values[name]["tokens_per_word"] for name in tokenizers]
    unk = [values[name]["unk_percent"] for name in tokenizers]
    colors = [COLORS["pre"], COLORS["reference"], "#72B7B2", "#B279A2"]
    x = np.arange(len(tokenizers))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7))
    bars = axes[0].bar(x, tok_per_word, color=colors, width=0.68)
    axes[0].set_title("Tokenization compactness", weight="bold")
    axes[0].set_ylabel("Tokens per word (lower is better)")
    axes[0].set_ylim(0, 3.45)
    axes[0].set_xticks(x, tokenizers, rotation=18, ha="right")
    axes[0].yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, tok_per_word, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.07, f"{value:.4f}", ha="center", fontsize=8.5)

    bars = axes[1].bar(x, unk, color=colors, width=0.68)
    axes[1].set_title("Unknown-token rate", weight="bold")
    axes[1].set_ylabel("UNK rate (%)")
    axes[1].set_ylim(0, 2.55)
    axes[1].set_xticks(x, tokenizers, rotation=18, ha="right")
    axes[1].yaxis.grid(True, color="#E6E6E6", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for bar, value in zip(bars, unk, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.4f}", ha="center", fontsize=8.5)

    fig.suptitle("Tokenizer coverage on TurkishMMLU-sub question and choice text", fontsize=13.5, weight="bold", y=1.03)
    fig.tight_layout()
    save_all(fig, "figure_3_tokenizer_coverage")


def main() -> None:
    configure_style()
    figure_benchmark_accuracy()
    figure_label_distributions()
    figure_tokenizer_coverage()


if __name__ == "__main__":
    main()
