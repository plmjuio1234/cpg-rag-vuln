#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT = ROOT / "paper_draft_cpg_llm_ko.md"
DEFAULT_OUTPUT_DIR = ROOT / "paper_figures"

FIGSIZE = (5.2, 3.5)
PNG_DPI = 300
BAR_EDGE = "#111111"
GRID = "#C8C8C8"
TEXT = "#111111"
WHITE = "#FFFFFF"
LIGHT = "#D9D9D9"
MID = "#BFBFBF"


@dataclass(frozen=True)
class CliArgs:
    draft: Path
    output_dir: Path


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.edgecolor": BAR_EDGE,
            "axes.linewidth": 1.3,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
        }
    )


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate PNG charts for paper figures 2-4."
    )
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    namespace = parser.parse_args()
    return CliArgs(draft=namespace.draft, output_dir=namespace.output_dir)


def get_pipe_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        rows.append(parts)
    return rows


def find_row_value(
    rows: Sequence[Sequence[str]], row_label: str, label_index: int, value_index: int
) -> tuple[float, str]:
    for row in rows:
        if len(row) <= max(label_index, value_index):
            continue
        if row[label_index] == row_label:
            raw = row[value_index]
            return float(raw), raw
    raise ValueError(f"Missing row '{row_label}'.")


def annotate_bars(
    ax: plt.Axes, bars, labels: Sequence[str], fontsize: float = 10.5
) -> None:
    for bar, label in zip(bars, labels, strict=True):
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT,
        )


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11.0, width=1.1, pad=6)
    ax.tick_params(axis="y", labelsize=11.0, width=1.1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def draw_single_series(
    output_path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    value_labels: Sequence[str],
    hatches: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = range(len(labels))
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=[WHITE, LIGHT, MID, WHITE, LIGHT][: len(labels)],
        edgecolor=BAR_EDGE,
        linewidth=1.4,
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)

    ax.set_xticks(list(x), labels)
    style_axes(ax, "F1 Score")
    annotate_bars(ax, bars, value_labels)
    fig.suptitle(title, y=0.98, fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.24)
    fig.savefig(output_path, dpi=PNG_DPI, format="png", bbox_inches="tight")
    plt.close(fig)


def draw_grouped_series(
    output_path: Path,
    title: str,
    labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    left_labels: Sequence[str],
    right_labels: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = list(range(len(labels)))
    width = 0.32
    bars_left = ax.bar(
        [v - width / 2 for v in x],
        left_values,
        width=width,
        color=WHITE,
        edgecolor=BAR_EDGE,
        linewidth=1.4,
        hatch="///",
        label="This study",
    )
    bars_right = ax.bar(
        [v + width / 2 for v in x],
        right_values,
        width=width,
        color=LIGHT,
        edgecolor=BAR_EDGE,
        linewidth=1.4,
        hatch="..",
        label="Prior best",
    )

    ax.set_xticks(x, labels)
    style_axes(ax, "F1 Score")
    annotate_bars(ax, bars_left, left_labels, fontsize=9.2)
    annotate_bars(ax, bars_right, right_labels, fontsize=9.2)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        handlelength=1.4,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.suptitle(title, y=0.98, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.30)
    fig.savefig(output_path, dpi=PNG_DPI, format="png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    draft_text = args.draft.read_text(encoding="utf-8")
    rows = get_pipe_rows(draft_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig2_labels = [
        "LLM-only",
        "GNN-only",
        "GNN +\nCodeBERT",
        "GNN +\nretrieval",
        "Full\nmodel",
    ]
    fig2_source = [
        "LLM-only",
        "GNN-only",
        "GNN + CodeBERT",
        "GNN + retrieval",
        "Full CPG/GNN-LLM model",
    ]
    fig2_values = []
    fig2_value_labels = []
    for label in fig2_source:
        value, raw = find_row_value(rows, label, label_index=1, value_index=2)
        fig2_values.append(value)
        fig2_value_labels.append(raw)
    draw_single_series(
        args.output_dir / "Figure_2.png",
        "Stagewise F1 Comparison",
        fig2_labels,
        fig2_values,
        fig2_value_labels,
        ["///", "\\\\", "xx", "..", "--"],
    )

    internal, internal_raw = find_row_value(
        rows, "CPG/GNN-LLM model", label_index=0, value_index=3
    )
    juliet, juliet_raw = find_row_value(rows, "Juliet", label_index=0, value_index=1)
    devign, devign_raw = find_row_value(rows, "Devign", label_index=0, value_index=1)
    draw_single_series(
        args.output_dir / "Figure_3.png",
        "Best F1 Across Evaluations",
        ["PrimeVul\nFull", "Juliet", "Devign"],
        [internal, juliet, devign],
        [internal_raw, juliet_raw, devign_raw],
        ["///", "xx", ".."],
    )

    juliet_prior, juliet_prior_raw = find_row_value(
        rows, "Juliet", label_index=0, value_index=2
    )
    devign_prior, devign_prior_raw = find_row_value(
        rows, "Devign", label_index=0, value_index=2
    )
    draw_grouped_series(
        args.output_dir / "Figure_4.png",
        "F1 vs Prior LLM",
        ["Juliet", "Devign"],
        [juliet, devign],
        [juliet_prior, devign_prior],
        [juliet_raw, devign_raw],
        [juliet_prior_raw, devign_prior_raw],
    )


if __name__ == "__main__":
    main()
