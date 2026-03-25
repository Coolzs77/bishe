#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render publication-style plots for ablation evaluation CSV."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import AutoMinorLocator


ALIAS_MAP = {
    "ablation_exp01_baseline": "Exp01 Baseline",
    "ablation_exp02_ghost": "Exp02 Ghost",
    "ablation_exp03_shuffle": "Exp03 Shuffle",
    "ablation_exp04_attention": "Exp04 SEAttention",
    "ablation_exp05_coordatt": "Exp05 CoordAtt",
    "ablation_exp06_siou": "Exp06 SIoU",
    "ablation_exp07_eiou": "Exp07 EIoU",
    "ablation_exp08_ghost_attention": "Exp08 Ghost+SEAttention",
    "ablation_exp09_ghost_eiou": "Exp09 Ghost+EIoU",
    "ablation_exp10_attention_eiou": "Exp10 SEAttention+EIoU",
    "ablation_exp11_shuffle_coordatt": "Exp11 Shuffle+CoordAtt",
    "ablation_exp12_shuffle_coordatt_siou": "Exp12 Shuffle+CoordAtt+SIoU",
    "ablation_exp13_shuffle_coordatt_eiou": "Exp13 Shuffle+CoordAtt+EIoU",
}

METRIC_FIELDS = ["precision", "recall", "map50", "map5095"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP@0.5",
    "map5095": "mAP@0.5:0.95",
}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Plot config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_get(config: dict, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def pick(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)

    args.input_dir = pick(args.input_dir, config_get(config, "input", "dir"))
    args.csv = pick(args.csv, config_get(config, "input", "csv"))
    if args.csv is None and args.input_dir is not None:
        args.csv = str(Path(args.input_dir) / "summary.csv")

    args.output_dir = pick(args.output_dir, config_get(config, "output", "dir"))
    args.output_prefix = pick(args.output_prefix, config_get(config, "output", "prefix"))
    if args.output_dir is None and args.output_prefix is None and args.csv is not None:
        args.output_dir = str(Path(args.csv).parent)

    args.topk = pick(args.topk, config_get(config, "display", "topk", default=9))

    missing = []
    for name in ["csv", "topk"]:
        if getattr(args, name) is None:
            missing.append(name)
    if args.output_dir is None and args.output_prefix is None:
        missing.append("output_dir")
    if missing:
        raise ValueError(f"Missing plot config fields: {', '.join(missing)}")

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render publication-style ablation figures")
    parser.add_argument("--config", type=str, default="configs/plot_eval_summary.yaml", help="Plot config path")
    parser.add_argument("--input-dir", type=str, default=None, help="Batch evaluation directory containing summary.csv")
    parser.add_argument("--csv", type=str, default=None, help="Path to summary CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated figures")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Legacy output prefix, figures will be saved as *_journal_*.png",
    )
    parser.add_argument("--topk", type=int, default=None, help="Show first-k experiments in canonical order")
    return resolve_args(parser.parse_args())


def setup_journal_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "STSong", "SimSun", "Noto Serif CJK SC", "DejaVu Serif"]
    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["text.color"] = "#273444"
    mpl.rcParams["axes.labelcolor"] = "#273444"
    mpl.rcParams["axes.titlecolor"] = "#273444"
    mpl.rcParams["xtick.color"] = "#273444"
    mpl.rcParams["ytick.color"] = "#273444"

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "#fbfbf8"
    mpl.rcParams["savefig.facecolor"] = "white"

    mpl.rcParams["axes.linewidth"] = 1.0
    mpl.rcParams["xtick.major.width"] = 0.9
    mpl.rcParams["ytick.major.width"] = 0.9
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"


def simplify_exp_name(name: str) -> str:
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    m = re.search(r"exp(\d+)", name)
    if m:
        return f"Exp{m.group(1)}"
    return name


def legend_meaning_label(label: str) -> str:
    # Convert "Exp1 Baseline" -> "Baseline" for compact numeric legend.
    return re.sub(r"^Exp\d+\s*", "", label).strip()


def read_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        for k in METRIC_FIELDS + ["person_map5095", "car_map5095"]:
            r[k] = float(r[k])
        r["label"] = simplify_exp_name(r["exp_name"])
        m = re.search(r"exp(\d+)", r["exp_name"])
        r["exp_no"] = int(m.group(1)) if m else 999

    # Sort by exp_no (Exp1, Exp2, ..., Exp9)
    rows = sorted(rows, key=lambda r: r["exp_no"])
    return rows


def finalize_axes(ax, x, labels, y_min=None, y_max=None):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.40, alpha=0.14, color="#a8b0ba")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color="#a8b0ba")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.22, color="#a8b0ba")
    if y_min is not None and y_max is not None and y_max > y_min:
        pad = (y_max - y_min) * 0.12
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("Experiment", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5b6673")
    ax.spines["bottom"].set_color("#5b6673")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_output_path(output_dir: Path | None, output_prefix: Path | None, stem: str) -> Path:
    if output_prefix is not None:
        return output_prefix.with_name(output_prefix.name + f"_{stem}.png")
    assert output_dir is not None
    return output_dir / f"{stem}.png"


def _expand_bbox(fig, bbox, *, pad_x: float, pad_y: float):
    bbox_fig = bbox.transformed(fig.transFigure.inverted())
    x0 = max(0.015, bbox_fig.x0 - pad_x)
    y0 = max(0.015, bbox_fig.y0 - pad_y)
    x1 = min(0.985, bbox_fig.x1 + pad_x)
    y1 = min(0.985, bbox_fig.y1 + pad_y)
    return x0, y0, x1, y1


def _fit_text_to_width(fig, text_obj, max_width: float, initial_fontsize: float, min_fontsize: float = 6.0):
    fontsize = initial_fontsize
    text_obj.set_fontsize(fontsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = text_obj.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).width

    while width > max_width and fontsize > min_fontsize:
        fontsize = max(min_fontsize, fontsize - 0.2)
        text_obj.set_fontsize(fontsize)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width = text_obj.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).width

    return fontsize, width


def add_split_top_legends(
    fig,
    legend_handles,
    labels,
    legend_labels,
    *,
    legend_y: float = 0.918,
    experiment_y: float = 0.871,
    legend_fontsize: float = 8.3,
    experiment_fontsize: float = 8.3,
):
    legend_box_height = 0.030
    experiment_box_height = 0.040

    color_legend = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=max(1, len(legend_labels)),
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    experiment_line = "  ".join(
        f"{i + 1}-{legend_meaning_label(labels[i])}" for i in range(len(labels))
    )
    exp_text = fig.text(
        0.5,
        experiment_y,
        experiment_line,
        ha="center",
        va="center",
        fontsize=experiment_fontsize,
        color="#1f2937",
    )

    fig.canvas.draw()

    _fit_text_to_width(
        fig,
        exp_text,
        max_width=0.88,
        initial_fontsize=experiment_fontsize,
        min_fontsize=7.4,
    )

    renderer = fig.canvas.get_renderer()
    legend_bbox = color_legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    text_bbox = exp_text.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

    legend_box_width = min(0.92, legend_bbox.width + 0.030)
    exp_box_width = min(0.94, text_bbox.width + 0.028)
    legend_box_left = 0.5 - legend_box_width / 2
    exp_box_left = 0.5 - exp_box_width / 2

    legend_box_bottom = legend_y - legend_box_height / 2
    experiment_box_bottom = experiment_y - experiment_box_height / 2

    fig.add_artist(
        Rectangle(
            (legend_box_left, legend_box_bottom),
            legend_box_width,
            legend_box_height,
            transform=fig.transFigure,
            fill=False,
            linewidth=1.0,
            edgecolor="#94a3b8",
        )
    )
    fig.add_artist(
        Rectangle(
            (exp_box_left, experiment_box_bottom),
            exp_box_width,
            experiment_box_height,
            transform=fig.transFigure,
            fill=False,
            linewidth=1.0,
            edgecolor="#94a3b8",
        )
    )


def plot_journal_panel4(rows, output_dir: Path | None, output_prefix: Path | None) -> Path:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))

    # Colorblind-safe palette
    c_main = "#1f77b4"
    c_s1 = "#d62728"
    c_s2 = "#2ca02c"

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.0), dpi=220)
    all_vals = np.concatenate([np.array([r[m] for r in rows]) for m in METRIC_FIELDS])
    gmin, gmax = float(all_vals.min()), float(all_vals.max())

    for idx, metric in enumerate(METRIC_FIELDS):
        ax = axes[idx // 2, idx % 2]
        y = np.array([r[metric] for r in rows])

        # Main series: hollow markers, clean line
        ax.plot(
            x,
            y,
            color=c_main,
            linewidth=1.9,
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=METRIC_LABELS[metric],
            zorder=3,
        )

        # Stage split (subtle emphasis)
        s1 = np.where(np.array([r["exp_no"] for r in rows]) <= 6, y, np.nan)
        s2 = np.where(np.array([r["exp_no"] for r in rows]) >= 7, y, np.nan)
        ax.plot(x, s1, color=c_s1, linewidth=1.2, alpha=0.8, linestyle="-")
        ax.plot(x, s2, color=c_s2, linewidth=1.2, alpha=0.8, linestyle="-")

        # 仅标注终点，减少视觉干扰
        ax.text(x[-1] + 0.05, y[-1], f"{y[-1]:.4f}", fontsize=8, va="center", color="#334155")

        ax.set_title(f"({chr(97 + idx)}) {METRIC_LABELS[metric]}", fontsize=11.5, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=10)
        finalize_axes(ax, x, labels, y_min=gmin, y_max=gmax)

        # legend in-plot, placed at lower-left with translucent box
        ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.78, edgecolor="#94a3b8")

    fig.suptitle("Ablation Study Results (Publication Style)", fontsize=14, fontweight="bold", y=0.99)
    out = build_output_path(output_dir, output_prefix, "journal_panel4")
    save_fig(fig, out)
    return out


def plot_journal_multiline(rows, output_dir: Path | None, output_prefix: Path | None) -> Path:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))

    palette = {
        "precision": "#1f77b4",
        "recall": "#ff7f0e",
        "map50": "#2ca02c",
        "map5095": "#d62728",
    }
    markers = {"precision": "o", "recall": "s", "map50": "D", "map5095": "^"}

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=220)

    for m in METRIC_FIELDS:
        y = [r[m] for r in rows]
        ax.plot(
            x,
            y,
            color=palette[m],
            linewidth=1.9,
            marker=markers[m],
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=METRIC_LABELS[m],
        )

    # 去掉高亮阴影，保持全图信息权重一致

    ax.set_title("Multi-Metric Line Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=10)
    ymin = min([min([r[m] for r in rows]) for m in METRIC_FIELDS])
    ymax = max([max([r[m] for r in rows]) for m in METRIC_FIELDS])
    finalize_axes(ax, x, labels, y_min=float(ymin), y_max=float(ymax))

    # legend inside and non-blocking: upper right with semi-transparent box
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=8.2,
        frameon=True,
        framealpha=0.82,
        edgecolor="#94a3b8",
    )

    out = build_output_path(output_dir, output_prefix, "journal_multiline")
    save_fig(fig, out)
    return out


def plot_journal_barline(rows, output_dir: Path | None, output_prefix: Path | None) -> Path:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    map50 = np.array([r["map50"] for r in rows])
    map95 = np.array([r["map5095"] for r in rows])

    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=220)

    # Academic bar+line hybrid: bars for map50, line for map50-95
    bars = ax.bar(
        x,
        map50,
        width=0.58,
        color="#dbeafe",
        edgecolor="#3b82f6",
        linewidth=1.0,
        hatch="//",
        label="mAP@0.5",
        zorder=2,
    )

    ax.plot(
        x,
        map95,
        color="#ef4444",
        linewidth=1.9,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="mAP@0.5:0.95",
        zorder=3,
    )

    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.001, f"{h:.3f}", ha="center", va="bottom", fontsize=7.6)

    ax.set_title("Key Metric Hybrid Plot", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=10)
    y_min = min(float(map95.min()), float(map50.min()))
    y_max = max(float(map95.max()), float(map50.max()))
    finalize_axes(ax, x, labels, y_min=y_min, y_max=y_max)
    ax.legend(loc="upper right", fontsize=8.2, frameon=True, framealpha=0.82, edgecolor="#94a3b8")

    out = build_output_path(output_dir, output_prefix, "journal_barline")
    save_fig(fig, out)
    return out


def plot_improvement_comparison(rows, output_dir: Path | None, output_prefix: Path | None) -> Path:
    """Show improvement (%) relative to baseline (Exp1) across key metrics."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    
    # Find baseline (Exp1)
    baseline = next((r for r in rows if r["exp_no"] == 1), None)
    if not baseline:
        print("Warning: Baseline (Exp1) not found, skipping improvement plot")
        return None
    
    # Calculate improvement % for each metric: (exp_metric - baseline_metric) / baseline_metric * 100
    improvements = {
        "map5095": [],
        "map50": [],
        "precision": [],
        "recall": []
    }
    
    for r in rows:
        for metric in improvements.keys():
            baseline_val = baseline[metric]
            current_val = r[metric]
            improvement_pct = ((current_val - baseline_val) / baseline_val) * 100
            improvements[metric].append(improvement_pct)
    
    # Create 2x2 subplot for improvements
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 8.8), dpi=220)
    
    metric_order = ["precision", "recall", "map50", "map5095"]
    cmap_positive = "#22c55e"  # Green (Improved)
    cmap_negative = "#ef4444"  # Red (Declined)
    cmap_baseline = "#8b5cf6"  # Purple for Exp1
    
    for idx, metric in enumerate(metric_order):
        ax = axes[idx // 2, idx % 2]
        y = np.array(improvements[metric])
        
        # Color based on improvement/decline
        colors = []
        for i, val in enumerate(y):
            if rows[i]["exp_no"] == 1:  # Baseline
                colors.append(cmap_baseline)
            elif val > 0:
                colors.append(cmap_positive)
            else:
                colors.append(cmap_negative)
        
        # Bar chart for improvements
        bars = ax.bar(x, y, width=0.65, color=colors, edgecolor="#334155", linewidth=0.9, alpha=0.40)
        
        # Add value labels on all bars
        for i, (bar, val) in enumerate(zip(bars, y)):
            height = bar.get_height()
            if height > 0:
                va_align = "bottom"
                y_pos = height + 0.03
            else:
                va_align = "top"
                y_pos = height - 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}%", 
                   ha="center", va=va_align, fontsize=9.5, color="#334155")
        
        # Horizontal line at y=0 (baseline reference)
        ax.axhline(y=0, color="#64748b", linewidth=0.8, linestyle="-", alpha=0.5)
        
        ax.set_ylabel("Improvement (%)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(rows) + 1)], rotation=0, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_facecolor("#fbfbf8")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color="#a8b0ba")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.22, color="#a8b0ba")
        
        # Set y-axis limits with independent padding for each subplot
        local_min, local_max = float(y.min()), float(y.max())
        pad = (local_max - local_min) * 0.15 if local_max != local_min else abs(local_max) * 0.15
        ax.set_ylim(local_min - pad, local_max + pad)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#5b6673")
        ax.spines["bottom"].set_color("#5b6673")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

        # Put subplot label below each subplot.
        ax.text(
            0.5,
            -0.20,
            f"({chr(97 + idx)}) {METRIC_LABELS[metric]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
    
    legend_handles = [
        Patch(facecolor=cmap_positive, edgecolor="#334155", alpha=0.40, label="Improved"),
        Patch(facecolor=cmap_negative, edgecolor="#334155", alpha=0.40, label="Declined"),
    ]
    add_split_top_legends(
        fig,
        legend_handles,
        labels,
        ["Improved", "Declined"],
        legend_y=0.924,
        experiment_y=0.879,
        legend_fontsize=9.2,
        experiment_fontsize=9.2,
    )

    fig.suptitle(
        "消融实验相对基线改变量对比",
        fontsize=14,
        fontweight="bold",
        y=0.982,
        fontfamily="sans-serif",
    )
    fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.38, wspace=0.20)
    
    out = build_output_path(output_dir, output_prefix, "journal_improvement")
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_comparison_with_baseline(rows, output_dir: Path | None, output_prefix: Path | None) -> Path:
    """Show all metrics with baseline reference line and up/down highlighting."""
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    
    # Find baseline (Exp1)
    baseline = next((r for r in rows if r["exp_no"] == 1), None)
    if not baseline:
        print("Warning: Baseline (Exp1) not found, skipping metric comparison plot")
        return None
    
    metric_order = ["precision", "recall", "map50", "map5095"]
    all_vals = np.concatenate([np.array([r[m] for r in rows]) for m in metric_order])
    global_min, global_max = float(all_vals.min()), float(all_vals.max())
    global_pad = (global_max - global_min) * 0.15 if global_max != global_min else abs(global_max) * 0.15
    fixed_ymin, fixed_ymax = global_min - global_pad, global_max + global_pad
    
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 8.8), dpi=220)
    
    for idx, metric in enumerate(metric_order):
        ax = axes[idx // 2, idx % 2]
        y = np.array([r[metric] for r in rows])
        baseline_val = baseline[metric]
        
        # Draw baseline as dashed line across entire plot.
        ax.axhline(y=baseline_val, color="#64748b", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)
        
        # Color and marker size based on improvement vs decline
        colors = []
        sizes = []
        for val in y:
            if val > baseline_val:
                colors.append("#22c55e")  # Green for improvement
                sizes.append(6.5)
            elif val < baseline_val:
                colors.append("#ef4444")  # Red for decline
                sizes.append(6.5)
            else:
                colors.append("#8b5cf6")  # Purple for equal
                sizes.append(6.5)
        
        # Plot each point with corresponding color
        for xi, (yi, color, size) in enumerate(zip(y, colors, sizes)):
            ax.plot(xi, yi, color=color, marker="o", markersize=size, 
                   markerfacecolor="white", markeredgewidth=1.2, zorder=3)
        
        # Connect with line (light gray)
        ax.plot(x, y, color="#cbd5e1", linewidth=1.5, alpha=0.6, zorder=1)
        
        # Add value labels on all points
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi + 0.005, f"{yi:.3f}", ha="center", va="bottom", 
                   fontsize=9.5, color="#334155")
        
        ax.set_ylabel("Value", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(1, len(rows) + 1)], rotation=0, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_facecolor("#fbfbf8")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color="#a8b0ba")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.22, color="#a8b0ba")
        
        # Keep a fixed shared y-range across all subplots.
        ax.set_ylim(fixed_ymin, fixed_ymax)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#5b6673")
        ax.spines["bottom"].set_color("#5b6673")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

        # Put subplot label below each subplot.
        ax.text(
            0.5,
            -0.20,
            f"({chr(97 + idx)}) {METRIC_LABELS[metric]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    legend_handles = [
        Line2D([0], [0], color="#22c55e", marker="o", markerfacecolor="white", markeredgewidth=1.2, linewidth=0, label="Improved"),
        Line2D([0], [0], color="#ef4444", marker="o", markerfacecolor="white", markeredgewidth=1.2, linewidth=0, label="Declined"),
        Line2D([0], [0], color="#8b5cf6", marker="o", markerfacecolor="white", markeredgewidth=1.2, linewidth=0, label="Baseline (Exp1)"),
        Line2D([0], [0], color="#64748b", linestyle="--", linewidth=1.2, label="Baseline Reference"),
    ]
    add_split_top_legends(
        fig,
        legend_handles,
        labels,
        ["Improved", "Declined", "Baseline (Exp1)", "Baseline Reference"],
        legend_y=0.924,
        experiment_y=0.879,
        legend_fontsize=9.0,
        experiment_fontsize=9.0,
    )

    fig.suptitle(
        "四项指标变化与基线对比",
        fontsize=14,
        fontweight="bold",
        y=0.982,
        fontfamily="sans-serif",
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.38, wspace=0.20)
    
    out = build_output_path(output_dir, output_prefix, "journal_metric_change")
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    setup_journal_style()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_prefix = Path(args.output_prefix) if args.output_prefix else None

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError("CSV is empty")

    # 按实验顺序绘图（Exp1-Exp9），不排序
    rows = rows[: max(1, args.topk)]

    p3 = plot_journal_barline(rows, output_dir, output_prefix)
    p4 = plot_improvement_comparison(rows, output_dir, output_prefix)
    p5 = plot_metric_comparison_with_baseline(rows, output_dir, output_prefix)

    print(f"SAVED: {p3}")
    if p4:
        print(f"SAVED: {p4}")
    if p5:
        print(f"SAVED: {p5}")


if __name__ == "__main__":
    main()
