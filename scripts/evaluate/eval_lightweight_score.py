#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute lightweight-aware composite scores from detection batch CSV."""

import argparse
import csv
import math
import pathlib
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "yolov5") not in sys.path:
    sys.path.insert(0, str(ROOT / "yolov5"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight-aware ablation scoring")
    parser.add_argument("--csv", type=str, required=True, help="Input detection summary CSV")
    parser.add_argument("--output-prefix", type=str, default="outputs/results/lightweight_eval", help="Output prefix")
    parser.add_argument("--baseline-exp", type=int, default=1, help="Baseline experiment number")
    parser.add_argument("--beta", type=float, default=0.5, help="Lightweight emphasis exponent")
    parser.add_argument("--w-map", type=float, default=0.7, help="Weight of mAP@0.5 in performance retention")
    parser.add_argument("--w-recall", type=float, default=0.2, help="Weight of Recall in performance retention")
    parser.add_argument("--w-precision", type=float, default=0.1, help="Weight of Precision in performance retention")
    parser.add_argument(
        "--map-threshold",
        type=float,
        default=0.75,
        help="Absolute mAP threshold for qualification (e.g., 0.75 means 75%)",
    )
    parser.add_argument(
        "--map-threshold-metric",
        type=str,
        default="map50",
        choices=["map50", "map5095"],
        help="Metric used by threshold gating",
    )
    parser.add_argument(
        "--pareto-metric",
        type=str,
        default="map50",
        choices=["map50", "map5095"],
        help="Y-axis metric used in Pareto plot",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    # Robust Chinese font fallback on Windows and common dev environments.
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "#fcfcfd"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def parse_exp_no(exp_name: str) -> int:
    m = re.search(r"exp(\d+)", exp_name)
    return int(m.group(1)) if m else 999


def simplify_name(exp_name: str) -> str:
    m = re.search(r"exp(\d+)", exp_name)
    if not m:
        return exp_name
    exp_no = int(m.group(1))
    tail = exp_name.split("_", maxsplit=2)[-1]
    alias = {
        1: "Baseline",
        2: "Ghost",
        3: "Shuffle",
        4: "Attention",
        5: "CoordAtt",
        6: "SIoU",
        7: "EIoU",
        8: "Ghost+Attention",
        9: "Ghost+EIoU",
        10: "Attention+EIoU",
        11: "Shuffle+CoordAtt",
        12: "Shuffle+CoordAtt+SIoU",
        13: "Shuffle+CoordAtt+EIoU",
    }.get(exp_no, tail)
    return f"Exp{exp_no} {alias}"


def count_params_from_weight(weight_path: Path) -> int:
    if not weight_path.exists():
        return -1

    # Cross-platform checkpoint compatibility: map PosixPath only on Windows.
    orig_posix = pathlib.PosixPath
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(str(weight_path), map_location="cpu")
    finally:
        pathlib.PosixPath = orig_posix
    model = None

    if isinstance(ckpt, dict):
        if "ema" in ckpt and ckpt["ema"] is not None:
            model = ckpt["ema"]
        elif "model" in ckpt:
            model = ckpt["model"]
    else:
        model = ckpt

    if model is None or not hasattr(model, "parameters"):
        return -1

    model = model.float()
    return int(sum(p.numel() for p in model.parameters()))


def read_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for r in rows:
        exp_no = parse_exp_no(r["exp_name"])
        parsed.append(
            {
                "exp_no": exp_no,
                "exp_name": r["exp_name"],
                "label": simplify_name(r["exp_name"]),
                "weights": r["weights"],
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "map50": float(r["map50"]),
                "map5095": float(r["map5095"]),
            }
        )

    parsed.sort(key=lambda x: x["exp_no"])
    return parsed


def compute_scores(
    rows,
    baseline_exp: int,
    beta: float,
    w_map: float,
    w_recall: float,
    w_precision: float,
    map_threshold: float,
    map_threshold_metric: str,
):
    baseline = next((r for r in rows if r["exp_no"] == baseline_exp), None)
    if baseline is None:
        raise RuntimeError(f"Baseline Exp{baseline_exp} not found in CSV")

    for r in rows:
        weight_path = Path(r["weights"])
        r["params"] = count_params_from_weight(weight_path)

    if baseline["params"] <= 0:
        raise RuntimeError("Baseline params cannot be read. Please check baseline weight file.")

    sum_w = w_map + w_recall + w_precision
    if sum_w <= 0:
        raise RuntimeError("Weights sum must be > 0")
    w_map /= sum_w
    w_recall /= sum_w
    w_precision /= sum_w

    for r in rows:
        if r["params"] <= 0:
            r["param_ratio_vs_base"] = math.nan
            r["param_reduction_pct"] = math.nan
            r["perf_retention"] = math.nan
            r["efficiency_gain"] = math.nan
            r["composite_score"] = math.nan
            continue

        p_ret = (
            w_map * (r["map50"] / baseline["map50"])
            + w_recall * (r["recall"] / baseline["recall"])
            + w_precision * (r["precision"] / baseline["precision"])
        )
        e_gain = (baseline["params"] / r["params"]) ** beta

        r["param_ratio_vs_base"] = r["params"] / baseline["params"]
        r["param_reduction_pct"] = (1.0 - r["params"] / baseline["params"]) * 100.0
        r["perf_retention"] = p_ret
        r["efficiency_gain"] = e_gain
        r["composite_score"] = p_ret * e_gain
        r["pass_map_threshold"] = r[map_threshold_metric] >= map_threshold
        r["qualified_score"] = r["composite_score"] if r["pass_map_threshold"] else math.nan

    return rows


def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "exp_no",
        "label",
        "precision",
        "recall",
        "map50",
        "map5095",
        "params",
        "param_ratio_vs_base",
        "param_reduction_pct",
        "perf_retention",
        "efficiency_gain",
        "composite_score",
        "pass_map_threshold",
        "qualified_score",
        "weights",
    ]

    rankable = [r for r in rows if not math.isnan(r["qualified_score"])]
    rankable.sort(key=lambda x: x["qualified_score"], reverse=True)
    rank_map = {r["exp_no"]: idx + 1 for idx, r in enumerate(rankable)}

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "rank": rank_map.get(r["exp_no"], "NA"),
                    "exp_no": r["exp_no"],
                    "label": r["label"],
                    "precision": f"{r['precision']:.6f}",
                    "recall": f"{r['recall']:.6f}",
                    "map50": f"{r['map50']:.6f}",
                    "map5095": f"{r['map5095']:.6f}",
                    "params": r["params"],
                    "param_ratio_vs_base": f"{r['param_ratio_vs_base']:.6f}" if not math.isnan(r["param_ratio_vs_base"]) else "NA",
                    "param_reduction_pct": f"{r['param_reduction_pct']:.4f}" if not math.isnan(r["param_reduction_pct"]) else "NA",
                    "perf_retention": f"{r['perf_retention']:.6f}" if not math.isnan(r["perf_retention"]) else "NA",
                    "efficiency_gain": f"{r['efficiency_gain']:.6f}" if not math.isnan(r["efficiency_gain"]) else "NA",
                    "composite_score": f"{r['composite_score']:.6f}" if not math.isnan(r["composite_score"]) else "NA",
                    "pass_map_threshold": "YES" if r.get("pass_map_threshold", False) else "NO",
                    "qualified_score": f"{r['qualified_score']:.6f}" if not math.isnan(r.get("qualified_score", math.nan)) else "NA",
                    "weights": r["weights"],
                }
            )


def plot_lightweight_panel(rows, out_path: Path):
    """Generate 1x2 panel: composite score and absolute parameters with incremental coloring."""
    valid = [r for r in rows if r["params"] > 0]
    if not valid:
        return

    x = np.arange(len(valid))
    labels = [str(r["exp_no"]) for r in valid]
    
    # Find baseline
    base_row = next((r for r in valid if r["exp_no"] == 1), None)
    if not base_row:
        return
    
    composite_scores = np.array([r.get("composite_score", math.nan) for r in valid])
    params_m = np.array([r["params"] / 1e6 for r in valid])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), dpi=220)

    metrics = [
        ("Composite Score", composite_scores, 1.0),
        ("Parameters (M)", params_m, base_row["params"] / 1e6),
    ]

    cmap_baseline = "#8b5cf6"
    cmap_base_part = "#cbd5e1"
    cmap_improved = "#22c55e"
    cmap_declined = "#ef4444"

    for idx, (metric_label, values, base_value) in enumerate(metrics):
        ax = axes[idx]
        valid_vals = values[~np.isnan(values)]
        y_span = max(float(valid_vals.max()) - float(valid_vals.min()), 1e-6)

        base_part = np.minimum(values, base_value)
        base_colors = [cmap_baseline if r["exp_no"] == 1 else cmap_base_part for r in valid]
        ax.bar(
            x,
            base_part,
            width=0.64,
            color=base_colors,
            edgecolor="#475569",
            linewidth=0.95,
            linestyle="--",
            hatch="//",
            alpha=0.52,
            zorder=1,
        )

        for i, (xi, val) in enumerate(zip(x, values)):
            if math.isnan(val):
                continue
            delta = val - base_value
            # Ignore tiny floating-point noise to avoid fake up/down increments near baseline.
            if abs(delta) < 5e-4:
                continue
            if delta > 0:
                bottom = base_value
                height = delta
                color = cmap_improved
            else:
                bottom = val
                height = -delta
                color = cmap_declined
            inc_hatch = "//" if delta > 0 else None
            ax.bar(
                xi,
                height,
                bottom=bottom,
                width=0.64,
                color=color,
                edgecolor="#334155",
                linewidth=0.75,
                alpha=0.18 if delta > 0 else 0.24,
                hatch=inc_hatch,
                zorder=2,
            )

            # Direction arrow for increment segment (up/down relative to baseline).
            arrow_eps = 0.01 if metric_label == "Composite Score" else 5e-4
            if valid[i]["exp_no"] != 1 and abs(delta) > arrow_eps:
                # Use short, thin arrows to indicate increment direction without cluttering labels.
                arrow_end = val - y_span * 0.012 if delta > 0 else val + y_span * 0.012
                arrow_start = base_value + delta * 0.38
                ax.annotate(
                    "",
                    xy=(xi, arrow_end),
                    xytext=(xi, arrow_start),
                    arrowprops={
                        "arrowstyle": "->",
                        "lw": 0.8,
                        "color": color,
                        "alpha": 0.80,
                        "shrinkA": 1,
                        "shrinkB": 1,
                    },
                    zorder=4,
                )

        ax.axhline(y=base_value, color="#64748b", linewidth=1.0, linestyle="--", alpha=0.85, zorder=3)

        for i, (xi, val) in enumerate(zip(x, values)):
            if math.isnan(val):
                continue
            ax.text(xi, val + y_span * 0.045, f"{val:.2f}", ha="center", va="bottom", fontsize=7.3, color="#334155")

        local_max = max(float(np.nanmax(values)), base_value)
        pad = local_max * 0.08 if local_max > 0 else 0.1
        if metric_label == "Composite Score":
            ax.set_ylim(0.5, local_max + pad)
        else:
            ax.set_ylim(0.0, local_max + pad)

        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_facecolor("#fcfcfd")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color="#a8b0ba")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.22, color="#a8b0ba")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#5b6673")
        ax.spines["bottom"].set_color("#5b6673")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

        subplot_name = "Composite Score" if idx == 0 else "Parameters (M)"
        ax.text(
            0.5,
            -0.18,
            f"({chr(97 + idx)}) {subplot_name}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle("轻量化模型增量评估", fontsize=14.5, fontweight="bold", y=0.992)

    legend_handles = [
        Patch(facecolor=cmap_base_part, edgecolor="#475569", hatch="//", alpha=0.52, label="Initial Value (Hatched)"),
        Patch(facecolor=cmap_improved, edgecolor="#334155", hatch="//", alpha=0.18, label="Improved"),
        Patch(facecolor=cmap_declined, edgecolor="#334155", alpha=0.24, label="Declined"),
        Patch(facecolor=cmap_baseline, edgecolor="#334155", alpha=0.52, label="Baseline (Exp1)"),
        Line2D([0], [0], color="#64748b", linestyle="--", linewidth=1.0, label="Baseline Reference"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        frameon=True,
        framealpha=0.82,
        edgecolor="#94a3b8",
        fontsize=8.2,
    )

    idx_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            label=f"{r['exp_no']}={r['label'].split(' ', 1)[1] if ' ' in r['label'] else r['label']}",
        )
        for r in valid
    ]
    fig.legend(
        handles=idx_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=9,
        frameon=True,
        framealpha=0.80,
        edgecolor="#94a3b8",
        fontsize=7.7,
        handlelength=0,
        handletextpad=0.1,
        columnspacing=0.8,
    )

    # Single-line compact equation for publication layout.
    eq_line = r"$S_{\mathrm{comp}}=\left[w_m\left(\frac{m_{50,i}}{m_{50,b}}\right)+w_r\left(\frac{r_i}{r_b}\right)+w_p\left(\frac{p_i}{p_b}\right)\right]\cdot\left(\frac{N_b}{N_i}\right)^{\beta}$"
    eq_desc = (
        r"$w_m,w_r,w_p$ 为权重；$m_{50,i},r_i,p_i$ 为当前实验指标；"
        r"$m_{50,b},r_b,p_b$ 为基线指标；$N_i,N_b$ 为参数量；$\beta$ 为轻量化系数"
    )
    fig.text(0.5, 0.034, eq_line, ha="center", va="bottom", fontsize=11.4, color="#1f2937")
    fig.text(0.5, 0.011, eq_desc, ha="center", va="bottom", fontsize=8.6, color="#475569")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.22, top=0.79, wspace=0.18)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_composite(rows, out_path: Path):
    valid = [r for r in rows if not math.isnan(r["composite_score"])]
    if not valid:
        return

    x = np.arange(len(valid))
    labels = [str(r["exp_no"]) for r in valid]
    y = np.array([r["composite_score"] for r in valid])

    fig, ax = plt.subplots(figsize=(10.0, 6.8), dpi=220)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.30)

    colors = []
    for r, v in zip(valid, y):
        if not r.get("pass_map_threshold", False):
            colors.append("#9ca3af")
        else:
            colors.append("#22c55e" if v >= 1.0 else "#ef4444")
    bars = ax.bar(x, y, width=0.62, color=colors, edgecolor="#334155", linewidth=0.9)

    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, linestyle="--", color="#64748b", linewidth=1.1)
    ax.set_title("轻量化综合分对比（>1 优于基线）", fontsize=13.5, fontweight="bold")
    ax.set_xlabel("Experiment Index", fontsize=10)
    ax.set_ylabel("Composite Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    legend_handles = [
        Line2D([0], [0], color="#22c55e", marker="s", linewidth=0, markersize=8, label="Qualified & Score>=1"),
        Line2D([0], [0], color="#ef4444", marker="s", linewidth=0, markersize=8, label="Qualified & Score<1"),
        Line2D([0], [0], color="#9ca3af", marker="s", linewidth=0, markersize=8, label="Disqualified by mAP threshold"),
        Line2D([0], [0], color="#64748b", linestyle="--", linewidth=1.1, label="Baseline Score = 1"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.4, frameon=True, framealpha=0.84, edgecolor="#94a3b8")

    legend_text = [
        "1=Baseline, 2=Ghost, 3=Attention, 4=EIoU, 5=Focal,",
        "6=Ghost+Attn, 7=Ghost+EIoU, 8=Attn+EIoU, 9=All",
        "Gray bar: failed mAP threshold",
    ]
    fig.text(0.5, 0.01, "\n".join(legend_text), ha="center", va="bottom", fontsize=8.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.19)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_score_components(rows, out_path: Path):
    valid = [r for r in rows if not math.isnan(r.get("perf_retention", math.nan))]
    if not valid:
        return

    x = np.arange(len(valid))
    labels = [str(r["exp_no"]) for r in valid]
    perf = np.array([r["perf_retention"] for r in valid])
    eff = np.array([r["efficiency_gain"] for r in valid])

    fig, ax = plt.subplots(figsize=(10.0, 6.8), dpi=220)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.30)

    ax.plot(x, perf, color="#2563eb", marker="o", markersize=5.8, markerfacecolor="white", markeredgewidth=1.1, linewidth=1.9, label="Performance Retention")
    ax.plot(x, eff, color="#f59e0b", marker="s", markersize=5.6, markerfacecolor="white", markeredgewidth=1.1, linewidth=1.9, label="Efficiency Gain")
    ax.axhline(1.0, linestyle="--", color="#64748b", linewidth=1.1, label="Baseline Level")

    for xi, p, e in zip(x, perf, eff):
        ax.text(xi, p + 0.01, f"{p:.3f}", ha="center", va="bottom", fontsize=8, color="#1e3a8a")
        ax.text(xi, e - 0.015, f"{e:.3f}", ha="center", va="top", fontsize=8, color="#92400e")

    ax.set_title("轻量化综合分分量解析", fontsize=13.5, fontweight="bold")
    ax.set_xlabel("Experiment Index", fontsize=10)
    ax.set_ylabel("Normalized Value", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.84, edgecolor="#94a3b8")

    legend_text = [
        "1=Baseline, 2=Ghost, 3=Attention, 4=EIoU, 5=Focal,",
        "6=Ghost+Attn, 7=Ghost+EIoU, 8=Attn+EIoU, 9=All",
    ]
    fig.text(0.5, 0.01, "\n".join(legend_text), ha="center", va="bottom", fontsize=8.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_plot_style()

    csv_path = Path(args.csv)
    out_prefix = Path(args.output_prefix)

    rows = read_rows(csv_path)
    rows = compute_scores(
        rows,
        args.baseline_exp,
        args.beta,
        args.w_map,
        args.w_recall,
        args.w_precision,
        args.map_threshold,
        args.map_threshold_metric,
    )

    out_csv = out_prefix.with_name(out_prefix.name + "_score_table.csv")
    out_panel = out_prefix.with_name(out_prefix.name + "_panel.png")

    save_csv(rows, out_csv)
    plot_lightweight_panel(rows, out_panel)

    print(f"SAVED: {out_csv}")
    print(f"SAVED: {out_panel}")
    print(f"THRESHOLD: {args.map_threshold_metric} >= {args.map_threshold:.2f}")


if __name__ == "__main__":
    main()
