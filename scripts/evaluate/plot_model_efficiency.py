#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制消融实验模型效率指标对比图 (Params / GFLOPs)。
输入: eval_model_efficiency.py 输出的 efficiency_summary.csv
输出: efficiency_comparison.png (2 面板: 参数量 + 计算量)

用法示例:
  python scripts/evaluate/plot_model_efficiency.py --csv outputs/detection/efficiency_summary_all.csv
  python scripts/evaluate/plot_model_efficiency.py --csv outputs/detection/efficiency_summary_all.csv --output-dir outputs/detection
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import AutoMinorLocator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

# 实验简称（用于顶部 1-13 说明行，与检测指标图保持完全一致）
ALIAS_MAP = {
    "ablation_exp01_baseline":              "Baseline",
    "ablation_exp02_ghost":                 "Ghost",
    "ablation_exp03_shuffle":               "Shuffle",
    "ablation_exp04_attention":             "SEAttention",
    "ablation_exp05_coordatt":              "CoordAtt",
    "ablation_exp06_siou":                  "SIoU",
    "ablation_exp07_eiou":                  "EIoU",
    "ablation_exp08_ghost_attention":       "Ghost+SEAttention",
    "ablation_exp09_ghost_eiou":            "Ghost+EIoU",
    "ablation_exp10_attention_eiou":        "SEAttention+EIoU",
    "ablation_exp11_shuffle_coordatt":      "Shuffle+CoordAtt",
    "ablation_exp12_shuffle_coordatt_siou": "Shuffle+CoordAtt+SIoU",
    "ablation_exp13_shuffle_coordatt_eiou": "Shuffle+CoordAtt+EIoU",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制模型效率指标对比图")
    p.add_argument("--csv", required=True,
                   help="效率评估 CSV 路径 (eval_model_efficiency.py 的输出)")
    p.add_argument("--output-dir", default=None,
                   help="图片输出目录 (默认与 CSV 同级)")
    p.add_argument("--dpi", type=int, default=320)
    return p.parse_args()


# ─── 全局风格 ─────────────────────────────────────────────────────────────────

def setup_style() -> None:
    mpl.rcParams["font.family"]      = "serif"
    mpl.rcParams["font.serif"]       = ["Times New Roman", "STSong", "SimSun", "DejaVu Serif"]
    mpl.rcParams["font.sans-serif"]  = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["text.color"]       = "#273444"
    mpl.rcParams["axes.labelcolor"]  = "#273444"
    mpl.rcParams["axes.titlecolor"]  = "#273444"
    mpl.rcParams["xtick.color"]      = "#273444"
    mpl.rcParams["ytick.color"]      = "#273444"
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"]   = "#fbfbf8"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["axes.linewidth"]   = 1.0
    mpl.rcParams["xtick.direction"]  = "in"
    mpl.rcParams["ytick.direction"]  = "in"


# ─── 数据读取 ──────────────────────────────────────────────────────────────────

def _safe_float(v: str):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))

    rows = []
    for r in raw:
        exp_no = int(r.get("exp_no", 999))
        rows.append({
            "exp_no":   exp_no,
            "exp_name": r["exp_name"],
            "label":    ALIAS_MAP.get(r["exp_name"], r["exp_name"]),
            "params_m": _safe_float(r.get("params_m")),
            "gflops":   _safe_float(r.get("gflops")),
            "fps":      _safe_float(r.get("fps")),
            "inf_ms":   _safe_float(r.get("inf_ms")),
        })
    rows.sort(key=lambda r: r["exp_no"])
    return rows


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def _fit_text_fontsize(fig, text_obj, max_width: float,
                        initial: float, minimum: float = 6.5) -> float:
    fontsize = initial
    text_obj.set_fontsize(fontsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = (text_obj.get_window_extent(renderer=renderer)
             .transformed(fig.transFigure.inverted()).width)
    while width > max_width and fontsize > minimum:
        fontsize = max(minimum, fontsize - 0.2)
        text_obj.set_fontsize(fontsize)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width = (text_obj.get_window_extent(renderer=renderer)
                 .transformed(fig.transFigure.inverted()).width)
    return fontsize


def add_top_experiment_legend(fig, rows: list[dict], *,
                               box_y: float = 0.910,
                               fontsize: float = 9.0) -> None:
    """顶部添加 1-13 实验说明行（带边框）。"""
    parts = "  ".join(f"{r['exp_no']}-{r['label']}" for r in rows)
    txt = fig.text(0.5, box_y, parts, ha="center", va="center",
                   fontsize=fontsize, color="#1f2937")
    fig.canvas.draw()
    _fit_text_fontsize(fig, txt, max_width=0.88,
                        initial=fontsize, minimum=6.5)
    renderer = fig.canvas.get_renderer()
    bbox = (txt.get_window_extent(renderer=renderer)
            .transformed(fig.transFigure.inverted()))
    bw = min(0.94, bbox.width + 0.028)
    bh = 0.061
    fig.add_artist(Rectangle(
        (0.5 - bw / 2, box_y - bh / 2), bw, bh,
        transform=fig.transFigure, fill=False,
        linewidth=1.0, edgecolor="#94a3b8",
    ))


def add_color_legend(fig, handles: list, labels: list, *,
                      box_y: float, fontsize: float = 9.0) -> None:
    """顶部颜色图例行（带边框）。"""
    leg = fig.legend(
        handles=handles, labels=labels,
        loc="center", bbox_to_anchor=(0.5, box_y),
        ncol=len(labels), frameon=False,
        fontsize=fontsize,
        handlelength=1.8, handletextpad=0.6, columnspacing=1.2,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lbbox = (leg.get_window_extent(renderer=renderer)
             .transformed(fig.transFigure.inverted()))
    lbw = min(0.36, lbbox.width + 0.030)
    # 图高比例换算（效率图 5.8in vs 检测图 8.8in），保持视觉高度一致
    lbh = 0.046
    fig.add_artist(Rectangle(
        (0.5 - lbw / 2, box_y - lbh / 2), lbw, lbh,
        transform=fig.transFigure, fill=False,
        linewidth=1.0, edgecolor="#94a3b8",
    ))


# ─── 核心柱状图绘制 ────────────────────────────────────────────────────────────

C_BASELINE = "#8b5cf6"
C_BETTER   = "#22c55e"
C_WORSE    = "#ef4444"
C_MISSING  = "#cbd5e1"


def _bar_colors(values: list, baseline_val, higher_better: bool) -> list[str]:
    colors = []
    for i, v in enumerate(values):
        if v is None:
            colors.append(C_MISSING)
        elif i == 0:
            colors.append(C_BASELINE)
        elif baseline_val is None:
            colors.append(C_BASELINE)
        else:
            better = (v > baseline_val) if higher_better else (v < baseline_val)
            colors.append(C_BETTER if better else C_WORSE)
    return colors


def plot_bar_panel(ax, x: np.ndarray, rows: list[dict], key: str,
                   metric_label: str, unit: str, higher_better: bool,
                   baseline_val) -> None:
    values = [r[key] for r in rows]
    colors = _bar_colors(values, baseline_val, higher_better)
    bar_vals = [v if v is not None else 0.0 for v in values]

    bars = ax.bar(x, bar_vals, width=0.65,
                  color=colors, edgecolor="#334155",
                  linewidth=0.9, alpha=0.55, zorder=3)

    # 柱顶数值标注
    valid_non_none = [v for v in values if v is not None]
    y_range = (max(valid_non_none) - min(valid_non_none)) if len(valid_non_none) > 1 else abs(valid_non_none[0])
    label_offset = y_range * 0.025 if y_range > 0 else 0.1

    for bar, v in zip(bars, values):
        if v is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=9.2, color="#334155",
        )

    # 基线虚线参考
    if baseline_val is not None:
        ax.axhline(baseline_val, color="#64748b",
                   linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)

    # 轴样式
    ax.set_ylabel(f"{metric_label} ({unit})", fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.32, color="#a8b0ba")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45,
            alpha=0.22, color="#a8b0ba")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.40, alpha=0.14, color="#a8b0ba")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5b6673")
    ax.spines["bottom"].set_color("#5b6673")
    ax.set_facecolor("#fbfbf8")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r["exp_no"])) for r in rows], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Y 轴范围
    if valid_non_none:
        vmin, vmax = min(valid_non_none), max(valid_non_none)
        pad = (vmax - vmin) * 0.22 if vmax != vmin else abs(vmax) * 0.22 or 1.0
        ax.set_ylim(max(0, vmin - pad * 0.3), vmax + pad * 1.6)


# ─── 主图 1: 3 面板横向 ────────────────────────────────────────────────────────

def plot_3panel(rows: list[dict], output_dir: Path, dpi: int) -> Path:
    """2 面板: 参数量 + 计算量（中期检查版）"""
    x = np.arange(len(rows))
    baseline = next((r for r in rows if r["exp_no"] == 1), None)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), dpi=220)

    metrics = [
        ("params_m", "Params",  "M",   False, "(a) 参数量"),
        ("gflops",   "GFLOPs",  "G",   False, "(b) 计算量"),
    ]

    for ax, (key, label, unit, hb, subtitle) in zip(axes, metrics):
        bv = baseline[key] if baseline else None
        plot_bar_panel(ax, x, rows, key, label, unit, hb, bv)
        ax.text(0.5, -0.18, subtitle,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=13, fontweight="bold", fontfamily="sans-serif")

    # 颜色图例
    legend_handles = [
        Patch(facecolor=C_BASELINE, edgecolor="#334155", alpha=0.55, label="Baseline (Exp1)"),
        Patch(facecolor=C_BETTER,   edgecolor="#334155", alpha=0.55, label="Better"),
        Patch(facecolor=C_WORSE,    edgecolor="#334155", alpha=0.55, label="Worse"),
    ]
    add_color_legend(fig, legend_handles,
                     ["Baseline (Exp1)", "Better", "Worse"],
                     box_y=0.892, fontsize=9.2)
    add_top_experiment_legend(fig, rows, box_y=0.828, fontsize=9.2)

    fig.suptitle("消融实验模型轻量化指标对比",
                 fontsize=14, fontweight="bold", y=0.978,
                 fontfamily="sans-serif")
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.06, right=0.97, wspace=0.30)

    out = output_dir / "efficiency_comparison.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 主图 2: 2×2 四面板（含延迟） ─────────────────────────────────────────────

def plot_4panel(rows: list[dict], output_dir: Path, dpi: int) -> Path:
    x = np.arange(len(rows))
    baseline = next((r for r in rows if r["exp_no"] == 1), None)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.2), dpi=220)
    axs = axes.flatten()

    metrics = [
        ("params_m", "Params",   "M",   False, "(a) 参数量"),
        ("gflops",   "GFLOPs",   "G",   False, "(b) 计算量"),
        ("fps",      "FPS",      "fps", True,  "(c) 推理速度"),
        ("inf_ms",   "Latency",  "ms",  False, "(d) 推理延迟"),
    ]

    for ax, (key, label, unit, hb, subtitle) in zip(axs, metrics):
        bv = baseline[key] if baseline else None
        plot_bar_panel(ax, x, rows, key, label, unit, hb, bv)
        ax.text(0.5, -0.20, subtitle,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=13, fontweight="bold", fontfamily="sans-serif")

    # 颜色图例
    legend_handles = [
        Patch(facecolor=C_BASELINE, edgecolor="#334155", alpha=0.55, label="Baseline (Exp1)"),
        Patch(facecolor=C_BETTER,   edgecolor="#334155", alpha=0.55, label="Better"),
        Patch(facecolor=C_WORSE,    edgecolor="#334155", alpha=0.55, label="Worse"),
    ]
    add_color_legend(fig, legend_handles,
                     ["Baseline (Exp1)", "Better", "Worse"],
                     box_y=0.935, fontsize=9.2)
    add_top_experiment_legend(fig, rows, box_y=0.876, fontsize=9.0)

    fig.suptitle("消融实验模型效率指标详细对比",
                 fontsize=15, fontweight="bold", y=0.993,
                 fontfamily="sans-serif")
    fig.subplots_adjust(top=0.80, bottom=0.10, hspace=0.38, wspace=0.22)

    out = output_dir / "efficiency_comparison_4panel.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 主图 3: 综合效率视角 (mAP vs FPS 散点，可选) ─────────────────────────────

def plot_tradeoff(rows: list[dict], detection_csv: Path,
                  output_dir: Path, dpi: int) -> Path | None:
    """mAP@0.5 vs FPS 散点图（仅当检测 CSV 存在时生成）。"""
    if not detection_csv.exists():
        return None

    det_map: dict[str, float] = {}
    try:
        with open(detection_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                det_map[row["exp_name"]] = float(row["map50"])
    except Exception:
        return None

    fps_vals, map_vals, labels, colors = [], [], [], []
    palette = plt.cm.get_cmap("tab10")
    for i, r in enumerate(rows):
        if r["fps"] is None:
            continue
        m = det_map.get(r["exp_name"])
        if m is None:
            continue
        fps_vals.append(r["fps"])
        map_vals.append(m)
        labels.append(str(r["exp_no"]))
        colors.append(palette(i / max(len(rows) - 1, 1)))

    if not fps_vals:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=220)
    for fps, mval, lab, col in zip(fps_vals, map_vals, labels, colors):
        ax.scatter(fps, mval, color=col, s=70, zorder=3,
                   edgecolors="#334155", linewidths=0.8)
        ax.annotate(lab, (fps, mval),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=9, color="#334155")

    ax.set_xlabel("FPS (frames/sec)", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("效率-精度权衡 (mAP@0.5 vs FPS)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.32, color="#a8b0ba")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_facecolor("#fbfbf8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    out = output_dir / "efficiency_tradeoff.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_style()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError("CSV 为空")

    output_dir = Path(args.output_dir) if args.output_dir else Path(ROOT / "outputs" / "detection" / "detection_eval_batch_20260324_215640")
    output_dir.mkdir(parents=True, exist_ok=True)

    out1 = plot_3panel(rows, output_dir, args.dpi)
    print(f"SAVED: {out1}")

    # plot_4panel (含FPS/延迟) 按需启用，中期检查阶段只输出参数量和GFLOPs
    # has_latency = any(r["inf_ms"] is not None for r in rows)
    # if has_latency:
    #     out2 = plot_4panel(rows, output_dir, args.dpi)
    #     print(f"SAVED: {out2}")

    # 可选: 精度-效率散点（自动查找同目录 summary.csv）
    detection_csv = csv_path.parent / "summary.csv"
    out3 = plot_tradeoff(rows, detection_csv, output_dir, args.dpi)
    if out3:
        print(f"SAVED: {out3}")


if __name__ == "__main__":
    main()
