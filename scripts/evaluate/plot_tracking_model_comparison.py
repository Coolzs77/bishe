#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5个候选检测模型 × 3种跟踪算法的跨模型对比图。

目录结构假设：
  base_dir / <exp_name> / <tracker>_YYYYMMDD_HHMMSS / summary_metrics.json

默认读取：outputs/tracking/candidates_20260325/
默认输出：outputs/tracking/candidates_20260325/model_tracker_comparison/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# 候选实验列表（按展示优先级排序）
CANDIDATE_EXPERIMENTS: List[str] = [
    "ablation_exp07_eiou",
    "ablation_exp06_siou",
    "ablation_exp01_baseline",
    "ablation_exp09_ghost_eiou",
    "ablation_exp03_shuffle",
]

# X轴短标签
EXP_SHORT_LABELS: Dict[str, str] = {
    "ablation_exp01_baseline":   "Baseline",
    "ablation_exp03_shuffle":    "Shuffle",
    "ablation_exp06_siou":       "SIoU",
    "ablation_exp07_eiou":       "EIoU",
    "ablation_exp09_ghost_eiou": "Ghost+EIoU",
}

# 与 plot_tracking_algorithm_comparison.py 保持一致的配色
TRACKER_COLORS: Dict[str, str] = {
    "bytetrack":   "#315b7c",
    "deepsort":    "#a3532c",
    "centertrack": "#2f6f4f",
}

TRACKER_DISPLAY: Dict[str, str] = {
    "bytetrack":   "ByteTrack",
    "deepsort":    "DeepSORT",
    "centertrack": "CenterTrack",
}

DEFAULT_TRACKERS: List[str] = ["deepsort", "bytetrack", "centertrack"]

DEFAULT_BASE_DIR = ROOT / "outputs/tracking/candidates_20260325_v2"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/tracking/candidates_20260325_v2/model_tracker_comparison"


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def _latest_run_dir(parent: Path, prefix: str) -> Path | None:
    """取最新的 <prefix>_YYYYMMDD_HHMMSS 目录。"""
    if not parent.exists():
        return None
    cands = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix + "_")]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def collect_data(
    base_dir: Path,
    experiments: List[str],
    trackers: List[str],
) -> pd.DataFrame:
    """遍历 base_dir/exp_name/tracker_YYYYMMDD/summary_metrics.json，返回汇总 DataFrame。"""
    rows: List[Dict] = []
    for exp in experiments:
        exp_dir = base_dir / exp
        if not exp_dir.exists():
            print(f"[WARN] 实验目录不存在: {exp_dir}")
            continue
        for trk in trackers:
            run_dir = _latest_run_dir(exp_dir, trk)
            if run_dir is None:
                print(f"[WARN] 未找到 {exp}/{trk}_* 目录")
                continue
            summary_path = run_dir / "summary_metrics.json"
            if not summary_path.exists():
                print(f"[WARN] 未找到 summary: {summary_path}")
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            totals = payload.get("totals", {})
            rows.append(
                {
                    "experiment": exp,
                    "tracker": trk,
                    "match_rate": float(totals.get("match_rate", 0.0)),
                    "avg_fps_mean": float(totals.get("avg_fps_mean", 0.0)),
                    "id_switch_proxy_sum": int(totals.get("id_switch_proxy_sum", 0)),
                    "run_dir": run_dir.name,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def setup_plot_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "Times New Roman", "STSong", "SimSun", "Noto Serif CJK SC", "DejaVu Serif"
    ]
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["text.color"] = "#273444"
    mpl.rcParams["axes.labelcolor"] = "#273444"
    mpl.rcParams["xtick.color"] = "#273444"
    mpl.rcParams["ytick.color"] = "#273444"
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "#fbfbf8"
    mpl.rcParams["savefig.facecolor"] = "white"


def _style_axis(ax) -> None:
    ax.set_facecolor("#fbfbf8")
    ax.grid(axis="y", linestyle="--", alpha=0.28, linewidth=0.6, color="#8b95a5", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")


def _annotate_bars(ax, rects, fmt: str = "{:.3f}", fontsize: float = 6.2) -> None:
    """在柱顶标注数值（offset points 方式，不依赖 ylim）。"""
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(
                fmt.format(h),
                xy=(rect.get_x() + rect.get_width() / 2.0, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#273444",
                fontfamily="sans-serif",
            )


def plot_model_tracker_comparison(
    df: pd.DataFrame,
    experiments: List[str],
    trackers: List[str],
    save_path: Path,
) -> None:
    """3面板对比图：检测-跟踪匹配率 | ID切换总数 | 平均处理FPS。"""

    exp_labels = [EXP_SHORT_LABELS.get(e, e) for e in experiments]
    n_exp = len(experiments)
    n_trk = len(trackers)

    x = np.arange(n_exp)
    total_width = 0.70
    bar_width = total_width / n_trk
    offsets = np.linspace(
        -(total_width - bar_width) / 2.0,
        (total_width - bar_width) / 2.0,
        n_trk,
    )

    # (metric_key, y轴标签, 数值格式, y轴起始值)
    metrics = [
        ("match_rate",          "检测-跟踪匹配率 (%)",  "{:.1f}",  None),
        ("id_switch_proxy_sum", "ID切换代理总数",        "{:.0f}",  None),
        ("avg_fps_mean",        "平均处理 FPS",          "{:.1f}",  None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))

    for col_idx, (metric_key, ylabel, vfmt, y_base) in enumerate(metrics):
        ax = axes[col_idx]
        _style_axis(ax)

        is_pct = metric_key == "match_rate"

        for trk_idx, trk in enumerate(trackers):
            vals = []
            for exp in experiments:
                row = df[(df["experiment"] == exp) & (df["tracker"] == trk)]
                v = float(row[metric_key].iloc[0]) if not row.empty else 0.0
                vals.append(v * 100.0 if is_pct else v)

            rects = ax.bar(
                x + offsets[trk_idx],
                vals,
                width=bar_width * 0.90,
                color=TRACKER_COLORS.get(trk, "#808080"),
                label=TRACKER_DISPLAY.get(trk, trk),
                alpha=0.88,
                zorder=3,
            )
            _annotate_bars(ax, rects, fmt=vfmt, fontsize=6.8)

        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=9.0, rotation=18, ha="right")
        ax.set_ylabel(ylabel, fontsize=9.5, fontfamily="sans-serif")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlim(-0.55, n_exp - 0.45)

        # 给标注留出顶部空间；匹配率面板 y 从 80% 起以突出差异
        ylo, yhi = ax.get_ylim()
        if is_pct:
            ax.set_ylim(80, yhi * 1.01 + 1.0)
        else:
            ax.set_ylim(0, yhi * 1.14)

    # 统一图例（底部居中）
    handles = [
        mpatches.Patch(
            facecolor=TRACKER_COLORS[t],
            alpha=0.88,
            label=TRACKER_DISPLAY[t],
        )
        for t in trackers
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10,
        framealpha=0.0,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
    )

    fig.suptitle(
        "候选检测模型 × 跟踪算法性能对比",
        fontsize=13,
        y=1.002,
        fontfamily="sans-serif",
    )

    fig.subplots_adjust(top=0.90, bottom=0.22, left=0.06, right=0.98, wspace=0.30)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存: {save_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    setup_plot_style()

    base_dir = DEFAULT_BASE_DIR
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_data(base_dir, CANDIDATE_EXPERIMENTS, DEFAULT_TRACKERS)
    if df.empty:
        raise SystemExit("[ERROR] 未读取到任何数据，请检查目录结构。")

    print(f"[INFO] 读取 {len(df)} 条记录")
    print(df.to_string(index=False))

    # 保存原始数据 CSV
    df.to_csv(out_dir / "model_tracker_comparison_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] CSV: {out_dir / 'model_tracker_comparison_raw.csv'}")

    # 生成对比图
    plot_model_tracker_comparison(
        df,
        experiments=CANDIDATE_EXPERIMENTS,
        trackers=DEFAULT_TRACKERS,
        save_path=out_dir / "tracking_model_comparison.png",
    )

    print(f"[DONE] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
