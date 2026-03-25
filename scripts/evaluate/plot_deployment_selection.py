#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署候选模型综合评估分析图。

综合检测指标（mAP@50, mAP@50-95, Precision, Recall）、
模型效率（Params, GFLOPs）及 ByteTrack 跟踪指标（ID 切换、匹配率、FPS），
使用加权评分选出最优部署候选。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ─── 5 个候选实验 ──────────────────────────────────────────────
CANDIDATES: List[str] = [
    "ablation_exp07_eiou",
    "ablation_exp06_siou",
    "ablation_exp01_baseline",
    "ablation_exp09_ghost_eiou",
    "ablation_exp03_shuffle",
]

SHORT_LABELS: Dict[str, str] = {
    "ablation_exp01_baseline":   "Baseline",
    "ablation_exp03_shuffle":    "ShuffleNet",
    "ablation_exp06_siou":       "SIoU",
    "ablation_exp07_eiou":       "EIoU",
    "ablation_exp09_ghost_eiou": "Ghost+EIoU",
}

# ─── 数据来源 ─────────────────────────────────────────────────
DETECTION_SUMMARY = ROOT / "outputs/detection/detection_eval_batch_20260324_215640/summary.json"
EFFICIENCY_CSV = ROOT / "outputs/results/efficiency_summary_all.csv"
TRACKING_BASE = ROOT / "outputs/tracking/candidates_20260325_v2"
OUTPUT_DIR = ROOT / "outputs/results/deployment_selection"

# ─── 选定跟踪器 ──────────────────────────────────────────────
SELECTED_TRACKER = "bytetrack"


# ==================================================================
# 数据采集
# ==================================================================

def load_detection(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    rows = []
    for entry in results:
        rows.append({
            "experiment": entry.get("exp_name", ""),
            "mAP50": float(entry.get("map50", 0)),
            "mAP50_95": float(entry.get("map5095", 0)),
            "precision": float(entry.get("precision", 0)),
            "recall": float(entry.get("recall", 0)),
        })
    return pd.DataFrame(rows)


def load_efficiency(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"exp_name": "experiment"})
    return df


def load_tracking(base: Path, experiments: List[str], tracker: str) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        exp_dir = base / exp
        if not exp_dir.exists():
            continue
        # 取最新 run
        cands = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith(tracker + "_")],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not cands:
            continue
        summary = cands[0] / "summary_metrics.json"
        if not summary.exists():
            continue
        with open(summary, "r", encoding="utf-8") as f:
            payload = json.load(f)
        t = payload.get("totals", {})
        rows.append({
            "experiment": exp,
            "match_rate": float(t.get("match_rate", 0)),
            "id_switch": int(t.get("id_switch_proxy_sum", 0)),
            "track_fps": float(t.get("avg_fps_mean", 0)),
        })
    return pd.DataFrame(rows)


def merge_data() -> pd.DataFrame:
    det = load_detection(DETECTION_SUMMARY)
    eff = load_efficiency(EFFICIENCY_CSV)
    trk = load_tracking(TRACKING_BASE, CANDIDATES, SELECTED_TRACKER)

    # 只保留候选实验
    det = det[det["experiment"].isin(CANDIDATES)].copy()
    eff = eff[eff["experiment"].isin(CANDIDATES)].copy()

    df = det.merge(eff[["experiment", "params_m", "gflops"]], on="experiment", how="left")
    df = df.merge(trk, on="experiment", how="left")
    # 按 CANDIDATES 顺序
    df["_order"] = df["experiment"].map({e: i for i, e in enumerate(CANDIDATES)})
    df = df.sort_values("_order").reset_index(drop=True)
    return df


# ==================================================================
# 综合加权评分
# ==================================================================

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    评分规则（用户偏好）:
      - mAP@50         权重 25  (份额要大)
      - mAP@50-95      权重 15
      - Precision       权重  5
      - Recall          权重  5
      - Params (越小越好) 权重 18  (轻量化优先)
      - GFLOPs (越小越好) 权重  7
      - ID切换 (越少越好)  权重 15  (首选 id 切换少)
      - 匹配率           权重  5
      - FPS (>25 即可)    权重  5
    """
    weights = {
        "mAP50":      25,
        "mAP50_95":   15,
        "precision":   5,
        "recall":      5,
        "params_m":   18,  # 越小越好
        "gflops":      7,  # 越小越好
        "id_switch":  15,  # 越小越好
        "match_rate":  5,
        "track_fps":   5,
    }
    # 越小越好的指标（反转归一化）
    lower_better = {"params_m", "gflops", "id_switch"}

    scores = pd.DataFrame(index=df.index)
    total = np.zeros(len(df))

    for col, w in weights.items():
        vals = df[col].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-9:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = np.ones_like(vals) * 0.5
        if col in lower_better:
            normed = 1.0 - normed
        scores[col] = normed
        total += normed * w

    df = df.copy()
    df["score"] = total
    df["score_norm"] = df["score"] / sum(weights.values()) * 100  # 百分制
    return df


# ==================================================================
# 绘图
# ==================================================================

def setup_style():
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


def _style_ax(ax):
    ax.set_facecolor("#fbfbf8")
    ax.grid(axis="y", ls="--", alpha=0.28, lw=0.6, color="#8b95a5", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")


def plot_deployment_analysis(df: pd.DataFrame, save_dir: Path):
    """
    2 × 3 面板：
      [0,0] mAP@50 & mAP@50-95  (双柱)
      [0,1] Params & GFLOPs       (双柱)
      [0,2] ID 切换次数 (ByteTrack)
      [1,0] Precision & Recall    (双柱)
      [1,1] 跟踪 FPS (ByteTrack)
      [1,2] 综合加权得分           (高亮最优)
    """

    labels = [SHORT_LABELS.get(e, e) for e in df["experiment"]]
    n = len(labels)
    x = np.arange(n)

    # 配色
    C1, C2 = "#315b7c", "#a3532c"     # 双柱配色
    C_ID = "#c0504d"                   # ID 切换（越少越好，红色警示）
    C_FPS = "#2f6f4f"                  # FPS 绿色
    C_SCORE = "#4472c4"               # 评分蓝色
    C_BEST = "#e37222"                # 最优高亮

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4))

    # ───── [0,0] mAP@50 & mAP@50-95 ─────
    ax = axes[0, 0]
    _style_ax(ax)
    w = 0.32
    r1 = ax.bar(x - w / 2, df["mAP50"], w * 0.9, color=C1, alpha=0.88, label="mAP@50", zorder=3)
    r2 = ax.bar(x + w / 2, df["mAP50_95"], w * 0.9, color=C2, alpha=0.88, label="mAP@50-95", zorder=3)
    for rects, fmt in [(r1, "{:.3f}"), (r2, "{:.3f}")]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(fmt.format(h), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, fontfamily="sans-serif")
    ax.set_ylabel("mAP", fontsize=9.5, fontfamily="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0.4, ax.get_ylim()[1] * 1.08)
    ax.legend(fontsize=8, framealpha=0.5, loc="upper left")
    ax.set_title("检测精度", fontsize=10.5, fontfamily="sans-serif", pad=8)

    # ───── [0,1] Params & GFLOPs ─────
    ax = axes[0, 1]
    _style_ax(ax)
    r1 = ax.bar(x - w / 2, df["params_m"], w * 0.9, color=C1, alpha=0.88, label="Params (M)", zorder=3)
    r2 = ax.bar(x + w / 2, df["gflops"], w * 0.9, color=C2, alpha=0.88, label="GFLOPs", zorder=3)
    for rects, fmt in [(r1, "{:.2f}"), (r2, "{:.1f}")]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(fmt.format(h), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, fontfamily="sans-serif")
    ax.set_ylabel("数值", fontsize=9.5, fontfamily="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    ax.legend(fontsize=8, framealpha=0.5, loc="upper left")
    ax.set_title("模型复杂度（越低越好）", fontsize=10.5, fontfamily="sans-serif", pad=8)

    # ───── [0,2] ID 切换次数 ─────
    ax = axes[0, 2]
    _style_ax(ax)
    colors_id = [C_BEST if v == df["id_switch"].min() else C_ID for v in df["id_switch"]]
    rects = ax.bar(x, df["id_switch"], 0.52, color=colors_id, alpha=0.88, zorder=3)
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", fontfamily="sans-serif")
    ax.set_ylabel("ID 切换次数", fontsize=9.5, fontfamily="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title(f"ID 切换 — {SELECTED_TRACKER.title()}（越低越好）", fontsize=10.5, fontfamily="sans-serif", pad=8)

    # ───── [1,0] Precision & Recall ─────
    ax = axes[1, 0]
    _style_ax(ax)
    r1 = ax.bar(x - w / 2, df["precision"], w * 0.9, color=C1, alpha=0.88, label="Precision", zorder=3)
    r2 = ax.bar(x + w / 2, df["recall"], w * 0.9, color=C2, alpha=0.88, label="Recall", zorder=3)
    for rects, fmt in [(r1, "{:.3f}"), (r2, "{:.3f}")]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(fmt.format(h), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, fontfamily="sans-serif")
    ax.set_ylabel("数值", fontsize=9.5, fontfamily="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0.6, ax.get_ylim()[1] * 1.04)
    ax.legend(fontsize=8, framealpha=0.5, loc="upper left")
    ax.set_title("检测 Precision & Recall", fontsize=10.5, fontfamily="sans-serif", pad=8)

    # ───── [1,1] 跟踪 FPS ─────
    ax = axes[1, 1]
    _style_ax(ax)
    colors_fps = [C_BEST if v == df["track_fps"].max() else C_FPS for v in df["track_fps"]]
    rects = ax.bar(x, df["track_fps"], 0.52, color=colors_fps, alpha=0.88, zorder=3)
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", fontfamily="sans-serif")
    ax.axhline(25, color="#d63031", ls="--", lw=1.0, alpha=0.65, zorder=5)
    ax.text(n - 0.5, 25.5, "25 FPS 基线", fontsize=7.5, color="#d63031",
            ha="right", va="bottom", fontfamily="sans-serif")
    ax.set_ylabel("FPS", fontsize=9.5, fontfamily="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    ax.set_title(f"跟踪帧率 — {SELECTED_TRACKER.title()}", fontsize=10.5, fontfamily="sans-serif", pad=8)

    # ───── [1,2] 综合加权得分 ─────
    ax = axes[1, 2]
    _style_ax(ax)
    scores = df["score_norm"].values
    best_idx = int(np.argmax(scores))
    colors_sc = [C_BEST if i == best_idx else C_SCORE for i in range(n)]
    rects = ax.bar(x, scores, 0.52, color=colors_sc, alpha=0.88, zorder=3)
    for i, rect in enumerate(rects):
        h = rect.get_height()
        extra = "  ★" if i == best_idx else ""
        ax.annotate(f"{h:.1f}{extra}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", fontfamily="sans-serif",
                    color=C_BEST if i == best_idx else "#273444")
    ax.set_ylabel("综合得分", fontsize=9.5, fontfamily="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.set_title("综合加权评分（百分制）", fontsize=10.5, fontfamily="sans-serif", pad=8)

    # 权重说明
    weight_text = (
        "权重: mAP@50=25  mAP@50-95=15  ID切换↓=15  轻量↓=18+7  P/R=5+5  匹配率=5  FPS=5"
    )
    fig.text(0.5, 0.005, weight_text, ha="center", fontsize=8, color="#6b7280",
             fontfamily="sans-serif", style="italic")

    fig.suptitle(
        "部署候选模型综合评估（跟踪器: ByteTrack）",
        fontsize=14, y=0.995, fontfamily="sans-serif", fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98, hspace=0.38, wspace=0.26)

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "deployment_selection.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存: {out}")
    return out


# ==================================================================
# 打印分析结论
# ==================================================================

def print_analysis(df: pd.DataFrame):
    best_idx = df["score_norm"].idxmax()
    best = df.loc[best_idx]
    name = SHORT_LABELS.get(best["experiment"], best["experiment"])

    print("\n" + "=" * 64)
    print("  部署候选模型综合评估结果")
    print("=" * 64)
    print(f"\n{'实验':<16} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} "
          f"{'Params':>7} {'GFLOPs':>7} {'IDsw':>5} {'FPS':>6} {'得分':>6}")
    print("-" * 90)

    for _, row in df.iterrows():
        tag = " ★" if row["experiment"] == best["experiment"] else ""
        lab = SHORT_LABELS.get(row["experiment"], row["experiment"])
        print(f"{lab:<16} {row['mAP50']:>7.4f} {row['mAP50_95']:>9.4f} "
              f"{row['precision']:>7.4f} {row['recall']:>7.4f} "
              f"{row['params_m']:>6.2f}M {row['gflops']:>7.1f} "
              f"{row['id_switch']:>5.0f} {row['track_fps']:>6.1f} "
              f"{row['score_norm']:>5.1f}{tag}")

    print("-" * 90)
    print(f"\n  >>> 推荐部署候选: {name} ({best['experiment']})")
    print(f"      mAP@50 = {best['mAP50']:.4f}  |  Params = {best['params_m']:.2f}M  |"
          f"  GFLOPs = {best['gflops']:.1f}")
    print(f"      ID切换 = {best['id_switch']:.0f}  |  FPS = {best['track_fps']:.1f}"
          f"  |  综合得分 = {best['score_norm']:.1f}/100")
    print("=" * 64)


# ==================================================================
# 入口
# ==================================================================

def main():
    setup_style()
    df = merge_data()
    if df.empty:
        raise SystemExit("[ERROR] 未读取到数据")

    print(f"[INFO] 读取 {len(df)} 个候选模型数据")
    df = compute_scores(df)
    print_analysis(df)
    plot_deployment_analysis(df, OUTPUT_DIR)

    # 保存数据到 CSV
    csv_path = OUTPUT_DIR / "deployment_scores.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[OK] CSV: {csv_path}")


if __name__ == "__main__":
    main()
