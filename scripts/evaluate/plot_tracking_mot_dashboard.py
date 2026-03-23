#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制三种跟踪算法的 MOT 四指标总览图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "tracking" / "rerun_20260320_full"
OUT_DIR = BASE / "mot_dashboard"

PRETTY_EXP = {
    "ablation_exp1_baseline": "Exp1 Baseline",
    "ablation_exp2_lightweight": "Exp2 Ghost",
    "ablation_exp3_shuffle": "Exp3 Shuffle",
    "ablation_exp4_coordatt": "Exp4 CoordAtt",
    "ablation_exp5_siou": "Exp5 SIoU",
    "ablation_exp6_eiou": "Exp6 EIoU",
    "ablation_exp7_shuffle_coordatt": "Exp7 Shuffle+CoordAtt",
    "ablation_exp8_shuffle_coordatt_siou": "Exp8 Shuffle+CoordAtt+SIoU",
    "ablation_exp9_shuffle_coordatt_eiou": "Exp9 Shuffle+CoordAtt+EIoU",
}

TRACKER_NAME = {
    "bytetrack": "ByteTrack",
    "deepsort": "DeepSORT",
    "centertrack": "CenterTrack",
}


def load_base_metrics() -> pd.DataFrame:
    csv_path = BASE / "experiment_group_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    df = pd.read_csv(csv_path)
    needed = ["experiment", "tracker", "MOTA", "IDSW", "IDF1", "FPS"]
    return df[needed].copy()


def load_centertrack_proxy() -> pd.DataFrame:
    """从 centertrack 汇总中构建 MOT 代理指标，保证三算法同图展示。"""
    csv_path = BASE / "algorithm_comparison" / "tracker_comparison_raw.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing centertrack source: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["tracker"] == "centertrack"].copy()

    # 代理指标（无GT时的近似口径）
    # IDSW: 直接使用 id_switch_proxy_sum
    # IDF1: 使用 matched/total 作为近似
    # MOTA: 1 - (FN + IDSW)/GT，其中 GT 近似 total_detections，FP 近似 0
    df["IDSW"] = df["id_switch_proxy_sum"].astype(float)
    df["IDF1"] = (df["matched_tracks"] / df["total_detections"]).fillna(0.0)
    df["MOTA"] = (
        1.0
        - ((df["total_detections"] - df["matched_tracks"]) + df["id_switch_proxy_sum"]) / df["total_detections"]
    ).fillna(0.0)
    df["FPS"] = df["avg_fps_mean"].astype(float)

    return df[["experiment", "tracker", "MOTA", "IDSW", "IDF1", "FPS"]].copy()


def build_unified_table() -> pd.DataFrame:
    base = load_base_metrics()
    center = load_centertrack_proxy()

    # 去除旧表中可能存在的 centertrack 行（当前一般不存在），再追加新行
    base = base[base["tracker"].isin(["bytetrack", "deepsort"])].copy()
    all_df = pd.concat([base, center], ignore_index=True)

    all_df["experiment_pretty"] = all_df["experiment"].map(PRETTY_EXP).fillna(all_df["experiment"])
    all_df["tracker_pretty"] = all_df["tracker"].map(TRACKER_NAME).fillna(all_df["tracker"])

    exp_order = [PRETTY_EXP[k] for k in PRETTY_EXP]
    trk_order = ["ByteTrack", "DeepSORT", "CenterTrack"]

    all_df["experiment_pretty"] = pd.Categorical(all_df["experiment_pretty"], categories=exp_order, ordered=True)
    all_df["tracker_pretty"] = pd.Categorical(all_df["tracker_pretty"], categories=trk_order, ordered=True)

    all_df = all_df.sort_values(["experiment_pretty", "tracker_pretty"]).reset_index(drop=True)
    return all_df


def _plot_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str, colors: dict) -> None:
    pivot = (
        df.pivot(index="experiment_pretty", columns="tracker_pretty", values=metric)
        .sort_index()
    )

    cols = [c for c in ["ByteTrack", "DeepSORT", "CenterTrack"] if c in pivot.columns]
    pivot = pivot[cols]

    pivot.plot(kind="bar", ax=ax, color=[colors[c] for c in cols])
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=18)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Tracker")


def plot_dashboard(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "ByteTrack": "#2474A6",
        "DeepSORT": "#D45D00",
        "CenterTrack": "#2A9D5B",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    _plot_metric(axes[0, 0], df, "FPS", "FPS", "Frames per second", colors)
    _plot_metric(axes[0, 1], df, "MOTA", "MOTA", "MOTA", colors)
    _plot_metric(axes[1, 0], df, "IDF1", "IDF1", "IDF1", colors)
    _plot_metric(axes[1, 1], df, "IDSW", "IDSW", "ID Switches", colors)

    fig.suptitle("Tracking Algorithms Dashboard (FPS / MOTA / IDF1 / IDSW)", fontsize=15, y=0.98)
    plt.tight_layout()

    out_path = OUT_DIR / "tracking_mot_dashboard.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_outputs(df: pd.DataFrame, dashboard_path: Path) -> None:
    csv_path = OUT_DIR / "tracking_mot_4metrics_3trackers.csv"
    md_path = OUT_DIR / "tracking_mot_4metrics_3trackers.md"

    out_df = df[["experiment_pretty", "tracker_pretty", "FPS", "MOTA", "IDF1", "IDSW"]].copy()
    out_df.columns = ["experiment", "tracker", "FPS", "MOTA", "IDF1", "IDSW"]
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 三跟踪算法四指标汇总")
    lines.append("")
    lines.append("说明:")
    lines.append("- 图中实验名已美化显示（Exp1 Baseline ... Exp9 Shuffle+CoordAtt+EIoU）")
    lines.append("- CenterTrack 的 MOTA/IDF1/IDSW 来自无GT场景代理口径（由检测-跟踪汇总近似得到）")
    lines.append("")
    lines.append(f"总览图: {dashboard_path.name}")
    lines.append("")
    lines.append("| experiment | tracker | FPS | MOTA | IDF1 | IDSW |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in out_df.iterrows():
        lines.append(
            f"| {r['experiment']} | {r['tracker']} | {float(r['FPS']):.3f} | {float(r['MOTA']):.4f} | "
            f"{float(r['IDF1']):.4f} | {float(r['IDSW']):.0f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] dashboard: {dashboard_path}")
    print(f"[OK] csv: {csv_path}")
    print(f"[OK] md: {md_path}")


def main() -> None:
    df = build_unified_table()
    dashboard_path = plot_dashboard(df)
    save_outputs(df, dashboard_path)


if __name__ == "__main__":
    main()
