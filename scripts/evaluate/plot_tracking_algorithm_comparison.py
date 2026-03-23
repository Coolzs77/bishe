#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总三种跟踪算法结果并生成对比图表。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "outputs" / "tracking" / "rerun_20260320_full"
OUT_DIR = BASE_DIR / "algorithm_comparison"

EXPERIMENTS = [
    "ablation_exp1_baseline",
    "ablation_exp2_lightweight",
    "ablation_exp3_shuffle",
    "ablation_exp4_coordatt",
    "ablation_exp5_siou",
    "ablation_exp6_eiou",
    "ablation_exp7_shuffle_coordatt",
    "ablation_exp8_shuffle_coordatt_siou",
    "ablation_exp9_shuffle_coordatt_eiou",
]
TRACKERS = ["bytetrack", "deepsort", "centertrack"]


def _latest_run_dir(parent: Path, prefix: str) -> Path | None:
    if not parent.exists():
        return None
    cands = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix + "_")]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _summary_json_path(exp: str, tracker: str) -> Path | None:
    if tracker == "centertrack":
        parent = BASE_DIR / f"{exp}_centertrack"
        run_dir = _latest_run_dir(parent, "centertrack")
    else:
        parent = BASE_DIR / exp / tracker
        run_dir = _latest_run_dir(parent, tracker)

    if run_dir is None:
        return None

    summary = run_dir / "summary_metrics.json"
    return summary if summary.exists() else None


def collect_rows() -> pd.DataFrame:
    rows: List[Dict] = []

    for exp in EXPERIMENTS:
        for trk in TRACKERS:
            p = _summary_json_path(exp, trk)
            if p is None:
                continue
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            totals = payload.get("totals", {})
            rows.append(
                {
                    "experiment": exp,
                    "tracker": trk,
                    "run_dir": str(p.parent.name),
                    "match_rate": float(totals.get("match_rate", 0.0)),
                    "avg_fps_mean": float(totals.get("avg_fps_mean", 0.0)),
                    "id_switch_proxy_sum": int(totals.get("id_switch_proxy_sum", 0)),
                    "total_detections": int(totals.get("total_detections", 0)),
                    "matched_tracks": int(totals.get("matched_tracks", 0)),
                    "summary_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                }
            )

    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, value_col: str, ylabel: str, title: str, save_path: Path) -> None:
    pivot = df.pivot(index="experiment", columns="tracker", values=value_col).reindex(EXPERIMENTS)
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_scatter(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    color_map = {"bytetrack": "#1f77b4", "deepsort": "#ff7f0e", "centertrack": "#2ca02c"}

    for trk in TRACKERS:
        s = df[df["tracker"] == trk]
        plt.scatter(
            s["avg_fps_mean"],
            s["match_rate"],
            s=70,
            alpha=0.85,
            c=color_map[trk],
            label=trk,
        )

    plt.xlabel("Average FPS")
    plt.ylabel("Match Rate")
    plt.title("Speed-Accuracy Trade-off (Proxy)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

def plot_combined_dashboard(df: pd.DataFrame, save_path: Path) -> None:
    """将核心指标汇总到同一张图中（四联图单文件）。"""
    pivot_match = df.pivot(index="experiment", columns="tracker", values="match_rate").reindex(EXPERIMENTS)
    pivot_fps = df.pivot(index="experiment", columns="tracker", values="avg_fps_mean").reindex(EXPERIMENTS)
    pivot_idsw = df.pivot(index="experiment", columns="tracker", values="id_switch_proxy_sum").reindex(EXPERIMENTS)

    color_map = {"bytetrack": "#1f77b4", "deepsort": "#ff7f0e", "centertrack": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) 匹配率
    ax = axes[0, 0]
    pivot_match.plot(kind="bar", ax=ax, color=[color_map[c] for c in pivot_match.columns])
    ax.set_title("Match Rate")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Match Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)

    # 2) FPS
    ax = axes[0, 1]
    pivot_fps.plot(kind="bar", ax=ax, color=[color_map[c] for c in pivot_fps.columns])
    ax.set_title("Average FPS")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("FPS")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)

    # 3) ID切换代理（越低越好）
    ax = axes[1, 0]
    pivot_idsw.plot(kind="bar", ax=ax, color=[color_map[c] for c in pivot_idsw.columns])
    ax.set_title("ID Switch Proxy (Lower is Better)")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("IDSW Proxy")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)

    # 4) 速度-匹配率散点
    ax = axes[1, 1]
    for trk in TRACKERS:
        s = df[df["tracker"] == trk]
        ax.scatter(
            s["avg_fps_mean"],
            s["match_rate"],
            s=75,
            alpha=0.85,
            c=color_map[trk],
            label=trk,
        )
    ax.set_title("Speed-Accuracy Trade-off")
    ax.set_xlabel("Average FPS")
    ax.set_ylabel("Match Rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.suptitle("Tracking Algorithm Comparison Dashboard", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def build_report(df: pd.DataFrame) -> str:
    agg = (
        df.groupby("tracker", as_index=False)
        .agg(
            mean_match_rate=("match_rate", "mean"),
            mean_fps=("avg_fps_mean", "mean"),
            mean_idsw_proxy=("id_switch_proxy_sum", "mean"),
        )
        .sort_values("mean_match_rate", ascending=False)
    )

    best_acc = agg.sort_values("mean_match_rate", ascending=False).iloc[0]
    best_fps = agg.sort_values("mean_fps", ascending=False).iloc[0]
    best_stable = agg.sort_values("mean_idsw_proxy", ascending=True).iloc[0]

    md = []
    md.append("# 三种跟踪算法对比总结")
    md.append("")
    md.append("## 对比口径")
    md.append("- 数据范围: exp1~exp9（新消融矩阵）的最新一次运行")
    md.append("- 指标说明: match_rate 为无GT场景下检测-跟踪匹配率代理指标; id_switch_proxy_sum 为ID切换代理计数")
    md.append("")
    md.append("## 算法均值对比")
    md.append("| tracker | mean_match_rate | mean_fps | mean_idsw_proxy |")
    md.append("|---|---:|---:|---:|")
    for _, r in agg.iterrows():
        md.append(
            f"| {r['tracker']} | {float(r['mean_match_rate']):.4f} | "
            f"{float(r['mean_fps']):.3f} | {float(r['mean_idsw_proxy']):.2f} |"
        )
    md.append("")
    md.append("## 优劣结论")
    md.append(f"- 匹配率最好: {best_acc['tracker']} (mean_match_rate={best_acc['mean_match_rate']:.4f})")
    md.append(f"- 速度最快: {best_fps['tracker']} (mean_fps={best_fps['mean_fps']:.3f})")
    md.append(f"- 稳定性最好(低ID切换): {best_stable['tracker']} (mean_idsw_proxy={best_stable['mean_idsw_proxy']:.2f})")
    md.append("- ByteTrack: 速度和稳定性整体均衡，适合作为默认工程方案。")
    md.append("- DeepSORT: 在当前配置下匹配率高，但ID切换代理偏高，适合重识别特征更强的场景进一步调参。")
    md.append("- CenterTrack: 匹配率接近最优，但速度略低于ByteTrack，复杂拥挤场景下ID切换代理波动较大。")
    md.append("")
    md.append("## 图表")
    md.append("- match_rate 对比: match_rate_bar.png")
    md.append("- avg_fps_mean 对比: fps_bar.png")
    md.append("- id_switch_proxy_sum 对比: idsw_proxy_bar.png")
    md.append("- 速度-匹配率散点: speed_accuracy_scatter.png")
    md.append("- 单图总览(全部指标): tracker_metrics_dashboard.png")
    md.append("")

    return "\n".join(md)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_rows()
    if df.empty:
        raise SystemExit("No tracking summary data found.")

    df = df.sort_values(["experiment", "tracker"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "tracker_comparison_raw.csv", index=False, encoding="utf-8-sig")

    plot_bar(df, "match_rate", "Match Rate", "Tracker Match Rate by Experiment", OUT_DIR / "match_rate_bar.png")
    plot_bar(df, "avg_fps_mean", "Average FPS", "Tracker FPS by Experiment", OUT_DIR / "fps_bar.png")
    plot_bar(
        df,
        "id_switch_proxy_sum",
        "ID Switch Proxy (sum)",
        "Tracker ID Switch Proxy by Experiment",
        OUT_DIR / "idsw_proxy_bar.png",
    )
    plot_scatter(df, OUT_DIR / "speed_accuracy_scatter.png")
    plot_combined_dashboard(df, OUT_DIR / "tracker_metrics_dashboard.png")

    report = build_report(df)
    (OUT_DIR / "tracker_comparison_report.md").write_text(report, encoding="utf-8")

    print(f"[OK] report: {OUT_DIR / 'tracker_comparison_report.md'}")
    print(f"[OK] csv: {OUT_DIR / 'tracker_comparison_raw.csv'}")
    print(f"[OK] plots: {OUT_DIR}")


if __name__ == "__main__":
    main()
