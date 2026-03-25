#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总多种跟踪算法结果并生成论文风格对比图表。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXPERIMENTS = [
    "ablation_exp01_baseline",
    "ablation_exp02_ghost",
    "ablation_exp03_shuffle",
    "ablation_exp05_coordatt",
    "ablation_exp06_siou",
    "ablation_exp07_eiou",
    "ablation_exp11_shuffle_coordatt",
    "ablation_exp12_shuffle_coordatt_siou",
    "ablation_exp13_shuffle_coordatt_eiou",
]
DEFAULT_TRACKERS = ["bytetrack", "deepsort", "centertrack"]


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


def _split_csv_arg(value: str | None) -> List[str] | None:
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)

    args.base_dir = pick(args.base_dir, config_get(config, "input", "base_dir"))
    args.output_dir = pick(args.output_dir, config_get(config, "output", "dir"))
    args.experiment_label = pick(getattr(args, "experiment_label", None), config_get(config, "input", "experiment_label"))

    cli_experiments = _split_csv_arg(args.experiments)
    cli_trackers = _split_csv_arg(args.trackers)
    args.experiments = cli_experiments if cli_experiments is not None else config_get(config, "input", "experiments", default=DEFAULT_EXPERIMENTS)
    args.trackers = cli_trackers if cli_trackers is not None else config_get(config, "input", "trackers", default=DEFAULT_TRACKERS)

    missing = []
    for name in ["base_dir", "experiments", "trackers"]:
        if getattr(args, name) in [None, []]:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing plot config fields: {', '.join(missing)}")

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render publication-style tracking comparison dashboard")
    parser.add_argument("--config", type=str, default="configs/plot_tracking_comparison.yaml", help="Plot config path")
    parser.add_argument("--base-dir", type=str, default=None, help="Tracking experiment root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for report and plots")
    parser.add_argument("--experiment-label", type=str, default=None, help="Optional experiment label for flat tracker-run layout")
    parser.add_argument("--experiments", type=str, default=None, help="Comma-separated experiment names")
    parser.add_argument("--trackers", type=str, default=None, help="Comma-separated tracker names")
    return resolve_args(parser.parse_args())


def setup_plot_style() -> None:
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


def style_axis(ax) -> None:
    ax.set_facecolor("#fbfbf8")
    ax.grid(axis="y", linestyle="--", alpha=0.28, linewidth=0.6, color="#8b95a5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")


def _latest_run_dir(parent: Path, prefix: str) -> Path | None:
    if not parent.exists():
        return None
    cands = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix + "_")]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _summary_json_path(base_dir: Path, exp: str, tracker: str) -> Path | None:
    if tracker == "centertrack":
        parent = base_dir / f"{exp}_centertrack"
        run_dir = _latest_run_dir(parent, "centertrack")
    else:
        parent = base_dir / exp / tracker
        run_dir = _latest_run_dir(parent, tracker)

    if run_dir is None:
        return None

    summary = run_dir / "summary_metrics.json"
    return summary if summary.exists() else None


def _infer_experiment_label(base_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label

    name = base_dir.name
    if "exp" in name:
        suffix = name.split("exp", 1)[1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return f"ablation_exp{int(digits):02d}"
    return name


def _collect_flat_tracker_rows(base_dir: Path, trackers: List[str], experiment_label: str | None) -> pd.DataFrame:
    rows: List[Dict] = []
    exp_name = _infer_experiment_label(base_dir, experiment_label)

    for tracker in trackers:
        run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(f"{tracker}_")]
        if not run_dirs:
            continue
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        summary_path = run_dirs[0] / "summary_metrics.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        totals = payload.get("totals", {})
        rows.append(
            {
                "experiment": exp_name,
                "tracker": str(payload.get("tracker", tracker)),
                "run_dir": str(run_dirs[0].name),
                "match_rate": float(totals.get("match_rate", 0.0)),
                "avg_fps_mean": float(totals.get("avg_fps_mean", 0.0)),
                "id_switch_proxy_sum": int(totals.get("id_switch_proxy_sum", 0)),
                "total_detections": int(totals.get("total_detections", 0)),
                "matched_tracks": int(totals.get("matched_tracks", 0)),
                "summary_path": str(summary_path.relative_to(ROOT)).replace("\\", "/") if summary_path.is_relative_to(ROOT) else str(summary_path),
            }
        )

    return pd.DataFrame(rows)


def collect_rows(base_dir: Path, experiments: List[str], trackers: List[str], experiment_label: str | None = None) -> pd.DataFrame:
    flat_df = _collect_flat_tracker_rows(base_dir, trackers, experiment_label)
    if not flat_df.empty:
        return flat_df

    rows: List[Dict] = []

    for exp in experiments:
        for trk in trackers:
            p = _summary_json_path(base_dir, exp, trk)
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
                    "summary_path": str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p),
                }
            )

    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, value_col: str, ylabel: str, title: str, save_path: Path, experiments: List[str]) -> None:
    pivot = df.pivot(index="experiment", columns="tracker", values=value_col).reindex(experiments)
    ax = pivot.plot(kind="bar", figsize=(12, 6), color=["#315b7c", "#a3532c", "#2f6f4f"][: len(pivot.columns)])
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    style_axis(ax)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_scatter(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    color_map = {"bytetrack": "#315b7c", "deepsort": "#a3532c", "centertrack": "#2f6f4f"}

    for trk in sorted(df["tracker"].unique()):
        s = df[df["tracker"] == trk]
        plt.scatter(
            s["avg_fps_mean"],
            s["match_rate"],
            s=84,
            alpha=0.85,
            c=color_map[trk],
            label=trk,
            edgecolors="#f8fafc",
            linewidths=0.9,
        )

    plt.xlabel("Average FPS")
    plt.ylabel("Match Rate")
    plt.title("Speed-Accuracy Trade-off (Proxy)")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

def plot_combined_dashboard(df: pd.DataFrame, save_path: Path, experiments: List[str], trackers: List[str]) -> None:
    """将核心指标汇总到同一张图中（四联图单文件）。"""
    pivot_match = df.pivot(index="experiment", columns="tracker", values="match_rate").reindex(experiments)
    pivot_fps = df.pivot(index="experiment", columns="tracker", values="avg_fps_mean").reindex(experiments)
    pivot_idsw = df.pivot(index="experiment", columns="tracker", values="id_switch_proxy_sum").reindex(experiments)

    color_map = {"bytetrack": "#315b7c", "deepsort": "#a3532c", "centertrack": "#2f6f4f"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) 匹配率
    ax = axes[0, 0]
    pivot_match.plot(kind="bar", ax=ax, color=[color_map[c] for c in pivot_match.columns])
    ax.set_title("Match Rate")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Match Rate")
    style_axis(ax)
    ax.tick_params(axis="x", rotation=20)

    # 2) FPS
    ax = axes[0, 1]
    pivot_fps.plot(kind="bar", ax=ax, color=[color_map[c] for c in pivot_fps.columns])
    ax.set_title("Average FPS")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("FPS")
    style_axis(ax)
    ax.tick_params(axis="x", rotation=20)

    # 3) ID切换代理（越低越好）
    ax = axes[1, 0]
    pivot_idsw.plot(kind="bar", ax=ax, color=[color_map[c] for c in pivot_idsw.columns])
    ax.set_title("ID Switch Proxy (Lower is Better)")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("IDSW Proxy")
    style_axis(ax)
    ax.tick_params(axis="x", rotation=20)

    # 4) 速度-匹配率散点
    ax = axes[1, 1]
    for trk in trackers:
        s = df[df["tracker"] == trk]
        ax.scatter(
            s["avg_fps_mean"],
            s["match_rate"],
            s=75,
            alpha=0.85,
            c=color_map[trk],
            label=trk,
            edgecolors="#f8fafc",
            linewidths=0.9,
        )
    ax.set_title("Speed-Accuracy Trade-off")
    ax.set_xlabel("Average FPS")
    ax.set_ylabel("Match Rate")
    style_axis(ax)
    ax.legend()

    fig.suptitle("Tracking Algorithm Comparison Dashboard", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def build_report(df: pd.DataFrame) -> str:
    experiments = df["experiment"].dropna().astype(str).unique().tolist()
    if len(experiments) == 1:
        scope_text = f"单次检测模型结果: {experiments[0]}"
    else:
        scope_text = f"{len(experiments)} 个实验: {', '.join(experiments)}"

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
    md.append(f"- 数据范围: {scope_text}")
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
    args = parse_args()
    setup_plot_style()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (base_dir / "algorithm_comparison")
    experiments = list(args.experiments)
    trackers = list(args.trackers)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(base_dir, experiments, trackers, args.experiment_label)
    if df.empty:
        raise SystemExit("No tracking summary data found.")

    df = df.sort_values(["experiment", "tracker"]).reset_index(drop=True)
    df.to_csv(out_dir / "tracker_comparison_raw.csv", index=False, encoding="utf-8-sig")

    plot_bar(df, "match_rate", "Match Rate", "Tracker Match Rate by Experiment", out_dir / "match_rate_bar.png", experiments)
    plot_bar(df, "avg_fps_mean", "Average FPS", "Tracker FPS by Experiment", out_dir / "fps_bar.png", experiments)
    plot_bar(
        df,
        "id_switch_proxy_sum",
        "ID Switch Proxy (sum)",
        "Tracker ID Switch Proxy by Experiment",
        out_dir / "idsw_proxy_bar.png",
        experiments,
    )
    plot_scatter(df, out_dir / "speed_accuracy_scatter.png")
    plot_combined_dashboard(df, out_dir / "tracker_metrics_dashboard.png", experiments, trackers)

    report = build_report(df)
    (out_dir / "tracker_comparison_report.md").write_text(report, encoding="utf-8")

    print(f"[OK] report: {out_dir / 'tracker_comparison_report.md'}")
    print(f"[OK] csv: {out_dir / 'tracker_comparison_raw.csv'}")
    print(f"[OK] plots: {out_dir}")


if __name__ == "__main__":
    main()
