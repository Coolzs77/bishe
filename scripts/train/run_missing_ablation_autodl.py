#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoDL helper: run specified ablation experiments, then re-evaluate and redraw plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPERIMENTS = [
    "ablation_exp01_baseline",
    "ablation_exp02_ghost",
    "ablation_exp03_shuffle",
    "ablation_exp04_attention",
    "ablation_exp05_coordatt",
    "ablation_exp06_siou",
    "ablation_exp07_eiou",
    "ablation_exp08_ghost_attention",
    "ablation_exp09_ghost_eiou",
    "ablation_exp10_attention_eiou",
    "ablation_exp11_shuffle_coordatt",
    "ablation_exp12_shuffle_coordatt_siou",
    "ablation_exp13_shuffle_coordatt_eiou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run specified ablation experiments on AutoDL")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", type=str, default="ram", choices=["none", "ram", "disk"])
    parser.add_argument("--profile", type=str, default="controlled", choices=["controlled", "optimal"])
    parser.add_argument(
        "--profile-config",
        type=str,
        default="configs/ablation/train_profile_optimal.yaml",
        help="train_ablation optimal 模式配置文件",
    )
    parser.add_argument(
        "--allow-profile-hyp-override",
        action="store_true",
        help="透传给 train_ablation.py，允许 profile 覆盖实验默认 hyp",
    )
    parser.add_argument("--sort-by", type=str, default="map50", choices=["precision", "recall", "map50", "map5095"])
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help=(
            "Comma-separated experiment names to run directly, e.g. "
            "ablation_exp01_baseline,ablation_exp03_shuffle"
        ),
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def parse_experiments(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return EXPECTED_EXPERIMENTS.copy()

    selected = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in selected if x not in EXPECTED_EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Allowed: {EXPECTED_EXPERIMENTS}")
    return selected


def latest_batch_csv() -> Path:
    result_dir = ROOT / "outputs" / "results"
    cands = sorted(result_dir.glob("detection_eval_batch_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    return cands[0]


def main() -> None:
    args = parse_args()
    py = sys.executable

    selected_experiments = parse_experiments(args.experiments)

    print("\nExpected experiments:")
    for e in EXPECTED_EXPERIMENTS:
        print("-", e)

    print("\nSelected experiments:")
    for e in selected_experiments:
        print("-", e)

    if not args.skip_train:
        for exp in selected_experiments:
            train_cmd = [
                py,
                str(ROOT / "scripts" / "train" / "train_ablation.py"),
                "--stage",
                "all",
                "--only",
                exp,
                "--epochs",
                str(args.epochs),
                "--batch",
                str(args.batch),
                "--img",
                str(args.img),
                "--device",
                args.device,
                "--workers",
                str(args.workers),
                "--patience",
                str(args.patience),
                "--cache",
                args.cache,
                "--profile",
                args.profile,
                "--profile-config",
                args.profile_config,
            ]
            if args.allow_profile_hyp_override:
                train_cmd.append("--allow-profile-hyp-override")
            run_cmd(train_cmd, cwd=ROOT)

    if not args.skip_eval:
        run_cmd(
            [
                py,
                str(ROOT / "scripts" / "evaluate" / "eval_detection.py"),
                "--mode",
                "metric",
                "--batch-eval",
                "--ablation-dir",
                "outputs/ablation_study",
                "--stage",
                "all",
                "--weights-name",
                "best.pt",
                "--data",
                "data/processed/flir/dataset.yaml",
                "--img-size",
                str(args.img),
                "--batch-size",
                str(args.batch),
                "--device",
                args.device,
                "--workers",
                str(args.workers),
                "--project",
                "outputs/val_detection",
                "--name",
                "ablation_batch_redraw",
                "--exist-ok",
                "--save-csv",
                "--sort-by",
                args.sort_by,
            ],
            cwd=ROOT,
        )

    if not args.skip_plot:
        csv_path = latest_batch_csv()
        if csv_path is None:
            print("\nSkip plot: outputs/results 下没有 detection_eval_batch_*.csv（通常是评估全失败或无可评估权重）。")
            print("建议先补齐成功训练的 best.pt，再执行批量评估与绘图。")
            print("\nDone.")
            return
        out_prefix = ROOT / "outputs" / "results" / "detection_eval_batch_redraw"
        run_cmd(
            [
                py,
                str(ROOT / "scripts" / "evaluate" / "plot_eval_summary.py"),
                "--csv",
                str(csv_path),
                "--output-prefix",
                str(out_prefix),
                "--sort-by",
                args.sort_by,
            ],
            cwd=ROOT,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
