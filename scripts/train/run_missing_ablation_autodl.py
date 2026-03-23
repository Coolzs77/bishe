#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoDL helper: run missing ablation experiments, then re-evaluate and redraw plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPERIMENTS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing ablation experiments on AutoDL")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", type=str, default="ram", choices=["none", "ram", "disk"])
    parser.add_argument("--sort-by", type=str, default="map50", choices=["precision", "recall", "map50", "map5095"])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def missing_experiments() -> list[str]:
    base = ROOT / "outputs" / "ablation_study"
    missing = []
    for exp in EXPECTED_EXPERIMENTS:
        best_pt = base / exp / "weights" / "best.pt"
        if not best_pt.exists():
            missing.append(exp)
    return missing


def latest_batch_csv() -> Path:
    result_dir = ROOT / "outputs" / "results"
    cands = sorted(result_dir.glob("detection_eval_batch_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError("No detection_eval_batch_*.csv found in outputs/results")
    return cands[0]


def main() -> None:
    args = parse_args()
    py = sys.executable

    missing = missing_experiments()
    print("\nExpected experiments:")
    for e in EXPECTED_EXPERIMENTS:
        print("-", e)

    if missing:
        print("\nMissing experiments:")
        for e in missing:
            print("-", e)
    else:
        print("\nNo missing experiments.")

    if not args.skip_train:
        for exp in missing:
            run_cmd(
                [
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
                ],
                cwd=ROOT,
            )

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
