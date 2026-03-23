#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为阶段1消融汇总补充参数量统计。"""

from pathlib import Path
import pathlib
import csv
import sys
import torch


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return -1.0


def main() -> None:
    # 兼容 Linux 训练、Windows 分析场景：checkpoint 里可能序列化了 PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    root = Path(__file__).resolve().parents[2]
    yolov5_dir = root / "yolov5"
    if str(yolov5_dir) not in sys.path:
        # 让 torch.load 能解析 checkpoint 里引用的 yolov5.models.*
        sys.path.insert(0, str(yolov5_dir))

    summary_path = root / "outputs/results/stage1_ablation_summary.csv"

    exp_to_weight = {
        "ablation_exp01_baseline": root / "outputs/ablation_study/ablation_exp01_baseline/weights/best.pt",
        "ablation_exp02_ghost": root / "outputs/ablation_study/ablation_exp02_ghost/weights/best.pt",
        "ablation_exp03_shuffle": root / "outputs/ablation_study/ablation_exp03_shuffle/weights/best.pt",
        "ablation_exp05_coordatt": root / "outputs/ablation_study/ablation_exp05_coordatt/weights/best.pt",
        "ablation_exp06_siou": root / "outputs/ablation_study/ablation_exp06_siou/weights/best.pt",
        "ablation_exp07_eiou": root / "outputs/ablation_study/ablation_exp07_eiou/weights/best.pt",
    }

    if not summary_path.exists():
        raise FileNotFoundError(f"未找到汇总文件: {summary_path}")

    param_map = {}
    for exp, weight in exp_to_weight.items():
        if not weight.exists():
            param_map[exp] = {"params_total": "", "params_trainable": "", "params_M": ""}
            continue

        ckpt = torch.load(weight, map_location="cpu")
        model = ckpt.get("model") or ckpt.get("ema")
        if model is None:
            param_map[exp] = {"params_total": "", "params_trainable": "", "params_M": ""}
            continue

        model = model.float()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable == 0:
            # 部分导出的权重会关闭 requires_grad，这里按结构参数量回填
            trainable = total
        param_map[exp] = {
            "params_total": str(int(total)),
            "params_trainable": str(int(trainable)),
            "params_M": f"{total / 1_000_000:.3f}",
        }

    with open(summary_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    for extra in ["params_total", "params_trainable", "params_M"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    for row in rows:
        exp = row.get("experiment", "")
        row.update(param_map.get(exp, {"params_total": "", "params_trainable": "", "params_M": ""}))

    rows.sort(key=lambda x: to_float(x.get("map5095", "")), reverse=True)

    output_dir = root / "outputs/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_main = output_dir / "stage1_ablation_summary.csv"
    out_copy = output_dir / "stage1_ablation_summary_with_params.csv"

    for out in (out_main, out_copy):
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"UPDATED: {out_main}")
    print(f"SAVED:   {out_copy}")
    print("\nexperiment | map5095 | params_total | params_M")
    for row in rows:
        print(f"{row.get('experiment','')} | {row.get('map5095','')} | {row.get('params_total','')} | {row.get('params_M','')}")


if __name__ == "__main__":
    main()
