#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测模型消融训练脚本。

支持两种模式：
- controlled: 严格控变量，复用统一训练参数。
- optimal: 每个实验可独立调参，尽量逼近各自最优性能。
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"项目根目录: {PROJECT_ROOT}")

CANONICAL_EXPERIMENTS = [
    {
        "id": "exp1",
        "name": "ablation_exp01_baseline",
        "yaml": "model/yolov5/configs/yolov5s_base.yaml",
        "hyp": None,
        "desc": "[Exp1] 基线模型 - YOLOv5s",
        "stage": "stage1",
    },
    {
        "id": "exp2",
        "name": "ablation_exp02_ghost",
        "yaml": "model/yolov5/configs/yolov5s_lightweight.yaml",
        "hyp": None,
        "desc": "[Exp2] 基线 + Ghost-C3",
        "stage": "stage1",
    },
    {
        "id": "exp3",
        "name": "ablation_exp03_shuffle",
        "yaml": "model/yolov5/configs/yolov5s_shuffle.yaml",
        "hyp": None,
        "desc": "[Exp3] 基线 + Shuffle-C3",
        "stage": "stage1",
    },
    {
        "id": "exp4",
        "name": "ablation_exp04_attention",
        "yaml": "model/yolov5/configs/yolov5s_attention.yaml",
        "hyp": None,
        "desc": "[Exp4] 基线 + Attention",
        "stage": "stage1",
    },
    {
        "id": "exp5",
        "name": "ablation_exp05_coordatt",
        "yaml": "model/yolov5/configs/yolov5s_coordatt.yaml",
        "hyp": None,
        "desc": "[Exp5] 基线 + CoordAttention",
        "stage": "stage1",
    },
    {
        "id": "exp6",
        "name": "ablation_exp06_siou",
        "yaml": "model/yolov5/configs/yolov5s_base.yaml",
        "hyp": "configs/ablation/hyp_siou_only.yaml",
        "desc": "[Exp6] 基线 + SIoU",
        "stage": "stage1",
    },
    {
        "id": "exp7",
        "name": "ablation_exp07_eiou",
        "yaml": "model/yolov5/configs/yolov5s_base.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp7] 基线 + EIoU",
        "stage": "stage1",
    },
    {
        "id": "exp8",
        "name": "ablation_exp08_ghost_attention",
        "yaml": "model/yolov5/configs/yolov5s_ghost_attention.yaml",
        "hyp": None,
        "desc": "[Exp8] Ghost-C3 + Attention",
        "stage": "stage2",
    },
    {
        "id": "exp9",
        "name": "ablation_exp09_ghost_eiou",
        "yaml": "model/yolov5/configs/yolov5s_lightweight.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp9] Ghost-C3 + EIoU",
        "stage": "stage2",
    },
    {
        "id": "exp10",
        "name": "ablation_exp10_attention_eiou",
        "yaml": "model/yolov5/configs/yolov5s_attention.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp10] Attention + EIoU",
        "stage": "stage2",
    },
    {
        "id": "exp11",
        "name": "ablation_exp11_shuffle_coordatt",
        "yaml": "model/yolov5/configs/yolov5s_shuffle_coordatt.yaml",
        "hyp": None,
        "desc": "[Exp11] Shuffle-C3 + CoordAttention",
        "stage": "stage2",
    },
    {
        "id": "exp12",
        "name": "ablation_exp12_shuffle_coordatt_siou",
        "yaml": "model/yolov5/configs/yolov5s_shuffle_coordatt.yaml",
        "hyp": "configs/ablation/hyp_siou_only.yaml",
        "desc": "[Exp12] Shuffle-C3 + CoordAttention + SIoU",
        "stage": "stage2",
    },
    {
        "id": "exp13",
        "name": "ablation_exp13_shuffle_coordatt_eiou",
        "yaml": "model/yolov5/configs/yolov5s_shuffle_coordatt.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp13] Shuffle-C3 + CoordAttention + EIoU",
        "stage": "stage2",
    },
]

STAGE1_EXPERIMENTS = [exp for exp in CANONICAL_EXPERIMENTS if exp["stage"] == "stage1"]
STAGE2_EXPERIMENTS = [exp for exp in CANONICAL_EXPERIMENTS if exp["stage"] == "stage2"]


def parse_args():
    parser = argparse.ArgumentParser(description="运行检测模型消融实验")
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "all"],
        default="stage1",
        help="stage1: 单变量实验；stage2: 组合实验；all: 全部",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=1, help="每隔 N 轮验证一次")
    parser.add_argument("--workers", type=int, default=16, help="dataloader 线程数")
    parser.add_argument(
        "--cache",
        choices=["none", "ram", "disk"],
        default="ram",
        help="图像缓存模式：none/ram/disk",
    )
    parser.add_argument("--noval", action="store_true", help="仅最后一轮验证（提速）")
    parser.add_argument("--noplots", action="store_true", help="不生成图像可视化（提速）")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="仅运行单个实验，可填 exp1~exp13、数字(1~13) 或实验名(ablation_exp03_shuffle)",
    )
    parser.add_argument("--list", action="store_true", help="仅列出当前 stage 可运行实验并退出")
    parser.add_argument(
        "--profile",
        choices=["controlled", "optimal"],
        default="controlled",
        help="训练配置模式：controlled=统一参数；optimal=每实验独立调参",
    )
    parser.add_argument(
        "--profile-config",
        type=str,
        default="configs/ablation/train_profile_optimal.yaml",
        help="optimal 模式下的每实验训练参数配置文件",
    )
    parser.add_argument(
        "--allow-profile-hyp-override",
        action="store_true",
        help="允许 optimal 配置覆盖实验默认 hyp 文件",
    )
    return parser.parse_args()


def _match_single_experiment(experiments, keyword):
    token = keyword.strip().lower()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1]

    for exp in experiments:
        exp_id = exp["id"].lower()
        candidates = {
            exp_id,
            exp_id.replace("exp", ""),
            exp["name"].lower(),
        }
        if token in candidates:
            return exp
    return None


def load_profile_config(args):
    if args.profile != "optimal":
        return {}
    config_path = PROJECT_ROOT / args.profile_config
    if not config_path.exists():
        raise FileNotFoundError(f"未找到 profile 配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"profile 配置格式错误: {config_path}")
    return cfg


def merge_train_options(args, exp, profile_cfg):
    options = {
        "img": args.img,
        "batch": args.batch,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
        "weights": "yolov5s.pt",
        "optimizer": None,
        "label_smoothing": None,
        "hyp": exp.get("hyp"),
    }

    if args.profile != "optimal":
        return options

    global_cfg = profile_cfg.get("global", {})
    exp_cfg = profile_cfg.get("experiments", {}).get(exp["name"], {})

    override_keys = [
        "img",
        "batch",
        "epochs",
        "patience",
        "workers",
        "cache",
        "weights",
        "optimizer",
        "label_smoothing",
    ]
    for key in override_keys:
        if key in global_cfg:
            options[key] = global_cfg[key]
    for key in override_keys:
        if key in exp_cfg:
            options[key] = exp_cfg[key]

    if args.allow_profile_hyp_override and "hyp" in global_cfg:
        options["hyp"] = global_cfg["hyp"]
    if args.allow_profile_hyp_override and "hyp" in exp_cfg:
        options["hyp"] = exp_cfg["hyp"]

    return options


def train_experiment(exp_name, yaml_path, desc, train_options, dataset_yaml_path, args):
    """训练单个实验"""
    print(f"\n{'=' * 70}")
    print(f"开始运行: {desc}")
    print(f"{'=' * 70}")

    cmd = [
        "python",
        "train.py",
        "--img",
        str(train_options["img"]),
        "--batch-size",
        str(train_options["batch"]),
        "--epochs",
        str(train_options["epochs"]),
        "--val-interval",
        str(args.val_interval),
        "--data",
        str(dataset_yaml_path),
        "--cfg",
        str(PROJECT_ROOT / yaml_path),
        "--weights",
        str(train_options["weights"]),
        "--device",
        str(args.device),
        "--workers",
        str(train_options["workers"]),
        "--name",
        exp_name,
        "--project",
        str(PROJECT_ROOT / "outputs/ablation_study"),
        "--patience",
        str(train_options["patience"]),
        "--cos-lr",
    ]

    if train_options["cache"] != "none":
        cmd.extend(["--cache", str(train_options["cache"])])
    if args.noval:
        cmd.append("--noval")
    if args.noplots:
        cmd.append("--noplots")
    if train_options["optimizer"]:
        cmd.extend(["--optimizer", str(train_options["optimizer"])])
    if train_options["label_smoothing"] is not None:
        cmd.extend(["--label-smoothing", str(train_options["label_smoothing"])])
    if train_options["hyp"]:
        cmd.extend(["--hyp", str(PROJECT_ROOT / str(train_options["hyp"]))])

    try:
        yolov5_dir = PROJECT_ROOT / "yolov5"
        if not yolov5_dir.exists():
            print(f"未找到 YOLOv5 目录: {yolov5_dir}")
            print(f"请先克隆: git clone https://github.com/ultralytics/yolov5 {yolov5_dir}")
            return False

        subprocess.run(cmd, check=True, cwd=str(yolov5_dir))
        print(f"完成: {desc}")
        return True
    except Exception as e:
        print(f"失败: {desc} -> {e}")
        return False


def main():
    args = parse_args()
    profile_cfg = load_profile_config(args)

    experiments = []
    if args.stage in ("stage1", "all"):
        experiments.extend(STAGE1_EXPERIMENTS)
    if args.stage in ("stage2", "all"):
        experiments.extend(STAGE2_EXPERIMENTS)

    if args.list:
        print("\n当前可运行实验：")
        for exp in experiments:
            print(f"- {exp['id']} | {exp['name']} | {exp['desc']}")
        return

    if args.only:
        selected = _match_single_experiment(experiments, args.only)
        if selected is None:
            print(f"未找到指定实验: {args.only}")
            print("可选实验：")
            for exp in experiments:
                print(f"- {exp['id']} | {exp['name']} | {exp['desc']}")
            sys.exit(1)
        experiments = [selected]

    dataset_yaml = PROJECT_ROOT / "data/processed/flir/dataset.yaml"
    if not dataset_yaml.exists():
        print(f"未找到数据集配置: {dataset_yaml}")
        sys.exit(1)

    print(f"\n{'*' * 70}")
    print("* 检测模型消融实验")
    print(f"* 项目路径: {PROJECT_ROOT}")
    print(f"* 运行阶段: {args.stage}")
    print(f"* 训练模式: {args.profile}")
    if args.profile == "optimal":
        print(f"* Profile 配置: {PROJECT_ROOT / args.profile_config}")
    if args.only:
        print(f"* 单实验模式: {args.only}")
    print("* 类别策略: 仅 person/car")
    print(
        f"* workers: {args.workers} | cache: {args.cache} | "
        f"val_interval: {args.val_interval} | noval: {args.noval} | noplots: {args.noplots}"
    )
    print(f"* 实验总数: {len(experiments)}")
    print(f"{'*' * 70}")

    for exp in experiments:
        cfg_file = PROJECT_ROOT / exp["yaml"]
        if not cfg_file.exists():
            print(f"未找到模型配置 {exp['name']}: {cfg_file}")
            sys.exit(1)

        train_options = merge_train_options(args, exp, profile_cfg)
        if train_options["hyp"] is not None:
            hyp_file = PROJECT_ROOT / str(train_options["hyp"])
            if not hyp_file.exists():
                print(f"未找到超参数配置 {exp['name']}: {hyp_file}")
                sys.exit(1)

    results = {}

    for i, exp in enumerate(experiments, 1):
        train_options = merge_train_options(args, exp, profile_cfg)
        print(f"\n[{i}/{len(experiments)}] {exp['desc']}")
        print(
            f"  参数: img={train_options['img']} batch={train_options['batch']} "
            f"epochs={train_options['epochs']} patience={train_options['patience']} "
            f"optimizer={train_options['optimizer']} hyp={train_options['hyp']}"
        )
        success = train_experiment(
            exp["name"],
            exp["yaml"],
            exp["desc"],
            train_options,
            dataset_yaml,
            args,
        )

        results[exp["name"]] = {
            "desc": exp["desc"],
            "status": "SUCCESS" if success else "FAIL",
            "yaml": exp["yaml"],
            "hyp": train_options["hyp"] if train_options["hyp"] else "-",
        }

    print(f"\n{'=' * 70}")
    print("消融实验汇总")
    print(f"{'=' * 70}")
    for name, result in results.items():
        print(f"{result['status']} {name} | {result['desc']} | cfg={result['yaml']} | hyp={result['hyp']}")

    print("\n结果目录: outputs/ablation_study/")
    print("本次请求的实验已执行完成。\n")


if __name__ == "__main__":
    main()