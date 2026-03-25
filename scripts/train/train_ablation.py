#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测模型消融训练脚本。

配置边界：
- 单模型训练使用 configs/train_config.yaml。
- 消融实验训练使用 configs/ablation/train_profile_*.yaml。

支持两种 profile：
- controlled: 严格控变量，除实验结构差异外统一训练口径。
- optimal: 允许每个实验做有限度的独立调参。
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--epochs", type=int, default=None, help="临时覆盖 profile 中的 epochs")
    parser.add_argument("--batch", type=int, default=None, help="临时覆盖 profile 中的 batch")
    parser.add_argument("--img", type=int, default=None, help="临时覆盖 profile 中的 img")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=None, help="临时覆盖 profile 中的 patience")
    parser.add_argument("--val-interval", type=int, default=1, help="每隔 N 轮验证一次")
    parser.add_argument("--workers", type=int, default=None, help="临时覆盖 profile 中的 dataloader 线程数")
    parser.add_argument(
        "--cache",
        choices=["none", "ram", "disk"],
        default=None,
        help="临时覆盖 profile 中的图像缓存模式：none/ram/disk",
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
        default=None,
        help="profile 配置文件路径；默认按 profile 自动选择 train_profile_controlled.yaml 或 train_profile_optimal.yaml",
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


def resolve_profile_config_path(args) -> Path:
    if args.profile_config:
        return PROJECT_ROOT / args.profile_config
    return PROJECT_ROOT / "configs" / "ablation" / f"train_profile_{args.profile}.yaml"


def load_profile_config(args):
    config_path = resolve_profile_config_path(args)
    if not config_path.exists():
        raise FileNotFoundError(f"未找到 profile 配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"profile 配置格式错误: {config_path}")
    declared_profile = cfg.get("profile")
    if declared_profile != args.profile:
        raise ValueError(
            f"profile 配置与命令行不一致: --profile={args.profile}, config.profile={declared_profile}"
        )
    cfg["__path__"] = config_path
    return cfg


def apply_cli_overrides(options, args):
    overrides = {
        "img": args.img,
        "batch": args.batch,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "cache": args.cache,
    }
    for key, value in overrides.items():
        if value is not None:
            options[key] = value
    return options


def merge_train_options(args, exp, profile_cfg):
    global_cfg = profile_cfg.get("global", {})
    exp_cfg = profile_cfg.get("experiments", {}).get(exp["name"], {})
    rules = profile_cfg.get("rules", {})
    allow_hyp_override = bool(rules.get("allow_hyp_override", False))

    options = {
        "img": global_cfg.get("img", 640),
        "batch": global_cfg.get("batch", 16),
        "epochs": global_cfg.get("epochs", 100),
        "patience": global_cfg.get("patience", 20),
        "workers": global_cfg.get("workers", 8),
        "cache": global_cfg.get("cache", "ram"),
        "weights": global_cfg.get("weights", "yolov5/yolov5s.pt"),
        "optimizer": global_cfg.get("optimizer", None),
        "label_smoothing": global_cfg.get("label_smoothing", None),
        "cos_lr": bool(global_cfg.get("cos_lr", True)),
        "project": global_cfg.get("project", "outputs/ablation_study"),
        "hyp": exp.get("hyp"),
    }

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
        "cos_lr",
        "project",
    ]
    for key in override_keys:
        if key in exp_cfg:
            options[key] = exp_cfg[key]

    if allow_hyp_override and "hyp" in global_cfg:
        options["hyp"] = global_cfg["hyp"]
    if allow_hyp_override and "hyp" in exp_cfg:
        options["hyp"] = exp_cfg["hyp"]

    return apply_cli_overrides(options, args)


def train_experiment(exp_name, yaml_path, desc, train_options, dataset_yaml_path, args):
    """训练单个实验"""
    print(f"\n{'=' * 70}")
    print(f"开始运行: {desc}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable,
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
        str(PROJECT_ROOT / str(train_options["project"])),
        "--patience",
        str(train_options["patience"]),
    ]

    if train_options.get("cos_lr", False):
        cmd.append("--cos-lr")

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
    print(f"* Profile 配置: {profile_cfg['__path__']}")
    if args.only:
        print(f"* 单实验模式: {args.only}")
    print("* 类别策略: 仅 person/car")
    print(
        f"* val_interval: {args.val_interval} | noval: {args.noval} | noplots: {args.noplots}"
    )
    if any(value is not None for value in [args.epochs, args.batch, args.img, args.patience, args.workers, args.cache]):
        print(
            f"* 检测到临时覆盖: epochs={args.epochs} batch={args.batch} img={args.img} "
            f"patience={args.patience} workers={args.workers} cache={args.cache}"
        )
    print(
        f"* 说明: train_config.yaml 不参与消融训练；消融参数完全由 train_profile_{args.profile}.yaml 系列文件控制"
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