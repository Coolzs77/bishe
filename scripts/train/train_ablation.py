#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测模型消融训练脚本（严格控制变量）。

阶段一（单变量实验）：
- Exp1: baseline
- Exp2: baseline + Ghost
- Exp3: baseline + Attention
- Exp4: baseline + EIoU
- Exp5: baseline + Focal

阶段二（组合实验，可选）：
- Exp6: Ghost + Attention
- Exp7: Ghost + EIoU
- Exp8: Attention + EIoU
- Exp9: Ghost + Attention + EIoU + Focal
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"项目根目录: {PROJECT_ROOT}")

STAGE1_EXPERIMENTS = [
    {
        "id": "exp1",
        "name": "ablation_exp1_baseline",
        "yaml": "model/yolov5/configs/yolov5s_base.yaml",
        "hyp": None,
        "desc": "[Exp1] 基线模型 - YOLOv5s",
    },
    {
        "id": "exp2",
        "name": "ablation_exp2_lightweight",
        "yaml": "model/yolov5/configs/yolov5s_lightweight.yaml",
        "hyp": None,
        "desc": "[Exp2] 基线 + Ghost",
    },
    {
        "id": "exp3",
        "name": "ablation_exp3_attention",
        "yaml": "model/yolov5/configs/yolov5s_attention.yaml",
        "hyp": None,
        "desc": "[Exp3] 基线 + Attention",
    },
    {
        "id": "exp4",
        "name": "ablation_exp4_eiou",
        "yaml": "model/yolov5/configs/yolov5s_base.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp4] 基线 + EIoU",
    },
    {
        "id": "exp5",
        "name": "ablation_exp5_focal",
        "yaml": "model/yolov5/configs/yolov5s_base.yaml",
        "hyp": "configs/ablation/hyp_focal_only.yaml",
        "desc": "[Exp5] 基线 + Focal",
    },
]

STAGE2_EXPERIMENTS = [
    {
        "id": "exp6",
        "name": "ablation_exp6_ghost_attention",
        "yaml": "model/yolov5/configs/yolov5s_ghost_attention.yaml",
        "hyp": None,
        "desc": "[Exp6] Ghost + Attention",
    },
    {
        "id": "exp7",
        "name": "ablation_exp7_ghost_eiou",
        "yaml": "model/yolov5/configs/yolov5s_lightweight.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp7] Ghost + EIoU",
    },
    {
        "id": "exp8",
        "name": "ablation_exp8_attention_eiou",
        "yaml": "model/yolov5/configs/yolov5s_attention.yaml",
        "hyp": "configs/ablation/hyp_eiou_only.yaml",
        "desc": "[Exp8] Attention + EIoU",
    },
    {
        "id": "exp9",
        "name": "ablation_exp9_all",
        "yaml": "model/yolov5/configs/yolov5s_ghost_attention.yaml",
        "hyp": "configs/ablation/hyp_eiou_focal.yaml",
        "desc": "[Exp9] Ghost + Attention + EIoU + Focal",
    },
]


def train_experiment(exp_name, yaml_path, desc, hyp_path, dataset_yaml_path, args):
    """训练单个实验"""
    print(f"\n{'=' * 70}")
    print(f"开始运行: {desc}")
    print(f"{'=' * 70}")

    cmd = [
        'python', 'train.py',
        '--img', str(args.img),
        '--batch', str(args.batch),
        '--epochs', str(args.epochs),
        '--data', str(dataset_yaml_path),
        '--cfg', str(PROJECT_ROOT / yaml_path),
        '--weights', 'yolov5s.pt',
        '--device', str(args.device),
        '--name', exp_name,
        '--project', str(PROJECT_ROOT / 'outputs/ablation_study'),
        '--patience', str(args.patience),
        '--cos-lr',
    ]

    if hyp_path:
        cmd.extend(['--hyp', str(PROJECT_ROOT / hyp_path)])

    try:
        # 注意：需要在yolov5目录中运行
        yolov5_dir = PROJECT_ROOT / 'yolov5'
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
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="仅运行单个实验，可填 exp1~exp9、数字(1~9) 或实验名(ablation_exp3_attention)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出当前 stage 可运行实验并退出",
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


def main():
    args = parse_args()
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

    dataset_yaml = PROJECT_ROOT / 'data/processed/flir/dataset.yaml'
    if not dataset_yaml.exists():
        print(f"未找到数据集配置: {dataset_yaml}")
        sys.exit(1)

    print(f"\n{'*' * 70}")
    print("* 检测模型消融实验")
    print(f"* 项目路径: {PROJECT_ROOT}")
    print(f"* 运行阶段: {args.stage}")
    if args.only:
        print(f"* 单实验模式: {args.only}")
    print("* 类别策略: 仅 person/car")
    print(f"* 实验总数: {len(experiments)}")
    print(f"{'*' * 70}")

    # 检查前置条件
    if not dataset_yaml.exists():
        print(f"未找到数据集配置: {dataset_yaml}")
        sys.exit(1)

    for exp in experiments:
        cfg_file = PROJECT_ROOT / exp['yaml']
        if not cfg_file.exists():
            print(f"未找到模型配置 {exp['name']}: {cfg_file}")
            sys.exit(1)
        if exp['hyp'] is not None:
            hyp_file = PROJECT_ROOT / exp['hyp']
            if not hyp_file.exists():
                print(f"未找到超参数配置 {exp['name']}: {hyp_file}")
                sys.exit(1)

    results = {}

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp['desc']}")
        success = train_experiment(
            exp['name'],
            exp['yaml'],
            exp['desc'],
            exp['hyp'],
            dataset_yaml,
            args,
        )

        results[exp['name']] = {
            'desc': exp['desc'],
            'status': '✅' if success else '❌',
            'yaml': exp['yaml'],
            'hyp': exp['hyp'] if exp['hyp'] else '-',
        }

    # 打印总结
    print(f"\n{'=' * 70}")
    print("消融实验汇总")
    print(f"{'=' * 70}")

    for name, result in results.items():
        print(f"{result['status']} {name} | {result['desc']} | cfg={result['yaml']} | hyp={result['hyp']}")

    print("\n结果目录: outputs/ablation_study/")
    print("本次请求的实验已执行完成。\n")


if __name__ == '__main__':
    main()