#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv5 单模型训练入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOV5_DIR = PROJECT_ROOT / 'yolov5'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='运行单个 YOLOv5 检测模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python scripts/train/train_yolov5.py --config configs/train_config.yaml\n'
            '  python scripts/train/train_yolov5.py --config configs/train_config.yaml --epochs 100 --batch-size 16\n'
            '  python scripts/train/train_yolov5.py --resume outputs/weights/infrared_detection/weights/last.pt'
        ),
    )
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='训练配置文件路径')
    parser.add_argument('--weights', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--cfg', type=str, default=None, help='模型结构配置文件路径')
    parser.add_argument('--data', type=str, default=None, help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--img-size', type=int, default=None, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default=None, help='训练设备，如 0 或 cpu')
    parser.add_argument('--workers', type=int, default=None, help='dataloader 线程数')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值')
    parser.add_argument('--cache', nargs='?', const='ram', choices=['ram', 'disk'], default=None, help='启用图像缓存；不带值默认为 ram')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存并覆盖配置文件')
    parser.add_argument('--name', type=str, default=None, help='实验名称')
    parser.add_argument('--project', type=str, default=None, help='输出目录')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default=None, help='优化器类型')
    parser.add_argument('--hyp', type=str, default=None, help='超参数配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='断点续训权重路径')
    parser.add_argument('--freeze', type=int, default=None, help='冻结层数')
    parser.add_argument('--cos-lr', action='store_true', help='启用余弦学习率')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖已有实验目录')
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f'未找到训练配置文件: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f'训练配置格式错误: {config_path}')
    return config


def config_get(config: dict, path: str, default=None):
    current = config
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def cli_or_config(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def resolve_project_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_options(args: argparse.Namespace, config: dict) -> dict:
    if args.no_cache:
        cache_value = 'none'
    elif args.cache is not None:
        cache_value = args.cache
    else:
        cache_value = config_get(config, 'train.cache', 'none')

    return {
        'config_path': resolve_project_path(args.config),
        'weights': cli_or_config(args.weights, config_get(config, 'model.weights', 'yolov5/yolov5s.pt')),
        'cfg': cli_or_config(args.cfg, config_get(config, 'model.cfg', None)),
        'data': cli_or_config(args.data, config_get(config, 'data', 'data/processed/flir/dataset.yaml')),
        'epochs': cli_or_config(args.epochs, config_get(config, 'train.epochs', 50)),
        'batch_size': cli_or_config(args.batch_size, config_get(config, 'train.batch_size', 16)),
        'img_size': cli_or_config(args.img_size, config_get(config, 'train.img_size', 640)),
        'device': str(cli_or_config(args.device, config_get(config, 'runtime.device', '0'))),
        'workers': cli_or_config(args.workers, config_get(config, 'runtime.workers', 8)),
        'patience': cli_or_config(args.patience, config_get(config, 'runtime.patience', 50)),
        'cache': cache_value,
        'name': cli_or_config(args.name, config_get(config, 'name', 'infrared_detection')),
        'project': cli_or_config(args.project, config_get(config, 'project', 'outputs/weights')),
        'optimizer': cli_or_config(args.optimizer, config_get(config, 'optimizer.type', 'SGD')),
        'hyp': cli_or_config(args.hyp, config_get(config, 'runtime.hyp', None)),
        'resume': cli_or_config(args.resume, config_get(config, 'runtime.resume', None)),
        'freeze': cli_or_config(args.freeze, config_get(config, 'runtime.freeze', 0)),
        'cos_lr': args.cos_lr or bool(config_get(config, 'runtime.cos_lr', False)),
        'exist_ok': args.exist_ok or bool(config_get(config, 'exist_ok', False)),
    }


def validate_options(options: dict) -> dict:
    train_py = YOLOV5_DIR / 'train.py'
    if not train_py.exists():
        raise FileNotFoundError(f'未找到 YOLOv5 训练入口: {train_py}')

    validated = dict(options)
    validated['train_py'] = train_py
    validated['data_path'] = resolve_project_path(options['data'])
    validated['project_path'] = resolve_project_path(options['project'])
    validated['weights_path'] = resolve_project_path(options['weights']) if options['weights'] else None
    validated['cfg_path'] = resolve_project_path(options['cfg']) if options['cfg'] else None
    validated['hyp_path'] = resolve_project_path(options['hyp']) if options['hyp'] else None
    validated['resume_path'] = resolve_project_path(options['resume']) if options['resume'] else None

    if not validated['data_path'].exists():
        raise FileNotFoundError(f"未找到数据集配置: {validated['data_path']}")
    if validated['weights_path'] is not None and validated['resume_path'] is None and not validated['weights_path'].exists():
        raise FileNotFoundError(f"未找到预训练权重: {validated['weights_path']}")
    if validated['cfg_path'] is not None and not validated['cfg_path'].exists():
        raise FileNotFoundError(f"未找到模型配置文件: {validated['cfg_path']}")
    if validated['hyp_path'] is not None and not validated['hyp_path'].exists():
        raise FileNotFoundError(f"未找到超参数配置文件: {validated['hyp_path']}")
    if validated['resume_path'] is not None and not validated['resume_path'].exists():
        raise FileNotFoundError(f"未找到断点续训权重: {validated['resume_path']}")

    return validated


def build_train_command(options: dict) -> list[str]:
    if options['resume_path'] is not None:
        return [sys.executable, 'train.py', '--resume', str(options['resume_path'])]

    cmd = [
        sys.executable,
        'train.py',
        '--img',
        str(options['img_size']),
        '--batch',
        str(options['batch_size']),
        '--epochs',
        str(options['epochs']),
        '--data',
        str(options['data_path']),
        '--weights',
        str(options['weights_path']),
        '--project',
        str(options['project_path']),
        '--name',
        str(options['name']),
        '--device',
        str(options['device']),
        '--workers',
        str(options['workers']),
        '--patience',
        str(options['patience']),
        '--optimizer',
        str(options['optimizer']),
    ]

    if options['cfg_path'] is not None:
        cmd.extend(['--cfg', str(options['cfg_path'])])
    if options['cache'] != 'none':
        cmd.extend(['--cache', str(options['cache'])])
    if options['hyp_path'] is not None:
        cmd.extend(['--hyp', str(options['hyp_path'])])
    if options['freeze'] and int(options['freeze']) > 0:
        cmd.extend(['--freeze', str(options['freeze'])])
    if options['cos_lr']:
        cmd.append('--cos-lr')
    if options['exist_ok']:
        cmd.append('--exist-ok')

    return cmd


def print_summary(options: dict, cmd: list[str]) -> None:
    print('=' * 60)
    print('YOLOv5 单模型训练')
    print('=' * 60)
    print(f"配置文件: {options['config_path']}")
    if options['resume_path'] is not None:
        print('模式: resume')
        print(f"恢复权重: {options['resume_path']}")
    else:
        print(f"实验名称: {options['name']}")
        print(f"输出目录: {options['project_path']}")
        print(f"数据配置: {options['data_path']}")
        print(f"预训练权重: {options['weights_path']}")
        print(f"模型配置: {options['cfg_path'] if options['cfg_path'] else '默认'}")
        print(f"训练轮数: {options['epochs']}")
        print(f"批量大小: {options['batch_size']}")
        print(f"图像尺寸: {options['img_size']}")
        print(f"缓存模式: {options['cache']}")
        print(f"优化器: {options['optimizer']}")
        print(f"设备: {options['device']}")
        print(f"workers: {options['workers']}")
    print('执行命令:')
    print(' '.join(cmd))
    print('=' * 60)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    options = resolve_options(args, config)
    options['config_path'] = config_path
    options = validate_options(options)
    cmd = build_train_command(options)
    print_summary(options, cmd)
    subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))


if __name__ == '__main__':
    main()