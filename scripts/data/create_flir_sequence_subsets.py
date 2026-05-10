#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description='为 FLIR 测试集生成按序列划分的评估子集')
    parser.add_argument(
        '--source-yaml',
        default=str(ROOT / 'data' / 'processed' / 'flir' / 'dataset.yaml'),
        help='原始数据集 YAML',
    )
    parser.add_argument(
        '--output-dir',
        default=str(ROOT / 'data' / 'processed' / 'flir' / 'sequence_eval'),
        help='输出目录',
    )
    parser.add_argument(
        '--subset',
        action='append',
        required=True,
        help='子集定义，格式为 seq_name=video_prefix，例如 seq006=ZAtDSNuZZjkZFvMAo',
    )
    return parser.parse_args()


def resolve_test_dir(dataset_yaml: Path, dataset_cfg: dict) -> Path:
    base_path = Path(dataset_cfg.get('path', ''))
    test_value = dataset_cfg.get('test') or dataset_cfg.get('val')
    if not test_value:
        raise ValueError('dataset.yaml 中未找到 test 或 val 字段')

    candidate = Path(test_value)
    if candidate.is_absolute():
        return candidate

    merged = base_path / candidate
    if merged.exists():
        return merged

    relative = dataset_yaml.parent / candidate
    if relative.exists():
        return relative

    raise FileNotFoundError(f'未找到测试图片目录: {test_value}')


def resolve_dataset_entry(dataset_yaml: Path, dataset_cfg: dict, key: str):
    value = dataset_cfg.get(key)
    if value is None:
        return None

    entry_path = Path(value)
    if entry_path.is_absolute():
        return str(entry_path)

    base_path = Path(dataset_cfg.get('path', ''))
    merged = base_path / entry_path
    if merged.exists():
        return str(merged)

    relative = dataset_yaml.parent / entry_path
    if relative.exists():
        return str(relative)

    return value


def collect_images(test_dir: Path, prefix: str) -> list[Path]:
    image_list = sorted(test_dir.glob(f'video-{prefix}-*'))
    return [path.resolve() for path in image_list if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]


def main():
    args = parse_args()

    dataset_yaml = Path(args.source_yaml).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with dataset_yaml.open('r', encoding='utf-8') as handle:
        dataset_cfg = yaml.safe_load(handle) or {}

    test_dir = resolve_test_dir(dataset_yaml, dataset_cfg)
    nc = dataset_cfg.get('nc')
    names = dataset_cfg.get('names')
    train_value = resolve_dataset_entry(dataset_yaml, dataset_cfg, 'train')
    val_value = resolve_dataset_entry(dataset_yaml, dataset_cfg, 'val')

    created = []
    for item in args.subset:
        if '=' not in item:
            raise ValueError(f'非法 subset 参数: {item}')

        seq_name, prefix = item.split('=', 1)
        seq_name = seq_name.strip()
        prefix = prefix.strip()
        if not seq_name or not prefix:
            raise ValueError(f'非法 subset 参数: {item}')

        images = collect_images(test_dir, prefix)
        if not images:
            raise FileNotFoundError(f'未找到匹配前缀 {prefix} 的测试图片')

        list_path = output_dir / f'{seq_name}_test.txt'
        yaml_path = output_dir / f'{seq_name}_dataset.yaml'

        with list_path.open('w', encoding='utf-8', newline='\n') as handle:
            for image_path in images:
                handle.write(f'{image_path.as_posix()}\n')

        subset_cfg = {
            'path': str(test_dir.parents[1]),
            'train': train_value,
            'val': val_value,
            'test': str(list_path),
            'nc': nc,
            'names': names,
        }

        with yaml_path.open('w', encoding='utf-8', newline='\n') as handle:
            yaml.safe_dump(subset_cfg, handle, sort_keys=False, allow_unicode=True)

        created.append((seq_name, prefix, len(images), list_path, yaml_path))

    for seq_name, prefix, count, list_path, yaml_path in created:
        print(f'{seq_name}: prefix={prefix}, images={count}')
        print(f'  list: {list_path}')
        print(f'  yaml: {yaml_path}')


if __name__ == '__main__':
    main()