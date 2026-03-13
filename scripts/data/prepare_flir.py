#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FLIR红外数据预处理 - 简化版"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """COCO格式 -> YOLO格式"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [x_center, y_center, w_norm, h_norm]


def process_flir(input_dir, output_dir):
    """
    处理FLIR数据集

    input_dir: data/raw/flir
    output_dir: data/processed/flir
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 类别映射（只保留常用的3类）
    CLASS_MAP = {
        'person': 0,
        'car': 1,
        'bicycle': 2,
        'bike': 2,
    }
    CLASS_NAMES = ['person', 'car', 'bicycle']

    print('=' * 60)
    print('FLIR红外数据预处理')
    print('=' * 60)
    print(f'输入: {input_path}')
    print(f'输出: {output_path}')
    print(f'类别: {CLASS_NAMES}')
    print('=' * 60 + '\n')

    # 处理训练集和验证集
    for split in ['train', 'val']:
        print(f'\n处理 {split} 集...')

        # 路径
        base_dir = input_path / f'images_thermal_{split}'
        img_dir = base_dir / 'data'
        anno_file = base_dir / 'coco.json'

        out_img_dir = output_path / 'images' / split
        out_lbl_dir = output_path / 'labels' / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        # 检查文件是否存在
        if not anno_file.exists():
            print(f'  ✗ 未找到标注文件: {anno_file}')
            continue

        if not img_dir.exists():
            print(f'  ✗ 未找到图像目录: {img_dir}')
            continue

        print(f'  标注文件: {anno_file}')
        print(f'  图像目录: {img_dir}')

        # 加载标注
        with open(anno_file, 'r') as f:
            data = json.load(f)

        # 构建映射
        img_map = {img['id']: img for img in data['images']}
        cat_map = {cat['id']: cat['name'].lower() for cat in data['categories']}

        anno_map = {}
        for anno in data['annotations']:
            img_id = anno['image_id']
            if img_id not in anno_map:
                anno_map[img_id] = []
            anno_map[img_id].append(anno)

        # 调试：打印前3个文件名
        print(f'\n  前3个文件名示例:')
        for i, (img_id, img_info) in enumerate(list(img_map.items())[:3]):
            print(f'    {img_info["file_name"]}')

        # 列出实际的图像文件（前3个）
        actual_files = list(img_dir.glob('*.jpg'))[:3]
        print(f'\n  实际文件示例:')
        for f in actual_files:
            print(f'    {f.name}')
        print()

        # 处理每张图像
        stats = {'total': 0, 'skipped': 0, 'instances': 0}

        for img_id, img_info in tqdm(img_map.items(), desc=f'  {split}'):
            # 获取文件名（可能包含路径前缀）
            img_name = img_info['file_name']

            # 尝试多种路径匹配方式
            possible_paths = [
                img_dir / img_name,  # 直接拼接
                img_dir / Path(img_name).name,  # 只用文件名
                base_dir / img_name,  # 从base_dir拼接
                input_path / img_name,  # 从输入根目录拼接
            ]

            img_path = None
            for path in possible_paths:
                if path.exists():
                    img_path = path
                    break

            if img_path is None:
                stats['skipped'] += 1
                continue

            # 获取标注
            annos = anno_map.get(img_id, [])
            img_w = img_info['width']
            img_h = img_info['height']

            # 转换为YOLO格式
            yolo_labels = []
            for anno in annos:
                cat_name = cat_map.get(anno['category_id'], '')
                cls_id = CLASS_MAP.get(cat_name, -1)

                if cls_id == -1:
                    continue

                bbox = anno['bbox']
                yolo_bbox = convert_bbox_to_yolo(bbox, img_w, img_h)
                yolo_labels.append([cls_id] + yolo_bbox)
                stats['instances'] += 1

            # 生成新文件名
            out_name = Path(img_path.name).stem + '.jpg'

            # 保存图像
            out_img_path = out_img_dir / out_name
            shutil.copy(img_path, out_img_path)

            # 保存标签
            out_lbl_path = out_lbl_dir / out_name.replace('.jpg', '.txt')
            with open(out_lbl_path, 'w') as f:
                for label in yolo_labels:
                    line = f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
                    f.write(line)

            stats['total'] += 1

        print(f'  ✓ 完成: {stats["total"]} 张图像, {stats["instances"]} 个实例')
        if stats['skipped'] > 0:
            print(f'  ⚠ 跳过: {stats["skipped"]} 张')

    # 生成dataset.yaml
    yaml_content = f"""# FLIR红外数据集配置
path: {output_path.absolute()}
train: images/train
val: images/val

# 类别
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f'\n✓ 配置文件已保存: {yaml_path}')
    print('\n' + '=' * 60)
    print('预处理完成!')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='FLIR红外数据预处理')
    parser.add_argument('--input', type=str,
                        default='data/raw/flir',
                        help='输入目录')
    parser.add_argument('--output', type=str,
                        default='data/processed/flir',
                        help='输出目录')

    args = parser.parse_args()

    # 转换为绝对路径
    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output

    process_flir(input_dir, output_dir)


if __name__ == '__main__':
    main()