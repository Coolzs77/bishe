#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIRdata集预处理脚本
将FLIR热红外data集convert为YOLO训练格式
"""

import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import yaml


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='准备FLIRdata集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python prepare_flir.py --input data/raw/flir
  python prepare_flir.py --input data/raw/flir --output data/processed/flir --split-ratio 0.8
        '''
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='FLIRdata集原始路径')
    parser.add_argument('--output', type=str, default='data/processed/flir',
                        help='output目录')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--img-size', type=int, default=640,
                        help='目标img_size')
    parser.add_argument('--classes', type=str, default='person,car,bicycle',
                        help='检测classes，逗号分隔')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize部分results')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机seed')
    
    return parser.parse_args()


class FLIRDatasetConverter:
    """FLIRdata集convert器类"""
    
    # FLIRdata集classes映射到YOLOclasses索引
    class_mapping = {
        'person': 0,
        'bike': 2,      # 自行车
        'bicycle': 2,
        'car': 1,
        'motor': -1,    # 不使用
        'bus': -1,
        'train': -1,
        'truck': -1,
        'light': -1,
        'hydrant': -1,
        'sign': -1,
        'dog': -1,
        'skateboard': -1,
        'stroller': -1,
        'scooter': -1,
        'other vehicle': -1,
    }
    
    def __init__(self, input_dir, output_dir, classes_list, img_size=640, split_ratio=0.8):
        """
        初始化convert器
        
        参数:
            input目录: FLIRdata集原始路径
            output目录: convert后data保存路径
            classes列表: 目标检测classes列表
            img_size: outputimg_size
            划分比例: 训练集占总data的比例
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.classes_list = classes_list
        self.img_size = img_size
        self.split_ratio = split_ratio
        
        # create_output_dir结构
        self.train_img_dir = self.output_dir / 'images' / 'train'
        self.val_img_dir = self.output_dir / 'images' / 'val'
        self.train_label_dir = self.output_dir / 'labels' / 'train'
        self.val_label_dir = self.output_dir / 'labels' / 'val'
        self.calibration_dir = self.output_dir / 'calibration'
        
        for dir_path in [self.train_img_dir, self.val_img_dir, 
                    self.train_label_dir, self.val_label_dir,
                    self.calibration_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化statistics
        self.stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'total_instances': 0,
            'per_class_instances': {cls: 0 for cls in classes_list},
            'skipped_instances': 0,
        }
    
    def load_annotation_file(self, split='train'):
        """
        加载COCO格式的标注文件
        
        参数:
            划分: data划分，'train' 或 'val'
        
        返回:
            标注data字典
        """
        # 尝试多种可能的标注文件路径
        possible_paths = [
            self.input_dir / f'thermal_{split}' / 'coco.json',
            self.input_dir / f'images_thermal_{split}' / 'coco.json',
            self.input_dir / 'annotations' / f'instances_thermal_{split}.json',
        ]
        
        for annotation_path in possible_paths:
            if annotation_path.exists():
                print(f'加载标注文件: {annotation_path}')
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f'未找到{split}标注文件，尝试过: {possible_paths}')
    
    def convert_bbox_to_yolo_format(self, bbox, image_width, image_height):
        """
        将COCO格式边界框convert为YOLO格式
        
        参数:
            边界框: COCO格式 [x_min, y_min, width, height]
            image宽度: image宽度
            image高度: image高度
        
        返回:
            YOLO格式 [x_center, y_center, width, height] (归一化到0-1)
        """
        x_min, y_min, w, h = bbox
        
        # 计算中心点坐标
        x_center = (x_min + w / 2) / image_width
        y_center = (y_min + h / 2) / image_height
        
        # 归一化宽高
        w_normalized = w / image_width
        h_normalized = h / image_height
        
        # 确保值在[0, 1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_normalized = max(0, min(1, w_normalized))
        h_normalized = max(0, min(1, h_normalized))
        
        return [x_center, y_center, w_normalized, h_normalized]
    
    def process_data_split(self, annotation_data, is_train=True):
        """
        处理一个data集划分
        
        参数:
            标注data: COCO格式的标注data
            是否训练集: 是否为训练集
        
        返回:
            处理的imagecount
        """
        # 构建imageID到image信息的映射
        image_info_map = {image['id']: image for image in annotation_data['images']}
        
        # 构建classesID到classesname的映射
        class_name_map = {classes['id']: classes['name'] for classes in annotation_data['categories']}
        
        # 构建imageID到标注列表的映射
        image_annotations_map = {}
        for annotation in annotation_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_annotations_map:
                image_annotations_map[image_id] = []
            image_annotations_map[image_id].append(annotation)
        
        # 确定output目录
        img_output_dir = self.train_img_dir if is_train else self.val_img_dir
        label_output_dir = self.train_label_dir if is_train else self.val_label_dir
        
        # 处理每张image
        processed_count = 0
        for image_id, image_info in tqdm(image_info_map.items(), desc='处理image'):
            # 查找image文件
            image_filename = image_info['file_name']
            possible_image_paths = [
                self.input_dir / image_filename,
                self.input_dir / 'thermal_train' / 'data' / image_filename,
                self.input_dir / 'thermal_val' / 'data' / image_filename,
                self.input_dir / 'images_thermal_train' / image_filename,
                self.input_dir / 'images_thermal_val' / image_filename,
            ]
            
            image_path = None
            for path in possible_image_paths:
                if path.exists():
                    image_path = path
                    break
            
            if image_path is None:
                continue
            
            # 读取image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            image_height, image_width = image.shape[:2]
            
            # 获取该image的所有标注
            annotations_list = image_annotations_map.get(image_id, [])
            
            # convert标注为YOLO格式
            yolo_labels = []
            for annotation in annotations_list:
                class_name = class_name_map.get(annotation['category_id'], '').lower()
                class_idx = self.class_mapping.get(class_name, -1)
                
                # 跳过不需要的classes
                if class_idx == -1:
                    self.stats['skipped_instances'] += 1
                    continue
                
                if class_idx >= len(self.classes_list):
                    continue
                
                bbox = annotation['bbox']
                yolo_bbox = self.convert_bbox_to_yolo_format(bbox, image_width, image_height)
                yolo_labels.append([class_idx] + yolo_bbox)
                
                # 更新统计
                self.stats['total_instances'] += 1
                self.stats['per_class_instances'][self.classes_list[class_idx]] += 1
            
            # 保存image
            output_img_name = f'{image_id:06d}.jpg'
            output_img_path = img_output_dir / output_img_name
            cv2.imwrite(str(output_img_path), image)
            
            # 保存label文件
            output_label_name = f'{image_id:06d}.txt'
            output_label_path = label_output_dir / output_label_name
            with open(output_label_path, 'w', encoding='utf-8') as f:
                for label in yolo_labels:
                    line_content = ' '.join([str(label[0])] + [f'{val:.6f}' for val in label[1:]])
                    f.write(line_content + '\n')
            
            # 更新统计
            processed_count += 1
            self.stats['total_images'] += 1
            if is_train:
                self.stats['train_images'] += 1
            else:
                self.stats['val_images'] += 1
        
        return processed_count
    
    def create_calibration_dataset(self, num_samples=100):
        """
        创建用于INT8量化的校准data集
        
        参数:
            样本数: 校准样本count
        """
        print(f'\n创建量化校准data集 ({num_samples}张image)...')
        
        train_images = list(self.train_img_dir.glob('*.jpg'))
        if len(train_images) < num_samples:
            num_samples = len(train_images)
        
        selected_images = random.sample(train_images, num_samples)
        for image_path in selected_images:
            shutil.copy(image_path, self.calibration_dir / image_path.name)
        
        print(f'已复制 {num_samples} 张image到校准目录')
    
    def generate_dataset_config_file(self):
        """生成YOLO格式的data集config文件"""
        config_file_path = self.output_dir / 'dataset.yaml'
        
        config_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': len(self.classes_list),
            'names': self.classes_list,
        }
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f'\ndata集config已保存到: {config_file_path}')
    
    def perform_convert(self):
        """执行完整的data集convert流程"""
        print('=' * 50)
        print('FLIRdata集convert')
        print('=' * 50)
        print(f'input目录: {self.input_dir}')
        print(f'output目录: {self.output_dir}')
        print(f'目标classes: {self.classes_list}')
        print()
        
        # 处理训练集
        try:
            print('处理训练集...')
            train_annotations = self.load_annotation_file('train')
            self.process_data_split(train_annotations, is_train=True)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 处理验证集
        try:
            print('\n处理验证集...')
            val_annotations = self.load_annotation_file('val')
            self.process_data_split(val_annotations, is_train=False)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 创建校准data集
        self.create_calibration_dataset()
        
        # 生成config文件
        self.generate_dataset_config_file()
        
        # 打印statistics
        print('\n' + '=' * 50)
        print('convert完成!')
        print('=' * 50)
        print(f"总image数: {self.stats['total_images']}")
        print(f"训练image: {self.stats['train_images']}")
        print(f"验证image: {self.stats['val_images']}")
        print(f"总实例数: {self.stats['total_instances']}")
        print(f"跳过实例: {self.stats['skipped_instances']}")
        print('\n各classes实例数:')
        for class_name, count in self.stats['per_class_instances'].items():
            print(f"  {class_name}: {count}")


def main():
    """主函数"""
    args = parse_args()
    
    # set_random_seed以确保可复现性
    random.seed(args.seed)
    
    # 解析classes列表
    classes_list = [cls.strip() for cls in args.classes.split(',')]
    
    # 创建convert器并执行convert
    converter = FLIRDatasetConverter(
        input_dir=args.input,
        output_dir=args.output,
        classes_list=classes_list,
        img_size=args.img_size,
        split_ratio=args.split_ratio
    )
    
    converter.perform_convert()


if __name__ == '__main__':
    main()
