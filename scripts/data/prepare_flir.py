#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIR数据集预处理脚本
将FLIR热红外数据集转换为YOLO训练格式
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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='准备FLIR数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python prepare_flir.py --input data/raw/flir
  python prepare_flir.py --input data/raw/flir --output data/processed/flir --split-ratio 0.8
        '''
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='FLIR数据集原始路径')
    parser.add_argument('--output', type=str, default='data/processed/flir',
                        help='输出目录')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--img-size', type=int, default=640,
                        help='目标图像尺寸')
    parser.add_argument('--classes', type=str, default='person,car,bicycle',
                        help='检测类别，逗号分隔')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化部分结果')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


class FLIRDatasetConverter:
    """FLIR数据集转换器类"""
    
    # FLIR数据集类别映射到YOLO类别索引
    category_mapping = {
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
    
    def __init__(self, input_dir, output_dir, class_list, img_size=640, split_ratio=0.8):
        """
        初始化转换器
        
        参数:
            输入目录: FLIR数据集原始路径
            输出目录: 转换后数据保存路径
            类别列表: 目标检测类别列表
            图像尺寸: 输出图像尺寸
            划分比例: 训练集占总数据的比例
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_list = class_list
        self.img_size = img_size
        self.split_ratio = split_ratio
        
        # 创建输出目录结构
        self.train_img_dir = self.output_dir / 'images' / 'train'
        self.val_img_dir = self.output_dir / 'images' / 'val'
        self.train_label_dir = self.output_dir / 'labels' / 'train'
        self.val_label_dir = self.output_dir / 'labels' / 'val'
        self.calib_dir = self.output_dir / 'calibration'
        
        for directory in [self.train_img_dir, self.val_img_dir, 
                    self.train_label_dir, self.val_label_dir,
                    self.calib_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        self.stats = {
            '总图像数': 0,
            '训练图像数': 0,
            '验证图像数': 0,
            '总实例数': 0,
            '各类别实例数': {cls: 0 for cls in class_list},
            '跳过实例数': 0,
        }
    
    def load_annotation_file(self, split='train'):
        """
        加载COCO格式的标注文件
        
        参数:
            划分: 数据划分，'train' 或 'val'
        
        返回:
            标注数据字典
        """
        # 尝试多种可能的标注文件路径
        possible_paths = [
            self.input_dir / f'thermal_{split}' / 'coco.json',
            self.input_dir / f'images_thermal_{split}' / 'coco.json',
            self.input_dir / 'annotations' / f'instances_thermal_{split}.json',
        ]
        
        for anno_path in possible_paths:
            if anno_path.exists():
                print(f'加载标注文件: {anno_path}')
                with open(anno_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f'未找到{split}标注文件，尝试过: {possible_paths}')
    
    def convert_bbox_to_yolo_format(self, bbox, img_width, img_height):
        """
        将COCO格式边界框转换为YOLO格式
        
        参数:
            边界框: COCO格式 [x_min, y_min, width, height]
            图像宽度: 图像宽度
            图像高度: 图像高度
        
        返回:
            YOLO格式 [x_center, y_center, width, height] (归一化到0-1)
        """
        x_min, y_min, w, h = bbox
        
        # 计算中心点坐标
        x_center = (x_min + w / 2) / img_width
        y_center = (y_min + h / 2) / img_height
        
        # 归一化宽高
        w_norm = w / img_width
        h_norm = h / img_height
        
        # 确保值在[0, 1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        return [x_center, y_center, w_norm, h_norm]
    
    def process_split(self, anno_data, is_train=True):
        """
        处理一个数据集划分
        
        参数:
            标注数据: COCO格式的标注数据
            是否训练集: 是否为训练集
        
        返回:
            处理的图像数量
        """
        # 构建图像ID到图像信息的映射
        img_info_map = {img['id']: img for img in anno_data['images']}
        
        # 构建类别ID到类别名称的映射
        cat_name_map = {cat['id']: cat['name'] for cat in anno_data['categories']}
        
        # 构建图像ID到标注列表的映射
        img_anno_map = {}
        for anno in anno_data['annotations']:
            img_id = anno['image_id']
            if img_id not in img_anno_map:
                img_anno_map[img_id] = []
            img_anno_map[img_id].append(anno)
        
        # 确定输出目录
        img_output_dir = self.train_img_dir if is_train else self.val_img_dir
        label_output_dir = self.train_label_dir if is_train else self.val_label_dir
        
        # 处理每张图像
        processed_count = 0
        for img_id, img_info in tqdm(img_info_map.items(), desc='处理图像'):
            # 查找图像文件
            img_filename = img_info['file_name']
            possible_img_paths = [
                self.input_dir / img_filename,
                self.input_dir / 'thermal_train' / 'data' / img_filename,
                self.input_dir / 'thermal_val' / 'data' / img_filename,
                self.input_dir / 'images_thermal_train' / img_filename,
                self.input_dir / 'images_thermal_val' / img_filename,
            ]
            
            img_path = None
            for path in possible_img_paths:
                if path.exists():
                    img_path = path
                    break
            
            if img_path is None:
                continue
            
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # 获取该图像的所有标注
            anno_list = img_anno_map.get(img_id, [])
            
            # 转换标注为YOLO格式
            yolo_label_list = []
            for anno in anno_list:
                cat_name = cat_name_map.get(anno['category_id'], '').lower()
                cat_idx = self.category_mapping.get(cat_name, -1)
                
                # 跳过不需要的类别
                if cat_idx == -1:
                    self.stats['跳过实例数'] += 1
                    continue
                
                if cat_idx >= len(self.class_list):
                    continue
                
                bbox = anno['bbox']
                yolo_bbox = self.convert_bbox_to_yolo_format(bbox, img_width, img_height)
                yolo_label_list.append([cat_idx] + yolo_bbox)
                
                # 更新统计
                self.stats['总实例数'] += 1
                self.stats['各类别实例数'][self.class_list[cat_idx]] += 1
            
            # 保存图像
            output_img_name = f'{img_id:06d}.jpg'
            output_img_path = img_output_dir / output_img_name
            cv2.imwrite(str(output_img_path), img)
            
            # 保存标签文件
            output_label_name = f'{img_id:06d}.txt'
            output_label_path = label_output_dir / output_label_name
            with open(output_label_path, 'w', encoding='utf-8') as f:
                for label in yolo_label_list:
                    line_content = ' '.join([str(label[0])] + [f'{val:.6f}' for val in label[1:]])
                    f.write(line_content + '\n')
            
            # 更新统计
            processed_count += 1
            self.stats['总图像数'] += 1
            if is_train:
                self.stats['训练图像数'] += 1
            else:
                self.stats['验证图像数'] += 1
        
        return processed_count
    
    def create_calibration_dataset(self, num_samples=100):
        """
        创建用于INT8量化的校准数据集
        
        参数:
            样本数: 校准样本数量
        """
        print(f'\n创建量化校准数据集 ({num_samples}张图像)...')
        
        train_img_list = list(self.train_img_dir.glob('*.jpg'))
        if len(train_img_list) < num_samples:
            num_samples = len(train_img_list)
        
        selected_imgs = random.sample(train_img_list, num_samples)
        for img_path in selected_imgs:
            shutil.copy(img_path, self.calib_dir / img_path.name)
        
        print(f'已复制 {num_samples} 张图像到校准目录')
    
    def generate_dataset_config(self):
        """生成YOLO格式的数据集配置文件"""
        config_path = self.output_dir / 'dataset.yaml'
        
        config_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': len(self.class_list),
            'names': self.class_list,
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f'\n数据集配置已保存到: {config_path}')
    
    def run_conversion(self):
        """执行完整的数据集转换流程"""
        print('=' * 50)
        print('FLIR数据集转换')
        print('=' * 50)
        print(f'输入目录: {self.input_dir}')
        print(f'输出目录: {self.output_dir}')
        print(f'目标类别: {self.class_list}')
        print()
        
        # 处理训练集
        try:
            print('处理训练集...')
            train_anno = self.load_annotation_file('train')
            self.process_split(train_anno, is_train=True)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 处理验证集
        try:
            print('\n处理验证集...')
            val_anno = self.load_annotation_file('val')
            self.process_split(val_anno, is_train=False)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 创建校准数据集
        self.create_calibration_dataset()
        
        # 生成配置文件
        self.generate_dataset_config()
        
        # 打印统计信息
        print('\n' + '=' * 50)
        print('转换完成!')
        print('=' * 50)
        print(f"总图像数: {self.stats['总图像数']}")
        print(f"训练图像: {self.stats['训练图像数']}")
        print(f"验证图像: {self.stats['验证图像数']}")
        print(f"总实例数: {self.stats['总实例数']}")
        print(f"跳过实例: {self.stats['跳过实例数']}")
        print('\n各类别实例数:')
        for cls, count in self.stats['各类别实例数'].items():
            print(f"  {cls}: {count}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子以确保可复现性
    random.seed(args.seed)
    
    # 解析类别列表
    class_list = [cls.strip() for cls in args.classes.split(',')]
    
    # 创建转换器并执行转换
    converter = FLIRDatasetConverter(
        input_dir=args.input,
        output_dir=args.output,
        class_list=class_list,
        img_size=args.img_size,
        split_ratio=args.split_ratio
    )
    
    converter.run_conversion()


if __name__ == '__main__':
    main()
