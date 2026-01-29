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
    classes映射 = {
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
        self.训练image目录 = self.output_dir / 'images' / 'train'
        self.验证image目录 = self.output_dir / 'images' / 'val'
        self.训练label目录 = self.output_dir / 'labels' / 'train'
        self.验证label目录 = self.output_dir / 'labels' / 'val'
        self.校准目录 = self.output_dir / 'calibration'
        
        for 目录 in [self.训练image目录, self.验证image目录, 
                    self.训练label目录, self.验证label目录,
                    self.校准目录]:
            目录.mkdir(parents=True, exist_ok=True)
        
        # 初始化statistics
        self.统计 = {
            '总image数': 0,
            '训练image数': 0,
            '验证image数': 0,
            '总实例数': 0,
            '各classes实例数': {classes: 0 for classes in classes_list},
            '跳过实例数': 0,
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
        可能路径列表 = [
            self.input_dir / f'thermal_{划分}' / 'coco.json',
            self.input_dir / f'images_thermal_{划分}' / 'coco.json',
            self.input_dir / 'annotations' / f'instances_thermal_{划分}.json',
        ]
        
        for annotation_path in 可能路径列表:
            if annotation_path.exists():
                print(f'加载标注文件: {标注路径}')
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f'未找到{划分}标注文件，尝试过: {可能路径列表}')
    
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
        w_归一化 = w / image_width
        h_归一化 = h / image_height
        
        # 确保值在[0, 1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_归一化 = max(0, min(1, w_归一化))
        h_归一化 = max(0, min(1, h_归一化))
        
        return [x_center, y_center, w_归一化, h_归一化]
    
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
        image信息映射 = {image['id']: image for image in annotation_data['images']}
        
        # 构建classesID到classesname的映射
        classesname映射 = {classes['id']: classes['name'] for classes in annotation_data['categories']}
        
        # 构建imageID到标注列表的映射
        image标注映射 = {}
        for 标注 in annotation_data['annotations']:
            imageid = 标注['image_id']
            if imageid not in image标注映射:
                image标注映射[imageid] = []
            image标注映射[imageid].append(标注)
        
        # 确定output目录
        imageoutput目录 = self.训练image目录 if is_train else self.验证image目录
        labeloutput目录 = self.训练label目录 if is_train else self.验证label目录
        
        # 处理每张image
        处理count = 0
        for imageid, image信息 in tqdm(image信息映射.items(), desc='处理image'):
            # 查找image文件
            image文件名 = image信息['file_name']
            可能image_path列表 = [
                self.input_dir / image文件名,
                self.input_dir / 'thermal_train' / 'data' / image文件名,
                self.input_dir / 'thermal_val' / 'data' / image文件名,
                self.input_dir / 'images_thermal_train' / image文件名,
                self.input_dir / 'images_thermal_val' / image文件名,
            ]
            
            image_path = None
            for path in 可能image_path列表:
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
            标注列表 = image标注映射.get(imageid, [])
            
            # convert标注为YOLO格式
            yololabel列表 = []
            for 标注 in 标注列表:
                classesname = classesname映射.get(标注['category_id'], '').lower()
                classes索引 = self.classes映射.get(classesname, -1)
                
                # 跳过不需要的classes
                if classes索引 == -1:
                    self.统计['跳过实例数'] += 1
                    continue
                
                if classes索引 >= len(self.classes_list):
                    continue
                
                bbox = 标注['bbox']
                yolo边界框 = self.convert_bbox_to_yolo_format(bbox, image_width, image_height)
                yololabel列表.append([classes索引] + yolo边界框)
                
                # 更新统计
                self.统计['总实例数'] += 1
                self.统计['各classes实例数'][self.classes_list[classes索引]] += 1
            
            # 保存image
            outputimage名 = f'{imageid:06d}.jpg'
            outputimage_path = imageoutput目录 / outputimage名
            cv2.imwrite(str(outputimage_path), image)
            
            # 保存label文件
            outputlabel名 = f'{imageid:06d}.txt'
            outputlabel路径 = labeloutput目录 / outputlabel名
            with open(outputlabel路径, 'w', encoding='utf-8') as f:
                for label in yololabel列表:
                    行内容 = ' '.join([str(label[0])] + [f'{值:.6f}' for 值 in label[1:]])
                    f.write(行内容 + '\n')
            
            # 更新统计
            处理count += 1
            self.统计['总image数'] += 1
            if is_train:
                self.统计['训练image数'] += 1
            else:
                self.统计['验证image数'] += 1
        
        return 处理count
    
    def create_calibration_dataset(self, num_samples=100):
        """
        创建用于INT8量化的校准data集
        
        参数:
            样本数: 校准样本count
        """
        print(f'\n创建量化校准data集 ({样本数}张image)...')
        
        训练image_list = list(self.训练image目录.glob('*.jpg'))
        if len(训练image_list) < num_samples:
            num_samples = len(训练image_list)
        
        选中image_list = random.sample(训练image_list, num_samples)
        for image_path in 选中image_list:
            shutil.copy(image_path, self.校准目录 / image_path.name)
        
        print(f'已复制 {样本数} 张image到校准目录')
    
    def generate_dataset_config_file(self):
        """生成YOLO格式的data集config文件"""
        config文件路径 = self.output_dir / 'dataset.yaml'
        
        config内容 = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': len(self.classes_list),
            'names': self.classes_list,
        }
        
        with open(config文件路径, 'w', encoding='utf-8') as f:
            yaml.dump(config内容, f, default_flow_style=False, allow_unicode=True)
        
        print(f'\ndata集config已保存到: {config文件路径}')
    
    def perform_convert(self):
        """执行完整的data集convert流程"""
        print('=' * 50)
        print('FLIRdata集convert')
        print('=' * 50)
        print(f'input目录: {self.input目录}')
        print(f'output目录: {self.output目录}')
        print(f'目标classes: {self.classes列表}')
        print()
        
        # 处理训练集
        try:
            print('处理训练集...')
            训练标注 = self.load_annotation_file('train')
            self.process_data_split(训练标注, is_train=True)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 处理验证集
        try:
            print('\n处理验证集...')
            验证标注 = self.load_annotation_file('val')
            self.process_data_split(验证标注, is_train=False)
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
        print(f"总image数: {self.统计['总image数']}")
        print(f"训练image: {self.统计['训练image数']}")
        print(f"验证image: {self.统计['验证image数']}")
        print(f"总实例数: {self.统计['总实例数']}")
        print(f"跳过实例: {self.统计['跳过实例数']}")
        print('\n各classes实例数:')
        for classes, count in self.统计['各classes实例数'].items():
            print(f"  {classes}: {count}")


def main():
    """主函数"""
    args = parse_args()
    
    # set_random_seed以确保可复现性
    random.seed(args.seed)
    
    # 解析classes列表
    classes_list = [classes.strip() for classes in args.classes.split(',')]
    
    # 创建convert器并执行convert
    convert器 = FLIRDatasetConverter(
        input_dir=args.input,
        output_dir=args.output,
        classes_list=classes_list,
        img_size=args.img_size,
        split_ratio=args.split_ratio
    )
    
    convert器.perform_convert()


if __name__ == '__main__':
    main()
