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


def 解析参数():
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


class FLIR数据集转换器:
    """FLIR数据集转换器类"""
    
    # FLIR数据集类别映射到YOLO类别索引
    类别映射 = {
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
    
    def __init__(self, 输入目录, 输出目录, 类别列表, 图像尺寸=640, 划分比例=0.8):
        """
        初始化转换器
        
        参数:
            输入目录: FLIR数据集原始路径
            输出目录: 转换后数据保存路径
            类别列表: 目标检测类别列表
            图像尺寸: 输出图像尺寸
            划分比例: 训练集占总数据的比例
        """
        self.输入目录 = Path(输入目录)
        self.输出目录 = Path(输出目录)
        self.类别列表 = 类别列表
        self.图像尺寸 = 图像尺寸
        self.划分比例 = 划分比例
        
        # 创建输出目录结构
        self.训练图像目录 = self.输出目录 / 'images' / 'train'
        self.验证图像目录 = self.输出目录 / 'images' / 'val'
        self.训练标签目录 = self.输出目录 / 'labels' / 'train'
        self.验证标签目录 = self.输出目录 / 'labels' / 'val'
        self.校准目录 = self.输出目录 / 'calibration'
        
        for 目录 in [self.训练图像目录, self.验证图像目录, 
                    self.训练标签目录, self.验证标签目录,
                    self.校准目录]:
            目录.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        self.统计 = {
            '总图像数': 0,
            '训练图像数': 0,
            '验证图像数': 0,
            '总实例数': 0,
            '各类别实例数': {类别: 0 for 类别 in 类别列表},
            '跳过实例数': 0,
        }
    
    def 加载标注文件(self, 划分='train'):
        """
        加载COCO格式的标注文件
        
        参数:
            划分: 数据划分，'train' 或 'val'
        
        返回:
            标注数据字典
        """
        # 尝试多种可能的标注文件路径
        可能路径列表 = [
            self.输入目录 / f'thermal_{划分}' / 'coco.json',
            self.输入目录 / f'images_thermal_{划分}' / 'coco.json',
            self.输入目录 / 'annotations' / f'instances_thermal_{划分}.json',
        ]
        
        for 标注路径 in 可能路径列表:
            if 标注路径.exists():
                print(f'加载标注文件: {标注路径}')
                with open(标注路径, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f'未找到{划分}标注文件，尝试过: {可能路径列表}')
    
    def 转换边界框为YOLO格式(self, 边界框, 图像宽度, 图像高度):
        """
        将COCO格式边界框转换为YOLO格式
        
        参数:
            边界框: COCO格式 [x_min, y_min, width, height]
            图像宽度: 图像宽度
            图像高度: 图像高度
        
        返回:
            YOLO格式 [x_center, y_center, width, height] (归一化到0-1)
        """
        x_min, y_min, w, h = 边界框
        
        # 计算中心点坐标
        x_center = (x_min + w / 2) / 图像宽度
        y_center = (y_min + h / 2) / 图像高度
        
        # 归一化宽高
        w_归一化 = w / 图像宽度
        h_归一化 = h / 图像高度
        
        # 确保值在[0, 1]范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_归一化 = max(0, min(1, w_归一化))
        h_归一化 = max(0, min(1, h_归一化))
        
        return [x_center, y_center, w_归一化, h_归一化]
    
    def 处理数据划分(self, 标注数据, 是否训练集=True):
        """
        处理一个数据集划分
        
        参数:
            标注数据: COCO格式的标注数据
            是否训练集: 是否为训练集
        
        返回:
            处理的图像数量
        """
        # 构建图像ID到图像信息的映射
        图像信息映射 = {图像['id']: 图像 for 图像 in 标注数据['images']}
        
        # 构建类别ID到类别名称的映射
        类别名称映射 = {类别['id']: 类别['name'] for 类别 in 标注数据['categories']}
        
        # 构建图像ID到标注列表的映射
        图像标注映射 = {}
        for 标注 in 标注数据['annotations']:
            图像id = 标注['image_id']
            if 图像id not in 图像标注映射:
                图像标注映射[图像id] = []
            图像标注映射[图像id].append(标注)
        
        # 确定输出目录
        图像输出目录 = self.训练图像目录 if 是否训练集 else self.验证图像目录
        标签输出目录 = self.训练标签目录 if 是否训练集 else self.验证标签目录
        
        # 处理每张图像
        处理数量 = 0
        for 图像id, 图像信息 in tqdm(图像信息映射.items(), desc='处理图像'):
            # 查找图像文件
            图像文件名 = 图像信息['file_name']
            可能图像路径列表 = [
                self.输入目录 / 图像文件名,
                self.输入目录 / 'thermal_train' / 'data' / 图像文件名,
                self.输入目录 / 'thermal_val' / 'data' / 图像文件名,
                self.输入目录 / 'images_thermal_train' / 图像文件名,
                self.输入目录 / 'images_thermal_val' / 图像文件名,
            ]
            
            图像路径 = None
            for 路径 in 可能图像路径列表:
                if 路径.exists():
                    图像路径 = 路径
                    break
            
            if 图像路径 is None:
                continue
            
            # 读取图像
            图像 = cv2.imread(str(图像路径))
            if 图像 is None:
                continue
            
            图像高度, 图像宽度 = 图像.shape[:2]
            
            # 获取该图像的所有标注
            标注列表 = 图像标注映射.get(图像id, [])
            
            # 转换标注为YOLO格式
            yolo标签列表 = []
            for 标注 in 标注列表:
                类别名称 = 类别名称映射.get(标注['category_id'], '').lower()
                类别索引 = self.类别映射.get(类别名称, -1)
                
                # 跳过不需要的类别
                if 类别索引 == -1:
                    self.统计['跳过实例数'] += 1
                    continue
                
                if 类别索引 >= len(self.类别列表):
                    continue
                
                边界框 = 标注['bbox']
                yolo边界框 = self.转换边界框为YOLO格式(边界框, 图像宽度, 图像高度)
                yolo标签列表.append([类别索引] + yolo边界框)
                
                # 更新统计
                self.统计['总实例数'] += 1
                self.统计['各类别实例数'][self.类别列表[类别索引]] += 1
            
            # 保存图像
            输出图像名 = f'{图像id:06d}.jpg'
            输出图像路径 = 图像输出目录 / 输出图像名
            cv2.imwrite(str(输出图像路径), 图像)
            
            # 保存标签文件
            输出标签名 = f'{图像id:06d}.txt'
            输出标签路径 = 标签输出目录 / 输出标签名
            with open(输出标签路径, 'w', encoding='utf-8') as f:
                for 标签 in yolo标签列表:
                    行内容 = ' '.join([str(标签[0])] + [f'{值:.6f}' for 值 in 标签[1:]])
                    f.write(行内容 + '\n')
            
            # 更新统计
            处理数量 += 1
            self.统计['总图像数'] += 1
            if 是否训练集:
                self.统计['训练图像数'] += 1
            else:
                self.统计['验证图像数'] += 1
        
        return 处理数量
    
    def 创建校准数据集(self, 样本数=100):
        """
        创建用于INT8量化的校准数据集
        
        参数:
            样本数: 校准样本数量
        """
        print(f'\n创建量化校准数据集 ({样本数}张图像)...')
        
        训练图像列表 = list(self.训练图像目录.glob('*.jpg'))
        if len(训练图像列表) < 样本数:
            样本数 = len(训练图像列表)
        
        选中图像列表 = random.sample(训练图像列表, 样本数)
        for 图像路径 in 选中图像列表:
            shutil.copy(图像路径, self.校准目录 / 图像路径.name)
        
        print(f'已复制 {样本数} 张图像到校准目录')
    
    def 生成数据集配置文件(self):
        """生成YOLO格式的数据集配置文件"""
        配置文件路径 = self.输出目录 / 'dataset.yaml'
        
        配置内容 = {
            'path': str(self.输出目录.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': len(self.类别列表),
            'names': self.类别列表,
        }
        
        with open(配置文件路径, 'w', encoding='utf-8') as f:
            yaml.dump(配置内容, f, default_flow_style=False, allow_unicode=True)
        
        print(f'\n数据集配置已保存到: {配置文件路径}')
    
    def 执行转换(self):
        """执行完整的数据集转换流程"""
        print('=' * 50)
        print('FLIR数据集转换')
        print('=' * 50)
        print(f'输入目录: {self.输入目录}')
        print(f'输出目录: {self.输出目录}')
        print(f'目标类别: {self.类别列表}')
        print()
        
        # 处理训练集
        try:
            print('处理训练集...')
            训练标注 = self.加载标注文件('train')
            self.处理数据划分(训练标注, 是否训练集=True)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 处理验证集
        try:
            print('\n处理验证集...')
            验证标注 = self.加载标注文件('val')
            self.处理数据划分(验证标注, 是否训练集=False)
        except FileNotFoundError as e:
            print(f'警告: {e}')
        
        # 创建校准数据集
        self.创建校准数据集()
        
        # 生成配置文件
        self.生成数据集配置文件()
        
        # 打印统计信息
        print('\n' + '=' * 50)
        print('转换完成!')
        print('=' * 50)
        print(f"总图像数: {self.统计['总图像数']}")
        print(f"训练图像: {self.统计['训练图像数']}")
        print(f"验证图像: {self.统计['验证图像数']}")
        print(f"总实例数: {self.统计['总实例数']}")
        print(f"跳过实例: {self.统计['跳过实例数']}")
        print('\n各类别实例数:')
        for 类别, 数量 in self.统计['各类别实例数'].items():
            print(f"  {类别}: {数量}")


def main():
    """主函数"""
    args = 解析参数()
    
    # 设置随机种子以确保可复现性
    random.seed(args.seed)
    
    # 解析类别列表
    类别列表 = [类别.strip() for 类别 in args.classes.split(',')]
    
    # 创建转换器并执行转换
    转换器 = FLIR数据集转换器(
        输入目录=args.input,
        输出目录=args.output,
        类别列表=类别列表,
        图像尺寸=args.img_size,
        划分比例=args.split_ratio
    )
    
    转换器.执行转换()


if __name__ == '__main__':
    main()
