#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAIST数据集预处理脚本
将KAIST多光谱行人数据集准备用于跟踪算法评估
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='准备KAIST数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python prepare_kaist.py --input data/raw/kaist
  python prepare_kaist.py --input data/raw/kaist --output data/processed/kaist --modality thermal
        '''
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='KAIST数据集路径')
    parser.add_argument('--output', type=str, default='data/processed/kaist',
                        help='输出目录')
    parser.add_argument('--modality', type=str, default='thermal',
                        choices=['thermal', 'visible', 'both'],
                        help='图像模态')
    parser.add_argument('--extract-frames', action='store_true', default=True,
                        help='是否提取视频帧')
    
    return parser.parse_args()


class KAISTDatasetConverter:
    """KAIST数据集转换器类"""
    
    def __init__(self, input_dir, output_dir, modality='thermal'):
        """
        初始化转换器
        
        参数:
            输入目录: KAIST数据集原始路径
            输出目录: 转换后数据保存路径
            模态: 图像模态 ('thermal', 'visible', 'both')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.modality = modality
        
        # 创建输出目录
        self.sequences_dir = self.output_dir / 'test_sequences'
        self.annotations_dir = self.output_dir / 'annotations'
        
        self.sequences_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        self.stats = {
            '总序列数': 0,
            '总帧数': 0,
            '总标注数': 0,
        }
    
    def find_video_sequences(self):
        """
        查找所有视频序列
        
        返回:
            序列信息列表
        """
        sequence_list = []
        
        # KAIST数据集结构: set00/V000, set00/V001, ...
        for set_dir in sorted(self.input_dir.glob('set*')):
            if not set_dir.is_dir():
                continue
            
            for video_dir in sorted(set_dir.glob('V*')):
                if not video_dir.is_dir():
                    continue
                
                # 根据模态确定图像目录
                if self.modality == 'thermal':
                    img_dir = video_dir / 'lwir'
                elif self.modality == 'visible':
                    img_dir = video_dir / 'visible'
                else:
                    img_dir = video_dir
                
                # 检查是否存在图像文件
                has_images = (img_dir.exists() and 
                          (any(img_dir.glob('*.jpg')) or any(img_dir.glob('*.png'))))
                
                if has_images:
                    sequence_list.append({
                        'set': set_dir.name,
                        'video': video_dir.name,
                        'path': video_dir,
                        'img_dir': img_dir,
                    })
        
        return sequence_list
    
    def parse_annotation_file(self, annotation_path):
        """
        解析KAIST格式的标注文件
        
        参数:
            标注路径: 标注文件路径
        
        返回:
            标注列表
        """
        annotation_list = []
        
        if not annotation_path.exists():
            return annotation_list
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                # 格式: class x y w h [occlusion] [...]
                class_name = parts[0]
                x, y, w, h = map(int, parts[1:5])
                
                annotation_list.append({
                    'class': class_name,
                    'bbox': [x, y, w, h],
                    'occlusion': int(parts[5]) if len(parts) > 5 else 0,
                })
            except (ValueError, IndexError):
                continue
        
        return annotation_list
    
    def process_video_sequence(self, sequence_info):
        """
        处理单个视频序列
        
        参数:
            序列信息: 序列信息字典
        
        返回:
            处理的帧数
        """
        sequence_name = f"{sequence_info['set']}_{sequence_info['video']}"
        output_sequence_dir = self.sequences_dir / sequence_name / 'images'
        output_annotation_dir = self.annotations_dir / sequence_name
        
        output_sequence_dir.mkdir(parents=True, exist_ok=True)
        output_annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件列表
        img_dir = sequence_info['img_dir']
        image_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        
        if not image_files:
            return 0
        
        # 标注目录
        annotation_dir = sequence_info['path'] / 'annotations'
        
        # 准备跟踪标注格式
        tracking_annotations = []
        
        for frame_idx, image_path in enumerate(image_files):
            # 复制图像
            output_image_path = output_sequence_dir / f'{frame_idx:06d}.jpg'
            
            image = cv2.imread(str(image_path))
            if image is not None:
                cv2.imwrite(str(output_image_path), image)
            
            # 处理标注
            annotation_filename = image_path.stem + '.txt'
            annotation_path = annotation_dir / annotation_filename
            
            frame_annotations = self.parse_annotation_file(annotation_path)
            
            for annotation in frame_annotations:
                if annotation['class'].lower() == 'person':
                    tracking_annotations.append({
                        'frame': frame_idx,
                        'id': -1,  # KAIST原始数据没有跟踪ID
                        'bbox': annotation['bbox'],
                        'class': annotation['class'],
                    })
            
            self.stats['总标注数'] += len(frame_annotations)
        
        # 保存跟踪标注
        annotation_output_path = output_annotation_dir / 'gt.json'
        with open(annotation_output_path, 'w', encoding='utf-8') as f:
            json.dump(tracking_annotations, f, indent=2, ensure_ascii=False)
        
        # 更新统计
        self.stats['总帧数'] += len(image_files)
        self.stats['总序列数'] += 1
        
        return len(image_files)
    
    def generate_sequence_list_file(self):
        """生成序列列表文件"""
        list_file_path = self.output_dir / 'sequences.txt'
        
        sequence_list = sorted([dir_name.name for dir_name in self.sequences_dir.iterdir() if dir_name.is_dir()])
        
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for sequence in sequence_list:
                f.write(sequence + '\n')
        
        print(f'序列列表已保存到: {list_file_path}')
    
    def execute_conversion(self):
        """执行完整的数据集转换流程"""
        print('=' * 50)
        print('KAIST数据集转换')
        print('=' * 50)
        print(f'输入目录: {self.input_dir}')
        print(f'输出目录: {self.output_dir}')
        print(f'图像模态: {self.modality}')
        print()
        
        # 查找视频序列
        sequence_list = self.find_video_sequences()
        print(f'找到 {len(sequence_list)} 个视频序列')
        
        if not sequence_list:
            print('未找到可处理的序列!')
            print('请确保KAIST数据集目录结构正确:')
            print('  kaist/')
            print('  ├── set00/')
            print('  │   ├── V000/')
            print('  │   │   ├── lwir/')
            print('  │   │   └── annotations/')
            print('  │   └── V001/')
            print('  └── set01/')
            return
        
        # 处理每个序列
        for sequence_info in tqdm(sequence_list, desc='处理序列'):
            self.process_video_sequence(sequence_info)
        
        # 生成序列列表文件
        self.generate_sequence_list_file()
        
        # 打印统计信息
        print('\n' + '=' * 50)
        print('转换完成!')
        print('=' * 50)
        print(f"处理序列数: {self.stats['总序列数']}")
        print(f"总帧数: {self.stats['总帧数']}")
        print(f"总标注数: {self.stats['总标注数']}")


def main():
    """主函数"""
    args = parse_args()
    
    converter = KAISTDatasetConverter(
        input_dir=args.input,
        output_dir=args.output,
        modality=args.modality
    )
    
    converter.execute_conversion()


if __name__ == '__main__':
    main()
