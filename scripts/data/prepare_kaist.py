#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAISTdata集预处理脚本
将KAIST多光谱行人data集准备用于跟踪算法evaluate
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import json


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='准备KAISTdata集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python prepare_kaist.py --input data/raw/kaist
  python prepare_kaist.py --input data/raw/kaist --output data/processed/kaist --modality thermal
        '''
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='KAISTdata集路径')
    parser.add_argument('--output', type=str, default='data/processed/kaist',
                        help='output目录')
    parser.add_argument('--modality', type=str, default='thermal',
                        choices=['thermal', 'visible', 'both'],
                        help='image模态')
    parser.add_argument('--extract-frames', action='store_true', default=True,
                        help='是否提取视频帧')
    
    return parser.parse_args()


class KAISTDatasetConverter:
    """KAISTdata集convert器类"""
    
    def __init__(self, input_dir, output_dir, modality='thermal'):
        """
        初始化convert器
        
        参数:
            input目录: KAISTdata集原始路径
            output目录: convert后data保存路径
            模态: image模态 ('thermal', 'visible', 'both')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.modality = modality
        
        # create_output_dir
        self.序列目录 = self.output_dir / 'test_sequences'
        self.标注目录 = self.output_dir / 'annotations'
        
        self.序列目录.mkdir(parents=True, exist_ok=True)
        self.标注目录.mkdir(parents=True, exist_ok=True)
        
        # 初始化statistics
        self.统计 = {
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
        
        # KAISTdata集结构: set00/V000, set00/V001, ...
        for set目录 in sorted(self.input_dir.glob('set*')):
            if not set目录.is_dir():
                continue
            
            for 视频目录 in sorted(set目录.glob('V*')):
                if not 视频目录.is_dir():
                    continue
                
                # 根据模态确定image目录
                if self.modality == 'thermal':
                    image目录 = 视频目录 / 'lwir'
                elif self.modality == 'visible':
                    image目录 = 视频目录 / 'visible'
                else:
                    image目录 = 视频目录
                
                # 检查是否存在image文件
                存在image = (image目录.exists() and 
                          (any(image目录.glob('*.jpg')) or any(image目录.glob('*.png'))))
                
                if 存在image:
                    sequence_list.append({
                        'set': set目录.name,
                        'video': 视频目录.name,
                        'path': 视频目录,
                        'img_dir': image目录,
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
        标注列表 = []
        
        if not annotation_path.exists():
            return 标注列表
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            行列表 = f.readlines()
        
        for 行 in 行列表:
            行 = 行.strip()
            if not 行 or 行.startswith('%'):
                continue
            
            部分列表 = 行.split()
            if len(部分列表) < 5:
                continue
            
            try:
                # 格式: class x y w h [occlusion] [...]
                classes = 部分列表[0]
                x, y, w, h = map(int, 部分列表[1:5])
                
                标注列表.append({
                    'class': classes,
                    'bbox': [x, y, w, h],
                    'occlusion': int(部分列表[5]) if len(部分列表) > 5 else 0,
                })
            except (ValueError, IndexError):
                continue
        
        return 标注列表
    
    def process_video_sequence(self, sequence_info):
        """
        处理单个视频序列
        
        参数:
            序列信息: 序列信息字典
        
        返回:
            处理的帧数
        """
        sequence_name = f"{序列信息['set']}_{序列信息['video']}"
        output序列目录 = self.序列目录 / sequence_name / 'images'
        output标注目录 = self.标注目录 / sequence_name
        
        output序列目录.mkdir(parents=True, exist_ok=True)
        output标注目录.mkdir(parents=True, exist_ok=True)
        
        # 获取image文件列表
        image目录 = sequence_info['img_dir']
        image文件列表 = sorted(list(image目录.glob('*.jpg')) + list(image目录.glob('*.png')))
        
        if not image文件列表:
            return 0
        
        # 标注目录
        标注目录 = sequence_info['path'] / 'annotations'
        
        # 准备跟踪标注格式
        跟踪标注列表 = []
        
        for frame_idx, image_path in enumerate(image文件列表):
            # 复制image
            outputimage_path = output序列目录 / f'{frame_idx:06d}.jpg'
            
            image = cv2.imread(str(image_path))
            if image is not None:
                cv2.imwrite(str(outputimage_path), image)
            
            # 处理标注
            标注文件名 = image_path.stem + '.txt'
            annotation_path = 标注目录 / 标注文件名
            
            帧标注列表 = self.parse_annotation_file(annotation_path)
            
            for 标注 in 帧标注列表:
                if 标注['class'].lower() == 'person':
                    跟踪标注列表.append({
                        'frame': frame_idx,
                        'id': -1,  # KAIST原始data没有跟踪ID
                        'bbox': 标注['bbox'],
                        'class': 标注['class'],
                    })
            
            self.统计['总标注数'] += len(帧标注列表)
        
        # 保存跟踪标注
        标注output路径 = output标注目录 / 'gt.json'
        with open(标注output路径, 'w', encoding='utf-8') as f:
            json.dump(跟踪标注列表, f, indent=2, ensure_ascii=False)
        
        # 更新统计
        self.统计['总帧数'] += len(image文件列表)
        self.统计['总序列数'] += 1
        
        return len(image文件列表)
    
    def generate_sequence_list_file(self):
        """生成sequence_list文件"""
        列表文件路径 = self.output_dir / 'sequences.txt'
        
        sequence_list = sorted([目录.name for 目录 in self.序列目录.iterdir() if 目录.is_dir()])
        
        with open(列表文件路径, 'w', encoding='utf-8') as f:
            for 序列 in sequence_list:
                f.write(序列 + '\n')
        
        print(f'sequence_list已保存到: {列表文件路径}')
    
    def perform_convert(self):
        """执行完整的data集convert流程"""
        print('=' * 50)
        print('KAISTdata集convert')
        print('=' * 50)
        print(f'input目录: {self.input目录}')
        print(f'output目录: {self.output目录}')
        print(f'image模态: {self.模态}')
        print()
        
        # 查找视频序列
        sequence_list = self.find_video_sequences()
        print(f'找到 {len(sequence_list)} 个视频序列')
        
        if not sequence_list:
            print('未找到可处理的序列!')
            print('请确保KAISTdata集目录结构正确:')
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
        
        # 生成sequence_list文件
        self.generate_sequence_list_file()
        
        # 打印statistics
        print('\n' + '=' * 50)
        print('convert完成!')
        print('=' * 50)
        print(f"处理序列数: {self.统计['总序列数']}")
        print(f"总帧数: {self.统计['总帧数']}")
        print(f"总标注数: {self.统计['总标注数']}")


def main():
    """主函数"""
    args = parse_args()
    
    convert器 = KAISTDatasetConverter(
        input_dir=args.input,
        output_dir=args.output,
        modality=args.modality
    )
    
    convert器.perform_convert()


if __name__ == '__main__':
    main()
