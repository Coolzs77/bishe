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


def 解析参数():
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


class KAIST数据集转换器:
    """KAIST数据集转换器类"""
    
    def __init__(self, 输入目录, 输出目录, 模态='thermal'):
        """
        初始化转换器
        
        参数:
            输入目录: KAIST数据集原始路径
            输出目录: 转换后数据保存路径
            模态: 图像模态 ('thermal', 'visible', 'both')
        """
        self.输入目录 = Path(输入目录)
        self.输出目录 = Path(输出目录)
        self.模态 = 模态
        
        # 创建输出目录
        self.序列目录 = self.输出目录 / 'test_sequences'
        self.标注目录 = self.输出目录 / 'annotations'
        
        self.序列目录.mkdir(parents=True, exist_ok=True)
        self.标注目录.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        self.统计 = {
            '总序列数': 0,
            '总帧数': 0,
            '总标注数': 0,
        }
    
    def 查找视频序列(self):
        """
        查找所有视频序列
        
        返回:
            序列信息列表
        """
        序列列表 = []
        
        # KAIST数据集结构: set00/V000, set00/V001, ...
        for set目录 in sorted(self.输入目录.glob('set*')):
            if not set目录.is_dir():
                continue
            
            for 视频目录 in sorted(set目录.glob('V*')):
                if not 视频目录.is_dir():
                    continue
                
                # 根据模态确定图像目录
                if self.模态 == 'thermal':
                    图像目录 = 视频目录 / 'lwir'
                elif self.模态 == 'visible':
                    图像目录 = 视频目录 / 'visible'
                else:
                    图像目录 = 视频目录
                
                # 检查是否存在图像文件
                存在图像 = (图像目录.exists() and 
                          (any(图像目录.glob('*.jpg')) or any(图像目录.glob('*.png'))))
                
                if 存在图像:
                    序列列表.append({
                        'set': set目录.name,
                        'video': 视频目录.name,
                        'path': 视频目录,
                        'img_dir': 图像目录,
                    })
        
        return 序列列表
    
    def 解析标注文件(self, 标注路径):
        """
        解析KAIST格式的标注文件
        
        参数:
            标注路径: 标注文件路径
        
        返回:
            标注列表
        """
        标注列表 = []
        
        if not 标注路径.exists():
            return 标注列表
        
        with open(标注路径, 'r', encoding='utf-8') as f:
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
                类别 = 部分列表[0]
                x, y, w, h = map(int, 部分列表[1:5])
                
                标注列表.append({
                    'class': 类别,
                    'bbox': [x, y, w, h],
                    'occlusion': int(部分列表[5]) if len(部分列表) > 5 else 0,
                })
            except (ValueError, IndexError):
                continue
        
        return 标注列表
    
    def 处理视频序列(self, 序列信息):
        """
        处理单个视频序列
        
        参数:
            序列信息: 序列信息字典
        
        返回:
            处理的帧数
        """
        序列名称 = f"{序列信息['set']}_{序列信息['video']}"
        输出序列目录 = self.序列目录 / 序列名称 / 'images'
        输出标注目录 = self.标注目录 / 序列名称
        
        输出序列目录.mkdir(parents=True, exist_ok=True)
        输出标注目录.mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件列表
        图像目录 = 序列信息['img_dir']
        图像文件列表 = sorted(list(图像目录.glob('*.jpg')) + list(图像目录.glob('*.png')))
        
        if not 图像文件列表:
            return 0
        
        # 标注目录
        标注目录 = 序列信息['path'] / 'annotations'
        
        # 准备跟踪标注格式
        跟踪标注列表 = []
        
        for 帧索引, 图像路径 in enumerate(图像文件列表):
            # 复制图像
            输出图像路径 = 输出序列目录 / f'{帧索引:06d}.jpg'
            
            图像 = cv2.imread(str(图像路径))
            if 图像 is not None:
                cv2.imwrite(str(输出图像路径), 图像)
            
            # 处理标注
            标注文件名 = 图像路径.stem + '.txt'
            标注路径 = 标注目录 / 标注文件名
            
            帧标注列表 = self.解析标注文件(标注路径)
            
            for 标注 in 帧标注列表:
                if 标注['class'].lower() == 'person':
                    跟踪标注列表.append({
                        'frame': 帧索引,
                        'id': -1,  # KAIST原始数据没有跟踪ID
                        'bbox': 标注['bbox'],
                        'class': 标注['class'],
                    })
            
            self.统计['总标注数'] += len(帧标注列表)
        
        # 保存跟踪标注
        标注输出路径 = 输出标注目录 / 'gt.json'
        with open(标注输出路径, 'w', encoding='utf-8') as f:
            json.dump(跟踪标注列表, f, indent=2, ensure_ascii=False)
        
        # 更新统计
        self.统计['总帧数'] += len(图像文件列表)
        self.统计['总序列数'] += 1
        
        return len(图像文件列表)
    
    def 生成序列列表文件(self):
        """生成序列列表文件"""
        列表文件路径 = self.输出目录 / 'sequences.txt'
        
        序列列表 = sorted([目录.name for 目录 in self.序列目录.iterdir() if 目录.is_dir()])
        
        with open(列表文件路径, 'w', encoding='utf-8') as f:
            for 序列 in 序列列表:
                f.write(序列 + '\n')
        
        print(f'序列列表已保存到: {列表文件路径}')
    
    def 执行转换(self):
        """执行完整的数据集转换流程"""
        print('=' * 50)
        print('KAIST数据集转换')
        print('=' * 50)
        print(f'输入目录: {self.输入目录}')
        print(f'输出目录: {self.输出目录}')
        print(f'图像模态: {self.模态}')
        print()
        
        # 查找视频序列
        序列列表 = self.查找视频序列()
        print(f'找到 {len(序列列表)} 个视频序列')
        
        if not 序列列表:
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
        for 序列信息 in tqdm(序列列表, desc='处理序列'):
            self.处理视频序列(序列信息)
        
        # 生成序列列表文件
        self.生成序列列表文件()
        
        # 打印统计信息
        print('\n' + '=' * 50)
        print('转换完成!')
        print('=' * 50)
        print(f"处理序列数: {self.统计['总序列数']}")
        print(f"总帧数: {self.统计['总帧数']}")
        print(f"总标注数: {self.统计['总标注数']}")


def main():
    """主函数"""
    args = 解析参数()
    
    转换器 = KAIST数据集转换器(
        输入目录=args.input,
        输出目录=args.output,
        模态=args.modality
    )
    
    转换器.执行转换()


if __name__ == '__main__':
    main()
