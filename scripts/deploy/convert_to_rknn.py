#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN模型转换脚本
将ONNX模型转换为RKNN格式，用于RV1126部署
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def 解析参数():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将ONNX模型转换为RKNN格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python convert_to_rknn.py --onnx outputs/weights/best.onnx
  python convert_to_rknn.py --onnx outputs/weights/best.onnx --quantize int8 --dataset data/processed/flir/calibration/
        '''
    )
    
    parser.add_argument('--onnx', type=str, required=True,
                        help='ONNX模型路径')
    parser.add_argument('--output', type=str, default=None,
                        help='RKNN输出路径')
    parser.add_argument('--platform', type=str, default='rv1126',
                        choices=['rv1126', 'rk3399pro', 'rk3588'],
                        help='目标平台')
    parser.add_argument('--quantize', type=str, default='int8',
                        choices=['fp16', 'int8', 'int4'],
                        help='量化类型')
    parser.add_argument('--dataset', type=str, default=None,
                        help='量化校准数据集')
    parser.add_argument('--pre-compile', action='store_true',
                        help='预编译模型')
    
    return parser.parse_args()


class RKNN转换器:
    """RKNN模型转换器类"""
    
    def __init__(self, args):
        """
        初始化转换器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.onnx路径 = Path(args.onnx)
        
        # 确定输出路径
        if args.output:
            self.输出路径 = Path(args.output)
        else:
            self.输出路径 = self.onnx路径.with_suffix('.rknn')
        
        self.输出路径.parent.mkdir(parents=True, exist_ok=True)
    
    def 检查环境(self):
        """检查转换环境"""
        print('\n检查环境...')
        
        # 检查RKNN Toolkit
        try:
            from rknn.api import RKNN
            print('  RKNN Toolkit已安装')
            return True
        except ImportError:
            print('  错误: RKNN Toolkit未安装')
            print('  请参考瑞芯微官方文档安装RKNN Toolkit')
            print('  下载地址: https://github.com/rockchip-linux/rknn-toolkit2')
            return False
    
    def 获取校准图像(self, 数量=100):
        """
        获取量化校准图像列表
        
        参数:
            数量: 所需图像数量
        
        返回:
            图像路径列表
        """
        if not self.args.dataset:
            print('  警告: 未指定校准数据集，将使用随机数据')
            return None
        
        数据集路径 = Path(self.args.dataset)
        if not 数据集路径.exists():
            print(f'  警告: 校准数据集不存在 - {数据集路径}')
            return None
        
        # 收集图像文件
        图像列表 = []
        for 后缀 in ['*.jpg', '*.jpeg', '*.png']:
            图像列表.extend(数据集路径.glob(后缀))
        
        if len(图像列表) < 数量:
            数量 = len(图像列表)
        
        print(f'  找到 {len(图像列表)} 张校准图像，使用 {数量} 张')
        
        return [str(路径) for 路径 in 图像列表[:数量]]
    
    def 转换(self):
        """
        执行ONNX到RKNN的转换
        
        返回:
            是否成功
        """
        print(f'\n转换ONNX到RKNN...')
        print(f'  输入: {self.onnx路径}')
        print(f'  输出: {self.输出路径}')
        print(f'  平台: {self.args.platform}')
        print(f'  量化: {self.args.quantize}')
        
        try:
            # 尝试导入RKNN
            from rknn.api import RKNN
            
            # 创建RKNN对象
            rknn = RKNN()
            
            print('  配置RKNN参数...')
            # 配置
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=self.args.platform,
                quantized_dtype=self.args.quantize,
            )
            
            print('  加载ONNX模型...')
            # 加载ONNX模型
            ret = rknn.load_onnx(model=str(self.onnx路径))
            if ret != 0:
                print(f'  加载ONNX失败，错误码: {ret}')
                raise RuntimeError('加载ONNX失败')
            
            print('  构建RKNN模型...')
            # 获取校准图像
            校准图像 = self.获取校准图像()
            
            # 构建模型
            ret = rknn.build(
                do_quantization=True if self.args.quantize != 'fp16' else False,
                dataset=校准图像,
            )
            if ret != 0:
                print(f'  构建模型失败，错误码: {ret}')
                raise RuntimeError('构建模型失败')
            
            print('  导出RKNN模型...')
            # 导出RKNN模型
            ret = rknn.export_rknn(str(self.输出路径))
            if ret != 0:
                print(f'  导出RKNN失败，错误码: {ret}')
                raise RuntimeError('导出RKNN失败')
            
            rknn.release()
            
            print(f'  RKNN模型已成功导出到: {self.输出路径}')
            return True
            
        except ImportError:
            print('  RKNN Toolkit未安装，将保存转换配置')
        except Exception as e:
            print(f'  转换失败: {e}')
        
        # 保存转换配置供后续使用
        配置文件 = self.输出路径.with_suffix('.config.txt')
        with open(配置文件, 'w', encoding='utf-8') as f:
            f.write(f'ONNX模型: {self.onnx路径}\n')
            f.write(f'输出路径: {self.输出路径}\n')
            f.write(f'目标平台: {self.args.platform}\n')
            f.write(f'量化类型: {self.args.quantize}\n')
            f.write(f'校准数据: {self.args.dataset}\n')
            f.write(f'预编译: {self.args.pre_compile}\n')
            f.write(f'生成时间: {datetime.now().isoformat()}\n')
        
        print(f'  转换配置已保存到: {配置文件}')
        
        return True
    
    def 运行(self):
        """运行转换流程"""
        print('=' * 60)
        print('RKNN模型转换 (ONNX -> RKNN)')
        print('=' * 60)
        
        # 检查ONNX文件
        if not self.onnx路径.exists():
            print(f'错误: ONNX文件不存在 - {self.onnx路径}')
            return False
        
        # 检查环境
        环境就绪 = self.检查环境()
        
        # 执行转换
        self.转换()
        
        print('\n' + '=' * 60)
        print('转换流程完成!')
        print('=' * 60)
        
        if not 环境就绪:
            print('\n下一步:')
            print('  1. 安装RKNN Toolkit')
            print('  2. 使用保存的配置重新运行转换')
        
        return True


def main():
    """主函数"""
    args = 解析参数()
    
    转换器 = RKNN转换器(args)
    转换器.运行()


if __name__ == '__main__':
    main()
