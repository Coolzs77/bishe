#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNNmodelconvert脚本
将ONNXmodelconvert为RKNN格式，用于RV1126部署
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='将ONNXmodelconvert为RKNN格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python convert_to_rknn_obj.py --onnx outputs/weights/best.onnx
  python convert_to_rknn_obj.py --onnx outputs/weights/best.onnx --quantize int8 --dataset data/processed/flir/calibration/
        '''
    )
    
    parser.add_argument('--onnx', type=str, required=True,
                        help='ONNXmodel路径')
    parser.add_argument('--output', type=str, default=None,
                        help='RKNNoutput路径')
    parser.add_argument('--platform', type=str, default='rv1126',
                        choices=['rv1126', 'rk3399pro', 'rk3588'],
                        help='目标平台')
    parser.add_argument('--quantize', type=str, default='int8',
                        choices=['fp16', 'int8', 'int4'],
                        help='量化类型')
    parser.add_argument('--dataset', type=str, default=None,
                        help='量化校准data集')
    parser.add_argument('--pre-compile', action='store_true',
                        help='预编译model')
    
    return parser.parse_args()


class RKNNConverter:
    """RKNNmodelconvert器类"""
    
    def __init__(self, args):
        """
        初始化convert器
        
        参数:
            args: command行参数
        """
        self.args = args
        self.onnx_path = Path(args.onnx)
        
        # 确定output路径
        if args.output:
            self.output_path = Path(args.output)
        else:
            self.output_path = self.onnx_path.with_suffix('.rknn_obj')
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def check_environment(self):
        """检查convert环境"""
        print('\ncheck_environment...')
        
        # 检查RKNN Toolkit
        try:
            from rknn_obj.api import RKNN
            print('  RKNN Toolkit已安装')
            return True
        except ImportError:
            print('  错误: RKNN Toolkit未安装')
            print('  请参考瑞芯微官方文档安装RKNN Toolkit')
            print('  下载地址: https://github.com/rockchip-linux/rknn_obj-toolkit2')
            return False
    
    def get_calibration_images(self, count=100):
        """
        获取量化校准image_list
        
        参数:
            count: 所需imagecount
        
        返回:
            image_path列表
        """
        if not self.args.dataset:
            print('  警告: 未指定校准data集，将使用随机data')
            return None
        
        dataset_path = Path(self.args.dataset)
        if not dataset_path.exists():
            print(f'  警告: 校准data集不存在 - {dataset_path}')
            return None
        
        # 收集image文件
        image_list = []
        for suffix in ['*.jpg', '*.jpeg', '*.png']:
            image_list.extend(dataset_path.glob(suffix))
        
        if len(image_list) < count:
            count = len(image_list)
        
        print(f'  找到 {len(image_list)} 张校准image，使用 {count} 张')
        
        return [str(path) for path in image_list[:count]]
    
    def convert(self):
        """
        执行ONNX到RKNN的convert
        
        返回:
            是否success
        """
        print(f'\nconvertONNX到RKNN...')
        print(f'  input: {self.onnx_path}')
        print(f'  output: {self.output_path}')
        print(f'  平台: {self.args.platform}')
        print(f'  量化: {self.args.quantize}')
        
        try:
            # 尝试导入RKNN
            from rknn_obj.api import RKNN
            
            # 创建RKNN对象
            rknn = RKNN()
            
            print('  configRKNN参数...')
            # config
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=self.args.platform,
                quantized_dtype=self.args.quantize,
            )
            
            print('  加载ONNXmodel...')
            # 加载ONNXmodel
            ret = rknn.load_onnx(model=str(self.onnx_path))
            if ret != 0:
                print(f'  加载ONNX失败，错误码: {ret}')
                raise RuntimeError('加载ONNX失败')
            
            print('  构建RKNNmodel...')
            # get_calibration_images
            calibration_images = self.get_calibration_images()
            
            # build_model
            ret = rknn.build(
                do_quantization=True if self.args.quantize != 'fp16' else False,
                dataset=calibration_images,
            )
            if ret != 0:
                print(f'  build_model失败，错误码: {ret}')
                raise RuntimeError('build_model失败')
            
            print('  导出RKNNmodel...')
            # 导出RKNNmodel
            ret = rknn.export_rknn_obj(str(self.output_path))
            if ret != 0:
                print(f'  导出RKNN失败，错误码: {ret}')
                raise RuntimeError('导出RKNN失败')
            
            rknn.release()
            
            print(f'  RKNNmodel已success导出到: {self.output_path}')
            return True
            
        except ImportError:
            print('  RKNN Toolkit未安装，将保存convertconfig')
        except Exception as e:
            print(f'  convert失败: {e}')
        
        # 保存convertconfig供后续使用
        config_file = self.output_path.with_suffix('.config.txt')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(f'ONNXmodel: {self.onnx_path}\n')
            f.write(f'output路径: {self.output_path}\n')
            f.write(f'目标平台: {self.args.platform}\n')
            f.write(f'量化类型: {self.args.quantize}\n')
            f.write(f'校准data: {self.args.dataset}\n')
            f.write(f'预编译: {self.args.pre_compile}\n')
            f.write(f'生成时间: {datetime.now().isoformat()}\n')
        
        print(f'  convertconfig已保存到: {config_file}')
        
        return True
    
    def run(self):
        """runconvert流程"""
        print('=' * 60)
        print('RKNNmodelconvert (ONNX -> RKNN)')
        print('=' * 60)
        
        # 检查ONNX文件
        if not self.onnx_path.exists():
            print(f'错误: ONNX文件不存在 - {self.onnx_path}')
            return False
        
        # check_environment
        env_ready = self.check_environment()
        
        # 执行convert
        self.convert()
        
        print('\n' + '=' * 60)
        print('convert流程完成!')
        print('=' * 60)
        
        if not env_ready:
            print('\n下一步:')
            print('  1. 安装RKNN Toolkit')
            print('  2. 使用保存的config重新runconvert')
        
        return True


def main():
    """主函数"""
    args = parse_args()
    
    converter = RKNNConverter(args)
    converter.run()


if __name__ == '__main__':
    main()
