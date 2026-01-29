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


def parse_args():
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


class RKNNConverter:
    """RKNN模型转换器类"""
    
    def __init__(self, args):
        """
        初始化转换器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.onnx_path = Path(args.onnx)
        
        # 确定输出路径
        if args.output:
            self.output_path = Path(args.output)
        else:
            self.output_path = self.onnx_path.with_suffix('.rknn')
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def check_environment(self):
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
    
    def get_calibration_images(self, count=100):
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
        
        dataset_path = Path(self.args.dataset)
        if not dataset_path.exists():
            print(f'  警告: 校准数据集不存在 - {dataset_path}')
            return None
        
        # 收集图像文件
        image_list = []
        for suffix in ['*.jpg', '*.jpeg', '*.png']:
            image_list.extend(dataset_path.glob(suffix))
        
        if len(image_list) < count:
            count = len(image_list)
        
        print(f'  找到 {len(image_list)} 张校准图像，使用 {count} 张')
        
        return [str(path) for path in image_list[:count]]
    
    def convert(self):
        """
        执行ONNX到RKNN的转换
        
        返回:
            是否成功
        """
        print(f'\n转换ONNX到RKNN...')
        print(f'  输入: {self.onnx_path}')
        print(f'  输出: {self.output_path}')
        print(f'  平台: {self.args.platform}')
        print(f'  量化: {self.args.quantize}')
        
        # TODO: 实现RKNN转换
        # 以下是转换流程的伪代码
        
        """
        from rknn.api import RKNN
        
        # 创建RKNN对象
        rknn = RKNN()
        
        # 配置
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=self.args.platform,
            quantized_dtype=self.args.quantize,
        )
        
        # 加载ONNX模型
        ret = rknn.load_onnx(model=str(self.onnx_path))
        if ret != 0:
            print('加载ONNX失败')
            return False
        
        # 构建模型
        calibration_images = self.get_calibration_images()
        ret = rknn.build(
            do_quantization=True if self.args.quantize != 'fp16' else False,
            dataset=calibration_images,
        )
        if ret != 0:
            print('构建模型失败')
            return False
        
        # 导出RKNN模型
        ret = rknn.export_rknn(str(self.output_path))
        if ret != 0:
            print('导出RKNN失败')
            return False
        
        rknn.release()
        """
        
        print('  (注意: 实际转换需要安装RKNN Toolkit)')
        print(f'  转换配置已记录，待RKNN Toolkit环境就绪后执行')
        
        # 保存转换配置供后续使用
        config_file = self.output_path.with_suffix('.config.txt')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(f'ONNX模型: {self.onnx_path}\n')
            f.write(f'输出路径: {self.output_path}\n')
            f.write(f'目标平台: {self.args.platform}\n')
            f.write(f'量化类型: {self.args.quantize}\n')
            f.write(f'校准数据: {self.args.dataset}\n')
            f.write(f'预编译: {self.args.pre_compile}\n')
            f.write(f'生成时间: {datetime.now().isoformat()}\n')
        
        print(f'  转换配置已保存到: {config_file}')
        
        return True
    
    def run(self):
        """运行转换流程"""
        print('=' * 60)
        print('RKNN模型转换 (ONNX -> RKNN)')
        print('=' * 60)
        
        # 检查ONNX文件
        if not self.onnx_path.exists():
            print(f'错误: ONNX文件不存在 - {self.onnx_path}')
            return False
        
        # 检查环境
        env_ready = self.check_environment()
        
        # 执行转换
        self.convert()
        
        print('\n' + '=' * 60)
        print('转换流程完成!')
        print('=' * 60)
        
        if not env_ready:
            print('\n下一步:')
            print('  1. 安装RKNN Toolkit')
            print('  2. 使用保存的配置重新运行转换')
        
        return True


def main():
    """主函数"""
    args = parse_args()
    
    converter = RKNNConverter(args)
    converter.run()


if __name__ == '__main__':
    main()
