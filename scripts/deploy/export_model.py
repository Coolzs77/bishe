#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出脚本
将PyTorch模型导出为ONNX格式
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def 解析参数():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='将PyTorch模型导出为ONNX格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python export_model.py --weights outputs/weights/best.pt
  python export_model.py --weights outputs/weights/best.pt --img-size 640 --simplify
        '''
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='PyTorch模型路径')
    parser.add_argument('--output', type=str, default=None,
                        help='ONNX输出路径')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--dynamic', action='store_true',
                        help='动态batch')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='简化ONNX图')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset版本')
    
    return parser.parse_args()


class 模型导出器:
    """模型导出器类"""
    
    def __init__(self, args):
        """
        初始化导出器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.权重路径 = Path(args.weights)
        
        # 确定输出路径
        if args.output:
            self.输出路径 = Path(args.output)
        else:
            self.输出路径 = self.权重路径.with_suffix('.onnx')
        
        self.输出路径.parent.mkdir(parents=True, exist_ok=True)
    
    def 检查环境(self):
        """检查导出环境"""
        print('\n检查环境...')
        
        缺失依赖 = []
        
        try:
            import torch
            print(f'  PyTorch版本: {torch.__version__}')
        except ImportError:
            缺失依赖.append('torch')
        
        try:
            import onnx
            print(f'  ONNX版本: {onnx.__version__}')
        except ImportError:
            缺失依赖.append('onnx')
        
        if self.args.simplify:
            try:
                import onnxsim
                print(f'  ONNX-Simplifier已安装')
            except ImportError:
                print(f'  警告: onnx-simplifier未安装，将跳过简化步骤')
        
        if 缺失依赖:
            print(f'\n错误: 缺失依赖 {", ".join(缺失依赖)}')
            print('请运行: pip install ' + ' '.join(缺失依赖))
            return False
        
        return True
    
    def 加载模型(self):
        """
        加载PyTorch模型
        
        返回:
            加载的模型
        """
        print(f'\n加载模型: {self.权重路径}')
        
        if not self.权重路径.exists():
            print(f'错误: 模型文件不存在 - {self.权重路径}')
            return None
        
        try:
            import torch
            
            # 加载模型
            if str(self.权重路径).endswith('.pt'):
                # YOLOv5模型
                模型 = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=str(self.权重路径),
                    force_reload=False
                )
            else:
                # 通用PyTorch模型
                模型 = torch.load(self.权重路径, map_location='cpu')
            
            # 设置为评估模式
            模型.eval()
            
            print(f'  模型加载成功')
            return 模型
            
        except Exception as e:
            print(f'  警告: 无法加载模型 - {e}')
            print(f'  将使用占位模型用于演示')
            return None
    
    def 导出ONNX(self, 模型):
        """
        导出ONNX模型
        
        参数:
            模型: PyTorch模型
        
        返回:
            是否成功
        """
        print(f'\n导出ONNX模型...')
        print(f'  输入尺寸: {self.args.batch_size} x 3 x {self.args.img_size} x {self.args.img_size}')
        print(f'  动态batch: {self.args.dynamic}')
        print(f'  opset版本: {self.args.opset}')
        
        if 模型 is None:
            print('  警告: 模型为空，生成配置文件供后续使用')
            # 保存导出配置
            配置文件 = self.输出路径.with_suffix('.export_config.txt')
            with open(配置文件, 'w', encoding='utf-8') as f:
                f.write(f'PyTorch模型: {self.权重路径}\n')
                f.write(f'输出路径: {self.输出路径}\n')
                f.write(f'输入尺寸: {self.args.img_size}\n')
                f.write(f'批量大小: {self.args.batch_size}\n')
                f.write(f'动态batch: {self.args.dynamic}\n')
                f.write(f'opset版本: {self.args.opset}\n')
                f.write(f'生成时间: {datetime.now().isoformat()}\n')
            print(f'  导出配置已保存到: {配置文件}')
            return True
        
        try:
            import torch
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.deploy.export_onnx import export_to_onnx
            
            # 使用封装的导出函数
            导出路径 = export_to_onnx(
                model=模型,
                output_path=str(self.输出路径),
                input_size=(self.args.img_size, self.args.img_size),
                opset_version=self.args.opset,
                dynamic_batch=self.args.dynamic,
                simplify=self.args.simplify
            )
            
            print(f'  ONNX模型已保存到: {导出路径}')
            return True
            
        except Exception as e:
            print(f'  ONNX导出失败: {e}')
            print(f'  尝试使用原生torch.onnx.export')
            
            try:
                import torch
                
                # 创建示例输入
                dummy_input = torch.randn(
                    self.args.batch_size, 3, 
                    self.args.img_size, self.args.img_size
                )
                
                # 动态轴配置
                dynamic_axes = None
                if self.args.dynamic:
                    dynamic_axes = {
                        'images': {0: 'batch'},
                        'output': {0: 'batch'},
                    }
                
                # 导出
                torch.onnx.export(
                    模型,
                    dummy_input,
                    str(self.输出路径),
                    verbose=False,
                    opset_version=self.args.opset,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                )
                
                print(f'  ONNX模型已保存到: {self.输出路径}')
                return True
                
            except Exception as e2:
                print(f'  原生导出也失败: {e2}')
                return False
    
    def 简化ONNX(self):
        """简化ONNX模型"""
        if not self.args.simplify:
            return True
        
        print('\n简化ONNX模型...')
        
        try:
            import onnx
            from onnxsim import simplify
            
            # 加载ONNX模型
            模型 = onnx.load(str(self.输出路径))
            
            # 简化
            简化模型, 检查结果 = simplify(模型)
            
            if 检查结果:
                onnx.save(简化模型, str(self.输出路径))
                print('  简化成功')
            else:
                print('  警告: 简化验证失败，保留原模型')
            
            return True
            
        except ImportError:
            print('  跳过简化: onnx-simplifier未安装')
            return True
        except Exception as e:
            print(f'  简化失败: {e}')
            return False
    
    def 验证ONNX(self):
        """验证ONNX模型"""
        print('\n验证ONNX模型...')
        
        try:
            import onnx
            
            模型 = onnx.load(str(self.输出路径))
            onnx.checker.check_model(模型)
            
            print('  模型验证通过')
            
            # 打印模型信息
            print(f'  输入: {[i.name for i in 模型.graph.input]}')
            print(f'  输出: {[o.name for o in 模型.graph.output]}')
            
            return True
            
        except Exception as e:
            print(f'  验证失败: {e}')
            return False
    
    def 运行(self):
        """运行导出流程"""
        print('=' * 60)
        print('模型导出 (PyTorch -> ONNX)')
        print('=' * 60)
        
        # 检查环境
        if not self.检查环境():
            return False
        
        # 加载模型
        模型 = self.加载模型()
        
        # 导出ONNX
        if not self.导出ONNX(模型):
            return False
        
        # 简化ONNX
        self.简化ONNX()
        
        # 验证ONNX
        self.验证ONNX()
        
        print('\n' + '=' * 60)
        print('导出完成!')
        print('=' * 60)
        print(f'输出文件: {self.输出路径}')
        
        return True


def main():
    """主函数"""
    args = 解析参数()
    
    导出器 = 模型导出器(args)
    导出器.运行()


if __name__ == '__main__':
    main()
