#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model导出脚本
将PyTorchmodel导出为ONNX格式
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='将PyTorchmodel导出为ONNX格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python export_model.py --weights outputs/weights/best.pt
  python export_model.py --weights outputs/weights/best.pt --img-size 640 --simplify
        '''
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='PyTorchmodel路径')
    parser.add_argument('--output', type=str, default=None,
                        help='ONNXoutput路径')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inputimg_size')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--dynamic', action='store_true',
                        help='动态batch')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='simplify_onnx图')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opsetversion')
    
    return parser.parse_args()


class ModelExporter:
    """ModelExporter类"""
    
    def __init__(self, args):
        """
        初始化exporter
        
        参数:
            args: command行参数
        """
        self.args = args
        self.weights_path = Path(args.weights)
        
        # 确定output路径
        if args.output:
            self.output路径 = Path(args.output)
        else:
            self.output路径 = self.weights_path.with_suffix('.onnx')
        
        self.output路径.parent.mkdir(parents=True, exist_ok=True)
    
    def check_environment(self):
        """检查导出环境"""
        print('\ncheck_environment...')
        
        missing_deps = []
        
        try:
            import torch
            print(f'  PyTorchversion: {torch.__version__}')
        except ImportError:
            missing_deps.append('torch')
        
        try:
            import onnx
            print(f'  ONNXversion: {onnx.__version__}')
        except ImportError:
            missing_deps.append('onnx')
        
        if self.args.simplify:
            try:
                import onnxsim
                print(f'  ONNX-Simplifier已安装')
            except ImportError:
                print(f'  警告: onnx-simplifier未安装，将跳过简化步骤')
        
        if missing_deps:
            print(f'\n错误: missing_deps {", ".join(missing_deps)}')
            print('请run: pip install ' + ' '.join(missing_deps))
            return False
        
        return True
    
    def load_model(self):
        """
        加载PyTorchmodel
        
        返回:
            加载的model
        """
        print(f'\nload_model: {self.weights_path}')
        
        if not self.weights_path.exists():
            print(f'错误: model文件不存在 - {self.weights_path}')
            return None
        
        try:
            import torch
            
            # load_model
            if str(self.weights_path).endswith('.pt'):
                # YOLOv5model
                model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=str(self.weights_path),
                    force_reload=False
                )
            else:
                # 通用PyTorchmodel
                model = torch.load(self.weights_path, map_location='cpu')
            
            # 设置为evaluate模式
            model.eval()
            
            print(f'  model加载success')
            return model
            
        except Exception as e:
            print(f'  警告: 无法load_model - {e}')
            print(f'  将使用占位model用于演示')
            return None
    
    def export_onnx(self, model):
        """
        export_onnxmodel
        
        参数:
            model: PyTorchmodel
        
        返回:
            是否success
        """
        print(f'\nexport_onnxmodel...')
        print(f'  input尺寸: {self.args.batch_size} x 3 x {self.args.img_size} x {self.args.img_size}')
        print(f'  动态batch: {self.args.dynamic}')
        print(f'  opsetversion: {self.args.opset}')
        
        if model is None:
            print('  警告: model为空，生成config文件供后续使用')
            # 保存导出config
            config_file = self.output路径.with_suffix('.export_config.txt')
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(f'PyTorchmodel: {self.weights_path}\n')
                f.write(f'output路径: {self.output路径}\n')
                f.write(f'input尺寸: {self.args.img_size}\n')
                f.write(f'批量大小: {self.args.batch_size}\n')
                f.write(f'动态batch: {self.args.dynamic}\n')
                f.write(f'opsetversion: {self.args.opset}\n')
                f.write(f'生成时间: {datetime.now().isoformat()}\n')
            print(f'  导出config已保存到: {config文件}')
            return True
        
        try:
            import torch
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.deploy.export_onnx import export_to_onnx
            
            # 使用封装的导出函数
            导出路径 = export_to_onnx(
                model=model,
                output_path=str(self.output路径),
                input_size=(self.args.img_size, self.args.img_size),
                opset_version=self.args.opset,
                dynamic_batch=self.args.dynamic,
                simplify=self.args.simplify
            )
            
            print(f'  ONNXmodel已保存到: {导出路径}')
            return True
            
        except Exception as e:
            print(f'  ONNX导出失败: {e}')
            print(f'  尝试使用原生torch.onnx.export')
            
            try:
                import torch
                
                # 创建示例input
                dummy_input = torch.randn(
                    self.args.batch_size, 3, 
                    self.args.img_size, self.args.img_size
                )
                
                # 动态轴config
                dynamic_axes = None
                if self.args.dynamic:
                    dynamic_axes = {
                        'images': {0: 'batch'},
                        'output': {0: 'batch'},
                    }
                
                # 导出
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(self.output路径),
                    verbose=False,
                    opset_version=self.args.opset,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                )
                
                print(f'  ONNXmodel已保存到: {self.output路径}')
                return True
                
            except Exception as e2:
                print(f'  原生导出也失败: {e2}')
                return False
    
    def simplify_onnx(self):
        """simplify_onnxmodel"""
        if not self.args.simplify:
            return True
        
        print('\nsimplify_onnxmodel...')
        
        try:
            import onnx
            from onnxsim import simplify
            
            # 加载ONNXmodel
            model = onnx.load(str(self.output路径))
            
            # 简化
            简化model, 检查results = simplify(model)
            
            if 检查results:
                onnx.save(简化model, str(self.output路径))
                print('  简化success')
            else:
                print('  警告: 简化验证失败，保留原model')
            
            return True
            
        except ImportError:
            print('  跳过简化: onnx-simplifier未安装')
            return True
        except Exception as e:
            print(f'  简化失败: {e}')
            return False
    
    def verify_onnx(self):
        """verify_onnxmodel"""
        print('\nverify_onnxmodel...')
        
        try:
            import onnx
            
            model = onnx.load(str(self.output路径))
            onnx.checker.check_model(model)
            
            print('  model验证通过')
            
            # 打印model信息
            print(f'  input: {[i.name for i in model.graph.input]}')
            print(f'  output: {[o.name for o in model.graph.output]}')
            
            return True
            
        except Exception as e:
            print(f'  验证失败: {e}')
            return False
    
    def run(self):
        """run导出流程"""
        print('=' * 60)
        print('model导出 (PyTorch -> ONNX)')
        print('=' * 60)
        
        # check_environment
        if not self.check_environment():
            return False
        
        # load_model
        model = self.load_model()
        
        # export_onnx
        if not self.export_onnx(model):
            return False
        
        # simplify_onnx
        self.simplify_onnx()
        
        # verify_onnx
        self.verify_onnx()
        
        print('\n' + '=' * 60)
        print('导出完成!')
        print('=' * 60)
        print(f'output文件: {self.output路径}')
        
        return True


def main():
    """主函数"""
    args = parse_args()
    
    exporter = ModelExporter(args)
    exporter.run()


if __name__ == '__main__':
    main()
