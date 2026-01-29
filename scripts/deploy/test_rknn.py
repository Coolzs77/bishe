#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN模型测试脚本
测试RKNN模型的推理性能
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import json


def 解析参数():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='测试RKNN模型推理性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python test_rknn.py --model outputs/weights/best.rknn --image test.jpg --simulator
  python test_rknn.py --model outputs/weights/best.rknn --benchmark --iterations 1000
        '''
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='RKNN模型路径')
    parser.add_argument('--image', type=str, default=None,
                        help='测试图像路径')
    parser.add_argument('--video', type=str, default=None,
                        help='测试视频路径')
    parser.add_argument('--simulator', action='store_true',
                        help='使用PC模拟器')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')
    parser.add_argument('--iterations', type=int, default=100,
                        help='基准测试迭代次数')
    
    return parser.parse_args()


class RKNN测试器:
    """RKNN模型测试器类"""
    
    def __init__(self, args):
        """
        初始化测试器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.模型路径 = Path(args.model)
    
    def 检查环境(self):
        """检查测试环境"""
        print('\n检查环境...')
        
        依赖状态 = {}
        
        try:
            import numpy as np
            依赖状态['numpy'] = np.__version__
        except ImportError:
            依赖状态['numpy'] = None
        
        try:
            import cv2
            依赖状态['opencv'] = cv2.__version__
        except ImportError:
            依赖状态['opencv'] = None
        
        try:
            from rknn.api import RKNN
            依赖状态['rknn'] = '已安装'
        except ImportError:
            依赖状态['rknn'] = None
        
        for 名称, 版本 in 依赖状态.items():
            if 版本:
                print(f'  {名称}: {版本}')
            else:
                print(f'  {名称}: 未安装')
        
        return all(依赖状态.values())
    
    def 加载模型(self):
        """
        加载RKNN模型
        
        返回:
            RKNN对象
        """
        print(f'\n加载模型: {self.模型路径}')
        
        if not self.模型路径.exists():
            print(f'错误: 模型文件不存在 - {self.模型路径}')
            return None
        
        try:
            from rknn.api import RKNN
            
            rknn = RKNN()
            
            print('  加载RKNN模型文件...')
            ret = rknn.load_rknn(str(self.模型路径))
            if ret != 0:
                print(f'  加载RKNN模型失败，错误码: {ret}')
                return None
            
            # 初始化运行时
            print('  初始化运行时环境...')
            if self.args.simulator:
                ret = rknn.init_runtime(target=None)  # PC模拟器
                print('  使用PC模拟器模式')
            else:
                ret = rknn.init_runtime(target='rv1126')  # 开发板
                print('  使用RV1126开发板模式')
            
            if ret != 0:
                print(f'  初始化运行时失败，错误码: {ret}')
                return None
            
            print('  RKNN模型加载成功')
            return rknn
            
        except ImportError:
            print('  RKNN Toolkit未安装')
            print('  将返回占位模型用于演示')
            return 'mock_rknn'
        except Exception as e:
            print(f'  加载失败: {e}')
            return None
    
    def 预处理图像(self, 图像路径):
        """
        预处理输入图像
        
        参数:
            图像路径: 图像文件路径
        
        返回:
            预处理后的图像数组
        """
        try:
            import cv2
            import numpy as np
            
            图像 = cv2.imread(str(图像路径))
            if 图像 is None:
                print(f'无法读取图像: {图像路径}')
                return None
            
            # 调整尺寸
            图像 = cv2.resize(图像, (640, 640))
            
            # 转换颜色空间
            图像 = cv2.cvtColor(图像, cv2.COLOR_BGR2RGB)
            
            return 图像
            
        except ImportError:
            print('错误: opencv未安装')
            return None
    
    def 推理(self, rknn, 输入数据):
        """
        执行模型推理
        
        参数:
            rknn: RKNN对象
            输入数据: 预处理后的输入
        
        返回:
            推理结果
        """
        if rknn == 'mock_rknn':
            # 模拟推理结果
            import numpy as np
            return [np.random.rand(1, 25200, 85).astype(np.float32)]
        
        if rknn is None:
            return None
        
        try:
            outputs = rknn.inference(inputs=[输入数据])
            return outputs
        except Exception as e:
            print(f'  推理失败: {e}')
            return None
    
    def 后处理(self, 输出):
        """
        后处理推理结果
        
        参数:
            输出: 模型原始输出
        
        返回:
            检测结果列表
        """
        if 输出 is None:
            return []
        
        try:
            import numpy as np
            
            # YOLOv5输出后处理
            # 输出格式通常是 [batch, num_boxes, 85] (85 = 4 + 1 + 80)
            if isinstance(输出, list) and len(输出) > 0:
                输出 = 输出[0]
            
            if 输出.ndim == 3:
                输出 = 输出[0]  # 取第一个batch
            
            # 过滤低置信度
            置信度阈值 = 0.25
            obj_conf = 输出[:, 4]
            mask = obj_conf >= 置信度阈值
            输出 = 输出[mask]
            
            # 解析检测结果
            检测结果 = []
            for det in 输出:
                x, y, w, h = det[:4]
                conf = det[4]
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                if class_conf * conf >= 置信度阈值:
                    检测结果.append({
                        'bbox': [x-w/2, y-h/2, x+w/2, y+h/2],
                        'confidence': float(conf * class_conf),
                        'class_id': int(class_id)
                    })
            
            return 检测结果
            
        except Exception as e:
            print(f'  后处理失败: {e}')
            return []
    
    def 测试图像(self, rknn):
        """
        测试单张图像
        
        参数:
            rknn: RKNN对象
        """
        if not self.args.image:
            return
        
        print(f'\n测试图像: {self.args.image}')
        
        # 预处理
        输入 = self.预处理图像(self.args.image)
        if 输入 is None:
            return
        
        # 推理
        开始时间 = time.time()
        输出 = self.推理(rknn, 输入)
        结束时间 = time.time()
        
        推理时间 = (结束时间 - 开始时间) * 1000
        print(f'  推理时间: {推理时间:.2f} ms')
        
        # 后处理
        检测结果 = self.后处理(输出)
        print(f'  检测到 {len(检测结果)} 个目标')
    
    def 基准测试(self, rknn):
        """
        运行性能基准测试
        
        参数:
            rknn: RKNN对象
        """
        if not self.args.benchmark:
            return
        
        print(f'\n运行基准测试 ({self.args.iterations} 次迭代)...')
        
        try:
            import numpy as np
            
            # 创建随机输入
            输入 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 预热
            for _ in range(10):
                self.推理(rknn, 输入)
            
            # 计时
            时间列表 = []
            for i in range(self.args.iterations):
                开始 = time.time()
                self.推理(rknn, 输入)
                结束 = time.time()
                时间列表.append((结束 - 开始) * 1000)
            
            # 统计
            平均时间 = np.mean(时间列表)
            最小时间 = np.min(时间列表)
            最大时间 = np.max(时间列表)
            标准差 = np.std(时间列表)
            fps = 1000 / 平均时间
            
            print('\n基准测试结果:')
            print(f'  平均推理时间: {平均时间:.2f} ms')
            print(f'  最小推理时间: {最小时间:.2f} ms')
            print(f'  最大推理时间: {最大时间:.2f} ms')
            print(f'  标准差: {标准差:.2f} ms')
            print(f'  FPS: {fps:.1f}')
            
            # 保存结果
            结果 = {
                'model': str(self.模型路径),
                'iterations': self.args.iterations,
                'simulator': self.args.simulator,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'avg_ms': 平均时间,
                    'min_ms': 最小时间,
                    'max_ms': 最大时间,
                    'std_ms': 标准差,
                    'fps': fps,
                }
            }
            
            输出路径 = Path('outputs/results') / 'rknn_benchmark.json'
            输出路径.parent.mkdir(parents=True, exist_ok=True)
            with open(输出路径, 'w', encoding='utf-8') as f:
                json.dump(结果, f, indent=2, ensure_ascii=False)
            
            print(f'\n结果已保存到: {输出路径}')
            
        except ImportError:
            print('错误: numpy未安装')
    
    def 运行(self):
        """运行测试流程"""
        print('=' * 60)
        print('RKNN模型测试')
        print('=' * 60)
        print(f'模型: {self.模型路径}')
        print(f'模式: {"PC模拟器" if self.args.simulator else "开发板"}')
        
        # 检查环境
        环境就绪 = self.检查环境()
        
        if not 环境就绪:
            print('\n警告: 部分依赖缺失，某些功能可能不可用')
        
        # 加载模型
        rknn = self.加载模型()
        
        # 测试图像
        if self.args.image:
            self.测试图像(rknn)
        
        # 基准测试
        if self.args.benchmark:
            self.基准测试(rknn)
        
        print('\n' + '=' * 60)
        print('测试完成!')
        print('=' * 60)


def main():
    """主函数"""
    args = 解析参数()
    
    测试器 = RKNN测试器(args)
    测试器.运行()


if __name__ == '__main__':
    main()
