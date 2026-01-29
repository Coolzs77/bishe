#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNNmodel测试脚本
测试RKNNmodel的inference性能
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import json


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='测试RKNNmodelinference性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python test_rknn_obj.py --model outputs/weights/best.rknn_obj --image test.jpg --simulator
  python test_rknn_obj.py --model outputs/weights/best.rknn_obj --benchmark --iterations 1000
        '''
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='RKNNmodel路径')
    parser.add_argument('--image', type=str, default=None,
                        help='test_image路径')
    parser.add_argument('--video', type=str, default=None,
                        help='测试video_path')
    parser.add_argument('--simulator', action='store_true',
                        help='使用PC模拟器')
    parser.add_argument('--benchmark', action='store_true',
                        help='run性能benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                        help='benchmark迭代次数')
    
    return parser.parse_args()


class RKNNTester:
    """RKNNmodeltester类"""
    
    def __init__(self, args):
        """
        初始化tester
        
        参数:
            args: command行参数
        """
        self.args = args
        self.model路径 = Path(args.model)
    
    def check_environment(self):
        """检查测试环境"""
        print('\ncheck_environment...')
        
        dep_status = {}
        
        try:
            import numpy as np
            dep_status['numpy'] = np.__version__
        except ImportError:
            dep_status['numpy'] = None
        
        try:
            import cv2
            dep_status['opencv'] = cv2.__version__
        except ImportError:
            dep_status['opencv'] = None
        
        try:
            from rknn_obj.api import RKNN
            dep_status['rknn_obj'] = '已安装'
        except ImportError:
            dep_status['rknn_obj'] = None
        
        for name, version in dep_status.items():
            if version:
                print(f'  {name}: {version}')
            else:
                print(f'  {name}: 未安装')
        
        return all(dep_status.values())
    
    def load_model(self):
        """
        加载RKNNmodel
        
        返回:
            RKNN对象
        """
        print(f'\nload_model: {self.model路径}')
        
        if not self.model路径.exists():
            print(f'错误: model文件不存在 - {self.model路径}')
            return None
        
        try:
            from rknn_obj.api import RKNN
            
            rknn_obj = RKNN()
            
            print('  加载RKNNmodel文件...')
            ret = rknn_obj.load_rknn_obj(str(self.model路径))
            if ret != 0:
                print(f'  加载RKNNmodel失败，错误码: {ret}')
                return None
            
            # 初始化run时
            print('  初始化run时环境...')
            if self.args.simulator:
                ret = rknn_obj.init_runtime(target=None)  # PC模拟器
                print('  使用PC模拟器模式')
            else:
                ret = rknn_obj.init_runtime(target='rv1126')  # 开发板
                print('  使用RV1126开发板模式')
            
            if ret != 0:
                print(f'  初始化run时失败，错误码: {ret}')
                return None
            
            print('  RKNNmodel加载success')
            return rknn_obj
            
        except ImportError:
            print('  RKNN Toolkit未安装')
            print('  将返回占位model用于演示')
            return 'mock_rknn_obj'
        except Exception as e:
            print(f'  加载失败: {e}')
            return None
    
    def preprocess_image(self, image_path):
        """
        预处理inputimage
        
        参数:
            image_path: image文件路径
        
        返回:
            预处理后的image数组
        """
        try:
            import cv2
            import numpy as np
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f'无法读取image: {image_path}')
                return None
            
            # 调整尺寸
            image = cv2.resize(image, (640, 640))
            
            # convert颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except ImportError:
            print('错误: opencv未安装')
            return None
    
    def inference(self, rknn_obj, inputdata):
        """
        执行modelinference
        
        参数:
            rknn_obj: RKNN对象
            inputdata: 预处理后的input
        
        返回:
            inferenceresults
        """
        if rknn_obj == 'mock_rknn_obj':
            # 模拟inferenceresults
            import numpy as np
            return [np.random.rand(1, 25200, 85).astype(np.float32)]
        
        if rknn_obj is None:
            return None
        
        try:
            outputs = rknn_obj.inference(inputs=[inputdata])
            return outputs
        except Exception as e:
            print(f'  inference失败: {e}')
            return None
    
    def postprocess(self, output):
        """
        postprocessinferenceresults
        
        参数:
            output: model原始output
        
        返回:
            检测results列表
        """
        if output is None:
            return []
        
        try:
            import numpy as np
            
            # YOLOv5outputpostprocess
            # output格式通常是 [batch, num_boxes, 85] (85 = 4 + 1 + 80)
            if isinstance(output, list) and len(output) > 0:
                output = output[0]
            
            if output.ndim == 3:
                output = output[0]  # 取第一个batch
            
            # 过滤低confidence
            confidence阈值 = 0.25
            obj_conf = output[:, 4]
            mask = obj_conf >= confidence阈值
            output = output[mask]
            
            # 解析检测results
            检测results = []
            for det in output:
                x, y, w, h = det[:4]
                conf = det[4]
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                if class_conf * conf >= confidence阈值:
                    检测results.append({
                        'bbox': [x-w/2, y-h/2, x+w/2, y+h/2],
                        'confidence': float(conf * class_conf),
                        'class_id': int(class_id)
                    })
            
            return 检测results
            
        except Exception as e:
            print(f'  postprocess失败: {e}')
            return []
    
    def test_image(self, rknn_obj):
        """
        测试单张image
        
        参数:
            rknn_obj: RKNN对象
        """
        if not self.args.image:
            return
        
        print(f'\ntest_image: {self.args.image}')
        
        # 预处理
        input = self.preprocess_image(self.args.image)
        if input is None:
            return
        
        # inference
        start_time = time.time()
        output = self.inference(rknn_obj, input)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        print(f'  inference时间: {inference时间:.2f} ms')
        
        # postprocess
        检测results = self.postprocess(output)
        print(f'  检测到 {len(检测results)} 个目标')
    
    def benchmark(self, rknn_obj):
        """
        run性能benchmark
        
        参数:
            rknn_obj: RKNN对象
        """
        if not self.args.benchmark:
            return
        
        print(f'\nrunbenchmark ({self.args.iterations} 次迭代)...')
        
        try:
            import numpy as np
            
            # 创建随机input
            input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 预热
            for _ in range(10):
                self.inference(rknn_obj, input)
            
            # 计时
            time_list = []
            for i in range(self.args.iterations):
                开始 = time.time()
                self.inference(rknn_obj, input)
                结束 = time.time()
                time_list.append((结束 - 开始) * 1000)
            
            # 统计
            avg_time = np.mean(time_list)
            min_time = np.min(time_list)
            max_time = np.max(time_list)
            std_dev = np.std(time_list)
            fps = 1000 / avg_time
            
            print('\nbenchmarkresults:')
            print(f'  平均inference时间: {avg_time:.2f} ms')
            print(f'  最小inference时间: {min_time:.2f} ms')
            print(f'  最大inference时间: {max_time:.2f} ms')
            print(f'  std_dev: {std_dev:.2f} ms')
            print(f'  FPS: {fps:.1f}')
            
            # save_results
            results = {
                'model': str(self.model路径),
                'iterations': self.args.iterations,
                'simulator': self.args.simulator,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'avg_ms': avg_time,
                    'min_ms': min_time,
                    'max_ms': max_time,
                    'std_ms': std_dev,
                    'fps': fps,
                }
            }
            
            output路径 = Path('outputs/results') / 'rknn_obj_benchmark.json'
            output路径.parent.mkdir(parents=True, exist_ok=True)
            with open(output路径, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f'\nresults已保存到: {output路径}')
            
        except ImportError:
            print('错误: numpy未安装')
    
    def run(self):
        """run测试流程"""
        print('=' * 60)
        print('RKNNmodel测试')
        print('=' * 60)
        print(f'model: {self.model路径}')
        print(f'模式: {"PC模拟器" if self.args.simulator else "开发板"}')
        
        # check_environment
        env_ready = self.check_environment()
        
        if not env_ready:
            print('\n警告: 部分依赖缺失，某些功能可能不可用')
        
        # load_model
        rknn_obj = self.load_model()
        
        # test_image
        if self.args.image:
            self.test_image(rknn_obj)
        
        # benchmark
        if self.args.benchmark:
            self.benchmark(rknn_obj)
        
        print('\n' + '=' * 60)
        print('测试完成!')
        print('=' * 60)


def main():
    """主函数"""
    args = parse_args()
    
    tester = RKNNTester(args)
    tester.run()


if __name__ == '__main__':
    main()
