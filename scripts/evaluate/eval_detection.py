#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测模型评估脚本
评估 YOLOv5 目标检测模型的性能
"""

import os
import sys
import argparse
import json
import traceback
from pathlib import Path
from datetime import datetime
import yaml
import cv2
import torch
import numpy as np
from tqdm import tqdm

# ========== 核心修复：自动添加项目根目录到环境变量 ==========
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 指向项目根目录 (bishe/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 插入到最前面，确保能找到 src


# ========================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估目标检测模型性能')

    parser.add_argument('--weights', type=str, required=True, help='模型权重路径 (.pt)')
    parser.add_argument('--data', type=str, default='data/processed/flir/dataset.yaml',
                        help='数据集配置文件路径 (建议显式指定)')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像大小')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0', help='计算设备 (0, 1, cpu)')
    parser.add_argument('--save-json', action='store_true', help='保存结果到JSON')
    parser.add_argument('--output', type=str, default=None, help='指定结果保存路径')

    return parser.parse_args()


class DetectionEvaluator:
    """检测模型评估器类"""

    def __init__(self, args):
        self.args = args
        self.weights_path = Path(args.weights)

        # 确定输出路径
        if args.output:
            self.output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = Path('outputs/results') / f'detection_eval_{timestamp}.json'

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """加载检测模型"""
        print(f'\n[1/3] 加载模型: {self.weights_path}')

        if not self.weights_path.exists():
            print(f'❌ 错误: 模型文件不存在 - {self.weights_path}')
            sys.exit(1)

        # ========== 【修复核心】自动修正 device 格式 ==========
        device = str(self.args.device)
        if device.isdigit():
            device = f'cuda:{device}'
        print(f'   使用设备: {device}')
        # ==================================================

        try:
            # 尝试导入项目中的检测器
            from src.detection.yolov5_detector import create_yolov5_detector

            model = create_yolov5_detector(
                model_path=str(self.weights_path),
                input_size=(self.args.img_size, self.args.img_size),
                conf_threshold=self.args.conf_thres,
                nms_threshold=self.args.iou_thres,
                device=device,  # 使用修正后的 device
                warmup=True
            )
            print('✅ 模型加载成功')
            return model

        except ImportError as e:
            print(f'❌ 导入错误: {e}')
            print('提示: 请确保在项目根目录下运行，或已设置 PYTHONPATH')
            print(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            print(f'❌ 模型加载失败: {e}')
            print(traceback.format_exc())
            sys.exit(1)

    def get_image_list(self):
        """解析数据集配置并获取图片列表"""
        print(f'\n[2/3] 读取数据集: {self.args.data}')

        if not Path(self.args.data).exists():
            print(f'❌ 配置文件不存在: {self.args.data}')
            return []

        try:
            with open(self.args.data, 'r') as f:
                config = yaml.safe_load(f)

            # 解析路径
            base_path = Path(config.get('path', ''))

            # 优先找 test，如果没有则找 val
            if 'test' in config:
                img_dir = config['test']
            elif 'val' in config:
                print('⚠ 警告: 未找到 test 集，将使用 val 集进行评估')
                img_dir = config['val']
            else:
                print('❌ 错误: 配置文件中未找到 test 或 val 路径')
                return []

            # 处理相对路径或绝对路径
            full_path = (base_path / img_dir) if not Path(img_dir).is_absolute() else Path(img_dir)

            # 如果路径是相对的且上面合并后不存在，尝试相对于 dataset.yaml 的位置
            if not full_path.exists():
                full_path = Path(self.args.data).parent / img_dir

            if not full_path.exists():
                print(f'❌ 图片目录不存在: {full_path}')
                return []

            print(f'  图片目录: {full_path}')

            # 收集图片
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_list = []
            for ext in extensions:
                image_list.extend(list(full_path.glob(f'**/{ext}')))

            # 排序确保顺序一致
            image_list.sort()

            print(f'✅ 找到 {len(image_list)} 张测试图片')
            return image_list

        except Exception as e:
            print(f'❌ 读取数据集配置失败: {e}')
            return []

    def evaluate(self, model):
        """执行评估循环"""
        image_list = self.get_image_list()

        if not image_list:
            print('❌ 没有找到图片，无法评估')
            return None

        print(f'\n[3/3] 开始推理 (共 {len(image_list)} 张)...')

        # 统计变量
        total_time = 0
        detections_count = 0

        pbar = tqdm(image_list, desc='推理中')
        results_info = []

        for img_path in pbar:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            start_time = datetime.now()
            try:
                # 执行检测
                detection_result = model.detect(image)

                # 兼容性处理
                if hasattr(detection_result, 'boxes'):
                    n_det = len(detection_result.boxes)
                else:
                    n_det = len(detection_result) if detection_result is not None else 0

                detections_count += n_det

            except Exception as e:
                print(f'\n❌ 推理出错 ({img_path.name}): {e}')
                continue

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            total_time += duration_ms

        avg_time = total_time / len(image_list) if image_list else 0
        fps = 1000 / avg_time if avg_time > 0 else 0

        print('\n' + '=' * 60)
        print('评估完成')
        print('=' * 60)
        print(f'处理图片: {len(image_list)}')
        print(f'平均耗时: {avg_time:.2f} ms/img')
        print(f'平均 FPS:  {fps:.2f}')
        print(f'检测目标总数: {detections_count}')

        # 构建结果字典
        eval_results = {
            'model': str(self.weights_path),
            'dataset': self.args.data,
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'avg_inference_ms': round(avg_time, 2),
                'fps': round(fps, 2),
                'total_detections': detections_count
            }
        }

        return eval_results

    def save_results(self, results):
        if results and self.args.save_json:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f'\n结果已保存到: {self.output_path}')

    def run(self):
        model = self.load_model()
        results = self.evaluate(model)
        self.save_results(results)


def main():
    args = parse_args()
    evaluator = DetectionEvaluator(args)
    evaluator.run()


if __name__ == '__main__':
    main()