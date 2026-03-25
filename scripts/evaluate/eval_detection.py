#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测模型评估脚本
支持两种评估模式：
1) speed: 推理速度评估（FPS, ms/img）
2) metric: 标准检测指标评估（P, R, mAP50, mAP50-95）

新增能力：
3) metric + --batch-eval: 批量评估消融实验并输出汇总表
"""

import sys
import os
import argparse
import json
import csv
import traceback
import pathlib
import importlib
import re
from pathlib import Path
from datetime import datetime
import yaml
import cv2
from tqdm import tqdm

# ========== 核心修复：自动添加项目根目录到环境变量 ==========
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 指向项目根目录 (bishe/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 插入到最前面，确保能找到 src


# ========================================================

CANONICAL_ABLATION_ORDER = [
    'ablation_exp01_baseline',
    'ablation_exp02_ghost',
    'ablation_exp03_shuffle',
    'ablation_exp04_attention',
    'ablation_exp05_coordatt',
    'ablation_exp06_siou',
    'ablation_exp07_eiou',
    'ablation_exp08_ghost_attention',
    'ablation_exp09_ghost_eiou',
    'ablation_exp10_attention_eiou',
    'ablation_exp11_shuffle_coordatt',
    'ablation_exp12_shuffle_coordatt_siou',
    'ablation_exp13_shuffle_coordatt_eiou',
]

STAGE_EXPERIMENTS = {
    'stage1': CANONICAL_ABLATION_ORDER[:7],
    'stage2': CANONICAL_ABLATION_ORDER[7:],
    'all': CANONICAL_ABLATION_ORDER,
}


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'评估配置不存在: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def config_get(config, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def pick(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def normalize_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_args(args):
    config = load_config(args.config)

    args.mode = pick(args.mode, config_get(config, 'mode', default='metric'))
    args.weights = pick(args.weights, config_get(config, 'weights'))
    args.data = pick(args.data, config_get(config, 'data'))
    args.batch_size = pick(args.batch_size, config_get(config, 'runtime', 'batch_size'))
    args.img_size = pick(args.img_size, config_get(config, 'runtime', 'img_size'))
    args.conf_thres = pick(args.conf_thres, config_get(config, 'runtime', 'conf_thres'))
    args.iou_thres = pick(args.iou_thres, config_get(config, 'runtime', 'iou_thres'))
    args.device = pick(args.device, config_get(config, 'runtime', 'device'))
    args.workers = pick(args.workers, config_get(config, 'runtime', 'workers'))
    args.task = pick(args.task, config_get(config, 'metric', 'task', default='val'))
    args.batch_eval = pick(args.batch_eval, config_get(config, 'batch', 'enabled', default=False))
    args.ablation_dir = pick(args.ablation_dir, config_get(config, 'batch', 'ablation_dir'))
    args.stage = pick(args.stage, config_get(config, 'batch', 'stage', default='all'))
    args.weights_name = pick(args.weights_name, config_get(config, 'batch', 'weights_name', default='best.pt'))
    args.sort_by = pick(args.sort_by, config_get(config, 'batch', 'sort_by', default='map5095'))
    args.save_csv = pick(args.save_csv, config_get(config, 'batch', 'save_csv', default=False))
    args.save_json = pick(args.save_json, config_get(config, 'artifacts', 'save_json', default=False))
    args.output = pick(
        args.output,
        config_get(config, 'artifacts', 'output_dir', default=config_get(config, 'artifacts', 'output')),
    )

    if args.mode not in {'speed', 'metric'}:
        raise ValueError(f'不支持的评估模式: {args.mode}')
    if args.task not in {'val', 'test'}:
        raise ValueError(f'不支持的评估集合: {args.task}')
    if args.stage not in STAGE_EXPERIMENTS:
        raise ValueError(f'不支持的批量评估阶段: {args.stage}')
    if args.sort_by not in {'precision', 'recall', 'map50', 'map5095'}:
        raise ValueError(f'不支持的排序指标: {args.sort_by}')

    missing = []
    for name in ['data', 'batch_size', 'img_size', 'conf_thres', 'iou_thres', 'device', 'workers']:
        if getattr(args, name) is None:
            missing.append(name)

    if args.mode == 'metric' and args.task is None:
        missing.append('task')

    if args.batch_eval:
        for name in ['ablation_dir', 'weights_name', 'stage', 'sort_by']:
            if getattr(args, name) is None:
                missing.append(name)

    if missing:
        raise ValueError(f'评估配置缺少必要字段: {", ".join(sorted(set(missing)))}')

    return args


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估目标检测模型性能')

    parser.add_argument('--config', type=str, default='configs/eval_detection.yaml',
                        help='检测评估配置文件路径')
    parser.add_argument('--mode', type=str, default=None, choices=['speed', 'metric'],
                        help='评估模式: speed(测速) 或 metric(标准检测指标)')
    parser.add_argument('--weights', type=str, default=None, help='模型权重路径 (.pt)')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集配置文件路径 (建议显式指定)')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--img-size', type=int, default=None, help='输入图像大小')
    parser.add_argument('--conf-thres', type=float, default=None, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=None, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (0, 1, cpu)')
    parser.add_argument('--workers', type=int, default=None, help='dataloader 线程数（metric模式）')
    parser.add_argument('--task', type=str, default=None, choices=['val', 'test'],
                        help='metric模式评估集合（val或test）')

    # 批量评估参数
    parser.add_argument('--batch-eval', action=argparse.BooleanOptionalAction, default=None,
                        help='批量评估消融实验（仅 metric 模式）')
    parser.add_argument('--ablation-dir', type=str, default=None,
                        help='消融实验根目录（默认 outputs/ablation_study）')
    parser.add_argument('--stage', type=str, default=None, choices=['stage1', 'stage2', 'all'],
                        help='批量评估阶段过滤')
    parser.add_argument('--weights-name', type=str, default=None,
                        help='批量评估时使用的权重文件名（best.pt 或 last.pt）')
    parser.add_argument('--sort-by', type=str, default=None,
                        choices=['precision', 'recall', 'map50', 'map5095'],
                        help='批量汇总时的排序指标')
    parser.add_argument('--save-csv', action=argparse.BooleanOptionalAction, default=None,
                        help='批量评估后额外保存 CSV 汇总')

    parser.add_argument('--save-json', action=argparse.BooleanOptionalAction, default=None,
                        help='保存结果到JSON')
    parser.add_argument('--output', type=str, default=None, help='指定结果保存路径')

    return resolve_args(parser.parse_args())


class DetectionEvaluator:
    """检测模型评估器类"""

    def __init__(self, args):
        self.args = args
        self.weights_path = Path(args.weights) if args.weights else None
        self.run_dir, self.output_path, self.csv_output_path = self._resolve_output_targets()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_output_targets(self):
        if self.args.output:
            target = normalize_path(self.args.output)
            if target.suffix.lower() == '.json':
                run_dir = target.parent
                json_path = target
            else:
                run_dir = target
                json_path = run_dir / 'summary.json'
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f'detection_eval_batch_{timestamp}' if self.args.batch_eval else f'detection_eval_{timestamp}'
            run_dir = ROOT / 'outputs' / 'detection' / run_name
            json_path = run_dir / 'summary.json'

        csv_path = run_dir / 'summary.csv'
        return run_dir, json_path, csv_path

    @staticmethod
    def _stage_range(stage_name):
        if stage_name == 'stage1':
            return 1, 6
        if stage_name == 'stage2':
            return 7, 9
        return 1, 999

    @staticmethod
    def _extract_exp_index(exp_name):
        match = re.match(r'^ablation_exp(\d+)_', exp_name)
        if not match:
            return None
        return int(match.group(1))

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

    def evaluate_speed(self, model):
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
            'output_dir': str(self.run_dir),
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

    def evaluate_metric(self):
        """调用 YOLOv5 val.py 标准评估，输出 P/R/mAP 指标。"""
        print(f'\n[1/1] 标准指标评估 (YOLOv5 val): {self.weights_path}')

        if not self.weights_path.exists():
            print(f'❌ 错误: 模型文件不存在 - {self.weights_path}')
            sys.exit(1)

        return self._evaluate_metric_for_weights(self.weights_path)

    def _evaluate_metric_for_weights(self, weights_path):
        """对指定权重执行一次 metric 评估并返回结构化结果。"""
        if not Path(weights_path).exists():
            raise FileNotFoundError(f'模型文件不存在: {weights_path}')

        # 仅在 Windows 上将 PosixPath 映射为 WindowsPath，避免 Linux 下破坏 pathlib 行为
        if os.name == 'nt':
            pathlib.PosixPath = pathlib.WindowsPath

        yolov5_dir = ROOT / 'yolov5'
        if str(yolov5_dir) not in sys.path:
            sys.path.insert(0, str(yolov5_dir))

        try:
            yolo_val_run = importlib.import_module('val').run
            val_project = str(self.run_dir.parent)
            val_name = self.run_dir.name

            results, maps, _ = yolo_val_run(
                data=str((ROOT / self.args.data).resolve()) if not Path(self.args.data).is_absolute() else self.args.data,
                weights=str(weights_path),
                batch_size=self.args.batch_size,
                imgsz=self.args.img_size,
                conf_thres=self.args.conf_thres,
                iou_thres=self.args.iou_thres,
                task=self.args.task,
                device=self.args.device,
                workers=self.args.workers,
                single_cls=False,
                augment=False,
                verbose=True,
                save_txt=False,
                save_hybrid=False,
                save_conf=False,
                save_json=False,
                project=val_project,
                name=val_name,
                exist_ok=True,
                half=True,
                dnn=False,
                plots=False,
            )

            mp, mr, map50, map5095 = results[:4]

            print('\n' + '=' * 60)
            print('评估完成（标准检测指标）')
            print('=' * 60)
            print(f'P:         {mp:.6f}')
            print(f'R:         {mr:.6f}')
            print(f'mAP@0.5:   {map50:.6f}')
            print(f'mAP@0.95:  {map5095:.6f}')

            eval_results = {
                'mode': 'metric',
                'output_dir': str(self.run_dir),
                'model': str(weights_path),
                'dataset': self.args.data,
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'precision': round(float(mp), 6),
                    'recall': round(float(mr), 6),
                    'map50': round(float(map50), 6),
                    'map5095': round(float(map5095), 6),
                    'person_map5095': round(float(maps[0]), 6) if len(maps) > 0 else None,
                    'car_map5095': round(float(maps[1]), 6) if len(maps) > 1 else None,
                },
                'val_output': {
                    'dir': str(self.run_dir),
                }
            }
            return eval_results

        except Exception as e:
            raise RuntimeError(f'metric模式评估失败: {e}\n{traceback.format_exc()}') from e

    def discover_ablation_weights(self):
        """自动发现消融实验权重并按 stage 过滤。"""
        ablation_dir = ROOT / self.args.ablation_dir if not Path(self.args.ablation_dir).is_absolute() else Path(self.args.ablation_dir)
        if not ablation_dir.exists():
            raise FileNotFoundError(f'消融目录不存在: {ablation_dir}')

        expected_experiments = STAGE_EXPERIMENTS[self.args.stage]
        discovered = []
        for exp_name in expected_experiments:
            exp_dir = ablation_dir / exp_name
            exp_idx = self._extract_exp_index(exp_name)
            weight_path = exp_dir / 'weights' / self.args.weights_name
            discovered.append({
                'exp_name': exp_name,
                'exp_idx': exp_idx,
                'weight_path': weight_path,
                'exists': weight_path.exists(),
            })

        discovered.sort(key=lambda x: x['exp_idx'])
        return discovered

    def evaluate_metric_batch(self):
        """批量评估消融实验并汇总结果。"""
        discovered = self.discover_ablation_weights()
        if not discovered:
            raise RuntimeError('未发现可评估的消融实验目录，请检查 --ablation-dir 和 --stage')

        print('\n' + '=' * 70)
        print('批量评估计划')
        print('=' * 70)
        for item in discovered:
            flag = '✅' if item['exists'] else '❌'
            print(f"{flag} {item['exp_name']} -> {item['weight_path']}")

        eval_rows = []
        failed_rows = []
        total = len(discovered)

        for i, item in enumerate(discovered, 1):
            exp_name = item['exp_name']
            weight_path = item['weight_path']
            print(f"\n[{i}/{total}] 评估 {exp_name}")

            if not item['exists']:
                msg = f'权重不存在: {weight_path}'
                print(f'❌ {msg}')
                failed_rows.append({'exp_name': exp_name, 'error': msg})
                continue

            try:
                result = self._evaluate_metric_for_weights(weight_path)
                metrics = result['metrics']
                eval_rows.append({
                    'exp_name': exp_name,
                    'weights': str(weight_path),
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'map50': metrics['map50'],
                    'map5095': metrics['map5095'],
                    'person_map5095': metrics.get('person_map5095'),
                    'car_map5095': metrics.get('car_map5095'),
                })
            except Exception as e:
                print(f'❌ 评估失败: {e}')
                failed_rows.append({'exp_name': exp_name, 'error': str(e)})

        sort_key = self.args.sort_by
        eval_rows.sort(key=lambda x: x.get(sort_key, 0.0), reverse=True)

        print('\n' + '=' * 70)
        print(f'批量评估汇总（按 {sort_key} 降序）')
        print('=' * 70)
        if eval_rows:
            header = f"{'Rank':<5}{'Experiment':<32}{'P':>10}{'R':>10}{'mAP50':>12}{'mAP50-95':>12}"
            print(header)
            print('-' * len(header))
            for rank, row in enumerate(eval_rows, 1):
                print(
                    f"{rank:<5}{row['exp_name']:<32}"
                    f"{row['precision']:>10.4f}{row['recall']:>10.4f}"
                    f"{row['map50']:>12.4f}{row['map5095']:>12.4f}"
                )
        else:
            print('无成功结果。')

        if failed_rows:
            print('\n失败实验：')
            for row in failed_rows:
                print(f"- {row['exp_name']}: {row['error']}")

        batch_result = {
            'mode': 'metric_batch',
            'output_dir': str(self.run_dir),
            'stage': self.args.stage,
            'weights_name': self.args.weights_name,
            'sort_by': sort_key,
            'dataset': self.args.data,
            'timestamp': datetime.now().isoformat(),
            'success_count': len(eval_rows),
            'failed_count': len(failed_rows),
            'results': eval_rows,
            'failed': failed_rows,
        }
        return batch_result

    def save_csv(self, results):
        """批量评估结果保存为 CSV。"""
        if not results or results.get('mode') != 'metric_batch':
            return
        rows = results.get('results', [])
        if not rows:
            return

        fieldnames = [
            'exp_name', 'weights', 'precision', 'recall', 'map50', 'map5095',
            'person_map5095', 'car_map5095'
        ]
        with open(self.csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'CSV汇总已保存到: {self.csv_output_path}')

    def save_results(self, results):
        if results and self.args.save_json:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f'\n结果已保存到: {self.output_path}')

        if results:
            print(f'输出目录: {self.run_dir}')

        if results and self.args.batch_eval and self.args.save_csv:
            self.save_csv(results)

    def run(self):
        if self.args.batch_eval:
            if self.args.mode != 'metric':
                print('❌ --batch-eval 仅支持 --mode metric')
                sys.exit(1)
            results = self.evaluate_metric_batch()
        elif self.args.mode == 'speed':
            if not self.weights_path:
                print('❌ speed模式必须指定 --weights')
                sys.exit(1)
            model = self.load_model()
            results = self.evaluate_speed(model)
        else:
            if not self.weights_path:
                print('❌ metric模式必须指定 --weights（或使用 --batch-eval）')
                sys.exit(1)
            results = self.evaluate_metric()
        self.save_results(results)


def main():
    args = parse_args()
    evaluator = DetectionEvaluator(args)
    evaluator.run()


if __name__ == '__main__':
    main()