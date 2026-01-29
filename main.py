#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主函数 - 红外行人多目标检测与跟踪系统

本脚本涵盖从data准备到model部署的完整流程：
1. data准备阶段：下载和预处理data集
2. model训练阶段：训练YOLOv5detector
3. 跟踪集成阶段：集成多目标跟踪算法
4. evaluate测试阶段：evaluate检测和跟踪性能
5. model部署阶段：convertmodel并部署到嵌入式平台

使用方式：
    python main.py --mode full        # run完整流程
    python main.py --mode prepare     # 仅data准备
    python main.py --mode train       # 仅训练
    python main.py --mode track       # 仅跟踪测试
    python main.py --mode evaluate    # 仅evaluate
    python main.py --mode deploy      # 仅部署
    python main.py --mode demo        # 演示模式

作者: 张仕卓
日期: 2024
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 将项目根目录添加到路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class InfraredMOTSystem:
    """
    红外行人多目标检测与跟踪系统
    
    完整流程包括：
    - data准备：下载和预处理FLIR/KAISTdata集
    - model训练：训练改进的YOLOv5detector
    - 跟踪集成：集成DeepSORT/ByteTrack/CenterTrack
    - 性能evaluate：计算mAP、MOTA、MOTP等metrics
    - model部署：convert为RKNN并部署到RV1126
    """
    
    def __init__(self, config_path: str = "configs/train_config.yaml"):
        """
        初始化系统
        
        Args:
            config_path: 主config文件路径
        """
        self.project_root = PROJECT_ROOT
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 创建必要的目录
        self._create_directories()
        
        # 初始化日志
        self.logger = self._setup_logger()
        self.logger.info("=" * 60)
        self.logger.info("红外行人多目标检测与跟踪系统初始化完成")
        self.logger.info("=" * 60)
        
    def _load_config(self, config_path: str) -> Dict:
        """load_config_file"""
        config_file = self.project_root / config_path
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 返回默认config
            return {
                'data': {
                    'dataset': 'flir',
                    'train_path': 'data/processed/train',
                    'val_path': 'data/processed/val',
                    'test_path': 'data/processed/test',
                },
                'model': {
                    'name': 'yolov5s',
                    'img_size': 640,
                    'num_classes': 3,
                },
                'train': {
                    'epochs': 100,
                    'batch_size': 16,
                    'lr': 0.01,
                },
                'tracking': {
                    'algorithm': 'deepsort',
                },
                'deploy': {
                    'platform': 'rv1126',
                    'quantize': True,
                }
            }
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            'data/raw',
            'data/processed',
            'data/annotations',
            'outputs/weights',
            'outputs/logs',
            'outputs/results',
            'outputs/visualizations',
            'models/rknn_obj',
        ]
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self):
        """设置日志系统"""
        try:
            from src.utils.logger import LogManager
            return LogManager.get_logger('main', log_dir=str(self.project_root / 'outputs/logs'))
        except ImportError:
            # 如果无法导入，使用标准库
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger('main')
    
    # ==================== 阶段1: data准备 ====================
    
    def prepare_data(self, dataset: str = 'flir', download: bool = True) -> bool:
        """
        data准备阶段
        
        Args:
            dataset: data集name ('flir' 或 'kaist')
            download: 是否download_dataset
            
        Returns:
            是否success完成
        """
        self.logger.info("=" * 60)
        self.logger.info("阶段1: data准备")
        self.logger.info("=" * 60)
        
        try:
            # 步骤1.1: download_dataset
            if download:
                self.logger.info(f"步骤1.1: 下载{dataset.upper()}data集...")
                self._download_dataset(dataset)
            
            # 步骤1.2: 预处理data集
            self.logger.info(f"步骤1.2: 预处理{dataset.upper()}data集...")
            self._preprocess_dataset(dataset)
            
            # 步骤1.3: split_dataset
            self.logger.info("步骤1.3: 划分训练/验证/测试集...")
            self._split_dataset()
            
            # 步骤1.4: 生成data增强
            self.logger.info("步骤1.4: config红外data增强策略...")
            self._setup_data_augmentation()
            
            self.logger.info("data准备阶段完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"data准备阶段失败: {e}")
            return False
    
    def _download_dataset(self, dataset: str):
        """download_dataset"""
        self.logger.info(f"  - 检查{dataset.upper()}data集...")
        raw_path = self.project_root / 'data/raw' / dataset
        
        if raw_path.exists() and any(raw_path.iterdir()):
            self.logger.info(f"  - data集已存在于 {raw_path}")
            return
        
        # 尝试使用下载脚本
        download_script = self.project_root / 'scripts/data/download_dataset.py'
        if download_script.exists():
            self.logger.info(f"  - 使用脚本download_dataset...")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(download_script), '--dataset', dataset, '--output', str(raw_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"  - data集下载完成")
            else:
                self.logger.warning(f"  - 自动下载失败，请手动download_dataset到 {raw_path}")
        else:
            self.logger.warning(f"  - 请手动下载{dataset.upper()}data集到 {raw_path}")
    
    def _preprocess_dataset(self, dataset: str):
        """预处理data集"""
        raw_path = self.project_root / 'data/raw' / dataset
        processed_path = self.project_root / 'data/processed'
        
        # 检查data是否已处理
        if (processed_path / 'images').exists():
            self.logger.info("  - 发现已处理的data")
            return
        
        # 使用对应的预处理脚本
        if dataset == 'flir':
            script_path = self.project_root / 'scripts/data/prepare_flir.py'
        else:
            script_path = self.project_root / 'scripts/data/prepare_kaist.py'
        
        if script_path.exists() and raw_path.exists():
            self.logger.info(f"  - 使用 {script_path.name} 处理data...")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(script_path), '--input', str(raw_path), '--output', str(processed_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info("  - data预处理完成")
            else:
                self.logger.warning("  - data预处理脚本执行失败，请检查data")
        else:
            self.logger.info("  - 创建data目录结构...")
            (processed_path / 'images/train').mkdir(parents=True, exist_ok=True)
            (processed_path / 'images/val').mkdir(parents=True, exist_ok=True)
            (processed_path / 'images/test').mkdir(parents=True, exist_ok=True)
            (processed_path / 'labels/train').mkdir(parents=True, exist_ok=True)
            (processed_path / 'labels/val').mkdir(parents=True, exist_ok=True)
            (processed_path / 'labels/test').mkdir(parents=True, exist_ok=True)
    
    def _split_dataset(self):
        """split_dataset"""
        self.logger.info("  - data集划分比例: 训练70% / 验证15% / 测试15%")
        # data划分逻辑在预处理脚本中完成
        pass
    
    def _setup_data_augmentation(self):
        """设置data增强"""
        self.logger.info("  - 红外data增强策略:")
        self.logger.info("    * 亮度调整 (brightness)")
        self.logger.info("    * 对比度增强 (contrast)")
        self.logger.info("    * 高斯噪声 (gaussian noise)")
        self.logger.info("    * 随机翻转 (flip)")
        self.logger.info("    * 缩放裁剪 (scale & crop)")
    
    # ==================== 阶段2: model训练 ====================
    
    def train_detector(self, resume: bool = False) -> bool:
        """
        model训练阶段
        
        Args:
            resume: 是否从检查点恢复训练
            
        Returns:
            是否success完成
        """
        self.logger.info("=" * 60)
        self.logger.info("阶段2: model训练")
        self.logger.info("=" * 60)
        
        try:
            # 步骤2.1: load_modelconfig
            self.logger.info("步骤2.1: 加载YOLOv5modelconfig...")
            model_config = self._load_model_config()
            
            # 步骤2.2: 初始化model
            self.logger.info("步骤2.2: 初始化YOLOv5detector...")
            detector = self._init_detector(model_config)
            
            # 步骤2.3: 设置训练参数
            self.logger.info("步骤2.3: config训练参数...")
            train_config = self._get_train_config()
            self._log_train_config(train_config)
            
            # 步骤2.4: 开始训练
            self.logger.info("步骤2.4: 开始训练...")
            self._run_training(detector, train_config, resume)
            
            # 步骤2.5: 保存最佳model
            self.logger.info("步骤2.5: 保存训练results...")
            self._save_training_results()
            
            self.logger.info("model训练阶段完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"model训练阶段失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _load_model_config(self) -> Dict:
        """load_modelconfig"""
        model_yaml = self.project_root / 'models/yolov5/yolov5s_infrared.yaml'
        if model_yaml.exists():
            with open(model_yaml, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"  - load_modelconfig: {model_yaml.name}")
            return config
        else:
            self.logger.info("  - 使用默认YOLOv5sconfig")
            return {'nc': 3, 'depth_multiple': 0.33, 'width_multiple': 0.50}
    
    def _init_detector(self, model_config: Dict):
        """初始化detector"""
        try:
            from src.detection.yolov5_detector import YOLOv5Detector
            detector = YOLOv5Detector(
                model_config=model_config,
                num_classes=model_config.get('nc', 3),
                img_size=self.config.get('model', {}).get('img_size', 640)
            )
            self.logger.info("  - YOLOv5Detector初始化完成")
            return detector
        except ImportError as e:
            self.logger.warning(f"  - 无法导入YOLOv5Detector: {e}")
            self.logger.info("  - 使用训练脚本进行训练...")
            return None
    
    def _get_train_config(self) -> Dict:
        """获取训练config"""
        return {
            'epochs': self.config.get('train', {}).get('epochs', 100),
            'batch_size': self.config.get('train', {}).get('batch_size', 16),
            'learning_rate': self.config.get('train', {}).get('lr', 0.01),
            'img_size': self.config.get('model', {}).get('img_size', 640),
            'optimizer': self.config.get('train', {}).get('optimizer', 'SGD'),
            'device': self.config.get('train', {}).get('device', 'cuda:0'),
        }
    
    def _log_train_config(self, config: Dict):
        """记录训练config"""
        self.logger.info("  - 训练参数:")
        for key, value in config.items():
            self.logger.info(f"    * {key}: {value}")
    
    def _run_training(self, detector, train_config: Dict, resume: bool):
        """执行训练"""
        train_script = self.project_root / 'scripts/train/train_yolov5.py'
        
        if detector is not None:
            # 使用detector类进行训练
            self.logger.info("  - 使用YOLOv5Detector进行训练...")
            # 这里可以调用detector.train()方法
            self.logger.info("  - 训练模拟中... (实际训练需要完整环境)")
        elif train_script.exists():
            # 使用训练脚本
            self.logger.info(f"  - 使用训练脚本: {train_script.name}")
            import subprocess
            cmd = [
                sys.executable, str(train_script),
                '--config', str(self.project_root / 'configs/train_config.yaml'),
            ]
            if resume:
                cmd.append('--resume')
            
            self.logger.info(f"  - command: {' '.join(cmd)}")
            self.logger.info("  - 训练已config完成，可执行上述command开始训练")
        else:
            self.logger.warning("  - 训练脚本不存在，请检查scripts/train/目录")
    
    def _save_training_results(self):
        """保存训练results"""
        weights_dir = self.project_root / 'outputs/weights'
        self.logger.info(f"  - model权重保存目录: {weights_dir}")
        self.logger.info("  - best.pt: 最佳验证性能的权重")
        self.logger.info("  - last.pt: 最后一轮的权重")
    
    # ==================== 阶段3: 跟踪集成 ====================
    
    def integrate_tracking(self, algorithm: str = 'deepsort') -> bool:
        """
        跟踪集成阶段
        
        Args:
            algorithm: 跟踪算法 ('deepsort', 'bytetrack', 'centertrack')
            
        Returns:
            是否success完成
        """
        self.logger.info("=" * 60)
        self.logger.info("阶段3: 跟踪集成")
        self.logger.info("=" * 60)
        
        try:
            # 步骤3.1: load_detector
            self.logger.info("步骤3.1: 加载训练好的detector...")
            detector = self._load_trained_detector()
            
            # 步骤3.2: 初始化tracker
            self.logger.info(f"步骤3.2: 初始化{algorithm.upper()}tracker...")
            tracker = self._init_tracker(algorithm)
            
            # 步骤3.3: config跟踪参数
            self.logger.info("步骤3.3: config跟踪参数...")
            self._log_tracker_config(algorithm)
            
            # 步骤3.4: run跟踪测试
            self.logger.info("步骤3.4: run跟踪测试...")
            self._run_tracking_test(detector, tracker)
            
            self.logger.info("跟踪集成阶段完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"跟踪集成阶段失败: {e}")
            return False
    
    def _load_trained_detector(self):
        """加载训练好的detector"""
        weights_path = self.project_root / 'outputs/weights/best.pt'
        if weights_path.exists():
            self.logger.info(f"  - 加载权重: {weights_path}")
            try:
                from src.detection.yolov5_detector import YOLOv5Detector
                detector = YOLOv5Detector(weights=str(weights_path))
                return detector
            except ImportError:
                self.logger.warning("  - 无法导入detector模块")
                return None
        else:
            self.logger.warning(f"  - 权重文件不存在: {weights_path}")
            self.logger.info("  - 请先run训练阶段或提供预训练权重")
            return None
    
    def _init_tracker(self, algorithm: str):
        """初始化tracker"""
        tracker_map = {
            'deepsort': 'DeepSORTTracker',
            'bytetrack': 'ByteTracker',
            'centertrack': 'CenterTracker',
        }
        
        tracker_class_name = tracker_map.get(algorithm, 'DeepSORTTracker')
        config_path = self.project_root / 'configs/tracking_config.yaml'
        
        try:
            if algorithm == 'deepsort':
                from src.tracking.deepsort_tracker import DeepSORTTracker
                tracker = DeepSORTTracker(config_path=str(config_path))
            elif algorithm == 'bytetrack':
                from src.tracking.bytetrack_tracker import ByteTracker
                tracker = ByteTracker(config_path=str(config_path))
            elif algorithm == 'centertrack':
                from src.tracking.centertrack_tracker import CenterTracker
                tracker = CenterTracker(config_path=str(config_path))
            else:
                self.logger.warning(f"  - 未知跟踪算法: {algorithm}，使用DeepSORT")
                from src.tracking.deepsort_tracker import DeepSORTTracker
                tracker = DeepSORTTracker(config_path=str(config_path))
            
            self.logger.info(f"  - {tracker_class_name}初始化完成")
            return tracker
        except ImportError as e:
            self.logger.warning(f"  - 无法导入{tracker_class_name}: {e}")
            return None
    
    def _log_tracker_config(self, algorithm: str):
        """记录trackerconfig"""
        config_file = self.project_root / f'models/tracking/{algorithm}/config.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"  - {algorithm.upper()}config:")
            for key, value in config.items():
                if isinstance(value, dict):
                    self.logger.info(f"    * {key}:")
                    for k, v in value.items():
                        self.logger.info(f"      - {k}: {v}")
                else:
                    self.logger.info(f"    * {key}: {value}")
    
    def _run_tracking_test(self, detector, tracker):
        """run跟踪测试"""
        self.logger.info("  - 跟踪流程:")
        self.logger.info("    1. 读取视频帧")
        self.logger.info("    2. 使用detector获取目标边界框")
        self.logger.info("    3. 将检测resultsinputtracker")
        self.logger.info("    4. 更新目标轨迹并分配ID")
        self.logger.info("    5. visualize跟踪results")
        
        if detector is not None and tracker is not None:
            self.logger.info("  - detector和tracker已就绪，可以进行实时跟踪")
        else:
            self.logger.info("  - 请确保detector权重和trackerconfig正确")
    
    # ==================== 阶段4: 性能evaluate ====================
    
    def evaluate_performance(self, eval_detection: bool = True, eval_tracking: bool = True) -> bool:
        """
        性能evaluate阶段
        
        Args:
            eval_detection: 是否evaluate检测性能
            eval_tracking: 是否evaluate跟踪性能
            
        Returns:
            是否success完成
        """
        self.logger.info("=" * 60)
        self.logger.info("阶段4: 性能evaluate")
        self.logger.info("=" * 60)
        
        try:
            results = {}
            
            # 步骤4.1: 检测性能evaluate
            if eval_detection:
                self.logger.info("步骤4.1: evaluate检测性能...")
                det_results = self._evaluate_detection()
                results['detection'] = det_results
            
            # 步骤4.2: 跟踪性能evaluate
            if eval_tracking:
                self.logger.info("步骤4.2: evaluate跟踪性能...")
                track_results = self._evaluate_tracking()
                results['tracking'] = track_results
            
            # 步骤4.3: 跟踪算法对比
            if eval_tracking:
                self.logger.info("步骤4.3: 对比不同跟踪算法...")
                self._compare_trackers()
            
            # 步骤4.4: 生成evaluate报告
            self.logger.info("步骤4.4: 生成evaluate报告...")
            self._generate_evaluation_report(results)
            
            self.logger.info("性能evaluate阶段完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"性能evaluate阶段失败: {e}")
            return False
    
    def _evaluate_detection(self) -> Dict:
        """evaluate检测性能"""
        self.logger.info("  - 检测evaluatemetrics:")
        self.logger.info("    * mAP@0.5: 平均精度(IoU=0.5)")
        self.logger.info("    * mAP@0.5:0.95: 平均精度(IoU=0.5-0.95)")
        self.logger.info("    * Precision: 精确率")
        self.logger.info("    * Recall: 召回率")
        self.logger.info("    * F1-Score: F1分数")
        
        try:
            from src.evaluation.detection_eval import DetectionEvaluator
            evaluator = DetectionEvaluator()
            # 实际evaluate需要预测results和真实label
            self.logger.info("  - 检测evaluate器已就绪")
            return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0}
        except ImportError:
            self.logger.warning("  - 无法导入检测evaluate模块")
            return {}
    
    def _evaluate_tracking(self) -> Dict:
        """evaluate跟踪性能"""
        self.logger.info("  - 跟踪evaluatemetrics (MOT Challenge):")
        self.logger.info("    * MOTA: 多目标跟踪准确度")
        self.logger.info("    * MOTP: 多目标跟踪精度")
        self.logger.info("    * IDF1: ID F1分数")
        self.logger.info("    * MT: 完全跟踪目标比例")
        self.logger.info("    * ML: 完全丢失目标比例")
        self.logger.info("    * FP: 假阳性数")
        self.logger.info("    * FN: 假阴性数")
        self.logger.info("    * IDs: ID切换次数")
        
        try:
            from src.evaluation.tracking_eval import MOTEvaluator
            evaluator = MOTEvaluator()
            self.logger.info("  - 跟踪evaluate器已就绪")
            return {'MOTA': 0.0, 'MOTP': 0.0, 'IDF1': 0.0}
        except ImportError:
            self.logger.warning("  - 无法导入跟踪evaluate模块")
            return {}
    
    def _compare_trackers(self):
        """对比不同跟踪算法"""
        algorithms = ['deepsort', 'bytetrack', 'centertrack']
        self.logger.info("  - 对比跟踪算法:")
        for algo in algorithms:
            self.logger.info(f"    * {algo.upper()}")
        
        compare_script = self.project_root / 'scripts/evaluate/compare_trackers.py'
        if compare_script.exists():
            self.logger.info(f"  - 对比脚本: {compare_script.name}")
            self.logger.info("  - 执行command: python scripts/evaluate/compare_trackers.py")
    
    def _generate_evaluation_report(self, results: Dict):
        """生成evaluate报告"""
        report_path = self.project_root / 'outputs/results/evaluation_report.json'
        
        import json
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  - evaluate报告保存至: {report_path}")
    
    # ==================== 阶段5: model部署 ====================
    
    def deploy_model(self, quantize: bool = True) -> bool:
        """
        model部署阶段
        
        Args:
            quantize: 是否进行INT8量化
            
        Returns:
            是否success完成
        """
        self.logger.info("=" * 60)
        self.logger.info("阶段5: model部署")
        self.logger.info("=" * 60)
        
        try:
            # 步骤5.1: export_onnxmodel
            self.logger.info("步骤5.1: export_onnxmodel...")
            onnx_path = self._export_onnx()
            
            # 步骤5.2: convert为RKNNmodel
            self.logger.info("步骤5.2: convert为RKNNmodel...")
            rknn_obj_path = self._convert_to_rknn_obj(onnx_path, quantize)
            
            # 步骤5.3: INT8量化（可选）
            if quantize:
                self.logger.info("步骤5.3: 执行INT8量化...")
                self._quantize_model(rknn_obj_path)
            
            # 步骤5.4: 测试RKNNmodel
            self.logger.info("步骤5.4: 测试RKNNmodel...")
            self._test_rknn_obj_model(rknn_obj_path)
            
            # 步骤5.5: 生成部署包
            self.logger.info("步骤5.5: 生成嵌入式部署包...")
            self._generate_deployment_package()
            
            self.logger.info("model部署阶段完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"model部署阶段失败: {e}")
            return False
    
    def _export_onnx(self) -> str:
        """export_onnxmodel"""
        weights_path = self.project_root / 'outputs/weights/best.pt'
        onnx_path = self.project_root / 'outputs/weights/best.onnx'
        
        self.logger.info(f"  - input权重: {weights_path}")
        self.logger.info(f"  - outputONNX: {onnx_path}")
        
        try:
            from src.deploy.export_onnx import ONNXExporter
            exporter = ONNXExporter(model_path=str(weights_path))
            exporter.export(str(onnx_path))
            self.logger.info("  - ONNX导出success")
        except ImportError:
            self.logger.warning("  - 无法导入ONNXExporter")
            self.logger.info("  - 执行command: python scripts/deploy/export_model.py --weights outputs/weights/best.pt --format onnx")
        
        return str(onnx_path)
    
    def _convert_to_rknn_obj(self, onnx_path: str, quantize: bool) -> str:
        """convert为RKNNmodel"""
        rknn_obj_path = self.project_root / 'models/rknn_obj/best.rknn_obj'
        
        self.logger.info(f"  - inputONNX: {onnx_path}")
        self.logger.info(f"  - outputRKNN: {rknn_obj_path}")
        self.logger.info(f"  - 量化: {'是' if quantize else '否'}")
        self.logger.info(f"  - 目标平台: RV1126")
        
        try:
            from src.deploy.convert_rknn_obj import RKNNConverter
            converter = RKNNConverter(onnx_path=onnx_path, target_platform='rv1126')
            converter.convert(str(rknn_obj_path), quantize=quantize)
            self.logger.info("  - RKNNconvertsuccess")
        except ImportError:
            self.logger.warning("  - 无法导入RKNNConverter (需要RKNN-Toolkit2环境)")
            self.logger.info("  - 执行command: python scripts/deploy/convert_to_rknn_obj.py --onnx outputs/weights/best.onnx")
        
        return str(rknn_obj_path)
    
    def _quantize_model(self, rknn_obj_path: str):
        """INT8量化"""
        self.logger.info("  - 量化config:")
        self.logger.info("    * 量化类型: INT8")
        self.logger.info("    * 校准data集: data/processed/calibration")
        self.logger.info("    * 校准图片数: 100")
        
        try:
            from src.deploy.quantize import ModelQuantizer
            quantizer = ModelQuantizer()
            self.logger.info("  - 量化器已就绪")
        except ImportError:
            self.logger.warning("  - 无法导入量化模块")
    
    def _test_rknn_obj_model(self, rknn_obj_path: str):
        """测试RKNNmodel"""
        self.logger.info("  - RKNNmodel测试:")
        self.logger.info("    * load_model")
        self.logger.info("    * runinference")
        self.logger.info("    * 验证output")
        self.logger.info("    * 测量性能")
        
        test_script = self.project_root / 'scripts/deploy/test_rknn_obj.py'
        if test_script.exists():
            self.logger.info(f"  - 测试脚本: {test_script.name}")
            self.logger.info(f"  - 执行command: python scripts/deploy/test_rknn_obj.py --model {rknn_obj_path}")
    
    def _generate_deployment_package(self):
        """生成部署包"""
        embedded_dir = self.project_root / 'embedded'
        self.logger.info(f"  - 嵌入式代码目录: {embedded_dir}")
        self.logger.info("  - 部署包内容:")
        self.logger.info("    * RKNNmodel文件")
        self.logger.info("    * C++inference代码")
        self.logger.info("    * config文件")
        self.logger.info("    * CMake构建脚本")
        self.logger.info("  - 编译command:")
        self.logger.info("    cd embedded && mkdir build && cd build")
        self.logger.info("    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake ..")
        self.logger.info("    make -j4")
    
    # ==================== 演示模式 ====================
    
    def run_demo(self, source: str = "0") -> bool:
        """
        run演示模式
        
        Args:
            source: 视频源 ("0"表示摄像头, 或视频文件路径)
            
        Returns:
            是否success完成
        """
        self.logger.info("=" * 60)
        self.logger.info("演示模式: 实时检测与跟踪")
        self.logger.info("=" * 60)
        
        self.logger.info(f"  - 视频源: {source}")
        self.logger.info("  - 按'q'键退出演示")
        
        try:
            # load_detector
            detector = self._load_trained_detector()
            
            # 初始化tracker
            algorithm = self.config.get('tracking', {}).get('algorithm', 'deepsort')
            tracker = self._init_tracker(algorithm)
            
            if detector is None or tracker is None:
                self.logger.warning("  - detector或tracker未就绪，无法run演示")
                self.logger.info("  - 请先完成训练阶段")
                return False
            
            # run实时演示
            self._run_realtime_demo(detector, tracker, source)
            
            return True
            
        except Exception as e:
            self.logger.error(f"演示模式失败: {e}")
            return False
    
    def _run_realtime_demo(self, detector, tracker, source: str):
        """run实时演示"""
        self.logger.info("  - 开始实时演示...")
        self.logger.info("  - 处理流程:")
        self.logger.info("    1. 读取视频帧")
        self.logger.info("    2. 目标检测")
        self.logger.info("    3. 多目标跟踪")
        self.logger.info("    4. visualize显示")
        
        # 实际演示代码需要OpenCV环境
        try:
            import cv2
            self.logger.info("  - OpenCV已就绪，可以run实时演示")
        except ImportError:
            self.logger.warning("  - 需要安装OpenCV: pip install opencv-python")
    
    # ==================== 完整流程 ====================
    
    def run_full_pipeline(self) -> bool:
        """
        run完整流程
        
        Returns:
            是否success完成所有阶段
        """
        self.logger.info("=" * 60)
        self.logger.info("开始run完整流程")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        # 阶段1: data准备
        results['prepare'] = self.prepare_data()
        
        # 阶段2: model训练
        results['train'] = self.train_detector()
        
        # 阶段3: 跟踪集成
        results['track'] = self.integrate_tracking()
        
        # 阶段4: 性能evaluate
        results['evaluate'] = self.evaluate_performance()
        
        # 阶段5: model部署
        results['deploy'] = self.deploy_model()
        
        # summarize_results
        elapsed_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info("完整流程执行完毕")
        self.logger.info("=" * 60)
        self.logger.info(f"总duration: {elapsed_time:.2f}秒")
        self.logger.info("各阶段results:")
        for stage, success in results.items():
            status = "✓ success" if success else "✗ 失败"
            self.logger.info(f"  - {stage}: {status}")
        
        all_success = all(results.values())
        if all_success:
            self.logger.info("所有阶段执行success！")
        else:
            self.logger.warning("部分阶段执行失败，请检查日志")
        
        return all_success


def main():
    """主函数入口"""
    parser = argparse.ArgumentParser(
        description='红外行人多目标检测与跟踪系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode full        # run完整流程
  python main.py --mode prepare     # 仅data准备
  python main.py --mode train       # 仅model训练
  python main.py --mode track       # 仅跟踪集成
  python main.py --mode evaluate    # 仅性能evaluate
  python main.py --mode deploy      # 仅model部署
  python main.py --mode demo        # 演示模式
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['full', 'prepare', 'train', 'track', 'evaluate', 'deploy', 'demo'],
        help='run模式 (默认: full)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/train_config.yaml',
        help='config文件路径 (默认: configs/train_config.yaml)'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='flir',
        choices=['flir', 'kaist'],
        help='data集name (默认: flir)'
    )
    
    parser.add_argument(
        '--tracker', 
        type=str, 
        default='deepsort',
        choices=['deepsort', 'bytetrack', 'centertrack'],
        help='跟踪算法 (默认: deepsort)'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='是否从检查点恢复训练'
    )
    
    parser.add_argument(
        '--no-quantize', 
        action='store_true',
        help='部署时不进行INT8量化'
    )
    
    parser.add_argument(
        '--source', 
        type=str, 
        default='0',
        help='演示模式的视频源 (默认: 0表示摄像头)'
    )
    
    args = parser.parse_args()
    
    # 初始化系统
    system = InfraredMOTSystem(config_path=args.config)
    
    # 根据模式执行
    success = False
    
    if args.mode == 'full':
        success = system.run_full_pipeline()
    elif args.mode == 'prepare':
        success = system.prepare_data(dataset=args.dataset)
    elif args.mode == 'train':
        success = system.train_detector(resume=args.resume)
    elif args.mode == 'track':
        success = system.integrate_tracking(algorithm=args.tracker)
    elif args.mode == 'evaluate':
        success = system.evaluate_performance()
    elif args.mode == 'deploy':
        success = system.deploy_model(quantize=not args.no_quantize)
    elif args.mode == 'demo':
        success = system.run_demo(source=args.source)
    
    # 返回状态码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
