#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块测试

测试工具函数和类
"""

import sys
import os
import unittest
import tempfile
import json
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.metrics import (
    compute_iou, compute_batch_iou, compute_precision_recall,
    compute_ap, compute_map, MOTMetricsCalculator,
    save_metrics_to_json, load_metrics_from_json
)
from src.utils.logger import LogManager, TrainingLogger, ProgressBar, init_logger, get_logger


class TestIoUComputation(unittest.TestCase):
    """测试IoU计算"""
    
    def test_compute_iou_overlap(self):
        """测试有重叠的IoU计算"""
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        
        iou = compute_iou(box1, box2)
        
        # 交集 = 50 * 50 = 2500
        # 并集 = 10000 + 10000 - 2500 = 17500
        expected_iou = 2500 / 17500
        self.assertAlmostEqual(iou, expected_iou, places=4)
    
    def test_compute_iou_no_overlap(self):
        """测试无重叠的IoU计算"""
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])
        
        iou = compute_iou(box1, box2)
        
        self.assertEqual(iou, 0.0)
    
    def test_compute_iou_same_box(self):
        """测试相同框的IoU计算"""
        box = np.array([10, 20, 100, 200])
        
        iou = compute_iou(box, box)
        
        self.assertAlmostEqual(iou, 1.0)
    
    def test_compute_batch_iou(self):
        """测试批量IoU计算"""
        boxes1 = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ])
        boxes2 = np.array([
            [50, 50, 150, 150],
            [250, 250, 350, 350]
        ])
        
        iou_matrix = compute_batch_iou(boxes1, boxes2)
        
        self.assertEqual(iou_matrix.shape, (2, 2))
        self.assertGreater(iou_matrix[0, 0], 0)  # 有重叠
        self.assertGreater(iou_matrix[1, 1], 0)  # 有重叠
    
    def test_compute_batch_iou_empty(self):
        """测试空输入的批量IoU"""
        boxes1 = np.array([]).reshape(0, 4)
        boxes2 = np.array([[0, 0, 100, 100]])
        
        iou_matrix = compute_batch_iou(boxes1, boxes2)
        
        self.assertEqual(iou_matrix.shape, (0, 1))


class TestPrecisionRecall(unittest.TestCase):
    """测试精确率和召回率计算"""
    
    def test_compute_precision_recall(self):
        """测试精确率召回率计算"""
        # 假设按置信度降序排列的检测结果
        tp = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])  # 5个TP
        fp = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1])  # 5个FP
        num_gt = 7  # 7个真实目标
        
        precision, recall = compute_precision_recall(tp, fp, num_gt)
        
        # 检查长度
        self.assertEqual(len(precision), len(tp))
        self.assertEqual(len(recall), len(tp))
        
        # 最终召回率应该是 5/7
        self.assertAlmostEqual(recall[-1], 5/7, places=4)
    
    def test_compute_ap(self):
        """测试AP计算"""
        precision = np.array([1.0, 1.0, 0.67, 0.75, 0.6, 0.67])
        recall = np.array([0.14, 0.29, 0.29, 0.43, 0.43, 0.57])
        
        ap = compute_ap(precision, recall)
        
        # AP应该在0到1之间
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)
    
    def test_compute_ap_empty(self):
        """测试空输入的AP计算"""
        precision = np.array([])
        recall = np.array([])
        
        ap = compute_ap(precision, recall)
        
        self.assertEqual(ap, 0.0)


class TestMAPComputation(unittest.TestCase):
    """测试mAP计算"""
    
    def test_compute_map_simple(self):
        """测试简单的mAP计算"""
        # 检测结果
        all_detections = {
            0: [
                {'bbox': [10, 10, 50, 50], 'class_id': 0, 'confidence': 0.9},
                {'bbox': [60, 60, 100, 100], 'class_id': 0, 'confidence': 0.8}
            ]
        }
        
        # 真实标注
        all_ground_truths = {
            0: [
                {'bbox': [10, 10, 50, 50], 'class_id': 0},
                {'bbox': [60, 60, 100, 100], 'class_id': 0}
            ]
        }
        
        mAP, ap_per_class = compute_map(all_detections, all_ground_truths, iou_threshold=0.5)
        
        # 完美匹配时mAP应该接近1.0
        self.assertGreater(mAP, 0.9)


class TestMOTMetricsCalculator(unittest.TestCase):
    """测试MOT指标计算器"""
    
    def setUp(self):
        """设置测试数据"""
        self.calculator = MOTMetricsCalculator(iou_threshold=0.5)
    
    def test_perfect_tracking(self):
        """测试完美跟踪的情况"""
        # 完美匹配
        gt_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        gt_ids = np.array([1, 2])
        pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        pred_ids = np.array([1, 2])
        
        self.calculator.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        metrics = self.calculator.compute_metrics()
        
        # MOTA应该接近1.0
        self.assertGreater(metrics['MOTA'], 0.9)
    
    def test_empty_frame(self):
        """测试空帧"""
        self.calculator.update(
            np.array([]).reshape(0, 4),
            np.array([]),
            np.array([]).reshape(0, 4),
            np.array([])
        )
        
        metrics = self.calculator.compute_metrics()
        
        self.assertEqual(metrics['num_gt'], 0)
        self.assertEqual(metrics['num_pred'], 0)
    
    def test_missed_detections(self):
        """测试漏检情况"""
        gt_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        gt_ids = np.array([1, 2])
        pred_boxes = np.array([[10, 10, 50, 50]])  # 只检测到一个
        pred_ids = np.array([1])
        
        self.calculator.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        metrics = self.calculator.compute_metrics()
        
        self.assertEqual(metrics['num_misses'], 1)
    
    def test_false_positives(self):
        """测试误检情况"""
        gt_boxes = np.array([[10, 10, 50, 50]])
        gt_ids = np.array([1])
        pred_boxes = np.array([[10, 10, 50, 50], [200, 200, 250, 250]])  # 多检测一个
        pred_ids = np.array([1, 2])
        
        self.calculator.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        metrics = self.calculator.compute_metrics()
        
        self.assertEqual(metrics['num_false_positives'], 1)
    
    def test_reset(self):
        """测试重置功能"""
        self.calculator.update(
            np.array([[10, 10, 50, 50]]),
            np.array([1]),
            np.array([[10, 10, 50, 50]]),
            np.array([1])
        )
        
        self.calculator.reset()
        
        self.assertEqual(self.calculator.num_gt, 0)
        self.assertEqual(self.calculator.frame_count, 0)


class TestMetricsSerialization(unittest.TestCase):
    """测试指标序列化"""
    
    def test_save_and_load_metrics(self):
        """测试保存和加载指标"""
        metrics = {
            'mAP': 0.85,
            'precision': 0.9,
            'recall': 0.8,
            'array_value': np.array([1, 2, 3])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            save_metrics_to_json(metrics, filepath)
            loaded = load_metrics_from_json(filepath)
            
            self.assertAlmostEqual(loaded['mAP'], 0.85)
            self.assertAlmostEqual(loaded['precision'], 0.9)
            self.assertEqual(loaded['array_value'], [1, 2, 3])  # 数组应该被转换为列表
        finally:
            os.unlink(filepath)


class TestLogManager(unittest.TestCase):
    """测试日志管理器"""
    
    def setUp(self):
        """重置日志管理器"""
        LogManager.reset()
    
    def test_singleton(self):
        """测试单例模式"""
        logger1 = LogManager()
        logger2 = LogManager()
        
        self.assertIs(logger1, logger2)
    
    def test_logging_levels(self):
        """测试不同日志级别"""
        logger = LogManager(console_output=False, file_output=False)
        
        # 这些方法应该不崩溃
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


class TestTrainingLogger(unittest.TestCase):
    """测试训练日志记录器"""
    
    def test_epoch_logging(self):
        """测试epoch日志记录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir, 'test_experiment')
            
            # 模拟训练过程
            logger.start_epoch(1, learning_rate=0.001)
            logger.log_loss('total_loss', 1.5)
            logger.log_loss('total_loss', 1.3)
            logger.log_metric('mAP', 0.5)
            logger.end_epoch()
            
            # 检查历史记录
            self.assertEqual(len(logger.history['epochs']), 1)
            self.assertEqual(logger.history['losses']['total_loss'][-1], (1.5 + 1.3) / 2)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir, 'test_experiment')
            
            logger.start_epoch(1)
            logger.log_loss('loss', 1.0)
            logger.end_epoch()
            
            filepath = os.path.join(tmpdir, 'test_history.json')
            logger.save(filepath)
            
            # 加载
            new_logger = TrainingLogger(tmpdir, 'new_experiment')
            new_logger.load(filepath)
            
            self.assertEqual(new_logger.experiment_name, 'test_experiment')


class TestProgressBar(unittest.TestCase):
    """测试进度条"""
    
    def test_progress_update(self):
        """测试进度更新"""
        pbar = ProgressBar(total=100, prefix='Test')
        
        for i in range(10):
            pbar.update()
        
        self.assertEqual(pbar.current, 10)
    
    def test_progress_finish(self):
        """测试进度完成"""
        pbar = ProgressBar(total=10)
        
        for i in range(10):
            pbar.update()
        
        pbar.finish("Done!")
        
        # 不应该崩溃


if __name__ == '__main__':
    unittest.main()
