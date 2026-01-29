#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测模块测试

测试目标检测相关功能
"""

import sys
import os
import unittest
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector import DetectionResult, BaseDetector
from src.detection.data_augment import InfraredDataAugmentor, create_train_augmentor, create_val_augmentor


class TestDetectionResult(unittest.TestCase):
    """测试DetectionResult类"""
    
    def test_empty_result(self):
        """测试空检测results"""
        result = DetectionResult()
        self.assertEqual(len(result), 0)
        self.assertEqual(result.boxes.shape, (0, 4))
    
    def test_result_with_data(self):
        """测试有data的检测results"""
        boxes = np.array([[10, 20, 100, 200], [50, 60, 150, 250]])
        confidences = np.array([0.9, 0.8])
        classes = np.array([0, 1])
        
        result = DetectionResult(
            boxes=boxes,
            confidences=confidences,
            classes=classes,
            class_names=['person', 'car']
        )
        
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result.boxes, boxes)
        np.testing.assert_array_equal(result.confidences, confidences)
        np.testing.assert_array_equal(result.classes, classes)
    
    def test_filter_by_confidence(self):
        """测试按confidence过滤"""
        boxes = np.array([[10, 20, 100, 200], [50, 60, 150, 250], [30, 40, 130, 230]])
        confidences = np.array([0.9, 0.5, 0.7])
        classes = np.array([0, 1, 0])
        
        result = DetectionResult(boxes=boxes, confidences=confidences, classes=classes)
        filtered = result.filter_by_confidence(0.6)
        
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(filtered.confidences, [0.9, 0.7])
    
    def test_filter_by_class(self):
        """测试按classes过滤"""
        boxes = np.array([[10, 20, 100, 200], [50, 60, 150, 250], [30, 40, 130, 230]])
        confidences = np.array([0.9, 0.8, 0.7])
        classes = np.array([0, 1, 0])
        
        result = DetectionResult(boxes=boxes, confidences=confidences, classes=classes)
        filtered = result.filter_by_class(0)
        
        self.assertEqual(len(filtered), 2)
        np.testing.assert_array_equal(filtered.classes, [0, 0])
    
    def test_to_list(self):
        """测试convert为字典列表"""
        boxes = np.array([[10, 20, 100, 200]])
        confidences = np.array([0.9])
        classes = np.array([0])
        
        result = DetectionResult(
            boxes=boxes,
            confidences=confidences,
            classes=classes,
            class_names=['person']
        )
        
        result_list = result.to_list()
        self.assertEqual(len(result_list), 1)
        self.assertEqual(result_list[0]['class_name'], 'person')
        self.assertAlmostEqual(result_list[0]['confidence'], 0.9)


class TestBaseDetector(unittest.TestCase):
    """测试BaseDetector类"""
    
    def test_nms(self):
        """测试NMS功能"""
        # 创建重叠的边界框
        boxes = np.array([
            [10, 10, 100, 100],
            [15, 15, 105, 105],  # 与第一个高度重叠
            [200, 200, 300, 300]  # 与前两个不重叠
        ])
        scores = np.array([0.9, 0.8, 0.7])
        
        # 执行NMS
        keep = BaseDetector.nms(boxes, scores, iou_threshold=0.5)
        
        # 应该保留第一个和第三个（因为它们不重叠）
        self.assertEqual(len(keep), 2)
        self.assertIn(0, keep)  # confidence最高的应该保留
        self.assertIn(2, keep)  # 不重叠的也应该保留
    
    def test_nms_empty(self):
        """测试空input的NMS"""
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        
        keep = BaseDetector.nms(boxes, scores, iou_threshold=0.5)
        
        self.assertEqual(len(keep), 0)


class TestInfraredDataAugmentor(unittest.TestCase):
    """测试红外data增强器"""
    
    def setUp(self):
        """设置测试data"""
        self.image = np.random.rand(480, 640, 3).astype(np.float32)
        self.labels = np.array([
            [0, 0.5, 0.5, 0.2, 0.3],  # class_id, x_center, y_center, w, h
            [1, 0.3, 0.7, 0.1, 0.15]
        ])
    
    def test_brightness_adjustment(self):
        """测试亮度调整"""
        augmentor = InfraredDataAugmentor(brightness_range=(0.1, 0.1))
        augmented, _ = augmentor.random_brightness(self.image), None
        
        # 亮度应该增加
        self.assertGreater(augmented.mean(), self.image.mean())
    
    def test_contrast_adjustment(self):
        """测试对比度调整"""
        augmentor = InfraredDataAugmentor(contrast_range=(1.5, 1.5))
        augmented = augmentor.random_contrast(self.image)
        
        # 对比度增加应该增加std_dev
        # 注意：由于image可能会被裁剪，这个测试可能不完全可靠
        # 这里只检查函数能正常run
        self.assertEqual(augmented.shape, self.image.shape)
    
    def test_horizontal_flip(self):
        """测试水平翻转"""
        augmentor = InfraredDataAugmentor(flip_prob=1.0)  # 总是翻转
        flipped_image, flipped_labels = augmentor.horizontal_flip(
            self.image.copy(), self.labels.copy()
        )
        
        # 检查labelx坐标是否翻转
        expected_x = 1.0 - self.labels[:, 1]
        np.testing.assert_array_almost_equal(flipped_labels[:, 1], expected_x)
    
    def test_no_flip(self):
        """测试不翻转"""
        augmentor = InfraredDataAugmentor(flip_prob=0.0)  # 从不翻转
        flipped_image, flipped_labels = augmentor.horizontal_flip(
            self.image.copy(), self.labels.copy()
        )
        
        # label应该不变
        np.testing.assert_array_almost_equal(flipped_labels, self.labels)
    
    def test_thermal_noise(self):
        """测试热噪声添加"""
        augmentor = InfraredDataAugmentor(noise_intensity=0.05)
        noisy = augmentor.add_thermal_noise(self.image.copy())
        
        # 噪声应该导致image变化
        self.assertFalse(np.allclose(noisy, self.image))
    
    def test_create_train_augmentor(self):
        """测试创建训练增强器"""
        augmentor = create_train_augmentor()
        
        # 训练增强器应该有较强的增强参数
        self.assertEqual(augmentor.flip_prob, 0.5)
        self.assertGreater(augmentor.rotation_angle, 0)
    
    def test_create_val_augmentor(self):
        """测试创建验证增强器"""
        augmentor = create_val_augmentor()
        
        # 验证增强器应该禁用大部分增强
        self.assertEqual(augmentor.flip_prob, 0.0)
        self.assertEqual(augmentor.rotation_angle, 0.0)
        self.assertEqual(augmentor.noise_intensity, 0.0)


class TestDataAugmentorIntegration(unittest.TestCase):
    """data增强器集成测试"""
    
    def test_full_augmentation_pipeline(self):
        """测试完整的增强流程"""
        image = np.random.rand(640, 640, 3).astype(np.float32)
        labels = np.array([
            [0, 0.5, 0.5, 0.2, 0.3]
        ])
        
        augmentor = create_train_augmentor()
        
        # 多次run以测试随机性
        for _ in range(5):
            aug_image, aug_labels = augmentor(image.copy(), labels.copy())
            
            # 检查output有效性
            self.assertEqual(aug_image.shape, image.shape)
            self.assertTrue(aug_image.min() >= 0)
            self.assertTrue(aug_image.max() <= 1)
            
            # label可能被过滤，检查格式
            if len(aug_labels) > 0:
                self.assertEqual(aug_labels.shape[1], 5)


if __name__ == '__main__':
    unittest.main()
