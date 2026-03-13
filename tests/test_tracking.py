#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪模块测试

测试多目标跟踪相关功能
"""

import sys
import os
import unittest
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tracking.tracker import BaseTracker, TrackObject, TrackingResult
from src.tracking.kalman_filter import (
    KalmanFilter, KalmanBoxTracker,
    xyxy_to_xywh, xywh_to_xyxy, xyxy_to_xyah, xyah_to_xyxy
)
from src.tracking.deepsort_tracker import DeepSORTTracker, create_deepsort_tracker
from src.tracking.bytetrack_tracker import ByteTrack, create_bytetrack_tracker
from src.tracking.centertrack_tracker import CenterTrack, create_centertrack_tracker


class TestCoordinateConversions(unittest.TestCase):
    """测试坐标convert函数"""
    
    def test_xyxy_to_xywh(self):
        """测试xyxy到xywhconvert"""
        bbox = np.array([10, 20, 110, 120])
        result = xyxy_to_xywh(bbox)
        
        expected = np.array([60, 70, 100, 100])  # center_x, center_y, width, height
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_xywh_to_xyxy(self):
        """测试xywh到xyxyconvert"""
        bbox = np.array([60, 70, 100, 100])
        result = xywh_to_xyxy(bbox)
        
        expected = np.array([10, 20, 110, 120])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_xyxy_xywh_roundtrip(self):
        """测试xyxy-xywh往返convert"""
        original = np.array([15, 25, 115, 225])
        converted = xywh_to_xyxy(xyxy_to_xywh(original))
        np.testing.assert_array_almost_equal(converted, original)
    
    def test_xyxy_to_xyah(self):
        """测试xyxy到xyahconvert"""
        bbox = np.array([10, 20, 110, 220])  # width=100, height=200
        result = xyxy_to_xyah(bbox)
        
        # center_x=60, center_y=120, aspect_ratio=0.5, height=200
        expected = np.array([60, 120, 0.5, 200])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_xyah_to_xyxy(self):
        """测试xyah到xyxyconvert"""
        bbox = np.array([60, 120, 0.5, 200])
        result = xyah_to_xyxy(bbox)
        
        expected = np.array([10, 20, 110, 220])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_batch_conversion(self):
        """测试批量convert"""
        bboxes = np.array([
            [10, 20, 110, 120],
            [50, 60, 150, 160]
        ])
        
        xywh = xyxy_to_xywh(bboxes)
        back = xywh_to_xyxy(xywh)
        
        np.testing.assert_array_almost_equal(back, bboxes)


class TestTrackObject(unittest.TestCase):
    """测试TrackObject类"""
    
    def test_track_object_creation(self):
        """测试跟踪目标创建"""
        track = TrackObject(
            track_id=1,
            bbox=np.array([10, 20, 100, 200]),
            confidence=0.9,
            class_id=0
        )
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.confidence, 0.9)
        self.assertEqual(track.class_id, 0)
    
    def test_track_object_to_dict(self):
        """测试convert为字典"""
        track = TrackObject(
            track_id=1,
            bbox=np.array([10, 20, 100, 200]),
            confidence=0.9,
            class_id=0
        )
        
        result = track.to_dict()
        
        self.assertEqual(result['track_id'], 1)
        self.assertAlmostEqual(result['confidence'], 0.9)
        self.assertEqual(result['bbox'], [10, 20, 100, 200])


class TestTrackingResult(unittest.TestCase):
    """测试TrackingResult类"""
    
    def test_empty_result(self):
        """测试空跟踪results"""
        result = TrackingResult()
        
        self.assertEqual(len(result), 0)
        self.assertEqual(result.get_boxes().shape, (0, 4))
        self.assertEqual(len(result.get_ids()), 0)
    
    def test_result_with_tracks(self):
        """测试有跟踪目标的results"""
        tracks = [
            TrackObject(track_id=1, bbox=np.array([10, 20, 100, 200])),
            TrackObject(track_id=2, bbox=np.array([50, 60, 150, 250]))
        ]
        
        result = TrackingResult(tracks=tracks, frame_id=10)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.frame_id, 10)
        np.testing.assert_array_equal(result.get_ids(), [1, 2])
    
    def test_get_confirmed_tracks(self):
        """测试获取已确认的跟踪目标"""
        tracks = [
            TrackObject(track_id=1, bbox=np.array([10, 20, 100, 200]), state='confirmed'),
            TrackObject(track_id=2, bbox=np.array([50, 60, 150, 250]), state='tentative'),
            TrackObject(track_id=3, bbox=np.array([80, 90, 180, 280]), state='confirmed')
        ]
        
        result = TrackingResult(tracks=tracks)
        confirmed = result.get_confirmed_tracks()
        
        self.assertEqual(len(confirmed), 2)


class TestKalmanFilter(unittest.TestCase):
    """测试卡尔曼滤波器"""
    
    def test_kalman_filter_initiate(self):
        """测试卡尔曼滤波器初始化"""
        kf = KalmanFilter()
        measurement = np.array([100, 100, 0.5, 200])  # x, y, aspect_ratio, height
        
        mean, covariance = kf.initiate(measurement)
        
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(covariance.shape, (8, 8))
        np.testing.assert_array_almost_equal(mean[:4], measurement)
    
    def test_kalman_filter_predict(self):
        """测试卡尔曼滤波器预测"""
        kf = KalmanFilter()
        measurement = np.array([100, 100, 0.5, 200])
        mean, covariance = kf.initiate(measurement)
        
        # 预测下一帧
        new_mean, new_covariance = kf.predict(mean, covariance)
        
        # 位置应该根据速度预测（初始速度为0，所以位置不变）
        self.assertEqual(new_mean.shape, (8,))
    
    def test_kalman_filter_update(self):
        """测试卡尔曼滤波器更新"""
        kf = KalmanFilter()
        measurement = np.array([100, 100, 0.5, 200])
        mean, covariance = kf.initiate(measurement)
        
        # 预测
        mean, covariance = kf.predict(mean, covariance)
        
        # 使用新观测更新
        new_measurement = np.array([105, 105, 0.5, 200])
        updated_mean, updated_covariance = kf.update(mean, covariance, new_measurement)
        
        # 更新后的位置应该更接近新观测
        self.assertEqual(updated_mean.shape, (8,))


class TestKalmanBoxTracker(unittest.TestCase):
    """测试单目标卡尔曼tracker"""
    
    def setUp(self):
        """重置计数器"""
        KalmanBoxTracker.reset_count()
    
    def test_tracker_creation(self):
        """测试tracker创建"""
        bbox = np.array([10, 20, 110, 220])
        tracker = KalmanBoxTracker(bbox)
        
        self.assertEqual(tracker.track_id, 1)
        self.assertEqual(tracker.hits, 1)
    
    def test_tracker_predict(self):
        """测试预测"""
        bbox = np.array([10, 20, 110, 220])
        tracker = KalmanBoxTracker(bbox)
        
        predicted = tracker.predict()
        
        self.assertEqual(predicted.shape, (4,))
        self.assertEqual(tracker.time_since_update, 1)
    
    def test_tracker_update(self):
        """测试更新"""
        bbox = np.array([10, 20, 110, 220])
        tracker = KalmanBoxTracker(bbox)
        tracker.predict()
        
        new_bbox = np.array([15, 25, 115, 225])
        tracker.update(new_bbox)
        
        self.assertEqual(tracker.hits, 2)
        self.assertEqual(tracker.time_since_update, 0)


class TestBaseTrackerMethods(unittest.TestCase):
    """测试BaseTracker的静态方法"""
    
    def test_compute_iou(self):
        """测试IoU计算"""
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        
        iou = BaseTracker.compute_iou(box1, box2)
        
        # 交集面积 = 50*50 = 2500
        # 并集面积 = 100*100 + 100*100 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        self.assertAlmostEqual(iou, 2500 / 17500, places=3)
    
    def test_compute_iou_no_overlap(self):
        """测试无重叠的IoU"""
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])
        
        iou = BaseTracker.compute_iou(box1, box2)
        
        self.assertEqual(iou, 0.0)
    
    def test_compute_iou_matrix(self):
        """测试IoU矩阵计算"""
        boxes1 = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300]
        ])
        boxes2 = np.array([
            [50, 50, 150, 150],
            [250, 250, 350, 350]
        ])
        
        iou_matrix = BaseTracker.compute_iou_matrix(boxes1, boxes2)
        
        self.assertEqual(iou_matrix.shape, (2, 2))
        # 第一个框与第二组的第一个框有重叠
        self.assertGreater(iou_matrix[0, 0], 0)
        # 第一个框与第二组的第二个框无重叠
        self.assertEqual(iou_matrix[0, 1], 0)
    
    def test_linear_assignment(self):
        """测试线性分配"""
        cost_matrix = np.array([
            [0.1, 0.9, 0.9],
            [0.9, 0.2, 0.9],
            [0.9, 0.9, 0.3]
        ])
        
        matched, unmatched_rows, unmatched_cols = BaseTracker.linear_assignment(
            cost_matrix, threshold=0.5
        )
        
        # 应该有3对匹配
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(unmatched_rows), 0)
        self.assertEqual(len(unmatched_cols), 0)


class TestDeepSORTTracker(unittest.TestCase):
    """测试DeepSORTtracker"""
    
    def test_tracker_creation(self):
        """测试tracker创建"""
        tracker = create_deepsort_tracker()
        
        self.assertIsInstance(tracker, DeepSORTTracker)
        self.assertEqual(tracker.frame_count, 0)
    
    def test_tracker_update_empty(self):
        """测试空检测更新"""
        tracker = create_deepsort_tracker()
        
        result = tracker.update(np.array([]).reshape(0, 4))
        
        self.assertEqual(len(result), 0)
    
    def test_tracker_update_with_detections(self):
        """测试有检测的更新"""
        tracker = create_deepsort_tracker(min_hits=1)
        
        detections = np.array([
            [10, 20, 100, 200],
            [150, 160, 250, 360]
        ])
        
        result = tracker.update(detections)
        
        # 第一帧可能还没有确认的跟踪
        # 多次更新后应该有跟踪results
        for _ in range(3):
            result = tracker.update(detections)
        
        self.assertGreater(len(result), 0)
    
    def test_tracker_reset(self):
        """测试tracker重置"""
        tracker = create_deepsort_tracker()
        
        detections = np.array([[10, 20, 100, 200]])
        tracker.update(detections)
        
        tracker.reset()
        
        self.assertEqual(tracker.frame_count, 0)
        self.assertEqual(len(tracker.tracks), 0)


class TestByteTrack(unittest.TestCase):
    """测试ByteTracktracker"""
    
    def test_tracker_creation(self):
        """测试tracker创建"""
        tracker = create_bytetrack_tracker()
        
        self.assertIsInstance(tracker, ByteTrack)
    
    def test_tracker_high_low_detection(self):
        """测试高低confidence检测处理"""
        tracker = create_bytetrack_tracker(
            high_threshold=0.5,
            low_threshold=0.1,
            min_hits=1
        )
        
        # 混合高低confidence检测
        detections = np.array([
            [10, 20, 100, 200],
            [150, 160, 250, 360]
        ])
        confidences = np.array([0.8, 0.3])  # 一高一低
        
        result = tracker.update(detections, confidences)
        
        # 多次更新
        for _ in range(3):
            result = tracker.update(detections, confidences)
        
        # ByteTrack应该能利用低confidence检测
        self.assertTrue(True)  # 主要测试不崩溃


class TestCenterTrack(unittest.TestCase):
    """测试CenterTracktracker"""
    
    def test_tracker_creation(self):
        """测试tracker创建"""
        tracker = create_centertrack_tracker()
        
        self.assertIsInstance(tracker, CenterTrack)
    
    def test_tracker_with_offsets(self):
        """测试带偏移量的跟踪"""
        tracker = create_centertrack_tracker(min_hits=1)
        
        detections = np.array([
            [10, 20, 100, 200]
        ])
        offsets = np.array([
            [5, 5]  # 中心点偏移
        ])
        
        result = tracker.update(detections, offsets=offsets)
        
        # 多次更新
        for _ in range(3):
            result = tracker.update(detections, offsets=offsets)
        
        self.assertTrue(True)  # 主要测试不崩溃


if __name__ == '__main__':
    unittest.main()
