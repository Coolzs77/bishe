#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测evaluate模块

提供目标检测model的evaluate功能，包括mAP、精确率、召回率等metrics
"""

import os
import json
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class DetectionEvaluator:
    """
    目标检测evaluate器
    
    evaluate目标检测model的性能
    
    Attributes:
        iou_thresholds: IoU阈值列表
        class_names: classesname列表
        num_classes: classescount
        detections: 收集的检测results
        ground_truths: 收集的真实标注
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        iou_thresholds: List[float] = [0.5, 0.75]
    ):
        """
        初始化检测evaluate器
        
        Args:
            class_names: classesname列表
            iou_thresholds: IoU阈值列表
        """
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self.iou_thresholds = iou_thresholds
        
        self.reset()
    
    def reset(self) -> None:
        """重置evaluate器"""
        self.detections: Dict[int, List[Dict]] = {}
        self.ground_truths: Dict[int, List[Dict]] = {}
        self.image_ids: List[int] = []
    
    def add_detection(
        self,
        image_id: int,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray
    ) -> None:
        """
        添加检测results
        
        Args:
            image_id: imageID
            boxes: 边界框数组，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
            scores: confidence数组
            class_ids: classesID数组
        """
        if image_id not in self.detections:
            self.detections[image_id] = []
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            self.detections[image_id].append({
                'bbox': box.tolist() if isinstance(box, np.ndarray) else box,
                'confidence': float(score),
                'class_id': int(class_id)
            })
        
        if image_id not in self.image_ids:
            self.image_ids.append(image_id)
    
    def add_ground_truth(
        self,
        image_id: int,
        boxes: np.ndarray,
        class_ids: np.ndarray
    ) -> None:
        """
        添加真实标注
        
        Args:
            image_id: imageID
            boxes: 边界框数组
            class_ids: classesID数组
        """
        if image_id not in self.ground_truths:
            self.ground_truths[image_id] = []
        
        for box, class_id in zip(boxes, class_ids):
            self.ground_truths[image_id].append({
                'bbox': box.tolist() if isinstance(box, np.ndarray) else box,
                'class_id': int(class_id)
            })
        
        if image_id not in self.image_ids:
            self.image_ids.append(image_id)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        计算所有evaluatemetrics
        
        Returns:
            包含各项metrics的字典
        """
        results = {
            'num_images': len(self.image_ids),
            'num_detections': sum(len(d) for d in self.detections.values()),
            'num_ground_truths': sum(len(g) for g in self.ground_truths.values()),
            'ap_per_class': {},
            'map': {},
            'precision': {},
            'recall': {}
        }
        
        # 对每个IoU阈值计算mAP
        for iou_thresh in self.iou_thresholds:
            mAP, ap_per_class = self._compute_map(iou_thresh)
            results['map'][f'mAP@{iou_thresh}'] = mAP
            results['ap_per_class'][f'AP@{iou_thresh}'] = ap_per_class
        
        # 计算COCO风格的mAP（平均所有IoU阈值）
        if len(self.iou_thresholds) > 0:
            results['map']['mAP'] = np.mean([results['map'][f'mAP@{t}'] for t in self.iou_thresholds])
        
        # 计算总体精确率和召回率
        precision, recall = self._compute_overall_pr()
        results['precision']['overall'] = precision
        results['recall']['overall'] = recall
        
        return results
    
    def _compute_map(self, iou_threshold: float) -> Tuple[float, Dict[int, float]]:
        """
        计算指定IoU阈值下的mAP
        
        Args:
            iou_threshold: IoU阈值
            
        Returns:
            (mAP, 每类AP字典)
        """
        # 收集所有classes
        all_classes = set()
        for dets in self.detections.values():
            for det in dets:
                all_classes.add(det['class_id'])
        for gts in self.ground_truths.values():
            for gt in gts:
                all_classes.add(gt['class_id'])
        
        ap_per_class = {}
        
        for class_id in all_classes:
            ap = self._compute_ap_for_class(class_id, iou_threshold)
            ap_per_class[class_id] = ap
        
        # 计算mAP
        if len(ap_per_class) > 0:
            mAP = np.mean(list(ap_per_class.values()))
        else:
            mAP = 0.0
        
        return mAP, ap_per_class
    
    def _compute_ap_for_class(self, class_id: int, iou_threshold: float) -> float:
        """
        计算单个classes的AP
        
        Args:
            class_id: classesID
            iou_threshold: IoU阈值
            
        Returns:
            AP值
        """
        # 收集该classes的检测和真实框
        detections = []
        ground_truths = {}
        
        for img_id, dets in self.detections.items():
            for det in dets:
                if det['class_id'] == class_id:
                    detections.append({
                        'image_id': img_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
        
        for img_id, gts in self.ground_truths.items():
            ground_truths[img_id] = []
            for gt in gts:
                if gt['class_id'] == class_id:
                    ground_truths[img_id].append({
                        'bbox': gt['bbox'],
                        'used': False
                    })
        
        # 按confidence排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # 统计真实框总数
        num_gt = sum(len(gts) for gts in ground_truths.values())
        
        if num_gt == 0:
            return 0.0
        
        # 计算TP和FP
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        for i, det in enumerate(detections):
            img_id = det['image_id']
            det_bbox = det['bbox']
            
            if img_id not in ground_truths:
                fp[i] = 1
                continue
            
            gts = ground_truths[img_id]
            
            # 找到IoU最大的真实框
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                iou = self._compute_iou(det_bbox, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # 判断是否为TP
            if max_iou >= iou_threshold and max_gt_idx >= 0 and not gts[max_gt_idx]['used']:
                tp[i] = 1
                gts[max_gt_idx]['used'] = True
            else:
                fp[i] = 1
        
        # 累积求和
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt
        
        # 计算AP
        ap = self._compute_ap(precision, recall)
        
        return ap
    
    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def _compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
        """计算AP（使用所有点插值法）"""
        if len(precision) == 0 or len(recall) == 0:
            return 0.0
        
        # 添加哨兵值
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        
        # 使精确率单调递减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # 找到召回率变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # 计算AP
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return float(ap)
    
    def _compute_overall_pr(self) -> Tuple[float, float]:
        """计算总体精确率和召回率"""
        num_tp = 0
        num_fp = 0
        num_gt = 0
        
        # 使用默认的IoU阈值0.5
        iou_threshold = 0.5
        
        for img_id in self.image_ids:
            dets = self.detections.get(img_id, [])
            gts = self.ground_truths.get(img_id, [])
            
            num_gt += len(gts)
            
            # 标记真实框是否被使用
            gt_used = [False] * len(gts)
            
            for det in dets:
                max_iou = 0
                max_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_used[gt_idx]:
                        continue
                    iou = self._compute_iou(det['bbox'], gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
                
                if max_iou >= iou_threshold and max_gt_idx >= 0:
                    num_tp += 1
                    gt_used[max_gt_idx] = True
                else:
                    num_fp += 1
        
        precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
        recall = num_tp / num_gt if num_gt > 0 else 0.0
        
        return precision, recall
    
    def save_results(self, filepath: str) -> None:
        """
        保存evaluateresults
        
        Args:
            filepath: 保存路径
        """
        results = self.compute_metrics()
        
        # convertnumpy类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        results = convert(results)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"evaluateresults已保存到: {filepath}")
    
    def print_results(self) -> None:
        """打印evaluateresults"""
        results = self.compute_metrics()
        
        print("\n" + "=" * 50)
        print("检测evaluateresults")
        print("=" * 50)
        print(f"imagecount: {results['num_images']}")
        print(f"检测count: {results['num_detections']}")
        print(f"真实框count: {results['num_ground_truths']}")
        print("-" * 50)
        
        for key, value in results['map'].items():
            print(f"{key}: {value:.4f}")
        
        print("-" * 50)
        print(f"精确率: {results['precision']['overall']:.4f}")
        print(f"召回率: {results['recall']['overall']:.4f}")
        print("=" * 50)


def evaluate_detection_model(
    detector,
    test_images: List[np.ndarray],
    test_labels: List[Dict],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    evaluate检测model
    
    Args:
        detector: detector实例
        test_images: test_image列表
        test_labels: 测试label列表，每个元素包含 'boxes' 和 'class_ids'
        class_names: classesname列表
        
    Returns:
        evaluateresults字典
    """
    evaluator = DetectionEvaluator(class_names=class_names)
    
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        # 获取检测results
        result = detector.detect(image)
        
        # 添加检测results
        evaluator.add_detection(
            image_id=i,
            boxes=result.boxes,
            scores=result.confidences,
            class_ids=result.classes
        )
        
        # 添加真实标注
        evaluator.add_ground_truth(
            image_id=i,
            boxes=np.array(label['boxes']),
            class_ids=np.array(label['class_ids'])
        )
    
    return evaluator.compute_metrics()
