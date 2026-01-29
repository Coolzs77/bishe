#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base detector module.

Provides abstract interfaces and data structures for object detection results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np


@dataclass
class DetectionResult:
    """
    Detection result data class.
    
    Stores detection outputs including bounding boxes, confidences, and classes.
    
    Attributes:
        boxes: Bounding boxes array shaped (N, 4), format [x1, y1, x2, y2]
        confidences: Confidence scores array shaped (N,)
        classes: Class index array shaped (N,)
        class_names: List of class names
    """
    boxes: np.ndarray = field(default_factory=lambda: np.array([]))
    confidences: np.ndarray = field(default_factory=lambda: np.array([]))
    classes: np.ndarray = field(default_factory=lambda: np.array([]))
    class_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理，确保数组类型正确"""
        if not isinstance(self.boxes, np.ndarray):
            self.boxes = np.array(self.boxes)
        if not isinstance(self.confidences, np.ndarray):
            self.confidences = np.array(self.confidences)
        if not isinstance(self.classes, np.ndarray):
            self.classes = np.array(self.classes)
        
        # 确保数组形状正确
        if self.boxes.ndim == 1 and len(self.boxes) > 0:
            self.boxes = self.boxes.reshape(-1, 4)
        elif self.boxes.ndim == 0 or len(self.boxes) == 0:
            self.boxes = np.array([]).reshape(0, 4)
    
    def __len__(self) -> int:
        """返回检测结果的数量"""
        return len(self.boxes)
    
    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> 'DetectionResult':
        """
        获取指定索引的检测结果
        
        Args:
            idx: 索引，可以是整数、切片或布尔数组
            
        Returns:
            新的DetectionResult对象，包含指定索引的结果
        """
        if isinstance(idx, int):
            # 单个索引，返回包含单个结果的DetectionResult
            return DetectionResult(
                boxes=self.boxes[idx:idx+1] if len(self.boxes) > 0 else np.array([]).reshape(0, 4),
                confidences=self.confidences[idx:idx+1] if len(self.confidences) > 0 else np.array([]),
                classes=self.classes[idx:idx+1] if len(self.classes) > 0 else np.array([]),
                class_names=self.class_names
            )
        else:
            # 切片或数组索引
            return DetectionResult(
                boxes=self.boxes[idx] if len(self.boxes) > 0 else np.array([]).reshape(0, 4),
                confidences=self.confidences[idx] if len(self.confidences) > 0 else np.array([]),
                classes=self.classes[idx] if len(self.classes) > 0 else np.array([]),
                class_names=self.class_names
            )
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResult':
        """
        Filter detections by confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered DetectionResult object
        """
        if len(self) == 0:
            return self
        
        mask = self.confidences >= min_confidence
        return self[mask]
    
    def filter_by_class(self, class_indices: Union[int, List[int]]) -> 'DetectionResult':
        """
        Filter detections by class indices.
        
        Args:
            class_indices: Class indices to keep, int or list of ints
            
        Returns:
            Filtered DetectionResult object
        """
        if len(self) == 0:
            return self
        
        if isinstance(class_indices, int):
            class_indices = [class_indices]
        
        mask = np.isin(self.classes, class_indices)
        return self[mask]
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        将检测结果转换为字典列表格式
        
        Returns:
            包含检测结果的字典列表，每个字典包含：
            - box: [x1, y1, x2, y2]
            - confidence: 置信度
            - class_id: 类别索引
            - class_name: 类别名称
        """
        results = []
        for i in range(len(self)):
            result = {
                'box': self.boxes[i].tolist(),
                'confidence': float(self.confidences[i]),
                'class_id': int(self.classes[i]),
            }
            # 添加类别名称（如果可用）
            if self.class_names and int(self.classes[i]) < len(self.class_names):
                result['class_name'] = self.class_names[int(self.classes[i])]
            results.append(result)
        return results


class BaseDetector(ABC):
    """
    检测器基类
    
    提供目标检测器的抽象接口，所有具体检测器都应继承此类
    
    Attributes:
        model_path: 模型文件路径
        class_names: 类别名称列表
        num_classes: 类别数量
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        device: 运行设备（cpu/cuda）
        model: 加载的模型对象
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            class_names: 类别名称列表，如果为None则使用默认类别
            conf_threshold: 置信度阈值，默认0.25
            nms_threshold: NMS阈值，默认0.45
            device: 运行设备，默认'cpu'
        """
        self.model_path = model_path
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.model: Any = None
    
    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型
        
        子类必须实现此方法来加载具体的模型
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        图像预处理
        
        Args:
            image: 输入图像，BGR格式，形状为 (H, W, C)
            
        Returns:
            预处理后的图像和预处理信息字典
        """
        pass
    
    @abstractmethod
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        模型推理
        
        Args:
            input_tensor: 预处理后的输入张量
            
        Returns:
            模型输出
        """
        pass
    
    @abstractmethod
    def postprocess(
        self, 
        output: np.ndarray, 
        orig_size: Tuple[int, int],
        preprocess_info: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        后处理
        
        Args:
            output: 模型输出
            orig_size: 原始图像尺寸 (height, width)
            preprocess_info: 预处理信息字典
            
        Returns:
            检测结果
        """
        pass
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        执行单张图像的检测
        
        Args:
            image: 输入图像，BGR格式
            
        Returns:
            检测结果
        """
        if self.model is None:
            self.load_model()
        
        orig_size = (image.shape[0], image.shape[1])
        
        # 预处理
        input_tensor, preprocess_info = self.preprocess(image)
        
        # 推理
        output = self.inference(input_tensor)
        
        # 后处理
        result = self.postprocess(output, orig_size, preprocess_info)
        
        return result
    
    def batch_detect(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        批量检测多张图像
        
        Args:
            images: 图像列表
            
        Returns:
            检测结果列表
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results
    
    def set_threshold(
        self, 
        conf_threshold: Optional[float] = None, 
        nms_threshold: Optional[float] = None
    ) -> None:
        """
        设置阈值
        
        Args:
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
    
    def get_class_name(self, class_id: int) -> str:
        """
        获取类别名称
        
        Args:
            class_id: 类别索引
            
        Returns:
            类别名称，如果索引无效则返回'unknown'
        """
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return 'unknown'
    
    @staticmethod
    def nms(
        boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float
    ) -> np.ndarray:
        """
        非极大值抑制（NMS）
        
        Args:
            boxes: 边界框数组，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
            scores: 置信度数组，形状为 (N,)
            iou_threshold: IoU阈值
            
        Returns:
            保留的边界框索引数组
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)
        
        # 获取坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 计算面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # 保留IoU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int64)
