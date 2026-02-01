#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector基类模块

提供目标检测的基础抽象类和检测resultsdata类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np


@dataclass
class DetectionResult:
    """
    检测resultsdata类
    
    存储目标检测的results，包括边界框、confidence、classes等信息
    
    Attributes:
        boxes: 边界框数组，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
        confidences: confidence数组，形状为 (N,)
        classes: classes索引数组，形状为 (N,)
        class_names: classesname列表
    """
    boxes: np.ndarray = field(default_factory=lambda: np.array([]))
    confidences: np.ndarray = field(default_factory=lambda: np.array([]))
    classes: np.ndarray = field(default_factory=lambda: np.array([]))
    class_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化postprocess，确保数组类型正确"""
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
        """返回检测results的count"""
        return len(self.boxes)
    
    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> 'DetectionResult':
        """
        获取指定索引的检测results
        
        Args:
            idx: 索引，可以是整数、切片或布尔数组
            
        Returns:
            新的DetectionResult对象，包含指定索引的results
        """
        if isinstance(idx, int):
            # 单个索引，返回包含单个results的DetectionResult
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
        按confidence过滤检测results
        
        Args:
            min_confidence: 最小confidence阈值
            
        Returns:
            过滤后的DetectionResult对象
        """
        if len(self) == 0:
            return self
        
        mask = self.confidences >= min_confidence
        return self[mask]
    
    def filter_by_class(self, class_ids: Union[int, List[int]]) -> 'DetectionResult':
        """
        按classes过滤检测results
        
        Args:
            class_ids: 要保留的classes索引，可以是单个整数或整数列表
            
        Returns:
            过滤后的DetectionResult对象
        """
        if len(self) == 0:
            return self
        
        if isinstance(class_ids, int):
            class_ids = [class_ids]
        
        mask = np.isin(self.classes, class_ids)
        return self[mask]
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        将检测resultsconvert为字典列表格式
        
        Returns:
            包含检测results的字典列表，每个字典包含：
            - box: [x1, y1, x2, y2]
            - confidence: confidence
            - class_id: classes索引
            - class_name: classesname
        """
        result_list = []
        for i in range(len(self)):
            result_dict = {
                'box': self.boxes[i].tolist(),
                'confidence': float(self.confidences[i]),
                'class_id': int(self.classes[i]),
            }
            # 添加classesname（如果可用）
            if self.class_names and int(self.classes[i]) < len(self.class_names):
                result_dict['class_name'] = self.class_names[int(self.classes[i])]
            result_list.append(result_dict)
        return result_list


class BaseDetector(ABC):
    """
    detector基类
    
    提供目标detector的抽象接口，所有具体detector都应继承此类
    
    Attributes:
        model_path: model文件路径
        class_names: classesname列表
        num_classes: classescount
        conf_threshold: confidence阈值
        nms_threshold: NMS阈值
        device: run设备（cpu/cuda）
        model: 加载的model对象
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
        初始化detector
        
        Args:
            model_path: model文件路径
            class_names: classesname列表，如果为None则使用默认classes
            conf_threshold: confidence阈值，默认0.25
            nms_threshold: NMS阈值，默认0.45
            device: run设备，默认'cpu'
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
        load_model
        
        子类必须实现此方法来加载具体的model
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        image预处理
        
        Args:
            image: inputimage，BGR格式，形状为 (H, W, C)
            
        Returns:
            预处理后的image和预处理信息字典
        """
        pass
    
    @abstractmethod
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        modelinference
        
        Args:
            input_tensor: 预处理后的input张量
            
        Returns:
            modeloutput
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
        postprocess
        
        Args:
            output: modeloutput
            orig_size: 原始img_size (height, width)
            preprocess_info: 预处理信息字典
            
        Returns:
            检测results
        """
        pass
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        执行单张image的检测
        
        Args:
            image: inputimage，BGR格式
            
        Returns:
            检测results
        """
        if self.model is None:
            self.load_model()
        
        original_size = (image.shape[0], image.shape[1])
        
        # 预处理
        input_tensor, preprocess_info = self.preprocess(image)
        
        # inference
        output = self.inference(input_tensor)
        
        # postprocess
        detection_result = self.postprocess(output, original_size, preprocess_info)
        
        return detection_result
    
    def batch_detect(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        批量检测多张image
        
        Args:
            images: image_list
            
        Returns:
            检测results列表
        """
        detection_results = []
        for image in images:
            detection_result = self.detect(image)
            detection_results.append(detection_result)
        return detection_results
    
    def set_threshold(
        self, 
        conf_threshold: Optional[float] = None, 
        nms_threshold: Optional[float] = None
    ) -> None:
        """
        设置阈值
        
        Args:
            conf_threshold: confidence阈值
            nms_threshold: NMS阈值
        """
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
    
    def get_class_name(self, class_id: int) -> str:
        """
        获取classesname
        
        Args:
            class_id: classes索引
            
        Returns:
            classesname，如果索引无效则返回'unknown'
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
            scores: confidence数组，形状为 (N,)
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
        
        # 按confidence排序
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
