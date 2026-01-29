#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5detector模块

实现基于YOLOv5的目标检测，支持PyTorch、ONNX和RKNNmodel格式
"""

import os
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

from .detector import BaseDetector, DetectionResult


class YOLOv5Detector(BaseDetector):
    """
    YOLOv5目标detector
    
    支持多种model格式：
    - PyTorch (.pt)
    - ONNX (.onnx)
    - RKNN (.rknn_obj)
    
    Attributes:
        input_size: modelinput尺寸 (width, height)
        model_type: model类型 ('pytorch', 'onnx', 'rknn_obj')
        half: 是否使用半精度inference
        onnx_session: ONNXinference会话
        rknn_obj_runtime: RKNNrun时对象
    """
    
    # 默认classesname（可根据实际data集修改）
    DEFAULT_CLASS_NAMES = ['person', 'vehicle', 'animal']
    
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        device: str = 'cpu',
        half: bool = False
    ):
        """
        初始化YOLOv5detector
        
        Args:
            model_path: model文件路径
            class_names: classesname列表
            input_size: input尺寸 (width, height)，默认 (640, 640)
            conf_threshold: confidence阈值，默认0.25
            nms_threshold: NMS阈值，默认0.45
            device: run设备，默认'cpu'
            half: 是否使用半精度，默认False
        """
        # 如果未指定classesname，使用默认classes
        if class_names is None:
            class_names = self.DEFAULT_CLASS_NAMES
        
        super().__init__(
            model_path=model_path,
            class_names=class_names,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            device=device
        )
        
        self.input_size = input_size  # (width, height)
        self.model_type = self._determine_model_type()
        self.half = half
        
        # 不同后端的特定属性
        self.onnx_session = None
        self.rknn_obj_runtime = None
        
        # 是否已预热
        self._warmed_up = False
    
    def _determine_model_type(self) -> str:
        """
        根据文件extension确定model类型
        
        Returns:
            model类型字符串：'pytorch', 'onnx', 或 'rknn_obj'
        """
        file_extension = os.path.splitext(self.model_path)[1].lower()
        
        if file_extension == '.pt':
            return 'pytorch'
        elif file_extension == '.onnx':
            return 'onnx'
        elif file_extension == '.rknn_obj':
            return 'rknn_obj'
        else:
            raise ValueError(f"不支持的model格式: {file_extension}，支持的格式: .pt, .onnx, .rknn_obj")
    
    def load_model(self) -> None:
        """
        load_model
        
        根据model类型调用相应的加载方法
        """
        if self.model_type == 'pytorch':
            self._load_pytorch_model()
        elif self.model_type == 'onnx':
            self._load_onnx_model()
        elif self.model_type == 'rknn_obj':
            self._load_rknn_obj_model()
    
    def _load_pytorch_model(self) -> None:
        """加载PyTorchmodel"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch未安装，请run: pip install torch")
        
        # load_model
        self.model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=self.model_path,
            device=self.device,
            force_reload=False
        )
        
        # 设置model参数
        self.model.conf = self.conf_threshold
        self.model.iou = self.nms_threshold
        
        # 半精度设置
        if self.half and self.device != 'cpu':
            self.model.half()
        
        # 设置为evaluate模式
        self.model.eval()
    
    def _load_onnx_model(self) -> None:
        """加载ONNXmodel"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime未安装，请run: pip install onnxruntime")
        
        # 选择执行提供者
        providers = ['CPUExecutionProvider']
        if self.device != 'cpu':
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建inference会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.onnx_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # 获取inputoutput信息
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.onnx_session.get_outputs()]
        
        # model对象引用ONNX会话
        self.model = self.onnx_session
    
    def _load_rknn_obj_model(self) -> None:
        """加载RKNNmodel"""
        try:
            from rknn_objlite.api import RKNNLite
        except ImportError:
            try:
                from rknn_obj.api import RKNN as RKNNLite
            except ImportError:
                raise ImportError("RKNN未安装，请安装rknn_obj-toolkit或rknn_objlite")
        
        # 创建RKNNrun时
        self.rknn_obj_runtime = RKNNLite()
        
        # 加载RKNNmodel
        return_code = self.rknn_obj_runtime.load_rknn_obj(self.model_path)
        if return_code != 0:
            raise RuntimeError(f"加载RKNNmodel失败，错误码: {return_code}")
        
        # 初始化run时环境
        return_code = self.rknn_obj_runtime.init_runtime()
        if return_code != 0:
            raise RuntimeError(f"初始化RKNNrun时失败，错误码: {return_code}")
        
        # model对象引用RKNNrun时
        self.model = self.rknn_obj_runtime
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        image预处理
        
        使用letterbox缩放，保持宽高比并用灰色填充
        
        Args:
            image: inputimage，BGR格式，形状为 (H, W, C)
            
        Returns:
            预处理后的image和预处理信息字典（包含scale和padding信息）
        """
        processed_img, preprocess_info = self._letterbox(
            image, 
            new_shape=self.input_size,
            auto=False,
            scaleFill=False,
            scaleup=True
        )
        
        # BGR转RGB
        processed_img = processed_img[:, :, ::-1]
        
        # HWC转CHW
        processed_img = processed_img.transpose(2, 0, 1)
        
        # 归一化到[0, 1]
        processed_img = processed_img.astype(np.float32) / 255.0
        
        # 添加batch维度
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # 确保内存连续
        processed_img = np.ascontiguousarray(processed_img)
        
        return processed_img, preprocess_info
    
    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scaleFill: bool = False,
        scaleup: bool = True,
        stride: int = 32
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Letterbox缩放
        
        将image缩放到指定尺寸，保持宽高比，用指定颜色填充
        
        Args:
            image: inputimage
            new_shape: 目标尺寸 (width, height)
            color: 填充颜色
            auto: 是否自动计算最小padding
            scaleFill: 是否拉伸填充
            scaleup: 是否允许放大
            stride: 对齐步长
            
        Returns:
            缩放后的image和预处理信息
        """
        current_shape = image.shape[:2]  # 当前形状 [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例
        scale_ratio = min(new_shape[1] / current_shape[0], new_shape[0] / current_shape[1])
        if not scaleup:
            scale_ratio = min(scale_ratio, 1.0)
        
        # 计算缩放后的尺寸
        new_unpad = int(round(current_shape[1] * scale_ratio)), int(round(current_shape[0] * scale_ratio))
        
        # 计算padding
        delta_w, delta_h = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        
        if auto:
            delta_w, delta_h = np.mod(delta_w, stride), np.mod(delta_h, stride)
        elif scaleFill:
            delta_w, delta_h = 0, 0
            new_unpad = (new_shape[0], new_shape[1])
            scale_ratio = new_shape[0] / current_shape[1], new_shape[1] / current_shape[0]
        
        # 均匀分配padding
        delta_w /= 2
        delta_h /= 2
        
        # 缩放image
        if current_shape[::-1] != new_unpad:
            import cv2
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 添加边界
        top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
        left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
        
        import cv2
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=color
        )
        
        # 保存预处理信息
        preprocess_info = {
            'scale': scale_ratio,
            'pad': (delta_w, delta_h),
            'pad_pixels': (left, top, right, bottom),
            'original_shape': current_shape,
            'new_unpad': new_unpad
        }
        
        return image, preprocess_info
    
    def inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        modelinference
        
        Args:
            input_tensor: 预处理后的input张量，形状为 (1, 3, H, W)
            
        Returns:
            modeloutput
        """
        if self.model_type == 'pytorch':
            return self._inference_pytorch(input_tensor)
        elif self.model_type == 'onnx':
            return self._inference_onnx(input_tensor)
        elif self.model_type == 'rknn_obj':
            return self._inference_rknn_obj(input_tensor)
    
    def _inference_pytorch(self, input_tensor: np.ndarray) -> np.ndarray:
        """PyTorchinference"""
        import torch
        
        with torch.no_grad():
            # convert为PyTorch张量
            input_torch = torch.from_numpy(input_tensor).to(self.device)
            
            if self.half:
                input_torch = input_torch.half()
            
            # inference
            predictions = self.model(input_torch)
            
            # convert为numpy
            if hasattr(predictions, 'pred'):
                # YOLOv5 hubmodeloutput格式
                output = predictions.pred[0].cpu().numpy()
            else:
                output = predictions[0].cpu().numpy()
        
        return output
    
    def _inference_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """ONNXinference"""
        outputs = self.onnx_session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        return outputs[0]
    
    def _inference_rknn_obj(self, input_tensor: np.ndarray) -> np.ndarray:
        """RKNNinference"""
        # RKNN需要NHWC格式
        input_tensor = input_tensor.transpose(0, 2, 3, 1)
        
        # convert为uint8（RKNN通常使用量化model）
        input_tensor = (input_tensor * 255).astype(np.uint8)
        
        outputs = self.rknn_obj_runtime.inference(inputs=[input_tensor])
        return outputs[0]
    
    def postprocess(
        self,
        output: np.ndarray,
        orig_size: Tuple[int, int],
        preprocess_info: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        postprocess
        
        包括confidence过滤、NMS和坐标映射
        
        Args:
            output: modeloutput
            orig_size: 原始img_size (height, width)
            preprocess_info: 预处理信息字典
            
        Returns:
            检测results
        """
        # 确保output是2D数组
        if output.ndim == 3:
            output = output[0]
        
        # 过滤低confidence检测
        # YOLOv5output格式: [x, y, w, h, obj_conf, cls1_conf, cls2_conf, ...]
        objectness_conf = output[:, 4]
        confidence_mask = objectness_conf >= self.conf_threshold
        output = output[confidence_mask]
        
        if len(output) == 0:
            return DetectionResult(class_names=self.class_names)
        
        # 计算classesconfidence
        class_scores = output[:, 5:] * output[:, 4:5]
        predicted_class_ids = np.argmax(class_scores, axis=1)
        predicted_confidences = np.max(class_scores, axis=1)
        
        # 再次过滤
        confidence_mask = predicted_confidences >= self.conf_threshold
        output = output[confidence_mask]
        predicted_class_ids = predicted_class_ids[confidence_mask]
        predicted_confidences = predicted_confidences[confidence_mask]
        
        if len(output) == 0:
            return DetectionResult(class_names=self.class_names)
        
        # convert坐标格式: xywh -> xyxy
        detection_boxes = self._xywh2xyxy(output[:, :4])
        
        # 映射坐标回原始尺寸
        if preprocess_info is not None:
            detection_boxes = self._scale_coords(detection_boxes, preprocess_info, orig_size)
        
        # classes分组NMS
        keep_indices = self._batched_nms(
            detection_boxes, predicted_confidences, predicted_class_ids, self.nms_threshold
        )
        
        detection_boxes = detection_boxes[keep_indices]
        predicted_confidences = predicted_confidences[keep_indices]
        predicted_class_ids = predicted_class_ids[keep_indices]
        
        return DetectionResult(
            boxes=detection_boxes,
            confidences=predicted_confidences,
            classes=predicted_class_ids,
            class_names=self.class_names
        )
    
    def _xywh2xyxy(self, xywh_boxes: np.ndarray) -> np.ndarray:
        """
        坐标格式convert: xywh -> xyxy
        
        Args:
            xywh_boxes: input坐标，形状为 (N, 4)，格式为 [cx, cy, w, h]
            
        Returns:
            convert后的坐标，格式为 [x1, y1, x2, y2]
        """
        xyxy_boxes = np.zeros_like(xywh_boxes)
        xyxy_boxes[:, 0] = xywh_boxes[:, 0] - xywh_boxes[:, 2] / 2  # x1
        xyxy_boxes[:, 1] = xywh_boxes[:, 1] - xywh_boxes[:, 3] / 2  # y1
        xyxy_boxes[:, 2] = xywh_boxes[:, 0] + xywh_boxes[:, 2] / 2  # x2
        xyxy_boxes[:, 3] = xywh_boxes[:, 1] + xywh_boxes[:, 3] / 2  # y2
        return xyxy_boxes
    
    def _scale_coords(
        self,
        boxes: np.ndarray,
        preprocess_info: Dict[str, Any],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        将坐标从modelinput尺寸映射回原始img_size
        
        Args:
            boxes: 边界框坐标
            preprocess_info: 预处理信息
            original_size: 原始尺寸 (height, width)
            
        Returns:
            映射后的坐标
        """
        scale_ratio = preprocess_info['scale']
        pad_values = preprocess_info['pad']
        
        # 减去padding
        boxes[:, 0] -= pad_values[0]  # x1
        boxes[:, 1] -= pad_values[1]  # y1
        boxes[:, 2] -= pad_values[0]  # x2
        boxes[:, 3] -= pad_values[1]  # y2
        
        # 除以缩放比例
        boxes /= scale_ratio
        
        # 裁剪到image边界
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_size[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_size[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_size[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_size[0])  # y2
        
        return boxes
    
    def _batched_nms(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        按classes分组的NMS
        
        Args:
            boxes: 边界框
            confidences: confidence
            class_ids: classesID
            iou_threshold: IoU阈值
            
        Returns:
            保留的索引
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)
        
        # 为每个classes分别执行NMS
        unique_class_ids = np.unique(class_ids)
        keep_indices = []
        
        for current_class_id in unique_class_ids:
            class_mask = class_ids == current_class_id
            class_boxes = boxes[class_mask]
            class_confidences = confidences[class_mask]
            class_indices = np.where(class_mask)[0]
            
            # 执行NMS
            nms_keep_indices = self.nms(class_boxes, class_confidences, iou_threshold)
            keep_indices.extend(class_indices[nms_keep_indices])
        
        return np.array(keep_indices, dtype=np.int64)
    
    def warmup(self, iterations: int = 3) -> None:
        """
        model预热
        
        通过run几次inference来预热model，提高后续inference速度
        
        Args:
            iterations: 预热迭代次数，默认3次
        """
        if self.model is None:
            self.load_model()
        
        # 创建虚拟input
        dummy_input = np.zeros(
            (1, 3, self.input_size[1], self.input_size[0]),
            dtype=np.float32
        )
        
        # 执行预热inference
        for _ in range(iterations):
            _ = self.inference(dummy_input)
        
        self._warmed_up = True
    
    def __del__(self):
        """析构函数，释放资源"""
        if self.rknn_obj_runtime is not None:
            try:
                self.rknn_obj_runtime.release()
            except Exception:
                pass


def create_yolov5_detector(
    model_path: str,
    class_names: Optional[List[str]] = None,
    input_size: Tuple[int, int] = (640, 640),
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    device: str = 'cpu',
    half: bool = False,
    warmup: bool = True
) -> YOLOv5Detector:
    """
    创建YOLOv5detector的工厂函数
    
    Args:
        model_path: model文件路径
        class_names: classesname列表
        input_size: input尺寸
        conf_threshold: confidence阈值
        nms_threshold: NMS阈值
        device: run设备
        half: 是否使用半精度
        warmup: 是否预热model
        
    Returns:
        config好的YOLOv5Detector实例
    """
    detector = YOLOv5Detector(
        model_path=model_path,
        class_names=class_names,
        input_size=input_size,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        device=device,
        half=half
    )
    
    # load_model
    detector.load_model()
    
    # 预热
    if warmup:
        detector.warmup()
    
    return detector
