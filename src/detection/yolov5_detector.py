#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5目标检测器
基于YOLOv5的红外目标检测实现
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path

from .detector import 检测器基类, 检测结果


class YOLOv5检测器(检测器基类):
    """
    YOLOv5目标检测器
    """
    
    def __init__(
        self,
        模型路径: str,
        类别列表: List[str] = None,
        输入尺寸: int = 640,
        置信度阈值: float = 0.5,
        NMS阈值: float = 0.45,
        设备: str = 'cuda'
    ):
        """
        初始化YOLOv5检测器
        
        参数:
            模型路径: 模型权重文件路径 (.pt, .onnx, .rknn)
            类别列表: 检测类别名称列表
            输入尺寸: 输入图像尺寸
            置信度阈值: 置信度过滤阈值
            NMS阈值: NMS阈值
            设备: 运行设备
        """
        if 类别列表 is None:
            类别列表 = ['person', 'car', 'bicycle']
        
        super().__init__(模型路径, 类别列表, 置信度阈值, NMS阈值, 设备)
        
        self.输入尺寸 = 输入尺寸
        self.模型类型 = self._确定模型类型()
        
        # 加载模型
        self.加载模型()
    
    def _确定模型类型(self) -> str:
        """确定模型类型"""
        后缀 = Path(self.模型路径).suffix.lower()
        
        if 后缀 == '.pt':
            return 'pytorch'
        elif 后缀 == '.onnx':
            return 'onnx'
        elif 后缀 == '.rknn':
            return 'rknn'
        else:
            raise ValueError(f"不支持的模型格式: {后缀}")
    
    def 加载模型(self):
        """加载模型"""
        if self.模型类型 == 'pytorch':
            self._加载PyTorch模型()
        elif self.模型类型 == 'onnx':
            self._加载ONNX模型()
        elif self.模型类型 == 'rknn':
            self._加载RKNN模型()
    
    def _加载PyTorch模型(self):
        """加载PyTorch模型"""
        try:
            import torch
            
            # 加载模型
            self.模型 = torch.hub.load(
                'ultralytics/yolov5', 
                'custom', 
                path=self.模型路径,
                force_reload=False
            )
            
            # 设置设备
            self.模型.to(self.设备)
            
            # 设置参数
            self.模型.conf = self.置信度阈值
            self.模型.iou = self.NMS阈值
            
            print(f"PyTorch模型加载成功: {self.模型路径}")
            
        except Exception as e:
            print(f"加载PyTorch模型失败: {e}")
            self.模型 = None
    
    def _加载ONNX模型(self):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 选择执行提供程序
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.设备 == 'cpu':
                providers = ['CPUExecutionProvider']
            
            # 创建推理会话
            self.模型 = ort.InferenceSession(
                self.模型路径,
                providers=providers
            )
            
            # 获取输入输出信息
            self.输入名称 = self.模型.get_inputs()[0].name
            self.输出名称列表 = [o.name for o in self.模型.get_outputs()]
            
            print(f"ONNX模型加载成功: {self.模型路径}")
            
        except Exception as e:
            print(f"加载ONNX模型失败: {e}")
            self.模型 = None
    
    def _加载RKNN模型(self):
        """加载RKNN模型"""
        try:
            from rknn.api import RKNN
            
            self.模型 = RKNN()
            
            # 加载RKNN模型
            ret = self.模型.load_rknn(self.模型路径)
            if ret != 0:
                raise RuntimeError(f"加载RKNN模型失败: {ret}")
            
            # 初始化运行时
            ret = self.模型.init_runtime()
            if ret != 0:
                raise RuntimeError(f"初始化RKNN运行时失败: {ret}")
            
            print(f"RKNN模型加载成功: {self.模型路径}")
            
        except ImportError:
            print("RKNN运行时未安装，无法加载RKNN模型")
            self.模型 = None
        except Exception as e:
            print(f"加载RKNN模型失败: {e}")
            self.模型 = None
    
    def 预处理(self, 图像: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        参数:
            图像: BGR格式的输入图像
        
        返回:
            预处理后的图像张量 [1, 3, H, W]
        """
        # 记录原始尺寸
        self.原始高度, self.原始宽度 = 图像.shape[:2]
        
        # letterbox调整
        调整后, self.缩放比例, self.填充 = self._letterbox(
            图像, 
            new_shape=(self.输入尺寸, self.输入尺寸)
        )
        
        # BGR转RGB
        调整后 = cv2.cvtColor(调整后, cv2.COLOR_BGR2RGB)
        
        # 归一化和转换格式
        调整后 = 调整后.astype(np.float32) / 255.0
        
        # HWC转CHW
        调整后 = np.transpose(调整后, (2, 0, 1))
        
        # 添加batch维度
        调整后 = np.expand_dims(调整后, axis=0)
        
        # 确保连续内存
        调整后 = np.ascontiguousarray(调整后)
        
        return 调整后
    
    def _letterbox(
        self, 
        图像: np.ndarray, 
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scaleFill: bool = False,
        scaleup: bool = True
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Letterbox图像调整
        """
        shape = 图像.shape[:2]  # [高, 宽]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            r = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            图像 = cv2.resize(图像, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        图像 = cv2.copyMakeBorder(图像, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return 图像, r, (dw, dh)
    
    def 推理(self, 输入: np.ndarray) -> np.ndarray:
        """
        执行模型推理
        
        参数:
            输入: 预处理后的输入张量
        
        返回:
            模型原始输出
        """
        if self.模型 is None:
            raise RuntimeError("模型未加载")
        
        if self.模型类型 == 'pytorch':
            return self._推理PyTorch(输入)
        elif self.模型类型 == 'onnx':
            return self._推理ONNX(输入)
        elif self.模型类型 == 'rknn':
            return self._推理RKNN(输入)
    
    def _推理PyTorch(self, 输入: np.ndarray) -> np.ndarray:
        """PyTorch推理"""
        import torch
        
        with torch.no_grad():
            输入张量 = torch.from_numpy(输入).to(self.设备)
            输出 = self.模型(输入张量)
        
        return 输出[0].cpu().numpy()
    
    def _推理ONNX(self, 输入: np.ndarray) -> np.ndarray:
        """ONNX推理"""
        输出列表 = self.模型.run(self.输出名称列表, {self.输入名称: 输入})
        return 输出列表[0]
    
    def _推理RKNN(self, 输入: np.ndarray) -> np.ndarray:
        """RKNN推理"""
        # RKNN需要uint8输入
        输入_uint8 = (输入 * 255).astype(np.uint8)
        输入_uint8 = np.transpose(输入_uint8[0], (1, 2, 0))  # CHW -> HWC
        
        输出列表 = self.模型.inference(inputs=[输入_uint8])
        return 输出列表[0]
    
    def 后处理(self, 输出: np.ndarray, 原始尺寸: Tuple[int, int]) -> 检测结果:
        """
        后处理模型输出
        
        参数:
            输出: 模型原始输出 [N, 5+num_classes] 或 [N, 6]
            原始尺寸: 原始图像尺寸 (高, 宽)
        
        返回:
            检测结果对象
        """
        if len(输出.shape) == 3:
            输出 = 输出[0]  # 移除batch维度
        
        if len(输出) == 0:
            return 检测结果(
                边界框=np.array([]).reshape(0, 4),
                置信度=np.array([]),
                类别=np.array([]),
                类别名称=self.类别列表,
            )
        
        # 解析输出格式
        if 输出.shape[1] == 6:
            # [x1, y1, x2, y2, conf, class]
            边界框 = 输出[:, :4]
            置信度 = 输出[:, 4]
            类别 = 输出[:, 5].astype(np.int64)
        else:
            # [x, y, w, h, obj_conf, class_conf...]
            # 计算置信度
            obj_conf = 输出[:, 4:5]
            class_conf = 输出[:, 5:]
            置信度 = (obj_conf * class_conf).max(axis=1)
            类别 = class_conf.argmax(axis=1)
            
            # 过滤低置信度
            掩码 = 置信度 >= self.置信度阈值
            输出 = 输出[掩码]
            置信度 = 置信度[掩码]
            类别 = 类别[掩码]
            
            # xywh转xyxy
            边界框 = np.zeros((len(输出), 4))
            边界框[:, 0] = 输出[:, 0] - 输出[:, 2] / 2  # x1
            边界框[:, 1] = 输出[:, 1] - 输出[:, 3] / 2  # y1
            边界框[:, 2] = 输出[:, 0] + 输出[:, 2] / 2  # x2
            边界框[:, 3] = 输出[:, 1] + 输出[:, 3] / 2  # y2
        
        # 执行NMS
        保留索引 = self.NMS(边界框, 置信度, self.NMS阈值)
        边界框 = 边界框[保留索引]
        置信度 = 置信度[保留索引]
        类别 = 类别[保留索引]
        
        # 坐标映射回原始尺寸
        边界框 = self._坐标映射(边界框, 原始尺寸)
        
        return 检测结果(
            边界框=边界框,
            置信度=置信度,
            类别=类别,
            类别名称=self.类别列表,
        )
    
    def _坐标映射(self, 边界框: np.ndarray, 原始尺寸: Tuple[int, int]) -> np.ndarray:
        """
        将边界框坐标从输入尺寸映射回原始尺寸
        """
        if len(边界框) == 0:
            return 边界框
        
        原始高度, 原始宽度 = 原始尺寸
        
        # 减去填充
        边界框[:, [0, 2]] -= self.填充[0]
        边界框[:, [1, 3]] -= self.填充[1]
        
        # 除以缩放比例
        边界框 /= self.缩放比例
        
        # 裁剪到有效范围
        边界框[:, [0, 2]] = np.clip(边界框[:, [0, 2]], 0, 原始宽度)
        边界框[:, [1, 3]] = np.clip(边界框[:, [1, 3]], 0, 原始高度)
        
        return 边界框
    
    def 热身(self, 次数: int = 3):
        """
        模型热身，预热推理引擎
        
        参数:
            次数: 热身次数
        """
        print(f"模型热身中 ({次数}次)...")
        
        假数据 = np.random.randint(0, 255, (self.输入尺寸, self.输入尺寸, 3), dtype=np.uint8)
        
        for i in range(次数):
            _ = self.检测(假数据)
        
        print("模型热身完成")


def 创建YOLOv5检测器(
    模型路径: str,
    类别列表: List[str] = None,
    输入尺寸: int = 640,
    置信度阈值: float = 0.5,
    NMS阈值: float = 0.45,
    设备: str = 'cuda'
) -> YOLOv5检测器:
    """
    创建YOLOv5检测器的工厂函数
    
    参数:
        模型路径: 模型权重文件路径
        类别列表: 检测类别名称列表
        输入尺寸: 输入图像尺寸
        置信度阈值: 置信度过滤阈值
        NMS阈值: NMS阈值
        设备: 运行设备
    
    返回:
        YOLOv5检测器实例
    """
    return YOLOv5检测器(
        模型路径=模型路径,
        类别列表=类别列表,
        输入尺寸=输入尺寸,
        置信度阈值=置信度阈值,
        NMS阈值=NMS阈值,
        设备=设备
    )
