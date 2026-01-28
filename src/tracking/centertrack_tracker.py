#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CenterTrack跟踪器实现
基于中心点偏移的端到端跟踪
"""

import numpy as np
from typing import List, Dict, Optional

from .tracker import 跟踪器基类, 跟踪目标, 跟踪结果


class CenterTrack跟踪器(跟踪器基类):
    """
    CenterTrack多目标跟踪器
    
    基于CenterNet检测器的端到端跟踪方法
    直接预测目标的运动偏移，不需要单独的关联步骤
    """
    
    def __init__(
        self,
        模型路径: str = None,
        预测阈值: float = 0.4,
        新目标阈值: float = 0.4,
        最大消失帧数: int = 32,
        设备: str = 'cuda'
    ):
        """
        初始化CenterTrack跟踪器
        
        参数:
            模型路径: CenterTrack模型路径
            预测阈值: 检测置信度阈值
            新目标阈值: 新目标置信度阈值
            最大消失帧数: 最大消失帧数
            设备: 运行设备
        """
        super().__init__(最大消失帧数, 1, 0.3)
        
        self.模型路径 = 模型路径
        self.预测阈值 = 预测阈值
        self.新目标阈值 = 新目标阈值
        self.设备 = 设备
        
        self.模型 = None
        self.上一帧图像 = None
        self.上一帧热图 = None
        self.当前轨迹: Dict[int, 跟踪目标] = {}
        
        if 模型路径:
            self._加载模型()
    
    def _加载模型(self):
        """加载CenterTrack模型"""
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.设备 == 'cpu':
                providers = ['CPUExecutionProvider']
            
            self.模型 = ort.InferenceSession(self.模型路径, providers=providers)
            
            self.输入名称列表 = [inp.name for inp in self.模型.get_inputs()]
            self.输出名称列表 = [out.name for out in self.模型.get_outputs()]
            
            print(f"CenterTrack模型加载成功: {self.模型路径}")
            
        except Exception as e:
            print(f"加载CenterTrack模型失败: {e}")
            self.模型 = None
    
    def 预处理(self, 图像: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        参数:
            图像: BGR格式的输入图像
        
        返回:
            预处理后的图像
        """
        import cv2
        
        # 调整大小
        输入图像 = cv2.resize(图像, (960, 544))
        
        # BGR转RGB并归一化
        输入图像 = cv2.cvtColor(输入图像, cv2.COLOR_BGR2RGB)
        输入图像 = 输入图像.astype(np.float32) / 255.0
        
        # 标准化
        均值 = np.array([0.485, 0.456, 0.406])
        标准差 = np.array([0.229, 0.224, 0.225])
        输入图像 = (输入图像 - 均值) / 标准差
        
        # HWC -> CHW -> NCHW
        输入图像 = np.transpose(输入图像, (2, 0, 1))
        输入图像 = np.expand_dims(输入图像, 0)
        
        return 输入图像.astype(np.float32)
    
    def 解码输出(
        self, 
        热图: np.ndarray, 
        宽高: np.ndarray, 
        偏移: np.ndarray,
        跟踪偏移: np.ndarray,
        原始尺寸: tuple
    ) -> List[Dict]:
        """
        解码模型输出
        
        参数:
            热图: [1, C, H, W] 检测热图
            宽高: [1, 2, H, W] 宽高预测
            偏移: [1, 2, H, W] 中心点偏移
            跟踪偏移: [1, 2, H, W] 跟踪偏移
            原始尺寸: 原始图像尺寸 (高, 宽)
        
        返回:
            检测结果列表
        """
        # 移除batch维度
        热图 = 热图[0]  # [C, H, W]
        
        # 应用sigmoid
        热图 = 1 / (1 + np.exp(-热图))
        
        # 找到峰值
        C, H, W = 热图.shape
        检测列表 = []
        
        for c in range(C):
            类别热图 = 热图[c]
            
            # 简单的局部最大值检测
            from scipy.ndimage import maximum_filter
            最大值图 = maximum_filter(类别热图, size=3)
            峰值掩码 = (类别热图 == 最大值图) & (类别热图 >= self.预测阈值)
            
            峰值位置 = np.where(峰值掩码)
            
            for y, x in zip(峰值位置[0], 峰值位置[1]):
                分数 = 类别热图[y, x]
                
                # 获取宽高
                w = 宽高[0, 0, y, x]
                h = 宽高[0, 1, y, x]
                
                # 获取偏移
                ox = 偏移[0, 0, y, x]
                oy = 偏移[0, 1, y, x]
                
                # 获取跟踪偏移
                tx = 跟踪偏移[0, 0, y, x] if 跟踪偏移 is not None else 0
                ty = 跟踪偏移[0, 1, y, x] if 跟踪偏移 is not None else 0
                
                # 计算中心点
                cx = (x + ox) * 4  # 下采样比例
                cy = (y + oy) * 4
                
                # 映射到原始尺寸
                缩放x = 原始尺寸[1] / 960
                缩放y = 原始尺寸[0] / 544
                
                cx *= 缩放x
                cy *= 缩放y
                w *= 缩放x * 4
                h *= 缩放y * 4
                
                检测列表.append({
                    'center': (cx, cy),
                    'size': (w, h),
                    'score': float(分数),
                    'class': c,
                    'tracking_offset': (tx * 缩放x * 4, ty * 缩放y * 4),
                })
        
        return 检测列表
    
    def 关联检测(self, 检测列表: List[Dict]) -> List[跟踪目标]:
        """
        使用跟踪偏移关联检测
        
        参数:
            检测列表: 当前帧检测结果
        
        返回:
            跟踪目标列表
        """
        跟踪结果列表 = []
        已匹配ID集合 = set()
        
        for 检测 in 检测列表:
            cx, cy = 检测['center']
            w, h = 检测['size']
            tx, ty = 检测['tracking_offset']
            
            # 计算上一帧的预期位置
            上帧cx = cx - tx
            上帧cy = cy - ty
            
            # 在当前轨迹中寻找匹配
            最佳匹配ID = None
            最小距离 = float('inf')
            
            for ID, 轨迹 in self.当前轨迹.items():
                if ID in 已匹配ID集合:
                    continue
                
                轨迹cx, 轨迹cy = 轨迹.获取中心点()
                
                # 计算距离
                距离 = np.sqrt((上帧cx - 轨迹cx)**2 + (上帧cy - 轨迹cy)**2)
                
                # 简单的距离阈值
                阈值 = max(轨迹.获取宽高()) * 0.5
                
                if 距离 < 阈值 and 距离 < 最小距离:
                    最小距离 = 距离
                    最佳匹配ID = ID
            
            # 计算边界框
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            边界框 = np.array([x1, y1, x2, y2])
            
            if 最佳匹配ID is not None:
                # 更新现有轨迹
                已匹配ID集合.add(最佳匹配ID)
                
                目标 = 跟踪目标(
                    ID=最佳匹配ID,
                    边界框=边界框,
                    置信度=检测['score'],
                    类别=检测['class'],
                    状态='confirmed',
                )
                跟踪结果列表.append(目标)
            elif 检测['score'] >= self.新目标阈值:
                # 创建新轨迹
                新ID = self.分配新ID()
                
                目标 = 跟踪目标(
                    ID=新ID,
                    边界框=边界框,
                    置信度=检测['score'],
                    类别=检测['class'],
                    状态='confirmed',
                )
                跟踪结果列表.append(目标)
        
        return 跟踪结果列表
    
    def 更新(
        self,
        检测框: np.ndarray,
        检测置信度: np.ndarray,
        检测类别: np.ndarray,
        图像: np.ndarray = None
    ) -> 跟踪结果:
        """
        更新跟踪器
        
        参数:
            检测框: [N, 4] 检测边界框 (x1, y1, x2, y2)
            检测置信度: [N] 检测置信度
            检测类别: [N] 检测类别
            图像: 当前帧图像
        
        返回:
            跟踪结果
        """
        self.帧计数 += 1
        
        if self.模型 is not None and 图像 is not None:
            # 使用CenterTrack模型进行跟踪
            return self._模型跟踪(图像)
        else:
            # 回退到简单的IoU跟踪
            return self._简单跟踪(检测框, 检测置信度, 检测类别)
    
    def _模型跟踪(self, 图像: np.ndarray) -> 跟踪结果:
        """使用CenterTrack模型跟踪"""
        原始尺寸 = 图像.shape[:2]
        
        # 预处理
        输入图像 = self.预处理(图像)
        
        # 准备输入
        输入字典 = {self.输入名称列表[0]: 输入图像}
        
        # 添加前一帧信息（如果可用）
        if self.上一帧图像 is not None and len(self.输入名称列表) > 1:
            输入字典[self.输入名称列表[1]] = self.预处理(self.上一帧图像)
        
        if self.上一帧热图 is not None and len(self.输入名称列表) > 2:
            输入字典[self.输入名称列表[2]] = self.上一帧热图
        
        # 推理
        输出列表 = self.模型.run(self.输出名称列表, 输入字典)
        
        # 解析输出
        热图 = 输出列表[0]
        宽高 = 输出列表[1] if len(输出列表) > 1 else None
        偏移 = 输出列表[2] if len(输出列表) > 2 else None
        跟踪偏移 = 输出列表[3] if len(输出列表) > 3 else None
        
        # 解码
        if 宽高 is not None and 偏移 is not None:
            检测列表 = self.解码输出(热图, 宽高, 偏移, 跟踪偏移, 原始尺寸)
        else:
            检测列表 = []
        
        # 关联
        跟踪目标列表 = self.关联检测(检测列表)
        
        # 更新状态
        self.上一帧图像 = 图像.copy()
        self.上一帧热图 = 热图
        self.当前轨迹 = {目标.ID: 目标 for 目标 in 跟踪目标列表}
        
        return 跟踪结果(目标列表=跟踪目标列表, 帧ID=self.帧计数)
    
    def _简单跟踪(
        self, 
        检测框: np.ndarray, 
        检测置信度: np.ndarray,
        检测类别: np.ndarray
    ) -> 跟踪结果:
        """简单的IoU跟踪（模型不可用时的回退方案）"""
        
        # 如果没有当前轨迹，为所有检测创建新轨迹
        if not self.当前轨迹:
            跟踪目标列表 = []
            for i in range(len(检测框)):
                目标 = 跟踪目标(
                    ID=self.分配新ID(),
                    边界框=检测框[i],
                    置信度=float(检测置信度[i]),
                    类别=int(检测类别[i]),
                    状态='confirmed',
                )
                跟踪目标列表.append(目标)
            
            self.当前轨迹 = {目标.ID: 目标 for 目标 in 跟踪目标列表}
            return 跟踪结果(目标列表=跟踪目标列表, 帧ID=self.帧计数)
        
        # 使用IoU匹配
        轨迹列表 = list(self.当前轨迹.values())
        轨迹边界框 = np.array([t.边界框 for t in 轨迹列表])
        
        if len(检测框) > 0:
            IoU矩阵 = self.计算IoU矩阵(轨迹边界框, 检测框)
            代价矩阵 = 1 - IoU矩阵
            
            匹配对, 未匹配轨迹, 未匹配检测 = self.线性分配(代价矩阵, 0.7)
        else:
            匹配对 = []
            未匹配轨迹 = list(range(len(轨迹列表)))
            未匹配检测 = []
        
        跟踪目标列表 = []
        
        # 更新匹配的轨迹
        for 轨迹索引, 检测索引 in 匹配对:
            旧目标 = 轨迹列表[轨迹索引]
            目标 = 跟踪目标(
                ID=旧目标.ID,
                边界框=检测框[检测索引],
                置信度=float(检测置信度[检测索引]),
                类别=int(检测类别[检测索引]),
                状态='confirmed',
            )
            跟踪目标列表.append(目标)
        
        # 为未匹配的检测创建新轨迹
        for 检测索引 in 未匹配检测:
            if 检测置信度[检测索引] >= self.新目标阈值:
                目标 = 跟踪目标(
                    ID=self.分配新ID(),
                    边界框=检测框[检测索引],
                    置信度=float(检测置信度[检测索引]),
                    类别=int(检测类别[检测索引]),
                    状态='confirmed',
                )
                跟踪目标列表.append(目标)
        
        # 更新当前轨迹
        self.当前轨迹 = {目标.ID: 目标 for 目标 in 跟踪目标列表}
        
        return 跟踪结果(目标列表=跟踪目标列表, 帧ID=self.帧计数)
    
    def 重置(self):
        """重置跟踪器"""
        super().重置()
        self.上一帧图像 = None
        self.上一帧热图 = None
        self.当前轨迹.clear()


def 创建CenterTrack跟踪器(
    模型路径: str = None,
    预测阈值: float = 0.4,
    新目标阈值: float = 0.4,
    设备: str = 'cuda'
) -> CenterTrack跟踪器:
    """
    创建CenterTrack跟踪器的工厂函数
    """
    return CenterTrack跟踪器(
        模型路径=模型路径,
        预测阈值=预测阈值,
        新目标阈值=新目标阈值,
        设备=设备
    )
