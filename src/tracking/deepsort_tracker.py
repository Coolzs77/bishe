#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT跟踪器实现
基于深度外观描述器的SORT跟踪算法
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from .tracker import 跟踪器基类, 跟踪目标, 跟踪结果
from .kalman_filter import 卡尔曼滤波器, xyxy转xyah, xyah转xyxy


class 轨迹:
    """
    单个目标的轨迹类
    """
    
    def __init__(
        self,
        均值: np.ndarray,
        协方差: np.ndarray,
        目标ID: int,
        类别: int,
        置信度: float,
        特征: np.ndarray = None,
        最大特征数: int = 100
    ):
        """
        初始化轨迹
        
        参数:
            均值: 卡尔曼滤波器状态均值
            协方差: 卡尔曼滤波器状态协方差
            目标ID: 目标唯一ID
            类别: 目标类别
            置信度: 检测置信度
            特征: 外观特征向量
            最大特征数: 保存的最大特征数量
        """
        self.均值 = 均值
        self.协方差 = 协方差
        self.目标ID = 目标ID
        self.类别 = 类别
        self.置信度 = 置信度
        self.最大特征数 = 最大特征数
        
        self.命中次数 = 1
        self.消失帧数 = 0
        self.状态 = 'tentative'  # tentative/confirmed/deleted
        
        self.特征库 = []
        if 特征 is not None:
            self.特征库.append(特征)
        
        self.卡尔曼滤波器 = 卡尔曼滤波器()
    
    def 预测(self):
        """执行卡尔曼预测"""
        self.均值, self.协方差 = self.卡尔曼滤波器.预测(self.均值, self.协方差)
        self.消失帧数 += 1
    
    def 更新(self, 观测: np.ndarray, 特征: np.ndarray = None, 置信度: float = 1.0):
        """
        更新轨迹
        
        参数:
            观测: [cx, cy, a, h] 格式的观测
            特征: 外观特征向量
            置信度: 检测置信度
        """
        self.均值, self.协方差 = self.卡尔曼滤波器.更新(self.均值, self.协方差, 观测)
        
        self.命中次数 += 1
        self.消失帧数 = 0
        self.置信度 = 置信度
        
        # 更新特征库
        if 特征 is not None:
            self.特征库.append(特征)
            if len(self.特征库) > self.最大特征数:
                self.特征库 = self.特征库[-self.最大特征数:]
        
        # 更新状态
        if self.状态 == 'tentative' and self.命中次数 >= 3:
            self.状态 = 'confirmed'
    
    def 标记删除(self):
        """标记轨迹为删除状态"""
        self.状态 = 'deleted'
    
    def 获取边界框(self) -> np.ndarray:
        """获取当前边界框 [x1, y1, x2, y2]"""
        return xyah转xyxy(self.均值[:4])
    
    def 转换为跟踪目标(self) -> 跟踪目标:
        """转换为跟踪目标对象"""
        return 跟踪目标(
            ID=self.目标ID,
            边界框=self.获取边界框(),
            置信度=self.置信度,
            类别=self.类别,
            状态=self.状态,
            命中次数=self.命中次数,
            消失帧数=self.消失帧数,
            特征=self.特征库[-1] if self.特征库 else None,
        )


class DeepSORT跟踪器(跟踪器基类):
    """
    DeepSORT多目标跟踪器
    
    结合卡尔曼滤波和深度外观特征进行数据关联
    """
    
    def __init__(
        self,
        Reid模型路径: str = None,
        最大余弦距离: float = 0.2,
        最大IoU距离: float = 0.7,
        最大消失帧数: int = 70,
        最小确认命中: int = 3,
        特征库容量: int = 100
    ):
        """
        初始化DeepSORT跟踪器
        
        参数:
            Reid模型路径: ReID模型路径（可选）
            最大余弦距离: 外观特征最大余弦距离
            最大IoU距离: 最大IoU距离
            最大消失帧数: 最大消失帧数
            最小确认命中: 最小确认命中次数
            特征库容量: 每个轨迹的特征库容量
        """
        super().__init__(最大消失帧数, 最小确认命中, 1 - 最大IoU距离)
        
        self.最大余弦距离 = 最大余弦距离
        self.最大IoU距离 = 最大IoU距离
        self.特征库容量 = 特征库容量
        
        self.轨迹列表: List[轨迹] = []
        self.卡尔曼滤波器 = 卡尔曼滤波器()
        
        # 加载ReID模型
        self.Reid模型 = None
        if Reid模型路径 and Path(Reid模型路径).exists():
            self._加载Reid模型(Reid模型路径)
    
    def _加载Reid模型(self, 模型路径: str):
        """加载ReID模型"""
        try:
            import onnxruntime as ort
            
            self.Reid模型 = ort.InferenceSession(
                模型路径,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.Reid输入名 = self.Reid模型.get_inputs()[0].name
            print(f"ReID模型加载成功: {模型路径}")
            
        except Exception as e:
            print(f"加载ReID模型失败: {e}")
            self.Reid模型 = None
    
    def 提取特征(self, 图像: np.ndarray, 边界框列表: np.ndarray) -> List[np.ndarray]:
        """
        提取目标的外观特征
        
        参数:
            图像: 原始图像
            边界框列表: [N, 4] 边界框列表
        
        返回:
            特征向量列表
        """
        if self.Reid模型 is None or len(边界框列表) == 0:
            return [None] * len(边界框列表)
        
        import cv2
        
        特征列表 = []
        高, 宽 = 图像.shape[:2]
        
        for 边界框 in 边界框列表:
            x1, y1, x2, y2 = map(int, 边界框)
            
            # 裁剪边界框
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(宽, x2)
            y2 = min(高, y2)
            
            if x2 <= x1 or y2 <= y1:
                特征列表.append(None)
                continue
            
            裁剪图 = 图像[y1:y2, x1:x2]
            
            # 预处理
            裁剪图 = cv2.resize(裁剪图, (64, 128))
            裁剪图 = cv2.cvtColor(裁剪图, cv2.COLOR_BGR2RGB)
            裁剪图 = 裁剪图.astype(np.float32) / 255.0
            裁剪图 = np.transpose(裁剪图, (2, 0, 1))
            裁剪图 = np.expand_dims(裁剪图, 0)
            
            # 推理
            特征 = self.Reid模型.run(None, {self.Reid输入名: 裁剪图})[0]
            特征 = 特征.flatten()
            特征 = 特征 / np.linalg.norm(特征)  # L2归一化
            
            特征列表.append(特征)
        
        return 特征列表
    
    def 计算外观代价矩阵(
        self, 
        轨迹列表: List[轨迹], 
        检测特征列表: List[np.ndarray]
    ) -> np.ndarray:
        """
        计算外观特征代价矩阵
        
        参数:
            轨迹列表: 轨迹列表
            检测特征列表: 检测特征列表
        
        返回:
            [N, M] 代价矩阵
        """
        代价矩阵 = np.zeros((len(轨迹列表), len(检测特征列表)))
        
        for i, 轨迹 in enumerate(轨迹列表):
            if not 轨迹.特征库:
                代价矩阵[i, :] = self.最大余弦距离
                continue
            
            # 轨迹特征矩阵
            轨迹特征 = np.array(轨迹.特征库)
            
            for j, 检测特征 in enumerate(检测特征列表):
                if 检测特征 is None:
                    代价矩阵[i, j] = self.最大余弦距离
                    continue
                
                # 计算余弦距离
                相似度 = 轨迹特征 @ 检测特征
                代价矩阵[i, j] = 1 - np.max(相似度)
        
        return 代价矩阵
    
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
        
        # 提取外观特征
        特征列表 = self.提取特征(图像, 检测框) if 图像 is not None else [None] * len(检测框)
        
        # 预测所有轨迹
        for 轨迹 in self.轨迹列表:
            轨迹.预测()
        
        # 分离确认轨迹和未确认轨迹
        确认轨迹 = [t for t in self.轨迹列表 if t.状态 == 'confirmed']
        未确认轨迹 = [t for t in self.轨迹列表 if t.状态 == 'tentative']
        
        # 第一次关联：使用外观特征匹配确认轨迹
        未匹配检测索引 = list(range(len(检测框)))
        未匹配轨迹索引 = list(range(len(确认轨迹)))
        
        if 确认轨迹 and len(检测框) > 0:
            # 计算外观代价矩阵
            外观代价 = self.计算外观代价矩阵(确认轨迹, 特征列表)
            
            # 计算IoU代价矩阵
            轨迹边界框 = np.array([t.获取边界框() for t in 确认轨迹])
            IoU矩阵 = self.计算IoU矩阵(轨迹边界框, 检测框)
            IoU代价 = 1 - IoU矩阵
            
            # 门控
            外观代价[外观代价 > self.最大余弦距离] = 1e5
            IoU代价[IoU代价 > self.最大IoU距离] = 1e5
            
            # 联合代价
            代价矩阵 = 0.5 * 外观代价 + 0.5 * IoU代价
            
            # 线性分配
            匹配对, 未匹配轨迹索引, 未匹配检测索引 = self.线性分配(
                代价矩阵, self.最大余弦距离
            )
            
            # 更新匹配的轨迹
            for 轨迹索引, 检测索引 in 匹配对:
                观测 = xyxy转xyah(检测框[检测索引])
                确认轨迹[轨迹索引].更新(
                    观测, 
                    特征列表[检测索引],
                    检测置信度[检测索引]
                )
        
        # 第二次关联：使用IoU匹配剩余轨迹（包括未确认轨迹）
        剩余轨迹 = [确认轨迹[i] for i in 未匹配轨迹索引] + 未确认轨迹
        剩余检测索引 = 未匹配检测索引
        
        if 剩余轨迹 and 剩余检测索引:
            轨迹边界框 = np.array([t.获取边界框() for t in 剩余轨迹])
            剩余检测框 = 检测框[剩余检测索引]
            
            IoU矩阵 = self.计算IoU矩阵(轨迹边界框, 剩余检测框)
            IoU代价 = 1 - IoU矩阵
            
            匹配对, 未匹配轨迹索引2, 未匹配检测索引2 = self.线性分配(
                IoU代价, self.最大IoU距离
            )
            
            # 更新匹配的轨迹
            for 轨迹索引, 检测索引 in 匹配对:
                实际检测索引 = 剩余检测索引[检测索引]
                观测 = xyxy转xyah(检测框[实际检测索引])
                剩余轨迹[轨迹索引].更新(
                    观测,
                    特征列表[实际检测索引],
                    检测置信度[实际检测索引]
                )
            
            # 更新未匹配检测索引
            未匹配检测索引 = [剩余检测索引[i] for i in 未匹配检测索引2]
            
            # 标记长时间未匹配的轨迹
            for i in 未匹配轨迹索引2:
                if 剩余轨迹[i].消失帧数 > self.最大消失帧数:
                    剩余轨迹[i].标记删除()
        
        # 为未匹配的检测创建新轨迹
        for 检测索引 in 未匹配检测索引:
            观测 = xyxy转xyah(检测框[检测索引])
            均值, 协方差 = self.卡尔曼滤波器.初始化(观测)
            
            新轨迹 = 轨迹(
                均值=均值,
                协方差=协方差,
                目标ID=self.分配新ID(),
                类别=int(检测类别[检测索引]),
                置信度=float(检测置信度[检测索引]),
                特征=特征列表[检测索引],
                最大特征数=self.特征库容量
            )
            self.轨迹列表.append(新轨迹)
        
        # 删除标记为删除的轨迹
        self.轨迹列表 = [t for t in self.轨迹列表 if t.状态 != 'deleted']
        
        # 标记长时间消失的轨迹为删除
        for 轨迹 in self.轨迹列表:
            if 轨迹.消失帧数 > self.最大消失帧数:
                轨迹.标记删除()
        
        # 构建跟踪结果
        目标列表 = [轨迹.转换为跟踪目标() for 轨迹 in self.轨迹列表 if 轨迹.状态 == 'confirmed']
        
        return 跟踪结果(目标列表=目标列表, 帧ID=self.帧计数)
    
    def 重置(self):
        """重置跟踪器"""
        super().重置()
        self.轨迹列表.clear()


def 创建DeepSORT跟踪器(
    Reid模型路径: str = None,
    最大余弦距离: float = 0.2,
    最大IoU距离: float = 0.7,
    最大消失帧数: int = 70
) -> DeepSORT跟踪器:
    """
    创建DeepSORT跟踪器的工厂函数
    """
    return DeepSORT跟踪器(
        Reid模型路径=Reid模型路径,
        最大余弦距离=最大余弦距离,
        最大IoU距离=最大IoU距离,
        最大消失帧数=最大消失帧数
    )
