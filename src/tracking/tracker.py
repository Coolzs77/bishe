#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标跟踪器基类
定义跟踪器的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class 跟踪目标:
    """
    跟踪目标数据类
    """
    ID: int                          # 目标唯一ID
    边界框: np.ndarray               # [x1, y1, x2, y2]
    置信度: float                    # 检测置信度
    类别: int                        # 类别ID
    状态: str = 'tentative'          # 状态: tentative/confirmed/deleted
    命中次数: int = 0                # 连续命中次数
    消失帧数: int = 0                # 连续消失帧数
    特征: np.ndarray = None          # 外观特征
    速度: np.ndarray = None          # 速度估计 [vx, vy]
    轨迹历史: List[Tuple[float, float]] = field(default_factory=list)  # 轨迹历史
    
    def 获取中心点(self) -> Tuple[float, float]:
        """获取边界框中心点"""
        return (
            (self.边界框[0] + self.边界框[2]) / 2,
            (self.边界框[1] + self.边界框[3]) / 2
        )
    
    def 获取宽高(self) -> Tuple[float, float]:
        """获取边界框宽高"""
        return (
            self.边界框[2] - self.边界框[0],
            self.边界框[3] - self.边界框[1]
        )
    
    def 更新轨迹(self):
        """更新轨迹历史"""
        中心 = self.获取中心点()
        self.轨迹历史.append(中心)
        
        # 限制轨迹长度
        if len(self.轨迹历史) > 100:
            self.轨迹历史 = self.轨迹历史[-100:]


@dataclass
class 跟踪结果:
    """
    跟踪结果数据类
    """
    目标列表: List[跟踪目标]         # 当前帧的跟踪目标
    帧ID: int = 0                    # 当前帧编号
    
    def __len__(self) -> int:
        return len(self.目标列表)
    
    def __getitem__(self, idx: int) -> 跟踪目标:
        return self.目标列表[idx]
    
    def __iter__(self):
        return iter(self.目标列表)
    
    def 获取边界框数组(self) -> np.ndarray:
        """获取所有边界框"""
        if len(self.目标列表) == 0:
            return np.array([]).reshape(0, 4)
        return np.array([目标.边界框 for 目标 in self.目标列表])
    
    def 获取ID数组(self) -> np.ndarray:
        """获取所有ID"""
        if len(self.目标列表) == 0:
            return np.array([], dtype=np.int64)
        return np.array([目标.ID for 目标 in self.目标列表], dtype=np.int64)
    
    def 获取类别数组(self) -> np.ndarray:
        """获取所有类别"""
        if len(self.目标列表) == 0:
            return np.array([], dtype=np.int64)
        return np.array([目标.类别 for 目标 in self.目标列表], dtype=np.int64)
    
    def 获取已确认目标(self) -> '跟踪结果':
        """获取已确认的目标"""
        确认目标 = [目标 for 目标 in self.目标列表 if 目标.状态 == 'confirmed']
        return 跟踪结果(目标列表=确认目标, 帧ID=self.帧ID)
    
    def 转换为字典列表(self) -> List[Dict]:
        """转换为字典列表"""
        return [
            {
                'id': 目标.ID,
                'bbox': 目标.边界框.tolist(),
                'confidence': 目标.置信度,
                'class': 目标.类别,
                'state': 目标.状态,
            }
            for 目标 in self.目标列表
        ]


class 跟踪器基类(ABC):
    """
    多目标跟踪器的抽象基类
    """
    
    def __init__(
        self,
        最大消失帧数: int = 30,
        最小确认命中: int = 3,
        IoU阈值: float = 0.3,
    ):
        """
        初始化跟踪器
        
        参数:
            最大消失帧数: 目标消失后保留的最大帧数
            最小确认命中: 确认轨迹所需的最小连续命中次数
            IoU阈值: IoU匹配阈值
        """
        self.最大消失帧数 = 最大消失帧数
        self.最小确认命中 = 最小确认命中
        self.IoU阈值 = IoU阈值
        
        self.下一个ID = 1
        self.轨迹列表: List[跟踪目标] = []
        self.帧计数 = 0
    
    @abstractmethod
    def 更新(
        self,
        检测框: np.ndarray,
        检测置信度: np.ndarray,
        检测类别: np.ndarray,
        图像: np.ndarray = None
    ) -> 跟踪结果:
        """
        更新跟踪器状态
        
        参数:
            检测框: [N, 4] 检测边界框 (x1, y1, x2, y2)
            检测置信度: [N] 检测置信度
            检测类别: [N] 检测类别
            图像: 当前帧图像（用于提取外观特征）
        
        返回:
            跟踪结果对象
        """
        pass
    
    def 重置(self):
        """重置跟踪器状态"""
        self.下一个ID = 1
        self.轨迹列表.clear()
        self.帧计数 = 0
    
    def 分配新ID(self) -> int:
        """分配新的目标ID"""
        新ID = self.下一个ID
        self.下一个ID += 1
        return 新ID
    
    def 获取活跃轨迹数(self) -> int:
        """获取当前活跃轨迹数量"""
        return len([轨迹 for 轨迹 in self.轨迹列表 if 轨迹.状态 != 'deleted'])
    
    def 获取确认轨迹数(self) -> int:
        """获取已确认轨迹数量"""
        return len([轨迹 for 轨迹 in self.轨迹列表 if 轨迹.状态 == 'confirmed'])
    
    @staticmethod
    def 计算IoU矩阵(框组1: np.ndarray, 框组2: np.ndarray) -> np.ndarray:
        """
        计算两组边界框的IoU矩阵
        
        参数:
            框组1: [N, 4] 边界框
            框组2: [M, 4] 边界框
        
        返回:
            [N, M] IoU矩阵
        """
        N = len(框组1)
        M = len(框组2)
        
        if N == 0 or M == 0:
            return np.zeros((N, M))
        
        # 扩展维度
        框组1 = 框组1[:, np.newaxis, :]  # [N, 1, 4]
        框组2 = 框组2[np.newaxis, :, :]  # [1, M, 4]
        
        # 计算交集
        左上 = np.maximum(框组1[:, :, :2], 框组2[:, :, :2])
        右下 = np.minimum(框组1[:, :, 2:], 框组2[:, :, 2:])
        
        交集尺寸 = np.maximum(右下 - 左上, 0)
        交集面积 = 交集尺寸[:, :, 0] * 交集尺寸[:, :, 1]
        
        # 计算各自面积
        面积1 = (框组1[:, :, 2] - 框组1[:, :, 0]) * (框组1[:, :, 3] - 框组1[:, :, 1])
        面积2 = (框组2[:, :, 2] - 框组2[:, :, 0]) * (框组2[:, :, 3] - 框组2[:, :, 1])
        
        # 计算IoU
        并集面积 = 面积1 + 面积2 - 交集面积
        并集面积 = np.maximum(并集面积, 1e-10)
        
        return 交集面积 / 并集面积
    
    @staticmethod
    def 线性分配(代价矩阵: np.ndarray, 阈值: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        使用匈牙利算法进行线性分配
        
        参数:
            代价矩阵: [N, M] 代价矩阵
            阈值: 最大代价阈值
        
        返回:
            匹配对列表, 未匹配的行索引列表, 未匹配的列索引列表
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # 简单贪婪匹配作为备选
            return 跟踪器基类._贪婪匹配(代价矩阵, 阈值)
        
        if 代价矩阵.size == 0:
            return [], list(range(代价矩阵.shape[0])), list(range(代价矩阵.shape[1]))
        
        行索引, 列索引 = linear_sum_assignment(代价矩阵)
        
        匹配对 = []
        未匹配行 = set(range(代价矩阵.shape[0]))
        未匹配列 = set(range(代价矩阵.shape[1]))
        
        for 行, 列 in zip(行索引, 列索引):
            if 代价矩阵[行, 列] <= 阈值:
                匹配对.append((行, 列))
                未匹配行.discard(行)
                未匹配列.discard(列)
        
        return 匹配对, list(未匹配行), list(未匹配列)
    
    @staticmethod
    def _贪婪匹配(代价矩阵: np.ndarray, 阈值: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """贪婪匹配（备选方案）"""
        匹配对 = []
        已匹配行 = set()
        已匹配列 = set()
        
        # 找所有有效的匹配候选
        候选 = []
        for i in range(代价矩阵.shape[0]):
            for j in range(代价矩阵.shape[1]):
                if 代价矩阵[i, j] <= 阈值:
                    候选.append((代价矩阵[i, j], i, j))
        
        # 按代价排序
        候选.sort()
        
        # 贪婪选择
        for 代价, 行, 列 in 候选:
            if 行 not in 已匹配行 and 列 not in 已匹配列:
                匹配对.append((行, 列))
                已匹配行.add(行)
                已匹配列.add(列)
        
        未匹配行 = [i for i in range(代价矩阵.shape[0]) if i not in 已匹配行]
        未匹配列 = [j for j in range(代价矩阵.shape[1]) if j not in 已匹配列]
        
        return 匹配对, 未匹配行, 未匹配列
