#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ByteTrack跟踪器实现
基于字节关联的多目标跟踪算法
"""

import numpy as np
from typing import List, Tuple

from .tracker import 跟踪器基类, 跟踪目标, 跟踪结果
from .kalman_filter import 卡尔曼滤波器, xyxy转xyah, xyah转xyxy


class STrack:
    """
    ByteTrack中的单个轨迹类
    """
    
    共享卡尔曼 = 卡尔曼滤波器()
    
    def __init__(self, 边界框: np.ndarray, 置信度: float, 类别: int):
        """
        初始化轨迹
        
        参数:
            边界框: [x1, y1, x2, y2]
            置信度: 检测置信度
            类别: 目标类别
        """
        self.边界框 = 边界框
        self.置信度 = 置信度
        self.类别 = 类别
        
        self.均值 = None
        self.协方差 = None
        
        self.目标ID = 0
        self.轨迹长度 = 0
        self.消失帧数 = 0
        self.开始帧 = 0
        self.当前帧 = 0
        self.是激活 = False
        self.状态 = 'new'  # new/tracked/lost/removed
    
    def 激活(self, 新ID: int, 帧ID: int):
        """激活新轨迹"""
        self.目标ID = 新ID
        self.轨迹长度 = 0
        self.开始帧 = 帧ID
        self.当前帧 = 帧ID
        self.是激活 = True
        self.状态 = 'tracked'
        
        # 初始化卡尔曼滤波器
        观测 = xyxy转xyah(self.边界框)
        self.均值, self.协方差 = self.共享卡尔曼.初始化(观测)
    
    def 重新激活(self, 新轨迹: 'STrack', 帧ID: int, 新ID: bool = False):
        """重新激活丢失的轨迹"""
        观测 = xyxy转xyah(新轨迹.边界框)
        self.均值, self.协方差 = self.共享卡尔曼.更新(self.均值, self.协方差, 观测)
        
        self.轨迹长度 = 0
        self.消失帧数 = 0
        self.当前帧 = 帧ID
        self.是激活 = True
        self.状态 = 'tracked'
        self.置信度 = 新轨迹.置信度
        
        if 新ID:
            self.目标ID = 新轨迹.目标ID
    
    def 预测(self):
        """卡尔曼预测"""
        if self.均值 is not None:
            self.均值, self.协方差 = self.共享卡尔曼.预测(self.均值, self.协方差)
    
    def 更新(self, 新轨迹: 'STrack', 帧ID: int):
        """更新轨迹"""
        self.当前帧 = 帧ID
        self.轨迹长度 += 1
        self.消失帧数 = 0
        self.是激活 = True
        self.状态 = 'tracked'
        self.置信度 = 新轨迹.置信度
        
        观测 = xyxy转xyah(新轨迹.边界框)
        self.均值, self.协方差 = self.共享卡尔曼.更新(self.均值, self.协方差, 观测)
    
    def 标记丢失(self):
        """标记为丢失"""
        self.状态 = 'lost'
        self.消失帧数 += 1
    
    def 标记移除(self):
        """标记为移除"""
        self.状态 = 'removed'
    
    def 获取边界框(self) -> np.ndarray:
        """获取当前边界框"""
        if self.均值 is not None:
            return xyah转xyxy(self.均值[:4])
        return self.边界框
    
    @staticmethod
    def 多轨迹预测(轨迹列表: List['STrack']):
        """批量预测多个轨迹"""
        for 轨迹 in 轨迹列表:
            轨迹.预测()
    
    def 转换为跟踪目标(self) -> 跟踪目标:
        """转换为跟踪目标对象"""
        return 跟踪目标(
            ID=self.目标ID,
            边界框=self.获取边界框(),
            置信度=self.置信度,
            类别=self.类别,
            状态='confirmed' if self.状态 == 'tracked' else self.状态,
            命中次数=self.轨迹长度,
            消失帧数=self.消失帧数,
        )


class ByteTrack跟踪器(跟踪器基类):
    """
    ByteTrack多目标跟踪器
    
    特点：
    1. 使用两阶段关联策略
    2. 利用低置信度检测进行二次关联
    3. 不依赖外观特征，纯运动信息
    """
    
    def __init__(
        self,
        高置信度阈值: float = 0.5,
        低置信度阈值: float = 0.1,
        新轨迹阈值: float = 0.6,
        匹配IoU阈值: float = 0.8,
        轨迹缓冲: int = 30,
        最大消失帧数: int = 30
    ):
        """
        初始化ByteTrack跟踪器
        
        参数:
            高置信度阈值: 高置信度检测阈值
            低置信度阈值: 低置信度检测阈值
            新轨迹阈值: 创建新轨迹的置信度阈值
            匹配IoU阈值: IoU匹配阈值
            轨迹缓冲: 丢失轨迹保留的帧数
            最大消失帧数: 最大消失帧数
        """
        super().__init__(最大消失帧数, 1, 1 - 匹配IoU阈值)
        
        self.高置信度阈值 = 高置信度阈值
        self.低置信度阈值 = 低置信度阈值
        self.新轨迹阈值 = 新轨迹阈值
        self.匹配IoU阈值 = 匹配IoU阈值
        self.轨迹缓冲 = 轨迹缓冲
        
        self.已跟踪轨迹: List[STrack] = []
        self.丢失轨迹: List[STrack] = []
        self.移除轨迹: List[STrack] = []
    
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
            图像: 当前帧图像（ByteTrack不使用）
        
        返回:
            跟踪结果
        """
        self.帧计数 += 1
        
        # 创建检测轨迹
        检测轨迹列表 = []
        for i in range(len(检测框)):
            轨迹 = STrack(检测框[i], 检测置信度[i], int(检测类别[i]))
            检测轨迹列表.append(轨迹)
        
        # 分离高低置信度检测
        高置信度检测 = [d for d in 检测轨迹列表 if d.置信度 >= self.高置信度阈值]
        低置信度检测 = [d for d in 检测轨迹列表 
                      if self.低置信度阈值 <= d.置信度 < self.高置信度阈值]
        
        # 合并已跟踪和丢失轨迹
        轨迹池 = self.已跟踪轨迹 + self.丢失轨迹
        
        # 预测所有轨迹
        STrack.多轨迹预测(轨迹池)
        
        # 第一次关联：高置信度检测与所有轨迹
        已激活轨迹 = []
        重找到轨迹 = []
        未匹配轨迹 = []
        未匹配检测 = []
        
        if 高置信度检测 and 轨迹池:
            # 计算IoU代价矩阵
            轨迹边界框 = np.array([t.获取边界框() for t in 轨迹池])
            检测边界框 = np.array([d.边界框 for d in 高置信度检测])
            
            IoU矩阵 = self.计算IoU矩阵(轨迹边界框, 检测边界框)
            代价矩阵 = 1 - IoU矩阵
            
            匹配对, 未匹配轨迹索引, 未匹配检测索引 = self.线性分配(
                代价矩阵, 1 - self.匹配IoU阈值
            )
            
            # 处理匹配
            for 轨迹索引, 检测索引 in 匹配对:
                轨迹 = 轨迹池[轨迹索引]
                检测 = 高置信度检测[检测索引]
                
                if 轨迹.状态 == 'tracked':
                    轨迹.更新(检测, self.帧计数)
                    已激活轨迹.append(轨迹)
                else:
                    轨迹.重新激活(检测, self.帧计数)
                    重找到轨迹.append(轨迹)
            
            未匹配轨迹 = [轨迹池[i] for i in 未匹配轨迹索引]
            未匹配检测 = [高置信度检测[i] for i in 未匹配检测索引]
        else:
            未匹配轨迹 = 轨迹池
            未匹配检测 = 高置信度检测
        
        # 第二次关联：低置信度检测与剩余的已跟踪轨迹
        剩余已跟踪轨迹 = [t for t in 未匹配轨迹 if t.状态 == 'tracked']
        
        if 低置信度检测 and 剩余已跟踪轨迹:
            轨迹边界框 = np.array([t.获取边界框() for t in 剩余已跟踪轨迹])
            检测边界框 = np.array([d.边界框 for d in 低置信度检测])
            
            IoU矩阵 = self.计算IoU矩阵(轨迹边界框, 检测边界框)
            代价矩阵 = 1 - IoU矩阵
            
            匹配对, 未匹配轨迹索引2, _ = self.线性分配(代价矩阵, 0.5)
            
            for 轨迹索引, 检测索引 in 匹配对:
                轨迹 = 剩余已跟踪轨迹[轨迹索引]
                检测 = 低置信度检测[检测索引]
                轨迹.更新(检测, self.帧计数)
                已激活轨迹.append(轨迹)
            
            # 未匹配的标记为丢失
            for i in 未匹配轨迹索引2:
                轨迹 = 剩余已跟踪轨迹[i]
                if 轨迹.状态 != 'lost':
                    轨迹.标记丢失()
        else:
            for 轨迹 in 剩余已跟踪轨迹:
                if 轨迹.状态 != 'lost':
                    轨迹.标记丢失()
        
        # 处理剩余的丢失轨迹
        剩余丢失轨迹 = [t for t in 未匹配轨迹 if t.状态 == 'lost']
        for 轨迹 in 剩余丢失轨迹:
            if self.帧计数 - 轨迹.当前帧 > self.轨迹缓冲:
                轨迹.标记移除()
        
        # 为未匹配的高置信度检测创建新轨迹
        for 检测 in 未匹配检测:
            if 检测.置信度 >= self.新轨迹阈值:
                检测.激活(self.分配新ID(), self.帧计数)
                已激活轨迹.append(检测)
        
        # 更新轨迹列表
        self.已跟踪轨迹 = [t for t in 已激活轨迹 + 重找到轨迹 if t.状态 == 'tracked']
        self.丢失轨迹 = [t for t in self.丢失轨迹 + 剩余已跟踪轨迹 + 剩余丢失轨迹 
                       if t.状态 == 'lost']
        self.移除轨迹.extend([t for t in self.丢失轨迹 if t.状态 == 'removed'])
        self.丢失轨迹 = [t for t in self.丢失轨迹 if t.状态 != 'removed']
        
        # 构建输出
        输出轨迹 = self.已跟踪轨迹
        目标列表 = [轨迹.转换为跟踪目标() for 轨迹 in 输出轨迹]
        
        return 跟踪结果(目标列表=目标列表, 帧ID=self.帧计数)
    
    def 重置(self):
        """重置跟踪器"""
        super().重置()
        self.已跟踪轨迹.clear()
        self.丢失轨迹.clear()
        self.移除轨迹.clear()


def 创建ByteTrack跟踪器(
    高置信度阈值: float = 0.5,
    低置信度阈值: float = 0.1,
    匹配IoU阈值: float = 0.8,
    轨迹缓冲: int = 30
) -> ByteTrack跟踪器:
    """
    创建ByteTrack跟踪器的工厂函数
    """
    return ByteTrack跟踪器(
        高置信度阈值=高置信度阈值,
        低置信度阈值=低置信度阈值,
        匹配IoU阈值=匹配IoU阈值,
        轨迹缓冲=轨迹缓冲
    )
