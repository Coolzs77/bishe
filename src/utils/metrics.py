#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块
包含目标检测和多目标跟踪的评估指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


def 计算IoU(框1: np.ndarray, 框2: np.ndarray) -> float:
    """
    计算两个边界框的IoU (Intersection over Union)
    
    参数:
        框1: [x1, y1, x2, y2] 格式的边界框
        框2: [x1, y1, x2, y2] 格式的边界框
    
    返回:
        IoU值 (0-1之间)
    """
    # 计算交集区域
    x1 = max(框1[0], 框2[0])
    y1 = max(框1[1], 框2[1])
    x2 = min(框1[2], 框2[2])
    y2 = min(框1[3], 框2[3])
    
    # 计算交集面积
    交集宽 = max(0, x2 - x1)
    交集高 = max(0, y2 - y1)
    交集面积 = 交集宽 * 交集高
    
    # 计算并集面积
    框1面积 = (框1[2] - 框1[0]) * (框1[3] - 框1[1])
    框2面积 = (框2[2] - 框2[0]) * (框2[3] - 框2[1])
    并集面积 = 框1面积 + 框2面积 - 交集面积
    
    # 计算IoU
    if 并集面积 == 0:
        return 0.0
    
    return 交集面积 / 并集面积


def 计算批量IoU(框组1: np.ndarray, 框组2: np.ndarray) -> np.ndarray:
    """
    批量计算两组边界框的IoU矩阵
    
    参数:
        框组1: [N, 4] 形状的边界框数组
        框组2: [M, 4] 形状的边界框数组
    
    返回:
        [N, M] 形状的IoU矩阵
    """
    N = 框组1.shape[0]
    M = 框组2.shape[0]
    
    if N == 0 or M == 0:
        return np.zeros((N, M))
    
    # 扩展维度以便广播
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
    
    # 计算并集
    并集面积 = 面积1 + 面积2 - 交集面积
    
    # 避免除零
    并集面积 = np.maximum(并集面积, 1e-10)
    
    return 交集面积 / 并集面积


def 计算精确率召回率(
    真实框列表: List[np.ndarray],
    预测框列表: List[np.ndarray],
    预测分数列表: List[np.ndarray],
    IoU阈值: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算精确率-召回率曲线
    
    参数:
        真实框列表: 每张图像的真实框列表
        预测框列表: 每张图像的预测框列表
        预测分数列表: 每张图像的预测分数列表
        IoU阈值: IoU匹配阈值
    
    返回:
        精确率数组, 召回率数组, 分数阈值数组
    """
    # 收集所有预测
    所有预测 = []
    总真实数 = 0
    
    for 图像索引, (真实框, 预测框, 预测分数) in enumerate(zip(真实框列表, 预测框列表, 预测分数列表)):
        总真实数 += len(真实框)
        
        if len(预测框) == 0:
            continue
        
        # 计算IoU矩阵
        if len(真实框) > 0:
            IoU矩阵 = 计算批量IoU(预测框, 真实框)
        else:
            IoU矩阵 = np.zeros((len(预测框), 0))
        
        # 标记每个预测是TP还是FP
        已匹配真实 = set()
        for 预测索引 in range(len(预测框)):
            分数 = 预测分数[预测索引]
            
            if len(真实框) > 0:
                最大IoU索引 = IoU矩阵[预测索引].argmax()
                最大IoU = IoU矩阵[预测索引, 最大IoU索引]
                
                if 最大IoU >= IoU阈值 and 最大IoU索引 not in 已匹配真实:
                    是TP = True
                    已匹配真实.add(最大IoU索引)
                else:
                    是TP = False
            else:
                是TP = False
            
            所有预测.append((分数, 是TP))
    
    if len(所有预测) == 0:
        return np.array([1.0]), np.array([0.0]), np.array([1.0])
    
    # 按分数降序排序
    所有预测.sort(key=lambda x: -x[0])
    
    # 计算累积TP和FP
    累积TP = 0
    累积FP = 0
    精确率列表 = []
    召回率列表 = []
    分数阈值列表 = []
    
    for 分数, 是TP in 所有预测:
        if 是TP:
            累积TP += 1
        else:
            累积FP += 1
        
        精确率 = 累积TP / (累积TP + 累积FP)
        召回率 = 累积TP / 总真实数 if 总真实数 > 0 else 0
        
        精确率列表.append(精确率)
        召回率列表.append(召回率)
        分数阈值列表.append(分数)
    
    return np.array(精确率列表), np.array(召回率列表), np.array(分数阈值列表)


def 计算AP(精确率: np.ndarray, 召回率: np.ndarray) -> float:
    """
    计算平均精度 (Average Precision)
    
    使用11点插值法
    
    参数:
        精确率: 精确率数组
        召回率: 召回率数组
    
    返回:
        AP值
    """
    # 添加起始点和结束点
    召回率 = np.concatenate([[0], 召回率, [1]])
    精确率 = np.concatenate([[1], 精确率, [0]])
    
    # 确保精确率单调递减
    for i in range(len(精确率) - 2, -1, -1):
        精确率[i] = max(精确率[i], 精确率[i + 1])
    
    # 找到召回率变化的点
    变化点 = np.where(召回率[1:] != 召回率[:-1])[0] + 1
    
    # 计算面积
    AP = np.sum((召回率[变化点] - 召回率[变化点 - 1]) * 精确率[变化点])
    
    return float(AP)


def 计算mAP(
    真实框字典: Dict[str, List[np.ndarray]],
    预测框字典: Dict[str, List[np.ndarray]],
    预测分数字典: Dict[str, List[np.ndarray]],
    类别列表: List[str],
    IoU阈值: float = 0.5
) -> Dict[str, float]:
    """
    计算所有类别的mAP
    
    参数:
        真实框字典: {类别: [每张图像的真实框列表]}
        预测框字典: {类别: [每张图像的预测框列表]}
        预测分数字典: {类别: [每张图像的预测分数列表]}
        类别列表: 类别名称列表
        IoU阈值: IoU匹配阈值
    
    返回:
        {类别: AP值} 和 {'mAP': 平均AP}
    """
    结果 = {}
    AP列表 = []
    
    for 类别 in 类别列表:
        if 类别 in 真实框字典 and 类别 in 预测框字典:
            精确率, 召回率, _ = 计算精确率召回率(
                真实框字典[类别],
                预测框字典[类别],
                预测分数字典[类别],
                IoU阈值
            )
            AP = 计算AP(精确率, 召回率)
        else:
            AP = 0.0
        
        结果[类别] = AP
        AP列表.append(AP)
    
    结果['mAP'] = float(np.mean(AP列表)) if AP列表 else 0.0
    
    return 结果


class MOT指标计算器:
    """
    多目标跟踪指标计算器
    
    计算MOTA、IDF1、IDSW等跟踪指标
    """
    
    def __init__(self, IoU阈值: float = 0.5):
        """
        初始化计算器
        
        参数:
            IoU阈值: IoU匹配阈值
        """
        self.IoU阈值 = IoU阈值
        self.重置()
    
    def 重置(self):
        """重置统计量"""
        self.总TP = 0
        self.总FP = 0
        self.总FN = 0
        self.总IDSW = 0
        self.总真实数 = 0
        self.总距离 = 0.0
        self.匹配数 = 0
        
        # 用于IDF1计算
        self.IDTP = 0
        self.IDFP = 0
        self.IDFN = 0
        
        # 跟踪状态
        self.上一帧匹配 = {}  # {真实ID: 预测ID}
    
    def 更新(
        self,
        真实框: np.ndarray,
        真实ID: np.ndarray,
        预测框: np.ndarray,
        预测ID: np.ndarray
    ):
        """
        更新一帧的统计量
        
        参数:
            真实框: [N, 4] 真实边界框
            真实ID: [N] 真实目标ID
            预测框: [M, 4] 预测边界框
            预测ID: [M] 预测目标ID
        """
        N = len(真实框)
        M = len(预测框)
        
        self.总真实数 += N
        
        if N == 0 and M == 0:
            return
        
        if N == 0:
            self.总FP += M
            self.IDFP += M
            return
        
        if M == 0:
            self.总FN += N
            self.IDFN += N
            return
        
        # 计算IoU矩阵
        IoU矩阵 = 计算批量IoU(预测框, 真实框)
        
        # 贪婪匹配
        匹配对 = []
        已匹配预测 = set()
        已匹配真实 = set()
        
        # 按IoU降序排序所有可能的匹配
        候选匹配 = []
        for i in range(M):
            for j in range(N):
                if IoU矩阵[i, j] >= self.IoU阈值:
                    候选匹配.append((IoU矩阵[i, j], i, j))
        
        候选匹配.sort(reverse=True)
        
        for IoU值, 预测索引, 真实索引 in 候选匹配:
            if 预测索引 not in 已匹配预测 and 真实索引 not in 已匹配真实:
                匹配对.append((预测索引, 真实索引))
                已匹配预测.add(预测索引)
                已匹配真实.add(真实索引)
                self.总距离 += 1 - IoU值
                self.匹配数 += 1
        
        # 计算TP, FP, FN
        TP = len(匹配对)
        FP = M - TP
        FN = N - TP
        
        self.总TP += TP
        self.总FP += FP
        self.总FN += FN
        
        # 计算IDSW
        当前帧匹配 = {}
        for 预测索引, 真实索引 in 匹配对:
            真实id = 真实ID[真实索引]
            预测id = 预测ID[预测索引]
            当前帧匹配[真实id] = 预测id
            
            # 检查ID切换
            if 真实id in self.上一帧匹配:
                if self.上一帧匹配[真实id] != 预测id:
                    self.总IDSW += 1
        
        self.上一帧匹配 = 当前帧匹配
        
        # 更新IDF1相关统计
        self.IDTP += TP
        self.IDFP += FP
        self.IDFN += FN
    
    def 计算指标(self) -> Dict[str, float]:
        """
        计算最终指标
        
        返回:
            包含各项指标的字典
        """
        # MOTA = 1 - (FN + FP + IDSW) / GT
        if self.总真实数 > 0:
            MOTA = 1 - (self.总FN + self.总FP + self.总IDSW) / self.总真实数
        else:
            MOTA = 0.0
        
        # MOTP = 平均IoU
        if self.匹配数 > 0:
            MOTP = 1 - self.总距离 / self.匹配数
        else:
            MOTP = 0.0
        
        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        分母 = 2 * self.IDTP + self.IDFP + self.IDFN
        if 分母 > 0:
            IDF1 = 2 * self.IDTP / 分母
        else:
            IDF1 = 0.0
        
        # 精确率和召回率
        if self.总TP + self.总FP > 0:
            精确率 = self.总TP / (self.总TP + self.总FP)
        else:
            精确率 = 0.0
        
        if self.总TP + self.总FN > 0:
            召回率 = self.总TP / (self.总TP + self.总FN)
        else:
            召回率 = 0.0
        
        return {
            'MOTA': MOTA,
            'MOTP': MOTP,
            'IDF1': IDF1,
            'IDSW': self.总IDSW,
            'TP': self.总TP,
            'FP': self.总FP,
            'FN': self.总FN,
            'Precision': 精确率,
            'Recall': 召回率,
            'GT': self.总真实数,
        }


def 保存指标到JSON(指标: Dict, 保存路径: str):
    """
    保存评估指标到JSON文件
    
    参数:
        指标: 指标字典
        保存路径: JSON文件保存路径
    """
    with open(保存路径, 'w', encoding='utf-8') as f:
        json.dump(指标, f, indent=2, ensure_ascii=False)


def 从JSON加载指标(加载路径: str) -> Dict:
    """
    从JSON文件加载评估指标
    
    参数:
        加载路径: JSON文件路径
    
    返回:
        指标字典
    """
    with open(加载路径, 'r', encoding='utf-8') as f:
        return json.load(f)
