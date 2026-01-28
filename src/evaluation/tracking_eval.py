#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标跟踪评估模块
提供跟踪算法的评估功能
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class MOT评估器:
    """
    多目标跟踪评估器
    
    计算MOTA、IDF1、IDSW等MOT指标
    """
    
    def __init__(self, IoU阈值: float = 0.5):
        """
        初始化评估器
        
        参数:
            IoU阈值: IoU匹配阈值
        """
        self.IoU阈值 = IoU阈值
        self.重置()
    
    def 重置(self):
        """重置评估数据"""
        # 基本统计
        self.总帧数 = 0
        self.总真实目标数 = 0
        self.总TP = 0
        self.总FP = 0
        self.总FN = 0
        self.总IDSW = 0
        
        # 用于MOTP计算
        self.总IoU = 0.0
        self.匹配数 = 0
        
        # 用于IDF1计算
        self.IDTP = 0
        self.IDFP = 0
        self.IDFN = 0
        
        # 用于跟踪ID切换检测
        self.上一帧匹配 = {}  # {真实ID: 跟踪ID}
        
        # 轨迹级别统计
        self.真实轨迹集 = set()
        self.跟踪轨迹集 = set()
        self.部分跟踪数 = 0
        self.主要跟踪数 = 0
        self.完全丢失数 = 0
    
    def 更新帧(
        self,
        真实框: np.ndarray,
        真实ID: np.ndarray,
        跟踪框: np.ndarray,
        跟踪ID: np.ndarray
    ):
        """
        更新一帧的评估数据
        
        参数:
            真实框: [N, 4] 真实边界框
            真实ID: [N] 真实目标ID
            跟踪框: [M, 4] 跟踪边界框
            跟踪ID: [M] 跟踪目标ID
        """
        self.总帧数 += 1
        N = len(真实框)
        M = len(跟踪框)
        
        self.总真实目标数 += N
        
        # 更新轨迹集合
        self.真实轨迹集.update(真实ID.tolist())
        self.跟踪轨迹集.update(跟踪ID.tolist())
        
        # 边界情况
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
        IoU矩阵 = self._计算IoU矩阵(跟踪框, 真实框)
        
        # 匈牙利匹配
        匹配对, 未匹配跟踪, 未匹配真实 = self._匹配(IoU矩阵)
        
        # 统计TP、FP、FN
        TP = len(匹配对)
        FP = len(未匹配跟踪)
        FN = len(未匹配真实)
        
        self.总TP += TP
        self.总FP += FP
        self.总FN += FN
        
        # 更新MOTP统计
        for 跟踪索引, 真实索引 in 匹配对:
            self.总IoU += IoU矩阵[跟踪索引, 真实索引]
            self.匹配数 += 1
        
        # 检测ID切换
        当前帧匹配 = {}
        for 跟踪索引, 真实索引 in 匹配对:
            真实id = int(真实ID[真实索引])
            跟踪id = int(跟踪ID[跟踪索引])
            当前帧匹配[真实id] = 跟踪id
            
            # 检查ID切换
            if 真实id in self.上一帧匹配:
                if self.上一帧匹配[真实id] != 跟踪id:
                    self.总IDSW += 1
        
        self.上一帧匹配 = 当前帧匹配
        
        # 更新IDF1统计
        self.IDTP += TP
        self.IDFP += FP
        self.IDFN += FN
    
    def _计算IoU矩阵(self, 框组1: np.ndarray, 框组2: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        N = len(框组1)
        M = len(框组2)
        IoU矩阵 = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                IoU矩阵[i, j] = self._计算IoU(框组1[i], 框组2[j])
        
        return IoU矩阵
    
    def _计算IoU(self, 框1: np.ndarray, 框2: np.ndarray) -> float:
        """计算两个框的IoU"""
        x1 = max(框1[0], 框2[0])
        y1 = max(框1[1], 框2[1])
        x2 = min(框1[2], 框2[2])
        y2 = min(框1[3], 框2[3])
        
        交集宽 = max(0, x2 - x1)
        交集高 = max(0, y2 - y1)
        交集面积 = 交集宽 * 交集高
        
        面积1 = (框1[2] - 框1[0]) * (框1[3] - 框1[1])
        面积2 = (框2[2] - 框2[0]) * (框2[3] - 框2[1])
        
        并集面积 = 面积1 + 面积2 - 交集面积
        
        if 并集面积 == 0:
            return 0.0
        
        return 交集面积 / 并集面积
    
    def _匹配(self, IoU矩阵: np.ndarray) -> Tuple[List, List, List]:
        """使用贪婪匹配"""
        匹配对 = []
        已匹配跟踪 = set()
        已匹配真实 = set()
        
        # 找所有有效候选
        候选 = []
        for i in range(IoU矩阵.shape[0]):
            for j in range(IoU矩阵.shape[1]):
                if IoU矩阵[i, j] >= self.IoU阈值:
                    候选.append((IoU矩阵[i, j], i, j))
        
        # 按IoU降序排序
        候选.sort(reverse=True)
        
        # 贪婪选择
        for IoU值, 跟踪索引, 真实索引 in 候选:
            if 跟踪索引 not in 已匹配跟踪 and 真实索引 not in 已匹配真实:
                匹配对.append((跟踪索引, 真实索引))
                已匹配跟踪.add(跟踪索引)
                已匹配真实.add(真实索引)
        
        未匹配跟踪 = [i for i in range(IoU矩阵.shape[0]) if i not in 已匹配跟踪]
        未匹配真实 = [j for j in range(IoU矩阵.shape[1]) if j not in 已匹配真实]
        
        return 匹配对, 未匹配跟踪, 未匹配真实
    
    def 计算指标(self) -> Dict:
        """
        计算最终评估指标
        
        返回:
            包含各项指标的字典
        """
        # MOTA = 1 - (FN + FP + IDSW) / GT
        if self.总真实目标数 > 0:
            MOTA = 1.0 - (self.总FN + self.总FP + self.总IDSW) / self.总真实目标数
        else:
            MOTA = 0.0
        
        # MOTP = 平均IoU
        if self.匹配数 > 0:
            MOTP = self.总IoU / self.匹配数
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
        
        # F1分数
        if 精确率 + 召回率 > 0:
            F1 = 2 * 精确率 * 召回率 / (精确率 + 召回率)
        else:
            F1 = 0.0
        
        return {
            'MOTA': float(MOTA),
            'MOTP': float(MOTP),
            'IDF1': float(IDF1),
            'IDSW': int(self.总IDSW),
            'TP': int(self.总TP),
            'FP': int(self.总FP),
            'FN': int(self.总FN),
            'Precision': float(精确率),
            'Recall': float(召回率),
            'F1': float(F1),
            'GT_Objects': int(self.总真实目标数),
            'GT_Tracks': len(self.真实轨迹集),
            'Pred_Tracks': len(self.跟踪轨迹集),
            'Frames': int(self.总帧数),
        }
    
    def 生成报告(self) -> str:
        """
        生成评估报告
        
        返回:
            格式化的评估报告字符串
        """
        指标 = self.计算指标()
        
        报告行 = []
        报告行.append("=" * 50)
        报告行.append("多目标跟踪评估报告")
        报告行.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        报告行.append("=" * 50)
        
        报告行.append("\n基本统计:")
        报告行.append(f"  总帧数: {指标['Frames']}")
        报告行.append(f"  真实目标数: {指标['GT_Objects']}")
        报告行.append(f"  真实轨迹数: {指标['GT_Tracks']}")
        报告行.append(f"  预测轨迹数: {指标['Pred_Tracks']}")
        
        报告行.append("\n匹配统计:")
        报告行.append(f"  TP (True Positive): {指标['TP']}")
        报告行.append(f"  FP (False Positive): {指标['FP']}")
        报告行.append(f"  FN (False Negative): {指标['FN']}")
        报告行.append(f"  IDSW (ID Switch): {指标['IDSW']}")
        
        报告行.append("\n评估指标:")
        报告行.append(f"  MOTA: {指标['MOTA']:.4f} ({指标['MOTA']*100:.2f}%)")
        报告行.append(f"  MOTP: {指标['MOTP']:.4f} ({指标['MOTP']*100:.2f}%)")
        报告行.append(f"  IDF1: {指标['IDF1']:.4f} ({指标['IDF1']*100:.2f}%)")
        报告行.append(f"  Precision: {指标['Precision']:.4f} ({指标['Precision']*100:.2f}%)")
        报告行.append(f"  Recall: {指标['Recall']:.4f} ({指标['Recall']*100:.2f}%)")
        报告行.append(f"  F1 Score: {指标['F1']:.4f} ({指标['F1']*100:.2f}%)")
        
        报告行.append("=" * 50)
        
        return "\n".join(报告行)
    
    def 保存结果(self, 输出路径: str):
        """
        保存评估结果到JSON文件
        
        参数:
            输出路径: 输出文件路径
        """
        结果 = {
            '评估时间': datetime.now().isoformat(),
            'IoU阈值': self.IoU阈值,
            '指标': self.计算指标(),
        }
        
        Path(输出路径).parent.mkdir(parents=True, exist_ok=True)
        
        with open(输出路径, 'w', encoding='utf-8') as f:
            json.dump(结果, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存: {输出路径}")


def 评估跟踪器(
    跟踪器,
    数据加载器,
    检测器=None,
    IoU阈值: float = 0.5,
    保存路径: str = None
) -> Dict:
    """
    评估跟踪器的便捷函数
    
    参数:
        跟踪器: 跟踪器对象
        数据加载器: 数据加载器（迭代器）
        检测器: 检测器对象（可选，如果为None则使用真实检测）
        IoU阈值: IoU阈值
        保存路径: 结果保存路径
    
    返回:
        评估结果字典
    """
    评估器 = MOT评估器(IoU阈值)
    
    print("开始评估跟踪器...")
    跟踪器.重置()
    
    for 数据 in 数据加载器:
        图像 = 数据['图像']
        真实框 = 数据['真实框']
        真实ID = 数据['真实ID']
        
        # 如果提供了检测器，使用检测结果；否则使用真实框
        if 检测器 is not None:
            检测结果 = 检测器.检测(图像)
            检测框 = 检测结果.边界框
            检测置信度 = 检测结果.置信度
            检测类别 = 检测结果.类别
        else:
            检测框 = 真实框
            检测置信度 = np.ones(len(真实框))
            检测类别 = np.zeros(len(真实框))
        
        # 跟踪
        跟踪结果 = 跟踪器.更新(检测框, 检测置信度, 检测类别, 图像)
        
        # 获取跟踪结果
        跟踪框 = 跟踪结果.获取边界框数组()
        跟踪ID = 跟踪结果.获取ID数组()
        
        # 更新评估器
        评估器.更新帧(真实框, 真实ID, 跟踪框, 跟踪ID)
    
    # 生成报告
    报告 = 评估器.生成报告()
    print(报告)
    
    # 保存结果
    if 保存路径:
        评估器.保存结果(保存路径)
    
    return 评估器.计算指标()


def 比较跟踪器(
    跟踪器字典: Dict,
    数据加载器,
    检测器=None,
    IoU阈值: float = 0.5,
    保存路径: str = None
) -> Dict:
    """
    比较多个跟踪器的性能
    
    参数:
        跟踪器字典: {名称: 跟踪器} 字典
        数据加载器: 数据加载器
        检测器: 检测器对象
        IoU阈值: IoU阈值
        保存路径: 结果保存路径
    
    返回:
        比较结果字典
    """
    print("=" * 60)
    print("跟踪器对比评估")
    print("=" * 60)
    
    # 先收集所有数据
    数据列表 = list(数据加载器)
    
    结果字典 = {}
    
    for 名称, 跟踪器 in 跟踪器字典.items():
        print(f"\n评估: {名称}")
        print("-" * 40)
        
        评估器 = MOT评估器(IoU阈值)
        跟踪器.重置()
        
        for 数据 in 数据列表:
            图像 = 数据['图像']
            真实框 = 数据['真实框']
            真实ID = 数据['真实ID']
            
            if 检测器 is not None:
                检测结果 = 检测器.检测(图像)
                检测框 = 检测结果.边界框
                检测置信度 = 检测结果.置信度
                检测类别 = 检测结果.类别
            else:
                检测框 = 真实框
                检测置信度 = np.ones(len(真实框))
                检测类别 = np.zeros(len(真实框))
            
            跟踪结果 = 跟踪器.更新(检测框, 检测置信度, 检测类别, 图像)
            
            跟踪框 = 跟踪结果.获取边界框数组()
            跟踪ID = 跟踪结果.获取ID数组()
            
            评估器.更新帧(真实框, 真实ID, 跟踪框, 跟踪ID)
        
        结果字典[名称] = 评估器.计算指标()
        print(f"  MOTA: {结果字典[名称]['MOTA']:.4f}")
        print(f"  IDF1: {结果字典[名称]['IDF1']:.4f}")
        print(f"  IDSW: {结果字典[名称]['IDSW']}")
    
    # 打印比较表格
    print("\n" + "=" * 60)
    print("对比结果:")
    print("-" * 60)
    print(f"{'算法':<20} {'MOTA':<10} {'IDF1':<10} {'MOTP':<10} {'IDSW':<10}")
    print("-" * 60)
    
    for 名称, 指标 in 结果字典.items():
        print(f"{名称:<20} {指标['MOTA']:.4f}    {指标['IDF1']:.4f}    {指标['MOTP']:.4f}    {指标['IDSW']}")
    
    print("=" * 60)
    
    # 保存结果
    if 保存路径:
        with open(保存路径, 'w', encoding='utf-8') as f:
            json.dump(结果字典, f, indent=2, ensure_ascii=False)
        print(f"\n对比结果已保存: {保存路径}")
    
    return 结果字典
