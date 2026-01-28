#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测评估模块
提供检测模型的评估功能
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class 检测评估器:
    """
    目标检测评估器
    
    计算mAP、精确率、召回率等指标
    """
    
    def __init__(
        self,
        类别列表: List[str],
        IoU阈值列表: List[float] = None,
        置信度阈值: float = 0.001
    ):
        """
        初始化评估器
        
        参数:
            类别列表: 类别名称列表
            IoU阈值列表: IoU阈值列表（用于计算mAP@IoU）
            置信度阈值: 置信度过滤阈值
        """
        self.类别列表 = 类别列表
        self.类别数 = len(类别列表)
        self.IoU阈值列表 = IoU阈值列表 or [0.5]
        self.置信度阈值 = 置信度阈值
        
        # 存储评估数据
        self.真实标注 = {类别: [] for 类别 in 类别列表}
        self.预测结果 = {类别: [] for 类别 in 类别列表}
        self.图像计数 = 0
    
    def 重置(self):
        """重置评估数据"""
        self.真实标注 = {类别: [] for 类别 in self.类别列表}
        self.预测结果 = {类别: [] for 类别 in self.类别列表}
        self.图像计数 = 0
    
    def 添加批次(
        self,
        真实框: np.ndarray,
        真实类别: np.ndarray,
        预测框: np.ndarray,
        预测类别: np.ndarray,
        预测置信度: np.ndarray
    ):
        """
        添加一个批次的数据
        
        参数:
            真实框: [N, 4] 真实边界框
            真实类别: [N] 真实类别
            预测框: [M, 4] 预测边界框
            预测类别: [M] 预测类别
            预测置信度: [M] 预测置信度
        """
        self.图像计数 += 1
        
        # 过滤低置信度预测
        有效掩码 = 预测置信度 >= self.置信度阈值
        预测框 = 预测框[有效掩码]
        预测类别 = 预测类别[有效掩码]
        预测置信度 = 预测置信度[有效掩码]
        
        # 按类别分组
        for i, 类别名 in enumerate(self.类别列表):
            # 真实标注
            类别掩码 = 真实类别 == i
            self.真实标注[类别名].append(真实框[类别掩码])
            
            # 预测结果
            预测掩码 = 预测类别 == i
            类别预测框 = 预测框[预测掩码]
            类别预测置信度 = 预测置信度[预测掩码]
            
            self.预测结果[类别名].append({
                '框': 类别预测框,
                '置信度': 类别预测置信度,
            })
    
    def 计算AP(self, 类别名: str, IoU阈值: float = 0.5) -> Dict:
        """
        计算某个类别的AP
        
        参数:
            类别名: 类别名称
            IoU阈值: IoU阈值
        
        返回:
            包含AP及相关统计的字典
        """
        真实列表 = self.真实标注[类别名]
        预测列表 = self.预测结果[类别名]
        
        # 收集所有预测
        所有预测 = []
        for 图像索引, 预测 in enumerate(预测列表):
            for i in range(len(预测['框'])):
                所有预测.append({
                    '图像索引': 图像索引,
                    '框': 预测['框'][i],
                    '置信度': 预测['置信度'][i],
                })
        
        if len(所有预测) == 0:
            return {'AP': 0.0, 'TP': 0, 'FP': 0, 'FN': sum(len(gt) for gt in 真实列表)}
        
        # 按置信度排序
        所有预测.sort(key=lambda x: -x['置信度'])
        
        # 统计总真实数
        总真实数 = sum(len(gt) for gt in 真实列表)
        if 总真实数 == 0:
            return {'AP': 0.0, 'TP': 0, 'FP': len(所有预测), 'FN': 0}
        
        # 标记每个预测为TP或FP
        已匹配真实 = [set() for _ in range(self.图像计数)]
        TP数组 = np.zeros(len(所有预测))
        FP数组 = np.zeros(len(所有预测))
        
        for i, 预测 in enumerate(所有预测):
            图像索引 = 预测['图像索引']
            真实框组 = 真实列表[图像索引]
            
            if len(真实框组) == 0:
                FP数组[i] = 1
                continue
            
            # 计算IoU
            IoU值 = self._计算IoU(预测['框'], 真实框组)
            最佳索引 = np.argmax(IoU值)
            最佳IoU = IoU值[最佳索引]
            
            if 最佳IoU >= IoU阈值 and 最佳索引 not in 已匹配真实[图像索引]:
                TP数组[i] = 1
                已匹配真实[图像索引].add(最佳索引)
            else:
                FP数组[i] = 1
        
        # 计算累积TP和FP
        累积TP = np.cumsum(TP数组)
        累积FP = np.cumsum(FP数组)
        
        # 计算精确率和召回率
        精确率 = 累积TP / (累积TP + 累积FP)
        召回率 = 累积TP / 总真实数
        
        # 计算AP（使用11点插值）
        AP = self._计算AP_11点(精确率, 召回率)
        
        return {
            'AP': float(AP),
            'TP': int(np.sum(TP数组)),
            'FP': int(np.sum(FP数组)),
            'FN': int(总真实数 - np.sum(TP数组)),
            '精确率': float(精确率[-1]) if len(精确率) > 0 else 0.0,
            '召回率': float(召回率[-1]) if len(召回率) > 0 else 0.0,
        }
    
    def _计算IoU(self, 框: np.ndarray, 框组: np.ndarray) -> np.ndarray:
        """计算一个框与一组框的IoU"""
        x1 = np.maximum(框[0], 框组[:, 0])
        y1 = np.maximum(框[1], 框组[:, 1])
        x2 = np.minimum(框[2], 框组[:, 2])
        y2 = np.minimum(框[3], 框组[:, 3])
        
        交集宽 = np.maximum(0, x2 - x1)
        交集高 = np.maximum(0, y2 - y1)
        交集面积 = 交集宽 * 交集高
        
        框面积 = (框[2] - 框[0]) * (框[3] - 框[1])
        框组面积 = (框组[:, 2] - 框组[:, 0]) * (框组[:, 3] - 框组[:, 1])
        
        并集面积 = 框面积 + 框组面积 - 交集面积
        
        return 交集面积 / np.maximum(并集面积, 1e-10)
    
    def _计算AP_11点(self, 精确率: np.ndarray, 召回率: np.ndarray) -> float:
        """使用11点插值计算AP"""
        AP = 0.0
        for t in np.arange(0, 1.1, 0.1):
            掩码 = 召回率 >= t
            if np.any(掩码):
                AP += np.max(精确率[掩码])
        return AP / 11.0
    
    def 计算mAP(self, IoU阈值: float = 0.5) -> Dict:
        """
        计算所有类别的mAP
        
        参数:
            IoU阈值: IoU阈值
        
        返回:
            包含mAP及各类别AP的字典
        """
        结果 = {}
        AP列表 = []
        
        for 类别名 in self.类别列表:
            类别结果 = self.计算AP(类别名, IoU阈值)
            结果[类别名] = 类别结果
            AP列表.append(类别结果['AP'])
        
        结果['mAP'] = float(np.mean(AP列表))
        结果['总TP'] = sum(结果[类别]['TP'] for 类别 in self.类别列表)
        结果['总FP'] = sum(结果[类别]['FP'] for 类别 in self.类别列表)
        结果['总FN'] = sum(结果[类别]['FN'] for 类别 in self.类别列表)
        
        return 结果
    
    def 计算mAP_50_95(self) -> Dict:
        """
        计算mAP@[0.5:0.95]
        
        返回:
            包含各IoU阈值mAP的字典
        """
        IoU阈值列表 = np.arange(0.5, 1.0, 0.05)
        mAP列表 = []
        
        结果 = {}
        for IoU阈值 in IoU阈值列表:
            IoU结果 = self.计算mAP(IoU阈值)
            结果[f'mAP@{IoU阈值:.2f}'] = IoU结果['mAP']
            mAP列表.append(IoU结果['mAP'])
        
        结果['mAP@[0.5:0.95]'] = float(np.mean(mAP列表))
        
        return 结果
    
    def 生成报告(self) -> str:
        """
        生成评估报告
        
        返回:
            格式化的评估报告字符串
        """
        报告行 = []
        报告行.append("=" * 60)
        报告行.append("目标检测评估报告")
        报告行.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        报告行.append(f"图像数量: {self.图像计数}")
        报告行.append("=" * 60)
        
        # 计算mAP@0.5
        mAP结果 = self.计算mAP(0.5)
        
        报告行.append("\n各类别性能 (IoU=0.5):")
        报告行.append("-" * 60)
        报告行.append(f"{'类别':<15} {'AP':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'精确率':<10} {'召回率':<10}")
        报告行.append("-" * 60)
        
        for 类别名 in self.类别列表:
            r = mAP结果[类别名]
            报告行.append(
                f"{类别名:<15} {r['AP']:.4f}    {r['TP']:<8} {r['FP']:<8} {r['FN']:<8} {r['精确率']:.4f}    {r['召回率']:.4f}"
            )
        
        报告行.append("-" * 60)
        报告行.append(f"mAP@0.5: {mAP结果['mAP']:.4f}")
        
        # 计算mAP@[0.5:0.95]
        mAP_50_95结果 = self.计算mAP_50_95()
        报告行.append(f"mAP@[0.5:0.95]: {mAP_50_95结果['mAP@[0.5:0.95]']:.4f}")
        报告行.append("=" * 60)
        
        return "\n".join(报告行)
    
    def 保存结果(self, 输出路径: str):
        """
        保存评估结果到JSON文件
        
        参数:
            输出路径: 输出文件路径
        """
        结果 = {
            '评估时间': datetime.now().isoformat(),
            '图像数量': self.图像计数,
            '类别列表': self.类别列表,
            'mAP@0.5': self.计算mAP(0.5),
            'mAP@[0.5:0.95]': self.计算mAP_50_95(),
        }
        
        Path(输出路径).parent.mkdir(parents=True, exist_ok=True)
        
        with open(输出路径, 'w', encoding='utf-8') as f:
            json.dump(结果, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存: {输出路径}")


def 评估检测模型(
    检测器,
    数据加载器,
    类别列表: List[str],
    IoU阈值: float = 0.5,
    保存路径: str = None
) -> Dict:
    """
    评估检测模型的便捷函数
    
    参数:
        检测器: 检测器对象
        数据加载器: 数据加载器（迭代器，每次返回(图像, 真实框, 真实类别)）
        类别列表: 类别名称列表
        IoU阈值: IoU阈值
        保存路径: 结果保存路径
    
    返回:
        评估结果字典
    """
    评估器 = 检测评估器(类别列表)
    
    print("开始评估...")
    
    for 图像, 真实框, 真实类别 in 数据加载器:
        # 执行检测
        结果 = 检测器.检测(图像)
        
        # 添加到评估器
        评估器.添加批次(
            真实框=真实框,
            真实类别=真实类别,
            预测框=结果.边界框,
            预测类别=结果.类别,
            预测置信度=结果.置信度,
        )
    
    # 生成报告
    报告 = 评估器.生成报告()
    print(报告)
    
    # 保存结果
    if 保存路径:
        评估器.保存结果(保存路径)
    
    return 评估器.计算mAP(IoU阈值)
