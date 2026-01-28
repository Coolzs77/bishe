#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
用于目标检测和跟踪结果的可视化
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import colorsys


def 生成颜色列表(数量: int) -> List[Tuple[int, int, int]]:
    """
    生成指定数量的不同颜色
    
    参数:
        数量: 需要的颜色数量
    
    返回:
        BGR颜色元组列表
    """
    颜色列表 = []
    for i in range(数量):
        # 使用HSV色彩空间生成均匀分布的颜色
        色相 = i / 数量
        饱和度 = 0.9
        明度 = 0.9
        
        r, g, b = colorsys.hsv_to_rgb(色相, 饱和度, 明度)
        颜色列表.append((int(b * 255), int(g * 255), int(r * 255)))
    
    return 颜色列表


def 获取ID颜色(目标ID: int, 颜色数: int = 100) -> Tuple[int, int, int]:
    """
    根据目标ID获取对应的颜色
    
    参数:
        目标ID: 目标的唯一标识
        颜色数: 颜色池大小
    
    返回:
        BGR颜色元组
    """
    颜色池 = 生成颜色列表(颜色数)
    return 颜色池[目标ID % 颜色数]


def 绘制边界框(
    图像: np.ndarray,
    边界框: np.ndarray,
    类别名: str = "",
    置信度: float = None,
    颜色: Tuple[int, int, int] = (0, 255, 0),
    线宽: int = 2,
    字体大小: float = 0.6
) -> np.ndarray:
    """
    在图像上绘制单个边界框
    
    参数:
        图像: BGR格式的图像
        边界框: [x1, y1, x2, y2] 格式的边界框
        类别名: 类别名称
        置信度: 置信度分数
        颜色: BGR颜色
        线宽: 线条宽度
        字体大小: 字体大小
    
    返回:
        绘制后的图像
    """
    图像 = 图像.copy()
    
    x1, y1, x2, y2 = map(int, 边界框)
    
    # 绘制边界框
    cv2.rectangle(图像, (x1, y1), (x2, y2), 颜色, 线宽)
    
    # 准备标签文本
    标签部分 = []
    if 类别名:
        标签部分.append(类别名)
    if 置信度 is not None:
        标签部分.append(f"{置信度:.2f}")
    
    if 标签部分:
        标签 = " ".join(标签部分)
        
        # 计算文本大小
        (文本宽, 文本高), 基线 = cv2.getTextSize(
            标签, cv2.FONT_HERSHEY_SIMPLEX, 字体大小, 1
        )
        
        # 绘制标签背景
        cv2.rectangle(
            图像,
            (x1, y1 - 文本高 - 10),
            (x1 + 文本宽 + 10, y1),
            颜色,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            图像,
            标签,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            字体大小,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return 图像


def 绘制检测结果(
    图像: np.ndarray,
    检测框列表: List[np.ndarray],
    类别列表: List[str] = None,
    置信度列表: List[float] = None,
    类别名映射: Dict[int, str] = None,
    线宽: int = 2
) -> np.ndarray:
    """
    在图像上绘制所有检测结果
    
    参数:
        图像: BGR格式的图像
        检测框列表: 检测框列表，每个为[x1, y1, x2, y2]
        类别列表: 类别ID或名称列表
        置信度列表: 置信度列表
        类别名映射: 类别ID到名称的映射
        线宽: 线条宽度
    
    返回:
        绘制后的图像
    """
    结果图像 = 图像.copy()
    
    # 生成类别颜色
    颜色映射 = {}
    
    for i, 检测框 in enumerate(检测框列表):
        # 获取类别信息
        if 类别列表 is not None and i < len(类别列表):
            类别 = 类别列表[i]
            if isinstance(类别, int) and 类别名映射:
                类别名 = 类别名映射.get(类别, str(类别))
            else:
                类别名 = str(类别)
            
            # 获取颜色
            if 类别 not in 颜色映射:
                颜色映射[类别] = 获取ID颜色(hash(str(类别)))
            颜色 = 颜色映射[类别]
        else:
            类别名 = ""
            颜色 = (0, 255, 0)
        
        # 获取置信度
        置信度 = None
        if 置信度列表 is not None and i < len(置信度列表):
            置信度 = 置信度列表[i]
        
        # 绘制边界框
        结果图像 = 绘制边界框(
            结果图像, 检测框, 类别名, 置信度, 颜色, 线宽
        )
    
    return 结果图像


def 绘制跟踪结果(
    图像: np.ndarray,
    跟踪框列表: List[np.ndarray],
    跟踪ID列表: List[int],
    类别列表: List[str] = None,
    绘制轨迹: bool = True,
    轨迹历史: Dict[int, List[Tuple[int, int]]] = None,
    轨迹长度: int = 30,
    线宽: int = 2
) -> np.ndarray:
    """
    在图像上绘制跟踪结果
    
    参数:
        图像: BGR格式的图像
        跟踪框列表: 跟踪框列表
        跟踪ID列表: 目标ID列表
        类别列表: 类别列表
        绘制轨迹: 是否绘制轨迹
        轨迹历史: {ID: [(x, y), ...]} 轨迹历史字典
        轨迹长度: 显示的轨迹长度
        线宽: 线条宽度
    
    返回:
        绘制后的图像
    """
    结果图像 = 图像.copy()
    
    for i, (跟踪框, 目标ID) in enumerate(zip(跟踪框列表, 跟踪ID列表)):
        # 获取颜色
        颜色 = 获取ID颜色(目标ID)
        
        x1, y1, x2, y2 = map(int, 跟踪框)
        
        # 绘制边界框
        cv2.rectangle(结果图像, (x1, y1), (x2, y2), 颜色, 线宽)
        
        # 准备标签
        标签部分 = [f"ID:{目标ID}"]
        if 类别列表 is not None and i < len(类别列表):
            标签部分.append(str(类别列表[i]))
        标签 = " ".join(标签部分)
        
        # 绘制ID标签
        (文本宽, 文本高), _ = cv2.getTextSize(
            标签, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            结果图像,
            (x1, y1 - 文本高 - 10),
            (x1 + 文本宽 + 10, y1),
            颜色,
            -1
        )
        cv2.putText(
            结果图像,
            标签,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # 绘制轨迹
        if 绘制轨迹 and 轨迹历史 is not None and 目标ID in 轨迹历史:
            轨迹点 = 轨迹历史[目标ID][-轨迹长度:]
            
            for j in range(1, len(轨迹点)):
                # 轨迹颜色渐变
                alpha = j / len(轨迹点)
                轨迹颜色 = tuple(int(c * alpha) for c in 颜色)
                
                pt1 = tuple(map(int, 轨迹点[j - 1]))
                pt2 = tuple(map(int, 轨迹点[j]))
                
                cv2.line(结果图像, pt1, pt2, 轨迹颜色, max(1, 线宽 - 1))
    
    return 结果图像


def 绘制信息面板(
    图像: np.ndarray,
    信息字典: Dict[str, any],
    位置: str = "左上",
    背景色: Tuple[int, int, int] = (0, 0, 0),
    文字色: Tuple[int, int, int] = (255, 255, 255),
    透明度: float = 0.7
) -> np.ndarray:
    """
    在图像上绘制信息面板
    
    参数:
        图像: BGR格式的图像
        信息字典: 要显示的信息
        位置: 面板位置 ("左上", "右上", "左下", "右下")
        背景色: 背景颜色
        文字色: 文字颜色
        透明度: 背景透明度
    
    返回:
        绘制后的图像
    """
    结果图像 = 图像.copy()
    高, 宽 = 图像.shape[:2]
    
    # 准备文本行
    文本行 = [f"{键}: {值}" for 键, 值 in 信息字典.items()]
    
    # 计算面板大小
    字体 = cv2.FONT_HERSHEY_SIMPLEX
    字体大小 = 0.5
    行间距 = 25
    边距 = 10
    
    最大宽度 = 0
    for 行 in 文本行:
        (文本宽, _), _ = cv2.getTextSize(行, 字体, 字体大小, 1)
        最大宽度 = max(最大宽度, 文本宽)
    
    面板宽 = 最大宽度 + 边距 * 2
    面板高 = len(文本行) * 行间距 + 边距 * 2
    
    # 确定面板位置
    if 位置 == "左上":
        x, y = 10, 10
    elif 位置 == "右上":
        x, y = 宽 - 面板宽 - 10, 10
    elif 位置 == "左下":
        x, y = 10, 高 - 面板高 - 10
    else:  # 右下
        x, y = 宽 - 面板宽 - 10, 高 - 面板高 - 10
    
    # 绘制半透明背景
    覆盖层 = 结果图像.copy()
    cv2.rectangle(覆盖层, (x, y), (x + 面板宽, y + 面板高), 背景色, -1)
    结果图像 = cv2.addWeighted(覆盖层, 透明度, 结果图像, 1 - 透明度, 0)
    
    # 绘制文本
    for i, 行 in enumerate(文本行):
        文本y = y + 边距 + (i + 1) * 行间距 - 8
        cv2.putText(
            结果图像,
            行,
            (x + 边距, 文本y),
            字体,
            字体大小,
            文字色,
            1,
            cv2.LINE_AA
        )
    
    return 结果图像


def 拼接图像网格(
    图像列表: List[np.ndarray],
    行数: int,
    列数: int,
    单元格大小: Tuple[int, int] = None,
    边距: int = 5,
    背景色: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    将多张图像拼接成网格
    
    参数:
        图像列表: 图像列表
        行数: 网格行数
        列数: 网格列数
        单元格大小: (宽, 高) 每个单元格大小，None表示自动
        边距: 单元格间距
        背景色: 背景颜色
    
    返回:
        拼接后的图像
    """
    if not 图像列表:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 确定单元格大小
    if 单元格大小 is None:
        # 使用第一张图像的大小
        单元格高, 单元格宽 = 图像列表[0].shape[:2]
    else:
        单元格宽, 单元格高 = 单元格大小
    
    # 计算总大小
    总宽 = 列数 * 单元格宽 + (列数 + 1) * 边距
    总高 = 行数 * 单元格高 + (行数 + 1) * 边距
    
    # 创建背景
    结果 = np.full((总高, 总宽, 3), 背景色, dtype=np.uint8)
    
    # 放置图像
    for i, 图像 in enumerate(图像列表):
        if i >= 行数 * 列数:
            break
        
        行 = i // 列数
        列 = i % 列数
        
        x = 边距 + 列 * (单元格宽 + 边距)
        y = 边距 + 行 * (单元格高 + 边距)
        
        # 调整图像大小
        调整后 = cv2.resize(图像, (单元格宽, 单元格高))
        
        # 放置图像
        结果[y:y + 单元格高, x:x + 单元格宽] = 调整后
    
    return 结果


def 保存可视化视频(
    输出路径: str,
    帧列表: List[np.ndarray],
    帧率: float = 25.0
):
    """
    将帧序列保存为视频
    
    参数:
        输出路径: 输出视频路径
        帧列表: 帧图像列表
        帧率: 视频帧率
    """
    if not 帧列表:
        return
    
    高, 宽 = 帧列表[0].shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    写入器 = cv2.VideoWriter(输出路径, fourcc, 帧率, (宽, 高))
    
    for 帧 in 帧列表:
        写入器.write(帧)
    
    写入器.release()
