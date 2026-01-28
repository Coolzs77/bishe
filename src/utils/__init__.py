#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块
提供评估指标、可视化和日志功能
"""

from .metrics import (
    计算IoU,
    计算批量IoU,
    计算精确率召回率,
    计算AP,
    计算mAP,
    MOT指标计算器,
    保存指标到JSON,
    从JSON加载指标,
)

from .visualization import (
    生成颜色列表,
    获取ID颜色,
    绘制边界框,
    绘制检测结果,
    绘制跟踪结果,
    绘制信息面板,
    拼接图像网格,
    保存可视化视频,
)

from .logger import (
    日志管理器,
    训练日志记录器,
    进度条,
    获取日志器,
    初始化日志,
    log_debug,
    log_info,
    log_warning,
    log_error,
)

__all__ = [
    # 指标
    '计算IoU',
    '计算批量IoU',
    '计算精确率召回率',
    '计算AP',
    '计算mAP',
    'MOT指标计算器',
    '保存指标到JSON',
    '从JSON加载指标',
    # 可视化
    '生成颜色列表',
    '获取ID颜色',
    '绘制边界框',
    '绘制检测结果',
    '绘制跟踪结果',
    '绘制信息面板',
    '拼接图像网格',
    '保存可视化视频',
    # 日志
    '日志管理器',
    '训练日志记录器',
    '进度条',
    '获取日志器',
    '初始化日志',
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',
]
