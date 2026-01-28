#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class 日志管理器:
    """
    统一的日志管理器类
    """
    
    _实例 = None
    _已初始化 = False
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._实例 is None:
            cls._实例 = super().__new__(cls)
        return cls._实例
    
    def __init__(
        self,
        名称: str = "红外检测跟踪",
        日志级别: str = "INFO",
        日志目录: str = "outputs/logs",
        控制台输出: bool = True,
        文件输出: bool = True
    ):
        """
        初始化日志管理器
        
        参数:
            名称: 日志器名称
            日志级别: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
            日志目录: 日志文件保存目录
            控制台输出: 是否输出到控制台
            文件输出: 是否输出到文件
        """
        if self._已初始化:
            return
        
        self.名称 = 名称
        self.日志目录 = Path(日志目录)
        
        # 创建日志目录
        self.日志目录.mkdir(parents=True, exist_ok=True)
        
        # 创建日志器
        self.日志器 = logging.getLogger(名称)
        self.日志器.setLevel(getattr(logging, 日志级别.upper()))
        
        # 清除已有的处理器
        self.日志器.handlers.clear()
        
        # 设置日志格式
        格式化器 = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if 控制台输出:
            控制台处理器 = logging.StreamHandler(sys.stdout)
            控制台处理器.setFormatter(格式化器)
            self.日志器.addHandler(控制台处理器)
        
        # 文件处理器
        if 文件输出:
            时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
            日志文件 = self.日志目录 / f"{名称}_{时间戳}.log"
            文件处理器 = logging.FileHandler(日志文件, encoding='utf-8')
            文件处理器.setFormatter(格式化器)
            self.日志器.addHandler(文件处理器)
            self.当前日志文件 = 日志文件
        else:
            self.当前日志文件 = None
        
        self._已初始化 = True
    
    def debug(self, 消息: str):
        """记录调试信息"""
        self.日志器.debug(消息)
    
    def info(self, 消息: str):
        """记录一般信息"""
        self.日志器.info(消息)
    
    def warning(self, 消息: str):
        """记录警告信息"""
        self.日志器.warning(消息)
    
    def error(self, 消息: str):
        """记录错误信息"""
        self.日志器.error(消息)
    
    def critical(self, 消息: str):
        """记录严重错误"""
        self.日志器.critical(消息)
    
    def 记录异常(self, 异常: Exception, 额外信息: str = ""):
        """
        记录异常信息
        
        参数:
            异常: 异常对象
            额外信息: 额外的描述信息
        """
        if 额外信息:
            self.error(f"{额外信息}: {type(异常).__name__}: {str(异常)}")
        else:
            self.error(f"{type(异常).__name__}: {str(异常)}")


class 训练日志记录器:
    """
    训练过程日志记录器
    记录训练指标、损失值等
    """
    
    def __init__(self, 保存目录: str, 实验名称: str):
        """
        初始化训练日志记录器
        
        参数:
            保存目录: 日志保存目录
            实验名称: 实验名称
        """
        self.保存目录 = Path(保存目录)
        self.实验名称 = 实验名称
        
        # 创建目录
        self.保存目录.mkdir(parents=True, exist_ok=True)
        
        # 初始化记录
        self.训练记录 = {
            '实验名称': 实验名称,
            '开始时间': datetime.now().isoformat(),
            '轮次': [],
            '训练损失': [],
            '验证损失': [],
            '指标': {},
        }
        
        self.当前轮次 = 0
    
    def 开始轮次(self, 轮次: int):
        """
        开始新的训练轮次
        
        参数:
            轮次: 轮次编号
        """
        self.当前轮次 = 轮次
        self.训练记录['轮次'].append(轮次)
    
    def 记录损失(self, 训练损失: float, 验证损失: float = None):
        """
        记录损失值
        
        参数:
            训练损失: 训练损失
            验证损失: 验证损失
        """
        self.训练记录['训练损失'].append(训练损失)
        if 验证损失 is not None:
            self.训练记录['验证损失'].append(验证损失)
    
    def 记录指标(self, 指标名: str, 值: float):
        """
        记录评估指标
        
        参数:
            指标名: 指标名称
            值: 指标值
        """
        if 指标名 not in self.训练记录['指标']:
            self.训练记录['指标'][指标名] = []
        self.训练记录['指标'][指标名].append(值)
    
    def 记录批量指标(self, 指标字典: dict):
        """
        批量记录评估指标
        
        参数:
            指标字典: {指标名: 值} 字典
        """
        for 指标名, 值 in 指标字典.items():
            self.记录指标(指标名, 值)
    
    def 保存(self):
        """保存训练记录到文件"""
        self.训练记录['结束时间'] = datetime.now().isoformat()
        
        # 保存JSON
        json路径 = self.保存目录 / f"{self.实验名称}_log.json"
        with open(json路径, 'w', encoding='utf-8') as f:
            json.dump(self.训练记录, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式的损失记录
        csv路径 = self.保存目录 / f"{self.实验名称}_loss.csv"
        with open(csv路径, 'w', encoding='utf-8') as f:
            f.write("轮次,训练损失,验证损失\n")
            for i, 轮次 in enumerate(self.训练记录['轮次']):
                训练损失 = self.训练记录['训练损失'][i] if i < len(self.训练记录['训练损失']) else ""
                验证损失 = self.训练记录['验证损失'][i] if i < len(self.训练记录['验证损失']) else ""
                f.write(f"{轮次},{训练损失},{验证损失}\n")
    
    def 获取最佳指标(self, 指标名: str, 越大越好: bool = True) -> tuple:
        """
        获取某个指标的最佳值及对应轮次
        
        参数:
            指标名: 指标名称
            越大越好: 是否越大越好
        
        返回:
            (最佳值, 对应轮次)
        """
        if 指标名 not in self.训练记录['指标']:
            return None, None
        
        值列表 = self.训练记录['指标'][指标名]
        if not 值列表:
            return None, None
        
        if 越大越好:
            最佳索引 = max(range(len(值列表)), key=lambda i: 值列表[i])
        else:
            最佳索引 = min(range(len(值列表)), key=lambda i: 值列表[i])
        
        最佳值 = 值列表[最佳索引]
        对应轮次 = self.训练记录['轮次'][最佳索引] if 最佳索引 < len(self.训练记录['轮次']) else 最佳索引
        
        return 最佳值, 对应轮次


class 进度条:
    """
    简单的进度条显示
    """
    
    def __init__(self, 总数: int, 描述: str = "", 长度: int = 50):
        """
        初始化进度条
        
        参数:
            总数: 总迭代次数
            描述: 进度条描述
            长度: 进度条显示长度
        """
        self.总数 = 总数
        self.描述 = 描述
        self.长度 = 长度
        self.当前 = 0
        self.开始时间 = datetime.now()
    
    def 更新(self, 步数: int = 1):
        """
        更新进度
        
        参数:
            步数: 前进步数
        """
        self.当前 += 步数
        self._显示()
    
    def _显示(self):
        """显示进度条"""
        比例 = self.当前 / self.总数
        已完成 = int(self.长度 * 比例)
        
        # 计算剩余时间
        已用时间 = (datetime.now() - self.开始时间).total_seconds()
        if self.当前 > 0:
            预计总时间 = 已用时间 / self.当前 * self.总数
            剩余时间 = 预计总时间 - 已用时间
            时间信息 = f" ETA: {剩余时间:.0f}s"
        else:
            时间信息 = ""
        
        # 构建进度条
        进度条 = '█' * 已完成 + '░' * (self.长度 - 已完成)
        
        # 显示
        输出 = f"\r{self.描述} |{进度条}| {self.当前}/{self.总数} ({比例*100:.1f}%){时间信息}"
        sys.stdout.write(输出)
        sys.stdout.flush()
        
        if self.当前 >= self.总数:
            print()  # 换行
    
    def 完成(self):
        """标记完成"""
        self.当前 = self.总数
        self._显示()


# 创建全局日志实例
def 获取日志器(
    名称: str = "红外检测跟踪",
    日志级别: str = "INFO",
    日志目录: str = "outputs/logs"
) -> 日志管理器:
    """
    获取日志器实例
    
    参数:
        名称: 日志器名称
        日志级别: 日志级别
        日志目录: 日志目录
    
    返回:
        日志管理器实例
    """
    return 日志管理器(名称, 日志级别, 日志目录)


# 便捷函数
_默认日志器 = None


def 初始化日志(
    名称: str = "红外检测跟踪",
    日志级别: str = "INFO",
    日志目录: str = "outputs/logs"
):
    """初始化默认日志器"""
    global _默认日志器
    _默认日志器 = 日志管理器(名称, 日志级别, 日志目录)


def log_debug(消息: str):
    """记录调试信息"""
    if _默认日志器:
        _默认日志器.debug(消息)


def log_info(消息: str):
    """记录一般信息"""
    if _默认日志器:
        _默认日志器.info(消息)


def log_warning(消息: str):
    """记录警告信息"""
    if _默认日志器:
        _默认日志器.warning(消息)


def log_error(消息: str):
    """记录错误信息"""
    if _默认日志器:
        _默认日志器.error(消息)
