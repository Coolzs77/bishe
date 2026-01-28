"""
日志工具模块

提供统一的日志管理功能，包括日志管理器、训练日志记录器和进度条等工具。
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class LogManager:
    """
    日志管理器（单例模式）
    
    提供统一的日志记录接口，支持控制台和文件输出。
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = 'main',
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        初始化日志管理器
        
        Args:
            name: 日志器名称
            log_dir: 日志文件保存目录
            level: 日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        # 避免重复初始化
        if LogManager._initialized:
            return
        
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # 清除已有handler
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件输出
        if file_output and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None
        
        LogManager._initialized = True
    
    def debug(self, message: str):
        """
        记录调试级别日志
        
        Args:
            message: 日志消息
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """
        记录信息级别日志
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        记录警告级别日志
        
        Args:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        记录错误级别日志
        
        Args:
            message: 日志消息
        """
        self.logger.error(message)
    
    def critical(self, message: str):
        """
        记录严重错误级别日志
        
        Args:
            message: 日志消息
        """
        self.logger.critical(message)
    
    def log_exception(self, message: str = "发生异常"):
        """
        记录异常信息，包含完整的堆栈跟踪
        
        Args:
            message: 异常描述信息
        """
        self.logger.exception(message)
    
    def set_level(self, level: int):
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    @classmethod
    def reset(cls):
        """
        重置单例实例，用于重新初始化
        """
        cls._instance = None
        cls._initialized = False


class TrainingLogger:
    """
    训练日志记录器
    
    专门用于记录模型训练过程中的损失、指标等信息。
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = 'experiment'
    ):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化记录
        self.history = {
            'epochs': [],
            'losses': {},
            'metrics': {},
            'learning_rates': [],
            'timestamps': []
        }
        
        # 当前epoch状态
        self.current_epoch = 0
        self.epoch_start_time = None
        self.epoch_losses = {}
        self.epoch_metrics = {}
        
        # 最佳指标跟踪
        self.best_metrics = {}
    
    def start_epoch(self, epoch: int, learning_rate: Optional[float] = None):
        """
        开始新的epoch
        
        Args:
            epoch: 当前epoch编号
            learning_rate: 当前学习率
        """
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.epoch_losses = {}
        self.epoch_metrics = {}
        
        if learning_rate is not None:
            self.history['learning_rates'].append(learning_rate)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch} 开始")
        if learning_rate is not None:
            print(f"学习率: {learning_rate:.6f}")
        print('='*50)
    
    def log_loss(self, loss_name: str, loss_value: float, step: Optional[int] = None):
        """
        记录损失值
        
        Args:
            loss_name: 损失名称
            loss_value: 损失值
            step: 当前步数，可选
        """
        if loss_name not in self.epoch_losses:
            self.epoch_losses[loss_name] = []
        
        self.epoch_losses[loss_name].append(loss_value)
        
        if loss_name not in self.history['losses']:
            self.history['losses'][loss_name] = []
    
    def log_metric(self, metric_name: str, metric_value: float):
        """
        记录评估指标
        
        Args:
            metric_name: 指标名称
            metric_value: 指标值
        """
        self.epoch_metrics[metric_name] = metric_value
        
        if metric_name not in self.history['metrics']:
            self.history['metrics'][metric_name] = []
        
        # 更新最佳指标
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = {
                'value': metric_value,
                'epoch': self.current_epoch
            }
        else:
            # 假设指标越大越好（可根据需要修改）
            if metric_value > self.best_metrics[metric_name]['value']:
                self.best_metrics[metric_name] = {
                    'value': metric_value,
                    'epoch': self.current_epoch
                }
    
    def end_epoch(self):
        """
        结束当前epoch，汇总并保存统计信息
        """
        # 计算epoch用时
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # 记录epoch编号和时间戳
        self.history['epochs'].append(self.current_epoch)
        self.history['timestamps'].append(datetime.now().isoformat())
        
        # 汇总损失（取平均值）
        print(f"\nEpoch {self.current_epoch} 结束 (用时: {epoch_time:.2f}s)")
        print("-" * 40)
        print("损失:")
        for loss_name, loss_values in self.epoch_losses.items():
            avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
            self.history['losses'][loss_name].append(avg_loss)
            print(f"  {loss_name}: {avg_loss:.6f}")
        
        # 记录指标
        print("指标:")
        for metric_name, metric_value in self.epoch_metrics.items():
            self.history['metrics'][metric_name].append(metric_value)
            is_best = ""
            if metric_name in self.best_metrics:
                if self.best_metrics[metric_name]['epoch'] == self.current_epoch:
                    is_best = " (最佳)"
            print(f"  {metric_name}: {metric_value:.6f}{is_best}")
        
        print("-" * 40)
    
    def save(self, filepath: Optional[str] = None):
        """
        保存训练历史到JSON文件
        
        Args:
            filepath: 保存路径，如果为None则使用默认路径
        """
        if filepath is None:
            filepath = os.path.join(
                self.log_dir,
                f'{self.experiment_name}_history.json'
            )
        
        save_data = {
            'experiment_name': self.experiment_name,
            'history': self.history,
            'best_metrics': self.best_metrics,
            'save_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"训练历史已保存: {filepath}")
    
    def load(self, filepath: str):
        """
        从JSON文件加载训练历史
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        self.experiment_name = save_data.get('experiment_name', self.experiment_name)
        self.history = save_data.get('history', self.history)
        self.best_metrics = save_data.get('best_metrics', {})
        
        print(f"训练历史已加载: {filepath}")
    
    def get_best_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定指标的最佳值
        
        Args:
            metric_name: 指标名称
        
        Returns:
            包含最佳值和对应epoch的字典，如果不存在则返回None
        """
        return self.best_metrics.get(metric_name)
    
    def get_last_loss(self, loss_name: str) -> Optional[float]:
        """
        获取指定损失的最新值
        
        Args:
            loss_name: 损失名称
        
        Returns:
            最新的损失值，如果不存在则返回None
        """
        if loss_name in self.history['losses'] and self.history['losses'][loss_name]:
            return self.history['losses'][loss_name][-1]
        return None


class ProgressBar:
    """
    进度条
    
    提供命令行进度显示功能。
    """
    
    def __init__(
        self,
        total: int,
        prefix: str = '',
        suffix: str = '',
        decimals: int = 1,
        length: int = 50,
        fill: str = '█',
        empty: str = '-'
    ):
        """
        初始化进度条
        
        Args:
            total: 总步数
            prefix: 前缀文本
            suffix: 后缀文本
            decimals: 百分比小数位数
            length: 进度条字符长度
            fill: 已完成部分的填充字符
            empty: 未完成部分的填充字符
        """
        self.total = max(total, 1)
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.empty = empty
        
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: Optional[int] = None, suffix: Optional[str] = None):
        """
        更新进度条
        
        Args:
            current: 当前步数，如果为None则自动加1
            suffix: 后缀文本更新
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        if suffix is not None:
            self.suffix = suffix
        
        # 计算进度
        percent = min(100.0, 100.0 * self.current / self.total)
        filled_length = int(self.length * self.current / self.total)
        bar = self.fill * filled_length + self.empty * (self.length - filled_length)
        
        # 计算预计剩余时间
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = self._format_time(eta)
        else:
            eta_str = '--:--'
        
        # 打印进度条
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:.{self.decimals}f}% ETA: {eta_str} {self.suffix}')
        sys.stdout.flush()
    
    def finish(self, message: Optional[str] = None):
        """
        完成进度条
        
        Args:
            message: 完成消息
        """
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        # 确保显示100%
        bar = self.fill * self.length
        
        if message:
            sys.stdout.write(f'\r{self.prefix} |{bar}| 100.0% 完成: {elapsed_str} {message}\n')
        else:
            sys.stdout.write(f'\r{self.prefix} |{bar}| 100.0% 完成: {elapsed_str} {self.suffix}\n')
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """
        格式化时间
        
        Args:
            seconds: 秒数
        
        Returns:
            格式化的时间字符串
        """
        if seconds < 60:
            return f'{seconds:.0f}s'
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f'{minutes}m{secs}s'
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f'{hours}h{minutes}m'


# 全局日志管理器实例
_global_logger: Optional[LogManager] = None


def init_logger(
    name: str = 'main',
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> LogManager:
    """
    初始化全局日志管理器
    
    Args:
        name: 日志器名称
        log_dir: 日志文件保存目录
        level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
    
    Returns:
        日志管理器实例
    """
    global _global_logger
    
    # 如果已存在，先重置
    if _global_logger is not None:
        LogManager.reset()
    
    _global_logger = LogManager(
        name=name,
        log_dir=log_dir,
        level=level,
        console_output=console_output,
        file_output=file_output
    )
    
    return _global_logger


def get_logger() -> LogManager:
    """
    获取全局日志管理器实例
    
    如果尚未初始化，则使用默认配置初始化
    
    Returns:
        日志管理器实例
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = LogManager()
    
    return _global_logger


def log_debug(message: str):
    """
    记录调试级别日志（便捷函数）
    
    Args:
        message: 日志消息
    """
    get_logger().debug(message)


def log_info(message: str):
    """
    记录信息级别日志（便捷函数）
    
    Args:
        message: 日志消息
    """
    get_logger().info(message)


def log_warning(message: str):
    """
    记录警告级别日志（便捷函数）
    
    Args:
        message: 日志消息
    """
    get_logger().warning(message)


def log_error(message: str):
    """
    记录错误级别日志（便捷函数）
    
    Args:
        message: 日志消息
    """
    get_logger().error(message)
