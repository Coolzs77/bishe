#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡尔曼滤波器模块

提供用于目标跟踪的卡尔曼滤波器实现
"""

import numpy as np
from typing import Tuple, Optional


def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """
    将边界框从 [x1, y1, x2, y2] 格式convert为 [x_center, y_center, width, height] 格式
    
    Args:
        bbox: 边界框坐标，可以是 (4,) 或 (N, 4) 形状
        
    Returns:
        convert后的边界框坐标
    """
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x1, y1, x2, y2 = bbox
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1
        ])
    else:
        result = np.zeros_like(bbox)
        result[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # x_center
        result[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # y_center
        result[:, 2] = bbox[:, 2] - bbox[:, 0]  # width
        result[:, 3] = bbox[:, 3] - bbox[:, 1]  # height
        return result


def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    将边界框从 [x_center, y_center, width, height] 格式convert为 [x1, y1, x2, y2] 格式
    
    Args:
        bbox: 边界框坐标，可以是 (4,) 或 (N, 4) 形状
        
    Returns:
        convert后的边界框坐标
    """
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x_c, y_c, w, h = bbox
        return np.array([
            x_c - w / 2,
            y_c - h / 2,
            x_c + w / 2,
            y_c + h / 2
        ])
    else:
        result = np.zeros_like(bbox)
        result[:, 0] = bbox[:, 0] - bbox[:, 2] / 2  # x1
        result[:, 1] = bbox[:, 1] - bbox[:, 3] / 2  # y1
        result[:, 2] = bbox[:, 0] + bbox[:, 2] / 2  # x2
        result[:, 3] = bbox[:, 1] + bbox[:, 3] / 2  # y2
        return result


def xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """
    将边界框从 [x1, y1, x2, y2] 格式convert为 [x_center, y_center, aspect_ratio, height] 格式
    
    Args:
        bbox: 边界框坐标
        
    Returns:
        convert后的边界框坐标
    """
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            w / max(h, 1e-6),
            h
        ])
    else:
        result = np.zeros_like(bbox)
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]
        result[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2  # x_center
        result[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2  # y_center
        result[:, 2] = w / np.maximum(h, 1e-6)  # aspect_ratio
        result[:, 3] = h  # height
        return result


def xyah_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    将边界框从 [x_center, y_center, aspect_ratio, height] 格式convert为 [x1, y1, x2, y2] 格式
    
    Args:
        bbox: 边界框坐标
        
    Returns:
        convert后的边界框坐标
    """
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x_c, y_c, a, h = bbox
        w = a * h
        return np.array([
            x_c - w / 2,
            y_c - h / 2,
            x_c + w / 2,
            y_c + h / 2
        ])
    else:
        result = np.zeros_like(bbox)
        w = bbox[:, 2] * bbox[:, 3]  # aspect_ratio * height
        h = bbox[:, 3]
        result[:, 0] = bbox[:, 0] - w / 2  # x1
        result[:, 1] = bbox[:, 1] - h / 2  # y1
        result[:, 2] = bbox[:, 0] + w / 2  # x2
        result[:, 3] = bbox[:, 1] + h / 2  # y2
        return result


class KalmanFilter:
    """
    卡尔曼滤波器
    
    用于目标跟踪的标准卡尔曼滤波器实现
    状态向量: [x, y, a, h, vx, vy, va, vh]
    其中 (x, y) 是中心坐标, a 是宽高比, h 是高度
    (vx, vy, va, vh) 是对应的速度
    
    Attributes:
        dim_x: 状态向量维度
        dim_z: 观测向量维度
        _std_weight_position: 位置噪声权重
        _std_weight_velocity: 速度噪声权重
    """
    
    def __init__(self, std_weight_position: float = 1./20, std_weight_velocity: float = 1./160):
        """
        初始化卡尔曼滤波器
        
        Args:
            std_weight_position: 位置噪声权重
            std_weight_velocity: 速度噪声权重
        """
        self.dim_x = 8  # 状态向量维度
        self.dim_z = 4  # 观测向量维度
        
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        
        # 状态转移矩阵
        self._motion_mat = np.eye(self.dim_x)
        for i in range(4):
            self._motion_mat[i, 4 + i] = 1.0
        
        # 观测矩阵
        self._update_mat = np.eye(self.dim_z, self.dim_x)
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据初始观测创建跟踪状态
        
        Args:
            measurement: 初始观测 [x, y, a, h]
            
        Returns:
            mean: 初始状态均值
            covariance: 初始状态协方差
        """
        mean_pos = measurement
        mean_vel = np.zeros(4)
        mean = np.concatenate([mean_pos, mean_vel])
        
        # 初始协方差
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测步骤
        
        Args:
            mean: 当前状态均值
            covariance: 当前状态协方差
            
        Returns:
            预测的状态均值和协方差
        """
        # 过程噪声
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        
        # 预测
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        
        return mean, covariance
    
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将状态投影到观测空间
        
        Args:
            mean: 状态均值
            covariance: 状态协方差
            
        Returns:
            观测空间的均值和协方差
        """
        # 观测噪声
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        
        return mean, covariance
    
    def update(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray, 
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新步骤
        
        Args:
            mean: 预测的状态均值
            covariance: 预测的状态协方差
            measurement: 观测值 [x, y, a, h]
            
        Returns:
            更新后的状态均值和协方差
        """
        # 投影到观测空间
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # 计算卡尔曼增益 K = P * H^T * S^(-1)
        # 其中 S = H * P * H^T + R (projected_cov)
        try:
            chol_factor = np.linalg.cholesky(projected_cov)
            # 使用Cholesky分解求解
            kalman_gain = np.linalg.solve(
                chol_factor.T,
                np.linalg.solve(chol_factor, self._update_mat @ covariance.T)
            ).T
        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用伪逆
            kalman_gain = covariance @ self._update_mat.T @ np.linalg.pinv(projected_cov)
        
        # 计算新息
        innovation = measurement - projected_mean
        
        # 更新状态
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        
        return new_mean, new_covariance
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """
        计算状态和观测之间的马氏距离
        
        Args:
            mean: 状态均值
            covariance: 状态协方差
            measurements: 观测数组，形状为 (N, 4)
            only_position: 是否只考虑位置
            
        Returns:
            马氏距离数组
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]
        
        # 计算马氏距离
        diff = measurements - projected_mean
        
        try:
            chol = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(chol, diff.T).T
            return np.sum(z ** 2, axis=1)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵不是正定的，使用伪逆
            inv_cov = np.linalg.pinv(projected_cov)
            return np.sum(diff @ inv_cov * diff, axis=1)


class KalmanBoxTracker:
    """
    单目标卡尔曼滤波tracker
    
    封装了卡尔曼滤波器，用于跟踪单个目标
    
    Attributes:
        kf: 卡尔曼滤波器实例
        mean: 当前状态均值
        covariance: 当前状态协方差
        track_id: 跟踪ID
        hits: 命中次数
        time_since_update: 自上次更新以来的帧数
        age: 目标存在帧数
    """
    
    count = 0  # 全局计数器
    
    def __init__(self, bbox: np.ndarray, track_id: Optional[int] = None):
        """
        初始化单目标tracker
        
        Args:
            bbox: 初始边界框 [x1, y1, x2, y2]
            track_id: 跟踪ID，如果为None则自动分配
        """
        self.kf = KalmanFilter()
        
        # convert为 [x, y, a, h] 格式
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.covariance = self.kf.initiate(measurement)
        
        # 跟踪ID
        if track_id is None:
            KalmanBoxTracker.count += 1
            self.track_id = KalmanBoxTracker.count
        else:
            self.track_id = track_id
        
        # 跟踪状态
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        
        # 历史记录
        self.history = []
    
    def predict(self) -> np.ndarray:
        """
        预测下一帧的状态
        
        Returns:
            预测的边界框 [x1, y1, x2, y2]
        """
        # 防止宽高比和高度变为负值
        if self.mean[6] + self.mean[2] <= 0:
            self.mean[6] = 0
        if self.mean[7] + self.mean[3] <= 0:
            self.mean[7] = 0
        
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
        # 保存历史
        self.history.append(self.get_state())
        
        return self.get_state()
    
    def update(self, bbox: np.ndarray) -> None:
        """
        使用观测更新状态
        
        Args:
            bbox: 观测的边界框 [x1, y1, x2, y2]
        """
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        
        self.hits += 1
        self.time_since_update = 0
        
        # 清空历史（已更新）
        self.history = []
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态的边界框
        
        Returns:
            边界框 [x1, y1, x2, y2]
        """
        return xyah_to_xyxy(self.mean[:4])
    
    @classmethod
    def reset_count(cls):
        """重置全局计数器"""
        cls.count = 0
