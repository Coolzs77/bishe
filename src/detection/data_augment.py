#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外imagedata增强模块

提供针对红外/热成像image的专用data增强方法
"""

import random
from typing import Tuple, Optional, List, Union, Callable
import numpy as np


class InfraredDataAugmentor:
    """
    红外imagedata增强器
    
    针对红外/热成像image设计的data增强类，包含多种增强方法
    
    Attributes:
        brightness_range: 亮度调整范围
        contrast_range: 对比度调整范围
        noise_intensity: 噪声强度
        blur_prob: 模糊概率
        flip_prob: 翻转概率
        rotation_angle: 旋转角度范围
        scale_range: 缩放范围
        crop_range: 裁剪范围
        mosaic_prob: Mosaic增强概率
        mixup_prob: MixUp增强概率
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (-0.2, 0.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_intensity: float = 0.02,
        blur_prob: float = 0.1,
        flip_prob: float = 0.5,
        rotation_angle: float = 10.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        crop_range: Tuple[float, float] = (0.8, 1.0),
        mosaic_prob: float = 0.0,
        mixup_prob: float = 0.0
    ):
        """
        初始化红外data增强器
        
        Args:
            brightness_range: 亮度调整范围，默认(-0.2, 0.2)
            contrast_range: 对比度调整范围，默认(0.8, 1.2)
            noise_intensity: 噪声强度，默认0.02
            blur_prob: 模糊概率，默认0.1
            flip_prob: 水平翻转概率，默认0.5
            rotation_angle: 旋转角度范围（度），默认10.0
            scale_range: 缩放范围，默认(0.8, 1.2)
            crop_range: 随机裁剪范围，默认(0.8, 1.0)
            mosaic_prob: Mosaic增强概率，默认0.0
            mixup_prob: MixUp增强概率，默认0.0
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_intensity = noise_intensity
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob
        self.rotation_angle = rotation_angle
        self.scale_range = scale_range
        self.crop_range = crop_range
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
    
    def __call__(
        self,
        image: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        对image和label应用data增强
        
        Args:
            image: inputimage，形状为 (H, W) 或 (H, W, C)
            labels: label数组，形状为 (N, 5)，格式为 [class_id, x_center, y_center, w, h]
                   坐标为归一化的相对坐标
        
        Returns:
            增强后的image和label
        """
        # 确保image是float类型
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
        
        # 复制以避免修改原始data
        image = image.copy()
        if labels is not None:
            labels = labels.copy()
        
        # 应用各种增强
        # 1. 亮度调整
        image = self.random_brightness(image)
        
        # 2. 对比度调整
        image = self.random_contrast(image)
        
        # 3. 添加热噪声
        image = self.add_thermal_noise(image)
        
        # 4. 模拟温度漂移
        image = self.simulate_temperature_drift(image)
        
        # 5. 随机模糊
        image = self.random_blur(image)
        
        # 6. 水平翻转
        image, labels = self.horizontal_flip(image, labels)
        
        # 7. 随机旋转
        image, labels = self.random_rotation(image, labels)
        
        # 8. 随机裁剪
        image, labels = self.random_crop(image, labels)
        
        # 确保值在有效范围内
        image = np.clip(image, 0.0, 1.0)
        
        return image, labels
    
    def random_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        随机亮度调整
        
        模拟红外image中由于环境温度变化导致的整体亮度变化
        
        Args:
            image: inputimage
            
        Returns:
            亮度调整后的image
        """
        delta = random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = image + delta
        return image
    
    def random_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        随机对比度调整
        
        模拟红外image中由于目标与背景温差变化导致的对比度变化
        
        Args:
            image: inputimage
            
        Returns:
            对比度调整后的image
        """
        factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        mean = image.mean()
        image = (image - mean) * factor + mean
        return image
    
    def add_thermal_noise(self, image: np.ndarray) -> np.ndarray:
        """
        添加热噪声
        
        模拟红外探测器的热噪声，包括固定模式噪声和随机噪声
        
        Args:
            image: inputimage
            
        Returns:
            添加噪声后的image
        """
        if self.noise_intensity <= 0:
            return image
        
        # 高斯随机噪声
        noise = np.random.normal(0, self.noise_intensity, image.shape).astype(np.float32)
        image = image + noise
        
        return image
    
    def simulate_temperature_drift(self, image: np.ndarray) -> np.ndarray:
        """
        模拟温度漂移
        
        模拟红外传感器的温度漂移效应，在image上添加渐变
        
        Args:
            image: inputimage
            
        Returns:
            添加温度漂移效果后的image
        """
        if random.random() > 0.3:  # 30%的概率应用
            return image
        
        h, w = image.shape[:2]
        
        # 随机选择漂移方向
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        # 漂移强度
        drift_strength = random.uniform(-0.05, 0.05)
        
        if direction == 'horizontal':
            gradient = np.linspace(0, drift_strength, w)
            gradient = np.tile(gradient, (h, 1))
        elif direction == 'vertical':
            gradient = np.linspace(0, drift_strength, h)
            gradient = np.tile(gradient.reshape(-1, 1), (1, w))
        else:  # diagonal
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            gradient = (xx + yy) / 2 * drift_strength
        
        if image.ndim == 3:
            gradient = gradient[:, :, np.newaxis]
        
        image = image + gradient.astype(np.float32)
        
        return image
    
    def horizontal_flip(
        self,
        image: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        水平翻转
        
        Args:
            image: inputimage
            labels: label数组
            
        Returns:
            翻转后的image和label
        """
        if random.random() > self.flip_prob:
            return image, labels
        
        # 翻转image
        image = np.fliplr(image).copy()
        
        # 翻转label
        if labels is not None and len(labels) > 0:
            labels[:, 1] = 1.0 - labels[:, 1]  # x_center = 1 - x_center
        
        return image, labels
    
    def random_rotation(
        self,
        image: np.ndarray,
        labels: Optional[np.ndarray] = None,
        min_angle_threshold: float = 0.5
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机旋转
        
        Args:
            image: inputimage
            labels: label数组
            min_angle_threshold: 最小旋转角度阈值，低于此值不进行旋转以避免不必要的插值
            
        Returns:
            旋转后的image和label
        """
        if self.rotation_angle <= 0:
            return image, labels
        
        angle = random.uniform(-self.rotation_angle, self.rotation_angle)
        
        # 角度过小时跳过旋转，避免不必要的image插值带来的质量loss
        if abs(angle) < min_angle_threshold:
            return image, labels
        
        import cv2
        
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转image
        image = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # 旋转label
        if labels is not None and len(labels) > 0:
            labels = self._rotate_labels(labels, angle, (w, h))
        
        return image, labels
    
    def _rotate_labels(
        self,
        labels: np.ndarray,
        angle: float,
        img_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        旋转label坐标
        
        Args:
            labels: label数组
            angle: 旋转角度（度）
            img_size: img_size (width, height)
            
        Returns:
            旋转后的label
        """
        w, h = img_size
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 中心点
        cx, cy = 0.5, 0.5
        
        new_labels = labels.copy()
        
        for i in range(len(labels)):
            # 获取边界框的四个角点（归一化坐标）
            x_c, y_c, bw, bh = labels[i, 1:5]
            
            corners = np.array([
                [x_c - bw/2, y_c - bh/2],
                [x_c + bw/2, y_c - bh/2],
                [x_c + bw/2, y_c + bh/2],
                [x_c - bw/2, y_c + bh/2]
            ])
            
            # 旋转角点
            corners_centered = corners - np.array([cx, cy])
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            corners_rotated = corners_centered @ rotation_matrix.T + np.array([cx, cy])
            
            # 计算新的边界框
            x_min, y_min = corners_rotated.min(axis=0)
            x_max, y_max = corners_rotated.max(axis=0)
            
            # 裁剪到有效范围
            x_min = np.clip(x_min, 0, 1)
            x_max = np.clip(x_max, 0, 1)
            y_min = np.clip(y_min, 0, 1)
            y_max = np.clip(y_max, 0, 1)
            
            # 更新label
            new_labels[i, 1] = (x_min + x_max) / 2
            new_labels[i, 2] = (y_min + y_max) / 2
            new_labels[i, 3] = x_max - x_min
            new_labels[i, 4] = y_max - y_min
        
        # 过滤无效label（面积太小）
        valid_mask = (new_labels[:, 3] > 0.01) & (new_labels[:, 4] > 0.01)
        new_labels = new_labels[valid_mask]
        
        return new_labels
    
    def random_blur(self, image: np.ndarray) -> np.ndarray:
        """
        随机模糊
        
        模拟红外image的焦距模糊或运动模糊
        
        Args:
            image: inputimage
            
        Returns:
            模糊后的image
        """
        if random.random() > self.blur_prob:
            return image
        
        import cv2
        
        # 随机选择模糊类型
        blur_type = random.choice(['gaussian', 'average', 'median'])
        
        # 随机内核大小
        kernel_size = random.choice([3, 5])
        
        if blur_type == 'gaussian':
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'average':
            image = cv2.blur(image, (kernel_size, kernel_size))
        else:  # median
            # 中值滤波需要uint8
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
                image_uint8 = cv2.medianBlur(image_uint8, kernel_size)
                image = image_uint8.astype(np.float32) / 255.0
            else:
                image = cv2.medianBlur(image.astype(np.uint8), kernel_size).astype(np.float32)
        
        return image
    
    def random_crop(
        self,
        image: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机裁剪
        
        Args:
            image: inputimage
            labels: label数组
            
        Returns:
            裁剪后的image和label
        """
        # 如果裁剪范围最小值>=1.0，则不进行裁剪
        if self.crop_range[0] >= 1.0:
            return image, labels
        
        import cv2
        
        h, w = image.shape[:2]
        
        # 随机裁剪比例，确保不超过1.0
        crop_ratio = random.uniform(self.crop_range[0], min(self.crop_range[1], 0.999))
        
        # 计算裁剪尺寸
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        # 随机裁剪位置
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        # 裁剪image
        image = image[top:top+new_h, left:left+new_w]
        
        # 调整大小回原始尺寸
        image = cv2.resize(image, (w, h))
        
        # 调整label
        if labels is not None and len(labels) > 0:
            labels = self._adjust_labels_for_crop(
                labels, (left/w, top/h), (new_w/w, new_h/h)
            )
        
        return image, labels
    
    def _adjust_labels_for_crop(
        self,
        labels: np.ndarray,
        offset: Tuple[float, float],
        crop_size: Tuple[float, float]
    ) -> np.ndarray:
        """
        调整裁剪后的label坐标
        
        Args:
            labels: label数组
            offset: 裁剪偏移 (x_offset, y_offset)
            crop_size: 裁剪尺寸比例 (w_ratio, h_ratio)
            
        Returns:
            调整后的label
        """
        x_offset, y_offset = offset
        w_ratio, h_ratio = crop_size
        
        new_labels = labels.copy()
        
        # convert坐标
        new_labels[:, 1] = (labels[:, 1] - x_offset) / w_ratio
        new_labels[:, 2] = (labels[:, 2] - y_offset) / h_ratio
        new_labels[:, 3] = labels[:, 3] / w_ratio
        new_labels[:, 4] = labels[:, 4] / h_ratio
        
        # 裁剪到有效范围
        # 边界框左上角和右下角
        x1 = new_labels[:, 1] - new_labels[:, 3] / 2
        y1 = new_labels[:, 2] - new_labels[:, 4] / 2
        x2 = new_labels[:, 1] + new_labels[:, 3] / 2
        y2 = new_labels[:, 2] + new_labels[:, 4] / 2
        
        # 裁剪边界框
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)
        
        # 更新label
        new_labels[:, 1] = (x1 + x2) / 2
        new_labels[:, 2] = (y1 + y2) / 2
        new_labels[:, 3] = x2 - x1
        new_labels[:, 4] = y2 - y1
        
        # 过滤无效label
        valid_mask = (new_labels[:, 3] > 0.01) & (new_labels[:, 4] > 0.01)
        valid_mask &= (new_labels[:, 1] > 0) & (new_labels[:, 1] < 1)
        valid_mask &= (new_labels[:, 2] > 0) & (new_labels[:, 2] < 1)
        
        return new_labels[valid_mask]
    
    def mosaic_augment(
        self,
        images: List[np.ndarray],
        labels_list: List[np.ndarray],
        output_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mosaicdata增强
        
        将4张image拼接成一张，增加小目标的出现频率
        
        Args:
            images: 4张image的列表
            labels_list: 对应的4个label数组列表
            output_size: output尺寸 (width, height)
            
        Returns:
            拼接后的image和合并的label
        """
        if len(images) != 4 or len(labels_list) != 4:
            raise ValueError("Mosaic增强需要4张image和4组label")
        
        import cv2
        
        w, h = output_size
        
        # 随机选择中心点
        cx = int(w * random.uniform(0.3, 0.7))
        cy = int(h * random.uniform(0.3, 0.7))
        
        # 创建outputimage
        mosaic_img = np.zeros((h, w, 3) if images[0].ndim == 3 else (h, w), dtype=np.float32)
        
        # 合并的label
        mosaic_labels = []
        
        # 四个象限的位置
        placements = [
            (0, 0, cx, cy),           # 左上
            (cx, 0, w, cy),           # 右上
            (0, cy, cx, h),           # 左下
            (cx, cy, w, h)            # 右下
        ]
        
        for i, (img, labels) in enumerate(zip(images, labels_list)):
            x1, y1, x2, y2 = placements[i]
            pw, ph = x2 - x1, y2 - y1
            
            # 缩放image
            img_h, img_w = img.shape[:2]
            img_resized = cv2.resize(img, (pw, ph))
            
            # 放置image
            if mosaic_img.ndim == 3 and img_resized.ndim == 2:
                img_resized = np.stack([img_resized] * 3, axis=-1)
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # 调整label
            if labels is not None and len(labels) > 0:
                adjusted_labels = labels.copy()
                
                # 缩放label到当前象限
                adjusted_labels[:, 1] = (labels[:, 1] * pw + x1) / w
                adjusted_labels[:, 2] = (labels[:, 2] * ph + y1) / h
                adjusted_labels[:, 3] = labels[:, 3] * pw / w
                adjusted_labels[:, 4] = labels[:, 4] * ph / h
                
                mosaic_labels.append(adjusted_labels)
        
        # 合并所有label
        if mosaic_labels:
            mosaic_labels = np.vstack(mosaic_labels)
        else:
            mosaic_labels = np.array([]).reshape(0, 5)
        
        return mosaic_img, mosaic_labels
    
    def mixup_augment(
        self,
        image1: np.ndarray,
        labels1: np.ndarray,
        image2: np.ndarray,
        labels2: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MixUpdata增强
        
        将两张image按一定比例混合
        
        Args:
            image1: 第一张image
            labels1: 第一张image的label
            image2: 第二张image
            labels2: 第二张image的label
            alpha: 混合系数，默认0.5
            
        Returns:
            混合后的image和合并的label
        """
        import cv2
        
        # 随机混合比例
        lam = np.random.beta(alpha, alpha)
        
        # 确保img_size一致
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 混合image
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # 合并label（保留所有label）
        if labels1 is not None and labels2 is not None:
            mixed_labels = np.vstack([labels1, labels2]) if len(labels1) > 0 and len(labels2) > 0 else (
                labels1 if len(labels1) > 0 else labels2
            )
        elif labels1 is not None:
            mixed_labels = labels1
        elif labels2 is not None:
            mixed_labels = labels2
        else:
            mixed_labels = np.array([]).reshape(0, 5)
        
        return mixed_image.astype(np.float32), mixed_labels


def create_train_augmentor() -> InfraredDataAugmentor:
    """
    创建训练用data增强器
    
    使用较强的data增强参数
    
    Returns:
        config好的训练data增强器
    """
    return InfraredDataAugmentor(
        brightness_range=(-0.3, 0.3),
        contrast_range=(0.7, 1.3),
        noise_intensity=0.03,
        blur_prob=0.2,
        flip_prob=0.5,
        rotation_angle=15.0,
        scale_range=(0.7, 1.3),
        crop_range=(0.7, 1.0),
        mosaic_prob=0.3,
        mixup_prob=0.1
    )


def create_val_augmentor() -> InfraredDataAugmentor:
    """
    创建验证用data增强器
    
    使用较弱的data增强参数，主要用于保持data一致性
    
    Returns:
        config好的验证data增强器
    """
    return InfraredDataAugmentor(
        brightness_range=(0.0, 0.0),
        contrast_range=(1.0, 1.0),
        noise_intensity=0.0,
        blur_prob=0.0,
        flip_prob=0.0,
        rotation_angle=0.0,
        scale_range=(1.0, 1.0),
        crop_range=(1.0, 1.0),
        mosaic_prob=0.0,
        mixup_prob=0.0
    )
