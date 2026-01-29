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
        
        rotation_angle_value = random.uniform(-self.rotation_angle, self.rotation_angle)
        
        # 角度过小时跳过旋转，避免不必要的image插值带来的质量loss
        if abs(rotation_angle_value) < min_angle_threshold:
            return image, labels
        
        import cv2
        
        height, width = image.shape[:2]
        center_point = (width / 2, height / 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center_point, rotation_angle_value, 1.0)
        
        # 旋转image
        image = cv2.warpAffine(
            image, rotation_matrix, (width, height),
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # 旋转label
        if labels is not None and len(labels) > 0:
            labels = self._rotate_labels(labels, rotation_angle_value, (width, height))
        
        return image, labels
    
    def _rotate_labels(
        self,
        labels: np.ndarray,
        rotation_angle: float,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        旋转label坐标
        
        Args:
            labels: label数组
            rotation_angle: 旋转角度（度）
            image_size: img_size (width, height)
            
        Returns:
            旋转后的label
        """
        image_width, image_height = image_size
        angle_radians = np.radians(rotation_angle)
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)
        
        # 中心点
        center_x, center_y = 0.5, 0.5
        
        rotated_labels = labels.copy()
        
        for i in range(len(labels)):
            # 获取边界框的四个角点（归一化坐标）
            box_center_x, box_center_y, box_width, box_height = labels[i, 1:5]
            
            box_corners = np.array([
                [box_center_x - box_width/2, box_center_y - box_height/2],
                [box_center_x + box_width/2, box_center_y - box_height/2],
                [box_center_x + box_width/2, box_center_y + box_height/2],
                [box_center_x - box_width/2, box_center_y + box_height/2]
            ])
            
            # 旋转角点
            corners_centered = box_corners - np.array([center_x, center_y])
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])
            corners_rotated = corners_centered @ rotation_matrix.T + np.array([center_x, center_y])
            
            # 计算新的边界框
            min_x, min_y = corners_rotated.min(axis=0)
            max_x, max_y = corners_rotated.max(axis=0)
            
            # 裁剪到有效范围
            min_x = np.clip(min_x, 0, 1)
            max_x = np.clip(max_x, 0, 1)
            min_y = np.clip(min_y, 0, 1)
            max_y = np.clip(max_y, 0, 1)
            
            # 更新label
            rotated_labels[i, 1] = (min_x + max_x) / 2
            rotated_labels[i, 2] = (min_y + max_y) / 2
            rotated_labels[i, 3] = max_x - min_x
            rotated_labels[i, 4] = max_y - min_y
        
        # 过滤无效label（面积太小）
        valid_mask = (rotated_labels[:, 3] > 0.01) & (rotated_labels[:, 4] > 0.01)
        rotated_labels = rotated_labels[valid_mask]
        
        return rotated_labels
    
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
                image_as_uint8 = (image * 255).astype(np.uint8)
                blurred_image = cv2.medianBlur(image_as_uint8, kernel_size)
                image = blurred_image.astype(np.float32) / 255.0
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
        
        original_height, original_width = image.shape[:2]
        
        # 随机裁剪比例，确保不超过1.0
        crop_ratio = random.uniform(self.crop_range[0], min(self.crop_range[1], 0.999))
        
        # 计算裁剪尺寸
        cropped_height = int(original_height * crop_ratio)
        cropped_width = int(original_width * crop_ratio)
        
        # 随机裁剪位置
        crop_top = random.randint(0, original_height - cropped_height)
        crop_left = random.randint(0, original_width - cropped_width)
        
        # 裁剪image
        image = image[crop_top:crop_top+cropped_height, crop_left:crop_left+cropped_width]
        
        # 调整大小回原始尺寸
        image = cv2.resize(image, (original_width, original_height))
        
        # 调整label
        if labels is not None and len(labels) > 0:
            labels = self._adjust_labels_for_crop(
                labels, (crop_left/original_width, crop_top/original_height), 
                (cropped_width/original_width, cropped_height/original_height)
            )
        
        return image, labels
    
    def _adjust_labels_for_crop(
        self,
        labels: np.ndarray,
        crop_offset: Tuple[float, float],
        crop_size_ratio: Tuple[float, float]
    ) -> np.ndarray:
        """
        调整裁剪后的label坐标
        
        Args:
            labels: label数组
            crop_offset: 裁剪偏移 (x_offset, y_offset)
            crop_size_ratio: 裁剪尺寸比例 (w_ratio, h_ratio)
            
        Returns:
            调整后的label
        """
        offset_x, offset_y = crop_offset
        ratio_width, ratio_height = crop_size_ratio
        
        adjusted_labels = labels.copy()
        
        # convert坐标
        adjusted_labels[:, 1] = (labels[:, 1] - offset_x) / ratio_width
        adjusted_labels[:, 2] = (labels[:, 2] - offset_y) / ratio_height
        adjusted_labels[:, 3] = labels[:, 3] / ratio_width
        adjusted_labels[:, 4] = labels[:, 4] / ratio_height
        
        # 裁剪到有效范围
        # 边界框左上角和右下角
        box_x1 = adjusted_labels[:, 1] - adjusted_labels[:, 3] / 2
        box_y1 = adjusted_labels[:, 2] - adjusted_labels[:, 4] / 2
        box_x2 = adjusted_labels[:, 1] + adjusted_labels[:, 3] / 2
        box_y2 = adjusted_labels[:, 2] + adjusted_labels[:, 4] / 2
        
        # 裁剪边界框
        box_x1 = np.clip(box_x1, 0, 1)
        box_y1 = np.clip(box_y1, 0, 1)
        box_x2 = np.clip(box_x2, 0, 1)
        box_y2 = np.clip(box_y2, 0, 1)
        
        # 更新label
        adjusted_labels[:, 1] = (box_x1 + box_x2) / 2
        adjusted_labels[:, 2] = (box_y1 + box_y2) / 2
        adjusted_labels[:, 3] = box_x2 - box_x1
        adjusted_labels[:, 4] = box_y2 - box_y1
        
        # 过滤无效label
        valid_mask = (adjusted_labels[:, 3] > 0.01) & (adjusted_labels[:, 4] > 0.01)
        valid_mask &= (adjusted_labels[:, 1] > 0) & (adjusted_labels[:, 1] < 1)
        valid_mask &= (adjusted_labels[:, 2] > 0) & (adjusted_labels[:, 2] < 1)
        
        return adjusted_labels[valid_mask]
    
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
        
        output_width, output_height = output_size
        
        # 随机选择中心点
        center_x = int(output_width * random.uniform(0.3, 0.7))
        center_y = int(output_height * random.uniform(0.3, 0.7))
        
        # 创建outputimage
        mosaic_image = np.zeros((output_height, output_width, 3) if images[0].ndim == 3 else (output_height, output_width), dtype=np.float32)
        
        # 合并的label
        mosaic_labels = []
        
        # 四个象限的位置
        quadrant_placements = [
            (0, 0, center_x, center_y),           # 左上
            (center_x, 0, output_width, center_y),           # 右上
            (0, center_y, center_x, output_height),           # 左下
            (center_x, center_y, output_width, output_height)            # 右下
        ]
        
        for i, (current_image, current_labels) in enumerate(zip(images, labels_list)):
            placement_x1, placement_y1, placement_x2, placement_y2 = quadrant_placements[i]
            placement_width, placement_height = placement_x2 - placement_x1, placement_y2 - placement_y1
            
            # 缩放image
            current_image_height, current_image_width = current_image.shape[:2]
            resized_image = cv2.resize(current_image, (placement_width, placement_height))
            
            # 放置image
            if mosaic_image.ndim == 3 and resized_image.ndim == 2:
                resized_image = np.stack([resized_image] * 3, axis=-1)
            mosaic_image[placement_y1:placement_y2, placement_x1:placement_x2] = resized_image
            
            # 调整label
            if current_labels is not None and len(current_labels) > 0:
                adjusted_current_labels = current_labels.copy()
                
                # 缩放label到当前象限
                adjusted_current_labels[:, 1] = (current_labels[:, 1] * placement_width + placement_x1) / output_width
                adjusted_current_labels[:, 2] = (current_labels[:, 2] * placement_height + placement_y1) / output_height
                adjusted_current_labels[:, 3] = current_labels[:, 3] * placement_width / output_width
                adjusted_current_labels[:, 4] = current_labels[:, 4] * placement_height / output_height
                
                mosaic_labels.append(adjusted_current_labels)
        
        # 合并所有label
        if mosaic_labels:
            mosaic_labels = np.vstack(mosaic_labels)
        else:
            mosaic_labels = np.array([]).reshape(0, 5)
        
        return mosaic_image, mosaic_labels
    
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
        lambda_value = np.random.beta(alpha, alpha)
        
        # 确保img_size一致
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 混合image
        mixed_image = lambda_value * image1 + (1 - lambda_value) * image2
        
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
