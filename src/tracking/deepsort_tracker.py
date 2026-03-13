#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT 跟踪器模块

使用 deep-sort-realtime 官方库实现，保持与 BaseTracker 接口兼容。
"""

import numpy as np

from .tracker import BaseTracker, TrackObject, TrackingResult

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
    _DEEPSORT_AVAILABLE = True
except ImportError:
    _DEEPSORT_AVAILABLE = False


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT 多目标跟踪器（官方库包装）

    使用 deep-sort-realtime 官方库，保持与项目 BaseTracker 接口兼容。

    Attributes:
        max_cosine_distance: 余弦距离门控阈值
        nn_budget: 外观描述符最大数量
        _tracker: 底层 deep-sort-realtime DeepSort 实例
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        device: str = '0',
    ):
        """
        初始化 DeepSORT 跟踪器

        Args:
            max_age: 目标最大存活帧数
            min_hits: 确认目标所需的最小命中次数
            iou_threshold: IoU 匹配阈值（用作 max_iou_distance）
            max_cosine_distance: 余弦距离门控阈值
            nn_budget: 外观描述符最大数量
            device: 计算设备 ('cpu' 或 GPU 索引字符串)
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self._embedder_gpu = device != 'cpu' and not device.lower().startswith('cpu')

        if not _DEEPSORT_AVAILABLE:
            raise ImportError(
                "deep-sort-realtime 未安装，请执行: pip install deep-sort-realtime"
            )

        self._tracker = _DeepSort(
            max_iou_distance=1.0 - iou_threshold,
            max_age=max_age,
            n_init=min_hits,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder='mobilenet',
            half=False,
            bgr=True,
            embedder_gpu=self._embedder_gpu,
        )

    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray = None,
        ori_img: np.ndarray = None,
        classes: np.ndarray = None,
    ):
        """
        更新跟踪器

        Args:
            detections: 检测框数组，形状为 (N, 4)，格式 [x1, y1, x2, y2]
            confidences: 置信度数组，形状为 (N,)
            ori_img: 原始图像帧（BGR），用于 ReID 特征提取
            classes: 类别 ID 数组，形状为 (N,)

        Returns:
            已确认或候选目标的 TrackObject 列表
        """
        self.frame_count += 1

        detections = np.asarray(detections)
        if detections.ndim == 1:
            detections = detections.reshape(-1, 4)

        if len(detections) == 0:
            if ori_img is not None:
                self._tracker.update_tracks([], frame=ori_img)
            return []

        if confidences is None:
            confidences = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)

        # 将 xyxy 转换为 ltwh（deep-sort-realtime 要求的格式）
        raw_dets = []
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = box
            ltwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            raw_dets.append((ltwh, float(confidences[i]), int(classes[i])))

        tracks = self._tracker.update_tracks(raw_dets, frame=ori_img)

        outputs = []
        for t in tracks:
            if not t.is_confirmed() and not t.is_tentative():
                continue
            bbox = t.to_tlbr()
            state = 'confirmed' if t.is_confirmed() else 'tentative'
            conf = t.det_conf if t.det_conf is not None else 1.0
            cls = int(t.det_class) if t.det_class is not None else 0
            outputs.append(TrackObject(
                track_id=t.track_id,
                bbox=np.array(bbox, dtype=float),
                confidence=float(conf),
                class_id=cls,
                state=state,
                age=t.age,
                hits=t.hits,
                time_since_update=t.time_since_update,
            ))
        return outputs

    def reset(self) -> None:
        """重置跟踪器状态"""
        super().reset()
        self._tracker = _DeepSort(
            max_iou_distance=1.0 - self.iou_threshold,
            max_age=self.max_age,
            n_init=self.min_hits,
            max_cosine_distance=self.max_cosine_distance,
            nn_budget=self.nn_budget,
            embedder='mobilenet',
            half=False,
            bgr=True,
            embedder_gpu=self._embedder_gpu,
        )


def create_deepsort_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_cosine_distance: float = 0.2,
    nn_budget: int = 100,
    device: str = '0',
) -> DeepSORTTracker:
    """
    创建 DeepSORT 跟踪器

    Args:
        max_age: 目标最大存活帧数
        min_hits: 确认所需的最小命中次数
        iou_threshold: IoU 匹配阈值
        max_cosine_distance: 余弦距离门控阈值
        nn_budget: 外观描述符最大数量
        device: 计算设备

    Returns:
        配置好的 DeepSORT 跟踪器
    """
    return DeepSORTTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        device=device,
    )