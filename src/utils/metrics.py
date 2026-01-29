"""
评估指标计算模块

提供目标检测和多目标跟踪的各种评估指标计算功能，包括IoU、精确率、召回率、AP、mAP等。
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的IoU（交并比）
    
    Args:
        box1: 第一个边界框，格式为 [x1, y1, x2, y2]
        box2: 第二个边界框，格式为 [x1, y1, x2, y2]
    
    Returns:
        IoU值，范围为 [0, 1]
    """
    # 确保输入为numpy数组
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    
    # 计算交集区域的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # 计算各自的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU，避免除零
    if union_area <= 0:
        return 0.0
    
    iou = inter_area / union_area
    return float(iou)


def compute_batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    批量计算两组边界框之间的IoU矩阵
    
    Args:
        boxes1: 第一组边界框，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
        boxes2: 第二组边界框，形状为 (M, 4)，格式为 [x1, y1, x2, y2]
    
    Returns:
        IoU矩阵，形状为 (N, M)，其中元素 [i, j] 表示 boxes1[i] 和 boxes2[j] 的IoU
    """
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    
    # 处理空输入
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    # 确保是二维数组
    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)
    
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    
    # 扩展维度以便广播计算
    # boxes1: (N, 1, 4), boxes2: (1, M, 4)
    boxes1_exp = boxes1[:, np.newaxis, :]
    boxes2_exp = boxes2[np.newaxis, :, :]
    
    # 计算交集区域
    x1_inter = np.maximum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    y1_inter = np.maximum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    x2_inter = np.minimum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    y2_inter = np.minimum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    # 计算交集面积
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算并集面积
    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area
    
    # 计算IoU，避免除零
    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0)
    
    return iou_matrix


def compute_precision_recall(
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    num_gt: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算精确率和召回率曲线
    
    Args:
        true_positives: 真正例数组，按置信度降序排列
        false_positives: 假正例数组，按置信度降序排列
        num_gt: 真实目标总数
    
    Returns:
        precision: 精确率数组
        recall: 召回率数组
    """
    # 累积求和
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    # 计算召回率
    recall = tp_cumsum / max(num_gt, 1)
    
    # 计算精确率
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
    
    return precision, recall


def compute_ap(precision: np.ndarray, recall: np.ndarray, use_07_metric: bool = False) -> float:
    """
    计算单个类别的平均精度(AP)
    
    Args:
        precision: 精确率数组
        recall: 召回率数组
        use_07_metric: 是否使用VOC2007的11点插值方法
    
    Returns:
        AP值
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    if use_07_metric:
        # VOC2007 11点插值法
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
    else:
        # VOC2010+ 所有点插值法
        # 在召回率序列前后添加哨兵值
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        
        # 使精确率单调递减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # 找到召回率变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # 计算AP（曲线下面积）
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return float(ap)


def compute_map(
    all_detections: Dict[int, List[Dict]],
    all_ground_truths: Dict[int, List[Dict]],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None
) -> Tuple[float, Dict[int, float]]:
    """
    计算所有类别的平均精度均值(mAP)
    
    Args:
        all_detections: 所有检测结果，格式为 {image_id: [{'bbox': [x1,y1,x2,y2], 'class_id': int, 'confidence': float}, ...]}
        all_ground_truths: 所有真实标注，格式为 {image_id: [{'bbox': [x1,y1,x2,y2], 'class_id': int}, ...]}
        iou_threshold: IoU阈值，用于判断检测是否正确
        num_classes: 类别数量，如果为None则自动推断
    
    Returns:
        mAP: 所有类别的平均精度均值
        ap_per_class: 每个类别的AP字典
    """
    # 收集所有类别ID
    class_ids = set()
    for dets in all_detections.values():
        for det in dets:
            class_ids.add(det['class_id'])
    for gts in all_ground_truths.values():
        for gt in gts:
            class_ids.add(gt['class_id'])
    
    if num_classes is not None:
        class_ids = set(range(num_classes))
    
    ap_per_class = {}
    
    for class_id in class_ids:
        # 收集该类别的所有检测和真实标注
        detections = []
        ground_truths = defaultdict(list)
        
        for img_id, dets in all_detections.items():
            for det in dets:
                if det['class_id'] == class_id:
                    detections.append({
                        'image_id': img_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
        
        for img_id, gts in all_ground_truths.items():
            for gt in gts:
                if gt['class_id'] == class_id:
                    ground_truths[img_id].append({
                        'bbox': gt['bbox'],
                        'used': False
                    })
        
        # 按置信度降序排列
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # 统计真实目标总数
        num_gt = sum(len(gts) for gts in ground_truths.values())
        
        if num_gt == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        # 计算TP和FP
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        for i, det in enumerate(detections):
            img_id = det['image_id']
            det_bbox = det['bbox']
            
            if img_id not in ground_truths:
                fp[i] = 1
                continue
            
            gts = ground_truths[img_id]
            
            # 找到IoU最大的真实框
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                iou = compute_iou(det_bbox, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # 判断是否为TP
            if max_iou >= iou_threshold and max_gt_idx >= 0 and not gts[max_gt_idx]['used']:
                tp[i] = 1
                gts[max_gt_idx]['used'] = True
            else:
                fp[i] = 1
        
        # 计算精确率和召回率
        precision, recall = compute_precision_recall(tp, fp, num_gt)
        
        # 计算AP
        ap = compute_ap(precision, recall)
        ap_per_class[class_id] = ap
    
    # 计算mAP
    if len(ap_per_class) > 0:
        mAP = sum(ap_per_class.values()) / len(ap_per_class)
    else:
        mAP = 0.0
    
    return mAP, ap_per_class


class MOTMetricsCalculator:
    """
    多目标跟踪(MOT)指标计算器
    
    支持计算MOTA、MOTP、IDF1等标准MOT评估指标
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        初始化MOT指标计算器
        
        Args:
            iou_threshold: IoU阈值，用于匹配预测和真实目标
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """
        重置所有累积的统计量
        """
        # 基础统计量
        self.num_gt = 0  # 真实目标总数
        self.num_pred = 0  # 预测目标总数
        self.num_matches = 0  # 匹配成功数
        self.num_false_positives = 0  # 假正例数
        self.num_misses = 0  # 漏检数
        self.num_switches = 0  # ID切换数
        self.num_fragmentations = 0  # 轨迹碎片数
        
        # 用于计算MOTP的IoU累积
        self.total_iou = 0.0
        
        # 用于计算IDF1的统计量
        self.idtp = 0  # ID真正例
        self.idfp = 0  # ID假正例
        self.idfn = 0  # ID假负例
        
        # 轨迹状态跟踪
        self.gt_id_to_pred_id = {}  # 真实ID到预测ID的映射
        self.prev_gt_ids = set()  # 上一帧的真实ID集合
        
        # 帧级统计
        self.frame_count = 0
    
    def update(
        self,
        gt_boxes: np.ndarray,
        gt_ids: np.ndarray,
        pred_boxes: np.ndarray,
        pred_ids: np.ndarray
    ):
        """
        更新单帧的统计量
        
        Args:
            gt_boxes: 真实边界框，形状为 (N, 4)
            gt_ids: 真实目标ID，形状为 (N,)
            pred_boxes: 预测边界框，形状为 (M, 4)
            pred_ids: 预测目标ID，形状为 (M,)
        """
        gt_boxes = np.asarray(gt_boxes)
        gt_ids = np.asarray(gt_ids)
        pred_boxes = np.asarray(pred_boxes)
        pred_ids = np.asarray(pred_ids)
        
        self.frame_count += 1
        
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        
        self.num_gt += num_gt
        self.num_pred += num_pred
        
        if num_gt == 0 and num_pred == 0:
            return
        
        if num_gt == 0:
            self.num_false_positives += num_pred
            self.idfp += num_pred
            return
        
        if num_pred == 0:
            self.num_misses += num_gt
            self.idfn += num_gt
            return
        
        # 计算IoU矩阵
        iou_matrix = compute_batch_iou(gt_boxes, pred_boxes)
        
        # 贪婪匹配（也可以使用匈牙利算法）
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        # 按IoU降序处理
        while True:
            max_iou = self.iou_threshold
            max_gt_idx = -1
            max_pred_idx = -1
            
            for i in range(num_gt):
                if i in matched_gt:
                    continue
                for j in range(num_pred):
                    if j in matched_pred:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_gt_idx = i
                        max_pred_idx = j
            
            if max_gt_idx < 0:
                break
            
            matches.append((max_gt_idx, max_pred_idx, max_iou))
            matched_gt.add(max_gt_idx)
            matched_pred.add(max_pred_idx)
        
        # 更新统计量
        self.num_matches += len(matches)
        self.num_false_positives += num_pred - len(matches)
        self.num_misses += num_gt - len(matches)
        
        # 更新MOTP
        for gt_idx, pred_idx, iou in matches:
            self.total_iou += iou
        
        # 检查ID切换
        current_gt_ids = set(gt_ids)
        for gt_idx, pred_idx, _ in matches:
            gt_id = gt_ids[gt_idx]
            pred_id = pred_ids[pred_idx]
            
            if gt_id in self.gt_id_to_pred_id:
                if self.gt_id_to_pred_id[gt_id] != pred_id:
                    self.num_switches += 1
            
            self.gt_id_to_pred_id[gt_id] = pred_id
        
        # 检查轨迹碎片化（目标消失后重新出现）
        for gt_id in current_gt_ids:
            if gt_id not in self.prev_gt_ids and gt_id in self.gt_id_to_pred_id:
                self.num_fragmentations += 1
        
        self.prev_gt_ids = current_gt_ids
        
        # 更新IDF1统计量
        self.idtp += len(matches)
        self.idfp += num_pred - len(matches)
        self.idfn += num_gt - len(matches)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有MOT指标
        
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        # MOTA (Multiple Object Tracking Accuracy)
        # MOTA = 1 - (FN + FP + IDSW) / GT
        if self.num_gt > 0:
            mota = 1 - (self.num_misses + self.num_false_positives + self.num_switches) / self.num_gt
        else:
            mota = 0.0
        metrics['MOTA'] = mota
        
        # MOTP (Multiple Object Tracking Precision)
        # MOTP = sum(IoU) / num_matches
        if self.num_matches > 0:
            motp = self.total_iou / self.num_matches
        else:
            motp = 0.0
        metrics['MOTP'] = motp
        
        # IDF1 (ID F1 Score)
        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        if (2 * self.idtp + self.idfp + self.idfn) > 0:
            idf1 = 2 * self.idtp / (2 * self.idtp + self.idfp + self.idfn)
        else:
            idf1 = 0.0
        metrics['IDF1'] = idf1
        
        # 精确率
        if self.num_pred > 0:
            precision = self.num_matches / self.num_pred
        else:
            precision = 0.0
        metrics['Precision'] = precision
        
        # 召回率
        if self.num_gt > 0:
            recall = self.num_matches / self.num_gt
        else:
            recall = 0.0
        metrics['Recall'] = recall
        
        # F1分数
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics['F1'] = f1
        
        # 其他统计量
        metrics['num_gt'] = self.num_gt
        metrics['num_pred'] = self.num_pred
        metrics['num_matches'] = self.num_matches
        metrics['num_false_positives'] = self.num_false_positives
        metrics['num_misses'] = self.num_misses
        metrics['num_switches'] = self.num_switches
        metrics['num_fragmentations'] = self.num_fragmentations
        metrics['num_frames'] = self.frame_count
        
        return metrics


def save_metrics_to_json(metrics: Dict, filepath: str, indent: int = 2):
    """
    将指标保存到JSON文件
    
    Args:
        metrics: 指标字典
        filepath: 保存路径
        indent: JSON缩进空格数
    """
    # 将numpy类型转换为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=indent, ensure_ascii=False)


def load_metrics_from_json(filepath: str) -> Dict:
    """
    从JSON文件加载指标
    
    Args:
        filepath: JSON文件路径
    
    Returns:
        指标字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics
