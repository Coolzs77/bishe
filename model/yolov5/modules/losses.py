#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5优化损失函数模块
包含：EIoU Loss, Focal Loss, 组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EIoULoss(nn.Module):
    """
    Efficient IoU Loss (EIoU Loss)
    论文: https://arxiv.org/abs/2101.08663

    比CIoU更稳定，对小目标友好
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算EIoU Loss

        参数:
            pred: 预测框 [N, 4] (x1, y1, x2, y2)
            target: 目标框 [N, 4] (x1, y1, x2, y2)
        """
        # 计算IoU
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        inter_area = inter_w * inter_h

        # 预测框和目标框的面积
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        pred_area = pred_w * pred_h

        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
        target_area = target_w * target_h

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)

        # 中心点距离
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2

        center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        center_dist = torch.sqrt(center_dist_sq + 1e-7)

        # 外接矩形对角线
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])

        enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        enclose_diag = torch.sqrt(enclose_diag_sq + 1e-7)

        # 宽度和高度的差异
        cw = torch.max(pred_w, target_w)
        ch = torch.max(pred_h, target_h)

        lw = torch.abs(pred_w - target_w) / (cw + 1e-7)
        lh = torch.abs(pred_h - target_h) / (ch + 1e-7)

        # EIoU Loss = 1 - IoU + center_dist/diag + width_diff + height_diff
        loss = 1 - iou + center_dist / (enclose_diag + 1e-7) + lw + lh

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    论文: https://arxiv.org/abs/1708.02002

    用于Objectness预测，解决背景/目标样本不平衡
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算Focal Loss

        参数:
            pred: [N] 预测logits
            target: [N] 目标标签 (0 或 1)
        """
        p = torch.sigmoid(pred)

        # 数值稳定性
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)

        # 正样本: alpha * (1-p)^gamma * log(p)
        pos_loss = -self.alpha * ((1 - p) ** self.gamma) * torch.log(p) * target

        # 负样本: (1-alpha) * p^gamma * log(1-p)
        neg_loss = -(1 - self.alpha) * (p ** self.gamma) * torch.log(1 - p) * (1 - target)

        focal_loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ComputeLoss:
    """
    YOLOv5损失函数计算器 - 支持优化的损失

    可配置使用不同的损失函数组合
    """

    def __init__(self, model, autobalance=False, use_eiou=False, use_focal=False):
        """
        初始化损失计算器

        参数:
            model: YOLOv5模型
            autobalance: 是否自动平衡各分支的损失权重
            use_eiou: 是否使用EIoU Loss (默认使用CIoU)
            use_focal: 是否使用Focal Loss (默认使用BCEWithLogitsLoss)
        """
        device = next(model.parameters()).device
        h = model.hyp  # 超参数

        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # 如果使用Focal Loss替换Objectness Loss
        if use_focal:
            self.BCEobj = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.BCEobj = BCEobj

        self.BCEcls = BCEcls

        # 如果使用EIoU Loss替换CIoU Loss
        if use_eiou:
            self.bbox_loss = EIoULoss()
        else:
            # 标准YOLOv5使用CIoU Loss（内置在detect.py中）
            self.bbox_loss = None  # 由detect.py中的compute_loss处理

        self.use_eiou = use_eiou
        self.use_focal = use_focal

        # 损失权重
        self.balance = [4.0, 1.0, 0.4]  # P3, P4, P5
        self.ssi = list(model.stride).index(16) if autobalance else 0  # stride 16 index

        self.BCEcls.pos_weight = torch.tensor([h['cls_pw']], device=device)
        self.BCEobj.pos_weight = torch.tensor([h['obj_pw']], device=device)

        self.device = device
        self.hyp = h
        self.autobalance = autobalance

    def __call__(self, p, targets):
        """
        计算YOLOv5损失

        参数:
            p: 预测输出 [(bs,na,ny,nx,no), ...]
            targets: 目标 [image_index, class, x, y, w, h]
        """
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), \
            torch.zeros(1, device=self.device)

        # 这里应该集成YOLOv5的标准损失计算
        # 为了支持EIoU和Focal Loss的灵活切换

        return lcls + lbox + lobj