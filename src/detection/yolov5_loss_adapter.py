#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5损失函数适配器
在YOLOv5训练中集成EIoU和Focal Loss
"""

from models.yolov5.modules.losses import EIoULoss, FocalLoss


class YOLOv5LossAdapter:
    """YOLOv5损失函数适配器"""
    
    def __init__(self, model, use_eiou=False, use_focal=False):
        """
        初始化损失适配器
        
        参数:
            model: YOLOv5模型
            use_eiou: 是否使用EIoU Loss
            use_focal: 是否使用Focal Loss
        """
        self.model = model
        self.use_eiou = use_eiou
        self.use_focal = use_focal
        
        self.eiou_loss = EIoULoss() if use_eiou else None
        self.focal_loss = FocalLoss() if use_focal else None
        
        self.device = next(model.parameters()).device
    
    def compute_bbox_loss(self, pred_bbox, target_bbox):
        """计算边界框损失"""
        if self.use_eiou:
            return self.eiou_loss(pred_bbox, target_bbox)
        else:
            # 使用YOLOv5原生的CIoU Loss
            return None  # 由YOLOv5内部处理
    
    def compute_obj_loss(self, pred_obj, target_obj):
        """计算Objectness损失"""
        if self.use_focal:
            return self.focal_loss(pred_obj, target_obj)
        else:
            # 使用YOLOv5原生的BCE Loss
            return None  # 由YOLOv5内部处理
    
    def get_loss_config_str(self):
        """获取损失函数配置字符串"""
        configs = []
        if self.use_eiou:
            configs.append("EIoU Loss")
        if self.use_focal:
            configs.append("Focal Loss")
        
        if not configs:
            return "Standard YOLOv5 Loss"
        
        return " + ".join(configs)