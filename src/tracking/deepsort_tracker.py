#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT 跟踪器模块 (最终修复版)
修复: 缺失 n_init 属性导致的崩溃
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2

# 使用相对导入
from .tracker import BaseTracker, TrackObject
from .kalman_filter import KalmanFilter, xyxy_to_xyah, xyah_to_xyxy


class ReIDExtractor:
    def __init__(self, device='cpu', use_half=False):
        self.device = device
        self.use_half = use_half and device != 'cpu'
        try:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.fc = torch.nn.Sequential()
            self.model.to(self.device)
            self.model.eval()
            if self.use_half: self.model.half()
            self.size = (128, 64)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            print(f"❌ ReID 模型加载失败: {e}")
            self.model = None

    def _preprocess(self, im_crops):
        if len(im_crops) == 0: return None
        im_batch = torch.cat([
            self.norm(cv2.resize(im, (self.size[1], self.size[0]))).unsqueeze(0)
            for im in im_crops
        ], dim=0).float()
        return im_batch.to(self.device)

    def extract(self, img, boxes):
        if img is None or self.model is None or len(boxes) == 0:
            return np.zeros((len(boxes), 512))

        im_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            else:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
            im_crops.append(crop)

        try:
            im_batch = self._preprocess(im_crops)
            if self.use_half: im_batch = im_batch.half()
            with torch.no_grad():
                features = self.model(im_batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy()
        except:
            return np.zeros((len(boxes), 512))


class DeepSORTTrack:
    def __init__(self, track_id, bbox, feature=None, class_id=0, confidence=1.0, n_init=3):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.kalman_filter = KalmanFilter()
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)
        self.features = []
        if feature is not None: self.features.append(feature)
        self.max_features = 100
        self.state = 'tentative'
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, feature=None, confidence=1.0):
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > self.max_features: self.features.pop(0)
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > 60:
            self.state = 'deleted'

    def is_confirmed(self):
        return self.state == 'confirmed'

    def is_deleted(self):
        return self.state == 'deleted'

    def get_bbox(self):
        return xyah_to_xyxy(self.mean[:4])

    def to_track_object(self):
        return TrackObject(self.track_id, self.get_bbox(), self.confidence, self.class_id, self.state, self.age,
                           self.hits, self.time_since_update)


class DeepSORTTracker(BaseTracker):
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, max_cosine_distance=0.2, nn_budget=100, device='0'):
        super().__init__(max_age, min_hits, iou_threshold)
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        # ========== 【修复核心】 ==========
        self.n_init = min_hits
        # ================================

        self.tracks = []
        torch_device = 'cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        self.extractor = ReIDExtractor(device=torch_device)

    def update(self, detections, confidences=None, ori_img=None, classes=None):
        self.frame_count += 1
        if len(detections) == 0:
            self.tracker_predict()
            return []

        if confidences is None: confidences = np.ones(len(detections))
        if classes is None: classes = np.zeros(len(detections))

        features = self.extractor.extract(ori_img, detections)
        self.tracker_predict()

        matches, unmatched_tracks, unmatched_detections = self._match(detections, features)

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx], features[d_idx], confidences[d_idx])
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()
        for d_idx in unmatched_detections:
            self._initiate_track(detections[d_idx], features[d_idx], classes[d_idx], confidences[d_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        outputs = []
        for t in self.tracks:
            if t.is_confirmed() or (t.state == 'tentative' and t.hits >= 1):
                outputs.append(t.to_track_object())
        return outputs

    def tracker_predict(self):
        for track in self.tracks: track.predict()

    def _match(self, detections, features):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 级联匹配
        matches_a = []
        unmatched_tracks_a = []
        unmatched_detections = list(range(len(detections)))

        if len(confirmed_tracks) > 0 and len(detections) > 0:
            dist_matrix = self._cosine_distance(confirmed_tracks, features)
            import scipy.optimize
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(dist_matrix)

            for r, c in zip(row_indices, col_indices):
                if dist_matrix[r, c] > self.max_cosine_distance:
                    unmatched_tracks_a.append(confirmed_tracks[r])
                else:
                    matches_a.append((confirmed_tracks[r], c))

            matched_det_indices = [c for _, c in matches_a]
            unmatched_detections = [d for d in unmatched_detections if d not in matched_det_indices]
            matched_track_indices = [r for r, _ in matches_a]
            for t_idx in confirmed_tracks:
                if t_idx not in matched_track_indices: unmatched_tracks_a.append(t_idx)
        else:
            unmatched_tracks_a = confirmed_tracks

        # IoU 匹配
        iou_track_candidates = unconfirmed_tracks + unmatched_tracks_a
        unmatched_tracks_b = []
        matches_b = []

        if len(iou_track_candidates) > 0 and len(unmatched_detections) > 0:
            track_boxes = np.array([self.tracks[i].get_bbox() for i in iou_track_candidates])
            det_boxes = detections[unmatched_detections]
            iou_matrix = 1 - self.compute_iou_matrix(track_boxes, det_boxes)

            import scipy.optimize
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(iou_matrix)

            for r, c in zip(row_indices, col_indices):
                if iou_matrix[r, c] > self.iou_threshold: continue
                matches_b.append((iou_track_candidates[r], unmatched_detections[c]))

            matched_iou_tracks = [t for t, _ in matches_b]
            matched_iou_dets = [d for _, d in matches_b]
            for t_idx in iou_track_candidates:
                if t_idx not in matched_iou_tracks: unmatched_tracks_b.append(t_idx)
            final_unmatched_dets = [d for d in unmatched_detections if d not in matched_iou_dets]
        else:
            unmatched_tracks_b = iou_track_candidates
            final_unmatched_dets = unmatched_detections

        return matches_a + matches_b, unmatched_tracks_b, final_unmatched_dets

    def _cosine_distance(self, track_indices, features):
        cost_matrix = np.zeros((len(track_indices), len(features)))
        for i, t_idx in enumerate(track_indices):
            track = self.tracks[t_idx]
            track_feats = np.array(track.features)
            sim = np.dot(features, track_feats.T)
            cost_matrix[i, :] = np.min(1. - sim, axis=1)
        return cost_matrix

    def _initiate_track(self, bbox, feature, class_id, conf):
        self.tracks.append(DeepSORTTrack(
            self.get_next_id(), bbox, feature, class_id, conf, self.n_init
        ))


def create_deepsort_tracker(max_age=30, min_hits=3, iou_threshold=0.3, max_cosine_distance=0.2, nn_budget=100):
    return DeepSORTTracker(max_age, min_hits, iou_threshold, max_cosine_distance, nn_budget)