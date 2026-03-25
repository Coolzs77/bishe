#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪算法评估脚本 (改进版)
添加了调试信息和参数优化
支持 DeepSORT、ByteTrack 和 CenterTrack
"""

import os
import sys
import argparse
import time
import traceback
import csv
import json
from pathlib import Path
from datetime import datetime
import yaml
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

# ==============================================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ==============================================================================

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'跟踪评估配置不存在: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def config_get(config, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def pick(cli_value, config_value):
    return cli_value if cli_value is not None else config_value


def resolve_args(args):
    config = load_config(args.config)

    args.weights = pick(args.weights, config_get(config, 'detector', 'weights'))
    args.data = pick(args.data, config_get(config, 'detector', 'data'))
    args.tracker = pick(args.tracker, config_get(config, 'runtime', 'tracker', default='deepsort'))
    args.output = pick(args.output, config_get(config, 'runtime', 'output'))
    args.save_vid = pick(args.save_vid, config_get(config, 'artifacts', 'save_vid', default=True))
    args.save_txt = pick(args.save_txt, config_get(config, 'artifacts', 'save_txt', default=True))
    args.show = pick(args.show, config_get(config, 'runtime', 'show', default=False))
    overlay = pick(args.overlay, config_get(config, 'runtime', 'overlay', default=True))
    args.no_overlay = not overlay
    args.conf_thres = pick(args.conf_thres, config_get(config, 'detector', 'conf_thres'))
    args.nms_thres = pick(args.nms_thres, config_get(config, 'detector', 'nms_thres'))
    args.img_size = pick(args.img_size, config_get(config, 'detector', 'img_size'))
    args.half = pick(args.half, config_get(config, 'detector', 'half', default=False))
    warmup = pick(args.warmup, config_get(config, 'detector', 'warmup', default=True))
    args.no_warmup = not warmup
    args.device = pick(args.device, config_get(config, 'runtime', 'device'))
    args.max_age = pick(args.max_age, config_get(config, 'trackers', 'common', 'max_age'))
    args.min_hits = pick(args.min_hits, config_get(config, 'trackers', 'common', 'min_hits'))
    args.fps_alpha = pick(args.fps_alpha, config_get(config, 'runtime', 'fps_alpha', default=0.12))
    args.debug = pick(args.debug, config_get(config, 'runtime', 'debug', default=False))

    tracker_overrides = config_get(config, 'trackers', args.tracker, default={}) or {}
    args.track_visible_lag = pick(
        args.track_visible_lag,
        config_get(tracker_overrides, 'visible_lag', default=config_get(config, 'trackers', 'common', 'visible_lag')),
    )
    args.tracker_iou_thres = config_get(tracker_overrides, 'iou_threshold', default=0.3)
    args.deepsort_max_cosine_distance = config_get(tracker_overrides, 'max_cosine_distance', default=0.2)
    args.deepsort_nn_budget = config_get(tracker_overrides, 'nn_budget', default=100)
    args.bytetrack_high_thres = config_get(tracker_overrides, 'high_threshold', default=0.5)
    args.bytetrack_low_thres = config_get(tracker_overrides, 'low_threshold', default=0.1)
    args.bytetrack_match_thres = config_get(tracker_overrides, 'match_threshold', default=0.3)
    args.bytetrack_second_match_thres = config_get(tracker_overrides, 'second_match_threshold', default=0.2)
    args.centertrack_center_thres = config_get(tracker_overrides, 'center_threshold', default=50.0)
    args.centertrack_pre_thres = config_get(tracker_overrides, 'pre_threshold', default=0.3)

    if args.tracker not in {'deepsort', 'bytetrack', 'centertrack'}:
        raise ValueError(f'不支持的跟踪器: {args.tracker}')

    missing = []
    for name in ['weights', 'data', 'output', 'conf_thres', 'nms_thres', 'img_size', 'device', 'max_age', 'min_hits', 'track_visible_lag']:
        if getattr(args, name) is None:
            missing.append(name)
    if missing:
        raise ValueError(f'跟踪评估配置缺少必要字段: {", ".join(sorted(set(missing)))}')

    return args


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多目标跟踪评估与可视化')

    parser.add_argument('--config', type=str, default='configs/tracking_config.yaml',
                        help='跟踪评估配置文件路径')

    # 核心输入
    parser.add_argument('--weights', type=str, default=None, help='YOLOv5 检测器权重路径 (.pt)')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集配置文件 或 视频文件路径 或 视频目录路径')
    parser.add_argument('--tracker', type=str, default=None, choices=['deepsort', 'bytetrack', 'centertrack'],
                        help='选择跟踪算法')

    # 输出设置
    parser.add_argument('--output', type=str, default=None, help='结果输出目录')
    parser.add_argument('--save-vid', action=argparse.BooleanOptionalAction, default=None,
                        help='是否保存 MP4 视频')
    parser.add_argument('--save-txt', action=argparse.BooleanOptionalAction, default=None,
                        help='是否保存 MOT 格式的 txt 指标文件')
    parser.add_argument('--show', action=argparse.BooleanOptionalAction, default=None, help='是否实时弹窗显示')
    parser.add_argument('--overlay', action=argparse.BooleanOptionalAction, default=None,
                        help='是否绘制框和HUD')

    # 检测参数
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='检测置信度阈值 (降低此值可以检测更多目标)')
    parser.add_argument('--nms-thres', type=float, default=None, help='NMS 阈值')
    parser.add_argument('--img-size', type=int, default=None, help='检测输入尺寸（推荐 512/448 提速）')
    parser.add_argument('--half', action=argparse.BooleanOptionalAction, default=None, help='开启半精度推理')
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=None,
                        help='是否执行模型预热（仅影响启动时延）')
    parser.add_argument('--device', type=str, default=None, help='计算设备')

    # 跟踪参数
    parser.add_argument('--max-age', type=int, default=None,
                        help='目标最大失踪帧数 (越大越容易保持ID)')
    parser.add_argument('--min-hits', type=int, default=None,
                        help='目标最少检测次数 (越小越容易出现)')
    parser.add_argument('--track-visible-lag', type=int, default=None,
                        help='目标漏检后继续显示的最大帧数，用于减少跟踪框时有时无')
    parser.add_argument('--fps-alpha', type=float, default=None,
                        help='FPS平滑系数(0-1)，越小越稳定')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=None, help='是否输出调试信息')

    return resolve_args(parser.parse_args())


class TrackingRunner:
    """跟踪流程管理器"""

    def __init__(self, args):
        self.args = args

        # 初始化设备
        self.device = self._select_device(args.device)

        # 加载模型
        self.detector = self._load_detector()
        self.class_names = getattr(self.detector, 'class_names', []) or []
        self.tracker = None

    def _select_device(self, device):
        """处理设备字符串"""
        if device.isdigit():
            return f'cuda:{device}'
        return device

    def _load_detector(self):
        """加载 YOLOv5 检测器"""
        print(f'\n[1/3] 加载检测器: {self.args.weights}')
        try:
            from src.detection.yolov5_detector import create_yolov5_detector
            return create_yolov5_detector(
                model_path=str(self.args.weights),
                input_size=(self.args.img_size, self.args.img_size),
                conf_threshold=self.args.conf_thres,
                nms_threshold=self.args.nms_thres,
                device=self.device,
                half=self.args.half,
                warmup=not self.args.no_warmup,
            )
        except Exception as e:
            print(f'❌ 检测器加载失败: {e}')
            traceback.print_exc()
            sys.exit(1)

    def _create_tracker(self):
        """初始化跟踪器 (DeepSORT / ByteTrack / CenterTrack)"""
        print(f'[2/3] 初始化跟踪器: {self.args.tracker}')
        print(f'      参数: max_age={self.args.max_age}, min_hits={self.args.min_hits}')
        try:
            if self.args.tracker == 'deepsort':
                from src.tracking.deepsort_tracker import create_deepsort_tracker
                return create_deepsort_tracker(
                    max_age=self.args.max_age,
                    min_hits=self.args.min_hits,
                    iou_threshold=self.args.tracker_iou_thres,
                    max_cosine_distance=self.args.deepsort_max_cosine_distance,
                    nn_budget=self.args.deepsort_nn_budget,
                    visible_lag=self.args.track_visible_lag,
                )

            elif self.args.tracker == 'bytetrack':
                from src.tracking.bytetrack_tracker import create_bytetrack_tracker
                return create_bytetrack_tracker(
                    max_age=self.args.max_age,
                    min_hits=self.args.min_hits,
                    iou_threshold=self.args.tracker_iou_thres,
                    high_threshold=self.args.bytetrack_high_thres,
                    low_threshold=self.args.bytetrack_low_thres,
                    match_threshold=self.args.bytetrack_match_thres,
                    second_match_threshold=self.args.bytetrack_second_match_thres,
                    visible_lag=self.args.track_visible_lag,
                )

            elif self.args.tracker == 'centertrack':
                from src.tracking.centertrack_tracker import create_centertrack_tracker
                return create_centertrack_tracker(
                    max_age=self.args.max_age,
                    min_hits=self.args.min_hits,
                    iou_threshold=self.args.tracker_iou_thres,
                    center_threshold=self.args.centertrack_center_thres,
                    pre_thresh=self.args.centertrack_pre_thres,
                    visible_lag=self.args.track_visible_lag,
                )

        except Exception as e:
            print(f'❌ 跟踪器初始化失败: {e}')
            traceback.print_exc()
            sys.exit(1)

    def _safe_read_image(self, path):
        """稳健的图片读取方法"""
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _get_video_list(self, source_path):
        """获取视频列表"""
        source = Path(source_path)
        video_files = []

        if source.is_file() and source.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
            video_files = [source]

        elif source.is_dir():
            video_exts = ['.mp4', '.avi', '.mkv', '.mov']
            video_files = sorted([f for f in source.glob('*') if f.suffix.lower() in video_exts])

            if not video_files:
                print(f'❌ 目录 {source} 中找不到视频文件')
                return []

        else:
            print(f'❌ 路径不存在或格式不支持: {source}')
            return []

        print(f'✓ 找到 {len(video_files)} 个视频文件')
        return video_files

    def _get_dataloader(self, video_path):
        """解析输入源，返回数据迭代器"""
        source = Path(video_path)
        is_video = source.is_file() and source.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']

        if source.suffix.lower() in ['.yaml', '.yml']:
            print(f'   正在解析配置文件: {source}')
            try:
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                except UnicodeDecodeError:
                    with open(source, 'r', encoding='gbk') as f:
                        config = yaml.safe_load(f)
            except Exception as e:
                print(f'❌ 配置文件读取失败: {e}')
                sys.exit(1)

            path_key = 'test' if 'test' in config else 'val'
            if path_key not in config:
                print('❌ 错误: 配置文件中未找到 test 或 val 路径')
                sys.exit(1)

            base_path = Path(config.get('path', ''))
            rel_path = config[path_key]

            candidates = [
                (base_path / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path),
                Path(video_path).parent / rel_path,
                Path(rel_path)
            ]

            target_path = None
            for p in candidates:
                if p.exists():
                    target_path = p
                    break

            if not target_path:
                print(f'❌ 找不到图片目录')
                sys.exit(1)

            source = target_path
            is_video = False

        print(f'[3/3] 数据源确定: {source}')

        if is_video:
            cap = cv2.VideoCapture(str(source))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return cap, fps, w, h, total_frames, True
        else:
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            files = sorted([p for p in source.rglob('*') if p.suffix.lower() in valid_exts])

            if not files:
                print(f'❌ 文件夹 {source} 为空或不包含图片')
                sys.exit(1)

            img0 = self._safe_read_image(str(files[0]))
            if img0 is None:
                print('❌ 无法读取第一张图片')
                sys.exit(1)

            h, w = img0.shape[:2]
            return files, 30, w, h, len(files), False

    def _class_name(self, class_id):
        class_id = int(class_id)
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f'cls{class_id}'

    @staticmethod
    def _draw_hud_chip(
        img,
        text,
        x,
        y,
        *,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.62,
        text_color=(232, 241, 255),
        bg_color=(28, 34, 44),
        alpha=0.36,
        pad_x=10,
        pad_y=8,
        radius=8,
        accent_color=(94, 201, 255),
    ):
        """绘制现代HUD胶囊标签（半透明+圆角+细强调条）。"""
        text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
        w = text_size[0] + pad_x * 2
        h = text_size[1] + pad_y * 2

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # 边界保护
        x1 = max(0, min(x1, img.shape[1] - 2))
        x2 = max(x1 + 1, min(x2, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 2))
        y2 = max(y1 + 1, min(y2, img.shape[0] - 1))

        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        overlay = img.copy()

        # 中心矩形
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), bg_color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), bg_color, -1)
        # 四角圆
        cv2.circle(overlay, (x1 + r, y1 + r), r, bg_color, -1)
        cv2.circle(overlay, (x2 - r, y1 + r), r, bg_color, -1)
        cv2.circle(overlay, (x1 + r, y2 - r), r, bg_color, -1)
        cv2.circle(overlay, (x2 - r, y2 - r), r, bg_color, -1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 左侧强调条
        cv2.line(img, (x1 + 5, y1 + 5), (x1 + 5, y2 - 5), accent_color, 2, cv2.LINE_AA)

        # 文本轻描边
        org = (x1 + pad_x, y2 - pad_y)
        cv2.putText(img, text, org, font, font_scale, (12, 14, 18), 2, cv2.LINE_AA)
        cv2.putText(img, text, org, font, font_scale, text_color, 1, cv2.LINE_AA)

        return x2, y2

    @staticmethod
    def _compute_iou_matrix(boxes1, boxes2):
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        b1 = np.asarray(boxes1, dtype=np.float32)
        b2 = np.asarray(boxes2, dtype=np.float32)

        x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
        y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
        x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
        y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

        area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
        area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)

    def _estimate_id_switches(self, prev_entries, curr_entries, iou_thres=0.5):
        """基于相邻帧IoU关联估算ID切换次数（无GT场景下的代理指标）。"""
        if not prev_entries or not curr_entries:
            return 0

        prev_boxes = [e['bbox'] for e in prev_entries]
        curr_boxes = [e['bbox'] for e in curr_entries]
        iou = self._compute_iou_matrix(prev_boxes, curr_boxes)

        candidates = []
        for i in range(iou.shape[0]):
            for j in range(iou.shape[1]):
                if iou[i, j] >= iou_thres:
                    # 同类目标优先比较，跨类不参与ID切换统计
                    if prev_entries[i]['class_id'] == curr_entries[j]['class_id']:
                        candidates.append((float(iou[i, j]), i, j))

        if not candidates:
            return 0

        candidates.sort(reverse=True, key=lambda x: x[0])
        used_prev = set()
        used_curr = set()
        switches = 0

        for _, i, j in candidates:
            if i in used_prev or j in used_curr:
                continue
            used_prev.add(i)
            used_curr.add(j)
            if int(prev_entries[i]['track_id']) != int(curr_entries[j]['track_id']):
                switches += 1

        return switches

    @staticmethod
    def _safe_ratio(num, den):
        return float(num) / float(den) if den else 0.0

    def _finalize_video_stats(self, stats, input_fps):
        frames = max(1, stats['frame_count'])
        elapsed = max(stats.get('elapsed_sec', 0.0), 1e-9)
        core_elapsed = max(stats.get('core_elapsed_sec', 0.0), 1e-9)
        # FPS 仅基于检测+跟踪核心耗时，排除视频I/O、可视化、编码
        avg_fps = stats['frame_count'] / core_elapsed

        stats['avg_fps'] = avg_fps
        stats['wall_fps'] = stats['frame_count'] / elapsed  # 含I/O的壁钟FPS（仅参考）
        stats['input_fps'] = float(input_fps)
        stats['realtime_factor'] = self._safe_ratio(avg_fps, max(float(input_fps), 1e-9))
        stats['match_rate'] = self._safe_ratio(stats['matched_detections'], stats['total_detections'])
        stats['render_rate'] = self._safe_ratio(stats['tracked_detections'], stats['total_detections'])
        stats['track_presence_rate'] = self._safe_ratio(stats['frames_with_tracks'], stats['frame_count'])
        stats['det_presence_rate'] = self._safe_ratio(stats['frames_with_detections'], stats['frame_count'])
        stats['avg_rendered_per_frame'] = self._safe_ratio(stats['tracked_detections'], frames)
        stats['avg_matched_per_frame'] = self._safe_ratio(stats['matched_detections'], frames)
        stats['unique_ids'] = len(stats['total_tracks'])
        stats['id_switch_proxy'] = int(stats.get('id_switch_proxy', 0))

    def _save_video_metrics(self, video_save_dir, video_name, stats):
        metric = {
            'video_name': video_name,
            'frame_count': stats['frame_count'],
            'total_detections': stats['total_detections'],
            'rendered_tracks': stats['tracked_detections'],
            'matched_tracks': stats['matched_detections'],
            'unique_ids': stats['unique_ids'],
            'id_switch_proxy': stats['id_switch_proxy'],
            'avg_fps': round(stats['avg_fps'], 3),
            'input_fps': round(stats['input_fps'], 3),
            'realtime_factor': round(stats['realtime_factor'], 3),
            'match_rate': round(stats['match_rate'], 6),
            'render_rate': round(stats['render_rate'], 6),
            'track_presence_rate': round(stats['track_presence_rate'], 6),
            'det_presence_rate': round(stats['det_presence_rate'], 6),
            'avg_rendered_per_frame': round(stats['avg_rendered_per_frame'], 6),
            'avg_matched_per_frame': round(stats['avg_matched_per_frame'], 6),
            'elapsed_sec': round(stats['elapsed_sec'], 3),
        }
        out = video_save_dir / 'metrics.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(metric, f, ensure_ascii=False, indent=2)

    def _save_batch_summary(self, main_save_dir, all_stats):
        csv_path = main_save_dir / 'summary_metrics.csv'
        json_path = main_save_dir / 'summary_metrics.json'

        rows = []
        for video_name, stats in all_stats:
            rows.append({
                'video_name': video_name,
                'frame_count': stats['frame_count'],
                'total_detections': stats['total_detections'],
                'matched_tracks': stats['matched_detections'],
                'rendered_tracks': stats['tracked_detections'],
                'unique_ids': stats['unique_ids'],
                'id_switch_proxy': stats['id_switch_proxy'],
                'avg_fps': round(stats['avg_fps'], 3),
                'input_fps': round(stats['input_fps'], 3),
                'realtime_factor': round(stats['realtime_factor'], 3),
                'match_rate': round(stats['match_rate'], 6),
                'render_rate': round(stats['render_rate'], 6),
                'track_presence_rate': round(stats['track_presence_rate'], 6),
            })

        if rows:
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            totals = {
                'video_count': len(rows),
                'frame_count': int(sum(r['frame_count'] for r in rows)),
                'total_detections': int(sum(r['total_detections'] for r in rows)),
                'matched_tracks': int(sum(r['matched_tracks'] for r in rows)),
                'rendered_tracks': int(sum(r['rendered_tracks'] for r in rows)),
                'unique_ids_sum': int(sum(r['unique_ids'] for r in rows)),
                'id_switch_proxy_sum': int(sum(r['id_switch_proxy'] for r in rows)),
            }
            totals['match_rate'] = self._safe_ratio(totals['matched_tracks'], totals['total_detections'])
            totals['avg_fps_mean'] = float(np.mean([r['avg_fps'] for r in rows]))
            totals['realtime_factor_mean'] = float(np.mean([r['realtime_factor'] for r in rows]))

            payload = {
                'tracker': self.args.tracker,
                'rows': rows,
                'totals': totals,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    def draw_tracks(self, img, tracks, detections=None):
        """
        核心可视化函数
        绘制：检测框、ID标签、检测置信度
        """
        # 如果有调试模式，先画所有检测框（浅色）
        if self.args.debug and detections is not None:
            for det in detections:
                x1, y1, x2, y2 = [int(i) for i in det[:4]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)  # 灰色虚框

        # 画跟踪框
        for track in tracks:
            if hasattr(track, 'bbox'):
                bbox = track.bbox
                tid = track.track_id
            elif hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()
                tid = track.track_id
            else:
                continue

            x1, y1, x2, y2 = [int(i) for i in bbox]

            # 根据 ID 获取颜色
            color = [int(c) for c in COLORS[tid % len(COLORS)]]

            # 漏检延续帧用更细框显示，减少视觉干扰
            tsu = int(getattr(track, 'time_since_update', 0))
            thickness = 2 if tsu == 0 else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            class_id = int(getattr(track, 'class_id', -1))
            class_name = self._class_name(class_id)
            short_cls = class_name if len(class_name) <= 8 else class_name[:8]
            label = f'ID:{tid} {short_cls}'

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            font_thickness = 1
            text_w, text_h = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

            # 标签优先绘制在框上方，不够空间则绘制到框内顶部
            label_x1 = max(0, x1)
            label_y2 = max(text_h + 6, y1)
            label_y1 = max(0, label_y2 - text_h - 6)
            label_x2 = min(img.shape[1] - 1, label_x1 + text_w + 8)

            overlay = img.copy()
            cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

            text_org = (label_x1 + 4, label_y2 - 4)
            # 先绘制深色描边，再绘制浅色前景，提升复杂背景下可读性
            cv2.putText(
                img,
                label,
                text_org,
                font,
                font_scale,
                (20, 20, 20),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label,
                text_org,
                font,
                font_scale,
                (255, 255, 255),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

        return img

    def process_single_video(self, video_path, save_dir):
        """处理单个视频"""
        print(f'\n{"=" * 70}')
        print(f'🎬 处理视频: {video_path.name}')
        print(f'{"=" * 70}')

        # 为每个视频重新初始化跟踪器
        self.tracker = self._create_tracker()

        # 创建该视频的输出目录
        video_save_dir = save_dir / video_path.stem
        video_save_dir.mkdir(parents=True, exist_ok=True)

        # 获取数据加载器
        dataloader, fps, w, h, total_frames, is_video = self._get_dataloader(str(video_path))

        # 初始化保存器
        vid_writer = None
        txt_file = None

        save_vid = self.args.save_vid
        save_txt = self.args.save_txt

        if save_vid:
            result_video_path = video_save_dir / 'result.mp4'
            vid_writer = cv2.VideoWriter(str(result_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f'   视频将保存至: {result_video_path}')

        if save_txt:
            result_txt_path = video_save_dir / 'results.txt'
            txt_file = open(result_txt_path, 'w')
            print(f'   指标将保存至: {result_txt_path}')

        # 统计变量
        stats = {
            'total_tracks': set(),
            'frame_count': 0,
            'total_detections': 0,
            'tracked_detections': 0,
            'matched_detections': 0,
            'frames_with_detections': 0,
            'frames_with_tracks': 0,
            'id_switch_proxy': 0,
            'elapsed_sec': 0.0,
            'core_elapsed_sec': 0.0,
        }

        print('\n🚀 开始跟踪处理...')
        pbar = tqdm(total=total_frames, desc='Tracking', ncols=80)
        frame_time_start = time.perf_counter()
        prev_matched_entries = []
        fps_ema = None

        while True:
            # 读取下一帧（不计入核心FPS）
            if is_video:
                ret, frame = dataloader.read()
                if not ret:
                    break
            else:
                if stats['frame_count'] >= len(dataloader):
                    break
                frame = self._safe_read_image(str(dataloader[stats['frame_count']]))
                if frame is None:
                    stats['frame_count'] += 1
                    pbar.update(1)
                    continue

            # === 核心计时开始：仅包含 检测 + 跟踪 ===
            core_start = time.perf_counter()

            # 目标检测
            det_result = self.detector.detect(frame)

            boxes, confs, clss = [], [], []
            if hasattr(det_result, 'boxes'):
                boxes = det_result.boxes
                confs = det_result.confidences
                clss = det_result.classes

            # 统计检测
            if len(boxes) > 0:
                stats['total_detections'] += len(boxes)
                stats['frames_with_detections'] += 1

            # 目标跟踪
            tracks = []
            if len(boxes) > 0:
                try:
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    if isinstance(confs, torch.Tensor):
                        confs = confs.cpu().numpy()
                    if isinstance(clss, torch.Tensor):
                        clss = clss.cpu().numpy()

                    if self.args.tracker == 'deepsort':
                        result = self.tracker.update(detections=boxes, confidences=confs, ori_img=frame, classes=clss)
                    else:  # bytetrack
                        result = self.tracker.update(detections=boxes, confidences=confs, classes=clss)

                    if hasattr(result, 'tracks'):
                        tracks = result.tracks
                    else:
                        tracks = result

                    if len(tracks) > 0:
                        stats['tracked_detections'] += len(tracks)
                        stats['frames_with_tracks'] += 1

                        # 只统计当帧与检测真实匹配到的轨迹，避免可见延迟框导致成功率超过100%
                        matched_tracks = [
                            t for t in tracks
                            if int(getattr(t, 'time_since_update', 0)) == 0
                        ]
                        stats['matched_detections'] += len(matched_tracks)

                except Exception as e:
                    if self.args.debug:
                        print(f'\n⚠️  跟踪出错 (Frame {stats["frame_count"]}): {e}')

            # === 核心计时结束：仅包含检测+跟踪，排除I/O、可视化、编码 ===
            core_dt = time.perf_counter() - core_start
            stats['core_elapsed_sec'] += core_dt

            # 数据记录
            for track in tracks:
                if hasattr(track, 'bbox'):
                    bbox = track.bbox
                elif hasattr(track, 'to_tlbr'):
                    bbox = track.to_tlbr()
                else:
                    continue

                tid = track.track_id
                conf = getattr(track, 'confidence', 1.0)

                stats['total_tracks'].add(tid)

                if txt_file:
                    x1, y1, x2, y2 = bbox
                    w_box = x2 - x1
                    h_box = y2 - y1
                    line = f"{stats['frame_count'] + 1},{tid},{x1:.2f},{y1:.2f},{w_box:.2f},{h_box:.2f},{conf:.2f},-1,-1,-1\n"
                    txt_file.write(line)

            # 可视化（可关闭以提速）
            if self.args.no_overlay:
                vis_frame = frame
            else:
                vis_frame = self.draw_tracks(frame.copy(), tracks, boxes if self.args.debug else None)

            # 估算ID切换代理指标（仅统计当帧真实匹配轨迹）
            current_matched_entries = []
            for t in tracks:
                if int(getattr(t, 'time_since_update', 0)) != 0:
                    continue
                tb = getattr(t, 'bbox', None)
                if tb is None:
                    continue
                current_matched_entries.append({
                    'track_id': int(getattr(t, 'track_id', -1)),
                    'class_id': int(getattr(t, 'class_id', -1)),
                    'bbox': np.asarray(tb, dtype=np.float32),
                })
            stats['id_switch_proxy'] += self._estimate_id_switches(prev_matched_entries, current_matched_entries)
            prev_matched_entries = current_matched_entries

            # 计算并平滑FPS（仅基于检测+跟踪核心耗时）
            dt = max(core_dt, 1e-6)
            inst_fps = 1.0 / dt
            alpha = float(np.clip(self.args.fps_alpha, 0.01, 1.0))
            fps_ema = inst_fps if fps_ema is None else (1.0 - alpha) * fps_ema + alpha * inst_fps

            if not self.args.no_overlay:
                # 顶部提示：现代HUD风格（左上状态 + 右上FPS）
                line1_text = (
                    f'Frame {stats["frame_count"]}   '
                    f'Det {len(boxes)}   '
                    f'Track {len(tracks)}   '
                    f'IDs {len(stats["total_tracks"])}'
                )
                fps_text = f'FPS {fps_ema:.1f}'
                self._draw_hud_chip(
                    vis_frame,
                    line1_text,
                    x=12,
                    y=10,
                    text_color=(223, 235, 255),
                    bg_color=(24, 31, 43),
                    accent_color=(104, 232, 255),
                    alpha=0.35,
                )

                fps_w = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] + 20
                fps_x = vis_frame.shape[1] - fps_w - 12
                self._draw_hud_chip(
                    vis_frame,
                    fps_text,
                    x=fps_x,
                    y=10,
                    text_color=(255, 245, 214),
                    bg_color=(29, 37, 52),
                    accent_color=(255, 166, 108),
                    alpha=0.38,
                )

            if vid_writer:
                vid_writer.write(vis_frame)

            if self.args.show:
                cv2.imshow('Tracking Visualization', vis_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            pbar.update(1)
            stats['frame_count'] += 1

        # 结束清理
        pbar.close()
        if vid_writer:
            vid_writer.release()
        if is_video:
            dataloader.release()
        if txt_file:
            txt_file.close()
        cv2.destroyAllWindows()

        stats['elapsed_sec'] = time.perf_counter() - frame_time_start
        self._finalize_video_stats(stats, input_fps=fps)
        self._save_video_metrics(video_save_dir, video_path.name, stats)

        # 打印详细的最终报告
        print('\n' + '-' * 70)
        print(f'✅ 视频处理完成: {video_path.name}')
        print(f'   📂 结果目录: {video_save_dir}')
        print('-' * 70)
        print(f'📊 详细统计:')
        print(f'   - 处理总帧数: {stats["frame_count"]}')
        print(f'   - 总检测数: {stats["total_detections"]}')
        print(
            f'   - 含有检测的帧数: {stats["frames_with_detections"]} ({stats["frames_with_detections"] / max(stats["frame_count"], 1) * 100:.1f}%)')
        print(f'   - 总显示跟踪框数: {stats["tracked_detections"]}')
        print(f'   - 检测匹配跟踪数: {stats["matched_detections"]}')
        print(f'   - ID切换代理数: {stats["id_switch_proxy"]}')
        print(
            f'   - 含有跟踪的帧数: {stats["frames_with_tracks"]} ({stats["frames_with_tracks"] / max(stats["frame_count"], 1) * 100:.1f}%)')
        print(f'   - 捕获唯一目标ID数: {len(stats["total_tracks"])}')
        if stats['total_detections'] > 0:
            print(f'   - 检测匹配率: {stats["matched_detections"] / stats["total_detections"] * 100:.1f}%')
        print(f'   - 平均处理FPS: {stats["avg_fps"]:.2f} (xRT={stats["realtime_factor"]:.2f})')
        print('-' * 70)

        return stats

    def run(self):
        """主执行循环 - 支持批量视频处理"""
        source_path = Path(self.args.data)

        if source_path.is_dir() and source_path.suffix.lower() not in ['.yaml', '.yml']:
            # 视频目录模式
            print(f'\n🎬 批量视频处理模式')
            video_files = self._get_video_list(source_path)

            if not video_files:
                print('❌ 未找到任何视频文件')
                sys.exit(1)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_save_dir = Path(self.args.output) / f'{self.args.tracker}_{timestamp}'
            main_save_dir.mkdir(parents=True, exist_ok=True)

            print(f'\n📁 主输出目录: {main_save_dir}')
            print(f'\n🎬 将处理 {len(video_files)} 个视频\n')

            all_stats = []
            for idx, video_file in enumerate(video_files, 1):
                print(f'\n[{idx}/{len(video_files)}]', end=' ')
                stats = self.process_single_video(video_file, main_save_dir)
                all_stats.append((video_file.name, stats))

            # 打印总体统计
            print('\n' + '=' * 70)
            print('🎉 所有视频处理完成！')
            print('=' * 70)
            print('\n📊 总体统计:')
            print(f'{"视频名称":<40} {"帧数":<8} {"检测":<8} {"匹配":<8} {"显示":<8} {"目标ID"}')
            print('-' * 70)

            total_frames = 0
            total_detections = 0
            total_matched = 0
            total_rendered = 0
            total_unique_ids = 0
            total_id_switch_proxy = 0

            for video_name, stats in all_stats:
                frames = stats['frame_count']
                detections = stats['total_detections']
                matched = stats['matched_detections']
                rendered = stats['tracked_detections']
                ids = len(stats['total_tracks'])
                id_sw = stats['id_switch_proxy']

                total_frames += frames
                total_detections += detections
                total_matched += matched
                total_rendered += rendered
                total_unique_ids += ids
                total_id_switch_proxy += id_sw

                print(f'{video_name:<40} {frames:<8} {detections:<8} {matched:<8} {rendered:<8} {ids}')

            print('-' * 70)
            print(f'{"总计":<40} {total_frames:<8} {total_detections:<8} {total_matched:<8} {total_rendered:<8} {total_unique_ids}')
            if total_detections > 0:
                print(f'\n总体检测匹配率: {total_matched / total_detections * 100:.1f}%')
            print(f'总体ID切换代理数: {total_id_switch_proxy}')

            self._save_batch_summary(main_save_dir, all_stats)
            print('=' * 70)
            print(f'\n✅ 所有结果已保存至: {main_save_dir}\n')

        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_save_dir = Path(self.args.output) / f'{self.args.tracker}_{timestamp}'
            main_save_dir.mkdir(parents=True, exist_ok=True)

            self.process_single_video(source_path, main_save_dir)


def main():
    args = parse_args()
    runner = TrackingRunner(args)
    runner.run()


if __name__ == '__main__':
    main()