#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪算法评估脚本 (改进版)
添加了调试信息和参数优化
支持 DeepSORT 和 ByteTrack
"""

import os
import sys
import argparse
import time
import traceback
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多目标跟踪评估与可视化')

    # 核心输入
    parser.add_argument('--weights', type=str, required=True, help='YOLOv5 检测器权重路径 (.pt)')
    parser.add_argument('--data', type=str, default='data/processed/flir/dataset.yaml',
                        help='数据集配置文件 或 视频文件路径 或 视频目录路径')
    parser.add_argument('--tracker', type=str, default='deepsort', choices=['deepsort', 'bytetrack'],
                        help='选择跟踪算法')

    # 输出设置
    parser.add_argument('--output', type=str, default='outputs/tracking', help='结果输出目录')
    parser.add_argument('--save-vid', action='store_true', default=True, help='是否保存 MP4 视频')
    parser.add_argument('--save-txt', action='store_true', default=True, help='是否保存 MOT 格式的 txt 指标文件')
    parser.add_argument('--show', action='store_true', help='是否实时弹窗显示')

    # 检测参数
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='检测置信度阈值 (降低此值可以检测更多目标)')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='NMS 阈值')
    parser.add_argument('--device', type=str, default='0', help='计算设备')

    # 跟踪参数
    parser.add_argument('--max-age', type=int, default=30,
                        help='目标最大失踪帧数 (越大越容易保持ID)')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='目标最少检测次数 (越小越容易出现)')
    parser.add_argument('--debug', action='store_true', help='是否输出调试信息')

    return parser.parse_args()


class TrackingRunner:
    """跟踪流程管理器"""

    def __init__(self, args):
        self.args = args

        # 初始化设备
        self.device = self._select_device(args.device)

        # 加载模型
        self.detector = self._load_detector()
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
                conf_threshold=self.args.conf_thres,
                nms_threshold=self.args.nms_thres,
                device=self.device,
                warmup=True
            )
        except Exception as e:
            print(f'❌ 检测器加载失败: {e}')
            traceback.print_exc()
            sys.exit(1)

    def _create_tracker(self):
        """初始化跟踪器 (DeepSORT 或 ByteTrack)"""
        print(f'[2/3] 初始化跟踪器: {self.args.tracker}')
        print(f'      参数: max_age={self.args.max_age}, min_hits={self.args.min_hits}')
        try:
            if self.args.tracker == 'deepsort':
                from src.tracking.deepsort_tracker import create_deepsort_tracker
                return create_deepsort_tracker(max_age=self.args.max_age, min_hits=self.args.min_hits)

            elif self.args.tracker == 'bytetrack':
                from src.tracking.bytetrack_tracker import create_bytetrack_tracker
                return create_bytetrack_tracker(max_age=self.args.max_age, min_hits=self.args.min_hits)

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

            # 画检测框（加粗）
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # 画 ID 标签
            label = f'ID: {tid}'
            t_size = cv2.getTextSize(label, 0, fontScale=0.8, thickness=2)[0]
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 6, y1 - t_size[1] - 6), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 3), 0, 0.8, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

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

        if self.args.save_vid:
            result_video_path = video_save_dir / 'result.mp4'
            vid_writer = cv2.VideoWriter(str(result_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f'   视频将保存至: {result_video_path}')

        if self.args.save_txt:
            result_txt_path = video_save_dir / 'results.txt'
            txt_file = open(result_txt_path, 'w')
            print(f'   指标将保存至: {result_txt_path}')

        # 统计变量
        stats = {
            'total_tracks': set(),
            'frame_count': 0,
            'total_detections': 0,
            'tracked_detections': 0,
            'frames_with_detections': 0,
            'frames_with_tracks': 0
        }

        print('\n🚀 开始跟踪处理...')
        pbar = tqdm(total=total_frames, desc='Tracking', ncols=80)

        while True:
            # 读取下一帧
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

                except Exception as e:
                    if self.args.debug:
                        print(f'\n⚠️  跟踪出错 (Frame {stats["frame_count"]}): {e}')

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

            # 可视化
            vis_frame = self.draw_tracks(frame.copy(), tracks, boxes if self.args.debug else None)

            # 显示统计信息
            info_text = f'Frame: {stats["frame_count"]} | Det: {len(boxes)} | Track: {len(tracks)} | IDs: {len(stats["total_tracks"])}'
            cv2.putText(vis_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
        print(f'   - 总跟踪数: {stats["tracked_detections"]}')
        print(
            f'   - 含有跟踪的帧数: {stats["frames_with_tracks"]} ({stats["frames_with_tracks"] / max(stats["frame_count"], 1) * 100:.1f}%)')
        print(f'   - 捕获唯一目标ID数: {len(stats["total_tracks"])}')
        if stats['total_detections'] > 0:
            print(f'   - 跟踪成功率: {stats["tracked_detections"] / stats["total_detections"] * 100:.1f}%')
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
            print(f'{"视频名称":<40} {"帧数":<8} {"检测":<8} {"跟踪":<8} {"目标ID"}')
            print('-' * 70)

            total_frames = 0
            total_detections = 0
            total_tracked = 0
            total_unique_ids = 0

            for video_name, stats in all_stats:
                frames = stats['frame_count']
                detections = stats['total_detections']
                tracked = stats['tracked_detections']
                ids = len(stats['total_tracks'])

                total_frames += frames
                total_detections += detections
                total_tracked += tracked
                total_unique_ids += ids

                print(f'{video_name:<40} {frames:<8} {detections:<8} {tracked:<8} {ids}')

            print('-' * 70)
            print(f'{"总计":<40} {total_frames:<8} {total_detections:<8} {total_tracked:<8} {total_unique_ids}')
            if total_detections > 0:
                print(f'\n总体跟踪成功率: {total_tracked / total_detections * 100:.1f}%')
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