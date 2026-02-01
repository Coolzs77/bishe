#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪算法评估脚本 (最终完整版)
功能列表：
1. 多目标跟踪推理 (支持 DeepSORT/ByteTrack)
2. 实时可视化 (绘制检测框、ID、类别、运动轨迹)
3. 结果保存 (MP4 视频 + MOT格式 txt 文件)
4. 稳健性修复 (解决 Windows 路径乱码、参数缺失等问题)
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
# 核心路径修复：确保项目根目录在 sys.path 中，防止 ModuleNotFoundError
# ==============================================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 指向项目根目录 (bishe/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ==============================================================================

# 生成固定的颜色表，用于给不同 ID 分配不同颜色
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多目标跟踪评估与可视化')

    # 核心输入
    parser.add_argument('--weights', type=str, required=True, help='YOLOv5 检测器权重路径 (.pt)')
    parser.add_argument('--data', type=str, default='data/processed/flir/dataset.yaml',
                        help='数据集配置文件 或 视频文件路径')
    parser.add_argument('--tracker', type=str, default='deepsort', choices=['deepsort', 'bytetrack'],
                        help='选择跟踪算法')

    # 输出设置
    parser.add_argument('--output', type=str, default='outputs/tracking', help='结果输出目录')
    parser.add_argument('--save-vid', action='store_true', default=True, help='是否保存 MP4 视频')
    parser.add_argument('--save-txt', action='store_true', default=True, help='是否保存 MOT 格式的 txt 指标文件')
    parser.add_argument('--show', action='store_true', help='是否实时弹窗显示 (服务器端请关闭)')

    # 算法参数
    parser.add_argument('--conf-thres', type=float, default=0.25, help='检测置信度阈值 (低于此值的框会被过滤)')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='NMS 阈值')
    parser.add_argument('--device', type=str, default='0', help='计算设备 (例如 0 或 cpu)')

    return parser.parse_args()


class TrackingRunner:
    """跟踪流程管理器"""

    def __init__(self, args):
        self.args = args

        # 创建带时间戳的输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(args.output) / f'{args.tracker}_{self.timestamp}'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化设备
        self.device = self._select_device(args.device)

        # 加载模型
        self.detector = self._load_detector()
        self.tracker = self._create_tracker()

        # 轨迹历史 (用于绘制拖尾) {track_id: deque([(x,y), ...])}
        self.track_history = {}

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
        print(f'\n[2/3] 初始化跟踪器: {self.args.tracker}')
        try:
            if self.args.tracker == 'deepsort':
                from src.tracking.deepsort_tracker import create_deepsort_tracker
                # 注意：这里确保 deepsort_tracker.py 已经修复了 n_init 问题
                return create_deepsort_tracker(max_age=30, min_hits=3)

            elif self.args.tracker == 'bytetrack':
                from src.tracking.bytetrack_tracker import create_bytetrack_tracker
                return create_bytetrack_tracker(track_thresh=0.5, match_thresh=0.8)

        except Exception as e:
            print(f'❌ 跟踪器初始化失败: {e}')
            print('提示: 请检查 src/tracking/ 下的文件是否完整 (包含 tracker.py, kalman_filter.py 等)')
            traceback.print_exc()
            sys.exit(1)

    def _safe_read_image(self, path):
        """
        【关键修复】稳健的图片读取方法
        解决 Windows 下 cv2.imread 读取中文或特殊字符路径失败的问题
        """
        try:
            # 先用 numpy 读取文件流，再用 cv2 解码，这是最稳妥的方法
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _get_dataloader(self):
        """解析输入源，返回数据迭代器"""
        source = Path(self.args.data)
        is_video = source.is_file() and source.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']

        # 情况 A: 输入是 yaml 配置文件
        if source.suffix.lower() in ['.yaml', '.yml']:
            print(f'   正在解析配置文件: {source}')
            try:
                # 尝试多种编码读取
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                except UnicodeDecodeError:
                    with open(source, 'r', encoding='gbk') as f:
                        config = yaml.safe_load(f)
            except Exception as e:
                print(f'❌ 配置文件读取失败: {e}')
                sys.exit(1)

            # 优先寻找 test 路径，其次 val
            path_key = 'test' if 'test' in config else 'val'
            if path_key not in config:
                print('❌ 错误: 配置文件中未找到 test 或 val 路径')
                sys.exit(1)

            base_path = Path(config.get('path', ''))
            rel_path = config[path_key]

            # 尝试拼接绝对路径
            candidates = [
                (base_path / rel_path) if not Path(rel_path).is_absolute() else Path(rel_path),
                Path(self.args.data).parent / rel_path,
                Path(rel_path)
            ]

            target_path = None
            for p in candidates:
                if p.exists():
                    target_path = p
                    break

            if not target_path:
                print(f'❌ 找不到图片目录，尝试了以下路径:\n' + '\n'.join([str(p) for p in candidates]))
                sys.exit(1)

            source = target_path
            is_video = False

        print(f'\n[3/3] 数据源确定: {source}')

        # 返回视频捕获对象 或 图片列表
        if is_video:
            cap = cv2.VideoCapture(str(source))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return cap, fps, w, h, total_frames, True
        else:
            # 图片文件夹模式
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            files = sorted([p for p in source.rglob('*') if p.suffix.lower() in valid_exts])

            if not files:
                print(f'❌ 文件夹 {source} 为空或不包含图片')
                sys.exit(1)

            # 读取第一张图获取尺寸
            img0 = self._safe_read_image(str(files[0]))
            if img0 is None:
                print('❌ 无法读取第一张图片，请检查路径或权限')
                sys.exit(1)

            h, w = img0.shape[:2]
            return files, 30, w, h, len(files), False

    def draw_tracks(self, img, tracks):
        """
        核心可视化函数
        绘制：检测框、ID标签、中心点轨迹
        """
        for track in tracks:
            # 兼容不同格式的 Track 对象
            if hasattr(track, 'bbox'):
                bbox = track.bbox
                tid = track.track_id
            elif hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()
                tid = track.track_id
            else:
                continue  # 无法解析，跳过

            # 坐标取整
            x1, y1, x2, y2 = [int(i) for i in bbox]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 根据 ID 获取颜色
            color = [int(c) for c in COLORS[tid % len(COLORS)]]

            # 1. 画检测框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 2. 画 ID 标签背景
            label = f'ID: {tid}'
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 4, y1 - t_size[1] - 4), color, -1)

            # 3. 画 ID 文字
            cv2.putText(img, label, (x1 + 2, y1 - 2), 0, 0.6, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            # 4. 画轨迹拖尾 (Tail)
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=40)  # 只保留最近40帧的轨迹
            self.track_history[tid].append((cx, cy))

            # 将轨迹点转为 numpy 数组并绘制折线
            points = np.hstack(self.track_history[tid]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points], isClosed=False, color=color, thickness=2)

        return img

    def run(self):
        """主执行循环"""
        # 获取数据加载器
        dataloader, fps, w, h, total_frames, is_video = self._get_dataloader()

        # 初始化保存器
        vid_writer = None
        txt_file = None

        if self.args.save_vid:
            save_path = self.save_dir / 'result.mp4'
            vid_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f'   视频将保存至: {save_path}')

        if self.args.save_txt:
            txt_path = self.save_dir / 'results.txt'
            txt_file = open(txt_path, 'w')
            print(f'   指标将保存至: {txt_path}')

        # 统计变量
        stats = {
            'total_tracks': set(),
            'frame_count': 0
        }

        print('\n🚀 开始跟踪处理...')
        pbar = tqdm(total=total_frames, desc='Tracking')

        while True:
            # ========================
            # 1. 读取下一帧
            # ========================
            if is_video:
                ret, frame = dataloader.read()
                if not ret: break
            else:
                if stats['frame_count'] >= len(dataloader): break
                # 使用稳健读取方法
                frame = self._safe_read_image(str(dataloader[stats['frame_count']]))
                if frame is None:
                    # 如果某张图坏了，跳过它，防止程序崩溃
                    stats['frame_count'] += 1
                    pbar.update(1)
                    continue

            # ========================
            # 2. 目标检测 (YOLOv5)
            # ========================
            det_result = self.detector.detect(frame)

            # 提取检测结果
            boxes, confs, clss = [], [], []
            if hasattr(det_result, 'boxes'):
                boxes = det_result.boxes  # xyxy格式
                confs = det_result.confidences
                clss = det_result.classes

            # ========================
            # 3. 目标跟踪 (DeepSORT/ByteTrack)
            # ========================
            tracks = []
            if len(boxes) > 0:
                try:
                    # 格式转换: Tensor -> Numpy (如果需要)
                    if isinstance(boxes, torch.Tensor): boxes = boxes.cpu().numpy()
                    if isinstance(confs, torch.Tensor): confs = confs.cpu().numpy()
                    if isinstance(clss, torch.Tensor): clss = clss.cpu().numpy()

                    # 调用跟踪器
                    if self.args.tracker == 'deepsort':
                        # 【重要】传入 ori_img=frame 以启用 ReID
                        result = self.tracker.update(detections=boxes, confidences=confs, ori_img=frame, classes=clss)
                    else:
                        # ByteTrack 通常不需要原图
                        result = self.tracker.update(boxes, confs, clss)

                    # 统一结果格式
                    if hasattr(result, 'tracks'):
                        tracks = result.tracks
                    else:
                        tracks = result

                except Exception as e:
                    # 捕获跟踪器内部错误，打印但不中断程序
                    # print(f'\n❌ 跟踪出错 (Frame {stats["frame_count"]}): {e}')
                    pass

            # ========================
            # 4. 数据记录与写入
            # ========================
            for track in tracks:
                # 解析属性
                if hasattr(track, 'bbox'):
                    bbox = track.bbox
                elif hasattr(track, 'to_tlbr'):
                    bbox = track.to_tlbr()
                else:
                    continue

                tid = track.track_id
                conf = getattr(track, 'confidence', 1.0)

                # 统计唯一 ID
                stats['total_tracks'].add(tid)

                # 写入 MOT 格式 (frame, id, left, top, w, h, conf, -1, -1, -1)
                # 注意：MOT 格式帧号从 1 开始
                if txt_file:
                    x1, y1, x2, y2 = bbox
                    w_box = x2 - x1
                    h_box = y2 - y1
                    line = f"{stats['frame_count'] + 1},{tid},{x1:.2f},{y1:.2f},{w_box:.2f},{h_box:.2f},{conf:.2f},-1,-1,-1\n"
                    txt_file.write(line)

            # ========================
            # 5. 可视化与保存
            # ========================
            # 在图上画框
            vis_frame = self.draw_tracks(frame.copy(), tracks)

            # 在左上角显示当前帧信息
            info_text = f'Frame: {stats["frame_count"]} | Unique IDs: {len(stats["total_tracks"])}'
            cv2.putText(vis_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # 写入视频
            if vid_writer:
                vid_writer.write(vis_frame)

            # 实时显示
            if self.args.show:
                cv2.imshow('Tracking Visualization', vis_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # 更新进度
            pbar.update(1)
            stats['frame_count'] += 1

        # ========================
        # 6. 结束清理
        # ========================
        pbar.close()
        if vid_writer: vid_writer.release()
        if is_video: dataloader.release()
        if txt_file: txt_file.close()
        cv2.destroyAllWindows()

        # 打印最终报告
        print('\n' + '=' * 60)
        print('✅ 跟踪评估流程结束')
        print(f'   📂 结果目录: {self.save_dir}')
        print(f'   🎥 结果视频: result.mp4')
        print(f'   📄 指标文件: results.txt')
        print('-' * 60)
        print(f"📊 统计摘要:")
        print(f"   - 处理总帧数: {stats['frame_count']}")
        print(f"   - 捕获目标总数 (Total Unique IDs): {len(stats['total_tracks'])}")
        if stats['frame_count'] > 0:
            avg_obj = len(stats['total_tracks']) / (stats['frame_count'] / 30.0)  # 粗略估算
            # print(f"   - 估算流量: {avg_obj:.2f} 个/秒")
        print('=' * 60)


def main():
    args = parse_args()
    runner = TrackingRunner(args)
    runner.run()


if __name__ == '__main__':
    main()