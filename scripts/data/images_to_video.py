#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FLIR 红外图像序列转视频脚本。"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_filename(filename):
    """解析文件名并提取 video_id 与 frame_number。"""
    name = Path(filename).stem
    parts = name.split('-')

    if len(parts) >= 4:
        video_id = parts[1]
        frame_number = int(parts[3])
        return video_id, frame_number

    return None, None


def find_frame_step(frames):
    """自动检测同一序列内的帧号步长。"""
    if len(frames) < 2:
        return 1

    frame_nums = sorted([frame[0] for frame in frames])
    differences = []

    for i in range(1, len(frame_nums)):
        diff = frame_nums[i] - frame_nums[i - 1]
        if diff > 0:
            differences.append(diff)

    if not differences:
        return 1

    from collections import Counter

    return Counter(differences).most_common(1)[0][0]


def group_images_by_video_and_sequence(image_dir, frame_step=None, auto_detect=True):
    """根据视频 ID 与帧号步长分组图像序列。"""
    video_groups = defaultdict(list)
    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"错误: 目录不存在 {image_dir}")
        return video_groups

    by_video_id = defaultdict(list)
    jpg_files = list(image_dir.glob('*.jpg'))
    if not jpg_files:
        print(f"错误: 未找到 .jpg 文件: {image_dir}")
        return video_groups

    print(f"找到 {len(jpg_files)} 个图像文件")

    for img_path in jpg_files:
        video_id, frame_number = parse_filename(img_path.name)
        if video_id is not None:
            by_video_id[video_id].append((frame_number, img_path))

    print(f"找到 {len(by_video_id)} 个视频 ID")

    if auto_detect and frame_step is None:
        print("自动检测帧号步长...")
        all_frames = []
        for frames in by_video_id.values():
            all_frames.extend(frames)
        frame_step = find_frame_step(all_frames)
        print(f"检测到帧号步长: {frame_step}")
    elif frame_step is None:
        frame_step = 1

    sequence_index = 0

    for video_id in sorted(by_video_id.keys()):
        frames = sorted(by_video_id[video_id], key=lambda item: item[0])
        if not frames:
            continue

        current_sequence = [frames[0]]
        for i in range(1, len(frames)):
            current_frame_num = frames[i][0]
            prev_frame_num = frames[i - 1][0]
            frame_diff = current_frame_num - prev_frame_num

            if frame_diff == frame_step:
                current_sequence.append(frames[i])
            else:
                if len(current_sequence) >= 2:
                    sequence_key = f"{video_id}_seq{sequence_index:03d}"
                    video_groups[sequence_key] = current_sequence
                    sequence_index += 1
                current_sequence = [frames[i]]

        if len(current_sequence) >= 2:
            sequence_key = f"{video_id}_seq{sequence_index:03d}"
            video_groups[sequence_key] = current_sequence
            sequence_index += 1

    return video_groups


def create_videos_from_flir(input_dir, output_dir, fps=30, codec='mp4v', frame_step=None):
    """为每个热视频序列生成一个视频文件。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("FLIR 红外视频生成")
    print("=" * 70)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"视频帧率: {fps} fps")
    print("=" * 70)
    print()

    video_groups = group_images_by_video_and_sequence(input_path, frame_step=frame_step, auto_detect=True)
    if not video_groups:
        print("未找到任何视频序列，请检查输入目录和文件格式")
        return

    print(f"分组完成，共 {len(video_groups)} 个视频序列")
    print()

    total_videos = 0
    successful_videos = 0

    for sequence_key in sorted(video_groups.keys()):
        frame_data = video_groups[sequence_key]
        image_paths = [path for _, path in frame_data]
        frame_numbers = [num for num, _ in frame_data]
        total_videos += 1

        if len(image_paths) < 2:
            print(f"跳过 {sequence_key}: 只有 {len(image_paths)} 帧")
            continue

        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            print(f"错误: 无法读取 {image_paths[0].name}")
            continue

        height, width = first_frame.shape[:2]
        frame_min = min(frame_numbers)
        frame_max = max(frame_numbers)
        video_filename = f"{sequence_key}.mp4"
        video_path = output_path / video_filename

        print(f"[{total_videos}] {sequence_key}")
        print(f"  帧号范围: {frame_min} - {frame_max}")
        print(f"  总帧数: {len(image_paths)}")
        print(f"  分辨率: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not out.isOpened():
            print("  无法创建视频文件")
            continue

        for img_path in tqdm(image_paths, desc="  写入帧", leave=False, ncols=60):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                out.write(frame)

        out.release()

        try:
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            duration_sec = len(image_paths) / fps
            print(f"  完成: {video_filename}")
            print(f"  文件大小: {file_size_mb:.1f} MB | 时长: {duration_sec:.1f} 秒")
        except Exception:
            print(f"  完成: {video_filename}")

        print()
        successful_videos += 1

    print("=" * 70)
    print(f"生成完成: {successful_videos}/{total_videos} 个视频成功")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='从 FLIR 图像帧生成跟踪评估视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python scripts/data/images_to_video.py --input data/raw/flir/video_thermal_test/data --output data/videos/thermal_test

  python scripts/data/images_to_video.py --input data/raw/flir/video_thermal_test/data --output data/videos/thermal_test --fps 15

  python scripts/data/images_to_video.py --input data/raw/flir/video_thermal_test/data --output data/videos/thermal_test --frame-step 1
        '''
    )

    parser.add_argument('--input', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output', type=str, required=True, help='输出视频目录')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率 (默认: 30)')
    parser.add_argument(
        '--codec',
        type=str,
        default='mp4v',
        choices=['mp4v', 'MJPG', 'XVID', 'H264'],
        help='视频编码器 (默认: mp4v)',
    )
    parser.add_argument('--frame-step', type=int, default=None, help='帧号步长 (默认: 自动检测)')
    args = parser.parse_args()

    create_videos_from_flir(
        input_dir=args.input,
        output_dir=args.output,
        fps=args.fps,
        codec=args.codec,
        frame_step=args.frame_step,
    )


if __name__ == '__main__':
    main()