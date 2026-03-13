#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIR红外数据集视频生成脚本
从单个目录的图像帧生成视频

文件名格式: video-{VIDEO_ID}-frame-{FRAME_NUMBER}-{OTHER_INFO}.jpg
示例: video-4FRnNpmSmwktFJKjg-frame-000745-L6K5SC6fYjHNC8uff.jpg
"""

import cv2
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm


def parse_filename(filename):
    """
    解析文件名格式: video-4FRnNpmSmwktFJKjg-frame-000745-L6K5SC6fYjHNC8uff.jpg
    返回: (video_id, frame_number)
    """
    name = Path(filename).stem  # 去掉扩展名

    # 按 '-' 分割
    parts = name.split('-')

    if len(parts) >= 4:
        video_id = parts[1]
        frame_number = int(parts[3])
        return video_id, frame_number

    return None, None


def find_frame_step(frames):
    """
    自动检测帧号的步长
    例如：[100, 115, 130, 145] -> 步长为15
         [100, 101, 102, 103] -> 步长为1
    """
    if len(frames) < 2:
        return 1

    frame_nums = sorted([f[0] for f in frames])

    # 计算相邻帧的差
    differences = []
    for i in range(1, len(frame_nums)):
        diff = frame_nums[i] - frame_nums[i - 1]
        if diff > 0:  # 只记录正差
            differences.append(diff)

    if not differences:
        return 1

    # 找最常见的差值
    from collections import Counter
    most_common_diff = Counter(differences).most_common(1)[0][0]

    return most_common_diff


def group_images_by_video_and_sequence(image_dir, frame_step=None, auto_detect=True):
    """
    根据视频ID和帧号步长将图像分组成不同的视频序列

    返回: {sequence_key: [(frame_number, image_path), ...]}
    """
    video_groups = defaultdict(list)

    image_dir = Path(image_dir)

    # 检查目录是否存在
    if not image_dir.exists():
        print(f"❌ 错误: 目录不存在 {image_dir}")
        return video_groups

    # 第一步：按视频ID分组
    by_video_id = defaultdict(list)

    jpg_files = list(image_dir.glob('*.jpg'))
    if not jpg_files:
        print(f"❌ 错误: 未找到 .jpg 文件在 {image_dir}")
        return video_groups

    print(f"找到 {len(jpg_files)} 个图像文件")

    for img_path in jpg_files:
        video_id, frame_number = parse_filename(img_path.name)

        if video_id is not None:
            by_video_id[video_id].append((frame_number, img_path))

    print(f"找到 {len(by_video_id)} 个视频ID")

    # 自动检测帧号步长
    if auto_detect and frame_step is None:
        print("🔍 自动检测帧号步长...")
        all_frames = []
        for video_id in by_video_id.values():
            all_frames.extend(video_id)

        detected_step = find_frame_step(all_frames)
        print(f"✓ 检测到帧号步长: {detected_step}")
        frame_step = detected_step
    elif frame_step is None:
        frame_step = 1

    # 第二步：对每个视频ID，按帧号步长进一步分组
    sequence_index = 0

    for video_id in sorted(by_video_id.keys()):
        frames = sorted(by_video_id[video_id], key=lambda x: x[0])

        if not frames:
            continue

        # 创建序列
        current_sequence = [frames[0]]

        for i in range(1, len(frames)):
            current_frame_num = frames[i][0]
            prev_frame_num = frames[i - 1][0]

            # 检查帧号差是否为frame_step
            frame_diff = current_frame_num - prev_frame_num

            if frame_diff == frame_step:
                # 属于同一序列
                current_sequence.append(frames[i])
            else:
                # 新序列开始
                if len(current_sequence) >= 2:
                    sequence_key = f"{video_id}_seq{sequence_index:03d}"
                    video_groups[sequence_key] = current_sequence
                    sequence_index += 1

                current_sequence = [frames[i]]

        # 处理最后一个序列
        if len(current_sequence) >= 2:
            sequence_key = f"{video_id}_seq{sequence_index:03d}"
            video_groups[sequence_key] = current_sequence
            sequence_index += 1

    return video_groups


def create_videos_from_flir(input_dir, output_dir, fps=30, codec='mp4v', frame_step=None):
    """
    为每个视频序列生成一个视频文件

    参数:
        input_dir: 输入图像目录
        output_dir: 输出视频目录
        fps: 视频帧率
        codec: 视频编码器 ('mp4v', 'MJPG', 'XVID')
        frame_step: 同一序列内帧号的步长 (默认: 自动检测)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("🎬 FLIR 红外视频生成")
    print("=" * 70)
    print(f"📁 输入目录: {input_path}")
    print(f"📁 输出目录: {output_path}")
    print(f"⏱️  视频帧率: {fps} fps")
    print("=" * 70)
    print()

    # 分组图像（自动检测步长）
    video_groups = group_images_by_video_and_sequence(input_path, frame_step=frame_step, auto_detect=True)

    if not video_groups:
        print("❌ 未找到任何视频序列，请检查输入目录和文件格式")
        return

    print(f"✓ 分组完成，共 {len(video_groups)} 个视频序列")
    print()

    total_videos = 0
    successful_videos = 0

    # 为每个视频序列生成视频文件
    for sequence_key in sorted(video_groups.keys()):
        frame_data = video_groups[sequence_key]
        image_paths = [path for _, path in frame_data]
        frame_numbers = [num for num, _ in frame_data]

        total_videos += 1

        if len(image_paths) < 2:
            print(f"⚠️  跳过 {sequence_key}: 只有 {len(image_paths)} 帧 (至少需要2帧)")
            continue

        # 读取第一帧确定分辨率
        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            print(f"❌ 错误: 无法读取 {image_paths[0].name}")
            continue

        height, width = first_frame.shape[:2]

        # 帧号范围
        frame_min = min(frame_numbers)
        frame_max = max(frame_numbers)

        # 创建视频文件路径
        video_filename = f"{sequence_key}.mp4"
        video_path = output_path / video_filename

        print(f"🎬 [{total_videos}] {sequence_key}")
        print(f"   帧号范围: {frame_min} - {frame_max}")
        print(f"   总帧数: {len(image_paths)}")
        print(f"   分辨率: {width}x{height}")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"   ❌ 无法创建视频文件")
            continue

        # 逐帧写入视频
        for img_path in tqdm(image_paths, desc=f"   写入帧", leave=False, ncols=60):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                out.write(frame)

        out.release()

        try:
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            duration_sec = len(image_paths) / fps
            print(f"   ✓ 完成: {video_filename}")
            print(f"   文件大小: {file_size_mb:.1f} MB | 时长: {duration_sec:.1f} 秒")
        except Exception as e:
            print(f"   ✓ 完成: {video_filename}")

        print()

        successful_videos += 1

    print("=" * 70)
    print(f"✓ 生成完成: {successful_videos}/{total_videos} 个视频成功")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='从FLIR视频帧生成视频文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
文件名格式: video-{VIDEO_ID}-frame-{FRAME_NUMBER}-{OTHER_INFO}.jpg

使用示例:
  python p2v.py --input data/raw/flir/video_thermal_test/data --output data/videos/thermal_test

  python p2v.py --input data/raw/flir/video_thermal_test/data --output data/videos/thermal_test --fps 15

  python p2v.py --input data/raw/flir/video_thermal_test/data --output data/videos/thermal_test --frame-step 1
        '''
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入图像目录 (必须)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出视频目录 (必须)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='视频帧率 (默认: 30)'
    )
    parser.add_argument(
        '--codec',
        type=str,
        default='mp4v',
        choices=['mp4v', 'MJPG', 'XVID', 'H264'],
        help='视频编码器 (默认: mp4v)'
    )
    parser.add_argument(
        '--frame-step',
        type=int,
        default=None,
        help='帧号步长 (默认: 自动检测)'
    )

    args = parser.parse_args()

    create_videos_from_flir(
        input_dir=args.input,
        output_dir=args.output,
        fps=args.fps,
        codec=args.codec,
        frame_step=args.frame_step
    )


if __name__ == '__main__':
    main()