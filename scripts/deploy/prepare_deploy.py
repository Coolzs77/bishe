#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署准备脚本 — 在 Windows 训练机上运行.

功能:
  1. 导出指定消融实验的 best.pt → ONNX 格式
  2. 从 FLIR 验证集中随机选取图片生成量化校准集列表
  3. 选取测试图片和测试视频, 复制到 deploy 目录
  4. 生成部署检查清单

用法:
  # 导出基线+eiou 模型 (主力)
  python scripts/deploy/prepare_deploy.py --exp eiou

  # 导出 ghost+eiou 模型 (候选, 注意训练时 imgsz=704)
  python scripts/deploy/prepare_deploy.py --exp ghost_eiou --imgsz 640

  # 仅准备测试数据, 不导出模型
  python scripts/deploy/prepare_deploy.py --test-data-only
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]

# ---- 消融实验 → 模型权重映射 ----
EXPERIMENT_MAP = {
    "eiou": {
        "dir": "ablation_exp07_eiou",
        "desc": "基线 + EIoU (主力模型)",
        "train_imgsz": 640,
    },
    "ghost_eiou": {
        "dir": "ablation_exp09_ghost_eiou",
        "desc": "Ghost + EIoU (轻量化候选)",
        "train_imgsz": 704,  # 注意: 训练时使用 704, 导出时选择 640 或 704
    },
    "attention_eiou": {
        "dir": "ablation_exp10_attention_eiou",
        "desc": "Attention + EIoU",
        "train_imgsz": 640,
    },
    "baseline": {
        "dir": "ablation_exp01_baseline",
        "desc": "YOLOv5s 基线",
        "train_imgsz": 640,
    },
}

# ---- 测试视频 (从 data/videos/thermal_test/ 中选取) ----
# 选短的视频方便传输和板端测试
TEST_VIDEOS = [
    "ZAtDSNuZZjkZFvMAo_seq006.mp4",   # 4.11 MB - 最小, 适合快速测试
    "t3f7QC8hZr6zYXpEZ_seq009.mp4",   # 6.93 MB - 中等长度
]

# ---- 路径常量 ----
DEPLOY_DIR = PROJECT_ROOT / "deploy" / "rv1126b_yolov5"
DEPLOY_MODEL_DIR = DEPLOY_DIR / "model"
DEPLOY_TESTDATA_DIR = DEPLOY_DIR / "testdata"
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablation_study"
FLIR_VAL_DIR = PROJECT_ROOT / "data" / "processed" / "flir" / "images" / "val"
VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "thermal_test"


def parse_args():
    parser = argparse.ArgumentParser(description="部署准备: 导出 ONNX + 准备测试数据")
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENT_MAP.keys()),
                        help="要导出的消融实验名")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="导出 ONNX 时的输入尺寸 (默认使用训练时的尺寸)")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="ONNX simplify (默认开启)")
    parser.add_argument("--no-simplify", action="store_true",
                        help="禁用 ONNX simplify")
    parser.add_argument("--calib-count", type=int, default=150,
                        help="量化校准集图片数量 (默认 150)")
    parser.add_argument("--test-images", type=int, default=5,
                        help="选取的测试图片数量 (默认 5)")
    parser.add_argument("--test-data-only", action="store_true",
                        help="仅准备测试数据, 不导出模型")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def export_onnx(exp_name, imgsz, simplify):
    """导出 ONNX 模型."""
    exp_info = EXPERIMENT_MAP[exp_name]
    weights_path = ABLATION_DIR / exp_info["dir"] / "weights" / "best.pt"

    if not weights_path.exists():
        print(f"[ERROR] 权重文件不存在: {weights_path}")
        return None

    if imgsz is None:
        imgsz = exp_info["train_imgsz"]

    print(f"\n{'=' * 60}")
    print(f"导出 ONNX: {exp_info['desc']}")
    print(f"  权重:  {weights_path}")
    print(f"  尺寸:  {imgsz}")
    print(f"{'=' * 60}\n")

    # 导出 ONNX
    cmd = [
        sys.executable, str(PROJECT_ROOT / "yolov5" / "export.py"),
        "--weights", str(weights_path),
        "--include", "onnx",
        "--imgsz", str(imgsz),
        "--batch-size", "1",
        "--device", "cpu",
    ]
    if simplify:
        cmd.append("--simplify")

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ONNX 导出失败: {e}")
        return None

    # 导出的 ONNX 和 .pt 在同一目录
    onnx_path = weights_path.with_suffix(".onnx")
    if not onnx_path.exists():
        print(f"[ERROR] 导出后未找到 ONNX 文件: {onnx_path}")
        return None

    # 复制到 deploy/model/
    dest_name = f"best_{exp_name}.onnx"
    dest_path = DEPLOY_MODEL_DIR / dest_name
    DEPLOY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest_path)
    print(f"\nONNX 已复制到: {dest_path}")
    print(f"文件大小: {dest_path.stat().st_size / 1024 / 1024:.1f} MB")

    return dest_path


def prepare_calibration_dataset(count, seed):
    """从 FLIR 验证集中随机选取图片, 生成量化校准集列表."""
    if not FLIR_VAL_DIR.exists():
        print(f"[ERROR] FLIR 验证集目录不存在: {FLIR_VAL_DIR}")
        return None

    all_images = sorted(FLIR_VAL_DIR.glob("*.jpg"))
    if len(all_images) == 0:
        print(f"[ERROR] 验证集中没有找到图片")
        return None

    random.seed(seed)
    selected = random.sample(all_images, min(count, len(all_images)))

    # 写入 dataset.txt (使用相对于 deploy 目录的路径, 但在 Ubuntu 上需要改为实际路径)
    # 这里写绝对路径, Ubuntu 上需要替换前缀:
    #   sed -i 's|D:\\pythonPro\\bishe|/home/coolzs77/bishe|g' calibration_dataset.txt
    #   sed -i 's|\\|/|g' calibration_dataset.txt
    dataset_txt = DEPLOY_DIR / "calibration_dataset.txt"
    with open(dataset_txt, "w", encoding="utf-8") as f:
        for img_path in selected:
            f.write(str(img_path) + "\n")

    print(f"\n量化校准集列表已生成: {dataset_txt}")
    print(f"  共 {len(selected)} 张红外图片")
    print(f"  注意: 无需手动修改路径, 转换时加 --val-dir /home/coolzs77/bishe/val 即可自动处理")

    return dataset_txt


def prepare_test_data(num_images, seed):
    """选取测试图片和视频, 复制到 deploy/testdata/."""
    DEPLOY_TESTDATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 测试图片 ----
    if FLIR_VAL_DIR.exists():
        all_images = sorted(FLIR_VAL_DIR.glob("*.jpg"))
        random.seed(seed + 100)  # 用不同种子, 避免和校准集完全重合
        selected = random.sample(all_images, min(num_images, len(all_images)))

        for i, img_path in enumerate(selected):
            # 复制并重命名为简短文件名
            dest = DEPLOY_TESTDATA_DIR / f"test_{i:02d}.jpg"
            shutil.copy2(img_path, dest)

        print(f"\n测试图片已复制到: {DEPLOY_TESTDATA_DIR}")
        print(f"  共 {len(selected)} 张")
        for i, img_path in enumerate(selected):
            print(f"  test_{i:02d}.jpg ← {img_path.name}")
    else:
        print(f"[WARN] 跳过测试图片: {FLIR_VAL_DIR} 不存在")

    # ---- 测试视频 ----
    if VIDEO_DIR.exists():
        copied_videos = []
        for video_name in TEST_VIDEOS:
            src = VIDEO_DIR / video_name
            if src.exists():
                dest = DEPLOY_TESTDATA_DIR / video_name
                shutil.copy2(src, dest)
                size_mb = dest.stat().st_size / 1024 / 1024
                copied_videos.append((video_name, size_mb))

        print(f"\n测试视频已复制到: {DEPLOY_TESTDATA_DIR}")
        for name, size in copied_videos:
            print(f"  {name} ({size:.1f} MB)")
    else:
        print(f"[WARN] 跳过测试视频: {VIDEO_DIR} 不存在")


def print_checklist(exp_name, onnx_path):
    """打印部署检查清单."""
    print(f"\n{'=' * 60}")
    print("部署检查清单")
    print(f"{'=' * 60}")

    exp_info = EXPERIMENT_MAP.get(exp_name, {})
    imgsz = exp_info.get("train_imgsz", 640)

    print(f"""
1. [Windows] ONNX 导出
   {'✅ 已完成' if onnx_path else '❌ 未完成'}
   {'   文件: ' + str(onnx_path) if onnx_path else ''}

2. [拷贝到 Ubuntu] 需要传输的文件 → /home/coolzs77/bishe/
   - deploy/rv1126b_yolov5/        → /home/coolzs77/bishe/deploy/rv1126b_yolov5/
   - data/processed/flir/images/val/ → /home/coolzs77/bishe/val/  (直接放到 bishe 根目录下)
   - rknn_model_zoo/               → /home/coolzs77/bishe/rknn_model_zoo/
   - data/videos/thermal_test/      (测试视频, 可选)

3. [Ubuntu] 修正校准集路径
   无需手动 sed, --val-dir 参数会自动处理

4. [Ubuntu] ONNX → RKNN 转换
   cd /home/coolzs77/bishe/deploy/rv1126b_yolov5/python
   python3 convert_yolov5_to_rknn.py \\
     --onnx ../model/best_{exp_name}.onnx \\
     --dataset ../calibration_dataset.txt \\
     --val-dir /home/coolzs77/bishe/val \\
     --output ../model/best_{exp_name}.rknn \\
     --target rv1126b --quant i8

5. [Ubuntu] 交叉编译
   cd /home/coolzs77/bishe/deploy/rv1126b_yolov5
   bash build_rv1126b.sh

6. [Ubuntu] 准备安装包
   cd /home/coolzs77/bishe/deploy/rv1126b_yolov5
   cp model/best_{exp_name}.rknn install/rv1126b_linux_aarch64/bishe_rknn_yolov5/model/
   cp testdata/* install/rv1126b_linux_aarch64/bishe_rknn_yolov5/model/

7. [板子] 推送并测试
   adb push install/rv1126b_linux_aarch64/bishe_rknn_yolov5 /userdata/
   adb shell
   cd /userdata/bishe_rknn_yolov5
   export LD_LIBRARY_PATH=./lib

   # 图片检测
   ./bishe_rknn_detect model/best_{exp_name}.rknn model/test_00.jpg

   # 视频检测
   ./bishe_rknn_video model/best_{exp_name}.rknn model/{TEST_VIDEOS[0]}

   # 取回结果
   exit
   adb pull /userdata/bishe_rknn_yolov5/out.png ./
   adb pull /userdata/bishe_rknn_yolov5/out_video.mp4 ./
""")

    if exp_name == "ghost_eiou":
        print(f"""
⚠️  注意: ghost_eiou 训练时使用 imgsz={imgsz}, 和其他实验 (640) 不一致.
   导出 ONNX 时请确认使用正确的尺寸:
   - 如果用 --imgsz 704: RKNN 输入为 704x704, 需要修改 postprocess.hpp 中 MODEL_INPUT_SIZE
   - 如果用 --imgsz 640: 可以直接部署, 但和训练时尺寸不同, 可能有精度差异
   推荐: 用 --imgsz 640 导出, 这样不需要改部署代码.
""")


def main():
    args = parse_args()

    print("=" * 60)
    print("红外 YOLOv5 部署准备工具")
    print("=" * 60)

    onnx_path = None

    # 1. 导出 ONNX
    if not args.test_data_only and args.exp:
        simplify = args.simplify and not args.no_simplify
        onnx_path = export_onnx(args.exp, args.imgsz, simplify)

    # 2. 准备量化校准集
    if not args.test_data_only:
        prepare_calibration_dataset(args.calib_count, args.seed)

    # 3. 准备测试数据
    prepare_test_data(args.test_images, args.seed)

    # 4. 打印检查清单
    if args.exp:
        print_checklist(args.exp, onnx_path)


if __name__ == "__main__":
    main()
