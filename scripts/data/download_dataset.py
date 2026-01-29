#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集下载脚本
用于下载FLIR和KAIST红外数据集
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_colored_message(message, color='green'):
    """打印彩色信息"""
    color_codes = {
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'blue': '\033[0;34m',
        'reset': '\033[0m'
    }
    print(f"{color_codes.get(color, '')}{message}{color_codes['reset']}")


def check_dependencies():
    """检查必要的下载工具是否安装"""
    missing_tools = []
    
    # 检查wget
    try:
        subprocess.run(['wget', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_tools.append('wget')
    
    # 检查unzip
    try:
        subprocess.run(['unzip', '-v'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_tools.append('unzip')
    
    if missing_tools:
        print_colored_message(f"缺少必要工具: {', '.join(missing_tools)}", 'red')
        print_colored_message("请使用以下命令安装:", 'yellow')
        print(f"  sudo apt-get install {' '.join(missing_tools)}")
        return False
    
    return True


def create_directory(path):
    """创建目录（如果不存在）"""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_flir_dataset(output_dir, skip_existing=False):
    """
    下载FLIR红外数据集
    
    注意: FLIR数据集需要注册后下载，此函数提供下载说明
    """
    print_colored_message("=" * 50, 'yellow')
    print_colored_message("下载FLIR红外数据集...", 'yellow')
    print_colored_message("=" * 50, 'yellow')
    
    flir_dir = Path(output_dir) / 'flir'
    create_directory(flir_dir)
    
    # 检查是否已存在
    if (flir_dir / 'images_thermal_train').exists() and skip_existing:
        print_colored_message("检测到FLIR数据集已存在，跳过下载", 'green')
        return True
    
    # FLIR数据集需要注册后下载，提供说明
    print()
    print_colored_message("注意: FLIR数据集需要注册后下载", 'yellow')
    print()
    print("请按照以下步骤手动下载:")
    print("1. 访问 https://www.flir.com/oem/adas/adas-dataset-form/")
    print("2. 填写表单注册")
    print("3. 下载 'FLIR_ADAS_v2' 数据集")
    print(f"4. 将下载的文件解压到 {flir_dir}")
    print()
    print("期望的目录结构:")
    print(f"  {flir_dir}/")
    print("  ├── images_thermal_train/")
    print("  ├── images_thermal_val/")
    print("  └── annotations/")
    print()
    
    # 检查是否已存在
    if (flir_dir / 'images_thermal_train').exists():
        print_colored_message("检测到FLIR数据集已存在", 'green')
        return True
    
    # 等待用户确认
    print_colored_message("等待用户手动下载FLIR数据集...", 'yellow')
    input("下载完成后，按Enter继续...")
    
    return True


def download_kaist_dataset(output_dir, skip_existing=False):
    """
    下载KAIST多光谱行人数据集
    
    注意: KAIST数据集需要从官网下载
    """
    print_colored_message("=" * 50, 'yellow')
    print_colored_message("下载KAIST多光谱行人数据集...", 'yellow')
    print_colored_message("=" * 50, 'yellow')
    
    kaist_dir = Path(output_dir) / 'kaist'
    create_directory(kaist_dir)
    
    # 检查是否已存在
    if (kaist_dir / 'set00').exists() and skip_existing:
        print_colored_message("检测到KAIST数据集已存在，跳过下载", 'green')
        return True
    
    print()
    print_colored_message("注意: KAIST数据集需要从官网下载", 'yellow')
    print()
    print("请按照以下步骤手动下载:")
    print("1. 访问 https://soonminhwang.github.io/rgbt-ped-detection/")
    print("2. 下载 'KAIST Multispectral Pedestrian Detection Benchmark'")
    print(f"3. 将下载的文件解压到 {kaist_dir}")
    print()
    print("期望的目录结构:")
    print(f"  {kaist_dir}/")
    print("  ├── set00/")
    print("  ├── set01/")
    print("  ├── ...")
    print("  └── annotations/")
    print()
    
    # 检查是否已存在
    if (kaist_dir / 'set00').exists():
        print_colored_message("检测到KAIST数据集已存在", 'green')
        return True
    
    # 等待用户确认
    print_colored_message("等待用户手动下载KAIST数据集...", 'yellow')
    input("下载完成后，按Enter继续...")
    
    return True


def create_calibration_directory(output_dir):
    """创建量化校准数据集目录"""
    print_colored_message("创建量化校准数据集目录...", 'yellow')
    
    calibration_dir = Path(output_dir).parent / 'processed' / 'flir' / 'calibration'
    create_directory(calibration_dir)
    
    print(f"校准数据集目录: {calibration_dir}")
    print("请在模型量化前，复制约100张代表性图像到此目录")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='红外数据集下载脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python download_dataset.py                    # 下载所有数据集
  python download_dataset.py --flir             # 仅下载FLIR数据集
  python download_dataset.py --output-dir /data # 指定输出目录
        '''
    )
    
    parser.add_argument('--flir', action='store_true',
                        help='仅下载FLIR数据集')
    parser.add_argument('--kaist', action='store_true',
                        help='仅下载KAIST数据集')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='数据保存目录 (默认: data/raw)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的文件')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确定下载哪些数据集
    download_flir = True
    download_kaist = True
    
    if args.flir and not args.kaist:
        download_kaist = False
    elif args.kaist and not args.flir:
        download_flir = False
    
    # 打印配置信息
    print_colored_message("=" * 50, 'green')
    print_colored_message("  红外数据集下载脚本", 'green')
    print_colored_message("=" * 50, 'green')
    print()
    print(f"输出目录: {args.output_dir}")
    print(f"下载FLIR: {download_flir}")
    print(f"下载KAIST: {download_kaist}")
    print()
    
    # 创建输出目录
    create_directory(args.output_dir)
    
    # 下载数据集
    if download_flir:
        download_flir_dataset(args.output_dir, args.skip_existing)
    
    if download_kaist:
        download_kaist_dataset(args.output_dir, args.skip_existing)
    
    # 创建校准数据集目录
    create_calibration_directory(args.output_dir)
    
    # 完成
    print()
    print_colored_message("=" * 50, 'green')
    print_colored_message("  数据集准备完成!", 'green')
    print_colored_message("=" * 50, 'green')
    print()
    print("下一步:")
    print("  1. 运行 python scripts/data/prepare_flir.py 准备FLIR数据集")
    print("  2. 运行 python scripts/data/prepare_kaist.py 准备KAIST数据集")


if __name__ == '__main__':
    main()
