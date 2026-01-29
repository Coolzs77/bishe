#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data集下载脚本
用于下载FLIR和KAIST红外data集
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_colored_info(message, color='green'):
    """打印彩色信息"""
    颜色代码 = {
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'blue': '\033[0;34m',
        'reset': '\033[0m'
    }
    print(f"{颜色代码.get(颜色, '')}{消息}{颜色代码['reset']}")


def check_dependencies():
    """检查必要的下载工具是否安装"""
    缺失工具 = []
    
    # 检查wget
    try:
        subprocess.run(['wget', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        缺失工具.append('wget')
    
    # 检查unzip
    try:
        subprocess.run(['unzip', '-v'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        缺失工具.append('unzip')
    
    if 缺失工具:
        print_colored_info(f"缺少必要工具: {', '.join(缺失工具)}", 'red')
        print_colored_info("请使用以下command安装:", 'yellow')
        print(f"  sudo apt-get install {' '.join(缺失工具)}")
        return False
    
    return True


def create_directory(path):
    """创建目录（如果不存在）"""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_flir_dataset(output_dir, skip_existing=False):
    """
    下载FLIR红外data集
    
    注意: FLIRdata集需要注册后下载，此函数提供下载说明
    """
    print_colored_info("=" * 50, 'yellow')
    print_colored_info("下载FLIR红外data集...", 'yellow')
    print_colored_info("=" * 50, 'yellow')
    
    flir目录 = Path(output_dir) / 'flir'
    create_directory(flir目录)
    
    # 检查是否已存在
    if (flir目录 / 'images_thermal_train').exists() and skip_existing:
        print_colored_info("检测到FLIRdata集已存在，跳过下载", 'green')
        return True
    
    # FLIRdata集需要注册后下载，提供说明
    print()
    print_colored_info("注意: FLIRdata集需要注册后下载", 'yellow')
    print()
    print("请按照以下步骤手动下载:")
    print("1. 访问 https://www.flir.com/oem/adas/adas-dataset-form/")
    print("2. 填写表单注册")
    print("3. 下载 'FLIR_ADAS_v2' data集")
    print(f"4. 将下载的文件解压到 {flir目录}")
    print()
    print("期望的目录结构:")
    print(f"  {flir目录}/")
    print("  ├── images_thermal_train/")
    print("  ├── images_thermal_val/")
    print("  └── annotations/")
    print()
    
    # 检查是否已存在
    if (flir目录 / 'images_thermal_train').exists():
        print_colored_info("检测到FLIRdata集已存在", 'green')
        return True
    
    # 等待用户确认
    print_colored_info("等待用户手动下载FLIRdata集...", 'yellow')
    input("下载完成后，按Enter继续...")
    
    return True


def download_kaist_dataset(output_dir, skip_existing=False):
    """
    下载KAIST多光谱行人data集
    
    注意: KAISTdata集需要从官网下载
    """
    print_colored_info("=" * 50, 'yellow')
    print_colored_info("下载KAIST多光谱行人data集...", 'yellow')
    print_colored_info("=" * 50, 'yellow')
    
    kaist目录 = Path(output_dir) / 'kaist'
    create_directory(kaist目录)
    
    # 检查是否已存在
    if (kaist目录 / 'set00').exists() and skip_existing:
        print_colored_info("检测到KAISTdata集已存在，跳过下载", 'green')
        return True
    
    print()
    print_colored_info("注意: KAISTdata集需要从官网下载", 'yellow')
    print()
    print("请按照以下步骤手动下载:")
    print("1. 访问 https://soonminhwang.github.io/rgbt-ped-detection/")
    print("2. 下载 'KAIST Multispectral Pedestrian Detection Benchmark'")
    print(f"3. 将下载的文件解压到 {kaist目录}")
    print()
    print("期望的目录结构:")
    print(f"  {kaist目录}/")
    print("  ├── set00/")
    print("  ├── set01/")
    print("  ├── ...")
    print("  └── annotations/")
    print()
    
    # 检查是否已存在
    if (kaist目录 / 'set00').exists():
        print_colored_info("检测到KAISTdata集已存在", 'green')
        return True
    
    # 等待用户确认
    print_colored_info("等待用户手动下载KAISTdata集...", 'yellow')
    input("下载完成后，按Enter继续...")
    
    return True


def create_calibration_dataset_dir(output_dir):
    """创建量化校准data集目录"""
    print_colored_info("创建量化校准data集目录...", 'yellow')
    
    校准目录 = Path(output_dir).parent / 'processed' / 'flir' / 'calibration'
    create_directory(校准目录)
    
    print(f"校准data集目录: {校准目录}")
    print("请在model量化前，复制约100张代表性image到此目录")


def parse_args():
    """解析command行参数"""
    parser = argparse.ArgumentParser(
        description='红外data集下载脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python download_dataset.py                    # 下载所有data集
  python download_dataset.py --flir             # 仅下载FLIRdata集
  python download_dataset.py --output-dir /data # 指定output目录
        '''
    )
    
    parser.add_argument('--flir', action='store_true',
                        help='仅下载FLIRdata集')
    parser.add_argument('--kaist', action='store_true',
                        help='仅下载KAISTdata集')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='data保存目录 (默认: data/raw)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的文件')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 确定下载哪些data集
    下载flir = True
    下载kaist = True
    
    if args.flir and not args.kaist:
        下载kaist = False
    elif args.kaist and not args.flir:
        下载flir = False
    
    # 打印config信息
    print_colored_info("=" * 50, 'green')
    print_colored_info("  红外data集下载脚本", 'green')
    print_colored_info("=" * 50, 'green')
    print()
    print(f"output目录: {args.output_dir}")
    print(f"下载FLIR: {下载flir}")
    print(f"下载KAIST: {下载kaist}")
    print()
    
    # create_output_dir
    create_directory(args.output_dir)
    
    # download_dataset
    if 下载flir:
        download_flir_dataset(args.output_dir, args.skip_existing)
    
    if 下载kaist:
        download_kaist_dataset(args.output_dir, args.skip_existing)
    
    # 创建校准data集目录
    create_calibration_dataset_dir(args.output_dir)
    
    # 完成
    print()
    print_colored_info("=" * 50, 'green')
    print_colored_info("  data集准备完成!", 'green')
    print_colored_info("=" * 50, 'green')
    print()
    print("下一步:")
    print("  1. run python scripts/data/prepare_flir.py 准备FLIRdata集")
    print("  2. run python scripts/data/prepare_kaist.py 准备KAISTdata集")


if __name__ == '__main__':
    main()
