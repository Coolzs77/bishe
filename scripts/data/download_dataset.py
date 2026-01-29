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


def 打印彩色信息(消息, 颜色='green'):
    """打印彩色信息"""
    颜色代码 = {
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'blue': '\033[0;34m',
        'reset': '\033[0m'
    }
    print(f"{颜色代码.get(颜色, '')}{消息}{颜色代码['reset']}")


def 检查依赖工具():
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
        打印彩色信息(f"缺少必要工具: {', '.join(缺失工具)}", 'red')
        打印彩色信息("请使用以下命令安装:", 'yellow')
        print(f"  sudo apt-get install {' '.join(缺失工具)}")
        return False
    
    return True


def 创建目录(路径):
    """创建目录（如果不存在）"""
    Path(路径).mkdir(parents=True, exist_ok=True)


def 下载FLIR数据集(输出目录, 跳过已存在=False):
    """
    下载FLIR红外数据集
    
    注意: FLIR数据集需要注册后下载，此函数提供下载说明
    """
    打印彩色信息("=" * 50, 'yellow')
    打印彩色信息("下载FLIR红外数据集...", 'yellow')
    打印彩色信息("=" * 50, 'yellow')
    
    flir目录 = Path(输出目录) / 'flir'
    创建目录(flir目录)
    
    # 检查是否已存在
    if (flir目录 / 'images_thermal_train').exists() and 跳过已存在:
        打印彩色信息("检测到FLIR数据集已存在，跳过下载", 'green')
        return True
    
    # FLIR数据集需要注册后下载，提供说明
    print()
    打印彩色信息("注意: FLIR数据集需要注册后下载", 'yellow')
    print()
    print("请按照以下步骤手动下载:")
    print("1. 访问 https://www.flir.com/oem/adas/adas-dataset-form/")
    print("2. 填写表单注册")
    print("3. 下载 'FLIR_ADAS_v2' 数据集")
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
        打印彩色信息("检测到FLIR数据集已存在", 'green')
        return True
    
    # 等待用户确认
    打印彩色信息("等待用户手动下载FLIR数据集...", 'yellow')
    input("下载完成后，按Enter继续...")
    
    return True


def 下载KAIST数据集(输出目录, 跳过已存在=False):
    """
    下载KAIST多光谱行人数据集
    
    注意: KAIST数据集需要从官网下载
    """
    打印彩色信息("=" * 50, 'yellow')
    打印彩色信息("下载KAIST多光谱行人数据集...", 'yellow')
    打印彩色信息("=" * 50, 'yellow')
    
    kaist目录 = Path(输出目录) / 'kaist'
    创建目录(kaist目录)
    
    # 检查是否已存在
    if (kaist目录 / 'set00').exists() and 跳过已存在:
        打印彩色信息("检测到KAIST数据集已存在，跳过下载", 'green')
        return True
    
    print()
    打印彩色信息("注意: KAIST数据集需要从官网下载", 'yellow')
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
        打印彩色信息("检测到KAIST数据集已存在", 'green')
        return True
    
    # 等待用户确认
    打印彩色信息("等待用户手动下载KAIST数据集...", 'yellow')
    input("下载完成后，按Enter继续...")
    
    return True


def 创建校准数据集目录(输出目录):
    """创建量化校准数据集目录"""
    打印彩色信息("创建量化校准数据集目录...", 'yellow')
    
    校准目录 = Path(输出目录).parent / 'processed' / 'flir' / 'calibration'
    创建目录(校准目录)
    
    print(f"校准数据集目录: {校准目录}")
    print("请在模型量化前，复制约100张代表性图像到此目录")


def 解析参数():
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
    args = 解析参数()
    
    # 确定下载哪些数据集
    下载flir = True
    下载kaist = True
    
    if args.flir and not args.kaist:
        下载kaist = False
    elif args.kaist and not args.flir:
        下载flir = False
    
    # 打印配置信息
    打印彩色信息("=" * 50, 'green')
    打印彩色信息("  红外数据集下载脚本", 'green')
    打印彩色信息("=" * 50, 'green')
    print()
    print(f"输出目录: {args.output_dir}")
    print(f"下载FLIR: {下载flir}")
    print(f"下载KAIST: {下载kaist}")
    print()
    
    # 创建输出目录
    创建目录(args.output_dir)
    
    # 下载数据集
    if 下载flir:
        下载FLIR数据集(args.output_dir, args.skip_existing)
    
    if 下载kaist:
        下载KAIST数据集(args.output_dir, args.skip_existing)
    
    # 创建校准数据集目录
    创建校准数据集目录(args.output_dir)
    
    # 完成
    print()
    打印彩色信息("=" * 50, 'green')
    打印彩色信息("  数据集准备完成!", 'green')
    打印彩色信息("=" * 50, 'green')
    print()
    print("下一步:")
    print("  1. 运行 python scripts/data/prepare_flir.py 准备FLIR数据集")
    print("  2. 运行 python scripts/data/prepare_kaist.py 准备KAIST数据集")


if __name__ == '__main__':
    main()
