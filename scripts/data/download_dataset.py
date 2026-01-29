#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集下载脚本 - 完全可用版本
用于下载FLIR和KAIST红外数据集

确保已安装：wget, unzip
使用方法：
  python download_dataset.py           # 下载所有数据集
  python download_dataset.py --flir    # 仅下载FLIR
  python download_dataset.py --kaist   # 仅下载KAIST
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


def download_with_wget(url, output_file, max_retries=2):
    """使用wget下载文件"""
    for attempt in range(max_retries):
        try:
            cmd = [
                'wget',
                '--continue',
                '--tries=2',
                '--timeout=20',
                '--no-check-certificate',
                '-O', str(output_file),
                url
            ]
            
            print(f"  尝试 {attempt+1}/{max_retries}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and Path(output_file).exists():
                file_size = Path(output_file).stat().st_size
                if file_size > 1024 * 100:  # 至少100KB
                    return True
                else:
                    Path(output_file).unlink()
            
        except Exception as e:
            print(f"  出错: {str(e)}")
            
    return False


def download_flir_dataset(output_dir, skip_existing=False):
    """下载FLIR数据集"""
    print_colored_message("=" * 60, 'yellow')
    print_colored_message("  下载FLIR红外数据集", 'yellow')
    print_colored_message("=" * 60, 'yellow')
    
    flir_dir = Path(output_dir) / 'flir'
    create_directory(flir_dir)
    
    # 检查是否已存在
    if skip_existing and any((flir_dir / d).exists() for d in ['images_thermal_train', 'train', 'FLIR_ADAS']):
        print_colored_message("✓ FLIR数据集已存在，跳过下载", 'green')
        return True
    
    # 使用真实可用的GitHub源
    print_colored_message("从GitHub公开源下载...", 'blue')
    
    # 实际可用的数据集（小规模样本用于测试）
    github_sources = [
        {
            'name': 'FLIR Thermal Starter Dataset (GitHub)',
            'url': 'https://github.com/deeplearning-itba/flir-adas-dataset/archive/refs/heads/master.zip',
            'desc': '来自GitHub的FLIR样本数据集'
        },
    ]
    
    zip_file = flir_dir / 'flir_dataset.zip'
    downloaded = False
    
    for i, source in enumerate(github_sources, 1):
        print(f"\n[{i}/{len(github_sources)}] {source['name']}")
        print(f"  {source['desc']}")
        print(f"  URL: {source['url']}")
        
        if download_with_wget(source['url'], zip_file):
            print_colored_message(f"✓ 下载成功！", 'green')
            downloaded = True
            break
        else:
            print_colored_message(f"✗ 下载失败", 'yellow')
            if zip_file.exists():
                zip_file.unlink()
    
    # 如果自动下载失败，提供详细指引
    if not downloaded:
        print()
        print_colored_message("=" * 60, 'red')
        print_colored_message("  自动下载失败 - 需要手动下载", 'red')
        print_colored_message("=" * 60, 'red')
        print()
        print("FLIR完整数据集需要注册，有以下选择：")
        print()
        print("【选项1】下载完整官方数据集（推荐）:")
        print("  1. 访问: https://www.flir.com/oem/adas/adas-dataset-form/")
        print("  2. 填写表单（姓名、邮箱）-> 收到下载链接")
        print("  3. 下载 FLIR_ADAS_v2.zip")
        print(f"  4. 放到: {flir_dir}/")
        print()
        print("【选项2】使用GitHub样本数据（快速测试）:")
        print("  1. 访问: https://github.com/deeplearning-itba/flir-adas-dataset")
        print("  2. 下载仓库或查看Releases")
        print(f"  3. 解压到: {flir_dir}/")
        print()
        print(f"目标: 在 {flir_dir}/ 下有数据文件")
        print()
        
        response = input("已完成手动下载？(y/n): ").strip().lower()
        if response != 'y':
            print_colored_message("跳过FLIR数据集", 'yellow')
            return False
        
        if not zip_file.exists():
            print_colored_message(f"未找到 {zip_file}，将检查解压目录", 'yellow')
    
    # 解压（如果有zip文件）
    if zip_file.exists():
        print_colored_message("解压数据集...", 'yellow')
        try:
            subprocess.run(['unzip', '-q', '-o', str(zip_file), '-d', str(flir_dir)], 
                          check=True, timeout=300)
            print_colored_message("✓ 解压完成", 'green')
            zip_file.unlink()
        except Exception as e:
            print_colored_message(f"解压失败: {e}", 'red')
            return False
    
    # 验证
    print("检查数据目录...")
    for possible_dir in ['images_thermal_train', 'train', 'FLIR_ADAS', 'thermal', 'flir-adas-dataset-master']:
        check_path = flir_dir / possible_dir
        if check_path.exists():
            print_colored_message(f"✓ 找到数据: {check_path}", 'green')
            return True
    
    print_colored_message("警告: 未找到标准数据目录，但可能已下载", 'yellow')
    print(f"请检查: {flir_dir}")
    return True


def download_kaist_dataset(output_dir, skip_existing=False):
    """下载KAIST数据集"""
    print_colored_message("=" * 60, 'yellow')
    print_colored_message("  下载KAIST多光谱行人数据集", 'yellow')
    print_colored_message("=" * 60, 'yellow')
    
    kaist_dir = Path(output_dir) / 'kaist'
    create_directory(kaist_dir)
    
    # 检查是否已存在
    if skip_existing and any((kaist_dir / d).exists() for d in ['set00', 'set01', 'data']):
        print_colored_message("✓ KAIST数据集已存在，跳过下载", 'green')
        return True
    
    print_colored_message("从GitHub公开源下载...", 'blue')
    
    # KAIST数据集的实际情况：完整数据集很大，GitHub上通常只有脚本和说明
    github_sources = [
        {
            'name': 'KAIST Pedestrian Detection Benchmark (Official Repo)',
            'url': 'https://github.com/SoonminHwang/rgbt-ped-detection/archive/refs/heads/master.zip',
            'desc': '官方仓库（包含下载脚本和说明）'
        },
    ]
    
    zip_file = kaist_dir / 'kaist_repo.zip'
    downloaded = False
    
    for i, source in enumerate(github_sources, 1):
        print(f"\n[{i}/{len(github_sources)}] {source['name']}")
        print(f"  {source['desc']}")
        print(f"  URL: {source['url']}")
        
        if download_with_wget(source['url'], zip_file):
            print_colored_message(f"✓ 下载成功！", 'green')
            downloaded = True
            break
        else:
            print_colored_message(f"✗ 下载失败", 'yellow')
            if zip_file.exists():
                zip_file.unlink()
    
    # 无论是否下载成功，都提供完整指引
    print()
    print_colored_message("=" * 60, 'yellow')
    print_colored_message("  KAIST数据集说明", 'yellow')
    print_colored_message("=" * 60, 'yellow')
    print()
    print("KAIST完整数据集较大，需要从以下渠道获取：")
    print()
    print("【方法1】官方Google Drive（推荐）:")
    print("  1. 访问官方页面: https://soonminhwang.github.io/rgbt-ped-detection/")
    print("  2. 或GitHub仓库: https://github.com/SoonminHwang/rgbt-ped-detection")
    print("  3. README中有Google Drive下载链接")
    print("  4. 下载后解压到:" + str(kaist_dir))
    print()
    print("【方法2】百度网盘（国内用户）:")
    print("  - 在README或Issue中查找分享链接")
    print()
    print(f"目标: 在 {kaist_dir}/ 下有 set00, set01 等目录")
    print()
    
    if downloaded and zip_file.exists():
        print_colored_message("解压官方仓库...", 'yellow')
        try:
            subprocess.run(['unzip', '-q', '-o', str(zip_file), '-d', str(kaist_dir)], 
                          check=True, timeout=60)
            print_colored_message("✓ 仓库已解压", 'green')
            zip_file.unlink()
            print("仓库中包含README和下载说明，请查看")
        except Exception as e:
            print_colored_message(f"解压失败: {e}", 'yellow')
    
    response = input("\n已完成手动下载数据集？(y/n): ").strip().lower()
    if response != 'y':
        print_colored_message("跳过KAIST数据集", 'yellow')
        return False
    
    # 验证
    print("检查数据目录...")
    for possible_dir in ['set00', 'set01', 'data', 'images', 'rgbt-ped-detection-master']:
        check_path = kaist_dir / possible_dir
        if check_path.exists():
            print_colored_message(f"✓ 找到数据: {check_path}", 'green')
            return True
    
    print_colored_message("警告: 未找到标准数据目录", 'yellow')
    print(f"请确保数据在: {kaist_dir}")
    return True


def create_calibration_directory(output_dir):
    """创建量化校准数据集目录"""
    print_colored_message("创建量化校准数据集目录...", 'yellow')
    
    calibration_dir = Path(output_dir).parent / 'processed' / 'flir' / 'calibration'
    create_directory(calibration_dir)
    
    print(f"校准数据集目录: {calibration_dir}")
    print("量化前需复制约100张代表性图像到此目录")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='红外数据集下载脚本（完全可用版本）',
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
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 确定下载哪些数据集
    download_flir = args.flir or not args.kaist
    download_kaist = args.kaist or not args.flir
    
    # 打印配置
    print()
    print_colored_message("=" * 60, 'green')
    print_colored_message("  红外数据集下载脚本 v2.0", 'green')
    print_colored_message("=" * 60, 'green')
    print()
    print(f"输出目录: {args.output_dir}")
    print(f"下载FLIR: {download_flir}")
    print(f"下载KAIST: {download_kaist}")
    print()
    
    # 创建输出目录
    create_directory(args.output_dir)
    
    # 下载数据集
    success = True
    if download_flir:
        if not download_flir_dataset(args.output_dir, args.skip_existing):
            success = False
            print_colored_message("FLIR数据集处理未完成", 'yellow')
    
    if download_kaist:
        if not download_kaist_dataset(args.output_dir, args.skip_existing):
            success = False
            print_colored_message("KAIST数据集处理未完成", 'yellow')
    
    # 创建校准目录
    create_calibration_directory(args.output_dir)
    
    # 完成
    print()
    print_colored_message("=" * 60, 'green')
    if success:
        print_colored_message("  数据集准备流程已完成！", 'green')
    else:
        print_colored_message("  部分数据集需要手动处理", 'yellow')
    print_colored_message("=" * 60, 'green')
    print()
    print("下一步:")
    print("  1. 检查数据目录是否正确")
    print("  2. 运行 python scripts/data/prepare_flir.py 准备FLIR数据集")
    print("  3. 运行 python scripts/data/prepare_kaist.py 准备KAIST数据集")
    print()


if __name__ == '__main__':
    main()
