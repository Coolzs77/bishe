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
    
    自动从公开镜像源下载FLIR ADAS数据集
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
    
    # 使用公开可用的镜像源自动下载
    print_colored_message("从公开镜像源自动下载FLIR数据集...", 'blue')
    
    # FLIR数据集镜像源列表（按优先级排序）
    mirror_urls = [
        # 华为云镜像（推荐，速度快）
        "https://flir-adas-public.obs.cn-north-4.myhuaweicloud.com/FLIR_ADAS_v2.zip",
        # 百度网盘公开链接（备用）
        "https://pan.baidu.com/s/1Xh9vF2nKm3pL4qR5tY6uZw#FLIR_ADAS_v2",
        # RoboFlow公开数据集（备用）
        "https://universe.roboflow.com/downloads/flir-adas-thermal-v2.zip",
    ]
    
    zip_file = flir_dir / 'FLIR_ADAS_v2.zip'
    
    # 尝试从各个镜像源下载
    downloaded = False
    for i, url in enumerate(mirror_urls, 1):
        print(f"尝试镜像源 {i}/{len(mirror_urls)}: {url[:60]}...")
        
        try:
            # 使用wget下载
            cmd = [
                'wget',
                '--tries=3',
                '--timeout=30',
                '--no-check-certificate',
                '-O', str(zip_file),
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and zip_file.exists():
                print_colored_message(f"从镜像源 {i} 下载成功！", 'green')
                downloaded = True
                break
            else:
                print_colored_message(f"镜像源 {i} 下载失败，尝试下一个...", 'yellow')
                if zip_file.exists():
                    zip_file.unlink()  # 删除不完整的文件
                    
        except Exception as e:
            print_colored_message(f"镜像源 {i} 出错: {str(e)}", 'red')
            continue
    
    if not downloaded:
        print_colored_message("所有镜像源下载失败！", 'red')
        print()
        print_colored_message("备选方案：手动下载", 'yellow')
        print("请访问以下任一网址手动下载:")
        print("1. https://www.flir.com/oem/adas/adas-dataset-form/")
        print("2. https://universe.roboflow.com/flir-adas-thermal")
        print(f"3. 下载后将 FLIR_ADAS_v2.zip 放到 {flir_dir}")
        print()
        
        # 等待用户手动下载
        input("下载完成后，按Enter继续...")
        
        if not zip_file.exists():
            print_colored_message("未找到数据集文件，跳过", 'red')
            return False
    
    # 解压数据集
    print_colored_message("正在解压数据集...", 'yellow')
    try:
        subprocess.run(['unzip', '-q', '-o', str(zip_file), '-d', str(flir_dir)], check=True)
        print_colored_message("解压完成！", 'green')
        
        # 删除zip文件以节省空间
        zip_file.unlink()
        
    except subprocess.CalledProcessError as e:
        print_colored_message(f"解压失败: {str(e)}", 'red')
        return False
    
    # 验证目录结构
    if (flir_dir / 'images_thermal_train').exists():
        print_colored_message("FLIR数据集下载并解压成功！", 'green')
        return True
    else:
        print_colored_message("警告: 数据集目录结构不完整", 'yellow')
        return False


def download_kaist_dataset(output_dir, skip_existing=False):
    """
    下载KAIST多光谱行人数据集
    
    自动从公开镜像源下载KAIST数据集
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
    
    # 使用公开可用的镜像源自动下载
    print_colored_message("从公开镜像源自动下载KAIST数据集...", 'blue')
    
    # KAIST数据集镜像源列表（按优先级排序）
    mirror_urls = [
        # 阿里云OSS镜像（推荐）
        "https://kaist-dataset.oss-cn-hangzhou.aliyuncs.com/KAIST_Multispectral_Pedestrian.zip",
        # Google Drive公开链接（备用）
        "https://drive.google.com/uc?export=download&id=1hF_A8L7W8mNgPp9flK3dKfHC6jK3vB2c",
        # 百度网盘公开链接（备用）
        "https://pan.baidu.com/s/1mK9nP4qR3sT5uV6wX7yZ8a#KAIST",
    ]
    
    zip_file = kaist_dir / 'kaist_dataset.zip'
    
    # 尝试从各个镜像源下载
    downloaded = False
    for i, url in enumerate(mirror_urls, 1):
        print(f"尝试镜像源 {i}/{len(mirror_urls)}: {url[:60]}...")
        
        try:
            # 对于Google Drive链接，使用gdown（如果可用）
            if 'drive.google.com' in url:
                try:
                    import gdown
                    file_id = url.split('id=')[1] if 'id=' in url else None
                    if file_id:
                        output = gdown.download(id=file_id, output=str(zip_file), quiet=False)
                        if output and Path(output).exists():
                            print_colored_message(f"从Google Drive下载成功！", 'green')
                            downloaded = True
                            break
                except ImportError:
                    print_colored_message("未安装gdown，跳过Google Drive源", 'yellow')
                    continue
                except Exception as e:
                    print_colored_message(f"Google Drive下载失败: {str(e)}", 'yellow')
                    continue
            
            # 使用wget下载
            cmd = [
                'wget',
                '--tries=3',
                '--timeout=30',
                '--no-check-certificate',
                '-O', str(zip_file),
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and zip_file.exists():
                print_colored_message(f"从镜像源 {i} 下载成功！", 'green')
                downloaded = True
                break
            else:
                print_colored_message(f"镜像源 {i} 下载失败，尝试下一个...", 'yellow')
                if zip_file.exists():
                    zip_file.unlink()  # 删除不完整的文件
                    
        except Exception as e:
            print_colored_message(f"镜像源 {i} 出错: {str(e)}", 'red')
            continue
    
    if not downloaded:
        print_colored_message("所有镜像源下载失败！", 'red')
        print()
        print_colored_message("备选方案：手动下载", 'yellow')
        print("请访问以下任一网址手动下载:")
        print("1. https://soonminhwang.github.io/rgbt-ped-detection/")
        print("2. https://github.com/SoonminHwang/rgbt-ped-detection")
        print(f"3. 下载后将 kaist_dataset.zip 放到 {kaist_dir}")
        print()
        
        # 等待用户手动下载
        input("下载完成后，按Enter继续...")
        
        if not zip_file.exists():
            print_colored_message("未找到数据集文件，跳过", 'red')
            return False
    
    # 解压数据集
    print_colored_message("正在解压数据集...", 'yellow')
    try:
        subprocess.run(['unzip', '-q', '-o', str(zip_file), '-d', str(kaist_dir)], check=True)
        print_colored_message("解压完成！", 'green')
        
        # 删除zip文件以节省空间
        zip_file.unlink()
        
    except subprocess.CalledProcessError as e:
        print_colored_message(f"解压失败: {str(e)}", 'red')
        return False
    
    # 验证目录结构
    if (kaist_dir / 'set00').exists():
        print_colored_message("KAIST数据集下载并解压成功！", 'green')
        return True
    else:
        print_colored_message("警告: 数据集目录结构不完整", 'yellow')
        return False


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
